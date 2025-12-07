from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max

from src.models.components import GraphEnv
from src.utils.pylogger import RankedLogger

debug_logger = RankedLogger("gflownet.debug", rank_zero_only=True)

MIN_TEMPERATURE = 1e-5
LARGE_NEG = -1e9


class GFlowNetActor(nn.Module):
    """封装策略前向与环境 roll-out，包括可选 GT teacher forcing。"""

    def __init__(
        self,
        *,
        policy: nn.Module,
        env: GraphEnv,
        max_steps: int,
        policy_temperature: float,
        eval_policy_temperature: Optional[float],
        stop_logit_bias: float,
        random_action_prob: float,
        debug_actions: bool,
        debug_actions_steps: int,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)
        self.eval_policy_temperature = (
            self.policy_temperature if eval_policy_temperature is None else float(eval_policy_temperature)
        )
        self.stop_logit_bias = float(stop_logit_bias)
        self.random_action_prob = float(random_action_prob)
        self.debug_actions = bool(debug_actions)
        self.debug_actions_steps = max(int(debug_actions_steps), 0)

    def rollout(
        self,
        *,
        batch: Any,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        edge_scores: torch.Tensor,
        path_mask: Optional[torch.Tensor],
        path_exists: Optional[torch.Tensor],
        training: bool,
        gt_replay: bool,
        prior_alpha: float = 0.0,
        prior_score_eps: float = 1e-4,
        batch_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        # 选择分布温度：训练/评估可分离，用于 ablation。
        base_temperature = self.policy_temperature if training else self.eval_policy_temperature
        is_greedy = base_temperature < MIN_TEMPERATURE
        temperature = max(base_temperature, MIN_TEMPERATURE)
        eps = torch.finfo(edge_tokens.dtype).eps
        prob_eps = 1e-12
        random_action_prob = float(self.random_action_prob) if training else 0.0
        random_action_prob = min(max(random_action_prob, 0.0), 1.0)

        device = edge_tokens.device
        num_graphs = int(node_ptr.numel() - 1)
        assert (
            batch._slice_dict["answer_entity_ids"].shape == batch._slice_dict["answer_node_locals"].shape
        ), "answer_entity_ids ptr must align with answer_node_locals ptr"
        if batch.gt_path_edge_local_ids.numel() > 0:
            max_gt = int(batch.gt_path_edge_local_ids.max().item())
            assert max_gt < batch.edge_index.size(1), "gt_path_edge_local_ids exceeds edge_index size"

        graph_dict = {
            "edge_index": batch.edge_index.to(device),
            "edge_batch": edge_batch,
            "node_global_ids": batch.node_global_ids.to(device),
            "node_ptr": node_ptr,
            "edge_ptr": edge_ptr,
            "start_node_locals": batch.start_node_locals.to(device),
            "start_ptr": batch._slice_dict["start_node_locals"].to(device),
            "start_entity_ids": batch.start_entity_ids.to(device),
            "start_entity_ptr": batch._slice_dict["start_entity_ids"].to(device),
            "answer_node_locals": batch.answer_node_locals.to(device),
            "answer_ptr": batch._slice_dict["answer_entity_ids"].to(device),
            "answer_entity_ids": batch.answer_entity_ids.to(device),
            "edge_relations": batch.edge_attr.to(device),
            "edge_labels": batch.edge_labels.to(device),
            "top_edge_mask": batch.top_edge_mask.to(device),
            "gt_path_edge_local_ids": batch.gt_path_edge_local_ids.to(device),
            "gt_edge_ptr": batch._slice_dict["gt_path_edge_local_ids"].to(device),
            "gt_path_exists": batch.gt_path_exists.to(device),
            "is_answer_reachable": batch.is_answer_reachable.to(device),
        }
        state = self.env.reset(graph_dict, device=device)

        log_pf = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        use_gt_replay = gt_replay and path_exists is not None and bool(path_exists.any())
        log_pf_gt = torch.zeros(num_graphs, dtype=torch.float32, device=device) if use_gt_replay else None
        gt_positions = torch.zeros(num_graphs, dtype=torch.long, device=device) if use_gt_replay else None
        gt_edges = batch.gt_path_edge_local_ids.to(device) if use_gt_replay else None
        gt_ptr = batch._slice_dict["gt_path_edge_local_ids"].to(device) if use_gt_replay else None

        debug_logged: bool = not self.debug_actions or self.debug_actions_steps <= 0 or (batch_idx is not None and batch_idx != 0)

        actions = edge_ptr[1:].clone()

        for step in range(self.max_steps + 1):
            if state.done.all():
                break

            action_mask_edges = self.env.action_mask_edges(state)
            frontier_mask_edges = self.env.frontier_mask_edges(state)
            edge_logits, stop_logits, _ = self.policy(
                edge_tokens,
                question_tokens,
                edge_batch,
                state.selected_mask,
                edge_heads=batch.node_global_ids[batch.edge_index[0]].to(device),
                edge_tails=batch.node_global_ids[batch.edge_index[1]].to(device),
                current_tail=state.current_tail,
                frontier_mask=frontier_mask_edges,
            )
            if prior_alpha > 0.0:
                safe_scores = torch.clamp(edge_scores, min=prior_score_eps)
                edge_logits = edge_logits + prior_alpha * safe_scores.log()

            if not debug_logged:
                debug_logged = self._log_debug_actions(
                    step=step,
                    batch_idx=batch_idx,
                    edge_ptr=edge_ptr,
                    graph_dict=graph_dict,
                    action_mask_edges=action_mask_edges,
                    edge_logits=edge_logits,
                    state=state,
                    batch=batch,
                )

            # 有效 mask：屏蔽已完成图与非法边，避免 per-graph Python 循环。
            graph_done = state.done
            # stop 索引定义为该图片段结束位置（合法范围内的虚拟停止），env.step 必须识别为 stop。
            stop_indices = edge_ptr[1:]
            valid_edges = action_mask_edges & (~graph_done[edge_batch])

            # Stop logits 仅使用静态偏置，避免随 edge score 漂移导致策略失真。
            scaled_stop_logits = (stop_logits.squeeze(-1) + self.stop_logit_bias) / temperature

            # 归一化常数：segment-wise logsumexp
            edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
            valid_counts = (
                torch.bincount(edge_batch[valid_edges], minlength=num_graphs)
                if valid_edges.any()
                else torch.zeros(num_graphs, device=device, dtype=torch.long)
            )
            has_edge = edge_counts > 0
            has_valid_edge = valid_counts > 0
            total_actions = valid_counts.float() + 1.0  # +1 for stop
            uniform_edge_prob = torch.zeros_like(edge_logits).float()
            if valid_edges.any():
                uniform_edge_prob[valid_edges] = 1.0 / total_actions[edge_batch[valid_edges]].clamp(min=1.0)
            uniform_stop_prob = 1.0 / total_actions.clamp(min=1.0)

            def _log_probs(masked_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                scaled_edge_logits = masked_logits / temperature
                max_edge, _ = scatter_max(scaled_edge_logits, edge_batch, dim=0, dim_size=num_graphs)
                max_edge = torch.where(has_edge, max_edge, torch.full_like(max_edge, LARGE_NEG))
                max_joint = torch.maximum(max_edge, scaled_stop_logits)
                exp_edges = scatter_add(
                    torch.exp(scaled_edge_logits - max_joint[edge_batch]),
                    edge_batch,
                    dim=0,
                    dim_size=num_graphs,
                )
                exp_stop = torch.exp(scaled_stop_logits - max_joint)
                log_denom = max_joint + torch.log(exp_edges + exp_stop + eps)

                log_edge_prob = scaled_edge_logits - log_denom[edge_batch]
                log_stop_prob = scaled_stop_logits - log_denom
                edge_prob = torch.exp(log_edge_prob)
                stop_prob = torch.exp(log_stop_prob)
                final_edge_prob = (1.0 - random_action_prob) * edge_prob + random_action_prob * uniform_edge_prob
                final_stop_prob = (1.0 - random_action_prob) * stop_prob + random_action_prob * uniform_stop_prob
                log_final_edge = torch.log(final_edge_prob.clamp(min=prob_eps))
                log_final_stop = torch.log(final_stop_prob.clamp(min=prob_eps))
                log_final_edge = log_final_edge.masked_fill(~valid_edges, LARGE_NEG)
                return log_final_edge, log_final_stop

            masked_logits = edge_logits.masked_fill(~valid_edges, LARGE_NEG)
            log_final_edge, log_final_stop = _log_probs(masked_logits)

            # 采样：温度极低时走贪心，否则走 Gumbel-Max，基于混合后的分布。
            if is_greedy:
                score_edges = log_final_edge
                score_stop = log_final_stop
            else:
                score_edges = log_final_edge + self._gumbel_like(log_final_edge)
                score_stop = log_final_stop + self._gumbel_like(log_final_stop)
            score_edges_max, edge_argmax = scatter_max(score_edges, edge_batch, dim=0, dim_size=num_graphs)
            score_edges_max = torch.where(has_valid_edge, score_edges_max, torch.full_like(score_edges_max, LARGE_NEG))
            edge_argmax = torch.where(has_valid_edge, edge_argmax, torch.zeros_like(edge_argmax))

            choose_edge = has_valid_edge & (score_edges_max > score_stop)
            sampled_actions = torch.where(choose_edge, edge_argmax, stop_indices)
            log_pf_sampled = torch.where(choose_edge, log_final_edge[edge_argmax], log_final_stop)

            # GT replay（teacher forcing）：仅对 path_exists 图替换动作/对数概率。
            use_gt = torch.zeros(num_graphs, dtype=torch.bool, device=device)
            if use_gt_replay and path_exists is not None:
                use_gt = path_exists.bool()
                # 对使用 GT 的图启用环境绕过动作 mask（允许重复/回退），仅限 GT teacher forcing。
                state.graph.bypass_action_mask = use_gt

            # Step 0: 强制至少走一条边（若存在），避免起点即 stop。
            if step == 0:
                choose_edge = has_valid_edge
                sampled_actions = torch.where(choose_edge, edge_argmax, stop_indices)
                log_pf_sampled = torch.where(choose_edge, log_final_edge[edge_argmax], log_final_stop)

            gt_continue = torch.zeros(num_graphs, dtype=torch.bool, device=device)
            gt_stop = torch.zeros_like(gt_continue)
            gt_edge_choices = torch.full_like(sampled_actions, fill_value=-1)
            if use_gt.any():
                if gt_ptr is None or gt_edges is None or gt_positions is None or log_pf_gt is None:
                    raise RuntimeError("GT replay structures are missing despite use_gt_replay=True.")
                gt_lengths = gt_ptr[1:] - gt_ptr[:-1]
                gt_continue = use_gt & (gt_lengths > 0) & (gt_positions < gt_lengths)
                gt_stop = use_gt & (~gt_continue)

                # 将本地 gt 边索引转为全局边索引：仅对活跃图计算 offset，避免 repeat_interleave。
                if gt_continue.any():
                    active_gt = torch.nonzero(gt_continue).view(-1)
                    local_pos = gt_ptr[active_gt] + gt_positions[active_gt]
                    local_pos = torch.clamp(local_pos, max=gt_edges.numel() - 1)
                    # 注意：gt_path_edge_local_ids 在 PyG collate 后已累加成 batch 级绝对边索引
                    global_edges = gt_edges[local_pos]
                    # 额外校验：确保 gt 边落在所属图的边段内
                    span_start = edge_ptr[active_gt]
                    span_end = edge_ptr[active_gt + 1]
                    out_of_span = (global_edges < span_start) | (global_edges >= span_end)
                    if out_of_span.any():
                        bad = torch.nonzero(out_of_span, as_tuple=False).view(-1)
                        raise ValueError(
                            f"gt_path_edge_local_ids out of range for graphs {active_gt[bad].tolist()} "
                            f"global_edges={global_edges[bad].tolist()} "
                            f"edge_span={[(int(span_start[i].item()), int(span_end[i].item())) for i in bad.tolist()]}"
                        )
                gt_edge_choices[active_gt] = global_edges

            gt_log_p = torch.zeros_like(log_pf_sampled)
            if gt_continue.any():
                chosen_edges = gt_edge_choices[gt_continue]
                gt_log_p[gt_continue] = log_final_edge[chosen_edges]
            if gt_stop.any():
                gt_log_p[gt_stop] = log_final_stop[gt_stop]

            actions = torch.where(gt_continue, gt_edge_choices, torch.where(gt_stop, stop_indices, sampled_actions))
            log_pf_vec = torch.where(use_gt, gt_log_p, log_pf_sampled)
            if log_pf_gt is not None:
                log_pf_gt = log_pf_gt + torch.where(use_gt, gt_log_p, torch.zeros_like(log_pf_gt))
                gt_positions = torch.where(gt_continue, gt_positions + 1, gt_positions)

            # 已完成图强制停留，log_pf 增量为 0。
            actions = torch.where(graph_done, stop_indices, actions)
            log_pf_vec = torch.where(graph_done, torch.zeros_like(log_pf_vec), log_pf_vec)

            log_pf = log_pf + log_pf_vec
            state = self.env.step(state, actions, step_index=step)

        reach_success = state.answer_hits.float()
        reach_fraction = reach_success
        length = scatter_add(
            state.selected_mask.float(),
            edge_batch,
            dim=0,
            dim_size=num_graphs,
        )

        result = {
            "log_pf": log_pf,
            "selected_mask": state.selected_mask,
            "selection_order": state.selection_order,
            "actions": actions,
            "reach_fraction": reach_fraction,
            "reach_success": reach_success.float(),
            "length": length,
        }
        if log_pf_gt is not None:
            result["log_pf_gt"] = log_pf_gt
        return result

    def _log_debug_actions(
        self,
        *,
        step: int,
        batch_idx: Optional[int],
        edge_ptr: torch.Tensor,
        graph_dict: Dict[str, torch.Tensor],
        action_mask_edges: torch.Tensor,
        edge_logits: torch.Tensor,
        state: Any,
        batch: Any,
    ) -> bool:
        """在首个 batch 内打印首图的动作信息，避免主循环膨胀。"""
        if (
            not self.debug_actions
            or self.debug_actions_steps <= 0
            or step >= self.debug_actions_steps
            or (batch_idx is not None and batch_idx != 0)
        ):
            return False

        g_idx = 0
        start_idx = int(edge_ptr[g_idx].item())
        end_idx = int(edge_ptr[g_idx + 1].item())
        edge_count = end_idx - start_idx
        start_ptr = graph_dict["start_ptr"]
        s0, s1 = int(start_ptr[g_idx].item()), int(start_ptr[g_idx + 1].item())
        start_nodes = graph_dict["start_node_locals"][s0:s1].detach().cpu().tolist()
        current_tail = int(state.current_tail[g_idx].item()) if state.current_tail.numel() > g_idx else -1
        debug_logger.info(
            "[DEBUG_ACTION] g0 step=%d edges=%d starts_local=%s current_tail=%d",
            step,
            edge_count,
            start_nodes,
            current_tail,
        )
        if edge_count > 0:
            mask_slice = action_mask_edges[start_idx:end_idx]
            local_logits = edge_logits[start_idx:end_idx]
            if mask_slice.any():
                masked_logits = local_logits.masked_fill(~mask_slice, LARGE_NEG)
                top_local_idx = int(torch.argmax(masked_logits).item())
                edge_id = start_idx + top_local_idx
                head_id = int(batch.node_global_ids[batch.edge_index[0, edge_id]].item())
                tail_id = int(batch.node_global_ids[batch.edge_index[1, edge_id]].item())
                debug_logger.info(
                    "[DEBUG_ACTION] g0 top_edge=%d head=%d tail=%d logit=%.4g masked=%s",
                    edge_id,
                    head_id,
                    tail_id,
                    float(local_logits[top_local_idx].item()),
                    bool(mask_slice[top_local_idx].item()),
                )
            else:
                debug_logger.info("[DEBUG_ACTION] g0 has no valid edges under mask at step %d", step)
        return True

    @staticmethod
    def _gumbel_like(tensor: torch.Tensor) -> torch.Tensor:
        """Gumbel(0,1) 噪声，用于并行化采样。"""
        u = torch.clamp(torch.rand_like(tensor), min=1e-6, max=1.0 - 1e-6)
        return -torch.log(-torch.log(u))


__all__ = ["GFlowNetActor"]
