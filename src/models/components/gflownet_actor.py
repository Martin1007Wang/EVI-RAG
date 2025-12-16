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
        score_eps: float,
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
        self.score_eps = float(score_eps)
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
        training: bool,
        batch_idx: Optional[int] = None,
        graph_cache: Optional[Dict[str, torch.Tensor]] = None,
        node_tokens: Optional[torch.Tensor] = None,
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
        score_eps = self.score_eps
        if score_eps <= 0.0:
            raise ValueError(f"score_eps must be > 0, got {score_eps}")

        if graph_cache is not None:
            graph_dict = graph_cache
            node_global_ids = graph_cache["node_global_ids"]
            edge_index = graph_cache["edge_index"]
            heads_global = graph_cache["heads_global"]
            tails_global = graph_cache["tails_global"]
            edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
            base_edge_logits = torch.log(edge_scores.clamp(min=score_eps))
        else:
            # 预计算静态张量，避免循环内重复搬运/索引
            node_global_ids = batch.node_global_ids.to(device)
            edge_index = batch.edge_index.to(device)
            heads_global = node_global_ids[edge_index[0]]
            tails_global = node_global_ids[edge_index[1]]
            base_edge_logits = torch.log(edge_scores.clamp(min=score_eps))
            edge_counts = torch.bincount(edge_batch, minlength=num_graphs)

            graph_dict = {
                "edge_index": edge_index,
                "edge_batch": edge_batch,
                "node_global_ids": node_global_ids,
                "heads_global": heads_global,
                "tails_global": tails_global,
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
                "is_answer_reachable": batch.is_answer_reachable.to(device),
            }
        state = self.env.reset(graph_dict, device=device)

        start_state_emb: Optional[torch.Tensor] = None
        if node_tokens is not None:
            if node_tokens.dim() != 2:
                raise ValueError(f"node_tokens must be [N_total,H], got shape={tuple(node_tokens.shape)}")
            if node_tokens.size(0) != int(state.graph.node_ptr[-1].item()):
                raise ValueError(
                    "node_tokens size mismatch with batch nodes: "
                    f"node_tokens.size(0)={int(node_tokens.size(0))} vs node_ptr[-1]={int(state.graph.node_ptr[-1].item())}"
                )
            counts = (state.graph.start_ptr[1:] - state.graph.start_ptr[:-1]).to(device=device)
            start_nodes = state.graph.start_node_locals.to(device=device, dtype=torch.long).view(-1)
            start_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), counts)
            start_sum = scatter_add(node_tokens[start_nodes], start_batch, dim=0, dim_size=num_graphs)
            denom = counts.clamp(min=1).to(dtype=start_sum.dtype).unsqueeze(-1)
            start_state_emb = start_sum / denom

        log_pf = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        num_steps = self.max_steps + 1  # max_edges + 1 stop step
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        actions_seq = torch.full((num_graphs, num_steps), -1, dtype=torch.long, device=device)
        state_emb_seq = torch.zeros(
            num_graphs,
            num_steps,
            int(question_tokens.size(-1)),
            dtype=edge_tokens.dtype,
            device=device,
        )

        debug_logged: bool = not self.debug_actions or self.debug_actions_steps <= 0 or (batch_idx is not None and batch_idx != 0)

        stop_indices = edge_ptr[1:]
        actions = stop_indices.clone()

        def _log_probs_from_logits(
            *,
            edge_logits: torch.Tensor,
            stop_logits: torch.Tensor,
            valid_edges: torch.Tensor,
            temp: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            masked_edge_logits = edge_logits.masked_fill(~valid_edges, LARGE_NEG)
            scaled_edge_logits = masked_edge_logits / temp
            scaled_stop_logits = stop_logits / temp

            # Segment-wise logsumexp(edge logits, stop logit) for normalization.
            max_edge, _ = scatter_max(scaled_edge_logits, edge_batch, dim=0, dim_size=num_graphs)
            has_edge = torch.bincount(edge_batch, minlength=num_graphs) > 0
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

            log_edge = scaled_edge_logits - log_denom[edge_batch]
            log_stop = scaled_stop_logits - log_denom
            log_edge = log_edge.masked_fill(~valid_edges, LARGE_NEG)
            return log_edge, log_stop

        for step in range(num_steps):
            if state.done.all():
                break

            action_mask_edges = self.env.action_mask_edges(state)
            frontier_mask_edges = self.env.frontier_mask_edges(state)
            edge_residual, stop_residual, state_emb = self.policy(
                edge_tokens,
                question_tokens,
                edge_batch,
                state.selected_mask,
                edge_heads=heads_global,
                edge_tails=tails_global,
                current_tail=state.current_tail,
                frontier_mask=frontier_mask_edges,
            )
            if node_tokens is not None and start_state_emb is not None:
                is_step0_graph = state.step_counts == 0
                tail_local = state.current_tail_local.clamp(min=0)
                tail_emb = node_tokens[tail_local].to(dtype=state_emb.dtype)
                start_emb = start_state_emb.to(dtype=state_emb.dtype)
                state_emb = state_emb + torch.where(is_step0_graph.view(-1, 1), start_emb, tail_emb)
            state_emb_seq[:, step] = state_emb

            if not debug_logged:
                debug_logged = self._log_debug_actions(
                    step=step,
                    batch_idx=batch_idx,
                    edge_ptr=edge_ptr,
                    graph_dict=graph_dict,
                    action_mask_edges=action_mask_edges,
                    edge_logits=edge_residual,
                    state=state,
                    batch=batch,
                )

            # 有效 mask：屏蔽已完成图与非法边，避免 per-graph Python 循环。
            graph_done = state.done
            # stop 索引定义为该图片段结束位置（合法范围内的虚拟停止），env.step 必须识别为 stop。
            valid_edges = action_mask_edges & (~graph_done[edge_batch])

            # 归一化常数：segment-wise logsumexp
            valid_counts = (
                torch.bincount(edge_batch[valid_edges], minlength=num_graphs)
                if valid_edges.any()
                else torch.zeros(num_graphs, device=device, dtype=torch.long)
            )
            has_edge = edge_counts > 0
            has_valid_edge = valid_counts > 0
            total_actions = valid_counts.float() + 1.0  # +1 for stop
            uniform_edge_prob = torch.zeros_like(edge_residual, dtype=torch.float32)
            if valid_edges.any():
                uniform_edge_prob[valid_edges] = 1.0 / total_actions[edge_batch[valid_edges]].clamp(min=1.0)
            uniform_stop_prob = 1.0 / total_actions.clamp(min=1.0)

            # --- Base distribution P_retriever(a|s): normalize retriever scores under the current action mask. ---
            base_stop_logits = torch.full((num_graphs,), self.stop_logit_bias, device=device, dtype=edge_tokens.dtype)
            base_log_edge, base_log_stop = _log_probs_from_logits(
                edge_logits=base_edge_logits,
                stop_logits=base_stop_logits,
                valid_edges=valid_edges,
                temp=1.0,
            )

            # --- Residual policy: logit = log P_base + residual, then renormalize. ---
            combined_edge_logits = base_log_edge + edge_residual
            combined_stop_logits = base_log_stop + stop_residual
            clean_log_edge, clean_log_stop = _log_probs_from_logits(
                edge_logits=combined_edge_logits,
                stop_logits=combined_stop_logits,
                valid_edges=valid_edges,
                temp=temperature,
            )

            # --- Sampling distribution: epsilon-greedy mixture (NO gradient through noise; log_pf uses clean probs). ---
            clean_edge_prob = torch.exp(clean_log_edge)
            clean_stop_prob = torch.exp(clean_log_stop)
            sample_edge_prob = clean_edge_prob
            sample_stop_prob = clean_stop_prob
            if random_action_prob > 0.0:
                sample_edge_prob = (1.0 - random_action_prob) * clean_edge_prob + random_action_prob * uniform_edge_prob
                sample_stop_prob = (1.0 - random_action_prob) * clean_stop_prob + random_action_prob * uniform_stop_prob
            log_sample_edge = torch.log(sample_edge_prob.clamp(min=prob_eps)).masked_fill(~valid_edges, LARGE_NEG)
            log_sample_stop = torch.log(sample_stop_prob.clamp(min=prob_eps))

            # 采样：温度极低时走贪心，否则走 Gumbel-Max，基于混合后的分布。
            if is_greedy:
                score_edges = log_sample_edge
                score_stop = log_sample_stop
            else:
                score_edges = log_sample_edge + self._gumbel_like(log_sample_edge)
                score_stop = log_sample_stop + self._gumbel_like(log_sample_stop)
            score_edges_max, edge_argmax = scatter_max(score_edges, edge_batch, dim=0, dim_size=num_graphs)
            score_edges_max = torch.where(has_valid_edge, score_edges_max, torch.full_like(score_edges_max, LARGE_NEG))
            edge_argmax = torch.where(has_valid_edge, edge_argmax, torch.zeros_like(edge_argmax))

            choose_edge = has_valid_edge & (score_edges_max > score_stop)
            sampled_actions = torch.where(choose_edge, edge_argmax, stop_indices)
            # 行为策略对应该采样分布（含 epsilon），保持 TB 自洽
            log_pf_behavior = torch.where(choose_edge, log_sample_edge[edge_argmax], log_sample_stop)

            # Step 0: 强制至少走一条边（若存在），避免起点即 stop。
            if step == 0:
                choose_edge = has_valid_edge
                sampled_actions = torch.where(choose_edge, edge_argmax, stop_indices)
                log_pf_behavior = torch.where(choose_edge, log_sample_edge[edge_argmax], log_sample_stop)

            actions = sampled_actions
            log_pf_vec = log_pf_behavior

            # 已完成图强制停留，log_pf 增量为 0。
            actions = torch.where(graph_done, stop_indices, actions)
            log_pf_vec = torch.where(graph_done, torch.zeros_like(log_pf_vec), log_pf_vec)

            log_pf = log_pf + log_pf_vec
            log_pf_steps[:, step] = log_pf_vec
            actions_seq[:, step] = actions
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
            "log_pf_steps": log_pf_steps,
            "actions_seq": actions_seq,
            "state_emb_seq": state_emb_seq,
            "selected_mask": state.selected_mask,
            "selection_order": state.selection_order,
            "actions": actions,
            "reach_fraction": reach_fraction,
            "reach_success": reach_success.float(),
            "length": length,
        }
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
