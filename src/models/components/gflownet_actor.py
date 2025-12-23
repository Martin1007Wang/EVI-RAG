from __future__ import annotations

from typing import Any, Dict, Optional
import contextlib

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max

from src.models.components import GraphEnv
from src.models.components.gflownet_state_encoder import GNNStateEncoder, GNNStateEncoderCache
from src.utils.pylogger import RankedLogger

debug_logger = RankedLogger("gflownet.debug", rank_zero_only=True)

MIN_TEMPERATURE = 1e-5
DIR_UNKNOWN = 0
DIR_FORWARD = 1
DIR_REVERSE = 2


def _neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


class GFlowNetActor(nn.Module):
    """封装策略前向与环境 roll-out。"""

    def __init__(
        self,
        *,
        policy: nn.Module,
        env: GraphEnv,
        state_encoder: GNNStateEncoder,
        max_steps: int,
        policy_temperature: float,
        action_topk: Optional[int] = None,
        debug: bool,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.state_encoder = state_encoder
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)
        self.action_topk = None if action_topk is None else int(action_topk)
        if self.action_topk is not None and self.action_topk <= 0:
            raise ValueError(f"action_topk must be positive when set, got {self.action_topk}")
        self.debug = bool(debug)
        self.debug_steps = 1

    @staticmethod
    def _select_topk_candidate_edges(
        *,
        candidate_edges: torch.Tensor,  # [E_cand]
        edge_scores: torch.Tensor,      # [E_total]
        edge_batch: torch.Tensor,       # [E_total]
        num_graphs: int,
        k: int,
    ) -> torch.Tensor:
        """Select up to top-k candidate edges per graph by `edge_scores` (descending)."""
        candidate_edges = candidate_edges.to(device=edge_scores.device, dtype=torch.long).view(-1)
        if candidate_edges.numel() == 0:
            return candidate_edges
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if (candidate_edges < 0).any() or (candidate_edges >= int(edge_scores.numel())).any():
            raise ValueError("candidate_edges contains out-of-range indices.")

        scores = edge_scores[candidate_edges].to(dtype=torch.float32)
        seg = edge_batch[candidate_edges].to(device=edge_scores.device, dtype=torch.long)
        if seg.numel() != candidate_edges.numel():
            raise ValueError("candidate_edges indexing mismatch for edge_batch.")

        order_score = torch.argsort(scores, descending=True)
        cand_sorted = candidate_edges[order_score]
        seg_sorted = seg[order_score]

        order_seg = torch.argsort(seg_sorted, stable=True)
        cand_sorted = cand_sorted[order_seg]
        seg_sorted = seg_sorted[order_seg]

        counts = torch.bincount(seg_sorted, minlength=num_graphs)
        start = torch.zeros(num_graphs, device=edge_scores.device, dtype=torch.long)
        if num_graphs > 1:
            start[1:] = counts.cumsum(0)[:-1]
        rank = torch.arange(cand_sorted.numel(), device=edge_scores.device, dtype=torch.long) - start[seg_sorted]
        keep = rank < int(k)
        return cand_sorted[keep]

    def rollout(
        self,
        *,
        batch: Any,
        edge_tokens: torch.Tensor,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        temperature: Optional[float] = None,
        batch_idx: Optional[int] = None,
        graph_cache: Optional[Dict[str, torch.Tensor]] = None,
        forced_actions_seq: Optional[torch.Tensor] = None,
        state_encoder_cache: Optional[GNNStateEncoderCache] = None,
    ) -> Dict[str, torch.Tensor]:
        # 显式温度控制：由上层 cfg 决定，0 表示贪心近似。
        base_temperature = self.policy_temperature if temperature is None else float(temperature)
        is_greedy = base_temperature < MIN_TEMPERATURE
        temperature = max(base_temperature, MIN_TEMPERATURE)
        # NOTE: When evaluating with greedy sampling (temperature≈0), computing log-probs with an equally tiny
        # temperature makes log_pf numerically ill-conditioned (scales ~1/T). For metrics we only need the greedy
        # actions, while SubTB loss/log_pf diagnostics remain meaningful under the training temperature.
        logprob_temperature = temperature
        if temperature is not None and is_greedy:
            logprob_temperature = max(self.policy_temperature, MIN_TEMPERATURE)

        device = edge_tokens.device
        num_graphs = int(node_ptr.numel() - 1)
        edge_tokens_f = edge_tokens.to(device=device, dtype=torch.float32)
        node_tokens_f = node_tokens.to(device=device, dtype=torch.float32)
        question_tokens_f = question_tokens.to(device=device, dtype=torch.float32)
        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False)
            if device.type != "cpu"
            else contextlib.nullcontext()
        )

        if graph_cache is not None:
            graph_dict = graph_cache
            node_global_ids = graph_cache["node_global_ids"]
            edge_index = graph_cache["edge_index"]
            heads_global = graph_cache["heads_global"]
            tails_global = graph_cache["tails_global"]
            start_node_locals = graph_cache["start_node_locals"]
            start_ptr = graph_cache["start_ptr"]
        else:
            # 预计算静态张量，避免循环内重复搬运/索引
            node_global_ids = batch.node_global_ids.to(device)
            edge_index = batch.edge_index.to(device)
            heads_global = node_global_ids[edge_index[0]]
            tails_global = node_global_ids[edge_index[1]]
            start_node_locals = batch.start_node_locals.to(device)
            start_ptr = batch._slice_dict["start_node_locals"].to(device)

            graph_dict = {
                "edge_index": edge_index,
                "edge_batch": edge_batch,
                "node_global_ids": node_global_ids,
                "heads_global": heads_global,
                "tails_global": tails_global,
                "edge_scores": batch.edge_scores.to(device=device, dtype=torch.float32).view(-1),
                "node_ptr": node_ptr,
                "edge_ptr": edge_ptr,
                "start_node_locals": start_node_locals,
                "start_ptr": start_ptr,
                "answer_node_locals": batch.answer_node_locals.to(device),
                "answer_ptr": batch._slice_dict["answer_node_locals"].to(device),
            }
        if state_encoder_cache is None:
            with autocast_ctx:
                encoder_cache = self.state_encoder.precompute(
                    edge_index=edge_index,
                    edge_batch=edge_batch,
                    node_ptr=node_ptr,
                    start_node_locals=start_node_locals,
                    start_ptr=start_ptr,
                    node_tokens=node_tokens_f,
                    edge_tokens=edge_tokens_f,
                    question_tokens=question_tokens_f,
                )
        else:
            encoder_cache = state_encoder_cache

        state = self.env.reset(graph_dict, device=device)

        log_pf_total = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        num_steps = self.max_steps + 1  # max_edges + 1 stop step
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        state_emb_steps: list[torch.Tensor] = []
        actions_seq = torch.full((num_graphs, num_steps), -1, dtype=torch.long, device=device)
        action_dirs_seq = torch.full((num_graphs, num_steps), DIR_UNKNOWN, dtype=torch.long, device=device)
        phi_states = torch.zeros(num_graphs, num_steps + 1, dtype=torch.float32, device=device)
        if forced_actions_seq is not None:
            forced_actions_seq = forced_actions_seq.to(device=device, dtype=torch.long)
            if forced_actions_seq.shape != (num_graphs, num_steps):
                raise ValueError(
                    f"forced_actions_seq must have shape (B,T)=({num_graphs},{num_steps}), got {tuple(forced_actions_seq.shape)}"
                )

        debug_logged: bool = (not self.debug) or (batch_idx is not None and batch_idx != 0)

        stop_indices = edge_ptr[1:]
        actions = stop_indices.clone()

        for step in range(num_steps):
            with autocast_ctx:
                state_tokens = self.state_encoder.encode_state(cache=encoder_cache, state=state)
            state_emb_steps.append(state_tokens.to(dtype=torch.float32))
            if not torch.isfinite(state_tokens).all():
                bad = (~torch.isfinite(state_tokens)).sum().item()
                raise ValueError(f"state_tokens contains {bad} non-finite values.")

            action_mask_edges = self.env.action_mask_edges(state)
            frontier_mask_edges = self.env.frontier_mask_edges(state)
            graph_done = state.done
            candidate_mask = action_mask_edges & (~state.selected_mask) & (~graph_done[edge_batch])

            forced_step: Optional[torch.Tensor] = None
            if forced_actions_seq is not None:
                forced_step = forced_actions_seq[:, step]
                forced_step = torch.where(graph_done, stop_indices, forced_step)
                if forced_step.shape != (num_graphs,):
                    raise ValueError(f"forced_actions_seq[:, {step}] must be [B], got {tuple(forced_step.shape)}")
                es = edge_ptr[:-1]
                ee = edge_ptr[1:]
                invalid_low = forced_step < es
                invalid_high = forced_step > ee
                if bool((invalid_low | invalid_high).any().item()):
                    bad = torch.nonzero(invalid_low | invalid_high, as_tuple=False).view(-1)
                    preview = []
                    for idx in bad[:5].tolist():
                        preview.append(
                            f"(g={idx} action={int(forced_step[idx].item())} es={int(es[idx].item())} ee={int(ee[idx].item())})"
                        )
                    raise ValueError(
                        "forced_actions_seq contains out-of-range indices for per-graph edge slices; "
                        f"bad_examples={', '.join(preview)}"
                    )

            pruned_mask = candidate_mask
            if self.action_topk is not None and bool(candidate_mask.any().item()):
                cand_edges = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
                topk_edges = self._select_topk_candidate_edges(
                    candidate_edges=cand_edges,
                    edge_scores=state.graph.edge_scores_z,
                    edge_batch=edge_batch,
                    num_graphs=num_graphs,
                    k=self.action_topk,
                )
                pruned_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
                if topk_edges.numel() > 0:
                    pruned_mask[topk_edges] = True
                if forced_step is not None:
                    is_stop_forced = forced_step == stop_indices
                    forced_edges = forced_step[~is_stop_forced]
                    if forced_edges.numel() > 0:
                        pruned_mask[forced_edges] = True
                pruned_mask = pruned_mask & candidate_mask

            phi_states[:, step] = self.env.potential(state, valid_edges_override=pruned_mask).detach()
            edge_direction = self._compute_edge_directions(state)
            with autocast_ctx:
                edge_logits, stop_logits, _state_emb = self.policy(
                    edge_tokens_f,
                    question_tokens_f,
                    state_tokens,
                    edge_batch,
                    state.selected_mask,
                    selection_order=state.selection_order,
                    edge_heads=heads_global,
                    edge_tails=tails_global,
                    current_tail=state.current_tail,
                    frontier_mask=frontier_mask_edges,
                    edge_direction=edge_direction,
                    valid_edges_mask=pruned_mask,
                )
            if not torch.isfinite(edge_logits).all():
                bad = (~torch.isfinite(edge_logits)).sum().item()
                raise ValueError(f"edge_logits contains {bad} non-finite values.")
            if not torch.isfinite(stop_logits).all():
                bad = (~torch.isfinite(stop_logits)).sum().item()
                raise ValueError(f"stop_logits contains {bad} non-finite values.")

            if not debug_logged:
                debug_logged = self._log_debug_actions(
                    step=step,
                    batch_idx=batch_idx,
                    edge_ptr=edge_ptr,
                    graph_dict=graph_dict,
                    action_mask_edges=action_mask_edges,
                    edge_logits=edge_logits,
                    stop_logits=stop_logits,
                    state=state,
                    batch=batch,
                )

            # 有效 mask：屏蔽已完成图与非法边，避免 per-graph Python 循环。
            # stop 索引定义为该图片段结束位置（合法范围内的虚拟停止），env.step 必须识别为 stop。
            valid_edges = pruned_mask

            # Candidate edges for this step (across the whole PyG batch).
            cand_edges = torch.nonzero(valid_edges, as_tuple=False).view(-1) if valid_edges.any() else torch.empty(0, dtype=torch.long, device=device)
            cand_batch, log_cand, log_stop, log_denom = self._log_probs_from_logits_sparse(
                edge_logits=edge_logits,
                stop_logits=stop_logits,
                candidate_edges=cand_edges,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
                temp=logprob_temperature,
                suppress_stop=False,
            )
            has_valid_edge = (torch.bincount(cand_batch, minlength=num_graphs) > 0) if cand_batch.numel() > 0 else torch.zeros(num_graphs, device=device, dtype=torch.bool)

            if forced_actions_seq is None:
                if cand_edges.numel() == 0:
                    actions = stop_indices
                    log_pf_vec = log_stop
                else:
                    # 采样：温度极低时走贪心，否则走 Gumbel-Max（直接对 log-prob 采样）。
                    if is_greedy:
                        score_edges = log_cand
                        score_stop = log_stop
                    else:
                        score_edges = log_cand + self._gumbel_like(log_cand)
                        score_stop = log_stop + self._gumbel_like(log_stop)
                    score_edges_max, cand_argmax = scatter_max(score_edges, cand_batch, dim=0, dim_size=num_graphs)
                    score_edges_max = torch.where(
                        has_valid_edge, score_edges_max, torch.full_like(score_edges_max, _neg_inf_value(score_edges_max))
                    )
                    cand_argmax = torch.where(has_valid_edge, cand_argmax, torch.zeros_like(cand_argmax))

                    choose_edge = has_valid_edge & (score_edges_max > score_stop)
                    edge_argmax = cand_edges[cand_argmax]
                    actions = torch.where(choose_edge, edge_argmax, stop_indices)
                    log_pf_vec = torch.where(choose_edge, log_cand[cand_argmax], log_stop)
            else:
                if forced_step is None:
                    raise RuntimeError("forced_step must be computed when forced_actions_seq is provided.")

                is_stop_forced = forced_step == stop_indices
                log_pf_step = log_stop.clone()
                edge_mask_forced = ~is_stop_forced
                if edge_mask_forced.any():
                    edge_actions = forced_step[edge_mask_forced]
                    if not valid_edges[edge_actions].all():
                        raise ValueError("forced_actions_seq selects edges violating action_mask_edges.")
                    graph_ids = edge_batch[edge_actions].to(device=device, dtype=torch.long)
                    edge_logits_f = edge_logits.to(device=device, dtype=torch.float32)
                    scaled_edge = edge_logits_f[edge_actions] / float(logprob_temperature)
                    log_pf_step[edge_mask_forced] = scaled_edge - log_denom[graph_ids]
                actions = forced_step
                log_pf_vec = log_pf_step
            action_dirs = self._gather_action_directions(
                actions=actions,
                edge_direction=edge_direction,
                stop_indices=stop_indices,
            )
            action_dirs_seq[:, step] = action_dirs

            # 已完成图强制停留，log_pf 增量为 0。
            actions = torch.where(graph_done, stop_indices, actions)
            log_pf_vec = torch.where(graph_done, torch.zeros_like(log_pf_vec), log_pf_vec)

            log_pf_total = log_pf_total + log_pf_vec
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

        if len(state_emb_steps) != num_steps:
            raise RuntimeError(f"state_emb_steps length {len(state_emb_steps)} != num_steps {num_steps}")
        state_emb_seq = torch.stack(state_emb_steps, dim=1)
        result = {
            "log_pf": log_pf_total,
            "log_pf_steps": log_pf_steps,
            "state_emb_seq": state_emb_seq,
            "actions_seq": actions_seq,
            "action_directions": action_dirs_seq,
            "selected_mask": state.selected_mask,
            "selection_order": state.selection_order,
            "actions": actions,
            "reach_fraction": reach_fraction,
            "reach_success": reach_success.float(),
            "length": length,
            "phi_states": phi_states,
        }
        return result

    @staticmethod
    def _gather_action_directions(
        *,
        actions: torch.Tensor,
        edge_direction: torch.Tensor,
        stop_indices: torch.Tensor,
    ) -> torch.Tensor:
        action_dirs = torch.full_like(actions, DIR_UNKNOWN, dtype=torch.long)
        valid = (actions != stop_indices) & (actions >= 0) & (actions < int(edge_direction.numel()))
        if valid.any():
            action_dirs[valid] = edge_direction[actions[valid]]
        return action_dirs

    @staticmethod
    def _compute_edge_directions(state: "GraphState") -> torch.Tensor:
        device = state.graph.edge_index.device
        edge_index = state.graph.edge_index
        heads_local = edge_index[0]
        tails_local = edge_index[1]
        head_is_start = state.graph.node_is_start[heads_local]
        tail_is_start = state.graph.node_is_start[tails_local]
        dir_step0 = torch.full((edge_index.size(1),), DIR_UNKNOWN, device=device, dtype=torch.long)
        dir_step0 = torch.where(head_is_start, torch.full_like(dir_step0, DIR_FORWARD), dir_step0)
        if not state.graph.directed:
            dir_step0 = torch.where((~head_is_start) & tail_is_start, torch.full_like(dir_step0, DIR_REVERSE), dir_step0)

        edge_batch = state.graph.edge_batch
        current_tail = state.current_tail[edge_batch]
        heads_global = state.graph.heads_global
        tails_global = state.graph.tails_global
        dir_step = torch.full_like(dir_step0, DIR_UNKNOWN)
        dir_step = torch.where(heads_global == current_tail, torch.full_like(dir_step, DIR_FORWARD), dir_step)
        if not state.graph.directed:
            dir_step = torch.where(tails_global == current_tail, torch.full_like(dir_step, DIR_REVERSE), dir_step)

        is_step0 = state.step_counts[edge_batch] == 0
        return torch.where(is_step0, dir_step0, dir_step)

    @staticmethod
    def _log_probs_from_logits(
        *,
        edge_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        valid_edges: torch.Tensor,
        temp: float,
        edge_batch: torch.Tensor,
        num_graphs: int,
        suppress_stop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if temp <= 0.0:
            raise ValueError("temp must be positive.")
        edge_logits_f = edge_logits.to(dtype=torch.float32)
        stop_logits_f = stop_logits.to(dtype=torch.float32)
        neg_inf = _neg_inf_value(edge_logits_f)
        masked_edge_logits = edge_logits_f.masked_fill(~valid_edges, neg_inf)
        scaled_edge_logits = masked_edge_logits / temp
        scaled_stop_logits = stop_logits_f / temp
        if suppress_stop:
            scaled_stop_logits = torch.full_like(scaled_stop_logits, neg_inf)

        max_edge, _ = scatter_max(scaled_edge_logits, edge_batch, dim=0, dim_size=num_graphs)
        has_edge = torch.bincount(edge_batch, minlength=num_graphs) > 0
        max_edge = torch.where(has_edge, max_edge, torch.full_like(max_edge, neg_inf))
        max_joint = torch.maximum(max_edge, scaled_stop_logits)

        exp_edges = scatter_add(
            torch.exp(scaled_edge_logits - max_joint[edge_batch]),
            edge_batch,
            dim=0,
            dim_size=num_graphs,
        )
        exp_stop = torch.exp(scaled_stop_logits - max_joint)
        eps = torch.finfo(edge_logits_f.dtype).eps
        log_denom = max_joint + torch.log(exp_edges + exp_stop + eps)

        log_edge = scaled_edge_logits - log_denom[edge_batch]
        log_stop = scaled_stop_logits - log_denom
        log_edge = log_edge.masked_fill(~valid_edges, neg_inf)
        return log_edge, log_stop

    @staticmethod
    def _log_probs_from_logits_sparse(
        *,
        edge_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        candidate_edges: torch.Tensor,  # [E_cand]
        edge_batch: torch.Tensor,       # [E_total]
        num_graphs: int,
        temp: float,
        suppress_stop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparse log-prob computation over candidate edges only.

        Returns:
          - cand_batch: [E_cand] graph id per candidate edge
          - log_cand:   [E_cand] log P(edge|graph)
          - log_stop:   [B]      log P(stop|graph)
          - log_denom:  [B]      log normalizer in scaled-logit space
        """
        if temp <= 0.0:
            raise ValueError("temp must be positive.")
        device = edge_logits.device
        edge_logits_f = edge_logits.to(device=device, dtype=torch.float32).view(-1)
        stop_logits_f = stop_logits.to(device=device, dtype=torch.float32).view(-1)
        if stop_logits_f.numel() != num_graphs:
            raise ValueError(f"stop_logits must have shape [B]=({num_graphs},), got {tuple(stop_logits_f.shape)}")
        candidate_edges = candidate_edges.to(device=device, dtype=torch.long).view(-1)
        neg_inf = _neg_inf_value(edge_logits_f)

        scaled_stop = stop_logits_f / float(temp)
        if suppress_stop:
            scaled_stop = torch.full_like(scaled_stop, neg_inf)

        if candidate_edges.numel() == 0:
            max_joint = scaled_stop
            exp_stop = torch.exp(scaled_stop - max_joint)
            eps = torch.finfo(edge_logits_f.dtype).eps
            log_denom = max_joint + torch.log(exp_stop + eps)
            log_stop = scaled_stop - log_denom
            return (
                torch.empty(0, device=device, dtype=torch.long),
                torch.empty(0, device=device, dtype=edge_logits_f.dtype),
                log_stop,
                log_denom,
            )

        if (candidate_edges < 0).any() or (candidate_edges >= int(edge_logits_f.numel())).any():
            raise ValueError("candidate_edges contains out-of-range indices.")

        cand_batch = edge_batch[candidate_edges].to(device=device, dtype=torch.long)
        if cand_batch.numel() != candidate_edges.numel():
            raise ValueError("candidate_edges indexing mismatch for edge_batch.")
        scaled_cand = edge_logits_f[candidate_edges] / float(temp)

        has_edge = torch.bincount(cand_batch, minlength=num_graphs) > 0
        max_edge, _ = scatter_max(scaled_cand, cand_batch, dim=0, dim_size=num_graphs)
        max_edge = torch.where(has_edge, max_edge, torch.full_like(max_edge, neg_inf))
        max_joint = torch.maximum(max_edge, scaled_stop)

        exp_edges = torch.exp(scaled_cand - max_joint[cand_batch])
        exp_sum = scatter_add(exp_edges, cand_batch, dim=0, dim_size=num_graphs)
        exp_stop = torch.exp(scaled_stop - max_joint)
        eps = torch.finfo(edge_logits_f.dtype).eps
        log_denom = max_joint + torch.log(exp_sum + exp_stop + eps)

        log_cand = scaled_cand - log_denom[cand_batch]
        log_stop = scaled_stop - log_denom
        return cand_batch, log_cand, log_stop, log_denom

    def _log_debug_actions(
        self,
        *,
        step: int,
        batch_idx: Optional[int],
        edge_ptr: torch.Tensor,
        graph_dict: Dict[str, torch.Tensor],
        action_mask_edges: torch.Tensor,
        edge_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        state: Any,
        batch: Any,
    ) -> bool:
        """在首个 batch 内打印首图的动作信息，避免主循环膨胀。"""
        if (
            not self.debug
            or step >= self.debug_steps
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
                masked_logits = local_logits.masked_fill(~mask_slice, _neg_inf_value(local_logits))
                top_local_idx = int(torch.argmax(masked_logits).item())
                edge_id = start_idx + top_local_idx
                head_id = int(batch.node_global_ids[batch.edge_index[0, edge_id]].item())
                tail_id = int(batch.node_global_ids[batch.edge_index[1, edge_id]].item())
                debug_logger.info(
                    "[DEBUG_ACTION] g0 top_edge=%d head=%d tail=%d edge_logit=%.4g stop_logit=%.4g masked=%s",
                    edge_id,
                    head_id,
                    tail_id,
                    float(local_logits[top_local_idx].item()),
                    float(stop_logits[g_idx].item()),
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
