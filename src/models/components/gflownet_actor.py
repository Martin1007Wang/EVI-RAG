from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import inspect
import contextlib

import torch
from torch import nn
from torch_scatter import scatter_max

from src.models.components.gflownet_env import (
    GraphEnv,
    STOP_RELATION,
    DIRECTION_FORWARD,
    DIRECTION_BACKWARD,
)
from src.models.components.state_encoder import StateEncoder, StateEncoderCache

MIN_TEMPERATURE = 1e-5


def _neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def _segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.dim() != 1 or segment_ids.dim() != 1:
        raise ValueError("logits and segment_ids must be 1D tensors.")
    if logits.numel() != segment_ids.numel():
        raise ValueError("logits and segment_ids must have the same length.")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if logits.numel() == 0:
        return torch.full((num_segments,), _neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    dtype = logits.dtype
    neg_inf = torch.finfo(dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


class GFlowNetActor(nn.Module):
    """Edge-level GFlowNet actor."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        env: GraphEnv,
        state_encoder: StateEncoder,
        max_steps: int,
        policy_temperature: float,
        action_topk: Optional[int] = None,
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
        self._assert_edge_policy()

    def _assert_edge_policy(self) -> None:
        sig = inspect.signature(self.policy.forward)
        params = set(sig.parameters.keys())
        if "relation_repr" in params or "relation_batch" in params:
            raise ValueError(
                "GFlowNetActor expects an edge-level policy in edge-action mode; " "relation-level policies are not supported."
            )

    @staticmethod
    def _select_topk_candidate_edges(
        *,
        candidate_edges: torch.Tensor,  # [E_cand]
        edge_scores: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        num_graphs: int,
        k: int,
    ) -> torch.Tensor:
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
        state_encoder_cache: Optional[StateEncoderCache] = None,
    ) -> Dict[str, torch.Tensor]:
        base_temperature = self.policy_temperature if temperature is None else float(temperature)
        is_greedy = base_temperature < MIN_TEMPERATURE
        temperature = max(base_temperature, MIN_TEMPERATURE)
        logprob_temperature = temperature

        device = edge_tokens.device
        num_graphs = int(node_ptr.numel() - 1)
        edge_tokens_f = edge_tokens.to(device=device, dtype=torch.float32)
        question_tokens_f = question_tokens.to(device=device, dtype=torch.float32)
        autocast_ctx = (
            torch.autocast(device_type=device.type, enabled=False) if device.type != "cpu" else contextlib.nullcontext()
        )

        if graph_cache is not None:
            graph_dict = graph_cache
            edge_index = graph_cache["edge_index"]
            edge_relations = graph_cache["edge_relations"]
        else:
            edge_index = batch.edge_index.to(device)
            edge_relations = batch.edge_attr.to(device)
            graph_dict = {
                "edge_index": edge_index,
                "edge_batch": edge_batch,
                "edge_relations": edge_relations,
                "node_ptr": node_ptr,
                "edge_ptr": edge_ptr,
                "start_node_locals": batch.start_node_locals.to(device),
                "start_ptr": batch._slice_dict["start_node_locals"].to(device),
                "answer_node_locals": batch.answer_node_locals.to(device),
                "answer_ptr": batch._slice_dict["answer_node_locals"].to(device),
                "edge_scores": batch.edge_scores.to(device=device, dtype=torch.float32).view(-1),
            }
        if state_encoder_cache is None:
            with autocast_ctx:
                encoder_cache = self.state_encoder.precompute(
                    node_ptr=node_ptr,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens_f,
                )
        else:
            encoder_cache = state_encoder_cache

        state = self.env.reset(graph_dict, device=device)
        if state.action_embeddings.numel() == 0:
            state.action_embeddings = torch.zeros(
                num_graphs,
                self.max_steps,
                edge_tokens_f.size(-1),
                device=device,
                dtype=edge_tokens_f.dtype,
            )

        log_pf_total = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        num_steps = self.max_steps + 1
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        state_emb_steps: list[torch.Tensor] = []
        actions_seq = torch.full((num_graphs, num_steps), STOP_RELATION, dtype=torch.long, device=device)
        phi_states = torch.zeros(num_graphs, num_steps + 1, dtype=torch.float32, device=device)

        for step in range(num_steps):
            with autocast_ctx:
                state_tokens = self.state_encoder.encode_state(cache=encoder_cache, state=state)
            if not torch.isfinite(state_tokens).all():
                bad = (~torch.isfinite(state_tokens)).sum().item()
                raise ValueError(f"state_tokens contains {bad} non-finite values.")

            forward_mask, backward_mask = self.env.candidate_edge_masks(state)
            unused_edges = ~state.used_edge_mask
            forward_mask = forward_mask & unused_edges
            backward_mask = backward_mask & unused_edges
            candidate_edges = forward_mask | backward_mask
            pruned_mask = candidate_edges
            if self.action_topk is not None and bool(candidate_edges.any().item()):
                cand_edges = torch.nonzero(candidate_edges, as_tuple=False).view(-1)
                topk_edges = self._select_topk_candidate_edges(
                    candidate_edges=cand_edges,
                    edge_scores=state.graph.edge_scores_norm,
                    edge_batch=edge_batch,
                    num_graphs=num_graphs,
                    k=self.action_topk,
                )
                pruned_mask = torch.zeros_like(candidate_edges, dtype=torch.bool)
                if topk_edges.numel() > 0:
                    pruned_mask[topk_edges] = True
                pruned_mask = pruned_mask & candidate_edges
                forward_mask = forward_mask & pruned_mask
                backward_mask = backward_mask & pruned_mask
            valid_edges = pruned_mask

            phi_states[:, step] = self.env.potential(state, valid_edges_override=valid_edges).detach()

            edge_direction = torch.full((edge_index.size(1),), DIRECTION_FORWARD, device=device, dtype=torch.long)
            if backward_mask.any():
                edge_direction[backward_mask & (~forward_mask)] = DIRECTION_BACKWARD

            with autocast_ctx:
                edge_logits, stop_logits, state_out = self.policy(
                    edge_tokens=edge_tokens_f,
                    state_tokens=state_tokens,
                    edge_batch=edge_batch,
                    valid_edges_mask=valid_edges,
                    edge_direction=edge_direction,
                    question_tokens=question_tokens_f,
                )
            if not torch.isfinite(edge_logits).all():
                bad = (~torch.isfinite(edge_logits)).sum().item()
                raise ValueError(f"edge_logits contains {bad} non-finite values.")
            if edge_logits.numel() != edge_tokens_f.size(0):
                raise ValueError(f"edge_logits length {edge_logits.numel()} != num_edges {edge_tokens_f.size(0)}")
            if not torch.isfinite(stop_logits).all():
                bad = (~torch.isfinite(stop_logits)).sum().item()
                raise ValueError(f"stop_logits contains {bad} non-finite values.")
            if stop_logits.numel() != num_graphs:
                raise ValueError(f"stop_logits length {stop_logits.numel()} != num_graphs {num_graphs}")
            if state_out is None:
                state_out = state_tokens
            state_emb_steps.append(state_out.to(dtype=torch.float32))

            log_prob_edge, log_prob_stop, log_denom, has_edge = self._log_probs_edges(
                edge_logits=edge_logits,
                stop_logits=stop_logits,
                edge_batch=edge_batch,
                valid_edges=valid_edges,
                num_graphs=num_graphs,
                temp=logprob_temperature,
            )

            if forced_actions_seq is not None:
                forced = forced_actions_seq[:, step].to(device=device, dtype=torch.long)
                if forced.numel() != num_graphs:
                    raise ValueError("forced_actions_seq batch mismatch with num_graphs.")
                forced_stop = forced == STOP_RELATION
                if bool((~forced_stop).any().item()):
                    if edge_logits.numel() == 0:
                        raise ValueError("forced_actions_seq contains edge actions but edge_logits is empty.")
                    if ((forced[~forced_stop] < 0) | (forced[~forced_stop] >= edge_logits.numel())).any():
                        raise ValueError("forced_actions_seq contains out-of-range edge ids.")
                    graph_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
                    if (edge_batch[forced[~forced_stop]] != graph_ids[~forced_stop]).any():
                        raise ValueError("forced_actions_seq contains edges from the wrong graph.")
                    if (~valid_edges[forced[~forced_stop]]).any():
                        raise ValueError("forced_actions_seq contains edges not in the valid action set.")
                actions = forced
                log_pf_vec = log_prob_stop.clone()
                if bool((~forced_stop).any().item()):
                    log_pf_vec[~forced_stop] = log_prob_edge[forced[~forced_stop]]
            else:
                if is_greedy:
                    score_edge = log_prob_edge
                    score_stop = log_prob_stop
                else:
                    score_edge = log_prob_edge + self._gumbel_like(log_prob_edge)
                    score_stop = log_prob_stop + self._gumbel_like(log_prob_stop)

                neg_inf = _neg_inf_value(score_edge)
                score_edge = torch.where(valid_edges, score_edge, torch.full_like(score_edge, neg_inf))
                score_edge_max, edge_argmax = scatter_max(score_edge, edge_batch, dim=0, dim_size=num_graphs)
                score_edge_max = torch.where(
                    has_edge,
                    score_edge_max,
                    torch.full_like(score_edge_max, neg_inf),
                )
                edge_argmax = torch.where(has_edge, edge_argmax, torch.zeros_like(edge_argmax))

                choose_edge = has_edge & (score_edge_max > score_stop)
                actions = torch.where(choose_edge, edge_argmax, torch.full_like(edge_argmax, STOP_RELATION))
                log_pf_vec = torch.where(choose_edge, log_prob_edge[edge_argmax], log_prob_stop)

            if state.done.any():
                done_mask = state.done
                actions = torch.where(done_mask, torch.full_like(actions, STOP_RELATION), actions)
                log_pf_vec = torch.where(done_mask, torch.zeros_like(log_pf_vec), log_pf_vec)

            selected_action_emb = torch.zeros(
                num_graphs,
                edge_tokens_f.size(-1),
                device=device,
                dtype=edge_tokens_f.dtype,
            )
            edge_selected = actions >= 0
            if bool(edge_selected.any().item()):
                selected_action_emb[edge_selected] = edge_tokens_f[actions[edge_selected]]

            actions_seq[:, step] = actions
            log_pf_steps[:, step] = log_pf_vec
            log_pf_total = log_pf_total + log_pf_vec

            state = self.env.step(state, actions, selected_action_emb, step_index=step)

        reach_success = state.answer_hits.float()
        length = state.step_counts.to(dtype=torch.float32)

        if len(state_emb_steps) != num_steps:
            raise RuntimeError(f"state_emb_steps length {len(state_emb_steps)} != num_steps {num_steps}")
        state_emb_seq = torch.stack(state_emb_steps, dim=1)
        result = {
            "log_pf": log_pf_total,
            "log_pf_steps": log_pf_steps,
            "state_emb_seq": state_emb_seq,
            "actions_seq": actions_seq,
            "selected_mask": state.used_edge_mask,
            "actions": actions,
            "reach_fraction": reach_success,
            "reach_success": reach_success,
            "length": length,
            "phi_states": phi_states,
        }
        return result

    @staticmethod
    def _log_probs_edges(
        *,
        edge_logits: torch.Tensor,
        stop_logits: torch.Tensor,
        edge_batch: torch.Tensor,
        valid_edges: torch.Tensor,
        num_graphs: int,
        temp: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_graphs <= 0:
            return (
                torch.zeros(0, device=stop_logits.device),
                torch.zeros(0, device=stop_logits.device),
                torch.zeros(0, device=stop_logits.device),
                torch.zeros(0, device=stop_logits.device, dtype=torch.bool),
            )
        stop_scaled = stop_logits / float(temp)
        if edge_logits.numel() == 0:
            log_denom = stop_scaled
            has_edge = torch.zeros(num_graphs, device=stop_logits.device, dtype=torch.bool)
            return edge_logits, stop_scaled - log_denom, log_denom, has_edge

        edge_scaled = edge_logits / float(temp)
        valid_edges = valid_edges.to(device=edge_scaled.device, dtype=torch.bool)
        if bool(valid_edges.any().item()):
            logsumexp_edges = _segment_logsumexp_1d(edge_scaled[valid_edges], edge_batch[valid_edges], num_graphs)
        else:
            neg_inf = _neg_inf_value(edge_scaled)
            logsumexp_edges = torch.full((num_graphs,), neg_inf, device=edge_scaled.device, dtype=edge_scaled.dtype)
        log_denom = torch.logaddexp(logsumexp_edges, stop_scaled)
        log_prob_edge = edge_scaled - log_denom[edge_batch]
        log_prob_edge = torch.where(
            valid_edges,
            log_prob_edge,
            torch.full_like(log_prob_edge, _neg_inf_value(log_prob_edge)),
        )
        log_prob_stop = stop_scaled - log_denom
        has_edge = logsumexp_edges > _neg_inf_value(logsumexp_edges)
        return log_prob_edge, log_prob_stop, log_denom, has_edge

    @staticmethod
    def _gumbel_like(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        u = torch.rand_like(tensor)
        return -torch.log(-torch.log(u.clamp(min=eps, max=1.0 - eps)))


__all__ = ["GFlowNetActor"]
