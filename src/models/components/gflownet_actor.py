from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import inspect
import contextlib

import torch
from torch import nn
from torch_scatter import scatter_max

from src.models.components.gflownet_env import GraphEnv, STOP_RELATION
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
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.state_encoder = state_encoder
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)
        self._policy_accepts_edge_base = False
        self._assert_edge_policy()

    def _assert_edge_policy(self) -> None:
        sig = inspect.signature(self.policy.forward)
        params = set(sig.parameters.keys())
        if "relation_repr" in params or "relation_batch" in params:
            raise ValueError(
                "GFlowNetActor expects an edge-level policy in edge-action mode; " "relation-level policies are not supported."
            )
        self._policy_accepts_edge_base = "edge_base" in params

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
        return_log_probs: bool = False,
        return_valid_edges: bool = False,
        return_state_tokens: bool = False,
        dag_edge_mask: Optional[torch.Tensor] = None,
        return_bc_stats: bool = False,
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
            torch.autocast(device_type=device.type, enabled=torch.is_autocast_enabled())
            if device.type != "cpu"
            else contextlib.nullcontext()
        )
        edge_base = None
        if self._policy_accepts_edge_base and hasattr(self.policy, "compute_edge_base"):
            with autocast_ctx:
                edge_base = self.policy.compute_edge_base(edge_tokens_f)

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
            dummy_mask = getattr(batch, "is_dummy_agent", None)
            if torch.is_tensor(dummy_mask):
                graph_dict["dummy_mask"] = dummy_mask.to(device=device, dtype=torch.bool).view(-1)
        if state_encoder_cache is None:
            with autocast_ctx:
                encoder_cache = self.state_encoder.precompute(
                    node_ptr=node_ptr,
                    node_tokens=node_tokens,
                    question_tokens=question_tokens_f,
                    edge_index=edge_index,
                    start_node_locals=graph_dict.get("start_node_locals"),
                )
        else:
            encoder_cache = state_encoder_cache

        cache_state = graph_cache is not None and (not torch.is_grad_enabled())
        if graph_cache is not None and torch.is_grad_enabled():
            graph_cache.pop("state_cache", None)
        state = self.env.reset(graph_dict, device=device)
        if cache_state:
            graph_cache["state_cache"] = state

        log_pf_total = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        num_steps = self.max_steps + 1
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        state_emb_steps: list[torch.Tensor] = []
        state_tokens_steps: list[torch.Tensor] = []
        log_prob_edge_steps: list[torch.Tensor] = []
        log_prob_stop_steps: list[torch.Tensor] = []
        valid_edges_steps: list[torch.Tensor] = []
        bc_loss_sum = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_step_counts = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        bc_has_dag = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        if return_bc_stats:
            if dag_edge_mask is None:
                raise ValueError("return_bc_stats requires dag_edge_mask.")
            dag_edge_mask = dag_edge_mask.to(device=device, dtype=torch.bool).view(-1)
            if dag_edge_mask.numel() != edge_tokens_f.size(0):
                raise ValueError("dag_edge_mask length mismatch with edge_tokens.")
            dag_counts = torch.zeros(num_graphs, device=device, dtype=torch.float32)
            dag_counts.index_add_(0, edge_batch, dag_edge_mask.to(dtype=torch.float32))
            bc_has_dag = (dag_counts > 0).to(dtype=dag_counts.dtype)
        actions_seq = torch.full((num_graphs, num_steps), STOP_RELATION, dtype=torch.long, device=device)

        for step in range(num_steps):
            with autocast_ctx:
                state_tokens = self.state_encoder.encode_state(cache=encoder_cache, state=state)
            if not torch.isfinite(state_tokens).all():
                bad = (~torch.isfinite(state_tokens)).sum().item()
                raise ValueError(f"state_tokens contains {bad} non-finite values.")
            if return_state_tokens:
                state_tokens_steps.append(state_tokens)

            forward_mask, backward_mask = self.env.candidate_edge_masks(state)
            unused_edges = ~state.used_edge_mask
            forward_mask = forward_mask & unused_edges
            backward_mask = backward_mask & unused_edges
            candidate_edges = forward_mask | backward_mask
            valid_edges = candidate_edges

            policy_kwargs = {}
            if self._policy_accepts_edge_base:
                policy_kwargs["edge_base"] = edge_base
            with autocast_ctx:
                edge_logits, stop_logits, state_out = self.policy(
                    edge_tokens=edge_tokens_f,
                    state_tokens=state_tokens,
                    edge_batch=edge_batch,
                    valid_edges_mask=valid_edges,
                    **policy_kwargs,
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
            if return_log_probs:
                log_prob_edge_steps.append(log_prob_edge)
                log_prob_stop_steps.append(log_prob_stop)
            if return_valid_edges:
                valid_edges_steps.append(valid_edges)
            if return_bc_stats:
                step_loss, step_counts = self._compute_step_bc_stats(
                    log_prob_edge=log_prob_edge,
                    valid_edges=valid_edges,
                    dag_edge_mask=dag_edge_mask,
                    edge_batch=edge_batch,
                    num_graphs=num_graphs,
                )
                bc_loss_sum = bc_loss_sum + step_loss
                bc_step_counts = bc_step_counts + step_counts

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
            if bool(state.done.all().item()):
                break

        reach_success = state.answer_hits.float()
        length = state.step_counts.to(dtype=torch.float32)

        if len(state_emb_steps) < num_steps:
            pad = num_steps - len(state_emb_steps)
            if state_emb_steps:
                state_emb_steps.extend([state_emb_steps[-1]] * pad)
            else:
                filler = torch.zeros(
                    num_graphs,
                    edge_tokens_f.size(-1),
                    device=device,
                    dtype=torch.float32,
                )
                state_emb_steps.extend([filler] * num_steps)
        state_emb_seq = torch.stack(state_emb_steps, dim=1)
        result = {
            "log_pf": log_pf_total,
            "log_pf_steps": log_pf_steps,
            "state_emb_seq": state_emb_seq,
            "actions_seq": actions_seq,
            "directions_seq": state.directions,
            "selected_mask": state.used_edge_mask,
            "actions": actions,
            "reach_fraction": reach_success,
            "reach_success": reach_success,
            "length": length,
            "answer_node_hit": state.answer_node_hit,
            "start_node_hit": state.start_node_hit,
            "active_nodes": state.active_nodes,
        }
        if return_log_probs:
            if log_prob_edge_steps:
                if len(log_prob_edge_steps) < num_steps:
                    pad = num_steps - len(log_prob_edge_steps)
                    log_prob_edge_steps.extend([log_prob_edge_steps[-1]] * pad)
                result["log_prob_edge_seq"] = torch.stack(log_prob_edge_steps, dim=0)
            else:
                result["log_prob_edge_seq"] = torch.zeros(0, device=device, dtype=torch.float32)
            if log_prob_stop_steps:
                if len(log_prob_stop_steps) < num_steps:
                    pad = num_steps - len(log_prob_stop_steps)
                    log_prob_stop_steps.extend([log_prob_stop_steps[-1]] * pad)
                result["log_prob_stop_seq"] = torch.stack(log_prob_stop_steps, dim=1)
            else:
                result["log_prob_stop_seq"] = torch.zeros(0, device=device, dtype=torch.float32)
        if return_valid_edges:
            if valid_edges_steps:
                if len(valid_edges_steps) < num_steps:
                    pad = num_steps - len(valid_edges_steps)
                    valid_edges_steps.extend([valid_edges_steps[-1]] * pad)
                result["valid_edges_seq"] = torch.stack(valid_edges_steps, dim=0)
            else:
                result["valid_edges_seq"] = torch.zeros(0, device=device, dtype=torch.bool)
        if return_state_tokens:
            if state_tokens_steps:
                if len(state_tokens_steps) < num_steps:
                    pad = num_steps - len(state_tokens_steps)
                    state_tokens_steps.extend([state_tokens_steps[-1]] * pad)
                result["state_tokens_seq"] = torch.stack(state_tokens_steps, dim=1)
            else:
                result["state_tokens_seq"] = torch.zeros(0, device=device, dtype=torch.float32)
        if return_bc_stats:
            bc_loss = bc_loss_sum / bc_step_counts.clamp(min=1.0)
            result["bc_loss_per_graph"] = bc_loss
            result["bc_steps_per_graph"] = bc_step_counts
            result["bc_has_dag"] = bc_has_dag
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
        if edge_logits.dtype != torch.float32:
            edge_logits = edge_logits.to(dtype=torch.float32)
        if stop_logits.dtype != torch.float32:
            stop_logits = stop_logits.to(dtype=torch.float32)
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

    @staticmethod
    def _compute_step_bc_stats(
        *,
        log_prob_edge: torch.Tensor,
        valid_edges: torch.Tensor,
        dag_edge_mask: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = valid_edges & dag_edge_mask
        device = log_prob_edge.device
        if not bool(mask.any().item()):
            zeros = torch.zeros(num_graphs, device=device, dtype=log_prob_edge.dtype)
            return zeros, zeros
        masked_logits = log_prob_edge[mask]
        segments = edge_batch[mask]
        logsumexp = _segment_logsumexp_1d(masked_logits, segments, num_graphs)
        counts = torch.bincount(segments, minlength=num_graphs).to(device=device)
        has_valid = counts > 0
        step_loss = torch.where(has_valid, -logsumexp, torch.zeros_like(logsumexp))
        step_counts = has_valid.to(dtype=step_loss.dtype)
        return step_loss, step_counts


__all__ = ["GFlowNetActor"]
