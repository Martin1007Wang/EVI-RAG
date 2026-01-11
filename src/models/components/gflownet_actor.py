from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import contextlib

import torch
from torch import nn
from torch_scatter import scatter_max

from src.models.components.gflownet_env import GraphEnv, STOP_EDGE_RELATION, STOP_RELATION
from src.gfn.ops import compute_factorized_log_probs, gumbel_noise_like, neg_inf_value

MIN_TEMPERATURE = 1e-5
_STOP_STEP_OFFSET = 1
_STOP_NODE_NONE = -1
_DEFAULT_CHECK_FINITE = True
_STOP_LOGIT_DIM = 1
_EDGE_HEAD_INDEX = 0
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_NAN = float("nan")


@dataclass(frozen=True)
class RolloutResult:
    actions_seq: Optional[torch.Tensor]
    directions_seq: Optional[torch.Tensor]
    log_pf: torch.Tensor
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    selected_mask: torch.Tensor
    reach_success: torch.Tensor
    length: torch.Tensor
    stop_node_locals: torch.Tensor
    answer_node_hit: torch.Tensor
    start_node_hit: torch.Tensor
    visited_nodes: Optional[torch.Tensor]


@dataclass
class RolloutBuffers:
    actions_seq: Optional[torch.Tensor]
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor

    @classmethod
    def create(
        cls,
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        *,
        record_actions: bool,
    ) -> "RolloutBuffers":
        actions_seq = None
        if record_actions:
            actions_seq = torch.full(
                (num_graphs, num_steps),
                STOP_RELATION,
                dtype=torch.long,
                device=device,
            )
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        log_pb_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        return cls(actions_seq=actions_seq, log_pf_steps=log_pf_steps, log_pb_steps=log_pb_steps)


class GFlowNetActor(nn.Module):
    """Edge-level GFlowNet actor."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        env: GraphEnv,
        forward_head: nn.Module,
        backward_head: nn.Module,
        edge_forward_head: nn.Module,
        edge_backward_head: nn.Module,
        state_encoder: nn.Module,
        state_input_dim: int,
        hidden_dim: int,
        max_steps: int,
        policy_temperature: float,
        backward_temperature: Optional[float] = None,
        relation_use_active_nodes: bool = True,
        check_finite: bool = _DEFAULT_CHECK_FINITE,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.forward_head = forward_head
        self.backward_head = backward_head
        self.edge_forward_head = edge_forward_head
        self.edge_backward_head = edge_backward_head
        self.state_encoder = state_encoder
        self.stop_head = nn.Linear(int(state_input_dim), _STOP_LOGIT_DIM)
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)
        self.backward_temperature = self._resolve_backward_temperature(backward_temperature)
        self.relation_use_active_nodes = bool(relation_use_active_nodes)
        self.check_finite = bool(check_finite)

    def set_temperatures(
        self,
        *,
        policy_temperature: float,
        backward_temperature: Optional[float] = None,
    ) -> None:
        policy_temp = max(float(policy_temperature), MIN_TEMPERATURE)
        if backward_temperature is None:
            backward_temp = policy_temp
        else:
            backward_temp = max(float(backward_temperature), MIN_TEMPERATURE)
        self.policy_temperature = policy_temp
        self.backward_temperature = backward_temp

    def _assert_finite_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        *,
        step: Optional[int] = None,
    ) -> None:
        if not self.check_finite:
            return
        if torch.isfinite(tensor).all():
            return
        summary = self._summarize_non_finite(tensor)
        step_part = f" step={step}" if step is not None else ""
        raise RuntimeError(f"Non-finite detected in {name}{step_part}: {summary}")

    @staticmethod
    def _summarize_non_finite(tensor: torch.Tensor) -> str:
        finite = torch.isfinite(tensor)
        non_finite = int((~finite).sum().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_vals = tensor[finite]
        if finite_vals.numel() > _ZERO:
            calc = finite_vals.to(dtype=torch.float32)
            min_val = float(calc.min().item())
            max_val = float(calc.max().item())
            mean_val = float(calc.mean().item())
            abs_max = float(calc.abs().max().item())
        else:
            min_val = _NAN
            max_val = _NAN
            mean_val = _NAN
            abs_max = _NAN
        return (
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"non_finite={non_finite} (nan={nan_count}, inf={inf_count}), "
            f"finite_min={min_val}, finite_max={max_val}, finite_mean={mean_val}, abs_max={abs_max}"
        )

    def rollout(
        self,
        *,
        graph: dict[str, torch.Tensor],
        temperature: Optional[float] = None,
        record_actions: bool = True,
        record_visited: bool = True,
    ) -> RolloutResult:
        temp, is_greedy = self._resolve_temperature(temperature)
        edge_index = graph["edge_index"]
        device = edge_index.device
        state = self.env.reset(graph, device=device)
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        num_steps = self.max_steps + _STOP_STEP_OFFSET
        buffers = self._init_buffers(num_graphs, num_steps, device, record_actions=record_actions)
        log_pf_total = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        stop_node_locals = torch.full((num_graphs,), _STOP_NODE_NONE, device=device, dtype=torch.long)
        edge_batch = graph["edge_batch"]
        edge_relations = graph["edge_relations"]
        node_tokens = graph["node_tokens"]
        node_batch = graph["node_batch"]
        autocast_ctx = self._autocast_context(device)
        state_vec = self.state_encoder.init_state(num_graphs, device=device, dtype=node_tokens.dtype)

        for step in range(num_steps):
            pre_done = state.done
            valid_edges = self._valid_edges(state)
            allow_stop = self._allow_stop(state)
            relation_logits, edge_scores, stop_logits = self._score_forward_with_stop(
                graph,
                state_vec,
                active_nodes=state.active_nodes,
                autocast_ctx=autocast_ctx,
            )
            self._assert_finite_tensor(relation_logits, "relation_logits", step=step)
            self._assert_finite_tensor(edge_scores, "edge_scores", step=step)
            if stop_logits is not None:
                self._assert_finite_tensor(stop_logits, "stop_logits", step=step)
            edge_logits = self._policy_logits(
                valid_edges=valid_edges,
                edge_logits=edge_scores,
                autocast_ctx=autocast_ctx,
            )
            self._assert_finite_tensor(edge_logits, "edge_logits", step=step)
            log_prob_edge, log_prob_stop, _, has_edge = compute_factorized_log_probs(
                relation_logits=relation_logits,
                edge_logits=edge_logits,
                edge_relations=edge_relations,
                edge_batch=edge_batch,
                valid_edges=valid_edges,
                num_graphs=num_graphs,
                temperature=temp,
                stop_logits=stop_logits,
                allow_stop=allow_stop,
            )
            self._assert_finite_tensor(log_prob_edge, "log_prob_edge", step=step)
            if log_prob_stop is not None:
                self._assert_finite_tensor(log_prob_stop, "log_prob_stop", step=step)
            actions, log_pf = self._sample_actions(
                log_prob_edge=log_prob_edge,
                log_prob_stop=log_prob_stop,
                valid_edges=valid_edges,
                edge_batch=edge_batch,
                has_edge=has_edge,
                is_greedy=is_greedy,
            )
            self._assert_finite_tensor(log_pf, "log_pf", step=step)
            actions, log_pf = self._apply_done_mask(pre_done, actions, log_pf)
            stop_node_locals = self._update_stop_nodes(
                stop_node_locals=stop_node_locals,
                actions=actions,
                done_mask=pre_done,
                edge_index=edge_index,
                edge_batch=edge_batch,
                edge_relations=edge_relations,
                node_ptr=state.graph.node_ptr,
            )
            if buffers.actions_seq is not None:
                buffers.actions_seq[:, step] = actions
            buffers.log_pf_steps[:, step] = log_pf
            log_pf_total = log_pf_total + log_pf
            relation_tokens = self._gather_relation_tokens(
                relation_tokens=graph["relation_tokens"],
                actions=actions,
                num_graphs=num_graphs,
            )
            update_mask = (actions >= _ZERO) & (~pre_done)
            state_vec = self.state_encoder.update_state(
                state_vec,
                relation_tokens=relation_tokens,
                update_mask=update_mask,
            )
            self._assert_finite_tensor(state_vec, "state_vec", step=step)
            state = self.env.step(state, actions, step_index=step)
            log_pb = self._compute_log_pb(
                state=state,
                graph=graph,
                state_vec=state_vec,
                actions=actions,
                autocast_ctx=autocast_ctx,
                num_graphs=num_graphs,
            )
            self._assert_finite_tensor(log_pb, "log_pb", step=step)
            log_pb = torch.where(pre_done, torch.zeros_like(log_pb), log_pb)
            buffers.log_pb_steps[:, step] = log_pb

        stop_node_locals = self._backfill_stop_nodes(stop_node_locals=stop_node_locals, state=state)
        reach_success = state.answer_hits.float()
        length = state.step_counts.to(dtype=torch.float32)
        return RolloutResult(
            actions_seq=buffers.actions_seq,
            directions_seq=state.directions if record_actions else None,
            log_pf=log_pf_total,
            log_pf_steps=buffers.log_pf_steps,
            log_pb_steps=buffers.log_pb_steps,
            selected_mask=state.used_edge_mask,
            reach_success=reach_success,
            length=length,
            stop_node_locals=stop_node_locals,
            answer_node_hit=state.answer_node_hit,
            start_node_hit=state.start_node_hit,
            visited_nodes=state.visited_nodes if record_visited else None,
        )

    @staticmethod
    def _autocast_context(device: torch.device):
        if device.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, enabled=torch.is_autocast_enabled())

    @staticmethod
    def _init_buffers(
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        *,
        record_actions: bool,
    ) -> RolloutBuffers:
        return RolloutBuffers.create(
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            record_actions=record_actions,
        )

    def _resolve_temperature(self, temperature: Optional[float]) -> tuple[float, bool]:
        base = self.policy_temperature if temperature is None else float(temperature)
        is_greedy = base < MIN_TEMPERATURE
        return max(base, MIN_TEMPERATURE), is_greedy

    def _resolve_backward_temperature(self, temperature: Optional[float]) -> float:
        base = self.policy_temperature if temperature is None else float(temperature)
        return max(base, MIN_TEMPERATURE)

    def _policy_logits(
        self,
        *,
        valid_edges: torch.Tensor,
        edge_logits: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> torch.Tensor:
        with autocast_ctx:
            edge_logits = self.policy(
                edge_scores=edge_logits,
                valid_edges_mask=valid_edges,
            )
        return edge_logits

    def _valid_edges(self, state: Any) -> torch.Tensor:
        unused_edges = ~state.used_edge_mask
        forward_mask = self.env.forward_edge_mask(state)
        return forward_mask & unused_edges

    def _allow_stop(self, state: Any) -> torch.Tensor:
        step_counts = state.step_counts
        allow_stop = ~state.done
        if step_counts.numel() > _ZERO:
            horizon_exhausted = step_counts >= self.max_steps
            allow_stop = allow_stop & (~horizon_exhausted)
        min_stop_steps = int(getattr(self.env, "min_stop_steps", 0))
        if min_stop_steps > _ZERO:
            allow_stop = allow_stop & (step_counts >= min_stop_steps)
        return allow_stop

    def _sample_actions(
        self,
        *,
        log_prob_edge: torch.Tensor,
        log_prob_stop: torch.Tensor | None,
        valid_edges: torch.Tensor,
        edge_batch: torch.Tensor,
        has_edge: torch.Tensor,
        is_greedy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if log_prob_stop is None:
            raise ValueError("log_prob_stop must be provided for stop-aware sampling.")
        if log_prob_edge.numel() == 0:
            actions = torch.full_like(has_edge, STOP_RELATION, dtype=torch.long)
            log_pf = log_prob_stop.to(dtype=torch.float32)
            return actions, log_pf

        if is_greedy:
            score_edge = log_prob_edge
            score_stop = log_prob_stop
        else:
            score_edge = log_prob_edge + gumbel_noise_like(log_prob_edge)
            score_stop = log_prob_stop + gumbel_noise_like(log_prob_stop)

        neg_inf = neg_inf_value(score_edge)
        score_edge = torch.where(valid_edges, score_edge, torch.full_like(score_edge, neg_inf))
        score_edge_max, edge_argmax = scatter_max(score_edge, edge_batch, dim=0, dim_size=has_edge.numel())
        score_edge_max = torch.where(has_edge, score_edge_max, torch.full_like(score_edge_max, neg_inf))
        edge_argmax = torch.where(has_edge, edge_argmax, torch.zeros_like(edge_argmax))
        choose_stop = score_stop > score_edge_max
        choose_stop = choose_stop | (~has_edge)
        actions = torch.where(choose_stop, torch.full_like(edge_argmax, STOP_RELATION), edge_argmax)
        log_pf = torch.where(choose_stop, log_prob_stop, log_prob_edge[edge_argmax])
        return actions, log_pf

    def _score_edges_forward(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> torch.Tensor:
        edge_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=True,
        )
        return self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_forward_head,
            use_tail=True,
            autocast_ctx=autocast_ctx,
        )

    def _score_forward_with_stop(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=self.relation_use_active_nodes,
        )
        edge_state_inputs = relation_state_inputs
        if not self.relation_use_active_nodes:
            edge_state_inputs = self._build_state_inputs(
                node_tokens=graph["node_tokens"],
                question_tokens=graph["question_tokens"],
                state_vec=state_vec,
                active_nodes=active_nodes,
                node_batch=graph["node_batch"],
                include_active_nodes=True,
            )
        with autocast_ctx:
            relation_logits = self.forward_head(relation_state_inputs)
            stop_logits = self.stop_head(relation_state_inputs).squeeze(-1)
        edge_scores = self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_forward_head,
            use_tail=True,
            autocast_ctx=autocast_ctx,
        )
        return relation_logits, edge_scores, stop_logits

    def _score_edges_backward(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        relation_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=self.relation_use_active_nodes,
        )
        edge_state_inputs = relation_state_inputs
        if not self.relation_use_active_nodes:
            edge_state_inputs = self._build_state_inputs(
                node_tokens=graph["node_tokens"],
                question_tokens=graph["question_tokens"],
                state_vec=state_vec,
                active_nodes=active_nodes,
                node_batch=graph["node_batch"],
                include_active_nodes=True,
            )
        with autocast_ctx:
            relation_logits = self.backward_head(relation_state_inputs)
        edge_scores = self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_backward_head,
            use_tail=False,
            autocast_ctx=autocast_ctx,
        )
        return relation_logits, edge_scores

    def _score_edge_entities(
        self,
        *,
        graph: dict[str, torch.Tensor],
        state_inputs: torch.Tensor,
        edge_head: nn.Module,
        use_tail: bool,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> torch.Tensor:
        edge_index = graph["edge_index"]
        node_tokens = graph["node_tokens"]
        relation_tokens = graph["relation_tokens"]
        edge_batch = graph["edge_batch"]
        edge_nodes = edge_index[_EDGE_TAIL_INDEX if use_tail else _EDGE_HEAD_INDEX]
        edge_node_tokens = node_tokens.index_select(0, edge_nodes)
        with autocast_ctx:
            edge_scores = edge_head(
                state_inputs=state_inputs,
                relation_tokens=relation_tokens,
                node_tokens=edge_node_tokens,
                edge_batch=edge_batch,
            )
        return edge_scores

    @staticmethod
    def _pool_active_nodes(
        *,
        node_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        active_nodes: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        device = node_tokens.device
        dtype = node_tokens.dtype
        out = torch.zeros((num_graphs, node_tokens.size(-1)), device=device, dtype=dtype)
        active_mask = active_nodes.to(device=device, dtype=torch.bool)
        if not torch.any(active_mask):
            return out
        active_tokens = node_tokens[active_mask]
        active_batch = node_batch[active_mask]
        out.index_add_(0, active_batch, active_tokens)
        counts = torch.bincount(active_batch, minlength=num_graphs).to(device=device, dtype=dtype).clamp(min=_ONE)
        return out / counts.unsqueeze(-1)

    @classmethod
    def _build_state_inputs(
        cls,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        state_vec: torch.Tensor,
        active_nodes: torch.Tensor,
        node_batch: torch.Tensor,
        include_active_nodes: bool,
    ) -> torch.Tensor:
        if include_active_nodes:
            num_graphs = int(state_vec.size(0))
            current_entities = cls._pool_active_nodes(
                node_tokens=node_tokens,
                node_batch=node_batch,
                active_nodes=active_nodes,
                num_graphs=num_graphs,
            )
            return torch.cat([question_tokens, current_entities, state_vec], dim=-1)
        return torch.cat([question_tokens, state_vec], dim=-1)

    @staticmethod
    def _gather_relation_tokens(
        *,
        relation_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        device = relation_tokens.device
        dtype = relation_tokens.dtype
        out = torch.zeros((num_graphs, relation_tokens.size(-1)), device=device, dtype=dtype)
        valid_action = actions >= _ZERO
        if not torch.any(valid_action):
            return out
        action_ids = actions[valid_action]
        out[valid_action] = relation_tokens.index_select(0, action_ids)
        return out

    def _backward_valid_edges(self, state: Any) -> torch.Tensor:
        return self.env.backward_edge_mask(state)

    def _compute_log_pb(
        self,
        *,
        state: Any,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        actions: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        num_graphs: int,
    ) -> torch.Tensor:
        valid_edges = self._backward_valid_edges(state)
        valid_action = actions >= _ZERO
        if valid_action.any():
            action_ids = actions[valid_action]
            if action_ids.numel() > 0 and int(action_ids.max().item()) >= int(valid_edges.numel()):
                raise ValueError("Action id out of range for backward valid_edges.")
            action_mask = torch.zeros_like(valid_edges, dtype=torch.bool)
            action_mask[action_ids] = True
            valid_edges = valid_edges | action_mask
        relation_logits, edge_scores = self._score_edges_backward(
            graph,
            state_vec,
            active_nodes=state.active_nodes,
            autocast_ctx=autocast_ctx,
        )
        edge_logits = self._policy_logits(
            valid_edges=valid_edges,
            edge_logits=edge_scores,
            autocast_ctx=autocast_ctx,
        )
        log_prob_edge, _, _, _ = compute_factorized_log_probs(
            relation_logits=relation_logits,
            edge_logits=edge_logits,
            edge_relations=graph["edge_relations"],
            edge_batch=graph["edge_batch"],
            valid_edges=valid_edges,
            num_graphs=num_graphs,
            temperature=self.backward_temperature,
        )
        log_pb = self._select_log_prob(log_prob_edge, actions)
        if log_pb.numel() == 0:
            return log_pb
        edge_relations = graph["edge_relations"]
        if edge_relations.numel() == 0:
            return log_pb
        if valid_action.any():
            action_ids = actions[valid_action]
            if action_ids.numel() > 0 and int(action_ids.max().item()) >= int(edge_relations.numel()):
                raise ValueError("Action id out of range for edge_relations in log_pb computation.")
            rel_ids = edge_relations.index_select(0, action_ids)
            stop_mask = torch.zeros_like(actions, dtype=torch.bool)
            stop_mask[valid_action] = rel_ids == STOP_EDGE_RELATION
            log_pb = torch.where(stop_mask, torch.zeros_like(log_pb), log_pb)
        return log_pb

    @staticmethod
    def _select_log_prob(log_prob_edge: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(actions.size(0), device=log_prob_edge.device, dtype=log_prob_edge.dtype)
        valid_action = actions >= _ZERO
        if valid_action.any():
            out[valid_action] = log_prob_edge.index_select(0, actions[valid_action])
        return out

    @staticmethod
    def _apply_done_mask(
        done_mask: torch.Tensor,
        actions: torch.Tensor,
        log_pf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = torch.where(done_mask, torch.full_like(actions, STOP_RELATION), actions)
        log_pf = torch.where(done_mask, torch.zeros_like(log_pf), log_pf)
        return actions, log_pf

    @staticmethod
    def _update_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> torch.Tensor:
        done_mask = done_mask.to(device=stop_node_locals.device, dtype=torch.bool)
        valid_action = actions >= _ZERO
        stop_action = torch.zeros_like(actions, dtype=torch.bool)
        if valid_action.any():
            edge_ids = actions[valid_action]
            rel_ids = edge_relations[edge_ids]
            stop_action[valid_action] = rel_ids == STOP_EDGE_RELATION
        update_mask = stop_action & (~done_mask) & (stop_node_locals == _STOP_NODE_NONE)
        if not update_mask.any():
            return stop_node_locals
        edge_ids = actions[update_mask]
        graph_ids = edge_batch[edge_ids]
        head_nodes = edge_index[0, edge_ids]
        local_idx = head_nodes - node_ptr[graph_ids]
        updated = stop_node_locals.clone()
        updated[update_mask] = local_idx
        return updated

    @staticmethod
    def _backfill_stop_nodes(*, stop_node_locals: torch.Tensor, state: Any) -> torch.Tensor:
        needs_backfill = (stop_node_locals == _STOP_NODE_NONE) & state.done
        if not needs_backfill.any():
            return stop_node_locals
        active_idx = torch.nonzero(state.active_nodes, as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            return stop_node_locals
        node_ptr = state.graph.node_ptr
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        sentinel = num_nodes_total + _ONE
        active_batch = state.graph.node_batch[active_idx]
        local_idx = active_idx - node_ptr[active_batch]
        min_local = torch.full(
            (num_graphs,),
            sentinel,
            device=stop_node_locals.device,
            dtype=stop_node_locals.dtype,
        )
        min_local.scatter_reduce_(0, active_batch, local_idx, reduce="amin", include_self=True)
        has_active = min_local != sentinel
        backfill_mask = needs_backfill & has_active
        return torch.where(backfill_mask, min_local, stop_node_locals)

__all__ = ["GFlowNetActor", "RolloutResult"]
