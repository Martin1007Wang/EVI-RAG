from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from torch import nn
from torch_scatter import scatter_max

from src.gfn.ops import (
    EDGE_POLICY_MASK_KEY,
    STOP_NODE_MASK_KEY,
    OutgoingEdges,
    compute_forward_log_probs,
    gather_outgoing_edges,
    neg_inf_value,
    sample_actions,
)
from src.gfn.trajectory_utils import derive_trajectory, stack_steps
from src.models.components.gflownet_env import GraphEnv, GraphState, STOP_RELATION
from src.models.components.gflownet_layers import TrajectoryAgent

MIN_TEMPERATURE = 1e-5
_STOP_STEP_OFFSET = 1
_STOP_LOGIT_DIM = 1
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_TWO = 2
_CONTEXT_QUESTION = "question"
_CONTEXT_START_NODE = "start_node"
_CONTEXT_MODES = {_CONTEXT_QUESTION, _CONTEXT_START_NODE}
_DEFAULT_CONTEXT_MODE = _CONTEXT_QUESTION
_FLOW_MODES = {"forward", "backward"}


@dataclass(frozen=True)
class _RolloutStepScores:
    outgoing: OutgoingEdges
    edge_scores: torch.Tensor
    stop_logits: torch.Tensor
    edge_guidance: Optional[torch.Tensor]
    allow_stop: torch.Tensor
    edge_valid_mask: torch.Tensor


class GFlowNetActor(nn.Module):
    """Edge-level GFlowNet actor with trajectory-aware state."""

    def __init__(
        self,
        *,
        env: GraphEnv,
        agent: TrajectoryAgent,
        max_steps: int,
        policy_temperature: float,
        stop_bias_init: Optional[float] = None,
        context_mode: str = _DEFAULT_CONTEXT_MODE,
        default_mode: str = "forward",
    ) -> None:
        super().__init__()
        self.env = env
        self.agent = agent
        mode = str(context_mode or _DEFAULT_CONTEXT_MODE).strip().lower()
        if mode not in _CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {sorted(_CONTEXT_MODES)}, got {mode!r}.")
        self.context_mode = mode
        default_mode = str(default_mode or "forward").strip().lower()
        if default_mode not in _FLOW_MODES:
            raise ValueError(f"default_mode must be one of {sorted(_FLOW_MODES)}, got {default_mode!r}.")
        self.default_mode = default_mode
        stop_input_dim = int(agent.hidden_dim) + _TWO
        self.stop_input_norm = nn.LayerNorm(stop_input_dim)
        self.stop_head = nn.Linear(stop_input_dim, _STOP_LOGIT_DIM)
        if stop_bias_init is not None:
            if self.stop_head.bias is None:
                raise RuntimeError("stop_head bias must be enabled for stop bias init.")
            with torch.no_grad():
                self.stop_head.bias.fill_(float(stop_bias_init))
        self.max_steps = int(max_steps)
        self.policy_temperature = nn.Parameter(torch.tensor(float(policy_temperature), dtype=torch.float32))

    def rollout(
        self,
        *,
        graph: dict[str, torch.Tensor],
        temperature: Optional[Union[float, torch.Tensor]] = None,
        record_actions: bool = True,
        max_steps_override: Optional[int] = None,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mode: Optional[str] = None,
        init_node_locals: Optional[torch.Tensor] = None,
        init_ptr: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        device = graph["edge_index"].device
        temp, is_greedy = self._resolve_temperature(temperature)
        mode_val = self._resolve_rollout_mode(mode)
        state = self._resolve_rollout_state(
            graph=graph,
            device=device,
            mode=mode_val,
            init_node_locals=init_node_locals,
            init_ptr=init_ptr,
            max_steps_override=max_steps_override,
        )
        num_graphs = int(state.batch["node_ptr"].numel() - 1)
        num_steps = int(state.max_steps) + _STOP_STEP_OFFSET
        condition_tokens = self._condition_node_tokens(graph=graph, state=state)
        hidden = self.agent.initialize_state(condition_tokens)
        action_keys = self._resolve_action_keys(graph=graph, mode=mode_val)
        log_pf_steps, actions_steps, hidden, state = self._run_rollout_steps(
            state=state,
            graph=graph,
            action_keys=action_keys,
            guidance_fn=guidance_fn,
            hidden=hidden,
            num_steps=num_steps,
            num_graphs=num_graphs,
            temp=temp,
            is_greedy=is_greedy,
            record_actions=record_actions,
        )
        return self._finalize_rollout_outputs(
            log_pf_steps=log_pf_steps,
            actions_steps=actions_steps,
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            record_actions=record_actions,
        )

    def _resolve_temperature(self, temperature: Optional[Union[float, torch.Tensor]]) -> tuple[torch.Tensor, bool]:
        if temperature is None:
            base = self.policy_temperature
        elif isinstance(temperature, torch.Tensor):
            base = temperature
        else:
            base = torch.tensor(
                float(temperature),
                device=self.policy_temperature.device,
                dtype=self.policy_temperature.dtype,
            )
        if base.numel() != _ONE:
            raise ValueError("temperature must be a scalar.")
        base = base.view(())
        base_value = float(base.item())
        is_greedy = base_value < MIN_TEMPERATURE
        return base.clamp(min=MIN_TEMPERATURE), is_greedy

    def _resolve_rollout_mode(self, mode: Optional[str]) -> str:
        mode_val = self.default_mode if mode is None else str(mode).strip().lower()
        if mode_val not in _FLOW_MODES:
            raise ValueError(f"mode must be one of {sorted(_FLOW_MODES)}, got {mode_val!r}.")
        return mode_val

    def _resolve_rollout_state(
        self,
        *,
        graph: dict[str, torch.Tensor],
        device: torch.device,
        mode: str,
        init_node_locals: Optional[torch.Tensor],
        init_ptr: Optional[torch.Tensor],
        max_steps_override: Optional[int],
    ) -> GraphState:
        if max_steps_override is not None:
            override = int(max_steps_override)
            if override <= _ZERO:
                raise ValueError(f"max_steps_override must be > 0, got {override}.")
        if init_node_locals is None:
            if mode == "backward":
                init_node_locals = graph["target_node_locals"]
                if init_ptr is None:
                    init_ptr = graph.get("target_ptr")
            else:
                init_node_locals = graph["start_node_locals"]
                if init_ptr is None:
                    init_ptr = graph.get("start_ptr")
        elif init_ptr is None:
            raise ValueError("init_ptr must be provided when init_node_locals is overridden.")
        return self.env.reset(
            graph,
            device=device,
            mode=mode,
            init_node_locals=init_node_locals,
            init_ptr=init_ptr,
            max_steps_override=max_steps_override,
        )

    def _resolve_action_keys(self, *, graph: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
        action_keys = graph.get("action_keys_shared")
        if action_keys is None:
            cache_key = "action_keys_backward" if mode == "backward" else "action_keys_forward"
            action_keys = graph.get(cache_key)
        if action_keys is None:
            action_keys = self.agent.precompute_action_keys(
                relation_tokens=graph["relation_tokens"],
                node_tokens=graph["node_tokens"],
                edge_index=graph["edge_index"],
            )
        return action_keys

    def _run_rollout_steps(
        self,
        *,
        state: GraphState,
        graph: dict[str, torch.Tensor],
        action_keys: torch.Tensor,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        hidden: torch.Tensor,
        num_steps: int,
        num_graphs: int,
        temp: torch.Tensor,
        is_greedy: bool,
        record_actions: bool,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, GraphState]:
        log_pf_steps: list[torch.Tensor] = []
        actions_steps: list[torch.Tensor] = []
        for step in range(num_steps):
            scores = self._compute_step_scores(
                state=state,
                graph=graph,
                action_keys=action_keys,
                guidance_fn=guidance_fn,
                hidden=hidden,
                num_graphs=num_graphs,
            )
            actions, log_pf = self._sample_actions_from_scores(
                scores=scores,
                graph=graph,
                num_graphs=num_graphs,
                temp=temp,
                is_greedy=is_greedy,
            )
            actions, log_pf, hidden, state = self._rollout_step_update(
                state=state,
                graph=graph,
                actions=actions,
                log_pf=log_pf,
                hidden=hidden,
                step_index=step,
            )
            log_pf_steps.append(log_pf)
            if record_actions:
                actions_steps.append(actions)
            if bool(state.stopped.all().item()):
                break
        return log_pf_steps, actions_steps, hidden, state

    def _finalize_rollout_outputs(
        self,
        *,
        log_pf_steps: list[torch.Tensor],
        actions_steps: list[torch.Tensor],
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        record_actions: bool,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        log_pf_dtype = log_pf_steps[0].dtype if log_pf_steps else torch.float32
        log_pf_steps_tensor = stack_steps(
            log_pf_steps,
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=log_pf_dtype,
            fill_value=float(_ZERO),
        )
        actions_seq = None
        num_moves = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        if record_actions:
            action_dtype = actions_steps[0].dtype if actions_steps else torch.long
            actions_seq = stack_steps(
                actions_steps,
                num_graphs=num_graphs,
                num_steps=num_steps,
                device=device,
                dtype=action_dtype,
                fill_value=STOP_RELATION,
            )
            stats = derive_trajectory(actions_seq=actions_seq, stop_value=STOP_RELATION)
            num_moves = stats.num_moves
        else:
            raise ValueError("rollout requires record_actions=True to derive num_moves from actions_seq.")
        log_pf_total = log_pf_steps_tensor.sum(dim=1)
        return actions_seq, log_pf_total, log_pf_steps_tensor, num_moves

    def _compute_step_scores(
        self,
        *,
        state: GraphState,
        graph: dict[str, torch.Tensor],
        action_keys: torch.Tensor,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        hidden: torch.Tensor,
        num_graphs: int,
    ) -> _RolloutStepScores:
        outgoing = self._gather_outgoing_edges(state=state, graph=graph)
        horizon_exhausted = state.step_counts >= int(state.max_steps)
        edge_valid_mask = self._resolve_edge_valid_mask(
            outgoing=outgoing,
            horizon_exhausted=horizon_exhausted,
        )
        has_edge = outgoing.has_edge & (~horizon_exhausted.view(-1))
        stop_node_mask = graph.get(STOP_NODE_MASK_KEY)
        if stop_node_mask is None:
            raise ValueError("stop_node_mask missing from graph cache; allow_stop requires explicit targets.")
        curr_nodes = state.curr_nodes.view(-1)
        valid_node = curr_nodes >= _ZERO
        safe_nodes = curr_nodes.clamp(min=_ZERO)
        is_target = stop_node_mask.to(device=curr_nodes.device, dtype=torch.bool).index_select(0, safe_nodes)
        is_target = is_target & valid_node
        active = ~state.stopped
        allow_stop = active & (is_target | (~has_edge))
        self._validate_horizon(state=state, allow_stop=allow_stop, horizon_exhausted=horizon_exhausted)
        edge_scores = self._compute_edge_scores(
            state_vec=hidden,
            action_keys=action_keys,
            edge_batch=outgoing.edge_batch,
            edge_ids=outgoing.edge_ids,
        )
        edge_guidance = self._compute_edge_guidance(
            guidance_fn=guidance_fn,
            step_counts=state.step_counts,
            edge_ids=outgoing.edge_ids,
            hidden=hidden,
        )
        stop_logits = self._compute_stop_logits(
            state_vec=hidden,
            edge_scores=edge_scores,
            edge_batch=outgoing.edge_batch,
            num_graphs=num_graphs,
            edge_valid_mask=edge_valid_mask,
        )
        return _RolloutStepScores(
            outgoing=outgoing,
            edge_scores=edge_scores,
            stop_logits=stop_logits,
            edge_guidance=edge_guidance,
            allow_stop=allow_stop,
            edge_valid_mask=edge_valid_mask,
        )

    def _sample_actions_from_scores(
        self,
        *,
        scores: _RolloutStepScores,
        graph: dict[str, torch.Tensor],
        num_graphs: int,
        temp: torch.Tensor,
        is_greedy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = compute_forward_log_probs(
            edge_scores=scores.edge_scores,
            stop_logits=scores.stop_logits,
            allow_stop=scores.allow_stop,
            edge_batch=scores.outgoing.edge_batch,
            num_graphs=num_graphs,
            temperature=temp,
            edge_guidance=scores.edge_guidance,
            edge_valid_mask=scores.edge_valid_mask,
        )
        return sample_actions(
            outgoing=scores.outgoing,
            policy=policy,
            edge_relations=graph["edge_relations"],
            is_greedy=is_greedy,
            stop_action=STOP_RELATION,
        )

    @staticmethod
    def _resolve_edge_valid_mask(
        *,
        outgoing: OutgoingEdges,
        horizon_exhausted: torch.Tensor,
    ) -> torch.Tensor:
        horizon_exhausted = horizon_exhausted.view(-1)
        return ~horizon_exhausted.index_select(0, outgoing.edge_batch)

    @staticmethod
    def _validate_horizon(
        *,
        state: GraphState,
        allow_stop: torch.Tensor,
        horizon_exhausted: torch.Tensor,
    ) -> None:
        active = ~state.stopped
        invalid = active & horizon_exhausted & (~allow_stop)
        if bool(invalid.any().item()):
            count = int(invalid.sum().item())
            raise ValueError(f"max_steps reached without STOP at target for {count} graphs.")

    @staticmethod
    def _compute_edge_guidance(
        *,
        guidance_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
        step_counts: torch.Tensor,
        edge_ids: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if guidance_fn is None or edge_ids.numel() == _ZERO:
            return None
        edge_guidance = guidance_fn(step_counts, edge_ids, hidden).reshape(-1)
        if edge_guidance.numel() != edge_ids.numel():
            raise ValueError("edge_guidance length mismatch with outgoing edges.")
        return edge_guidance

    def _rollout_step_update(
        self,
        *,
        state: GraphState,
        graph: dict[str, torch.Tensor],
        actions: torch.Tensor,
        log_pf: torch.Tensor,
        hidden: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, GraphState]:
        actions, log_pf = self._apply_stop_mask(state.stopped, actions, log_pf)
        hidden = self._update_hidden(
            hidden=hidden,
            graph=graph,
            actions=actions,
            stopped_mask=state.stopped,
        )
        state = self.env.step(state, actions, step_index=step_index)
        return actions, log_pf, hidden, state

    def _condition_node_tokens(self, *, graph: dict[str, torch.Tensor], state: GraphState) -> torch.Tensor:
        if self.context_mode == _CONTEXT_QUESTION:
            return graph["question_tokens"]
        curr_nodes = state.curr_nodes.view(-1)
        valid = curr_nodes >= _ZERO
        if not bool(valid.any().item()):
            return torch.zeros_like(graph["question_tokens"])
        safe_nodes = curr_nodes.clamp(min=_ZERO)
        node_tokens = graph["node_tokens"].index_select(0, safe_nodes)
        return torch.where(valid.unsqueeze(-1), node_tokens, torch.zeros_like(node_tokens))

    def _gather_outgoing_edges(self, *, state: GraphState, graph: dict[str, torch.Tensor]) -> OutgoingEdges:
        active = ~state.stopped
        outgoing = gather_outgoing_edges(
            curr_nodes=state.curr_nodes,
            edge_ids_by_head=graph["edge_ids_by_head"],
            edge_ptr_by_head=graph["edge_ptr_by_head"],
            active_mask=active,
        )
        return self._filter_outgoing_edges_by_policy(outgoing=outgoing, state=state, graph=graph)

    @staticmethod
    def _filter_outgoing_edges_by_policy(
        *,
        outgoing: OutgoingEdges,
        state: GraphState,
        graph: dict[str, torch.Tensor],
    ) -> OutgoingEdges:
        if outgoing.edge_ids.numel() == _ZERO:
            return outgoing
        policy_mask = graph.get(EDGE_POLICY_MASK_KEY)
        if policy_mask is None:
            raise ValueError("edge_policy_mask missing from graph cache; strict edge policy requires it.")
        policy_mask = policy_mask.view(-1)
        keep = policy_mask.index_select(0, outgoing.edge_ids)
        edge_ids = outgoing.edge_ids[keep]
        edge_batch = outgoing.edge_batch[keep]
        num_graphs = int(state.curr_nodes.numel())
        if edge_ids.numel() == _ZERO:
            empty = edge_ids.new_empty((_ZERO,))
            edge_counts = edge_ids.new_zeros((num_graphs,))
            has_edge = edge_ids.new_zeros((num_graphs,), dtype=torch.bool)
            return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=edge_counts, has_edge=has_edge)
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
        has_edge = edge_counts > _ZERO
        return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=edge_counts, has_edge=has_edge)

    def _compute_edge_scores(
        self,
        *,
        state_vec: torch.Tensor,
        action_keys: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.agent.score_cached(
            hidden=state_vec,
            action_keys=action_keys,
            edge_batch=edge_batch,
            edge_ids=edge_ids,
        )

    def _compute_stop_logits(
        self,
        *,
        state_vec: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        edge_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        max_edge_score, has_edge = self._max_edge_score(
            edge_scores=edge_scores,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            edge_valid_mask=edge_valid_mask,
        )
        stop_input = torch.cat((state_vec, max_edge_score.unsqueeze(-1), has_edge.unsqueeze(-1)), dim=-1)
        stop_input = self.stop_input_norm(stop_input)
        return self.stop_head(stop_input).squeeze(-1)

    @staticmethod
    def _max_edge_score(
        *,
        edge_scores: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        edge_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_graphs <= _ZERO:
            zeros = edge_scores.new_zeros((_ZERO,))
            return zeros, zeros
        if edge_scores.numel() == _ZERO:
            zeros = edge_scores.new_zeros((num_graphs,))
            return zeros, zeros
        edge_batch = edge_batch.view(-1)
        if edge_scores.numel() != edge_batch.numel():
            raise ValueError("edge_scores length mismatch with edge_batch for stop logits.")
        edge_valid_mask = edge_valid_mask.view(-1)
        if edge_valid_mask.numel() != edge_scores.numel():
            raise ValueError("edge_valid_mask length mismatch with edge_scores for stop logits.")
        neg_inf = neg_inf_value(edge_scores)
        edge_scores = torch.where(edge_valid_mask, edge_scores, torch.full_like(edge_scores, neg_inf))
        max_scores, _ = scatter_max(edge_scores, edge_batch, dim=0, dim_size=num_graphs)
        neg_inf = neg_inf_value(max_scores)
        has_edge = max_scores > neg_inf
        max_scores = torch.where(has_edge, max_scores, torch.zeros_like(max_scores))
        return max_scores, has_edge.to(dtype=edge_scores.dtype)

    def _update_hidden(
        self,
        *,
        hidden: torch.Tensor,
        graph: dict[str, torch.Tensor],
        actions: torch.Tensor,
        stopped_mask: torch.Tensor,
    ) -> torch.Tensor:
        move_mask = (actions >= _ZERO) & (~stopped_mask)
        if not bool(move_mask.any().item()):
            return hidden
        idx = move_mask.nonzero(as_tuple=False).view(-1)
        edge_ids = actions[move_mask]
        rel_tokens = graph["relation_tokens"].index_select(0, edge_ids)
        tail_nodes = graph["edge_index"][_EDGE_TAIL_INDEX].index_select(0, edge_ids)
        node_tokens = graph["node_tokens"].index_select(0, tail_nodes)
        hidden_updates = self.agent.step(
            hidden=hidden.index_select(0, idx),
            relation_tokens=rel_tokens,
            node_tokens=node_tokens,
        )
        updated = hidden.clone()
        updated.index_copy_(0, idx, hidden_updates)
        return updated

    @staticmethod
    def _apply_stop_mask(
        stopped_mask: torch.Tensor,
        actions: torch.Tensor,
        log_pf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.where(stopped_mask, torch.full_like(actions, STOP_RELATION), actions)
        log_pf = torch.where(stopped_mask, torch.zeros_like(log_pf), log_pf)
        return actions, log_pf

__all__ = ["GFlowNetActor"]
