from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from src.models.components.gflownet_ops import (
    EDGE_POLICY_MASK_KEY,
    STOP_NODE_MASK_KEY,
    OutgoingEdges,
    PolicyLogProbs,
    apply_edge_policy_mask,
    compute_forward_log_probs,
    gather_outgoing_edges,
    neg_inf_value,
    sample_actions,
    segment_max,
    segment_logsumexp_1d,
)
from src.models.components.trajectory_utils import derive_trajectory, stack_steps
from src.models.components.gflownet_env import GraphEnv, GraphState, STOP_RELATION
from src.models.components.gflownet_layers import TrajectoryAgent

MIN_TEMPERATURE = 1e-5
_STOP_STEP_OFFSET = 1
_STOP_LOGIT_DIM = 1
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_TWO = 2
_CONTEXT_QUESTION_START = "question_start"
_CONTEXT_MODES = {_CONTEXT_QUESTION_START}
_DEFAULT_CONTEXT_MODE = _CONTEXT_QUESTION_START
_FLOW_MODES = {"forward", "backward"}
_DEFAULT_H_TRANSFORM_BIAS = 1.0
_DEFAULT_H_TRANSFORM_CLIP = 0.0
_DEFAULT_DIRECTION_EMBEDDING = False
_DEFAULT_DIRECTION_EMB_SCALE = 1.0
_DIRECTION_ID_FORWARD = 0
_DIRECTION_ID_BACKWARD = 1


@dataclass(frozen=True)
class _RolloutStepScores:
    outgoing: OutgoingEdges
    edge_scores: torch.Tensor
    stop_logits: torch.Tensor
    allow_stop: torch.Tensor
    edge_valid_mask: torch.Tensor


@dataclass(frozen=True)
class _SampledActions:
    actions: torch.Tensor
    log_pf: torch.Tensor
    policy: PolicyLogProbs


@dataclass(frozen=True)
class RolloutDiagnostics:
    has_edge_seq: torch.Tensor
    stop_margin_seq: torch.Tensor
    allow_stop_seq: torch.Tensor
    max_edge_score_seq: torch.Tensor
    stop_logit_seq: torch.Tensor


@dataclass(frozen=True)
class RolloutOutput:
    actions_seq: Optional[torch.Tensor]
    log_pf_total: torch.Tensor
    log_pf_steps: torch.Tensor
    num_moves: torch.Tensor
    diagnostics: Optional[RolloutDiagnostics]
    state_vec_seq: Optional[torch.Tensor]


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
        h_transform_bias: Optional[float] = None,
        h_transform_clip: Optional[float] = None,
        direction_embedding: Optional[bool] = None,
        direction_embedding_scale: Optional[float] = None,
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
        bias_value = _DEFAULT_H_TRANSFORM_BIAS if h_transform_bias is None else float(h_transform_bias)
        if bias_value < float(_ZERO):
            raise ValueError("h_transform_bias must be >= 0.")
        clip_value = _DEFAULT_H_TRANSFORM_CLIP if h_transform_clip is None else float(h_transform_clip)
        if clip_value < float(_ZERO):
            raise ValueError("h_transform_clip must be >= 0.")
        self.h_transform_bias = bias_value
        self.h_transform_clip = clip_value
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
        direction_enabled = _DEFAULT_DIRECTION_EMBEDDING if direction_embedding is None else bool(direction_embedding)
        if direction_enabled:
            self.direction_embed = nn.Embedding(len(_FLOW_MODES), int(agent.hidden_dim))
            scale = _DEFAULT_DIRECTION_EMB_SCALE if direction_embedding_scale is None else float(direction_embedding_scale)
            if scale < float(_ZERO):
                raise ValueError("direction_embedding_scale must be >= 0.")
            self.direction_scale = scale
        else:
            self.direction_embed = None
            self.direction_scale = float(_ZERO)

    def rollout(
        self,
        *,
        graph: dict[str, torch.Tensor],
        temperature: Optional[Union[float, torch.Tensor]] = None,
        record_actions: bool = True,
        record_diagnostics: bool = False,
        record_state: bool = False,
        max_steps_override: Optional[int] = None,
        mode: Optional[str] = None,
        init_node_locals: Optional[torch.Tensor] = None,
        init_ptr: Optional[torch.Tensor] = None,
    ) -> RolloutOutput:
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
        question_tokens, node_tokens = self._condition_context_tokens(graph=graph, state=state)
        hidden = self.agent.initialize_state(question_tokens=question_tokens, node_tokens=node_tokens)
        action_keys = self._resolve_action_keys(graph=graph, mode=mode_val)
        log_pf_steps, actions_steps, diag_steps, state_steps, hidden, state = self._run_rollout_steps(
            state=state,
            graph=graph,
            action_keys=action_keys,
            hidden=hidden,
            num_steps=num_steps,
            num_graphs=num_graphs,
            temp=temp,
            is_greedy=is_greedy,
            record_actions=record_actions,
            record_diagnostics=record_diagnostics,
            record_state=record_state,
        )
        return self._finalize_rollout_outputs(
            log_pf_steps=log_pf_steps,
            actions_steps=actions_steps,
            diag_steps=diag_steps,
            state_steps=state_steps,
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            record_actions=record_actions,
            record_diagnostics=record_diagnostics,
            record_state=record_state,
        )

    def resolve_temperature(self, temperature: Optional[Union[float, torch.Tensor]]) -> tuple[torch.Tensor, bool]:
        return self._resolve_temperature(temperature)

    def condition_question_tokens(self, question_tokens: torch.Tensor) -> torch.Tensor:
        if self.direction_embed is None:
            return question_tokens
        direction_id = _DIRECTION_ID_FORWARD if self.default_mode == "forward" else _DIRECTION_ID_BACKWARD
        dir_vec = self.direction_embed.weight[direction_id].to(device=question_tokens.device, dtype=question_tokens.dtype)
        scale = float(self.direction_scale)
        if question_tokens.dim() == _TWO:
            return question_tokens + dir_vec.view(1, -1) * scale
        if question_tokens.dim() == 3:
            return question_tokens + dir_vec.view(1, 1, -1) * scale
        raise ValueError("question_tokens must be [B, H] or [B, 1, H] for direction conditioning.")

    def _resolve_temperature(self, temperature: Optional[Union[float, torch.Tensor]]) -> tuple[torch.Tensor, bool]:
        if temperature is None:
            base = self.policy_temperature
        elif isinstance(temperature, torch.Tensor):
            base = temperature.to(device=self.policy_temperature.device, dtype=self.policy_temperature.dtype)
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
                question_tokens=graph["question_tokens"],
                edge_batch=graph["edge_batch"],
            )
        return action_keys

    def _run_rollout_steps(
        self,
        *,
        state: GraphState,
        graph: dict[str, torch.Tensor],
        action_keys: torch.Tensor,
        hidden: torch.Tensor,
        num_steps: int,
        num_graphs: int,
        temp: torch.Tensor,
        is_greedy: bool,
        record_actions: bool,
        record_diagnostics: bool,
        record_state: bool,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        Optional[dict[str, list[torch.Tensor]]],
        Optional[list[torch.Tensor]],
        torch.Tensor,
        GraphState,
    ]:
        log_pf_steps: list[torch.Tensor] = []
        actions_steps: list[torch.Tensor] = []
        state_steps: Optional[list[torch.Tensor]] = [] if record_state else None
        diag_steps = self._init_diag_steps(record_diagnostics)
        for step in range(num_steps):
            if state_steps is not None:
                state_steps.append(hidden)
            scores = self._compute_step_scores(
                state=state,
                graph=graph,
                action_keys=action_keys,
                hidden=hidden,
                num_graphs=num_graphs,
            )
            sampled = self._sample_actions_from_scores(
                scores=scores,
                graph=graph,
                num_graphs=num_graphs,
                temp=temp,
                is_greedy=is_greedy,
            )
            if diag_steps is not None:
                self._record_diag_steps(
                    diag_steps=diag_steps,
                    scores=scores,
                    policy=sampled.policy,
                    num_graphs=num_graphs,
                )
            actions, log_pf, hidden, state = self._rollout_step_update(
                state=state,
                graph=graph,
                actions=sampled.actions,
                log_pf=sampled.log_pf,
                hidden=hidden,
                step_index=step,
            )
            log_pf_steps.append(log_pf)
            if record_actions:
                actions_steps.append(actions)
            if bool(state.stopped.all().item()):
                break
        return log_pf_steps, actions_steps, diag_steps, state_steps, hidden, state

    def _finalize_rollout_outputs(
        self,
        *,
        log_pf_steps: list[torch.Tensor],
        actions_steps: list[torch.Tensor],
        diag_steps: Optional[dict[str, list[torch.Tensor]]],
        state_steps: Optional[list[torch.Tensor]],
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        record_actions: bool,
        record_diagnostics: bool,
        record_state: bool,
    ) -> RolloutOutput:
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
        diagnostics = None
        if record_diagnostics and diag_steps is not None:
            diagnostics = self._finalize_rollout_diagnostics(
                diag_steps=diag_steps,
                num_graphs=num_graphs,
                num_steps=num_steps,
                device=device,
                dtype=log_pf_dtype,
            )
        state_vec_seq = None
        if record_state:
            if state_steps is None or not state_steps:
                state_vec_seq = torch.zeros(
                    (num_graphs, num_steps, int(self.agent.hidden_dim)),
                    device=device,
                    dtype=log_pf_dtype,
                )
            else:
                state_dtype = state_steps[0].dtype
                stacked = torch.stack(state_steps, dim=1)
                if stacked.size(1) > num_steps:
                    raise ValueError("state stack exceeds expected rollout horizon.")
                if stacked.size(1) < num_steps:
                    pad = torch.zeros(
                        (num_graphs, num_steps - stacked.size(1), stacked.size(2)),
                        device=device,
                        dtype=state_dtype,
                    )
                    stacked = torch.cat([stacked, pad], dim=1)
                state_vec_seq = stacked
        return RolloutOutput(
            actions_seq=actions_seq,
            log_pf_total=log_pf_total,
            log_pf_steps=log_pf_steps_tensor,
            num_moves=num_moves,
            diagnostics=diagnostics,
            state_vec_seq=state_vec_seq,
        )

    def _compute_step_scores(
        self,
        *,
        state: GraphState,
        graph: dict[str, torch.Tensor],
        action_keys: torch.Tensor,
        hidden: torch.Tensor,
        num_graphs: int,
    ) -> _RolloutStepScores:
        outgoing = self._gather_outgoing_edges(state=state, graph=graph)
        horizon_exhausted = state.step_counts >= int(state.max_steps)
        last_step = state.step_counts + _ONE >= int(state.max_steps)
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
        allow_stop = active & (is_target | (~has_edge) | last_step)
        self._validate_horizon(state=state, allow_stop=allow_stop, horizon_exhausted=horizon_exhausted)
        edge_scores = self._compute_edge_scores(
            state_vec=hidden,
            action_keys=action_keys,
            edge_batch=outgoing.edge_batch,
            edge_ids=outgoing.edge_ids,
            graph=graph,
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
    ) -> _SampledActions:
        policy = compute_forward_log_probs(
            edge_scores=scores.edge_scores,
            stop_logits=scores.stop_logits,
            allow_stop=scores.allow_stop,
            edge_batch=scores.outgoing.edge_batch,
            num_graphs=num_graphs,
            temperature=temp,
            edge_valid_mask=scores.edge_valid_mask,
        )
        actions, log_pf = sample_actions(
            outgoing=scores.outgoing,
            policy=policy,
            edge_relations=graph["edge_relations"],
            is_greedy=is_greedy,
            stop_action=STOP_RELATION,
        )
        return _SampledActions(actions=actions, log_pf=log_pf, policy=policy)

    @staticmethod
    def _init_diag_steps(record_diagnostics: bool) -> Optional[dict[str, list[torch.Tensor]]]:
        if not record_diagnostics:
            return None
        return {
            "has_edge": [],
            "stop_margin": [],
            "allow_stop": [],
            "max_edge_score": [],
            "stop_logits": [],
        }

    def _record_diag_steps(
        self,
        *,
        diag_steps: dict[str, list[torch.Tensor]],
        scores: _RolloutStepScores,
        policy: PolicyLogProbs,
        num_graphs: int,
    ) -> None:
        max_edge_score, _ = self._max_edge_score(
            edge_scores=scores.edge_scores,
            edge_batch=scores.outgoing.edge_batch,
            num_graphs=num_graphs,
            edge_valid_mask=scores.edge_valid_mask,
        )
        stop_margin = policy.stop - policy.not_stop
        stop_margin = torch.where(policy.has_edge, stop_margin, torch.zeros_like(stop_margin))
        diag_steps["stop_margin"].append(stop_margin)
        diag_steps["has_edge"].append(policy.has_edge)
        diag_steps["allow_stop"].append(scores.allow_stop)
        diag_steps["max_edge_score"].append(max_edge_score)
        diag_steps["stop_logits"].append(scores.stop_logits)

    @staticmethod
    def _finalize_rollout_diagnostics(
        *,
        diag_steps: dict[str, list[torch.Tensor]],
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RolloutDiagnostics:
        has_edge_seq = stack_steps(
            diag_steps["has_edge"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=torch.bool,
            fill_value=_ZERO,
        )
        stop_margin_seq = stack_steps(
            diag_steps["stop_margin"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        allow_stop_seq = stack_steps(
            diag_steps["allow_stop"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=torch.bool,
            fill_value=_ZERO,
        )
        max_edge_score_seq = stack_steps(
            diag_steps["max_edge_score"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        stop_logit_seq = stack_steps(
            diag_steps["stop_logits"],
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            fill_value=float(_ZERO),
        )
        return RolloutDiagnostics(
            has_edge_seq=has_edge_seq,
            stop_margin_seq=stop_margin_seq,
            allow_stop_seq=allow_stop_seq,
            max_edge_score_seq=max_edge_score_seq,
            stop_logit_seq=stop_logit_seq,
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

    def _condition_context_tokens(
        self,
        *,
        graph: dict[str, torch.Tensor],
        state: GraphState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        question_tokens = graph["question_tokens"]
        curr_nodes = state.curr_nodes.view(-1)
        valid = curr_nodes >= _ZERO
        if not bool(valid.any().item()):
            return question_tokens, torch.zeros_like(question_tokens)
        safe_nodes = curr_nodes.clamp(min=_ZERO)
        node_tokens = graph["node_tokens"].index_select(0, safe_nodes)
        node_tokens = torch.where(valid.unsqueeze(-1), node_tokens, torch.zeros_like(node_tokens))
        return question_tokens, node_tokens

    def _gather_outgoing_edges(self, *, state: GraphState, graph: dict[str, torch.Tensor]) -> OutgoingEdges:
        active = ~state.stopped
        outgoing = gather_outgoing_edges(
            curr_nodes=state.curr_nodes,
            edge_ids_by_head=graph["edge_ids_by_head"],
            edge_ptr_by_head=graph["edge_ptr_by_head"],
            active_mask=active,
        )
        policy_mask = graph.get(EDGE_POLICY_MASK_KEY)
        if policy_mask is None:
            raise ValueError("edge_policy_mask missing from graph cache; strict edge policy requires it.")
        num_graphs = int(state.curr_nodes.numel())
        return apply_edge_policy_mask(outgoing=outgoing, edge_policy_mask=policy_mask, num_graphs=num_graphs)

    def _compute_edge_scores(
        self,
        *,
        state_vec: torch.Tensor,
        action_keys: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ids: torch.Tensor,
        graph: dict[str, torch.Tensor],
        detach_agent: bool = False,
    ) -> torch.Tensor:
        if edge_ids.numel() == _ZERO:
            return edge_ids.new_empty((_ZERO,))
        prior_scores = self._compute_prior_scores(
            state_vec=state_vec,
            action_keys=action_keys,
            edge_batch=edge_batch,
            edge_ids=edge_ids,
            detach_agent=detach_agent,
        )
        flow_scores = self._compute_flow_scores(graph=graph, edge_ids=edge_ids)
        return prior_scores + flow_scores * float(self.h_transform_bias)

    def _compute_prior_scores(
        self,
        *,
        state_vec: torch.Tensor,
        action_keys: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ids: torch.Tensor,
        detach_agent: bool,
    ) -> torch.Tensor:
        scores = self.agent.score_cached(
            hidden=state_vec,
            action_keys=action_keys,
            edge_batch=edge_batch,
            edge_ids=edge_ids,
        )
        return scores.detach() if detach_agent else scores

    def _compute_flow_scores(self, *, graph: dict[str, torch.Tensor], edge_ids: torch.Tensor) -> torch.Tensor:
        log_f_nodes = graph.get("log_f_nodes")
        if log_f_nodes is None:
            raise ValueError("log_f_nodes missing from graph cache for h_transform score mode.")
        edge_index = graph["edge_index"]
        tail_nodes = edge_index[_EDGE_TAIL_INDEX].index_select(0, edge_ids)
        log_f_tail = log_f_nodes.index_select(0, tail_nodes.to(device=log_f_nodes.device))
        return self._apply_flow_clip(log_f_tail)

    def _apply_flow_clip(self, log_f_tail: torch.Tensor) -> torch.Tensor:
        clip_value = float(self.h_transform_clip)
        if clip_value <= float(_ZERO):
            return log_f_tail
        clip = torch.tensor(clip_value, device=log_f_tail.device, dtype=log_f_tail.dtype)
        return torch.tanh(log_f_tail / clip) * clip

    def _compute_stop_logits(
        self,
        *,
        state_vec: torch.Tensor,
        edge_scores: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        edge_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        logsumexp_edges, has_edge = self._logsumexp_edge_score(
            edge_scores=edge_scores,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            edge_valid_mask=edge_valid_mask,
        )
        stop_input = torch.cat((state_vec, logsumexp_edges.unsqueeze(-1), has_edge.unsqueeze(-1)), dim=-1)
        stop_input = self.stop_input_norm(stop_input)
        return self.stop_head(stop_input).squeeze(-1)

    @staticmethod
    def _logsumexp_edge_score(
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
        logsumexp_edges = segment_logsumexp_1d(edge_scores, edge_batch, num_graphs)
        neg_inf = neg_inf_value(logsumexp_edges)
        has_edge = logsumexp_edges > neg_inf
        logsumexp_edges = torch.where(has_edge, logsumexp_edges, torch.zeros_like(logsumexp_edges))
        return logsumexp_edges, has_edge.to(dtype=edge_scores.dtype)

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
        max_scores, _ = segment_max(edge_scores, edge_batch, num_graphs)
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

__all__ = ["GFlowNetActor", "RolloutDiagnostics", "RolloutOutput"]
