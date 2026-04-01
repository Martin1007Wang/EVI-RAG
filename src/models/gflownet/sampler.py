from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from src.utils.cuda_memory import profile_cuda_memory

from .actor import HierarchicalStateActionDistribution
from .policy import SubgraphPolicy
from .prepared_batch import SubgraphPreparedBatch
from .reward import SubgraphTerminalReward
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch, SubgraphState


@dataclass(frozen=True)
class SubgraphTrajectorySampleBatch:
    state_log_flows: torch.Tensor
    log_pf_actions: torch.Tensor
    log_pb_actions: torch.Tensor
    log_reward_actions: torch.Tensor
    action_mask: torch.Tensor
    termination_action_steps: torch.Tensor
    chosen_edge_ids: torch.Tensor
    stop_actions: torch.Tensor
    terminal_commit_candidate_counts: torch.Tensor
    terminal_gold_answer_counts: torch.Tensor
    terminal_hit_mask: torch.Tensor
    terminal_component_counts: torch.Tensor
    terminal_edge_ids: tuple[tuple[int, ...], ...]
    terminal_node_ids: tuple[tuple[int, ...], ...]
    terminal_reachability_bits: tuple[dict[int, int], ...]
    sample_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    num_graphs: int
    num_rollouts: int
    terminal_answer_entity_ids: tuple[tuple[int, ...], ...] = field(
        default_factory=tuple
    )
    chosen_answer_entity_ids: torch.Tensor | None = None
    chosen_source_graph_nodes: torch.Tensor | None = None
    state_component_counts: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected_shape = (self.num_graphs, self.num_rollouts)
        if tuple(self.termination_action_steps.shape) != expected_shape:
            raise ValueError("termination_action_steps shape mismatch.")
        if tuple(self.terminal_commit_candidate_counts.shape) != expected_shape:
            raise ValueError("terminal_commit_candidate_counts shape mismatch.")
        if tuple(self.terminal_gold_answer_counts.shape) != expected_shape:
            raise ValueError("terminal_gold_answer_counts shape mismatch.")
        if tuple(self.terminal_hit_mask.shape) != expected_shape:
            raise ValueError("terminal_hit_mask shape mismatch.")
        if tuple(self.terminal_component_counts.shape) != expected_shape:
            raise ValueError("terminal_component_counts shape mismatch.")
        if (
            self.chosen_answer_entity_ids is not None
            and tuple(self.chosen_answer_entity_ids.shape) != expected_shape
        ):
            raise ValueError("chosen_answer_entity_ids shape mismatch.")
        if self.chosen_source_graph_nodes is not None and tuple(
            self.chosen_source_graph_nodes.shape
        ) != tuple(self.chosen_edge_ids.shape):
            raise ValueError("chosen_source_graph_nodes shape mismatch.")
        if self.state_component_counts is not None and tuple(
            self.state_component_counts.shape
        ) != tuple(self.state_log_flows.shape):
            raise ValueError("state_component_counts shape mismatch.")
        expected_flat = int(self.num_graphs) * int(self.num_rollouts)
        if len(self.terminal_edge_ids) != expected_flat:
            raise ValueError("terminal_edge_ids length mismatch.")
        if len(self.terminal_node_ids) != expected_flat:
            raise ValueError("terminal_node_ids length mismatch.")
        if len(self.terminal_reachability_bits) != expected_flat:
            raise ValueError("terminal_reachability_bits length mismatch.")
        if len(self.terminal_answer_entity_ids) == 0:
            derived_answer_entities: list[tuple[int, ...]] = []
            if self.chosen_answer_entity_ids is not None:
                for value in (
                    self.chosen_answer_entity_ids.view(-1).detach().cpu().tolist()
                ):
                    answer_entity_id = int(value)
                    derived_answer_entities.append(
                        () if answer_entity_id < 0 else (int(answer_entity_id),)
                    )
            else:
                derived_answer_entities = [() for _ in range(expected_flat)]
            object.__setattr__(
                self,
                "terminal_answer_entity_ids",
                tuple(derived_answer_entities),
            )
        if len(self.terminal_answer_entity_ids) != expected_flat:
            raise ValueError("terminal_answer_entity_ids length mismatch.")

    @property
    def success_rate(self) -> torch.Tensor:
        return self.terminal_hit_mask.to(dtype=torch.float32).mean()


@dataclass(frozen=True)
class _UniqueStateLayout:
    active_state_indices: tuple[int, ...]
    active_state_tensor: torch.Tensor
    unique_state_indices: tuple[int, ...]
    unique_state_tensor: torch.Tensor
    active_to_unique: torch.Tensor


@dataclass(frozen=True)
class _ActiveStateContext:
    unique_layout: _UniqueStateLayout
    active_analyses: tuple[SubgraphAnalysis, ...]
    active_log_flows: torch.Tensor
    active_component_counts: torch.Tensor
    active_state_distributions: tuple[HierarchicalStateActionDistribution, ...]


def _log_softmax_choice(logits: torch.Tensor, index: int) -> torch.Tensor:
    return torch.log_softmax(logits.to(dtype=torch.float32), dim=0)[int(index)]


def _sample_index(
    logits: torch.Tensor, *, temperature: float
) -> tuple[int, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    scaled_logits = logits.to(dtype=torch.float32) / float(temperature)
    probabilities = torch.softmax(scaled_logits, dim=0)
    sample = torch.multinomial(probabilities, num_samples=1)
    return int(sample.item()), scaled_logits


def _build_unique_state_layout(
    *,
    rollout_batch: SubgraphRolloutBatch,
    active_state_indices: tuple[int, ...] | list[int],
    device: torch.device,
) -> _UniqueStateLayout:
    normalized_indices = tuple(int(state_idx) for state_idx in active_state_indices)
    unique_index_by_key: dict[tuple[int, tuple[Any, ...]], int] = {}
    unique_state_indices: list[int] = []
    active_to_unique: list[int] = []
    for flat_state_idx in normalized_indices:
        graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
        cache_key = (graph_idx, rollout_batch.state_key(int(flat_state_idx)))
        unique_local_idx = unique_index_by_key.get(cache_key)
        if unique_local_idx is None:
            unique_local_idx = int(len(unique_state_indices))
            unique_index_by_key[cache_key] = unique_local_idx
            unique_state_indices.append(int(flat_state_idx))
        active_to_unique.append(int(unique_local_idx))
    return _UniqueStateLayout(
        active_state_indices=normalized_indices,
        active_state_tensor=torch.tensor(
            normalized_indices,
            device=device,
            dtype=torch.long,
        ),
        unique_state_indices=tuple(unique_state_indices),
        unique_state_tensor=torch.tensor(
            unique_state_indices,
            device=device,
            dtype=torch.long,
        ),
        active_to_unique=torch.tensor(
            active_to_unique,
            device=device,
            dtype=torch.long,
        ),
    )


def _lookup_analysis(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
) -> SubgraphAnalysis:
    cache_key = (int(graph_idx), state.key())
    analysis = analysis_cache.get(cache_key)
    if analysis is None:
        analysis = policy.analyze_state(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            state=state,
        )
        analysis_cache[cache_key] = analysis
    return analysis


def _resolve_unique_state_analyses(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    rollout_batch: SubgraphRolloutBatch,
    unique_layout: _UniqueStateLayout,
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
) -> tuple[tuple[SubgraphAnalysis, ...], dict[int, SubgraphAnalysis]]:
    unique_analyses: list[SubgraphAnalysis] = []
    analysis_lookup: dict[int, SubgraphAnalysis] = {}
    for flat_state_idx in unique_layout.unique_state_indices:
        graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
        state = rollout_batch.states[int(flat_state_idx)]
        analysis = _lookup_analysis(
            policy=policy,
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
            analysis_cache=analysis_cache,
        )
        unique_analyses.append(analysis)
        analysis_lookup[int(flat_state_idx)] = analysis
    return tuple(unique_analyses), analysis_lookup


def _lookup_backward_log_prob(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    backward_log_prob_cache: dict[tuple[int, tuple[Any, ...]], float],
) -> float:
    cache_key = (int(graph_idx), state.key())
    cached = backward_log_prob_cache.get(cache_key)
    if cached is None:
        cached = float(
            policy.compute_backward_log_prob(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                state=state,
            )
        )
        backward_log_prob_cache[cache_key] = cached
    return float(cached)


def _lookup_terminal_reward(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis: SubgraphAnalysis,
    terminal_reward_cache: dict[tuple[int, tuple[Any, ...]], SubgraphTerminalReward],
) -> SubgraphTerminalReward:
    cache_key = (int(graph_idx), state.key())
    cached = terminal_reward_cache.get(cache_key)
    if cached is None:
        cached = policy.compute_terminal_reward(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            analysis=analysis,
        )
        terminal_reward_cache[cache_key] = cached
    return cached


def _build_active_state_context(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    rollout_batch: SubgraphRolloutBatch,
    active_state_indices: tuple[int, ...] | list[int],
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
    action_pruning: Mapping[str, Any] | None,
    profile_prefix: str | None = None,
    profile_extra: str = "",
) -> _ActiveStateContext:
    device = prepared_batch.device
    analyze_context = (
        nullcontext()
        if profile_prefix is None
        else profile_cuda_memory(
            f"{profile_prefix}.analyze_rollout_batch",
            device=device,
            extra=profile_extra,
        )
    )
    with analyze_context:
        unique_layout = _build_unique_state_layout(
            rollout_batch=rollout_batch,
            active_state_indices=active_state_indices,
            device=device,
        )
        unique_analyses, analysis_lookup = _resolve_unique_state_analyses(
            policy=policy,
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            unique_layout=unique_layout,
            analysis_cache=analysis_cache,
        )
    encode_context = (
        nullcontext()
        if profile_prefix is None
        else profile_cuda_memory(
            f"{profile_prefix}.encode_state_features",
            device=device,
            extra=profile_extra,
        )
    )
    with encode_context:
        unique_state_features = policy.encode_state_features(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analysis_lookup,
            state_indices=unique_layout.unique_state_indices,
        )
    log_flow_context = (
        nullcontext()
        if profile_prefix is None
        else profile_cuda_memory(
            f"{profile_prefix}.compute_log_flows",
            device=device,
            extra=profile_extra,
        )
    )
    with log_flow_context:
        unique_log_flows = policy.compute_log_flows_from_state_features(
            prepared_batch=prepared_batch,
            state_features=unique_state_features,
            graph_ids=rollout_batch.graph_ids.index_select(
                0, unique_layout.unique_state_tensor
            ),
        )
    active_to_unique = [
        int(local_idx)
        for local_idx in unique_layout.active_to_unique.detach().cpu().tolist()
    ]
    active_analyses = tuple(
        unique_analyses[local_idx] for local_idx in active_to_unique
    )
    distribution_context = (
        nullcontext()
        if profile_prefix is None
        else profile_cuda_memory(
            f"{profile_prefix}.compute_action_distribution",
            device=device,
            extra=profile_extra,
        )
    )
    with distribution_context:
        unique_rollout_batch = SubgraphRolloutBatch(
            graph_ids=rollout_batch.graph_ids.index_select(
                0, unique_layout.unique_state_tensor
            ),
            states=tuple(
                rollout_batch.states[int(flat_state_idx)]
                for flat_state_idx in unique_layout.unique_state_indices
            ),
            done_mask=torch.zeros_like(
                unique_layout.unique_state_tensor,
                dtype=torch.bool,
            ),
            view_shape=(int(len(unique_layout.unique_state_indices)), 1),
        )
        unique_distribution_batch = (
            policy.build_action_distribution_from_state_features(
                prepared_batch=prepared_batch,
                rollout_batch=unique_rollout_batch,
                analyses=unique_analyses,
                state_features=unique_state_features,
                action_pruning=action_pruning,
            )
        )
    return _ActiveStateContext(
        unique_layout=unique_layout,
        active_analyses=active_analyses,
        active_log_flows=unique_log_flows.index_select(
            0, unique_layout.active_to_unique
        ),
        active_component_counts=torch.tensor(
            [float(analysis.anchor_component_count) for analysis in active_analyses],
            device=device,
            dtype=torch.float32,
        ),
        active_state_distributions=tuple(
            unique_distribution_batch.state_distributions[local_idx]
            for local_idx in active_to_unique
        ),
    )


def _resolve_rollout_terminal_analyses(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphPreparedBatch,
    rollout_batch: SubgraphRolloutBatch,
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
) -> tuple[SubgraphAnalysis, ...]:
    terminal_analyses: list[SubgraphAnalysis] = []
    for flat_state_idx, state in enumerate(rollout_batch.states):
        terminal_analyses.append(
            _lookup_analysis(
                policy=policy,
                prepared_batch=prepared_batch,
                graph_idx=int(rollout_batch.graph_ids[int(flat_state_idx)].item()),
                state=state,
                analysis_cache=analysis_cache,
            )
        )
    return tuple(terminal_analyses)


def _gate_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    return torch.stack(
        (state_distribution.stop_logit, state_distribution.continue_logit),
        dim=0,
    ).to(dtype=torch.float32)


def _node_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    return state_distribution.node_choice_logits.to(dtype=torch.float32)


def _relation_logits(
    state_distribution: HierarchicalStateActionDistribution,
    node_choice_idx: int,
) -> torch.Tensor:
    relation_slice = state_distribution.relation_slice(int(node_choice_idx))
    return state_distribution.relation_choice_logits[relation_slice].to(
        dtype=torch.float32
    )


def _edge_logits(
    state_distribution: HierarchicalStateActionDistribution,
    relation_choice_idx: int,
) -> torch.Tensor:
    edge_slice = state_distribution.edge_slice(int(relation_choice_idx))
    return state_distribution.edge_choice_logits[edge_slice].to(dtype=torch.float32)


def _stop_logits(
    state_distribution: HierarchicalStateActionDistribution,
) -> torch.Tensor:
    if int(state_distribution.stop_choice_logits.numel()) <= 0:
        raise RuntimeError("Stop distributions must expose at least one stop choice.")
    return state_distribution.stop_choice_logits.to(dtype=torch.float32)


def _select_stop_choice(
    *, state_distribution: HierarchicalStateActionDistribution
) -> int:
    if int(state_distribution.stop_choice_logits.numel()) <= 0:
        raise RuntimeError("Stop distributions must expose at least one stop choice.")
    return 0


def _sample_state_action(
    *,
    state_distribution: HierarchicalStateActionDistribution,
    temperature: float,
) -> tuple[SubgraphAction, torch.Tensor, int]:
    gate_logits = _gate_logits(state_distribution)
    gate_idx, _ = _sample_index(gate_logits, temperature=temperature)
    gate_log_prob = _log_softmax_choice(gate_logits, gate_idx)
    if gate_idx == 0 or int(state_distribution.node_choice_logits.numel()) <= 0:
        # STOP now terminates the current subgraph directly. Any answer supervision
        # is derived from the terminal topology, not from a separate answer choice.
        stop_logits = _stop_logits(state_distribution)
        stop_idx, _ = _sample_index(stop_logits, temperature=temperature)
        stop_log_prob = _log_softmax_choice(stop_logits, stop_idx)
        return (
            state_distribution.build_stop_action(int(stop_idx)),
            gate_log_prob + stop_log_prob,
            state_distribution.current_component_count,
        )
    node_logits = _node_logits(state_distribution)
    node_idx, _ = _sample_index(node_logits, temperature=temperature)
    node_log_prob = _log_softmax_choice(node_logits, node_idx)
    relation_logits = _relation_logits(state_distribution, int(node_idx))
    relation_idx, _ = _sample_index(relation_logits, temperature=temperature)
    relation_log_prob = _log_softmax_choice(relation_logits, relation_idx)
    relation_choice_slice = state_distribution.relation_slice(int(node_idx))
    relation_choice_idx = int(relation_choice_slice.start + int(relation_idx))
    edge_logits = _edge_logits(state_distribution, relation_choice_idx)
    edge_idx, _ = _sample_index(edge_logits, temperature=temperature)
    edge_log_prob = _log_softmax_choice(edge_logits, edge_idx)
    edge_choice_slice = state_distribution.edge_slice(relation_choice_idx)
    edge_choice_idx = int(edge_choice_slice.start + int(edge_idx))
    return (
        state_distribution.build_edge_action(edge_choice_idx),
        gate_log_prob + node_log_prob + relation_log_prob + edge_log_prob,
        state_distribution.edge_next_component_count(edge_choice_idx),
    )


def _teacher_forced_state_action(
    *,
    state_distribution: HierarchicalStateActionDistribution,
    planned_edge_id: int | None,
) -> tuple[SubgraphAction, torch.Tensor, int]:
    gate_logits = _gate_logits(state_distribution)
    if planned_edge_id is None:
        stop_idx = _select_stop_choice(state_distribution=state_distribution)
        stop_logits = _stop_logits(state_distribution)
        return (
            state_distribution.build_stop_action(int(stop_idx)),
            _log_softmax_choice(gate_logits, 0)
            + _log_softmax_choice(stop_logits, stop_idx),
            state_distribution.current_component_count,
        )
    if int(state_distribution.node_choice_logits.numel()) <= 0:
        raise RuntimeError(
            "Teacher-forced subgraph replay could not resolve any expandable node."
        )
    matching_edge_indices = torch.nonzero(
        state_distribution.edge_choice_edge_ids == int(planned_edge_id),
        as_tuple=False,
    ).view(-1)
    if int(matching_edge_indices.numel()) > 0:
        edge_choice_idx = int(matching_edge_indices[0].item())
        relation_choice_idx = int(
            state_distribution.edge_choice_relation_choice_indices[
                edge_choice_idx
            ].item()
        )
        node_choice_idx = int(
            state_distribution.relation_choice_node_choice_indices[
                relation_choice_idx
            ].item()
        )
        gate_log_prob = _log_softmax_choice(gate_logits, 1)
        node_log_prob = _log_softmax_choice(
            _node_logits(state_distribution), node_choice_idx
        )
        relation_choice_slice = state_distribution.relation_slice(node_choice_idx)
        relation_local_idx = int(relation_choice_idx - relation_choice_slice.start)
        relation_log_prob = _log_softmax_choice(
            _relation_logits(state_distribution, node_choice_idx),
            relation_local_idx,
        )
        edge_choice_slice = state_distribution.edge_slice(relation_choice_idx)
        edge_local_idx = int(edge_choice_idx - edge_choice_slice.start)
        edge_log_prob = _log_softmax_choice(
            _edge_logits(state_distribution, relation_choice_idx),
            edge_local_idx,
        )
        return (
            state_distribution.build_edge_action(edge_choice_idx),
            gate_log_prob + node_log_prob + relation_log_prob + edge_log_prob,
            state_distribution.edge_next_component_count(edge_choice_idx),
        )
    raise RuntimeError(
        "Teacher-forced subgraph replay could not resolve the planned edge under the "
        f"semantic hierarchical policy. planned_edge_id={planned_edge_id}"
    )


class SubgraphSampler:
    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = int(max_steps)

    def sample(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphPreparedBatch,
        rollouts_per_graph: int,
        temperature: float,
        proposal_bias_scale: float = 1.0,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphTrajectorySampleBatch:
        del proposal_bias_scale
        num_graphs = int(prepared_batch.num_graphs)
        num_rollouts = int(rollouts_per_graph)
        flat_states = num_graphs * num_rollouts
        max_actions = self.max_steps + 1
        analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis] = {}
        backward_log_prob_cache: dict[tuple[int, tuple[Any, ...]], float] = {}
        terminal_reward_cache: dict[
            tuple[int, tuple[Any, ...]], SubgraphTerminalReward
        ] = {}
        with profile_cuda_memory(
            "sampler.sample.initialize_rollout_batch",
            device=prepared_batch.device,
            extra=(
                f"num_graphs={num_graphs} num_rollouts={num_rollouts} "
                f"max_steps={self.max_steps}"
            ),
        ):
            rollout_batch = policy.initialize_rollout_batch(
                prepared_batch=prepared_batch,
                num_rollouts=num_rollouts,
            )
        with profile_cuda_memory(
            "sampler.sample.allocate_buffers",
            device=prepared_batch.device,
            extra=(
                f"flat_states={flat_states} max_actions={max_actions} "
                f"max_steps={self.max_steps}"
            ),
        ):
            state_log_flows = torch.zeros(
                (flat_states, max_actions),
                device=prepared_batch.device,
                dtype=torch.float32,
            )
            state_component_counts = torch.zeros_like(state_log_flows)
            log_pf_actions = torch.zeros_like(state_log_flows)
            log_pb_actions = torch.zeros_like(state_log_flows)
            log_reward_actions = torch.zeros_like(state_log_flows)
            action_mask = torch.zeros_like(state_log_flows, dtype=torch.bool)
            chosen_edge_ids = torch.full(
                (flat_states, self.max_steps),
                fill_value=-1,
                device=prepared_batch.device,
                dtype=torch.long,
            )
            chosen_source_graph_nodes = torch.full(
                (flat_states, self.max_steps),
                fill_value=-1,
                device=prepared_batch.device,
                dtype=torch.long,
            )
            stop_actions = torch.zeros(
                (flat_states, max_actions),
                device=prepared_batch.device,
                dtype=torch.bool,
            )
            termination_action_steps = torch.zeros(
                (flat_states,), device=prepared_batch.device, dtype=torch.long
            )
            terminal_commit_candidate_counts = torch.zeros_like(
                termination_action_steps
            )
            terminal_gold_answer_counts = torch.zeros_like(termination_action_steps)
            terminal_hit_mask = torch.zeros_like(
                termination_action_steps, dtype=torch.bool
            )
            terminal_component_counts = torch.zeros_like(termination_action_steps)
            terminal_answer_entity_ids: list[tuple[int, ...]] = [
                () for _ in range(flat_states)
            ]
        for action_step in range(max_actions):
            active_state_indices = rollout_batch.active_state_indices()
            active_states = int(len(active_state_indices))
            if active_states <= 0:
                break
            stage_extra = (
                f"action_step={action_step} active_states={active_states} "
                f"num_graphs={num_graphs} num_rollouts={num_rollouts}"
            )
            active_context = _build_active_state_context(
                policy=policy,
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                active_state_indices=active_state_indices,
                analysis_cache=analysis_cache,
                action_pruning=action_pruning,
                profile_prefix="sampler.sample",
                profile_extra=stage_extra,
            )
            current_log_flow = state_log_flows.new_zeros(
                (flat_states,), dtype=torch.float32
            )
            current_log_flow.index_copy_(
                0,
                active_context.unique_layout.active_state_tensor,
                active_context.active_log_flows,
            )
            state_log_flows[:, action_step] = current_log_flow
            state_component_counts[
                active_context.unique_layout.active_state_tensor, action_step
            ] = active_context.active_component_counts
            chosen_actions: list[SubgraphAction] = [SubgraphAction.stop()] * flat_states
            for state_idx in range(flat_states):
                if bool(rollout_batch.done_mask[state_idx].item()):
                    chosen_actions[state_idx] = SubgraphAction.stop()
            for local_state_idx, flat_state_idx in enumerate(
                active_context.unique_layout.active_state_indices
            ):
                state_distribution = active_context.active_state_distributions[
                    local_state_idx
                ]
                current_analysis = active_context.active_analyses[local_state_idx]
                current_state = rollout_batch.states[int(flat_state_idx)]
                graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
                action, log_pf, _next_components = _sample_state_action(
                    state_distribution=state_distribution,
                    temperature=float(temperature),
                )
                action_mask[flat_state_idx, action_step] = True
                chosen_actions[flat_state_idx] = action
                stop_actions[flat_state_idx, action_step] = bool(action.is_stop)
                log_pf_actions[flat_state_idx, action_step] = log_pf.to(
                    dtype=torch.float32
                )
                if action.is_stop:
                    terminal_reward = _lookup_terminal_reward(
                        policy=policy,
                        prepared_batch=prepared_batch,
                        graph_idx=graph_idx,
                        state=current_state,
                        analysis=current_analysis,
                        terminal_reward_cache=terminal_reward_cache,
                    )
                    log_reward_actions[flat_state_idx, action_step] = float(
                        terminal_reward.log_reward
                    )
                    termination_action_steps[flat_state_idx] = int(action_step + 1)
                    terminal_commit_candidate_counts[flat_state_idx] = int(
                        terminal_reward.answer_set.count
                    )
                    terminal_gold_answer_counts[flat_state_idx] = int(
                        terminal_reward.answer_set.gold_count
                    )
                    terminal_hit_mask[flat_state_idx] = bool(terminal_reward.hit)
                    terminal_component_counts[flat_state_idx] = int(
                        current_analysis.anchor_component_count
                    )
                    terminal_answer_entity_ids[flat_state_idx] = tuple(
                        int(entity_id) for entity_id in terminal_reward.answer_entities
                    )
                    continue
                if action_step >= self.max_steps:
                    raise RuntimeError(
                        "Expand chosen beyond max_steps in subgraph sampler."
                    )
                if action.edge_id is None:
                    raise RuntimeError("Expand actions must carry edge_id.")
                next_state = current_state.with_edge(int(action.edge_id))
                chosen_edge_ids[flat_state_idx, action_step] = int(action.edge_id)
                if action.source_graph_node is not None:
                    chosen_source_graph_nodes[flat_state_idx, action_step] = int(
                        action.source_graph_node
                    )
                log_pb_actions[flat_state_idx, action_step] = _lookup_backward_log_prob(
                    policy=policy,
                    prepared_batch=prepared_batch,
                    graph_idx=graph_idx,
                    state=next_state,
                    backward_log_prob_cache=backward_log_prob_cache,
                )
            with profile_cuda_memory(
                "sampler.sample.transition",
                device=prepared_batch.device,
                extra=stage_extra,
            ):
                rollout_batch = policy.transition(
                    rollout_batch=rollout_batch,
                    chosen_actions=tuple(chosen_actions),
                )
        if bool((termination_action_steps <= 0).any().item()):
            missing = torch.nonzero(termination_action_steps <= 0, as_tuple=False).view(
                -1
            )
            raise RuntimeError(
                "Subgraph sampler ended without explicit STOP for some rollouts. "
                f"missing={missing.detach().cpu().tolist()}"
            )
        with profile_cuda_memory(
            "sampler.sample.terminal_analyses",
            device=prepared_batch.device,
            extra=f"num_graphs={num_graphs} num_rollouts={num_rollouts}",
        ):
            terminal_analyses = _resolve_rollout_terminal_analyses(
                policy=policy,
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                analysis_cache=analysis_cache,
            )
        terminal_edge_ids = tuple(state.edge_ids for state in rollout_batch.states)
        terminal_node_ids = tuple(
            analysis.selected_node_ids for analysis in terminal_analyses
        )
        terminal_reachability_bits = tuple(
            dict(analysis.reachability_bits) for analysis in terminal_analyses
        )
        return SubgraphTrajectorySampleBatch(
            state_log_flows=state_log_flows.view(num_graphs, num_rollouts, max_actions),
            state_component_counts=state_component_counts.view(
                num_graphs, num_rollouts, max_actions
            ),
            log_pf_actions=log_pf_actions.view(num_graphs, num_rollouts, max_actions),
            log_pb_actions=log_pb_actions.view(num_graphs, num_rollouts, max_actions),
            log_reward_actions=log_reward_actions.view(
                num_graphs, num_rollouts, max_actions
            ),
            action_mask=action_mask.view(num_graphs, num_rollouts, max_actions),
            termination_action_steps=termination_action_steps.view(
                num_graphs, num_rollouts
            ),
            chosen_edge_ids=chosen_edge_ids.view(
                num_graphs, num_rollouts, self.max_steps
            ),
            chosen_source_graph_nodes=chosen_source_graph_nodes.view(
                num_graphs, num_rollouts, self.max_steps
            ),
            stop_actions=stop_actions.view(num_graphs, num_rollouts, max_actions),
            terminal_commit_candidate_counts=terminal_commit_candidate_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_gold_answer_counts=terminal_gold_answer_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_hit_mask=terminal_hit_mask.view(num_graphs, num_rollouts),
            terminal_component_counts=terminal_component_counts.view(
                num_graphs, num_rollouts
            ),
            terminal_edge_ids=terminal_edge_ids,
            terminal_node_ids=terminal_node_ids,
            terminal_reachability_bits=terminal_reachability_bits,
            terminal_answer_entity_ids=tuple(terminal_answer_entity_ids),
            sample_ids=prepared_batch.sample_ids,
            question_ids=prepared_batch.sample_ids,
            num_graphs=num_graphs,
            num_rollouts=num_rollouts,
        )

    def teacher_force(
        self,
        *,
        policy: SubgraphPolicy,
        prepared_batch: SubgraphPreparedBatch,
        edge_sequences: tuple[tuple[int, ...], ...],
    ) -> SubgraphTrajectorySampleBatch:
        num_graphs = int(prepared_batch.num_graphs)
        if int(len(edge_sequences)) != num_graphs:
            raise ValueError(
                "edge_sequences must align with prepared_batch.num_graphs."
            )
        flat_states = num_graphs
        max_actions = self.max_steps + 1
        analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis] = {}
        backward_log_prob_cache: dict[tuple[int, tuple[Any, ...]], float] = {}
        terminal_reward_cache: dict[
            tuple[int, tuple[Any, ...]], SubgraphTerminalReward
        ] = {}
        rollout_batch = policy.initialize_rollout_batch(
            prepared_batch=prepared_batch,
            num_rollouts=1,
        )
        state_log_flows = torch.zeros(
            (flat_states, max_actions),
            device=prepared_batch.device,
            dtype=torch.float32,
        )
        state_component_counts = torch.zeros_like(state_log_flows)
        log_pf_actions = torch.zeros_like(state_log_flows)
        log_pb_actions = torch.zeros_like(state_log_flows)
        log_reward_actions = torch.zeros_like(state_log_flows)
        action_mask = torch.zeros_like(state_log_flows, dtype=torch.bool)
        chosen_edge_ids = torch.full(
            (flat_states, self.max_steps),
            fill_value=-1,
            device=prepared_batch.device,
            dtype=torch.long,
        )
        chosen_source_graph_nodes = torch.full(
            (flat_states, self.max_steps),
            fill_value=-1,
            device=prepared_batch.device,
            dtype=torch.long,
        )
        stop_actions = torch.zeros(
            (flat_states, max_actions),
            device=prepared_batch.device,
            dtype=torch.bool,
        )
        termination_action_steps = torch.zeros(
            (flat_states,), device=prepared_batch.device, dtype=torch.long
        )
        terminal_commit_candidate_counts = torch.zeros_like(termination_action_steps)
        terminal_gold_answer_counts = torch.zeros_like(termination_action_steps)
        terminal_hit_mask = torch.zeros_like(termination_action_steps, dtype=torch.bool)
        terminal_component_counts = torch.zeros_like(termination_action_steps)
        terminal_answer_entity_ids: list[tuple[int, ...]] = [
            () for _ in range(flat_states)
        ]
        for action_step in range(max_actions):
            active_state_indices = rollout_batch.active_state_indices()
            active_states = int(len(active_state_indices))
            if active_states <= 0:
                break
            active_context = _build_active_state_context(
                policy=policy,
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                active_state_indices=active_state_indices,
                analysis_cache=analysis_cache,
                action_pruning=None,
            )
            current_log_flow = state_log_flows.new_zeros(
                (flat_states,), dtype=torch.float32
            )
            current_log_flow.index_copy_(
                0,
                active_context.unique_layout.active_state_tensor,
                active_context.active_log_flows,
            )
            state_log_flows[:, action_step] = current_log_flow
            state_component_counts[
                active_context.unique_layout.active_state_tensor, action_step
            ] = active_context.active_component_counts
            chosen_actions: list[SubgraphAction] = [SubgraphAction.stop()] * flat_states
            for local_state_idx, flat_state_idx in enumerate(
                active_context.unique_layout.active_state_indices
            ):
                state_distribution = active_context.active_state_distributions[
                    local_state_idx
                ]
                current_analysis = active_context.active_analyses[local_state_idx]
                current_state = rollout_batch.states[int(flat_state_idx)]
                graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
                planned_edges = edge_sequences[int(flat_state_idx)]
                planned_edge_id = (
                    int(planned_edges[action_step])
                    if action_step < int(len(planned_edges))
                    else None
                )
                action, log_pf, _next_components = _teacher_forced_state_action(
                    state_distribution=state_distribution,
                    planned_edge_id=planned_edge_id,
                )
                action_mask[flat_state_idx, action_step] = True
                chosen_actions[flat_state_idx] = action
                stop_actions[flat_state_idx, action_step] = bool(action.is_stop)
                log_pf_actions[flat_state_idx, action_step] = log_pf.to(
                    dtype=torch.float32
                )
                if action.is_stop:
                    terminal_reward = _lookup_terminal_reward(
                        policy=policy,
                        prepared_batch=prepared_batch,
                        graph_idx=graph_idx,
                        state=current_state,
                        analysis=current_analysis,
                        terminal_reward_cache=terminal_reward_cache,
                    )
                    log_reward_actions[flat_state_idx, action_step] = float(
                        terminal_reward.log_reward
                    )
                    termination_action_steps[flat_state_idx] = int(action_step + 1)
                    terminal_commit_candidate_counts[flat_state_idx] = int(
                        terminal_reward.answer_set.count
                    )
                    terminal_gold_answer_counts[flat_state_idx] = int(
                        terminal_reward.answer_set.gold_count
                    )
                    terminal_hit_mask[flat_state_idx] = bool(terminal_reward.hit)
                    terminal_component_counts[flat_state_idx] = int(
                        current_analysis.anchor_component_count
                    )
                    terminal_answer_entity_ids[flat_state_idx] = tuple(
                        int(entity_id) for entity_id in terminal_reward.answer_entities
                    )
                    continue
                if action.edge_id is None:
                    raise RuntimeError(
                        "Teacher-forced expand actions must carry edge_id."
                    )
                next_state = current_state.with_edge(int(action.edge_id))
                chosen_edge_ids[flat_state_idx, action_step] = int(action.edge_id)
                if action.source_graph_node is not None:
                    chosen_source_graph_nodes[flat_state_idx, action_step] = int(
                        action.source_graph_node
                    )
                log_pb_actions[flat_state_idx, action_step] = _lookup_backward_log_prob(
                    policy=policy,
                    prepared_batch=prepared_batch,
                    graph_idx=graph_idx,
                    state=next_state,
                    backward_log_prob_cache=backward_log_prob_cache,
                )
            rollout_batch = policy.transition(
                rollout_batch=rollout_batch,
                chosen_actions=tuple(chosen_actions),
            )
        if bool((termination_action_steps <= 0).any().item()):
            missing = torch.nonzero(termination_action_steps <= 0, as_tuple=False).view(
                -1
            )
            raise RuntimeError(
                "Teacher-forced subgraph replay ended without explicit STOP for some trajectories. "
                f"missing={missing.detach().cpu().tolist()}"
            )
        terminal_analyses = _resolve_rollout_terminal_analyses(
            policy=policy,
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analysis_cache=analysis_cache,
        )
        terminal_edge_ids = tuple(state.edge_ids for state in rollout_batch.states)
        terminal_node_ids = tuple(
            analysis.selected_node_ids for analysis in terminal_analyses
        )
        terminal_reachability_bits = tuple(
            dict(analysis.reachability_bits) for analysis in terminal_analyses
        )
        return SubgraphTrajectorySampleBatch(
            state_log_flows=state_log_flows.view(num_graphs, 1, max_actions),
            state_component_counts=state_component_counts.view(
                num_graphs, 1, max_actions
            ),
            log_pf_actions=log_pf_actions.view(num_graphs, 1, max_actions),
            log_pb_actions=log_pb_actions.view(num_graphs, 1, max_actions),
            log_reward_actions=log_reward_actions.view(num_graphs, 1, max_actions),
            action_mask=action_mask.view(num_graphs, 1, max_actions),
            termination_action_steps=termination_action_steps.view(num_graphs, 1),
            chosen_edge_ids=chosen_edge_ids.view(num_graphs, 1, self.max_steps),
            chosen_source_graph_nodes=chosen_source_graph_nodes.view(
                num_graphs, 1, self.max_steps
            ),
            stop_actions=stop_actions.view(num_graphs, 1, max_actions),
            terminal_commit_candidate_counts=terminal_commit_candidate_counts.view(
                num_graphs, 1
            ),
            terminal_gold_answer_counts=terminal_gold_answer_counts.view(num_graphs, 1),
            terminal_hit_mask=terminal_hit_mask.view(num_graphs, 1),
            terminal_component_counts=terminal_component_counts.view(num_graphs, 1),
            terminal_edge_ids=terminal_edge_ids,
            terminal_node_ids=terminal_node_ids,
            terminal_reachability_bits=terminal_reachability_bits,
            terminal_answer_entity_ids=tuple(terminal_answer_entity_ids),
            sample_ids=prepared_batch.sample_ids,
            question_ids=prepared_batch.sample_ids,
            num_graphs=num_graphs,
            num_rollouts=1,
        )


__all__ = ["SubgraphSampler", "SubgraphTrajectorySampleBatch"]
