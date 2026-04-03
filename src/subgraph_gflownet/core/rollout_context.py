from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .actor import HierarchicalStateActionDistribution
from .cuda_memory import profile_cuda_memory
from .policy import SubgraphPolicy
from .state import SubgraphAnalysis, SubgraphRolloutBatch, SubgraphState
from .subgraph_batch import SubgraphBatch


@dataclass(frozen=True)
class UniqueStateLayout:
    active_state_indices: tuple[int, ...]
    active_state_tensor: torch.Tensor
    unique_state_indices: tuple[int, ...]
    unique_state_tensor: torch.Tensor
    active_to_unique: torch.Tensor


@dataclass(frozen=True)
class ActiveStateContext:
    unique_layout: UniqueStateLayout
    active_analyses: tuple[SubgraphAnalysis, ...]
    active_log_flows: torch.Tensor
    active_component_counts: torch.Tensor
    active_state_distributions: tuple[HierarchicalStateActionDistribution, ...]


def build_unique_state_layout(
    *,
    rollout_batch: SubgraphRolloutBatch,
    active_state_indices: tuple[int, ...] | list[int],
    device: torch.device,
) -> UniqueStateLayout:
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
    return UniqueStateLayout(
        active_state_indices=normalized_indices,
        active_state_tensor=torch.tensor(
            normalized_indices, device=device, dtype=torch.long
        ),
        unique_state_indices=tuple(unique_state_indices),
        unique_state_tensor=torch.tensor(
            unique_state_indices, device=device, dtype=torch.long
        ),
        active_to_unique=torch.tensor(
            active_to_unique, device=device, dtype=torch.long
        ),
    )


def lookup_analysis(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
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


def resolve_unique_state_analyses(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    unique_layout: UniqueStateLayout,
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
) -> tuple[tuple[SubgraphAnalysis, ...], dict[int, SubgraphAnalysis]]:
    unique_analyses: list[SubgraphAnalysis] = []
    analysis_lookup: dict[int, SubgraphAnalysis] = {}
    for flat_state_idx in unique_layout.unique_state_indices:
        graph_idx = int(rollout_batch.graph_ids[int(flat_state_idx)].item())
        state = rollout_batch.states[int(flat_state_idx)]
        analysis = lookup_analysis(
            policy=policy,
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
            analysis_cache=analysis_cache,
        )
        unique_analyses.append(analysis)
        analysis_lookup[int(flat_state_idx)] = analysis
    return tuple(unique_analyses), analysis_lookup


def build_active_state_context(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    active_state_indices: tuple[int, ...] | list[int],
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
    action_pruning: Mapping[str, Any] | None,
    profile_prefix: str | None = None,
    profile_extra: str = "",
) -> ActiveStateContext:
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
        unique_layout = build_unique_state_layout(
            rollout_batch=rollout_batch,
            active_state_indices=active_state_indices,
            device=device,
        )
        unique_analyses, analysis_lookup = resolve_unique_state_analyses(
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
                unique_layout.unique_state_tensor, dtype=torch.bool
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

    return ActiveStateContext(
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


def resolve_rollout_terminal_analyses(
    *,
    policy: SubgraphPolicy,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    analysis_cache: dict[tuple[int, tuple[Any, ...]], SubgraphAnalysis],
) -> tuple[SubgraphAnalysis, ...]:
    terminal_analyses: list[SubgraphAnalysis] = []
    for flat_state_idx, state in enumerate(rollout_batch.states):
        terminal_analyses.append(
            lookup_analysis(
                policy=policy,
                prepared_batch=prepared_batch,
                graph_idx=int(rollout_batch.graph_ids[int(flat_state_idx)].item()),
                state=state,
                analysis_cache=analysis_cache,
            )
        )
    return tuple(terminal_analyses)


__all__ = [
    "ActiveStateContext",
    "UniqueStateLayout",
    "build_active_state_context",
    "lookup_analysis",
    "resolve_rollout_terminal_analyses",
]
