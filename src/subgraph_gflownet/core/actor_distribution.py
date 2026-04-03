from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import torch

from .action_surface import StateActionSurfaceContext, collect_state_action_surface
from .actor_types import (
    HierarchicalStateActionDistribution,
    SubgraphActionDistribution,
    _ActionDistributionBuildStats,
)
from .state import SubgraphAnalysis, SubgraphRolloutBatch
from .subgraph_batch import SubgraphBatch

if TYPE_CHECKING:
    from .actor import SubgraphActor


def _offsets(counts: list[int]) -> list[int]:
    offsets = [0]
    for count in counts:
        offsets.append(int(offsets[-1] + int(count)))
    return offsets


def _build_stop_choice_tensors(
    *, device: torch.device, current_answer_candidate_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((1,), -1, device=device, dtype=torch.long),
        torch.tensor(
            [int(current_answer_candidate_count)], device=device, dtype=torch.long
        ),
        torch.zeros((1,), device=device, dtype=torch.float32),
    )


def _masked_continue_logit(
    self: SubgraphActor,
    *,
    continue_logit: torch.Tensor,
    context: StateActionSurfaceContext,
) -> torch.Tensor:
    if (
        int(context.linearized.node_choice_graph_node_ids.numel()) <= 0
        or int(context.state_num_edges) >= self.max_steps
    ):
        return torch.full_like(continue_logit, float("-inf"))
    return continue_logit


def _build_distribution(
    *,
    context: StateActionSurfaceContext,
    stop_logit: torch.Tensor,
    continue_logit: torch.Tensor,
    stop_choice_logits: torch.Tensor,
    stop_choice_answer_entity_ids: torch.Tensor,
    stop_choice_support_node_counts: torch.Tensor,
    node_choice_logits: torch.Tensor,
    relation_choice_logits: torch.Tensor,
    edge_choice_logits: torch.Tensor,
) -> HierarchicalStateActionDistribution:
    linearized = context.linearized
    return HierarchicalStateActionDistribution(
        flat_state_index=int(context.flat_state_index),
        stop_logit=stop_logit,
        continue_logit=continue_logit,
        stop_choice_logits=stop_choice_logits,
        stop_choice_answer_entity_ids=stop_choice_answer_entity_ids,
        stop_choice_support_node_counts=stop_choice_support_node_counts,
        node_choice_graph_node_ids=linearized.node_choice_graph_node_ids,
        node_choice_logits=node_choice_logits,
        relation_choice_relation_ids=linearized.relation_choice_relation_ids,
        relation_choice_logits=relation_choice_logits,
        relation_choice_node_choice_indices=linearized.relation_choice_node_choice_indices,
        edge_choice_edge_ids=linearized.edge_choice_edge_ids,
        edge_choice_source_graph_nodes=linearized.edge_choice_source_graph_nodes,
        edge_choice_relation_ids=linearized.edge_choice_relation_ids,
        edge_choice_target_graph_nodes=linearized.edge_choice_target_graph_nodes,
        edge_choice_logits=edge_choice_logits,
        edge_choice_next_component_counts=linearized.edge_choice_next_component_counts,
        edge_choice_question_similarity=linearized.edge_choice_question_similarity,
        edge_choice_semantic_overlap=linearized.edge_choice_semantic_overlap,
        edge_choice_action_new_bit_gain=linearized.edge_choice_action_new_bit_gain,
        edge_choice_answer_candidate_counts=linearized.edge_choice_answer_candidate_counts,
        edge_choice_target_answer_distance=linearized.edge_choice_target_answer_distance,
        edge_choice_relation_choice_indices=linearized.edge_choice_relation_choice_indices,
        node_relation_ptr=linearized.node_relation_ptr,
        relation_edge_ptr=linearized.relation_edge_ptr,
        current_component_count=int(context.current_components),
        current_answer_candidate_count=int(context.current_answer_candidate_count),
        current_oracle_distance=float(context.current_oracle_distance),
    )


def _collect_state_distribution_context(
    self: SubgraphActor,
    *,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    flat_state_index: int,
    analysis: SubgraphAnalysis,
    action_pruning: Mapping[str, Any] | None = None,
) -> StateActionSurfaceContext:
    return collect_state_action_surface(
        batch=prepared_batch,
        rollout_batch=rollout_batch,
        flat_state_index=int(flat_state_index),
        analysis=analysis,
        max_steps=self.max_steps,
        action_pruning=action_pruning,
    )


def build_state_distribution(
    self: SubgraphActor,
    *,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    flat_state_index: int,
    analysis: SubgraphAnalysis,
    state_feature: torch.Tensor,
    action_pruning: Mapping[str, Any] | None = None,
) -> tuple[HierarchicalStateActionDistribution, _ActionDistributionBuildStats]:
    device = prepared_batch.device
    context = self._collect_state_distribution_context(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
        flat_state_index=int(flat_state_index),
        analysis=analysis,
        action_pruning=action_pruning,
    )
    linearized = context.linearized

    if int(linearized.node_choice_graph_node_ids.numel()) > 0:
        node_choice_logits = self._build_node_logits_batch(
            state_feature=state_feature,
            node_features=prepared_batch.node_tokens.index_select(
                0, linearized.node_choice_graph_node_ids
            ),
            node_is_anchor=linearized.node_choice_is_anchor,
            node_reachability_bits=linearized.node_choice_reachability_bits,
            max_anchor_count=context.max_anchor_count,
            full_mask=context.full_mask,
            node_candidate_counts=linearized.node_choice_candidate_counts,
        )
    else:
        node_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

    if int(linearized.relation_choice_relation_ids.numel()) > 0:
        relation_source_graph_nodes = (
            linearized.node_choice_graph_node_ids.index_select(
                0, linearized.relation_choice_node_choice_indices
            )
        )
        relation_choice_logits = self._build_relation_logits_batch(
            state_feature=state_feature,
            src_features=prepared_batch.node_tokens.index_select(
                0, relation_source_graph_nodes
            ),
            relation_features=prepared_batch.relation_tokens.index_select(
                0, linearized.relation_choice_relation_ids
            ),
            relation_num_edges=linearized.relation_choice_num_edges,
            relation_max_new_bit_gain=linearized.relation_choice_max_new_bit_gain,
            relation_max_answer_candidate_counts=(
                linearized.relation_choice_max_answer_candidate_counts
            ),
            relation_max_semantic_overlap=(
                linearized.relation_choice_max_semantic_overlap
            ),
        )
    else:
        relation_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

    if int(linearized.edge_choice_edge_ids.numel()) > 0:
        edge_choice_logits = self._build_edge_logits_batch(
            state_feature=state_feature,
            src_features=prepared_batch.node_tokens.index_select(
                0, linearized.edge_choice_source_graph_nodes
            ),
            relation_features=prepared_batch.relation_tokens.index_select(
                0, linearized.edge_choice_relation_ids
            ),
            dst_features=prepared_batch.node_tokens.index_select(
                0, linearized.edge_choice_target_graph_nodes
            ),
            current_components=context.current_components,
            next_component_counts=linearized.edge_choice_next_component_counts,
            semantic_overlap=linearized.edge_choice_semantic_overlap,
            action_new_bit_gain=linearized.edge_choice_action_new_bit_gain,
            answer_candidate_counts=linearized.edge_choice_answer_candidate_counts,
        )
    else:
        edge_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

    (
        stop_choice_answer_entity_ids,
        stop_choice_support_node_counts,
        stop_choice_logits,
    ) = _build_stop_choice_tensors(
        device=device,
        current_answer_candidate_count=int(context.current_answer_candidate_count),
    )
    stop_logit = self.stop_head(state_feature.unsqueeze(0)).squeeze(0).squeeze(-1)
    stop_logit = stop_logit.to(dtype=torch.float32) + self._build_failure_stop_logit(
        state_feature=state_feature,
        num_answer_ready=int(context.current_answer_candidate_count),
        current_components=context.current_components,
        current_edges=context.state_num_edges,
        device=device,
    ).to(dtype=torch.float32)
    continue_logit = (
        self.continue_head(state_feature.unsqueeze(0)).squeeze(0).squeeze(-1)
    )
    continue_logit = _masked_continue_logit(
        self,
        continue_logit=continue_logit.to(dtype=torch.float32),
        context=context,
    )
    distribution = _build_distribution(
        context=context,
        stop_logit=stop_logit,
        continue_logit=continue_logit,
        stop_choice_logits=stop_choice_logits,
        stop_choice_answer_entity_ids=stop_choice_answer_entity_ids,
        stop_choice_support_node_counts=stop_choice_support_node_counts,
        node_choice_logits=node_choice_logits,
        relation_choice_logits=relation_choice_logits,
        edge_choice_logits=edge_choice_logits,
    )
    return (
        distribution,
        _ActionDistributionBuildStats(
            raw_candidate_count=int(linearized.edge_choice_edge_ids.numel()),
            stop_choice_count=int(distribution.stop_choice_logits.numel()),
        ),
    )


def build_action_distribution(
    self: SubgraphActor,
    *,
    prepared_batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    analyses: tuple[SubgraphAnalysis, ...],
    state_features: torch.Tensor,
    action_pruning: Mapping[str, Any] | None = None,
) -> SubgraphActionDistribution:
    device = prepared_batch.device
    active_state_indices = rollout_batch.active_state_indices()
    if not active_state_indices:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        empty_state = state_features.new_empty((0, self.hidden_dim))
        return SubgraphActionDistribution(
            flat_state_indices=empty,
            state_features=empty_state,
            state_distributions=(),
        )

    active_state_tensor = torch.tensor(
        active_state_indices, device=device, dtype=torch.long
    )
    active_state_features = state_features.index_select(0, active_state_tensor)
    contexts = [
        self._collect_state_distribution_context(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            flat_state_index=int(flat_state_idx),
            analysis=analyses[int(flat_state_idx)],
            action_pruning=action_pruning,
        )
        for flat_state_idx in active_state_indices
    ]
    num_states = int(len(contexts))
    current_components = torch.tensor(
        [int(context.current_components) for context in contexts],
        device=device,
        dtype=torch.long,
    )
    state_num_edges = torch.tensor(
        [int(context.state_num_edges) for context in contexts],
        device=device,
        dtype=torch.long,
    )
    answer_ready_count_tensor = torch.tensor(
        [int(context.current_answer_candidate_count) for context in contexts],
        device=device,
        dtype=torch.long,
    )
    stop_logits = (
        self.stop_head(active_state_features).squeeze(-1).to(dtype=torch.float32)
    )
    continue_logits = (
        self.continue_head(active_state_features).squeeze(-1).to(dtype=torch.float32)
    )
    stop_logits = stop_logits + self._build_failure_stop_logits(
        state_features=active_state_features,
        num_answer_ready=answer_ready_count_tensor,
        current_components=current_components,
        current_edges=state_num_edges,
    )

    node_counts = [
        int(context.linearized.node_choice_graph_node_ids.numel())
        for context in contexts
    ]
    node_offsets = _offsets(node_counts)
    if node_offsets[-1] > 0:
        node_state_indices = torch.repeat_interleave(
            torch.arange(num_states, device=device, dtype=torch.long),
            torch.tensor(node_counts, device=device, dtype=torch.long),
        )
        node_graph_node_ids = torch.cat(
            [
                context.linearized.node_choice_graph_node_ids
                for context in contexts
                if int(context.linearized.node_choice_graph_node_ids.numel()) > 0
            ],
            dim=0,
        )
        node_choice_logits = self._build_node_logits(
            state_features=active_state_features.index_select(0, node_state_indices),
            node_features=prepared_batch.node_tokens.index_select(
                0, node_graph_node_ids
            ),
            node_is_anchor=torch.cat(
                [
                    context.linearized.node_choice_is_anchor
                    for context in contexts
                    if int(context.linearized.node_choice_is_anchor.numel()) > 0
                ],
                dim=0,
            ),
            node_reachability_bits=torch.cat(
                [
                    context.linearized.node_choice_reachability_bits
                    for context in contexts
                    if int(context.linearized.node_choice_reachability_bits.numel()) > 0
                ],
                dim=0,
            ),
            max_anchor_counts=torch.tensor(
                [int(context.max_anchor_count) for context in contexts],
                device=device,
                dtype=torch.long,
            ).index_select(0, node_state_indices),
            full_masks=torch.tensor(
                [int(context.full_mask) for context in contexts],
                device=device,
                dtype=torch.long,
            ).index_select(0, node_state_indices),
            node_candidate_counts=torch.cat(
                [
                    context.linearized.node_choice_candidate_counts
                    for context in contexts
                    if int(context.linearized.node_choice_candidate_counts.numel()) > 0
                ],
                dim=0,
            ),
        )
    else:
        node_choice_logits = active_state_features.new_empty((0,), dtype=torch.float32)

    relation_counts = [
        int(context.linearized.relation_choice_relation_ids.numel())
        for context in contexts
    ]
    relation_offsets = _offsets(relation_counts)
    if relation_offsets[-1] > 0:
        relation_state_indices = torch.repeat_interleave(
            torch.arange(num_states, device=device, dtype=torch.long),
            torch.tensor(relation_counts, device=device, dtype=torch.long),
        )
        relation_choice_relation_ids = torch.cat(
            [
                context.linearized.relation_choice_relation_ids
                for context in contexts
                if int(context.linearized.relation_choice_relation_ids.numel()) > 0
            ],
            dim=0,
        )
        relation_source_graph_nodes = torch.cat(
            [
                context.linearized.node_choice_graph_node_ids.index_select(
                    0, context.linearized.relation_choice_node_choice_indices
                )
                for context in contexts
                if int(context.linearized.relation_choice_relation_ids.numel()) > 0
            ],
            dim=0,
        )
        relation_choice_logits = self._build_relation_logits(
            state_features=active_state_features.index_select(
                0, relation_state_indices
            ),
            src_features=prepared_batch.node_tokens.index_select(
                0, relation_source_graph_nodes
            ),
            relation_features=prepared_batch.relation_tokens.index_select(
                0, relation_choice_relation_ids
            ),
            relation_num_edges=torch.cat(
                [
                    context.linearized.relation_choice_num_edges
                    for context in contexts
                    if int(context.linearized.relation_choice_num_edges.numel()) > 0
                ],
                dim=0,
            ),
            relation_max_new_bit_gain=torch.cat(
                [
                    context.linearized.relation_choice_max_new_bit_gain
                    for context in contexts
                    if int(context.linearized.relation_choice_max_new_bit_gain.numel())
                    > 0
                ],
                dim=0,
            ),
            relation_max_answer_candidate_counts=torch.cat(
                [
                    context.linearized.relation_choice_max_answer_candidate_counts
                    for context in contexts
                    if int(
                        context.linearized.relation_choice_max_answer_candidate_counts.numel()
                    )
                    > 0
                ],
                dim=0,
            ),
            relation_max_semantic_overlap=torch.cat(
                [
                    context.linearized.relation_choice_max_semantic_overlap
                    for context in contexts
                    if int(
                        context.linearized.relation_choice_max_semantic_overlap.numel()
                    )
                    > 0
                ],
                dim=0,
            ),
        )
    else:
        relation_choice_logits = active_state_features.new_empty(
            (0,), dtype=torch.float32
        )

    edge_counts = [
        int(context.linearized.edge_choice_edge_ids.numel()) for context in contexts
    ]
    edge_offsets = _offsets(edge_counts)
    if edge_offsets[-1] > 0:
        edge_state_indices = torch.repeat_interleave(
            torch.arange(num_states, device=device, dtype=torch.long),
            torch.tensor(edge_counts, device=device, dtype=torch.long),
        )
        edge_source_graph_nodes = torch.cat(
            [
                context.linearized.edge_choice_source_graph_nodes
                for context in contexts
                if int(context.linearized.edge_choice_source_graph_nodes.numel()) > 0
            ],
            dim=0,
        )
        edge_relation_ids = torch.cat(
            [
                context.linearized.edge_choice_relation_ids
                for context in contexts
                if int(context.linearized.edge_choice_relation_ids.numel()) > 0
            ],
            dim=0,
        )
        edge_target_graph_nodes = torch.cat(
            [
                context.linearized.edge_choice_target_graph_nodes
                for context in contexts
                if int(context.linearized.edge_choice_target_graph_nodes.numel()) > 0
            ],
            dim=0,
        )
        edge_choice_logits = self._build_edge_logits(
            state_features=active_state_features.index_select(0, edge_state_indices),
            src_features=prepared_batch.node_tokens.index_select(
                0, edge_source_graph_nodes
            ),
            relation_features=prepared_batch.relation_tokens.index_select(
                0, edge_relation_ids
            ),
            dst_features=prepared_batch.node_tokens.index_select(
                0, edge_target_graph_nodes
            ),
            current_components=current_components.index_select(0, edge_state_indices),
            next_component_counts=torch.cat(
                [
                    context.linearized.edge_choice_next_component_counts
                    for context in contexts
                    if int(context.linearized.edge_choice_next_component_counts.numel())
                    > 0
                ],
                dim=0,
            ),
            semantic_overlap=torch.cat(
                [
                    context.linearized.edge_choice_semantic_overlap
                    for context in contexts
                    if int(context.linearized.edge_choice_semantic_overlap.numel()) > 0
                ],
                dim=0,
            ),
            action_new_bit_gain=torch.cat(
                [
                    context.linearized.edge_choice_action_new_bit_gain
                    for context in contexts
                    if int(context.linearized.edge_choice_action_new_bit_gain.numel())
                    > 0
                ],
                dim=0,
            ),
            answer_candidate_counts=torch.cat(
                [
                    context.linearized.edge_choice_answer_candidate_counts
                    for context in contexts
                    if int(
                        context.linearized.edge_choice_answer_candidate_counts.numel()
                    )
                    > 0
                ],
                dim=0,
            ),
        )
    else:
        edge_choice_logits = active_state_features.new_empty((0,), dtype=torch.float32)

    state_distributions: list[HierarchicalStateActionDistribution] = []
    total_raw_candidates = 0
    max_raw_candidates = 0
    total_stop_choices = 0
    max_stop_choices = 0
    for local_state_idx, context in enumerate(contexts):
        node_choice_slice = slice(
            node_offsets[local_state_idx], node_offsets[local_state_idx + 1]
        )
        relation_choice_slice = slice(
            relation_offsets[local_state_idx], relation_offsets[local_state_idx + 1]
        )
        edge_choice_slice = slice(
            edge_offsets[local_state_idx], edge_offsets[local_state_idx + 1]
        )
        (
            stop_choice_answer_entity_ids,
            stop_choice_support_node_counts,
            stop_choice_logits,
        ) = _build_stop_choice_tensors(
            device=device,
            current_answer_candidate_count=int(context.current_answer_candidate_count),
        )
        distribution = _build_distribution(
            context=context,
            stop_logit=stop_logits[local_state_idx],
            continue_logit=_masked_continue_logit(
                self,
                continue_logit=continue_logits[local_state_idx],
                context=context,
            ),
            stop_choice_logits=stop_choice_logits,
            stop_choice_answer_entity_ids=stop_choice_answer_entity_ids,
            stop_choice_support_node_counts=stop_choice_support_node_counts,
            node_choice_logits=node_choice_logits[node_choice_slice],
            relation_choice_logits=relation_choice_logits[relation_choice_slice],
            edge_choice_logits=edge_choice_logits[edge_choice_slice],
        )
        raw_candidate_count = int(context.linearized.edge_choice_edge_ids.numel())
        stop_choice_count = int(stop_choice_logits.numel())
        total_raw_candidates += raw_candidate_count
        max_raw_candidates = max(max_raw_candidates, raw_candidate_count)
        total_stop_choices += stop_choice_count
        max_stop_choices = max(max_stop_choices, stop_choice_count)
        state_distributions.append(distribution)

    self._log_action_distribution_stats(
        active_state_count=int(len(active_state_indices)),
        total_raw_candidates=total_raw_candidates,
        max_raw_candidates=max_raw_candidates,
        total_stop_choices=total_stop_choices,
        max_stop_choices=max_stop_choices,
    )
    return SubgraphActionDistribution(
        flat_state_indices=active_state_tensor,
        state_features=active_state_features,
        state_distributions=tuple(state_distributions),
    )


__all__ = [
    "_collect_state_distribution_context",
    "build_action_distribution",
    "build_state_distribution",
]
