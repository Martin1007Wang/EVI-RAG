from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .semantic_oracles import oracle_distance, resolve_admissible_answer_set
from .state import SubgraphAnalysis, SubgraphRolloutBatch
from .subgraph_batch import SubgraphBatch


def _bit_count(bits: int) -> int:
    return int(int(bits).bit_count())


def resolve_action_pruning_limits(
    action_pruning: Mapping[str, Any] | None,
) -> tuple[int | None, int | None]:
    if action_pruning is None:
        return None, None
    per_node_top_k = int(action_pruning.get("per_node_top_k", 0)) or None
    per_state_top_k = int(action_pruning.get("per_state_top_k", 0)) or None
    return per_node_top_k, per_state_top_k


def estimate_next_components(
    *,
    analysis: SubgraphAnalysis,
    source_graph_node: int,
    target_graph_node: int,
) -> int:
    current_components = int(analysis.anchor_component_count)
    source_component = int(analysis.component_labels.get(int(source_graph_node), -1))
    target_component = int(analysis.component_labels.get(int(target_graph_node), -1))
    if (
        source_component >= 0
        and target_component >= 0
        and int(source_component) != int(target_component)
    ):
        return max(current_components - 1, 1)
    return current_components


@dataclass(frozen=True)
class LinearizedActionSurface:
    edge_choice_edge_ids: torch.Tensor
    edge_choice_source_graph_nodes: torch.Tensor
    edge_choice_relation_ids: torch.Tensor
    edge_choice_target_graph_nodes: torch.Tensor
    edge_choice_next_component_counts: torch.Tensor
    edge_choice_question_similarity: torch.Tensor
    edge_choice_semantic_overlap: torch.Tensor
    edge_choice_action_new_bit_gain: torch.Tensor
    edge_choice_answer_candidate_counts: torch.Tensor
    edge_choice_target_answer_distance: torch.Tensor
    edge_choice_relation_choice_indices: torch.Tensor
    node_choice_graph_node_ids: torch.Tensor
    node_choice_candidate_counts: torch.Tensor
    node_choice_reachability_bits: torch.Tensor
    node_choice_is_anchor: torch.Tensor
    relation_choice_relation_ids: torch.Tensor
    relation_choice_node_choice_indices: torch.Tensor
    relation_choice_num_edges: torch.Tensor
    relation_choice_max_new_bit_gain: torch.Tensor
    relation_choice_max_answer_candidate_counts: torch.Tensor
    relation_choice_max_semantic_overlap: torch.Tensor
    node_relation_ptr: torch.Tensor
    relation_edge_ptr: torch.Tensor

    @classmethod
    def empty(cls, *, device: torch.device) -> "LinearizedActionSurface":
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_float = torch.empty((0,), device=device, dtype=torch.float32)
        zero_ptr = torch.zeros((1,), device=device, dtype=torch.long)
        empty_bool = torch.empty((0,), device=device, dtype=torch.bool)
        return cls(
            edge_choice_edge_ids=empty_long,
            edge_choice_source_graph_nodes=empty_long,
            edge_choice_relation_ids=empty_long,
            edge_choice_target_graph_nodes=empty_long,
            edge_choice_next_component_counts=empty_long,
            edge_choice_question_similarity=empty_float,
            edge_choice_semantic_overlap=empty_float,
            edge_choice_action_new_bit_gain=empty_long,
            edge_choice_answer_candidate_counts=empty_long,
            edge_choice_target_answer_distance=empty_float,
            edge_choice_relation_choice_indices=empty_long,
            node_choice_graph_node_ids=empty_long,
            node_choice_candidate_counts=empty_long,
            node_choice_reachability_bits=empty_long,
            node_choice_is_anchor=empty_bool,
            relation_choice_relation_ids=empty_long,
            relation_choice_node_choice_indices=empty_long,
            relation_choice_num_edges=empty_long,
            relation_choice_max_new_bit_gain=empty_long,
            relation_choice_max_answer_candidate_counts=empty_long,
            relation_choice_max_semantic_overlap=empty_float,
            node_relation_ptr=zero_ptr,
            relation_edge_ptr=zero_ptr,
        )


@dataclass(frozen=True)
class StateActionSurfaceContext:
    flat_state_index: int
    current_components: int
    current_answer_candidate_count: int
    current_oracle_distance: float
    max_anchor_count: int
    full_mask: int
    state_num_edges: int
    linearized: LinearizedActionSurface


def _edge_question_similarity_or_zeros(
    *, batch: SubgraphBatch, candidate_edge_ids: torch.Tensor
) -> torch.Tensor:
    if batch.edge_question_similarity is None:
        return torch.zeros(
            (int(candidate_edge_ids.numel()),),
            device=candidate_edge_ids.device,
            dtype=torch.float32,
        )
    return batch.edge_question_similarity.index_select(0, candidate_edge_ids).to(
        dtype=torch.float32
    )


def collect_state_action_surface(
    *,
    batch: SubgraphBatch,
    rollout_batch: SubgraphRolloutBatch,
    flat_state_index: int,
    analysis: SubgraphAnalysis,
    max_steps: int,
    action_pruning: Mapping[str, Any] | None = None,
) -> StateActionSurfaceContext:
    device = batch.device
    flat_state_idx = int(flat_state_index)
    graph_idx = int(rollout_batch.graph_ids[flat_state_idx].item())
    graph = batch.graph(graph_idx)
    state = rollout_batch.states[flat_state_idx]
    current_components = int(analysis.anchor_component_count)
    answer_set = resolve_admissible_answer_set(
        batch=batch,
        graph_idx=graph_idx,
        analysis=analysis,
    )
    current_oracle_distance = float(
        oracle_distance(batch=batch, graph_idx=graph_idx, analysis=analysis)
    )
    full_mask = int(graph.anchor_full_mask)
    max_anchor_count = max(_bit_count(full_mask), 1)
    anchor_nodes = {int(node_id) for node_id in graph.anchor_abs_node_ids}
    reachable_entity_ids = {
        int(entity_id) for entity_id in analysis.entity_reachability_bits
    }
    per_node_top_k, per_state_top_k = resolve_action_pruning_limits(action_pruning)

    if int(state.num_edges) >= int(max_steps):
        return StateActionSurfaceContext(
            flat_state_index=flat_state_idx,
            current_components=current_components,
            current_answer_candidate_count=int(answer_set.count),
            current_oracle_distance=float(current_oracle_distance),
            max_anchor_count=max_anchor_count,
            full_mask=full_mask,
            state_num_edges=int(state.num_edges),
            linearized=LinearizedActionSurface.empty(device=device),
        )

    selected_edge_ids = {int(edge_id) for edge_id in state.edge_ids}
    candidate_edge_ids_list: list[int] = []
    candidate_source_graph_nodes_list: list[int] = []
    for graph_node_id in analysis.selected_node_ids:
        outgoing_edge_ids = graph.outgoing_edge_ids.get(int(graph_node_id), ())
        if per_node_top_k is not None:
            outgoing_edge_ids = outgoing_edge_ids[: int(per_node_top_k)]
        for edge_id in outgoing_edge_ids:
            if int(edge_id) in selected_edge_ids:
                continue
            candidate_edge_ids_list.append(int(edge_id))
            candidate_source_graph_nodes_list.append(int(graph_node_id))
    if not candidate_edge_ids_list:
        surface = LinearizedActionSurface.empty(device=device)
        return StateActionSurfaceContext(
            flat_state_index=flat_state_idx,
            current_components=current_components,
            current_answer_candidate_count=int(answer_set.count),
            current_oracle_distance=float(current_oracle_distance),
            max_anchor_count=max_anchor_count,
            full_mask=full_mask,
            state_num_edges=int(state.num_edges),
            linearized=surface,
        )

    candidate_edge_ids = torch.tensor(
        candidate_edge_ids_list, device=device, dtype=torch.long
    )
    candidate_source_graph_nodes = torch.tensor(
        candidate_source_graph_nodes_list,
        device=device,
        dtype=torch.long,
    )
    candidate_question_similarity = _edge_question_similarity_or_zeros(
        batch=batch,
        candidate_edge_ids=candidate_edge_ids,
    )

    if per_state_top_k is not None and len(candidate_edge_ids_list) > int(
        per_state_top_k
    ):
        if batch.edge_question_similarity is not None:
            ranking = torch.argsort(
                candidate_question_similarity, descending=True, stable=True
            )
            keep_tensor = ranking[: int(per_state_top_k)]
        else:
            keep_tensor = torch.argsort(candidate_edge_ids, stable=True)[
                : int(per_state_top_k)
            ]
        candidate_edge_ids = candidate_edge_ids.index_select(0, keep_tensor)
        candidate_source_graph_nodes = candidate_source_graph_nodes.index_select(
            0, keep_tensor
        )
        candidate_question_similarity = candidate_question_similarity.index_select(
            0, keep_tensor
        )
        candidate_edge_ids_list = [
            int(value) for value in candidate_edge_ids.detach().cpu().tolist()
        ]
        candidate_source_graph_nodes_list = [
            int(value) for value in candidate_source_graph_nodes.detach().cpu().tolist()
        ]

    candidate_target_graph_nodes = batch.topology.edge_index[1].index_select(
        0, candidate_edge_ids
    )
    candidate_relation_ids = batch.topology.edge_type.index_select(
        0, candidate_edge_ids
    )
    candidate_target_graph_nodes_list = [
        int(value) for value in candidate_target_graph_nodes.detach().cpu().tolist()
    ]
    candidate_relation_ids_list = [
        int(value) for value in candidate_relation_ids.detach().cpu().tolist()
    ]
    candidate_target_entity_ids_list = [
        int(value)
        for value in batch.node_entity_ids.index_select(0, candidate_target_graph_nodes)
        .detach()
        .cpu()
        .tolist()
    ]
    oracle_distance_map = graph.oracle_answer_distance
    candidate_source_bits_list = [
        int(analysis.reachability_bits.get(int(graph_node_id), 0))
        for graph_node_id in candidate_source_graph_nodes_list
    ]
    candidate_current_entity_bits_list = [
        int(analysis.entity_reachability_bits.get(int(entity_id), 0))
        for entity_id in candidate_target_entity_ids_list
    ]
    candidate_next_component_counts_list = [
        estimate_next_components(
            analysis=analysis,
            source_graph_node=int(source_graph_node),
            target_graph_node=int(target_graph_node),
        )
        for source_graph_node, target_graph_node in zip(
            candidate_source_graph_nodes_list,
            candidate_target_graph_nodes_list,
        )
    ]
    candidate_semantic_overlap_list = [
        1.0 if int(entity_id) in reachable_entity_ids else 0.0
        for entity_id in candidate_target_entity_ids_list
    ]
    answer_candidate_counts_list = [
        1
        if int(full_mask) > 0
        and (int(current_entity_bits) | int(source_bits)) == int(full_mask)
        else 0
        for source_bits, current_entity_bits in zip(
            candidate_source_bits_list,
            candidate_current_entity_bits_list,
        )
    ]
    candidate_new_bit_gain_list = [
        int(_bit_count(int(source_bits) & ~int(current_entity_bits)))
        for source_bits, current_entity_bits in zip(
            candidate_source_bits_list,
            candidate_current_entity_bits_list,
        )
    ]
    candidate_target_answer_distance_list = [
        float(oracle_distance_map.get(int(graph_node_id), -1))
        for graph_node_id in candidate_target_graph_nodes_list
    ]

    candidates_by_node: dict[int, list[int]] = {}
    for candidate_idx, graph_node_id in enumerate(candidate_source_graph_nodes_list):
        candidates_by_node.setdefault(int(graph_node_id), []).append(int(candidate_idx))

    ordered_candidate_indices: list[int] = []
    node_choice_graph_node_ids_list: list[int] = []
    node_choice_candidate_counts_list: list[int] = []
    node_choice_reachability_bits_list: list[int] = []
    node_choice_is_anchor_list: list[bool] = []
    relation_choice_relation_ids_list: list[int] = []
    relation_choice_node_choice_indices_list: list[int] = []
    relation_choice_num_edges_list: list[int] = []
    relation_choice_max_new_bit_gain_list: list[int] = []
    relation_choice_max_answer_candidate_counts_list: list[int] = []
    relation_choice_max_semantic_overlap_list: list[float] = []
    edge_choice_relation_choice_indices_list: list[int] = []
    node_relation_ptr = [0]
    relation_edge_ptr = [0]
    relation_choice_count = 0

    for node_choice_idx, graph_node_id in enumerate(sorted(candidates_by_node)):
        node_candidate_indices = candidates_by_node[int(graph_node_id)]
        node_choice_graph_node_ids_list.append(int(graph_node_id))
        node_choice_candidate_counts_list.append(int(len(node_candidate_indices)))
        node_choice_reachability_bits_list.append(
            int(analysis.reachability_bits.get(int(graph_node_id), 0))
        )
        node_choice_is_anchor_list.append(int(graph_node_id) in anchor_nodes)
        relation_groups: dict[int, list[int]] = {}
        for candidate_idx in node_candidate_indices:
            relation_groups.setdefault(
                int(candidate_relation_ids_list[int(candidate_idx)]), []
            ).append(int(candidate_idx))
        for relation_id in sorted(relation_groups):
            relation_candidate_indices = relation_groups[int(relation_id)]
            relation_choice_relation_ids_list.append(int(relation_id))
            relation_choice_node_choice_indices_list.append(int(node_choice_idx))
            relation_choice_num_edges_list.append(int(len(relation_candidate_indices)))
            relation_choice_max_new_bit_gain_list.append(
                max(
                    int(candidate_new_bit_gain_list[int(candidate_idx)])
                    for candidate_idx in relation_candidate_indices
                )
            )
            relation_choice_max_answer_candidate_counts_list.append(
                max(
                    int(answer_candidate_counts_list[int(candidate_idx)])
                    for candidate_idx in relation_candidate_indices
                )
            )
            relation_choice_max_semantic_overlap_list.append(
                max(
                    float(candidate_semantic_overlap_list[int(candidate_idx)])
                    for candidate_idx in relation_candidate_indices
                )
            )
            ordered_candidate_indices.extend(relation_candidate_indices)
            edge_choice_relation_choice_indices_list.extend(
                [int(relation_choice_count)] * int(len(relation_candidate_indices))
            )
            relation_choice_count += 1
            relation_edge_ptr.append(int(len(ordered_candidate_indices)))
        node_relation_ptr.append(int(relation_choice_count))

    order_tensor = torch.tensor(
        ordered_candidate_indices, device=device, dtype=torch.long
    )
    surface = LinearizedActionSurface(
        edge_choice_edge_ids=candidate_edge_ids.index_select(0, order_tensor),
        edge_choice_source_graph_nodes=candidate_source_graph_nodes.index_select(
            0, order_tensor
        ),
        edge_choice_relation_ids=candidate_relation_ids.index_select(0, order_tensor),
        edge_choice_target_graph_nodes=candidate_target_graph_nodes.index_select(
            0, order_tensor
        ),
        edge_choice_next_component_counts=torch.tensor(
            candidate_next_component_counts_list,
            device=device,
            dtype=torch.long,
        ).index_select(0, order_tensor),
        edge_choice_question_similarity=candidate_question_similarity.index_select(
            0, order_tensor
        ),
        edge_choice_semantic_overlap=torch.tensor(
            candidate_semantic_overlap_list,
            device=device,
            dtype=torch.float32,
        ).index_select(0, order_tensor),
        edge_choice_action_new_bit_gain=torch.tensor(
            candidate_new_bit_gain_list,
            device=device,
            dtype=torch.long,
        ).index_select(0, order_tensor),
        edge_choice_answer_candidate_counts=torch.tensor(
            answer_candidate_counts_list,
            device=device,
            dtype=torch.long,
        ).index_select(0, order_tensor),
        edge_choice_target_answer_distance=torch.tensor(
            candidate_target_answer_distance_list,
            device=device,
            dtype=torch.float32,
        ).index_select(0, order_tensor),
        edge_choice_relation_choice_indices=torch.tensor(
            edge_choice_relation_choice_indices_list,
            device=device,
            dtype=torch.long,
        ),
        node_choice_graph_node_ids=torch.tensor(
            node_choice_graph_node_ids_list,
            device=device,
            dtype=torch.long,
        ),
        node_choice_candidate_counts=torch.tensor(
            node_choice_candidate_counts_list,
            device=device,
            dtype=torch.long,
        ),
        node_choice_reachability_bits=torch.tensor(
            node_choice_reachability_bits_list,
            device=device,
            dtype=torch.long,
        ),
        node_choice_is_anchor=torch.tensor(
            node_choice_is_anchor_list,
            device=device,
            dtype=torch.bool,
        ),
        relation_choice_relation_ids=torch.tensor(
            relation_choice_relation_ids_list,
            device=device,
            dtype=torch.long,
        ),
        relation_choice_node_choice_indices=torch.tensor(
            relation_choice_node_choice_indices_list,
            device=device,
            dtype=torch.long,
        ),
        relation_choice_num_edges=torch.tensor(
            relation_choice_num_edges_list,
            device=device,
            dtype=torch.long,
        ),
        relation_choice_max_new_bit_gain=torch.tensor(
            relation_choice_max_new_bit_gain_list,
            device=device,
            dtype=torch.long,
        ),
        relation_choice_max_answer_candidate_counts=torch.tensor(
            relation_choice_max_answer_candidate_counts_list,
            device=device,
            dtype=torch.long,
        ),
        relation_choice_max_semantic_overlap=torch.tensor(
            relation_choice_max_semantic_overlap_list,
            device=device,
            dtype=torch.float32,
        ),
        node_relation_ptr=torch.tensor(
            node_relation_ptr, device=device, dtype=torch.long
        ),
        relation_edge_ptr=torch.tensor(
            relation_edge_ptr, device=device, dtype=torch.long
        ),
    )
    return StateActionSurfaceContext(
        flat_state_index=flat_state_idx,
        current_components=current_components,
        current_answer_candidate_count=int(answer_set.count),
        current_oracle_distance=float(current_oracle_distance),
        max_anchor_count=max_anchor_count,
        full_mask=full_mask,
        state_num_edges=int(state.num_edges),
        linearized=surface,
    )


__all__ = [
    "LinearizedActionSurface",
    "StateActionSurfaceContext",
    "collect_state_action_surface",
    "estimate_next_components",
    "resolve_action_pruning_limits",
]
