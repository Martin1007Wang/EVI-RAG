from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class SplitFilter:
    require_answer_in_graph: bool
    require_reachable_answer: bool


@dataclass(frozen=True)
class RawSample:
    dataset: str
    split: str
    question_id: str
    question: str
    graph: tuple[tuple[str, str, str], ...]
    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]


@dataclass(frozen=True)
class PreparedSample:
    dataset: str
    split: str
    question_id: str
    question: str

    edge_index: torch.Tensor  # [2, num_edges]
    num_nodes: int
    num_edges: int

    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]  # full in-graph answers

    anchor_node_ids: torch.Tensor  # [num_anchors]
    target_node_ids: torch.Tensor  # [num_answers_in_graph]
    reachable_target_node_ids: torch.Tensor  # [num_reachable_targets]

    node_entity_catalog_ids: torch.Tensor  # [num_nodes]
    edge_relation_catalog_ids: torch.Tensor  # [num_edges]

    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor
    node_target_distance: torch.Tensor  # [num_nodes]
    target_node_distances_flat: torch.Tensor

    target_shortest_path_count_flat: torch.Tensor
    target_shortest_path_edge_mask_flat: torch.Tensor
