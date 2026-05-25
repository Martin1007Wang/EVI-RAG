from __future__ import annotations
from dataclasses import dataclass
import torch
from src.data.schema.fields import SampleFields


StorageRecord = dict[str, torch.Tensor]


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

    # Each edge column is one cleaned triple-id edge, not one endpoint pair.
    # Relation-distinct parallel triples share endpoints but keep distinct ids.
    edge_index: torch.Tensor  # [2, num_edges]
    num_nodes: int
    num_edges: int

    question_entities: tuple[str, ...]
    answer_entities: tuple[str, ...]  # filtered in-graph answer entities

    anchor_node_ids: torch.Tensor  # [num_anchors]
    target_node_ids: torch.Tensor  # [num_answers_in_graph]
    reachable_target_node_ids: torch.Tensor  # [num_reachable_targets]

    node_entity_catalog_ids: torch.Tensor  # [num_nodes]
    edge_relation_catalog_ids: torch.Tensor  # [num_edges], aligned with edge_index columns

    anchor_node_forward_distances_flat: torch.Tensor  # [num_nodes]
    anchor_node_backward_distances_flat: torch.Tensor  # [num_nodes]
    node_target_distance: torch.Tensor  # [num_nodes]
    node_target_distances_flat: torch.Tensor  # [T * num_nodes], T = num_reachable_targets

    node_target_shortest_path_count_flat: torch.Tensor  # [T * num_nodes]
    # Runtime-only dense edge labels at triple-id level: [target_idx, edge_id].
    # Materialized artifacts persist sparse nonzero edge counts instead.
    node_target_shortest_path_edge_mask_flat: torch.Tensor  # [T * num_edges]
    node_target_shortest_path_edge_count_flat: torch.Tensor  # [T * num_edges]
    replay_trajectory_edge_ids: torch.Tensor  # [sum replay trajectory lengths]
    replay_trajectory_lengths: torch.Tensor  # [num replay trajectories]

    def storage_key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.question_id}"

    def to_storage_record(self) -> StorageRecord:
        edge_count_indices, edge_count_values = _sparse_nonzero_counts(
            self.node_target_shortest_path_edge_count_flat
        )
        sample_id = self.storage_key().encode("utf-8")
        return {
            SampleFields.SAMPLE_ID: torch.tensor(list(sample_id), dtype=torch.uint8),
            SampleFields.EDGE_INDEX: self.edge_index.long().contiguous(),
            SampleFields.NODE_ENTITY_CATALOG_IDS: self.node_entity_catalog_ids.long().contiguous(),
            SampleFields.EDGE_RELATION_CATALOG_IDS: self.edge_relation_catalog_ids.long().contiguous(),
            SampleFields.NUM_NODES: torch.as_tensor(self.num_nodes, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.as_tensor(self.num_edges, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: self.anchor_node_ids.long().contiguous(),
            SampleFields.TARGET_NODE_IDS: self.target_node_ids.long().contiguous(),
            SampleFields.REACHABLE_TARGET_NODE_IDS: self.reachable_target_node_ids.long().contiguous(),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: self.anchor_node_forward_distances_flat.long().contiguous(),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: self.anchor_node_backward_distances_flat.long().contiguous(),
            SampleFields.NODE_TARGET_DISTANCE: self.node_target_distance.long().contiguous(),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: self.node_target_distances_flat.long().contiguous(),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: self.node_target_shortest_path_count_flat.float().contiguous(),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: edge_count_indices,
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: edge_count_values,
            SampleFields.REPLAY_TRAJECTORY_EDGE_IDS: self.replay_trajectory_edge_ids.long().contiguous(),
            SampleFields.REPLAY_TRAJECTORY_LENGTHS: self.replay_trajectory_lengths.long().contiguous(),
        }


def _sparse_nonzero_counts(counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dense = counts.to(dtype=torch.float32, device="cpu").contiguous().view(-1)
    indices = dense.nonzero(as_tuple=False).view(-1).long().contiguous()
    if indices.numel() == 0:
        empty_indices = torch.empty((0,), dtype=torch.long)
        empty_values = torch.empty((0,), dtype=torch.float32)
        return empty_indices, empty_values
    values = dense.index_select(0, indices).to(dtype=torch.float32).contiguous()
    return indices.contiguous(), values
