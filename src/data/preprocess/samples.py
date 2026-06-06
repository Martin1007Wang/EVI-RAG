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
    reachable_target_max_distance: int

    node_entity_catalog_ids: torch.Tensor  # [num_nodes]
    edge_relation_catalog_ids: torch.Tensor  # [num_edges], aligned with edge_index columns

    node_target_distance: torch.Tensor  # [num_nodes]
    edge_on_shortest_path: torch.Tensor  # [num_edges]
    replay_bank_edge_ids: torch.Tensor  # [variants, slots, max_edges]
    replay_bank_edge_count: torch.Tensor  # [variants, slots]
    replay_bank_priority: torch.Tensor  # [variants, slots]

    def storage_key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.question_id}"

    def to_storage_record(self) -> StorageRecord:
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
            SampleFields.REACHABLE_TARGET_MAX_DISTANCE: torch.as_tensor(
                self.reachable_target_max_distance,
                dtype=torch.long,
            ),
            SampleFields.NODE_TARGET_DISTANCE: self.node_target_distance.long().contiguous(),
            SampleFields.EDGE_ON_SHORTEST_PATH: self.edge_on_shortest_path.bool().contiguous(),
            SampleFields.REPLAY_BANK_EDGE_IDS: self.replay_bank_edge_ids.long().contiguous(),
            SampleFields.REPLAY_BANK_EDGE_COUNT: self.replay_bank_edge_count.long().contiguous(),
            SampleFields.REPLAY_BANK_PRIORITY: self.replay_bank_priority.float().contiguous(),
        }
