from __future__ import annotations

from typing import TYPE_CHECKING, Any
import torch
from torch_geometric.data.batch import Batch

from .fields import SampleFields


class RetrievalBatch(Batch):
    if TYPE_CHECKING:
        ptr: torch.Tensor
        batch: torch.Tensor
        edge_index: torch.Tensor
        num_nodes: int

        edge_relation_ids_global: torch.Tensor
        node_entity_ids_global: torch.Tensor
        question_emb: torch.Tensor

        is_anchor_mask: torch.Tensor
        is_target_mask: torch.Tensor
        anchor_signed_distance: torch.Tensor
        answer_entity_ids_global: torch.Tensor
        positive_edge_mask: torch.Tensor
        node_to_target_distance: torch.Tensor
        shortest_suffix_count: torch.Tensor
        bounded_suffix_count: torch.Tensor
        max_path_length: torch.Tensor

        node_tokens: torch.Tensor
        non_text_node_mask: torch.Tensor
        relation_tokens: torch.Tensor
        is_cvt: torch.Tensor
        heuristic_log_v: torch.Tensor

        node_ptr: torch.Tensor
        edge_batch: torch.Tensor
        edge_ptr: torch.Tensor

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == SampleFields.BOUNDED_SUFFIX_COUNT:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)  # type: ignore[attr-defined]

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key in SampleFields.NO_INCREMENT_KEYS:
            return 0
        return super().__inc__(key, value, *args, **kwargs) # type: ignore[attr-defined]

    @property
    def num_nodes_total(self) -> int:
        return int(self.num_nodes)
