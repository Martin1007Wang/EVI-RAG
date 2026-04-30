from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from .fields import SampleFields


_NODE_ID_KEYS = frozenset(
    {
        SampleFields.ANCHOR_NODE_IDS,
        SampleFields.TARGET_NODE_IDS,
        SampleFields.REACHABLE_TARGET_NODE_IDS,
    }
)


def _num_nodes(data: Any) -> int:
    num_nodes = data.num_nodes
    if num_nodes is None:
        raise ValueError("num_nodes is required for batching node-index fields")
    return int(num_nodes)


class RetrievalData(Data):
    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in _NODE_ID_KEYS:
            return _num_nodes(self)

        if key in SampleFields.NO_INCREMENT_KEYS:
            return 0

        return super().__inc__(key, value, *args, **kwargs)


class RetrievalBatch(Batch):
    ptr: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int

    node_entity_catalog_ids: torch.Tensor
    edge_relation_catalog_ids: torch.Tensor

    question_emb: torch.Tensor

    anchor_node_ids: torch.Tensor
    target_node_ids: torch.Tensor
    reachable_target_node_ids: torch.Tensor

    anchor_node_forward_distances_flat: torch.Tensor
    anchor_node_backward_distances_flat: torch.Tensor

    node_target_distance: torch.Tensor
    target_node_distances_flat: torch.Tensor
    target_shortest_path_count_flat: torch.Tensor
    target_shortest_path_edge_mask_flat: torch.Tensor

    node_tokens: torch.Tensor
    non_text_node_mask: torch.Tensor
    relation_tokens: torch.Tensor
    is_non_text_entity: torch.Tensor
    heuristic_log_v: torch.Tensor

    node_ptr: torch.Tensor
    edge_batch: torch.Tensor
    edge_ptr: torch.Tensor

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        return super().__cat_dim__(key, value, *args, **kwargs)  # type: ignore[attr-defined]

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in _NODE_ID_KEYS:
            return self.num_nodes_total

        if key in SampleFields.NO_INCREMENT_KEYS:
            return 0

        return super().__inc__(key, value, *args, **kwargs)  # type: ignore[attr-defined]

    @property
    def num_nodes_total(self) -> int:
        return _num_nodes(self)

    @property
    def num_edges_total(self) -> int:
        if not hasattr(self, "edge_index"):
            raise ValueError("edge_index is required to infer num_edges_total.")

        edge_index = self.edge_index
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
            )

        return int(edge_index.size(1))

    @property
    def num_graphs_total(self) -> int:
        if hasattr(self, "ptr") and self.ptr is not None:
            return int(self.ptr.numel() - 1)

        if hasattr(self, "batch") and self.batch is not None and self.batch.numel() > 0:
            return int(self.batch.max().item()) + 1

        if hasattr(self, "num_graphs"):
            return int(self.num_graphs)

        raise ValueError(
            "Cannot infer number of graphs from ptr, batch, or num_graphs."
        )
