from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class CatalogLookup:
    entity_labels: Sequence[str]
    relation_labels: Sequence[str]

    def local_node_global_id(self, graph: object, local_node_id: int) -> int:
        ids = _graph_tensor(graph, name="node_entity_catalog_ids")
        return _local_id(ids, int(local_node_id), name="local_node_id")

    def local_edge_global_id(self, graph: object, local_edge_id: int) -> int:
        ids = _graph_tensor(graph, name="edge_relation_catalog_ids")
        return _local_id(ids, int(local_edge_id), name="local_edge_id")

    def local_node_label(self, graph: object, local_node_id: int) -> str:
        global_id = self.local_node_global_id(graph, local_node_id)
        return _label(self.entity_labels, global_id, name="entity_labels")

    def local_edge_label(self, graph: object, local_edge_id: int) -> str:
        global_id = self.local_edge_global_id(graph, local_edge_id)
        return _label(self.relation_labels, global_id, name="relation_labels")


def _graph_tensor(graph: object, *, name: str) -> torch.Tensor:
    value = getattr(graph, name, None)
    if not isinstance(value, torch.Tensor):
        raise AttributeError(f"graph must define tensor attribute {name!r}.")
    return value.to(dtype=torch.long).view(-1)


def _local_id(ids: torch.Tensor, index: int, *, name: str) -> int:
    if index < 0 or index >= int(ids.numel()):
        raise IndexError(f"{name} out of range: {index}")
    return int(ids[index].item())


def _label(labels: Sequence[str], index: int, *, name: str) -> str:
    if index < 0 or index >= len(labels):
        raise IndexError(f"{name} id out of range: {index}")
    return str(labels[index])


__all__ = ["CatalogLookup"]
