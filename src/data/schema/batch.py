from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch

from .fields import SampleFields

"""
Coordinate contract
-------------------

PyG batching operates on sample-local tensors and produces mini-batch physical
coordinates.

After collation:
- edge_index contains batch-physical node ids.
- anchor_node_ids / target_node_ids / reachable_target_node_ids contain
  batch-physical node ids.
- node_entity_catalog_ids / edge_relation_catalog_ids remain global catalog ids.
- weak_replay_edge_ids remain graph-local edge ids and must be converted with
  their follow_batch graph ids before use.

Do not add edge_batch as a required truth source. Infer edge_to_graph from:

    node_to_graph[edge_index[0]]

inside GraphContext.
"""


_NODE_INDEX_KEYS = frozenset(
    {
        SampleFields.ANCHOR_NODE_IDS,
        SampleFields.TARGET_NODE_IDS,
        SampleFields.REACHABLE_TARGET_NODE_IDS,
        SampleFields.WITNESS_PATH_TARGET_NODE_IDS,
    }
)

_GLOBAL_ID_KEYS = frozenset(
    {
        SampleFields.NODE_ENTITY_CATALOG_IDS,
        SampleFields.EDGE_RELATION_CATALOG_IDS,
    }
)

_COUNT_KEYS = frozenset(
    {
        SampleFields.NUM_NODES,
        SampleFields.NUM_EDGES,
    }
)

_FLAT_SUPERVISION_KEYS = frozenset(
    {
        SampleFields.NODE_TARGET_DISTANCE,
    }
)

_LOCAL_WEAK_REPLAY_KEYS = frozenset(
    {
        SampleFields.WEAK_REPLAY_EDGE_IDS,
        SampleFields.WEAK_REPLAY_EDGE_WEIGHT,
        SampleFields.WITNESS_PATH_EDGE_IDS,
    }
)

_LOCAL_PATH_ID_KEYS = frozenset(
    {
        SampleFields.WITNESS_PATH_EDGE_PATH_IDS,
    }
)

_RUNTIME_ONLY_NO_INCREMENT_KEYS = frozenset(
    {
        "node_target_shortest_path_edge_mask_flat",
    }
)

_NO_INCREMENT_KEYS = _GLOBAL_ID_KEYS | _COUNT_KEYS | _FLAT_SUPERVISION_KEYS | _LOCAL_WEAK_REPLAY_KEYS | _RUNTIME_ONLY_NO_INCREMENT_KEYS


def _num_nodes(data: Any) -> int:
    num_nodes = data.num_nodes
    if num_nodes is None:
        raise ValueError("num_nodes is required for batching node-index fields.")
    return int(num_nodes)


def _num_edges(data: Any) -> int:
    if hasattr(data, "edge_index"):
        edge_index = data.edge_index
        if edge_index is not None:
            if edge_index.ndim != 2 or edge_index.size(0) != 2:
                raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
            return int(edge_index.size(1))

    if hasattr(data, "num_edges") and data.num_edges is not None:
        return int(data.num_edges)

    raise ValueError("edge_index or num_edges is required to infer edge count.")


def _num_witness_paths(data: Any) -> int:
    if hasattr(data, SampleFields.WITNESS_PATH_TARGET_NODE_IDS):
        value = getattr(data, SampleFields.WITNESS_PATH_TARGET_NODE_IDS)
        if value is not None:
            return int(torch.as_tensor(value).numel())
    return 0


def _retrieval_increment(data: Any, key: str) -> int | None:
    """
    Return a custom PyG batching increment for known retrieval fields.

    None means: delegate to PyG default behavior.
    """

    if key in _NODE_INDEX_KEYS:
        return _num_nodes(data)

    if key in _LOCAL_PATH_ID_KEYS:
        return _num_witness_paths(data)

    if key in _NO_INCREMENT_KEYS:
        return 0

    return None


class RetrievalData(Data):
    """
    Single-sample retrieval graph.

    Local-coordinate fields before batching:
    - edge_index uses sample-local node ids.
    - anchor_node_ids / target_node_ids / reachable_target_node_ids use
      sample-local node ids.
    - weak_replay_edge_ids uses sample-local edge ids and is intentionally not
      incremented by PyG; use weak_replay_edge_ids_batch for graph ids.
    - witness_path_edge_ids uses sample-local edge ids and is intentionally not
      incremented by PyG; use witness_path_edge_ids_batch for graph ids.
    - witness_path_edge_path_ids uses graph-local path ids and is incremented by
      the number of witness paths in the sample.

    Global-id fields:
    - node_entity_catalog_ids
    - edge_relation_catalog_ids
    """

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        increment = _retrieval_increment(self, key)
        if increment is not None:
            return increment

        return super().__inc__(key, value, *args, **kwargs)


class RetrievalBatch(Batch):
    """
    PyG batch contract for retrieval samples.

    Required physical-coordinate fields after batching:
    - edge_index: [2, E], batch-physical node ids.
    - batch: [N], node id -> graph id.
    - ptr: [G + 1], graph id -> node range.
    - anchor_node_ids / target_node_ids / reachable_target_node_ids:
      batch-physical node ids.

    Required global-id feature fields:
    - node_entity_catalog_ids
    - edge_relation_catalog_ids

    Required supervision fields:
    - target / reachable target fields
    - node-target shortest-path materialized tensors

    Weak replay edge labels remain graph-local and are converted explicitly by
    TargetContext before loss code uses them.
    """

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

    node_target_distance: torch.Tensor

    weak_replay_edge_ids: torch.Tensor
    weak_replay_edge_weight: torch.Tensor
    witness_path_edge_ids: torch.Tensor
    witness_path_edge_path_ids: torch.Tensor
    witness_path_target_node_ids: torch.Tensor

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        return super().__cat_dim__(key, value, *args, **kwargs)  # type: ignore[attr-defined]

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        increment = _retrieval_increment(self, key)
        if increment is not None:
            return increment

        return super().__inc__(key, value, *args, **kwargs)  # type: ignore[attr-defined]

    @property
    def num_nodes_total(self) -> int:
        return _num_nodes(self)

    @property
    def num_edges_total(self) -> int:
        return _num_edges(self)

    @property
    def num_graphs_total(self) -> int:
        if hasattr(self, "ptr") and self.ptr is not None:
            return int(self.ptr.numel() - 1)

        if hasattr(self, "batch") and self.batch is not None and self.batch.numel() > 0:
            return int(self.batch.max().item()) + 1

        if hasattr(self, "num_graphs"):
            return int(self.num_graphs)

        raise ValueError("Cannot infer number of graphs from ptr, batch, or num_graphs.")

    @property
    def edge_graph_ids(self) -> torch.Tensor:
        """
        Derived edge -> graph mapping.

        This is not a stored truth source. It is equivalent to the edge_to_graph
        used by GraphContext:

            edge_graph_ids[e] = batch[edge_index[0, e]]

        Assumption:
        - no cross-graph edges.
        """

        return self.batch.index_select(0, self.edge_index[0])


def validate_retrieval_batch(
    batch: RetrievalBatch,
    *,
    check_cross_graph_edges: bool = True,
    check_edge_batch_if_present: bool = True,
) -> None:
    """
    Debug-only contract validation.

    Do not call this in the hot training loop unless you explicitly accept the
    overhead.
    """

    if batch.edge_index.ndim != 2 or batch.edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(batch.edge_index.shape)}.")

    if batch.edge_index.dtype != torch.long:
        raise TypeError(f"edge_index must have dtype torch.long, got {batch.edge_index.dtype}.")

    if batch.batch.ndim != 1:
        raise ValueError(f"batch must have shape [N], got {tuple(batch.batch.shape)}.")

    if batch.batch.dtype != torch.long:
        raise TypeError(f"batch must have dtype torch.long, got {batch.batch.dtype}.")

    num_nodes = batch.num_nodes_total
    num_edges = batch.num_edges_total

    if batch.batch.numel() != num_nodes:
        raise ValueError(f"batch has length {batch.batch.numel()}, but num_nodes_total is {num_nodes}.")

    if num_edges > 0:
        min_node_id = int(batch.edge_index.min().item())
        max_node_id = int(batch.edge_index.max().item())
        if min_node_id < 0 or max_node_id >= num_nodes:
            raise ValueError(
                "edge_index contains node ids outside the physical batch range: " f"min={min_node_id}, max={max_node_id}, num_nodes={num_nodes}."
            )

    _validate_node_id_tensor(
        batch.anchor_node_ids,
        name="anchor_node_ids",
        num_nodes=num_nodes,
    )
    _validate_node_id_tensor(
        batch.target_node_ids,
        name="target_node_ids",
        num_nodes=num_nodes,
    )
    _validate_node_id_tensor(
        batch.reachable_target_node_ids,
        name="reachable_target_node_ids",
        num_nodes=num_nodes,
    )

    if check_cross_graph_edges and num_edges > 0:
        src_graph = batch.batch.index_select(0, batch.edge_index[0])
        dst_graph = batch.batch.index_select(0, batch.edge_index[1])
        if not torch.equal(src_graph, dst_graph):
            mismatch = src_graph.ne(dst_graph).nonzero(as_tuple=False).flatten()
            first = int(mismatch[0].item())
            raise ValueError(
                "Cross-graph edges are not allowed. "
                f"First mismatch edge_id={first}, "
                f"src={int(batch.edge_index[0, first].item())}, "
                f"dst={int(batch.edge_index[1, first].item())}, "
                f"src_graph={int(src_graph[first].item())}, "
                f"dst_graph={int(dst_graph[first].item())}."
            )

    if check_edge_batch_if_present and hasattr(batch, "edge_batch"):
        edge_batch = getattr(batch, "edge_batch")
        if edge_batch is not None:
            if edge_batch.ndim != 1:
                raise ValueError(f"edge_batch must have shape [E], got {tuple(edge_batch.shape)}.")
            if int(edge_batch.numel()) != num_edges:
                raise ValueError(f"edge_batch has length {edge_batch.numel()}, " f"but num_edges_total is {num_edges}.")

            inferred = batch.edge_graph_ids
            edge_batch = edge_batch.to(device=inferred.device, dtype=torch.long)
            if not torch.equal(edge_batch, inferred):
                raise ValueError("edge_batch disagrees with batch[edge_index[0]]. " "Do not keep two inconsistent edge->graph truth sources.")


def _validate_node_id_tensor(
    value: torch.Tensor,
    *,
    name: str,
    num_nodes: int,
) -> None:
    if value.dtype != torch.long:
        raise TypeError(f"{name} must have dtype torch.long, got {value.dtype}.")

    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [K], got {tuple(value.shape)}.")

    if int(value.numel()) == 0:
        return

    min_id = int(value.min().item())
    max_id = int(value.max().item())

    if min_id < 0 or max_id >= int(num_nodes):
        raise ValueError(f"{name} contains node ids outside the physical batch range: " f"min={min_id}, max={max_id}, num_nodes={num_nodes}.")


__all__ = [
    "RetrievalData",
    "RetrievalBatch",
    "validate_retrieval_batch",
]
