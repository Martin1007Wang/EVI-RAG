from __future__ import annotations

import torch

from .observation import GraphObservation, GroupedLocalNodeIndex
from .protocol import GraphBatchProtocol
from .topology import CsrAdjacency, GraphTopology


def _require_tensor(*, value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}.")
    return value


def _require_1d_long(*, value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value=value, name=name)
    if tensor.dtype != torch.long or tensor.dim() != 1:
        raise ValueError(
            f"{name} must be 1D torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_2d_float(*, value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value=value, name=name)
    if tensor.dim() != 2 or not torch.is_floating_point(tensor):
        raise ValueError(
            f"{name} must be 2D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_3d_float(*, value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value=value, name=name)
    if tensor.dim() != 3 or not torch.is_floating_point(tensor):
        raise ValueError(
            f"{name} must be 3D floating point, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_2d_bool(*, value: object, name: str) -> torch.Tensor:
    tensor = _require_tensor(value=value, name=name)
    if tensor.dtype != torch.bool or tensor.dim() != 2:
        raise ValueError(
            f"{name} must be 2D bool, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _require_edge_index(*, value: object) -> torch.Tensor:
    tensor = _require_tensor(value=value, name="edge_index")
    if tensor.dtype != torch.long or tensor.dim() != 2 or int(tensor.size(0)) != 2:
        raise ValueError(
            f"edge_index must be [2, E] torch.long, got {tensor.dtype} {tuple(tensor.shape)}."
        )
    return tensor


def _validate_graph_batch_protocol(
    batch: GraphBatchProtocol,
) -> dict[str, torch.Tensor]:
    num_graphs = int(batch.num_graphs)
    if num_graphs < 1:
        raise ValueError("graph batch num_graphs must be >= 1.")

    tensors = {
        "node_ptr": _require_1d_long(value=batch.node_ptr, name="node_ptr"),
        "edge_index": _require_edge_index(value=batch.edge_index),
        "edge_rel_global": _require_1d_long(
            value=batch.edge_rel_global,
            name="edge_rel_global",
        ),
        "node_embeddings": _require_2d_float(
            value=batch.node_embeddings,
            name="node_embeddings",
        ),
        "edge_embeddings": _require_2d_float(
            value=batch.edge_embeddings,
            name="edge_embeddings",
        ),
        "question_emb": _require_2d_float(
            value=batch.question_emb,
            name="question_emb",
        ),
        "question_ctx": _require_3d_float(
            value=batch.question_ctx,
            name="question_ctx",
        ),
        "question_ctx_mask": _require_2d_bool(
            value=batch.question_ctx_mask,
            name="question_ctx_mask",
        ),
        "q_local_indices": _require_1d_long(
            value=batch.q_local_indices,
            name="q_local_indices",
        ),
        "q_ptr": _require_1d_long(
            value=batch.q_ptr,
            name="q_ptr",
        ),
        "node_global_ids": _require_1d_long(
            value=batch.node_global_ids,
            name="node_global_ids",
        ),
    }
    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        raise ValueError(
            f"graph batch tensors must share one device, got {sorted(str(device) for device in devices)}."
        )

    node_ptr = tensors["node_ptr"]
    if int(node_ptr.numel()) != num_graphs + 1:
        raise ValueError("node_ptr must have length num_graphs + 1 in graph batch.")
    if int(node_ptr[0].item()) != 0:
        raise ValueError("node_ptr must start at 0 in graph batch.")
    if bool((node_ptr[1:] < node_ptr[:-1]).any().item()):
        raise ValueError("node_ptr must be non-decreasing in graph batch.")
    num_nodes = int(node_ptr[-1].item())
    if int(tensors["node_embeddings"].size(0)) != num_nodes:
        raise ValueError(
            "node_embeddings row count mismatch with node_ptr in graph batch."
        )
    if int(tensors["node_global_ids"].numel()) != num_nodes:
        raise ValueError(
            "node_global_ids length mismatch with node_ptr in graph batch."
        )

    edge_index = tensors["edge_index"]
    num_edges = int(edge_index.size(1))
    if int(tensors["edge_rel_global"].numel()) != num_edges:
        raise ValueError(
            "edge_rel_global length mismatch with edge_index column count in graph batch."
        )
    if int(tensors["edge_embeddings"].size(0)) != num_edges:
        raise ValueError(
            "edge_embeddings row count mismatch with edge_index column count in graph batch."
        )
    if num_edges > 0:
        if bool((edge_index < 0).any().item()) or bool(
            (edge_index >= num_nodes).any().item()
        ):
            raise ValueError(
                "edge_index contains out-of-range node ids in graph batch."
            )
        graph_end_offsets = node_ptr[1:]
        source_graph = torch.searchsorted(graph_end_offsets, edge_index[0], right=True)
        target_graph = torch.searchsorted(graph_end_offsets, edge_index[1], right=True)
        if bool((source_graph != target_graph).any().item()):
            raise ValueError(
                "edge_index crosses graph boundaries in graph batch construction."
            )

    if int(tensors["question_emb"].size(0)) != num_graphs:
        raise ValueError("question_emb batch mismatch with num_graphs in graph batch.")
    question_ctx = tensors["question_ctx"]
    if int(question_ctx.size(0)) != num_graphs:
        raise ValueError("question_ctx batch mismatch with num_graphs in graph batch.")
    question_ctx_mask = tensors["question_ctx_mask"]
    if tuple(question_ctx_mask.shape) != tuple(question_ctx.shape[:2]):
        raise ValueError(
            "question_ctx_mask shape mismatch with question_ctx in graph batch."
        )
    if bool((~question_ctx_mask).all(dim=1).any().item()):
        raise ValueError(
            "question_ctx_mask contains rows without valid tokens in graph batch."
        )

    GroupedLocalNodeIndex.from_group_ptr(
        local_indices=tensors["q_local_indices"],
        group_ptr=tensors["q_ptr"],
        num_groups=num_graphs,
        field_name="q_local_indices",
    )
    return tensors


def _build_csr_with_edge_ids(
    *,
    edge_index: torch.Tensor,
    num_nodes_total: int,
) -> CsrAdjacency:
    device = edge_index.device
    edge_ids = torch.arange(int(edge_index.size(1)), device=device, dtype=torch.long)
    if int(edge_ids.numel()) == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        crow = torch.zeros((num_nodes_total + 1,), device=device, dtype=torch.long)
        return CsrAdjacency(crow=crow, col=empty, edge_ids=empty)
    heads = edge_index[0]
    order = torch.argsort(heads)
    heads_sorted = heads.index_select(0, order)
    tails_sorted = edge_index[1].index_select(0, order)
    edge_ids_sorted = edge_ids.index_select(0, order)
    row_ids = torch.arange(num_nodes_total + 1, device=device, dtype=torch.long)
    crow = torch.searchsorted(heads_sorted, row_ids, right=False)
    return CsrAdjacency(crow=crow, col=tails_sorted, edge_ids=edge_ids_sorted)


def _build_relation_table(
    *,
    edge_rel_global: torch.Tensor,
    edge_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(edge_rel_global.numel()) == 0:
        return edge_embeddings.new_empty(
            (0, int(edge_embeddings.size(-1)))
        ), edge_rel_global.new_empty((0,))
    _, edge_relations = torch.unique(edge_rel_global, sorted=True, return_inverse=True)
    num_rel = int(edge_relations.max().item()) + 1
    first_occ = torch.full(
        (num_rel,),
        fill_value=int(edge_relations.numel()),
        device=edge_relations.device,
        dtype=torch.long,
    )
    edge_ids = torch.arange(
        int(edge_relations.numel()),
        device=edge_relations.device,
        dtype=torch.long,
    )
    first_occ.scatter_reduce_(
        0,
        edge_relations,
        edge_ids,
        reduce="amin",
        include_self=True,
    )
    relation_embeddings = edge_embeddings.index_select(0, first_occ)
    return relation_embeddings, edge_relations


def build_graph_batch(
    batch: GraphBatchProtocol,
) -> tuple[GraphTopology, GraphObservation]:
    tensors = _validate_graph_batch_protocol(batch)
    relation_embeddings, edge_relations = _build_relation_table(
        edge_rel_global=tensors["edge_rel_global"],
        edge_embeddings=tensors["edge_embeddings"],
    )
    num_nodes = int(tensors["node_ptr"][-1].item())
    q_local_indices = GroupedLocalNodeIndex.from_group_ptr(
        local_indices=tensors["q_local_indices"],
        group_ptr=tensors["q_ptr"],
        num_groups=batch.num_graphs,
        field_name="q_local_indices",
    )
    sample_ids = tuple(str(sample_id) for sample_id in batch.sample_ids)
    if len(sample_ids) != int(batch.num_graphs):
        raise ValueError(
            "sample_ids length mismatch with num_graphs in environment builder: "
            f"sample_ids={len(sample_ids)}, num_graphs={int(batch.num_graphs)}."
        )

    topology = GraphTopology(
        num_graphs=batch.num_graphs,
        num_nodes=num_nodes,
        edge_index=tensors["edge_index"],
        edge_type=edge_relations,
        _graph_node_offsets=tensors["node_ptr"],
        adjacency=_build_csr_with_edge_ids(
            edge_index=tensors["edge_index"],
            num_nodes_total=num_nodes,
        ),
    )
    observation = GraphObservation(
        node_features=tensors["node_embeddings"],
        relation_features=relation_embeddings,
        node_ids=tensors["node_global_ids"],
        question_embedding=tensors["question_emb"],
        question_context=tensors["question_ctx"],
        question_valid_mask=tensors["question_ctx_mask"],
        q_local_indices=q_local_indices,
        sample_ids=sample_ids,
    )

    topology.validate()
    observation.validate(topology=topology)
    topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
        validate_grouping=False,
    )
    return topology, observation


__all__ = ["build_graph_batch"]
