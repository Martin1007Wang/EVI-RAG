from __future__ import annotations

import torch

from .batch import TrajectoryBatch
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


def _resolve_node_entity_ids(*, batch: GraphBatchProtocol) -> torch.Tensor:
    return _require_1d_long(value=batch.node_entity_ids, name="node_entity_ids")


def _validate_graph_batch_protocol(
    batch: GraphBatchProtocol,
) -> dict[str, torch.Tensor | None]:
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
        "anchor_local_indices": _require_1d_long(
            value=batch.anchor_local_indices,
            name="anchor_local_indices",
        ),
        "anchor_ptr": _require_1d_long(
            value=batch.anchor_ptr,
            name="anchor_ptr",
        ),
        "node_entity_ids": _resolve_node_entity_ids(batch=batch),
    }
    edge_embeddings = getattr(batch, "edge_embeddings", None)
    if edge_embeddings is not None:
        edge_embeddings = _require_2d_float(
            value=edge_embeddings,
            name="edge_embeddings",
        )
    relation_embeddings = getattr(batch, "relation_embeddings", None)
    if relation_embeddings is not None:
        relation_embeddings = _require_2d_float(
            value=relation_embeddings,
            name="relation_embeddings",
        )
    edge_rel_local = getattr(batch, "edge_rel_local", None)
    if edge_rel_local is not None:
        edge_rel_local = _require_1d_long(
            value=edge_rel_local,
            name="edge_rel_local",
        )
    tensors["edge_embeddings"] = edge_embeddings
    tensors["relation_embeddings"] = relation_embeddings
    tensors["edge_rel_local"] = edge_rel_local
    devices = {tensor.device for tensor in tensors.values() if tensor is not None}
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
    if int(tensors["node_entity_ids"].numel()) != num_nodes:
        raise ValueError(
            "node_entity_ids length mismatch with node_ptr in graph batch."
        )

    edge_index = tensors["edge_index"]
    assert edge_index is not None
    num_edges = int(edge_index.size(1))
    if int(tensors["edge_rel_global"].numel()) != num_edges:
        raise ValueError(
            "edge_rel_global length mismatch with edge_index column count in graph batch."
        )
    has_relation_table = relation_embeddings is not None or edge_rel_local is not None
    if has_relation_table:
        if relation_embeddings is None or edge_rel_local is None:
            raise ValueError(
                "graph batch must provide relation_embeddings and edge_rel_local together."
            )
        if int(edge_rel_local.numel()) != num_edges:
            raise ValueError(
                "edge_rel_local length mismatch with edge_index column count in graph batch."
            )
        if int(edge_rel_local.numel()) > 0 and bool(
            (edge_rel_local >= int(relation_embeddings.size(0))).any().item()
            or (edge_rel_local < 0).any().item()
        ):
            raise ValueError(
                "edge_rel_local contains out-of-range relation table indices in graph batch."
            )
    elif edge_embeddings is not None:
        if int(edge_embeddings.size(0)) != num_edges:
            raise ValueError(
                "edge_embeddings row count mismatch with edge_index column count in graph batch."
            )
    else:
        raise ValueError(
            "graph batch must provide either edge_embeddings or "
            "(relation_embeddings, edge_rel_local)."
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
    assert question_ctx is not None
    if int(question_ctx.size(0)) != num_graphs:
        raise ValueError("question_ctx batch mismatch with num_graphs in graph batch.")
    question_ctx_mask = tensors["question_ctx_mask"]
    assert question_ctx_mask is not None
    if tuple(question_ctx_mask.shape) != tuple(question_ctx.shape[:2]):
        raise ValueError(
            "question_ctx_mask shape mismatch with question_ctx in graph batch."
        )
    if bool((~question_ctx_mask).all(dim=1).any().item()):
        raise ValueError(
            "question_ctx_mask contains rows without valid tokens in graph batch."
        )

    GroupedLocalNodeIndex.from_group_ptr(
        local_indices=tensors["anchor_local_indices"],
        group_ptr=tensors["anchor_ptr"],
        num_groups=num_graphs,
        field_name="anchor_local_indices",
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


def _compact_relation_table(
    *,
    relation_embeddings: torch.Tensor,
    edge_rel_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(edge_rel_local.numel()) == 0:
        return relation_embeddings.new_empty(
            (0, int(relation_embeddings.size(-1)))
        ), edge_rel_local.new_empty((0,))
    used_local_ids, compact_edge_rel_local = torch.unique(
        edge_rel_local, sorted=True, return_inverse=True
    )
    return (
        relation_embeddings.index_select(0, used_local_ids),
        compact_edge_rel_local,
    )


def _resolve_relation_table(
    *,
    edge_rel_global: torch.Tensor,
    edge_embeddings: torch.Tensor | None,
    relation_embeddings: torch.Tensor | None,
    edge_rel_local: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if relation_embeddings is not None or edge_rel_local is not None:
        if relation_embeddings is None or edge_rel_local is None:
            raise ValueError(
                "relation_embeddings and edge_rel_local must be provided together."
            )
        return _compact_relation_table(
            relation_embeddings=relation_embeddings,
            edge_rel_local=edge_rel_local,
        )
    if edge_embeddings is None:
        raise ValueError(
            "build_graph_batch requires either edge_embeddings or "
            "(relation_embeddings, edge_rel_local)."
        )
    return _build_relation_table(
        edge_rel_global=edge_rel_global,
        edge_embeddings=edge_embeddings,
    )


def build_graph_batch(
    batch: GraphBatchProtocol,
    *,
    validate: bool = True,
) -> tuple[GraphTopology, GraphObservation]:
    if isinstance(batch, TrajectoryBatch) and not validate:
        batch.require_raw_features()
        tensors = {
            "node_ptr": batch.node_ptr,
            "edge_index": batch.edge_index,
            "edge_rel_global": batch.edge_rel_global,
            "node_embeddings": batch.node_embeddings,
            "edge_embeddings": batch.edge_embeddings,
            "relation_embeddings": batch.relation_embeddings,
            "edge_rel_local": batch.edge_rel_local,
            "question_emb": batch.question_emb,
            "question_ctx": batch.question_ctx,
            "question_ctx_mask": batch.question_ctx_mask,
            "anchor_local_indices": batch.anchor_local_indices,
            "anchor_ptr": batch.anchor_ptr,
            "node_entity_ids": batch.node_entity_ids,
        }
    else:
        tensors = _validate_graph_batch_protocol(batch)
    relation_embeddings, edge_relations = _resolve_relation_table(
        edge_rel_global=tensors["edge_rel_global"],
        edge_embeddings=tensors["edge_embeddings"],
        relation_embeddings=tensors["relation_embeddings"],
        edge_rel_local=tensors["edge_rel_local"],
    )
    node_ptr = tensors["node_ptr"]
    edge_index = tensors["edge_index"]
    node_embeddings = tensors["node_embeddings"]
    question_emb = tensors["question_emb"]
    question_ctx = tensors["question_ctx"]
    question_ctx_mask = tensors["question_ctx_mask"]
    anchor_local_indices_tensor = tensors["anchor_local_indices"]
    anchor_ptr = tensors["anchor_ptr"]
    node_entity_ids = tensors["node_entity_ids"]
    assert node_ptr is not None
    assert edge_index is not None
    assert node_embeddings is not None
    assert question_emb is not None
    assert question_ctx is not None
    assert question_ctx_mask is not None
    assert anchor_local_indices_tensor is not None
    assert anchor_ptr is not None
    assert node_entity_ids is not None
    num_nodes = int(node_ptr[-1].item())
    anchor_local_indices = GroupedLocalNodeIndex.from_group_ptr(
        local_indices=anchor_local_indices_tensor,
        group_ptr=anchor_ptr,
        num_groups=batch.num_graphs,
        field_name="anchor_local_indices",
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
        edge_index=edge_index,
        edge_type=edge_relations,
        _graph_node_offsets=node_ptr,
        adjacency=_build_csr_with_edge_ids(
            edge_index=edge_index,
            num_nodes_total=num_nodes,
        ),
        reverse_adjacency=_build_csr_with_edge_ids(
            edge_index=edge_index.flip(0),
            num_nodes_total=num_nodes,
        ),
    )
    observation = GraphObservation(
        node_features=node_embeddings,
        relation_features=relation_embeddings,
        node_entity_ids=node_entity_ids,
        question_embedding=question_emb,
        question_context=question_ctx,
        question_valid_mask=question_ctx_mask,
        anchor_local_indices=anchor_local_indices,
        sample_ids=sample_ids,
    )

    if validate:
        topology.validate()
        observation.validate(topology=topology)
        topology.resolve_local_node_indices(
            observation.anchor_local_indices,
            field_name="anchor_local_indices",
            validate_grouping=False,
        )
    return topology, observation


__all__ = ["build_graph_batch"]
