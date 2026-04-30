from __future__ import annotations

import torch

from .batch import RetrievalBatch


_NODE_OFFSET_FIELDS = (
    "anchor_node_ids",
    "target_node_ids",
    "reachable_target_node_ids",
)

_NODE_ALIGNED_FIELDS = (
    "node_entity_catalog_ids",
    "non_text_node_mask",
    "is_non_text_entity",
)

_EDGE_ALIGNED_FIELDS = ("edge_relation_catalog_ids",)

_OPTIONAL_NODE_ALIGNED_FIELDS = (
    "node_tokens",
    "anchor_node_forward_distances_flat",
    "anchor_node_backward_distances_flat",
    "node_target_distance",
    "heuristic_log_v",
)

_OPTIONAL_EDGE_ALIGNED_FIELDS = ("relation_tokens",)

_FLAT_LABEL_FIELDS = (
    "target_node_distances_flat",
    "target_shortest_path_edge_mask_flat",
)

_GRAPH_SEQUENCE_FIELDS = (
    "question_id",
    "question",
    "dataset",
    "split",
)


def repeat_retrieval_batch(batch: RetrievalBatch, repeats: int) -> RetrievalBatch:
    """
    Physically repeat a RetrievalBatch.

    Layout:
        repeated_graph_id = repeat_id * B + graph_id
        repeated_node_id  = repeat_id * N + original_node_id
        repeated_edge_id  = repeat_id * E + original_edge_id
    """

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")
    if repeats == 1:
        return batch

    _validate_batch(batch)

    device = batch.edge_index.device
    num_graphs = int(batch.ptr.numel()) - 1
    num_nodes = int(batch.num_nodes_total)
    num_edges = int(batch.edge_index.size(1))

    out = RetrievalBatch()
    out.num_nodes = num_nodes * repeats
    out.num_nodes_total = num_nodes * repeats

    out.edge_index = torch.cat(
        [batch.edge_index + i * num_nodes for i in range(repeats)],
        dim=1,
    ).contiguous()

    out.batch = torch.cat(
        [batch.batch + i * num_graphs for i in range(repeats)],
        dim=0,
    ).contiguous()

    out.edge_batch = torch.cat(
        [batch.edge_batch + i * num_graphs for i in range(repeats)],
        dim=0,
    ).contiguous()

    zero = torch.zeros(1, dtype=torch.long, device=device)

    out.ptr = torch.cat(
        [zero] + [batch.ptr[1:] + i * num_nodes for i in range(repeats)],
        dim=0,
    ).contiguous()
    out.node_ptr = out.ptr

    out.edge_ptr = torch.cat(
        [zero] + [batch.edge_ptr[1:] + i * num_edges for i in range(repeats)],
        dim=0,
    ).contiguous()

    _repeat_question_emb(
        batch=batch,
        out=out,
        repeats=repeats,
        num_graphs=num_graphs,
    )

    _repeat_node_offset_fields(
        batch=batch,
        out=out,
        repeats=repeats,
        num_nodes=num_nodes,
    )

    _repeat_tensor_fields(
        batch=batch,
        out=out,
        names=(
            _NODE_ALIGNED_FIELDS
            + _EDGE_ALIGNED_FIELDS
            + _OPTIONAL_NODE_ALIGNED_FIELDS
            + _OPTIONAL_EDGE_ALIGNED_FIELDS
            + _FLAT_LABEL_FIELDS
        ),
        repeats=repeats,
    )

    _repeat_graph_sequence_fields(
        batch=batch,
        out=out,
        repeats=repeats,
    )

    return out


def _repeat_node_offset_fields(
    *,
    batch: RetrievalBatch,
    out: RetrievalBatch,
    repeats: int,
    num_nodes: int,
) -> None:
    for name in _NODE_OFFSET_FIELDS:
        if not hasattr(batch, name):
            continue

        value = getattr(batch, name).to(dtype=torch.long)

        setattr(
            out,
            name,
            torch.cat(
                [value + i * num_nodes for i in range(repeats)],
                dim=0,
            ).contiguous(),
        )


def _repeat_tensor_fields(
    *,
    batch: RetrievalBatch,
    out: RetrievalBatch,
    names: tuple[str, ...],
    repeats: int,
) -> None:
    for name in names:
        if hasattr(batch, name):
            setattr(out, name, _repeat_tensor(getattr(batch, name), repeats))


def _repeat_question_emb(
    *,
    batch: RetrievalBatch,
    out: RetrievalBatch,
    repeats: int,
    num_graphs: int,
) -> None:
    question_emb = batch.question_emb

    if question_emb.ndim == 1:
        question_emb = question_emb.view(1, -1)

    if question_emb.ndim != 2:
        raise ValueError(
            f"question_emb must be 1D or 2D, got shape={tuple(question_emb.shape)}."
        )

    if int(question_emb.size(0)) != num_graphs:
        raise ValueError(
            "question_emb batch dimension must match num_graphs: "
            f"{int(question_emb.size(0))} != {num_graphs}."
        )

    out.question_emb = question_emb.repeat((repeats, 1)).contiguous()


def _repeat_graph_sequence_fields(
    *,
    batch: RetrievalBatch,
    out: RetrievalBatch,
    repeats: int,
) -> None:
    for name in _GRAPH_SEQUENCE_FIELDS:
        if not hasattr(batch, name):
            continue

        value = getattr(batch, name)

        if isinstance(value, list | tuple):
            setattr(out, name, value * repeats)


def _repeat_tensor(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.repeat(repeats).contiguous()

    return torch.cat([tensor] * repeats, dim=0).contiguous()


def _validate_batch(batch: RetrievalBatch) -> None:
    required = (
        "edge_index",
        "batch",
        "edge_batch",
        "ptr",
        "edge_ptr",
        "question_emb",
        "num_nodes_total",
    )
    missing = [name for name in required if not hasattr(batch, name)]
    if missing:
        raise RuntimeError(f"RetrievalBatch is missing required fields: {missing}.")

    if batch.edge_index.ndim != 2 or int(batch.edge_index.size(0)) != 2:
        raise ValueError(
            "batch.edge_index must have shape [2, num_edges], got "
            f"{tuple(batch.edge_index.shape)}."
        )

    num_graphs = int(batch.ptr.numel()) - 1

    if batch.batch.ndim != 1:
        raise ValueError(f"batch.batch must be 1D, got {tuple(batch.batch.shape)}.")

    if batch.edge_batch.ndim != 1:
        raise ValueError(
            f"batch.edge_batch must be 1D, got {tuple(batch.edge_batch.shape)}."
        )

    if batch.question_emb.ndim == 2 and int(batch.question_emb.size(0)) != num_graphs:
        raise ValueError(
            "batch.question_emb first dimension must match num_graphs: "
            f"{int(batch.question_emb.size(0))} != {num_graphs}."
        )


__all__ = ["repeat_retrieval_batch"]
