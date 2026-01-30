from __future__ import annotations

from dataclasses import dataclass

import torch

_ZERO = 0
_INVALID_EDGE_ID = -1
_SELF_RELATION_ID = -1

try:
    from torch import _dynamo as _torch_dynamo
except Exception:  # pragma: no cover - optional torch compile dependency
    _torch_dynamo = None

_EDGE_BATCH_INVERSION_PREVIEW = 5
_EDGE_BATCH_MISMATCH_PREVIEW = 5
_EDGE_BATCH_MIN = 0
_EDGE_BATCH_SAMPLE_PREVIEW = 5
_EDGE_BATCH_POS_PREVIEW = 5
_EDGE_PTR_MIN_LEN = 2
_QA_MASK_DIM = 2
_QA_MASK_EDGE_DIM = 2


def _dynamo_disable(fn):
    if _torch_dynamo is None:
        return fn
    return _torch_dynamo.disable(fn)


@dataclass(frozen=True)
class EdgeBatchDebugContext:
    sample_ids: list[str]
    edge_ptr: torch.Tensor


def build_edge_batch_debug_context(debug_batch: object) -> EdgeBatchDebugContext | None:
    sample_ids = getattr(debug_batch, "sample_id", None)
    slice_dict = getattr(debug_batch, "_slice_dict", None)
    if sample_ids is None or not isinstance(slice_dict, dict):
        return None
    edge_ptr = slice_dict.get("edge_index")
    if edge_ptr is None:
        return None
    if not torch.is_tensor(edge_ptr):
        edge_ptr = torch.as_tensor(edge_ptr, dtype=torch.long)
    edge_ptr = edge_ptr.detach().to(device="cpu")
    if edge_ptr.numel() < _EDGE_PTR_MIN_LEN:
        return None
    return EdgeBatchDebugContext(sample_ids=[str(sid) for sid in sample_ids], edge_ptr=edge_ptr)


def _preview_sample_ids(debug_context: EdgeBatchDebugContext, edge_positions: torch.Tensor) -> list[str]:
    edge_ptr = debug_context.edge_ptr
    if edge_ptr.numel() < _EDGE_PTR_MIN_LEN:
        return []
    edge_positions = edge_positions.detach().to(device="cpu")
    graph_ids = torch.bucketize(edge_positions, edge_ptr[1:], right=False)
    unique_graphs = sorted(set(graph_ids.tolist()))
    preview: list[str] = []
    for gid in unique_graphs[:_EDGE_BATCH_SAMPLE_PREVIEW]:
        if 0 <= gid < len(debug_context.sample_ids):
            preview.append(debug_context.sample_ids[gid])
    return preview


@_dynamo_disable
def compute_edge_batch(
    edge_index: torch.Tensor,
    *,
    node_ptr: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    validate: bool = True,
    debug_context: EdgeBatchDebugContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if node_ptr.numel() != num_graphs + 1:
        raise ValueError(f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}")
    # NOTE: right=True assigns boundary nodes to their owning graph: ptr[g] <= i < ptr[g+1].
    edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
    tail_batch = None
    if validate:
        tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
    if validate:
        if edge_batch.numel() > 0:
            min_idx = int(edge_batch.min().detach().tolist())
            max_idx = int(edge_batch.max().detach().tolist())
            if min_idx < _EDGE_BATCH_MIN or max_idx >= num_graphs:
                invalid = torch.nonzero(edge_batch >= num_graphs, as_tuple=False).view(-1)
                preview = invalid[:_EDGE_BATCH_POS_PREVIEW].detach().to(device="cpu").tolist()
                sample_preview = _preview_sample_ids(debug_context, invalid) if debug_context is not None else []
                detail = f"min={min_idx} max={max_idx} num_graphs={num_graphs}"
                if preview:
                    detail += f" invalid_edge_positions={preview}"
                if sample_preview:
                    detail += f" sample_id_preview={sample_preview}"
                raise ValueError(f"edge_batch contains out-of-range indices; {detail}.")
        if tail_batch is not None and not torch.equal(edge_batch, tail_batch):
            mismatch = torch.nonzero(edge_batch != tail_batch, as_tuple=False).view(-1)
            preview = mismatch[:_EDGE_BATCH_MISMATCH_PREVIEW].detach().to(device="cpu").tolist()
            sample_preview = _preview_sample_ids(debug_context, mismatch) if debug_context is not None else []
            detail = f"first_mismatches={preview}"
            if sample_preview:
                detail += f" sample_id_preview={sample_preview}"
            raise ValueError(
                "edge_index crosses graph boundaries; head/tail graph assignments differ. "
                f"{detail}"
            )
        if edge_batch.numel() > 1 and not bool((edge_batch[:-1] <= edge_batch[1:]).all().detach().tolist()):
            inv = torch.nonzero(edge_batch[:-1] > edge_batch[1:], as_tuple=False).view(-1)
            preview = inv[:_EDGE_BATCH_INVERSION_PREVIEW].detach().to(device="cpu").tolist()
            sample_preview = _preview_sample_ids(debug_context, inv) if debug_context is not None else []
            detail = f"first_inversions={preview}"
            if sample_preview:
                detail += f" sample_id_preview={sample_preview}"
            raise ValueError(
                "edge_batch is not non-decreasing along the flattened edge list, which breaks per-graph slicing; "
                f"{detail}. Ensure edges are concatenated per-graph (PyG Batch)."
            )
    edge_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
    edge_counts.scatter_add_(0, edge_batch, torch.ones_like(edge_batch, dtype=torch.long))
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = edge_counts.cumsum(0)
    return edge_batch, edge_ptr


def compute_invalid_nodes(
    *,
    edge_index: torch.Tensor,
    node_is_start: torch.Tensor,
    node_is_target: torch.Tensor,
) -> torch.Tensor:
    num_nodes_total = int(node_is_start.numel())
    neighbors = torch.zeros(num_nodes_total, device=edge_index.device, dtype=torch.bool)
    if edge_index.numel() > _ZERO:
        heads = edge_index[0]
        tails = edge_index[1]
        start_heads = node_is_start[heads]
        if bool(start_heads.any().detach().tolist()):
            neighbors[tails[start_heads]] = True
        start_tails = node_is_start[tails]
        if bool(start_tails.any().detach().tolist()):
            neighbors[heads[start_tails]] = True
    return (node_is_start | neighbors) & (~node_is_target)


def compute_qa_edge_mask(
    edge_index: torch.Tensor,
    *,
    num_nodes: int,
    q_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
) -> torch.Tensor:
    if edge_index.dim() != _QA_MASK_DIM or edge_index.size(0) != _QA_MASK_EDGE_DIM:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be positive, got {num_nodes}")
    if not torch.is_tensor(q_local_indices):
        q_local_indices = torch.as_tensor(q_local_indices, dtype=torch.long, device=edge_index.device)
    else:
        q_local_indices = q_local_indices.to(device=edge_index.device, dtype=torch.long)
    if not torch.is_tensor(a_local_indices):
        a_local_indices = torch.as_tensor(a_local_indices, dtype=torch.long, device=edge_index.device)
    else:
        a_local_indices = a_local_indices.to(device=edge_index.device, dtype=torch.long)

    q_local_indices = q_local_indices.view(-1)
    a_local_indices = a_local_indices.view(-1)
    if q_local_indices.numel() == 0 and a_local_indices.numel() == 0:
        return edge_index.new_zeros(edge_index.size(1), dtype=torch.bool)
    if q_local_indices.numel() == 0:
        qa_indices = a_local_indices
    elif a_local_indices.numel() == 0:
        qa_indices = q_local_indices
    else:
        qa_indices = torch.cat([q_local_indices, a_local_indices], dim=0)

    if qa_indices.numel() > 0:
        torch._assert(
            (qa_indices >= 0).all(),
            "q/a local indices contain negative values.",
        )
        torch._assert(
            (qa_indices < num_nodes).all(),
            "q/a local indices exceed num_nodes; batch collation is invalid.",
        )

    node_mask = torch.zeros(num_nodes, device=edge_index.device, dtype=torch.bool)
    if qa_indices.numel() > 0:
        node_mask[qa_indices] = True
    head_idx, tail_idx = edge_index
    return node_mask[head_idx] | node_mask[tail_idx]


def _coerce_edge_inverse_inputs(
    *,
    edge_index: torch.Tensor,
    edge_relations: torch.Tensor,
    inverse_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = edge_relations.device
    if edge_index.device != device:
        edge_index = edge_index.to(device=device)
    if edge_index.dtype != torch.long:
        edge_index = edge_index.to(dtype=torch.long)
    if edge_relations.dtype != torch.long:
        edge_relations = edge_relations.to(dtype=torch.long)
    if inverse_map.device != device:
        inverse_map = inverse_map.to(device=device)
    if inverse_map.dtype != torch.long:
        inverse_map = inverse_map.to(dtype=torch.long)
    heads = edge_index[0].view(-1)
    tails = edge_index[1].view(-1)
    rel = edge_relations.view(-1)
    return heads, tails, rel, inverse_map


def _build_edge_keys(
    *,
    heads: torch.Tensor,
    tails: torch.Tensor,
    rel: torch.Tensor,
    num_nodes_total: int,
    num_relations: int,
) -> torch.Tensor:
    return (heads * num_nodes_total + tails) * num_relations + rel


def _sorted_edge_keys(
    *,
    keys: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_idx = torch.nonzero(valid_mask, as_tuple=False).view(-1)
    if valid_idx.numel() == 0:
        return keys.new_empty((0,), dtype=keys.dtype), valid_idx
    keys_valid = keys.index_select(0, valid_idx)
    sorted_keys, order = torch.sort(keys_valid)
    sorted_edge_idx = valid_idx.index_select(0, order)
    if sorted_keys.numel() > 1:
        dup = sorted_keys[1:] == sorted_keys[:-1]
        torch._assert(~dup.any(), "Parallel edges detected: duplicate (head, tail, relation) entries.")
    return sorted_keys, sorted_edge_idx


def _assign_inverse_edges(
    *,
    inverse_edge: torch.Tensor,
    sorted_keys: torch.Tensor,
    sorted_edge_idx: torch.Tensor,
    inv_keys: torch.Tensor,
    valid_inv: torch.Tensor,
) -> None:
    if sorted_keys.numel() == 0:
        return
    inv_keys_valid = inv_keys[valid_inv]
    pos = torch.searchsorted(sorted_keys, inv_keys_valid)
    in_range = pos < sorted_keys.numel()
    if in_range.any():
        pos_safe = pos[in_range]
        matches = sorted_keys.index_select(0, pos_safe) == inv_keys_valid[in_range]
        if matches.any():
            inverse_ids = sorted_edge_idx.index_select(0, pos_safe[matches])
            inv_idx = torch.nonzero(valid_inv, as_tuple=False).view(-1)
            inv_idx = inv_idx[in_range][matches]
            inverse_edge[inv_idx] = inverse_ids


def _compute_inverse_relations(
    *,
    rel: torch.Tensor,
    inverse_map: torch.Tensor,
    invalid_edge_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_rel = rel >= 0
    inv_rel = rel.new_full(rel.shape, invalid_edge_id)
    if valid_rel.any():
        inv_rel[valid_rel] = inverse_map.index_select(0, rel[valid_rel])
    valid_inv = valid_rel & (inv_rel >= 0)
    return valid_rel, inv_rel, valid_inv


def _assign_inverse_edge_ids(
    *,
    inverse_edge: torch.Tensor,
    heads: torch.Tensor,
    tails: torch.Tensor,
    rel: torch.Tensor,
    inv_rel: torch.Tensor,
    valid_rel: torch.Tensor,
    valid_inv: torch.Tensor,
    num_nodes_total: int,
    num_relations: int,
) -> None:
    keys = _build_edge_keys(
        heads=heads,
        tails=tails,
        rel=rel,
        num_nodes_total=num_nodes_total,
        num_relations=num_relations,
    )
    sorted_keys, sorted_edge_idx = _sorted_edge_keys(keys=keys, valid_mask=valid_rel)
    if valid_inv.any():
        inv_keys = _build_edge_keys(
            heads=tails,
            tails=heads,
            rel=inv_rel,
            num_nodes_total=num_nodes_total,
            num_relations=num_relations,
        )
        _assign_inverse_edges(
            inverse_edge=inverse_edge,
            sorted_keys=sorted_keys,
            sorted_edge_idx=sorted_edge_idx,
            inv_keys=inv_keys,
            valid_inv=valid_inv,
        )


def build_edge_inverse_map(
    *,
    edge_index: torch.Tensor,
    edge_relations: torch.Tensor,
    num_nodes_total: int,
    inverse_map: torch.Tensor,
    num_relations: int,
    self_relation_id: int = _SELF_RELATION_ID,
    invalid_edge_id: int = _INVALID_EDGE_ID,
) -> torch.Tensor:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    num_edges = int(edge_relations.numel())
    if num_edges == 0:
        return edge_relations.new_zeros((0,), dtype=torch.long)
    if num_nodes_total <= 0 or num_relations <= 0:
        return edge_relations.new_full((num_edges,), invalid_edge_id, dtype=torch.long)

    heads, tails, rel, inverse_map = _coerce_edge_inverse_inputs(
        edge_index=edge_index,
        edge_relations=edge_relations,
        inverse_map=inverse_map,
    )
    inverse_edge = rel.new_full((num_edges,), invalid_edge_id, dtype=torch.long)
    self_loop = (rel == self_relation_id) & (heads == tails)

    valid_rel, inv_rel, valid_inv = _compute_inverse_relations(
        rel=rel,
        inverse_map=inverse_map,
        invalid_edge_id=invalid_edge_id,
    )
    if valid_rel.any():
        _assign_inverse_edge_ids(
            inverse_edge=inverse_edge,
            heads=heads,
            tails=tails,
            rel=rel,
            inv_rel=inv_rel,
            valid_rel=valid_rel,
            valid_inv=valid_inv,
            num_nodes_total=int(num_nodes_total),
            num_relations=int(num_relations),
        )

    if self_loop.any():
        self_ids = torch.arange(num_edges, device=rel.device, dtype=torch.long)
        inverse_edge = torch.where(self_loop, self_ids, inverse_edge)
    return inverse_edge


__all__ = [
    "EdgeBatchDebugContext",
    "build_edge_batch_debug_context",
    "compute_edge_batch",
    "compute_qa_edge_mask",
    "build_edge_inverse_map",
]
