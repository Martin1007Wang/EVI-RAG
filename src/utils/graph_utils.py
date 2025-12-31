from __future__ import annotations

import torch

try:
    from torch import _dynamo as _torch_dynamo
except Exception:  # pragma: no cover - optional torch compile dependency
    _torch_dynamo = None

_EDGE_BATCH_INVERSION_PREVIEW = 5
_EDGE_BATCH_MISMATCH_PREVIEW = 5
_EDGE_BATCH_MIN = 0
_EDGE_BATCH_SAMPLE_PREVIEW = 5
_EDGE_BATCH_POS_PREVIEW = 5
_QA_MASK_DIM = 2
_QA_MASK_EDGE_DIM = 2
POS_LABEL_THRESHOLD = 0.5


def _dynamo_disable(fn):
    if _torch_dynamo is None:
        return fn
    return _torch_dynamo.disable(fn)


def _preview_sample_ids(debug_batch, edge_positions: torch.Tensor) -> list[str]:
    sample_ids = getattr(debug_batch, "sample_id", None)
    slice_dict = getattr(debug_batch, "_slice_dict", None)
    if sample_ids is None or not isinstance(slice_dict, dict):
        return []
    edge_ptr = slice_dict.get("edge_index")
    if edge_ptr is None:
        return []
    if not torch.is_tensor(edge_ptr):
        edge_ptr = torch.as_tensor(edge_ptr, dtype=torch.long)
    edge_ptr = edge_ptr.detach().cpu()
    if edge_ptr.numel() < 2:
        return []
    edge_positions = edge_positions.detach().cpu()
    graph_ids = torch.bucketize(edge_positions, edge_ptr[1:], right=False)
    unique_graphs = sorted(set(graph_ids.tolist()))
    preview: list[str] = []
    for gid in unique_graphs[:_EDGE_BATCH_SAMPLE_PREVIEW]:
        if 0 <= gid < len(sample_ids):
            preview.append(str(sample_ids[gid]))
    return preview


@_dynamo_disable
def compute_edge_batch(
    edge_index: torch.Tensor,
    *,
    node_ptr: torch.Tensor,
    num_graphs: int,
    device: torch.device,
    debug_batch: object | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if node_ptr.numel() != num_graphs + 1:
        raise ValueError(f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}")
    # NOTE: right=True assigns boundary nodes to their owning graph: ptr[g] <= i < ptr[g+1].
    edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
    tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
    if edge_batch.numel() > 0:
        min_idx = int(edge_batch.min().item())
        max_idx = int(edge_batch.max().item())
        if min_idx < _EDGE_BATCH_MIN or max_idx >= num_graphs:
            invalid = torch.nonzero(edge_batch >= num_graphs, as_tuple=False).view(-1)
            preview = invalid[:_EDGE_BATCH_POS_PREVIEW].detach().cpu().tolist()
            sample_preview = _preview_sample_ids(debug_batch, invalid) if debug_batch is not None else []
            detail = f"min={min_idx} max={max_idx} num_graphs={num_graphs}"
            if preview:
                detail += f" invalid_edge_positions={preview}"
            if sample_preview:
                detail += f" sample_id_preview={sample_preview}"
            raise ValueError(f"edge_batch contains out-of-range indices; {detail}.")
    if not torch.equal(edge_batch, tail_batch):
        mismatch = torch.nonzero(edge_batch != tail_batch, as_tuple=False).view(-1)
        preview = mismatch[:_EDGE_BATCH_MISMATCH_PREVIEW].detach().cpu().tolist()
        sample_preview = _preview_sample_ids(debug_batch, mismatch) if debug_batch is not None else []
        detail = f"first_mismatches={preview}"
        if sample_preview:
            detail += f" sample_id_preview={sample_preview}"
        raise ValueError(
            "edge_index crosses graph boundaries; head/tail graph assignments differ. "
            f"{detail}"
        )
    if edge_batch.numel() > 1 and not bool((edge_batch[:-1] <= edge_batch[1:]).all().item()):
        inv = torch.nonzero(edge_batch[:-1] > edge_batch[1:], as_tuple=False).view(-1)
        preview = inv[:_EDGE_BATCH_INVERSION_PREVIEW].detach().cpu().tolist()
        sample_preview = _preview_sample_ids(debug_batch, inv) if debug_batch is not None else []
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


__all__ = ["compute_edge_batch", "compute_qa_edge_mask", "POS_LABEL_THRESHOLD"]
