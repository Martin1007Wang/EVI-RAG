from __future__ import annotations

import torch

_EDGE_BATCH_INVERSION_PREVIEW = 5
_EDGE_BATCH_MISMATCH_PREVIEW = 5


def compute_edge_batch(
    edge_index: torch.Tensor,
    *,
    node_ptr: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if node_ptr.numel() != num_graphs + 1:
        raise ValueError(f"node_ptr length mismatch: got {node_ptr.numel()} expected {num_graphs + 1}")
    # NOTE: right=True assigns boundary nodes to their owning graph: ptr[g] <= i < ptr[g+1].
    edge_batch = torch.bucketize(edge_index[0], node_ptr[1:], right=True)
    tail_batch = torch.bucketize(edge_index[1], node_ptr[1:], right=True)
    if not torch.equal(edge_batch, tail_batch):
        mismatch = torch.nonzero(edge_batch != tail_batch, as_tuple=False).view(-1)
        preview = mismatch[:_EDGE_BATCH_MISMATCH_PREVIEW].detach().cpu().tolist()
        raise ValueError(
            "edge_index crosses graph boundaries; head/tail graph assignments differ. "
            f"first_mismatches={preview}"
        )
    if edge_batch.numel() > 1 and not bool((edge_batch[:-1] <= edge_batch[1:]).all().item()):
        inv = torch.nonzero(edge_batch[:-1] > edge_batch[1:], as_tuple=False).view(-1)
        preview = inv[:_EDGE_BATCH_INVERSION_PREVIEW].detach().cpu().tolist()
        raise ValueError(
            "edge_batch is not non-decreasing along the flattened edge list, which breaks per-graph slicing; "
            f"first_inversions={preview}. Ensure edges are concatenated per-graph (PyG Batch)."
        )
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_ptr = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    edge_ptr[1:] = edge_counts.cumsum(0)
    return edge_batch, edge_ptr


__all__ = ["compute_edge_batch"]
