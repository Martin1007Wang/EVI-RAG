from __future__ import annotations

import torch


def gather_actions_from_csr_lock_free(
    *,
    adj_t,
    active_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather outgoing edges for each active node from a CSR adjacency."""
    crow = adj_t.crow_indices()
    col = adj_t.col_indices()
    values = adj_t.values()

    start_ptrs = crow[active_nodes]
    end_ptrs = crow[active_nodes + 1]
    out_degrees = end_ptrs - start_ptrs

    total_edges = int(out_degrees.sum().item())
    if total_edges == 0:
        empty = torch.empty(0, dtype=torch.long, device=active_nodes.device)
        return empty, empty, out_degrees

    base_idx = start_ptrs.repeat_interleave(out_degrees)
    segment_starts = out_degrees.cumsum(0) - out_degrees
    flat_offsets = torch.arange(
        total_edges, device=active_nodes.device, dtype=torch.long
    )
    increments = flat_offsets - segment_starts.repeat_interleave(out_degrees)
    gather_idx = base_idx + increments

    target_nodes = col[gather_idx]
    gathered_edge_ids = values[gather_idx]
    return gathered_edge_ids, target_nodes, out_degrees


__all__ = ["gather_actions_from_csr_lock_free"]
