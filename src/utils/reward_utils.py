from __future__ import annotations

import torch
from torch_scatter import scatter_sum


def build_anchor_induced_edge_mask(
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    return anchor_mask.index_select(0, src) & anchor_mask.index_select(0, dst)


def prune_to_protected_core(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    protected_nodes: torch.Tensor,
    max_iters: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iteratively prune non-protected leaves from the active subgraph."""
    if active_nodes.numel() == 0:
        return active_nodes, active_edges

    src, dst = edge_index[0], edge_index[1]
    alive_nodes = active_nodes.clone()

    for _ in range(max_iters):
        alive_edges = active_edges & alive_nodes[src] & alive_nodes[dst]
        degree = torch.zeros(
            active_nodes.size(0), dtype=torch.long, device=active_nodes.device
        )
        if bool(alive_edges.any().item()):
            ones = torch.ones(
                int(alive_edges.sum().item()),
                dtype=torch.long,
                device=active_nodes.device,
            )
            degree.index_add_(0, src[alive_edges], ones)
            degree.index_add_(0, dst[alive_edges], ones)
        prune_nodes = alive_nodes & ~protected_nodes & degree.le(1)
        if not bool(prune_nodes.any().item()):
            break
        alive_nodes = alive_nodes & ~prune_nodes

    alive_edges = active_edges & alive_nodes[src] & alive_nodes[dst]
    return alive_nodes, alive_edges


def per_graph_mask_count(
    mask: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return scatter_sum(mask.to(dtype=dtype), batch_index, dim=0, dim_size=num_graphs)


__all__ = [
    "build_anchor_induced_edge_mask",
    "per_graph_mask_count",
    "prune_to_protected_core",
]
