from __future__ import annotations

from typing import Sequence

import torch


def check_edge_index(edge_index: torch.Tensor) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )


def rebuild_active_nodes(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct node state from anchors and active edges.

        V_s = A union endpoints(E_s)
    """
    check_edge_index(edge_index)

    if active_edges.dtype != torch.bool:
        raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")
    if anchor_mask.dtype != torch.bool:
        raise TypeError(f"anchor_mask must be bool, got {anchor_mask.dtype}.")

    num_edges = int(edge_index.size(1))
    if active_edges.numel() != num_edges:
        raise ValueError(
            f"active_edges must have shape [{num_edges}], got {tuple(active_edges.shape)}."
        )

    device = edge_index.device
    active_edges = active_edges.to(device=device, dtype=torch.bool)
    active_nodes = anchor_mask.to(device=device, dtype=torch.bool).clone()

    if active_edges.numel() == 0 or not bool(active_edges.any()):
        return active_nodes

    src = edge_index[0, active_edges]
    dst = edge_index[1, active_edges]

    active_nodes[src] = True
    active_nodes[dst] = True

    return active_nodes


def prune_to_protected_core(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    protected_nodes: torch.Tensor,
    max_iters: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluation utility: iteratively prune non-protected leaves from an active
    subgraph.
    """
    check_edge_index(edge_index)

    if active_nodes.dtype != torch.bool:
        raise TypeError(f"active_nodes must be bool, got {active_nodes.dtype}.")
    if active_edges.dtype != torch.bool:
        raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")
    if protected_nodes.dtype != torch.bool:
        raise TypeError(f"protected_nodes must be bool, got {protected_nodes.dtype}.")

    num_nodes = int(active_nodes.numel())
    num_edges = int(edge_index.size(1))

    if active_edges.numel() != num_edges:
        raise ValueError(
            f"active_edges must have shape [{num_edges}], got {tuple(active_edges.shape)}."
        )
    if protected_nodes.numel() != num_nodes:
        raise ValueError(
            f"protected_nodes must have shape [{num_nodes}], got {tuple(protected_nodes.shape)}."
        )

    device = active_nodes.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    active_edges = active_edges.to(device=device, dtype=torch.bool)
    protected_nodes = protected_nodes.to(device=device, dtype=torch.bool)

    if num_nodes == 0 or num_edges == 0 or not bool(active_edges.any()):
        return active_nodes.clone(), torch.zeros_like(active_edges)

    src, dst = edge_index
    alive_nodes = active_nodes.clone()
    max_steps = num_nodes if max_iters is None else int(max_iters)

    for _ in range(max_steps):
        alive_edges = (
            active_edges
            & alive_nodes.index_select(0, src)
            & alive_nodes.index_select(0, dst)
        )

        degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        if bool(alive_edges.any()):
            edge_src = src[alive_edges]
            edge_dst = dst[alive_edges]
            ones = torch.ones(edge_src.numel(), dtype=torch.long, device=device)
            degree.index_add_(0, edge_src, ones)
            degree.index_add_(0, edge_dst, ones)

        prune_nodes = alive_nodes & ~protected_nodes & degree.le(1)
        if not bool(prune_nodes.any()):
            break

        alive_nodes = alive_nodes & ~prune_nodes

    core_edges = (
        active_edges
        & alive_nodes.index_select(0, src)
        & alive_nodes.index_select(0, dst)
    )

    return alive_nodes, core_edges


def build_local_graph(
    graph_edges: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, int], torch.Tensor]:
    """Build edge_index with one column per input triple-id edge."""
    node_index: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []

    for head, _, tail in graph_edges:
        src.append(node_index.setdefault(head, len(node_index)))
        dst.append(node_index.setdefault(tail, len(node_index)))

    return node_index, torch.tensor([src, dst], dtype=torch.long)


__all__ = [
    "build_local_graph",
    "check_edge_index",
    "prune_to_protected_core",
    "rebuild_active_nodes",
]
