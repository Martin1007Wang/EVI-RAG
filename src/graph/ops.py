from __future__ import annotations

from typing import Sequence

import torch

Edge = tuple[str, str, str]


def check_edge_index(edge_index: torch.Tensor) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )


def check_bool_vector(x: torch.Tensor, *, name: str, size: int | None = None) -> None:
    if x.dtype != torch.bool:
        raise TypeError(f"{name} must be bool, got {x.dtype}.")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(x.shape)}.")
    if size is not None and x.numel() != size:
        raise ValueError(f"{name} must have shape [{size}], got {tuple(x.shape)}.")


def build_local_graph(
    graph_edges: Sequence[Edge],
) -> tuple[dict[str, int], torch.Tensor]:
    """
    Build local node ids and directed edge_index.

    Contract:
    - one edge_index column per input triple;
    - parallel triples are preserved;
    - node ids follow first occurrence order in graph_edges.
    """
    node_index: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []

    for head, _, tail in graph_edges:
        src.append(node_index.setdefault(head, len(node_index)))
        dst.append(node_index.setdefault(tail, len(node_index)))

    return node_index, torch.tensor((src, dst), dtype=torch.long)


def rebuild_active_nodes(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Reconstruct active nodes from anchors and selected edges:

        V_s = A ∪ endpoints(E_s)
    """
    check_edge_index(edge_index)

    num_nodes = int(anchor_mask.numel())
    num_edges = int(edge_index.size(1))

    check_bool_vector(anchor_mask, name="anchor_mask")
    check_bool_vector(active_edges, name="active_edges", size=num_edges)

    edge_index = edge_index.to(device=anchor_mask.device, dtype=torch.long)
    active_edges = active_edges.to(device=anchor_mask.device)

    active_nodes = anchor_mask.clone()
    active_nodes[edge_index[:, active_edges].reshape(-1)] = True
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
    Iteratively remove non-protected leaves from the active undirected support.

    This is an evaluation utility, not a training transition.
    Direction is ignored only for degree pruning; edge identity is preserved.
    """
    check_edge_index(edge_index)

    num_nodes = int(active_nodes.numel())
    num_edges = int(edge_index.size(1))

    check_bool_vector(active_nodes, name="active_nodes")
    check_bool_vector(protected_nodes, name="protected_nodes", size=num_nodes)
    check_bool_vector(active_edges, name="active_edges", size=num_edges)

    device = active_nodes.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    active_edges = active_edges.to(device=device)
    protected_nodes = protected_nodes.to(device=device)

    src, dst = edge_index
    alive_nodes = active_nodes.clone()
    max_steps = num_nodes if max_iters is None else int(max_iters)

    for _ in range(max_steps):
        alive_edges = active_edges & alive_nodes[src] & alive_nodes[dst]

        degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
        edge_src = src[alive_edges]
        edge_dst = dst[alive_edges]
        ones = torch.ones(edge_src.numel(), dtype=torch.long, device=device)

        degree.index_add_(0, edge_src, ones)
        degree.index_add_(0, edge_dst, ones)

        prune_nodes = alive_nodes & ~protected_nodes & degree.le(1)
        if not bool(prune_nodes.any()):
            break

        alive_nodes &= ~prune_nodes

    core_edges = active_edges & alive_nodes[src] & alive_nodes[dst]
    return alive_nodes, core_edges


__all__ = [
    "Edge",
    "build_local_graph",
    "check_bool_vector",
    "check_edge_index",
    "prune_to_protected_core",
    "rebuild_active_nodes",
]
