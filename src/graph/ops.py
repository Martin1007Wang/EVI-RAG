from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch
from torch_scatter import scatter_sum


def check_edge_index(edge_index: torch.Tensor) -> None:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(edge_index.shape)}."
        )


def build_anchor_induced_edge_mask(
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Root-edge convention:

        E_0 = {(u,v) in E : u in A and v in A}
    """
    check_edge_index(edge_index)

    if anchor_mask.dtype != torch.bool:
        raise TypeError(f"anchor_mask must be bool, got {anchor_mask.dtype}.")

    num_edges = edge_index.size(1)
    if num_edges == 0:
        return torch.zeros(0, dtype=torch.bool, device=edge_index.device)

    src, dst = edge_index
    return anchor_mask.index_select(0, src) & anchor_mask.index_select(0, dst)


def rebuild_active_nodes(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Canonical node-state reconstruction:

        V_s = A union endpoints(E_s)

    This keeps active_nodes derived from active_edges and anchors.
    """
    check_edge_index(edge_index)

    if active_edges.dtype != torch.bool:
        raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")
    if anchor_mask.dtype != torch.bool:
        raise TypeError(f"anchor_mask must be bool, got {anchor_mask.dtype}.")
    if active_edges.numel() != edge_index.size(1):
        raise ValueError(
            f"active_edges must have shape [{edge_index.size(1)}], got {tuple(active_edges.shape)}."
        )

    active_nodes = anchor_mask.to(device=edge_index.device).clone()

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
    Iteratively prune non-protected leaves from an active subgraph.

    This is an evaluation utility.

    It keeps the structural core connecting protected nodes and removes
    dangling branches. Protected nodes are usually anchors plus active targets.

    Args:
        active_nodes:
            Boolean active-node mask, shape [num_nodes].
        active_edges:
            Boolean active-edge mask, shape [num_edges].
        edge_index:
            Graph edge index, shape [2, num_edges].
        protected_nodes:
            Boolean protected-node mask, shape [num_nodes].
        max_iters:
            Optional safety cap. Defaults to num_nodes.

    Returns:
        core_nodes:
            Active nodes remaining after pruning.
        core_edges:
            Active edges remaining after pruning.
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


def compute_uniform_nonroot_backward_removals(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    root_active_edges: torch.Tensor,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform backward parent set.

    Forward rule:

        C(s) = {e=(u,v) not in E_s : u in V_s or v in V_s}

    Backward rule:

        R(s') = {
            e in E_s' \\ E_0 :
                parent E_p = E_s' \\ {e} is constructible
                and e in C(parent)
        }

    The second condition makes backward removals exactly one-step reversible
    under the forward frontier rule.

    Returns:
        removable_mask:
            Boolean mask over physical edge ids.
        removable_counts:
            Number of removable edges per graph.
    """
    check_edge_index(edge_index)

    if active_edges.dtype != torch.bool:
        raise TypeError(f"active_edges must be bool, got {active_edges.dtype}.")
    if is_anchor_mask.dtype != torch.bool:
        raise TypeError(f"is_anchor_mask must be bool, got {is_anchor_mask.dtype}.")
    if root_active_edges.dtype != torch.bool:
        raise TypeError(
            f"root_active_edges must be bool, got {root_active_edges.dtype}."
        )

    num_edges = edge_index.size(1)
    num_graphs = int(num_graphs)

    if active_edges.numel() != num_edges:
        raise ValueError(
            f"active_edges must have shape [{num_edges}], got {tuple(active_edges.shape)}."
        )
    if root_active_edges.numel() != num_edges:
        raise ValueError(
            f"root_active_edges must have shape [{num_edges}], got {tuple(root_active_edges.shape)}."
        )
    if edge_batch.numel() != num_edges:
        raise ValueError(
            f"edge_batch must have shape [{num_edges}], got {tuple(edge_batch.shape)}."
        )
    if num_graphs < 0:
        raise ValueError(f"num_graphs must be non-negative, got {num_graphs}.")

    device = active_edges.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    anchor_mask = is_anchor_mask.to(device=device, dtype=torch.bool)
    root_edges = root_active_edges.to(device=device, dtype=torch.bool)

    if num_edges == 0:
        return (
            torch.zeros(0, dtype=torch.bool, device=device),
            torch.zeros(num_graphs, dtype=torch.long, device=device),
        )

    if validate:
        missing_root = root_edges & ~active_edges
        if bool(missing_root.any()):
            bad = missing_root.nonzero(as_tuple=False).view(-1)
            raise RuntimeError(
                "root_active_edges must remain active in every state, "
                f"but inactive root edges were found at ids={bad.tolist()}."
            )

    candidates = (active_edges & ~root_edges).nonzero(as_tuple=False).view(-1)

    removable = torch.zeros(num_edges, dtype=torch.bool, device=device)
    if candidates.numel() == 0:
        return removable, torch.zeros(num_graphs, dtype=torch.long, device=device)

    removable_ids = _local_backward_removable_edge_ids(
        edge_ids=candidates,
        edge_index=edge_index,
        edge_batch=edge_batch,
        anchor_mask=anchor_mask,
    )
    if removable_ids:
        removable[torch.tensor(removable_ids, dtype=torch.long, device=device)] = True

    counts = scatter_sum(
        removable.to(torch.long),
        edge_batch,
        dim=0,
        dim_size=num_graphs,
    ).to(dtype=torch.long)

    return removable, counts


def _local_backward_removable_edge_ids(
    *,
    edge_ids: torch.Tensor,
    edge_index: torch.Tensor,
    edge_batch: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> list[int]:
    """
    Exact removable-parent check over the small selected non-root edge set.

    Rollout states select at most expand_budget non-root edges per graph. Moving
    this tiny graph search to Python avoids launching and synchronizing many
    small CUDA kernels while preserving the constructibility contract.
    """
    device = edge_index.device
    edge_ids = edge_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_ids.numel() == 0:
        return []

    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)
    graph_ids = edge_batch.index_select(0, edge_ids)

    endpoint_ids = torch.cat([src, dst], dim=0)
    endpoint_is_anchor = anchor_mask.index_select(0, endpoint_ids).view(2, -1)

    records_by_graph: dict[int, list[tuple[int, int, int, bool, bool]]] = defaultdict(
        list
    )
    for edge_id, graph_id, u, v, u_anchor, v_anchor in zip(
        edge_ids.detach().cpu().tolist(),
        graph_ids.detach().cpu().tolist(),
        src.detach().cpu().tolist(),
        dst.detach().cpu().tolist(),
        endpoint_is_anchor[0].detach().cpu().tolist(),
        endpoint_is_anchor[1].detach().cpu().tolist(),
    ):
        records_by_graph[int(graph_id)].append(
            (int(edge_id), int(u), int(v), bool(u_anchor), bool(v_anchor))
        )

    removable: list[int] = []
    for records in records_by_graph.values():
        removable.extend(_local_graph_removable_edge_ids(records))

    return removable


def _local_graph_removable_edge_ids(
    records: Sequence[tuple[int, int, int, bool, bool]],
) -> list[int]:
    removable: list[int] = []
    graph_anchor_nodes = {
        node
        for _, src, dst, src_is_anchor, dst_is_anchor in records
        for node, is_anchor in ((src, src_is_anchor), (dst, dst_is_anchor))
        if is_anchor
    }

    for remove_idx, (edge_id, src, dst, _, _) in enumerate(records):
        parent = [record for idx, record in enumerate(records) if idx != remove_idx]
        if not _local_edge_set_constructible(parent):
            continue

        parent_active_nodes = set(graph_anchor_nodes)
        for _, parent_src, parent_dst, _, _ in parent:
            parent_active_nodes.add(parent_src)
            parent_active_nodes.add(parent_dst)

        if src in parent_active_nodes or dst in parent_active_nodes:
            removable.append(edge_id)

    return removable


def _local_edge_set_constructible(
    records: Sequence[tuple[int, int, int, bool, bool]],
) -> bool:
    if not records:
        return True

    reachable_nodes = {
        node
        for _, src, dst, src_is_anchor, dst_is_anchor in records
        for node, is_anchor in ((src, src_is_anchor), (dst, dst_is_anchor))
        if is_anchor
    }
    if not reachable_nodes:
        return False

    remaining = [(src, dst) for _, src, dst, _, _ in records]
    while remaining:
        next_remaining: list[tuple[int, int]] = []
        progressed = False
        for src, dst in remaining:
            if src in reachable_nodes or dst in reachable_nodes:
                reachable_nodes.add(src)
                reachable_nodes.add(dst)
                progressed = True
            else:
                next_remaining.append((src, dst))

        if not progressed:
            return False
        remaining = next_remaining

    return True


def _is_constructible_edge_set(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
    root_edges: torch.Tensor,
) -> bool:
    """
    Whether active_edges admits at least one forward construction order
    from the root state under incident expansion.
    """
    if bool((root_edges & ~active_edges).any()):
        return False

    nonroot_edges = active_edges & ~root_edges
    if not bool(nonroot_edges.any()):
        return True

    reachable_nodes = rebuild_active_nodes(
        active_edges=root_edges,
        edge_index=edge_index,
        anchor_mask=anchor_mask,
    )

    edge_ids = nonroot_edges.nonzero(as_tuple=False).view(-1)
    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)

    remaining = torch.ones(edge_ids.numel(), dtype=torch.bool, device=edge_ids.device)

    while bool(remaining.any()):
        incident = remaining & (
            reachable_nodes.index_select(0, src) | reachable_nodes.index_select(0, dst)
        )

        if not bool(incident.any()):
            return False

        remaining[incident] = False
        reachable_nodes[src[incident]] = True
        reachable_nodes[dst[incident]] = True

    return True


def _constructible_reachable_nodes(
    *,
    nonroot_edge_ids: torch.Tensor,
    edge_index: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> tuple[bool, torch.Tensor]:
    """
    Check constructibility for a small selected non-root edge set.

    Rollout states contain at most ``expand_budget`` non-root edges. Restricting
    this check to those selected ids avoids cloning and rescanning the full graph
    once for every possible backward removal.
    """
    reachable_nodes = anchor_mask.to(device=edge_index.device, dtype=torch.bool).clone()

    if nonroot_edge_ids.numel() == 0:
        return True, reachable_nodes

    edge_ids = nonroot_edge_ids.to(device=edge_index.device, dtype=torch.long).view(-1)
    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)

    remaining = torch.ones(edge_ids.numel(), dtype=torch.bool, device=edge_ids.device)

    while bool(remaining.any()):
        incident = remaining & (
            reachable_nodes.index_select(0, src) | reachable_nodes.index_select(0, dst)
        )

        if not bool(incident.any()):
            return False, reachable_nodes

        remaining[incident] = False
        reachable_nodes[src[incident]] = True
        reachable_nodes[dst[incident]] = True

    return True, reachable_nodes


def build_local_graph(
    graph_edges: Sequence[tuple[str, str, str]],
) -> tuple[dict[str, int], torch.Tensor]:
    node_index: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []

    for head, _, tail in graph_edges:
        src.append(node_index.setdefault(head, len(node_index)))
        dst.append(node_index.setdefault(tail, len(node_index)))

    return node_index, torch.tensor([src, dst], dtype=torch.long)


__all__ = [
    "build_anchor_induced_edge_mask",
    "build_local_graph",
    "check_edge_index",
    "compute_uniform_nonroot_backward_removals",
    "prune_to_protected_core",
    "rebuild_active_nodes",
]
