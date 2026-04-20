from __future__ import annotations

import torch
from torch_scatter import scatter_min, scatter_sum


def compute_component_labels(
    *,
    num_nodes: int,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    max_iters: int = 20,
) -> torch.Tensor:
    """Compute weakly connected-component labels for the active subgraph."""
    device = active_nodes.device
    node_labels = torch.arange(num_nodes, device=device)
    node_labels[~active_nodes] = num_nodes

    if edge_index.numel() == 0 or not bool(active_edges.any().item()):
        return node_labels

    src = edge_index[0][active_edges]
    dst = edge_index[1][active_edges]
    if src.numel() == 0:
        return node_labels

    u = torch.cat([src, dst])
    v = torch.cat([dst, src])

    for _ in range(max_iters):
        min_neighbor_labels, _ = scatter_min(node_labels[u], v, dim_size=num_nodes)
        min_neighbor_labels[min_neighbor_labels > num_nodes] = num_nodes
        new_labels = torch.min(node_labels, min_neighbor_labels)
        if torch.equal(new_labels, node_labels):
            break
        node_labels = new_labels

    return node_labels


def count_components(
    *,
    component_labels: torch.Tensor,
    active_nodes: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    node_ids = torch.arange(component_labels.size(0), device=component_labels.device)
    is_component_root = (component_labels == node_ids) & active_nodes
    return scatter_sum(is_component_root.int(), batch_index, dim=0, dim_size=num_graphs)


def _rebuild_active_nodes_from_edges(
    *,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
) -> torch.Tensor:
    active_nodes = is_anchor_mask.clone()
    if edge_index.numel() == 0 or not bool(active_edges.any().item()):
        return active_nodes

    src = edge_index[0][active_edges]
    dst = edge_index[1][active_edges]
    active_nodes[src] = True
    active_nodes[dst] = True
    return active_nodes


def _components_all_anchor_reachable(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
) -> bool:
    if not bool(is_anchor_mask.any().item()):
        return False
    if not bool(active_nodes.any().item()):
        return True

    component_labels = compute_component_labels(
        num_nodes=int(active_nodes.numel()),
        active_nodes=active_nodes,
        active_edges=active_edges,
        edge_index=edge_index,
        max_iters=max(int(active_nodes.numel()), 1),
    )

    active_labels = torch.unique(component_labels[active_nodes])
    anchor_labels = torch.unique(component_labels[is_anchor_mask])
    membership = active_labels.unsqueeze(1) == anchor_labels.unsqueeze(0)
    return bool(membership.any(dim=1).all().item())


def _is_forward_reachable_state(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
) -> bool:
    """
    Return whether a state is reachable by the forward edge-expansion MDP.

    In the current MDP, a state is forward-reachable iff every active connected
    component contains at least one anchor. This helper gives that contract a
    name so backward-parent checks can be phrased as exact inverse-transition
    checks rather than only as a structural heuristic.
    """
    return _components_all_anchor_reachable(
        active_nodes=active_nodes,
        active_edges=active_edges,
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
    )


def _is_exact_forward_parent(
    *,
    child_active_nodes: torch.Tensor,
    child_active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    removed_edge_pos: int,
) -> bool:
    """
    Check whether deleting one edge yields an exact one-step forward parent.

    The returned parent must satisfy all of the following:
    1. The parent state is itself forward-reachable.
    2. The removed edge is a legal frontier expansion from that parent, i.e. at
       least one endpoint is already active in the parent state.
    3. Re-applying that one forward expansion reconstructs the exact child
       state, including the rebuilt active-node mask.
    """
    parent_active_edges = child_active_edges.clone()
    parent_active_edges[removed_edge_pos] = False
    parent_active_nodes = _rebuild_active_nodes_from_edges(
        active_edges=parent_active_edges,
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
    )

    if not _is_forward_reachable_state(
        active_nodes=parent_active_nodes,
        active_edges=parent_active_edges,
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
    ):
        return False

    removed_src = int(edge_index[0, removed_edge_pos].item())
    removed_dst = int(edge_index[1, removed_edge_pos].item())
    if not bool(parent_active_nodes[removed_src] | parent_active_nodes[removed_dst]):
        return False

    reconstructed_child_edges = parent_active_edges.clone()
    reconstructed_child_edges[removed_edge_pos] = True
    reconstructed_child_nodes = _rebuild_active_nodes_from_edges(
        active_edges=reconstructed_child_edges,
        edge_index=edge_index,
        is_anchor_mask=is_anchor_mask,
    )
    return torch.equal(reconstructed_child_edges, child_active_edges) and torch.equal(
        reconstructed_child_nodes, child_active_nodes
    )


def compute_valid_backward_removals(
    *,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    edge_index: torch.Tensor,
    is_anchor_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact one-step forward parents of the current child state.

    A non-root active edge is backward-removable iff deleting it yields a parent
    state that is forward-reachable and the removed edge is a legal one-step
    frontier expansion from that parent. This is the precise inverse-transition
    relation needed by the trajectory-balance backward policy.

    root_active_edges（锚点诱导子图边）由函数内部从 is_anchor_mask 重建，
    不再作为外部参数传入——调用方不需要持有或传递这个不变量。
    """
    if active_nodes.dtype != torch.bool:
        raise TypeError("active_nodes must be a torch.bool tensor.")
    if active_edges.dtype != torch.bool:
        raise TypeError("active_edges must be a torch.bool tensor.")
    if is_anchor_mask.dtype != torch.bool:
        raise TypeError("is_anchor_mask must be a torch.bool tensor.")

    # 从 is_anchor_mask 内部重建 root_active_edges，无需外部传入
    src_global, dst_global = edge_index[0], edge_index[1]
    root_active_edges = is_anchor_mask[src_global] & is_anchor_mask[dst_global]

    removable_mask = torch.zeros_like(active_edges)
    candidate_mask = active_edges & ~root_active_edges
    if not bool(candidate_mask.any().item()):
        removable_counts = scatter_sum(
            removable_mask.int(), edge_batch, dim=0, dim_size=num_graphs
        )
        return removable_mask, removable_counts

    for graph_idx in range(num_graphs):
        graph_edge_ids = torch.nonzero(
            edge_batch == graph_idx, as_tuple=False
        ).view(-1)
        graph_node_ids = torch.nonzero(
            node_batch == graph_idx, as_tuple=False
        ).view(-1)
        if graph_edge_ids.numel() == 0 or graph_node_ids.numel() == 0:
            continue

        local_candidate_ids = graph_edge_ids[candidate_mask[graph_edge_ids]]
        if local_candidate_ids.numel() == 0:
            continue

        node_offset = int(graph_node_ids[0].item())
        local_edge_index = edge_index[:, graph_edge_ids] - node_offset
        local_anchor_mask = is_anchor_mask[graph_node_ids]
        local_active_edges = active_edges[graph_edge_ids]

        expected_active_nodes = _rebuild_active_nodes_from_edges(
            active_edges=local_active_edges,
            edge_index=local_edge_index,
            is_anchor_mask=local_anchor_mask,
        )
        if not torch.equal(expected_active_nodes, active_nodes[graph_node_ids]):
            raise ValueError(
                "active_nodes is inconsistent with active_edges for backward-removal "
                f"check on graph {graph_idx}."
            )

        for edge_id in local_candidate_ids:
            local_edge_pos = int(
                (graph_edge_ids == edge_id).nonzero(as_tuple=False)[0].item()
            )
            if _is_exact_forward_parent(
                child_active_nodes=expected_active_nodes,
                child_active_edges=local_active_edges,
                edge_index=local_edge_index,
                is_anchor_mask=local_anchor_mask,
                removed_edge_pos=local_edge_pos,
            ):
                removable_mask[edge_id] = True

    removable_counts = scatter_sum(
        removable_mask.int(), edge_batch, dim=0, dim_size=num_graphs
    )
    return removable_mask, removable_counts


__all__ = [
    "compute_component_labels",
    "compute_valid_backward_removals",
    "count_components",
]