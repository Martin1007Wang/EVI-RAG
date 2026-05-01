from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.weaver.state import State


def frontier_edges(
    *,
    batch: RetrievalBatch,
    state: State,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Legal Expand candidates.

        C(s) = {e=(u,r,v) in E \\ E_s : u in V_s or v in V_s}

    active_edges is the full current edge set E_s, including root edges.
    Root edges are excluded from frontier only because they are already active.

    Returns:
        edge_ids:
            Physical edge ids in the batched graph.

        edge_batch:
            Graph id for each returned edge.
    """
    device = device or state.device

    edge_index = batch.edge_index.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=device, dtype=torch.bool)

    src, dst = edge_index
    incident = active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
    frontier = incident & ~active_edges

    edge_ids = frontier.nonzero(as_tuple=False).flatten()
    return edge_ids, edge_batch.index_select(0, edge_ids)


def has_frontier_edge_per_graph(
    *,
    edge_batch: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    num_graphs = int(num_graphs)
    frontier_edge_ids = frontier_edge_ids.to(device=device, dtype=torch.long).view(-1)

    if frontier_edge_ids.numel() == 0:
        return torch.zeros(num_graphs, dtype=torch.bool, device=device)

    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    graph_ids = edge_batch.index_select(0, frontier_edge_ids)

    return torch.bincount(graph_ids, minlength=num_graphs).gt(0)


__all__ = [
    "frontier_edges",
    "has_frontier_edge_per_graph",
]
