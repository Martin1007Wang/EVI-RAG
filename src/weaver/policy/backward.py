from __future__ import annotations

import torch

from src.weaver.context import GraphContext
from src.weaver.state import StateBatch, root_reachable_mask_from_edges

Tensor = torch.Tensor


def legal_predecessor_count(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    width = int(child_state.budget)
    valid = torch.arange(width, device=child_state.device).view(1, -1).lt(child_state.edge_count.view(-1, 1))
    child_rows, remove_pos = valid.nonzero(as_tuple=True)
    counts = torch.zeros(child_state.num_states, dtype=torch.long, device=child_state.device)
    if int(child_rows.numel()) == 0:
        return counts

    removed_edges = child_state.edge_ids[child_rows, remove_pos]
    parent_edges = child_state.edge_ids.index_select(0, child_rows).clone()
    parent_edges[torch.arange(int(child_rows.numel()), device=child_state.device), remove_pos] = -1
    sentinel = max(int(graph_context.num_edges), 1)
    parent_edges = torch.sort(torch.where(parent_edges.lt(0), sentinel, parent_edges), dim=1).values
    parent_edges = torch.where(parent_edges.eq(sentinel), -1, parent_edges)
    parent_count = child_state.edge_count.index_select(0, child_rows) - 1
    parent_graph_ids = child_state.graph_ids.index_select(0, child_rows)
    parent = StateBatch(
        graph_ids=parent_graph_ids,
        edge_ids=parent_edges,
        edge_count=parent_count,
        budget=width,
    )

    root_reachable = root_reachable_mask_from_edges(
        edge_ids=parent_edges,
        edge_count=parent_count,
        graph=graph_context,
    )
    active = parent.active_node_index(graph_context)
    node_span = max(int(graph_context.num_nodes), 1)
    active_keys = active.row_ids * node_span + active.node_ids
    removed_src = graph_context.edge_src.index_select(0, removed_edges)
    request_keys = torch.arange(int(child_rows.numel()), device=child_state.device) * node_span + removed_src
    legal = root_reachable & torch.isin(request_keys, active_keys)
    return counts.scatter_add_(0, child_rows, legal.long())


def uniform_backward_log_prob(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    counts = legal_predecessor_count(
        child_state=child_state,
        graph_context=graph_context,
    )
    if bool(counts.le(0).any()):
        raise ValueError("Every child state must have a legal root-reachable predecessor.")
    return -torch.log(counts.float())


__all__ = [
    "legal_predecessor_count",
    "uniform_backward_log_prob",
]
