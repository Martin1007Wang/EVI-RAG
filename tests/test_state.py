from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.state import State


def build_graph() -> GraphContext:
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        adjacency=build_directed_adjacency_index(edge_index=edge_index, num_nodes=3),
        num_nodes=3,
        num_edges=3,
        num_graphs=1,
    )


def test_sparse_state_exposes_dense_masks_compatibly() -> None:
    graph = build_graph()
    state = State.from_selected_edges(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        selected_edge_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        expand_budget=3,
    )

    assert torch.equal(state.edge_mask, torch.tensor([[True, False, True]], dtype=torch.bool))
    assert torch.equal(state.selected_edge_mask, state.edge_mask)
    assert torch.equal(state.active_node_mask, torch.tensor([[True, True, True]], dtype=torch.bool))
    rows, edge_ids = state.selected_edges()
    assert torch.equal(rows, torch.tensor([0, 0], dtype=torch.long))
    assert torch.equal(edge_ids, torch.tensor([0, 2], dtype=torch.long))


def test_expand_defaults_to_no_frontier_revalidation() -> None:
    graph = build_graph()
    state = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=3,
    )

    frontier = state.frontier(graph, expand_budget=3)
    assert torch.equal(frontier.edge_ids, torch.tensor([0, 1], dtype=torch.long))

    child = state.expand(
        graph=graph,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([1], dtype=torch.long),
        expand_budget=3,
    )
    assert torch.equal(child.selected_edge_mask, torch.tensor([[False, True, False]], dtype=torch.bool))
    assert torch.equal(child.active_node_mask, torch.tensor([[True, False, True]], dtype=torch.bool))
