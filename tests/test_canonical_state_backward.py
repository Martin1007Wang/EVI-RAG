from __future__ import annotations

import math

import pytest
import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.policy.backward import legal_predecessor_count, uniform_backward_log_prob
from src.weaver.state import ExpansionBatch, StateBatch, root_reachable_mask_from_edges


def test_state_is_canonical_across_action_orders() -> None:
    graph = _graph([(0, 1), (0, 2)])
    root = StateBatch.initial(graph_ids=torch.tensor([0, 0]), budget=2, graph_context=graph)
    first = root.advance(ExpansionBatch(torch.tensor([0, 1]), torch.tensor([0, 1])), graph_context=graph)
    state = first.advance(ExpansionBatch(torch.tensor([0, 1]), torch.tensor([1, 0])), graph_context=graph)
    assert state.edge_ids.tolist() == [[0, 1], [0, 1]]
    assert state.active_node_index(graph).node_ids.tolist() == [0, 1, 2, 0, 1, 2]


def test_chain_has_only_one_root_reachable_predecessor() -> None:
    graph = _graph([(0, 1), (1, 2)])
    child = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]]),
        edge_count=torch.tensor([2]),
        budget=2,
        graph_context=graph,
    )
    assert legal_predecessor_count(child_state=child, graph_context=graph).tolist() == [1]
    assert uniform_backward_log_prob(child_state=child, graph_context=graph).tolist() == [0.0]


def test_independent_edges_have_two_predecessors() -> None:
    graph = _graph([(0, 1), (0, 2)])
    child = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]]),
        edge_count=torch.tensor([2]),
        budget=2,
        graph_context=graph,
    )
    assert legal_predecessor_count(child_state=child, graph_context=graph).tolist() == [2]
    assert torch.allclose(uniform_backward_log_prob(child_state=child, graph_context=graph), torch.tensor([-math.log(2.0)]))


def test_from_selected_edges_rejects_disconnected_state() -> None:
    graph = _graph([(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="root-reachable"):
        StateBatch.from_selected_edges(
            graph_ids=torch.tensor([0]),
            edge_ids=torch.tensor([[1, -1]]),
            edge_count=torch.tensor([1]),
            budget=2,
            graph_context=graph,
        )


def test_root_reachable_mask_handles_padding_cycles_and_multiple_anchors() -> None:
    graph = _graph(
        [(0, 1), (1, 2), (2, 0), (3, 4), (4, 3), (5, 6), (6, 7)],
        anchors=[0, 5],
    )
    edge_ids = torch.tensor(
        [
            [-1, -1, -1],
            [0, 1, -1],
            [1, -1, -1],
            [3, 4, -1],
            [5, 6, -1],
        ]
    )
    edge_count = torch.tensor([0, 2, 1, 2, 2])

    assert root_reachable_mask_from_edges(edge_ids=edge_ids, edge_count=edge_count, graph=graph).tolist() == [
        True,
        True,
        False,
        False,
        True,
    ]


def test_root_reachable_mask_matches_reference_dfs() -> None:
    graph = _graph(
        [(0, 1), (1, 2), (2, 3), (0, 4), (4, 3), (5, 6), (6, 5), (3, 7)],
        anchors=[0],
    )
    edge_ids = torch.tensor(
        [
            [-1, -1, -1, -1],
            [0, 1, 2, -1],
            [3, 4, -1, -1],
            [1, 2, -1, -1],
            [5, 6, -1, -1],
            [0, 7, -1, -1],
        ]
    )
    edge_count = torch.tensor([0, 3, 2, 2, 2, 2])

    actual = root_reachable_mask_from_edges(edge_ids=edge_ids, edge_count=edge_count, graph=graph)
    expected = torch.tensor(
        [_reference_root_reachable(edge_ids=row, edge_count=int(count.item()), graph=graph) for row, count in zip(edge_ids, edge_count, strict=True)]
    )

    assert actual.tolist() == expected.tolist()


def _reference_root_reachable(*, edge_ids: torch.Tensor, edge_count: int, graph: GraphContext) -> bool:
    selected = [int(edge_ids[i].item()) for i in range(edge_count)]
    reachable_nodes = set(graph.anchor_mask.nonzero(as_tuple=True)[0].tolist())
    reachable_edges: set[int] = set()
    changed = True
    while changed:
        changed = False
        for edge_id in selected:
            src = int(graph.edge_src[edge_id].item())
            dst = int(graph.edge_dst[edge_id].item())
            if edge_id not in reachable_edges and src in reachable_nodes:
                reachable_edges.add(edge_id)
                reachable_nodes.add(dst)
                changed = True
    return len(reachable_edges) == edge_count


def _graph(edges: list[tuple[int, int]], anchors: list[int] | None = None) -> GraphContext:
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = int(edge_index.max().item()) + 1
    src, dst = edge_index
    edge_ids = torch.arange(len(edges))
    out_order = torch.argsort(src)
    anchors = [0] if anchors is None else anchors
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    anchor_mask[torch.tensor(anchors)] = True
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.zeros(num_nodes, dtype=torch.long),
        edge_to_graph=torch.zeros(len(edges), dtype=torch.long),
        edge_ptr=torch.tensor([0, len(edges)]),
        anchor_mask=anchor_mask,
        anchor_ptr=torch.tensor([0, len(anchors)]),
        anchor_node_ids=torch.tensor(anchors),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=_ptr(torch.bincount(src, minlength=num_nodes)),
            edge_ids_by_src=edge_ids.index_select(0, out_order),
        ),
        num_nodes=num_nodes,
        num_edges=len(edges),
        num_graphs=1,
    )


def _ptr(counts: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.tensor([0]), torch.cumsum(counts, dim=0)])
