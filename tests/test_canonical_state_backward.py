from __future__ import annotations

import math

import pytest
import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.objectives.subtb import SubTBEventBatch, expansion_db_residual
from src.weaver.policy.backward import canonical_backward_log_prob, valid_predecessor_count
from src.weaver.state import ExpansionBatch, StateBatch, legal_state_mask


def test_state_advance_canonicalizes_edge_order() -> None:
    graph = _graph([(0, 1), (0, 2)])
    del graph

    root = StateBatch.initial(graph_ids=torch.tensor([0, 0]), budget=2)
    first = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0, 1]),
            edge_ids=torch.tensor([0, 1]),
        )
    )
    second = first.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0, 1]),
            edge_ids=torch.tensor([1, 0]),
        )
    )

    assert second.edge_count.tolist() == [2, 2]
    assert second.edge_ids.tolist() == [[0, 1], [0, 1]]


def test_state_advance_rejects_duplicate_edge() -> None:
    root = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2)
    state = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0]),
            edge_ids=torch.tensor([0]),
        )
    )

    with pytest.raises(ValueError, match="duplicate selected edges"):
        state.advance(
            ExpansionBatch(
                state_ids=torch.tensor([0]),
                edge_ids=torch.tensor([0]),
            )
        )


def test_legal_state_requires_anchor_reachability() -> None:
    graph = _graph([(0, 1), (1, 2), (3, 4)])
    state = StateBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor(
            [
                [0, 1],
                [1, -1],
                [2, -1],
            ],
            dtype=torch.long,
        ),
        edge_count=torch.tensor([2, 1, 1]),
        budget=2,
    )

    assert legal_state_mask(state=state, graph=graph).tolist() == [True, False, False]


def test_backward_counts_independent_valid_parents() -> None:
    graph = _graph([(0, 1), (0, 2)])
    parent = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, -1]], dtype=torch.long),
        edge_count=torch.tensor([1]),
        budget=2,
    )
    child = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        edge_count=torch.tensor([2]),
        budget=2,
    )

    log_prob = canonical_backward_log_prob(
        parent_state=parent,
        child_state=child,
        action_edge_ids=torch.tensor([1]),
        graph_context=graph,
        validate=True,
    )

    assert valid_predecessor_count(child_state=child, graph_context=graph, row=0) == 2
    assert torch.allclose(log_prob, torch.tensor([-math.log(2.0)]))


def test_expansion_db_residual_uses_multi_parent_backward_term() -> None:
    log_pb = torch.tensor([-math.log(2.0)])
    child_flow = torch.tensor([5.0])
    events = SubTBEventBatch(
        trajectory_ids=torch.tensor([0]),
        step_ids=torch.tensor([0]),
        source_ids=torch.tensor([0]),
        parent_state_log_flow=torch.tensor([0.0]),
        child_state_log_flow=child_flow,
        action_log_flow=child_flow + log_pb,
        backward_log_prob=log_pb,
        terminal_log_reward=torch.tensor([0.0]),
        terminal_reason=torch.tensor([-1]),
        is_terminal=torch.tensor([False]),
    )

    assert torch.allclose(expansion_db_residual(events), torch.zeros(1))


def test_backward_excludes_illegal_disconnected_parent() -> None:
    graph = _graph([(0, 1), (1, 2)])
    parent = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, -1]], dtype=torch.long),
        edge_count=torch.tensor([1]),
        budget=2,
    )
    child = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        edge_count=torch.tensor([2]),
        budget=2,
    )

    log_prob = canonical_backward_log_prob(
        parent_state=parent,
        child_state=child,
        action_edge_ids=torch.tensor([1]),
        graph_context=graph,
        validate=True,
    )

    assert valid_predecessor_count(child_state=child, graph_context=graph, row=0) == 1
    assert torch.allclose(log_prob, torch.tensor([0.0]))


def test_backward_validation_rejects_wrong_parent() -> None:
    graph = _graph([(0, 1), (0, 2)])
    wrong_parent = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[1, -1]], dtype=torch.long),
        edge_count=torch.tensor([1]),
        budget=2,
    )
    child = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        edge_count=torch.tensor([2]),
        budget=2,
    )

    with pytest.raises(ValueError, match="child minus action edge"):
        canonical_backward_log_prob(
            parent_state=wrong_parent,
            child_state=child,
            action_edge_ids=torch.tensor([1]),
            graph_context=graph,
            validate=True,
        )


def _graph(edges: list[tuple[int, int]]) -> GraphContext:
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = int(edge_index.max().item()) + 1
    edge_ids = torch.arange(len(edges), dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]

    out_order = torch.argsort(src)
    in_order = torch.argsort(dst)
    out_counts = torch.bincount(src, minlength=num_nodes)
    in_counts = torch.bincount(dst, minlength=num_nodes)

    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.zeros(num_nodes, dtype=torch.long),
        edge_to_graph=torch.zeros(len(edges), dtype=torch.long),
        edge_ptr=torch.tensor([0, len(edges)], dtype=torch.long),
        anchor_mask=torch.tensor([True] + [False] * (num_nodes - 1)),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=_ptr(out_counts),
            edge_ids_by_src=edge_ids.index_select(0, out_order),
            in_ptr=_ptr(in_counts),
            edge_ids_by_dst=edge_ids.index_select(0, in_order),
        ),
        num_nodes=num_nodes,
        num_edges=len(edges),
        num_graphs=1,
    )


def _ptr(counts: torch.Tensor) -> torch.Tensor:
    ptr = torch.empty(int(counts.numel()) + 1, dtype=torch.long)
    ptr[0] = 0
    ptr[1:] = torch.cumsum(counts, dim=0)
    return ptr
