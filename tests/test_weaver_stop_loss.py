from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.module import backward_action_log_prob
from src.weaver.policy import UniformValidPredecessorBackwardPolicy, valid_predecessor_count
from src.weaver.state import State


def test_uniform_backward_log_prob_counts_valid_predecessors() -> None:
    context = _graph_context()
    state = State(
        graph_ids=torch.zeros(2, dtype=torch.long),
        selected_edge_mask=torch.tensor(
            [
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        ),
        active_node_mask=torch.zeros((2, 4), dtype=torch.bool),
        step=torch.tensor([2, 2], dtype=torch.long),
    )

    assert valid_predecessor_count(
        state=state,
        context=context,
    ).tolist() == [1, 2]

    backward = UniformValidPredecessorBackwardPolicy()
    assert torch.allclose(
        backward.log_prob(
            child_state=state,
            context=context,
            action_edge_ids=torch.tensor([0, 1], dtype=torch.long),
        ),
        torch.tensor(
            [
                0.0,
                -torch.log(torch.tensor(2.0)),
            ]
        ),
    )


def test_backward_action_log_prob_is_zero_for_stop_and_predecessor_based_for_edges() -> None:
    context = _graph_context()
    state = State(
        graph_ids=torch.zeros(2, dtype=torch.long),
        selected_edge_mask=torch.tensor(
            [
                [True, False, True],
                [True, True, False],
            ],
            dtype=torch.bool,
        ),
        active_node_mask=torch.zeros((2, 4), dtype=torch.bool),
        step=torch.tensor([2, 2], dtype=torch.long),
    )

    log_pb = backward_action_log_prob(
        backward_policy=UniformValidPredecessorBackwardPolicy(),
        child_state=state,
        context=context,
        action_edge_ids=torch.tensor([0, -1], dtype=torch.long),
    )

    assert torch.allclose(log_pb, torch.tensor([0.0, 0.0]))


def _graph_context() -> GraphContext:
    edge_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.zeros(4, dtype=torch.long),
        edge_to_graph=torch.zeros(3, dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False, False], dtype=torch.bool),
        adjacency=build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=4,
        ),
        num_nodes=4,
        num_edges=3,
        num_graphs=1,
    )
