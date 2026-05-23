from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.policy.backward import UniformValidPredecessorBackwardPolicy, valid_predecessor_count
from src.weaver.state import State


def chain_graph() -> GraphContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        adjacency=build_directed_adjacency_index(edge_index=edge_index, num_nodes=3),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def test_backward_policy_uses_rollout_budget_for_deep_child_state() -> None:
    graph = chain_graph()
    parent = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    ).expand(
        graph=graph,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    )
    child = parent.expand(
        graph=graph,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([1], dtype=torch.long),
        expand_budget=2,
    )

    counts = valid_predecessor_count(
        state=child,
        context=graph,
        expand_budget=2,
    )
    assert torch.equal(counts, torch.tensor([1], dtype=torch.long))

    log_prob = UniformValidPredecessorBackwardPolicy().log_prob(
        child_state=child,
        context=graph,
        action_edge_ids=torch.tensor([1], dtype=torch.long),
        expand_budget=2,
    )
    assert torch.equal(log_prob, torch.tensor([-0.0]))


def test_backward_policy_rejects_child_beyond_rollout_budget() -> None:
    graph = chain_graph()
    child = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    ).expand(
        graph=graph,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    ).expand(
        graph=graph,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([1], dtype=torch.long),
        expand_budget=2,
    )

    policy = UniformValidPredecessorBackwardPolicy()
    try:
        policy.log_prob(
            child_state=child,
            context=graph,
            action_edge_ids=torch.tensor([1], dtype=torch.long),
            expand_budget=1,
        )
    except ValueError as exc:
        assert str(exc) == "Expansion child state has no exact forward predecessor."
    else:
        raise AssertionError("Expected invalid over-budget child state to be rejected.")
