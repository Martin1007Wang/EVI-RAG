from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.reward import EvidenceSubgraphReward, compute_zero_gain_edge_count
from src.weaver.state import StateBatch


def test_zero_gain_edge_count_is_order_invariant() -> None:
    graph = _graph()
    target = _target()
    state = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]]),
        edge_count=torch.tensor([2]),
        budget=2,
    )

    zero_gain = compute_zero_gain_edge_count(
        state=state,
        graph=graph,
        target_mask=target.target_mask,
    )

    reversed_state = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[1, 0]]),
        edge_count=torch.tensor([2]),
        budget=2,
    )
    reversed_zero_gain = compute_zero_gain_edge_count(
        state=reversed_state,
        graph=graph,
        target_mask=target.target_mask,
    )

    assert torch.allclose(zero_gain, torch.tensor([1.0]))
    assert torch.allclose(reversed_zero_gain, torch.tensor([1.0]))


def test_redundant_edge_cost_penalizes_zero_gain_edges() -> None:
    graph = _graph()
    target = _target()
    state = StateBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]]),
        edge_count=torch.tensor([2]),
        budget=2,
    )

    base = EvidenceSubgraphReward(
        answer_weight=4.0,
        proximity_weight=0.0,
        path_weight=0.0,
        fail_proximity_weight=0.0,
        fail_path_weight=0.0,
        edge_cost=0.0,
        redundant_edge_cost=0.0,
    )(state=state, graph_context=graph, target_context=target)
    compact = EvidenceSubgraphReward(
        answer_weight=4.0,
        proximity_weight=0.0,
        path_weight=0.0,
        fail_proximity_weight=0.0,
        fail_path_weight=0.0,
        edge_cost=0.0,
        redundant_edge_cost=0.7,
    )(state=state, graph_context=graph, target_context=target)

    assert torch.allclose(base.log_reward - compact.log_reward, torch.tensor([0.7]))
    assert torch.allclose(compact.metrics["reward/zero_gain_edge_count_mean"], torch.tensor(1.0))


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0]),
        edge_to_graph=torch.tensor([0, 0]),
        edge_ptr=torch.tensor([0, 2]),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 2, 2]),
            edge_ids_by_src=torch.tensor([0, 1]),
            in_ptr=torch.tensor([0, 0, 1, 2]),
            edge_ids_by_dst=torch.tensor([0, 1]),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, True, False]),
        reachable_target_node_ids=torch.tensor([1]),
        reachable_target_node_ids_ptr=torch.tensor([0, 1]),
        target_count_by_graph=torch.tensor([1]),
        node_target_distance=torch.tensor([1, 0, -1]),
        shortest_path_edge_mask=torch.tensor([True, False]),
        shortest_path_edge_weight=torch.tensor([1.0, 0.0]),
    )
