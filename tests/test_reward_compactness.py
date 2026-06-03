from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.reward import EvidenceSubgraphReward
from src.weaver.state import StateBatch


def test_reward_uses_canonical_state_from_selected_edges() -> None:
    graph = _graph()
    target = _target()

    state = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
        edge_count=torch.tensor([2], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    reward = EvidenceSubgraphReward(
        answer_weight=4.0,
        coverage_weight=0.0,
        edge_cost=0.5,
        fail_cost=0.0,
        answer_prize=1.0,
    )(state=state, target_context=target, graph_context=graph)

    assert reward.edge_count.tolist() == [2.0]
    assert reward.target_recall.tolist() == [1.0]
    assert reward.log_reward.tolist() == [3.0]


def test_reward_handles_missing_targets_without_nan() -> None:
    graph = _graph()
    target = _empty_target()

    state = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, -1]], dtype=torch.long),
        edge_count=torch.tensor([1], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    reward = EvidenceSubgraphReward(
        answer_weight=4.0,
        coverage_weight=0.0,
        edge_cost=0.25,
        fail_cost=2.0,
        answer_prize=1.0,
    )(state=state, target_context=target, graph_context=graph)

    assert reward.valid_mask.tolist() == [False]
    assert reward.success_mask.tolist() == [False]
    assert reward.target_recall.tolist() == [0.0]
    assert reward.metrics["reward/valid_rate"].item() == 0.0


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 2], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 2, 2], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 1], dtype=torch.long),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, True, False]),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        target_count_by_graph=torch.tensor([1], dtype=torch.long),
        node_target_distance=torch.tensor([1, 0, -1], dtype=torch.long),
    )


def _empty_target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, False, False]),
        reachable_target_node_ids=torch.empty(0, dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 0], dtype=torch.long),
        target_count_by_graph=torch.tensor([0], dtype=torch.long),
        node_target_distance=torch.tensor([-1, -1, -1], dtype=torch.long),
    )
