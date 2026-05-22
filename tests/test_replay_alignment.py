from __future__ import annotations

from types import SimpleNamespace

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.replay import (
    replay_trajectories_with_stats,
    training_from_trajectories,
)
from src.weaver.rollout.result import RolloutResult
from src.weaver.utility import TrueTerminalReward


def replay_batch():
    return SimpleNamespace(
        edge_index=torch.tensor(
            [
                [0, 0, 1, 2, 3, 3, 4, 5],
                [1, 1, 2, 0, 4, 4, 5, 3],
            ],
            dtype=torch.long,
        ),
        batch=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
        ptr=torch.tensor([0, 3, 6], dtype=torch.long),
        num_nodes_total=6,
        num_edges_total=8,
        num_graphs_total=2,
        anchor_node_ids=torch.tensor([0, 3], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2, 5], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
        node_target_distances_flat=torch.tensor([2, 1, 0, 2, 1, 0], dtype=torch.long),
        node_target_shortest_path_edge_count_flat=torch.tensor(
            [1.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0, 0.0],
            dtype=torch.float32,
        ),
        node_target_shortest_path_edge_mask_flat=torch.tensor(
            [True, True, True, False, True, True, True, False],
            dtype=torch.bool,
        ),
        anchor_node_forward_distances_flat=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long),
    )


def rollout(edge_rows: list[list[int]], terminal_step: int = 3) -> RolloutResult:
    selected = torch.tensor(edge_rows, dtype=torch.long)
    return RolloutResult(
        source_graph_id=torch.arange(selected.size(0), dtype=torch.long),
        selected_edge_ids=selected,
        policy_action_log_prob=torch.zeros_like(selected, dtype=torch.float32),
        behavior_action_log_prob=torch.zeros_like(selected, dtype=torch.float32),
        terminal_step=torch.full((selected.size(0),), int(terminal_step), dtype=torch.long),
        stop_reason=torch.full((selected.size(0),), RolloutResult.POLICY_STOP, dtype=torch.long),
        expand_budget=3,
    )


def test_replay_budget_is_per_eligible_graph() -> None:
    batch = replay_batch()
    context = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=context)
    reward = TrueTerminalReward(edge_cost=0.05, fail_cost=1.0)

    trajectories, stats = replay_trajectories_with_stats(
        batch=batch,
        context=context,
        budget=3,
        max_trajectories_per_graph=2,
        reward_model=reward,
        target_context=target,
    )

    assert stats.eligible_graphs == 2
    assert stats.covered_graphs == 2
    assert len(trajectories) == 4
    assert sum(trajectory.graph_id == 0 for trajectory in trajectories) == 2
    assert sum(trajectory.graph_id == 1 for trajectory in trajectories) == 2


def test_reward_sufficient_rollout_skips_replay_but_low_quality_hit_does_not() -> None:
    batch = replay_batch()
    context = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=context)
    reward = TrueTerminalReward(edge_cost=0.05, fail_cost=1.0)

    optimal_hit = rollout(
        [
            [0, 2, -1, -1],
            [4, 6, -1, -1],
        ],
        terminal_step=2,
    )
    skipped, skipped_stats = replay_trajectories_with_stats(
        batch=batch,
        context=context,
        budget=3,
        max_trajectories_per_graph=2,
        rollouts=[optimal_hit],
        reward_model=reward,
        target_context=target,
    )
    assert skipped == []
    assert skipped_stats.skipped_by_reward == 2

    low_quality_hit = rollout(
        [
            [0, 2, 3, -1],
            [4, 6, 7, -1],
        ],
        terminal_step=3,
    )
    replayed, replayed_stats = replay_trajectories_with_stats(
        batch=batch,
        context=context,
        budget=3,
        max_trajectories_per_graph=2,
        rollouts=[low_quality_hit],
        reward_model=reward,
        target_context=target,
    )
    assert len(replayed) == 4
    assert replayed_stats.skipped_by_reward == 0


def test_replay_transitions_use_current_frontier_semantics() -> None:
    batch = replay_batch()
    context = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=context)
    reward = TrueTerminalReward(edge_cost=0.05, fail_cost=1.0)
    trajectories, _ = replay_trajectories_with_stats(
        batch=batch,
        context=context,
        budget=3,
        max_trajectories_per_graph=2,
        reward_model=reward,
        target_context=target,
    )

    training = training_from_trajectories(
        trajectories=trajectories,
        graph=context,
        budget=3,
    )

    for row, edge_id in enumerate(training.expansions.edge_ids.tolist()):
        frontier = training.expansions.parent.select_rows(
            torch.tensor([row], dtype=torch.long)
        ).frontier(context, expand_budget=3)
        assert int(edge_id) in set(frontier.edge_ids.tolist())
