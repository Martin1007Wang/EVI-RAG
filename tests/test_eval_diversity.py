from __future__ import annotations

import torch

from src.eval.diversity import terminal_f1_std, unique_selected_edge_set_rate
from src.weaver.rollout.trajectory import BUDGET, POLICY_STOP, TrajectoryBatch


def test_unique_selected_edge_set_rate_collapses_permutations() -> None:
    rollouts = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0]),
        edge_ids=torch.tensor(
            [
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        edge_logp=torch.zeros((2, 2)),
        edge_count=torch.tensor([2, 2]),
        stop_reason=torch.tensor([BUDGET, BUDGET], dtype=torch.uint8),
        stop_logp=torch.zeros(2),
        source=torch.zeros(2, dtype=torch.bool),
    )

    assert unique_selected_edge_set_rate(rollouts) == 0.5


def test_terminal_f1_std_uses_stop_reason_mask() -> None:
    rollouts = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.full((3, 1), -1, dtype=torch.long),
        edge_logp=torch.zeros((3, 1)),
        edge_count=torch.zeros(3, dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, BUDGET, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.tensor([1.0, 100.0, 3.0]),
        source=torch.zeros(3, dtype=torch.bool),
    )

    assert terminal_f1_std(rollouts) == 1.0


def test_trajectory_stop_reason_masks_split_policy_and_forced_terminals() -> None:
    rollouts = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.full((3, 1), -1, dtype=torch.long),
        edge_logp=torch.zeros((3, 1)),
        edge_count=torch.zeros(3, dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, BUDGET, BUDGET], dtype=torch.uint8),
        stop_logp=torch.tensor([1.0, 0.0, 0.0]),
        source=torch.zeros(3, dtype=torch.bool),
    )

    assert rollouts.is_policy_stop.tolist() == [True, False, False]
    assert rollouts.is_budget_boundary.tolist() == [False, True, True]
    assert rollouts.is_forced_terminal.tolist() == [False, True, True]
