from __future__ import annotations

import torch

from src.eval.diversity import terminal_f1_std, unique_selected_edge_set_rate
from src.weaver.rollout.trajectory import BUDGET, POLICY_STOP, SRC_POLICY, TrajectoryBatch


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
        stop_reason=torch.tensor([BUDGET, BUDGET]),
        stop_logp=torch.zeros(2),
        source=torch.full((2,), SRC_POLICY, dtype=torch.long),
    )

    assert unique_selected_edge_set_rate(rollouts) == 0.5


def test_terminal_f1_std_uses_stop_reason_mask() -> None:
    rollouts = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.full((3, 1), -1, dtype=torch.long),
        edge_logp=torch.zeros((3, 1)),
        edge_count=torch.zeros(3, dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, BUDGET, POLICY_STOP]),
        stop_logp=torch.tensor([1.0, 100.0, 3.0]),
        source=torch.full((3,), SRC_POLICY, dtype=torch.long),
    )

    assert terminal_f1_std(rollouts) == 1.0
