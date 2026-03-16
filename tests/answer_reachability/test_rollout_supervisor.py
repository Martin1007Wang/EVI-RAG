from __future__ import annotations

import pytest
import torch

from src.metrics.answer_reachability import AnswerReachabilityTrajectorySupervisor

from .conftest import make_toy_batch


def test_rollout_supervisor_marks_answer_nodes_as_terminal_targets() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
    )

    terminal_target_mask = supervisor.build_terminal_target_mask(batch=make_toy_batch())

    assert terminal_target_mask.dtype == torch.bool
    assert terminal_target_mask.tolist() == [False, False, True]


def test_rollout_supervisor_graph_normalized_failure_reward() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
    )
    batch = make_toy_batch()

    rewards, log_rewards = supervisor.compute_terminal_rewards(
        batch=batch,
        terminal_nodes=torch.tensor([[2, 1]], dtype=torch.long),
        success_mask=torch.tensor([[True, False]], dtype=torch.bool),
    )

    assert rewards.shape == (1, 2)
    assert rewards[0, 0].item() == pytest.approx(1.0)
    assert rewards[0, 1].item() == pytest.approx(5.0e-4)
    assert torch.allclose(log_rewards, rewards.log())
