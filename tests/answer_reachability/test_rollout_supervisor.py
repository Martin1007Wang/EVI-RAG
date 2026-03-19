from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from src.models.configs import AnswerRewardConfig
from src.models.gflownet import AnswerReachabilityTrajectorySupervisor

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


def test_rollout_supervisor_graph_normalized_failure_reward_multi_graph() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
    )
    batch = SimpleNamespace(
        num_graphs=2,
        node_ptr=torch.tensor([0, 3, 8], dtype=torch.long),
        a_ptr=torch.tensor([0, 1, 3], dtype=torch.long),
    )

    rewards, log_rewards = supervisor.compute_terminal_rewards(
        batch=cast(Any, batch),
        terminal_nodes=torch.tensor([[2, 1], [7, 4]], dtype=torch.long),
        success_mask=torch.tensor([[True, False], [False, True]], dtype=torch.bool),
    )

    assert rewards.shape == (2, 2)
    assert torch.allclose(
        rewards,
        torch.tensor([[1.0, 5.0e-4], [1.0e-3 / 3.0, 1.0]], dtype=torch.float32),
    )
    assert torch.allclose(log_rewards, rewards.log())


def test_rollout_supervisor_binary_ranking_reward_normalizes_entity_multiplicity() -> (
    None
):
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
        answer_reward=AnswerRewardConfig(
            mode="binary_ranking",
            positive_utility=1.0,
            negative_utility=-1.0,
            beta=1.0,
            normalize_by_entity_count=True,
        ),
    )
    batch = SimpleNamespace(
        num_graphs=1,
        node_ptr=torch.tensor([0, 4], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 102], dtype=torch.long),
    )

    rewards, log_rewards = supervisor.compute_terminal_rewards(
        batch=cast(Any, batch),
        terminal_nodes=torch.tensor([[2, 3, 1]], dtype=torch.long),
        success_mask=torch.tensor([[True, False, False]], dtype=torch.bool),
    )

    expected_gold = (1.0e-3 + torch.exp(torch.tensor(1.0))).item() / 2.0
    expected_non_gold = 1.0e-3 + torch.exp(torch.tensor(-1.0)).item()
    assert rewards.shape == (1, 3)
    assert rewards[0, 0].item() == pytest.approx(expected_gold)
    assert rewards[0, 1].item() == pytest.approx(expected_gold)
    assert rewards[0, 2].item() == pytest.approx(expected_non_gold)
    assert torch.allclose(log_rewards, rewards.log())
