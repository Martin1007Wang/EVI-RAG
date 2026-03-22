from __future__ import annotations

import math
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


def test_answer_ranking_reward_keeps_aliases_at_same_terminal_mass() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
        answer_reward=AnswerRewardConfig(
            mode="binary_ranking",
            positive_utility=1.0,
            negative_utility=-1.0,
            beta=1.0,
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
        success_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
    )

    expected_gold = 1.0e-3 + torch.exp(torch.tensor(1.0)).item()
    expected_non_gold = 1.0e-3 + torch.exp(torch.tensor(-1.0)).item()
    assert rewards.shape == (1, 3)
    assert rewards[0, 0].item() == pytest.approx(expected_gold)
    assert rewards[0, 1].item() == pytest.approx(expected_gold)
    assert rewards[0, 2].item() == pytest.approx(expected_non_gold)
    assert torch.allclose(log_rewards, rewards.log())


def test_rollout_supervisor_uses_asymmetric_length_penalty_only_for_failures() -> None:
    alpha = 0.2
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
        answer_reward=AnswerRewardConfig(
            mode="entity_sink",
            positive_utility=1.0,
            negative_utility=-1.0,
            beta=1.0,
            failure_length_penalty_alpha=alpha,
        ),
    )
    batch = SimpleNamespace(
        num_graphs=1,
        node_ptr=torch.tensor([0, 4], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
    )

    terminal_transition = supervisor.resolve_terminal_transitions(
        batch=cast(Any, batch),
        terminal_nodes=torch.tensor([[2, 3]], dtype=torch.long),
        success_mask=torch.tensor([[True, False]], dtype=torch.bool),
        terminal_num_steps=torch.tensor([[3, 3]], dtype=torch.long),
    )

    base_reward = 1.0e-3 + torch.exp(torch.tensor(1.0)).item()
    failure_reward = 1.0e-3 + torch.exp(torch.tensor(-1.0)).item()
    assert terminal_transition.terminal_rewards[0, 0].item() == pytest.approx(
        base_reward
    )
    assert terminal_transition.terminal_rewards[0, 1].item() == pytest.approx(
        failure_reward * math.exp(-3.0 * alpha)
    )
    assert terminal_transition.terminal_backward_log_probs.tolist() == [[0.0, 0.0]]


def test_rollout_supervisor_penalizes_cycles_for_success_and_failure() -> None:
    cycle_penalty = 0.5
    supervisor = AnswerReachabilityTrajectorySupervisor(
        epsilon=1.0e-3,
        failure_reward_mode="graph_normalized",
        answer_reward=AnswerRewardConfig(
            mode="entity_sink",
            positive_utility=1.0,
            negative_utility=-1.0,
            beta=1.0,
            cycle_penalty=cycle_penalty,
        ),
    )
    batch = SimpleNamespace(
        num_graphs=1,
        node_ptr=torch.tensor([0, 4], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 103], dtype=torch.long),
    )

    terminal_transition = supervisor.resolve_terminal_transitions(
        batch=cast(Any, batch),
        terminal_nodes=torch.tensor([[2, 3]], dtype=torch.long),
        success_mask=torch.tensor([[True, False]], dtype=torch.bool),
        terminal_cycle_counts=torch.tensor([[2, 1]], dtype=torch.long),
    )

    expected_gold = (1.0e-3 + torch.exp(torch.tensor(1.0)).item()) * (cycle_penalty**2)
    expected_fail = (1.0e-3 + torch.exp(torch.tensor(-1.0)).item()) * cycle_penalty
    assert terminal_transition.terminal_rewards[0, 0].item() == pytest.approx(
        expected_gold
    )
    assert terminal_transition.terminal_rewards[0, 1].item() == pytest.approx(
        expected_fail
    )


def test_answer_reward_config_maps_legacy_length_penalty_alias() -> None:
    cfg = AnswerRewardConfig(length_penalty_alpha=0.3)

    assert cfg.failure_length_penalty_alpha == pytest.approx(0.3)


def test_answer_reward_config_rejects_negative_failure_length_penalty() -> None:
    with pytest.raises(ValueError, match="failure_length_penalty_alpha must be >= 0"):
        AnswerRewardConfig(failure_length_penalty_alpha=-0.1)


def test_answer_reward_config_rejects_invalid_cycle_penalty() -> None:
    with pytest.raises(ValueError, match="cycle_penalty must be in \(0, 1\]"):
        AnswerRewardConfig(cycle_penalty=1.5)
