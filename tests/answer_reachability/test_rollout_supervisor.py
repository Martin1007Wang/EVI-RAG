from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from src.models.gflownet import AnswerReachabilityTrajectorySupervisor

from .conftest import make_toy_batch


_NON_GOLD_LOG_REWARD = -3.0
_NON_GOLD_REWARD = torch.exp(torch.tensor(_NON_GOLD_LOG_REWARD)).item()


def test_rollout_supervisor_marks_answer_nodes_as_terminal_targets() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor()

    terminal_target_mask = supervisor.build_terminal_target_mask(batch=make_toy_batch())

    assert terminal_target_mask.dtype == torch.bool
    assert terminal_target_mask.tolist() == [False, False, True]


def test_rollout_supervisor_uses_fixed_energy_rewards() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor()
    batch = make_toy_batch()

    rewards, log_rewards = supervisor.compute_terminal_rewards(
        batch=batch,
        terminal_nodes=torch.tensor([[2, 1]], dtype=torch.long),
    )

    assert rewards.shape == (1, 2)
    assert rewards[0, 0].item() == pytest.approx(1.0)
    assert rewards[0, 1].item() == pytest.approx(_NON_GOLD_REWARD)
    assert torch.allclose(
        log_rewards,
        torch.tensor([[0.0, _NON_GOLD_LOG_REWARD]], dtype=torch.float32),
    )


def test_rollout_supervisor_uses_fixed_energy_rewards_multi_graph() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor()
    batch = SimpleNamespace(
        num_graphs=2,
        node_ptr=torch.tensor([0, 3, 8], dtype=torch.long),
        a_ptr=torch.tensor([0, 1, 3], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1, 3], dtype=torch.long),
        answer_entity_ids=torch.tensor([102, 204, 207], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 200, 201, 204, 205, 207]),
    )

    rewards, log_rewards = supervisor.compute_terminal_rewards(
        batch=cast(Any, batch),
        terminal_nodes=torch.tensor([[2, 1], [7, 4]], dtype=torch.long),
    )

    assert rewards.shape == (2, 2)
    assert torch.allclose(
        rewards,
        torch.tensor(
            [[1.0, _NON_GOLD_REWARD], [1.0, _NON_GOLD_REWARD]],
            dtype=torch.float32,
        ),
    )
    assert torch.allclose(
        log_rewards,
        torch.tensor(
            [[0.0, _NON_GOLD_LOG_REWARD], [0.0, _NON_GOLD_LOG_REWARD]],
            dtype=torch.float32,
        ),
    )


def test_energy_reward_keeps_aliases_at_same_terminal_mass() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor()
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
    )

    assert rewards.shape == (1, 3)
    assert rewards[0, 0].item() == pytest.approx(1.0)
    assert rewards[0, 1].item() == pytest.approx(1.0)
    assert rewards[0, 2].item() == pytest.approx(_NON_GOLD_REWARD)
    assert torch.allclose(
        log_rewards,
        torch.tensor([[0.0, 0.0, _NON_GOLD_LOG_REWARD]], dtype=torch.float32),
    )


def test_rollout_supervisor_assigns_zero_terminal_backward_for_energy_reward() -> None:
    supervisor = AnswerReachabilityTrajectorySupervisor()
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
    )

    assert torch.allclose(
        terminal_transition.terminal_rewards,
        torch.tensor([[1.0, _NON_GOLD_REWARD]], dtype=torch.float32),
    )
    assert torch.allclose(
        terminal_transition.terminal_log_rewards,
        torch.tensor([[0.0, _NON_GOLD_LOG_REWARD]], dtype=torch.float32),
    )
    assert terminal_transition.terminal_backward_log_probs.tolist() == [[0.0, 0.0]]
