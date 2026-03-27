from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from src.models.configs import SubTrajectoryBalanceConfig
from src.models.gflownet import SubTrajectoryBalanceLoss


def _make_sample_batch(**overrides: Any) -> SimpleNamespace:
    sample_batch = {
        "graph_log_z": torch.tensor([0.0], dtype=torch.float32),
        "start_log_probs": torch.tensor([[0.0]], dtype=torch.float32),
        "start_state_log_f": torch.tensor([[0.0]], dtype=torch.float32),
        "log_pf_steps": torch.zeros((1, 1, 2), dtype=torch.float32),
        "log_pb_steps": torch.zeros((1, 1, 2), dtype=torch.float32),
        "next_state_log_f_steps": torch.zeros((1, 1, 2), dtype=torch.float32),
        "terminal_num_steps": torch.tensor([[1]], dtype=torch.long),
        "terminal_log_rewards": torch.tensor([[0.0]], dtype=torch.float32),
        "success_mask": torch.ones((1, 1), dtype=torch.bool),
    }
    sample_batch.update(overrides)
    return SimpleNamespace(**sample_batch)


def test_subtb_loss_zero_for_consistent_move_then_submit_rollout() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([-0.4], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[-0.4]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[-0.3, -0.2]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.tensor([[[-0.7, 0.0]]], dtype=torch.float32),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        termination_action_steps=torch.tensor([[2]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[-0.9]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert torch.allclose(loss_output.loss, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(loss_output.subtb_loss, torch.tensor(0.0), atol=1.0e-6)


def test_subtb_loss_handles_zero_move_rollout_with_finite_anchor() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([-0.4], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[-0.4]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[0]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[-0.4]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert torch.isfinite(loss_output.loss)
    assert torch.allclose(loss_output.loss, torch.tensor(0.0), atol=1.0e-6)


def test_subtb_loss_reports_log_z_statistics() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([1.0, 3.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[-0.5], [-0.5]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.5], [2.5]], dtype=torch.float32),
        log_pb_steps=torch.zeros((2, 1, 2), dtype=torch.float32),
        log_pf_steps=torch.tensor(
            [[[-0.25, 0.0]], [[-0.25, 0.0]]], dtype=torch.float32
        ),
        next_state_log_f_steps=torch.tensor(
            [[[0.25, 0.0]], [[2.25, 0.0]]], dtype=torch.float32
        ),
        terminal_num_steps=torch.tensor([[1], [1]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[0.25], [2.25]], dtype=torch.float32),
        success_mask=torch.ones((2, 1), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert torch.allclose(loss_output.loss, torch.tensor(0.0), atol=1.0e-6)
    assert loss_output.log_z_mean.item() == pytest.approx(2.0)
    assert loss_output.log_z_variance.item() == pytest.approx(1.0)


def test_subtb_loss_matches_ms_subtrajectory_objective() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([1.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[1.0]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        log_pb_steps=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        next_state_log_f_steps=torch.tensor([[[2.0, 7.0]]], dtype=torch.float32),
        terminal_num_steps=torch.tensor([[2]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[3.0]], dtype=torch.float32),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert loss_output.root_component_loss.item() == pytest.approx(0.0)
    assert loss_output.pairwise_component_loss.item() == pytest.approx(1.0)
    assert loss_output.terminal_component_loss.item() == pytest.approx(2.5)
    assert loss_output.loss.item() == pytest.approx(3.5)


def test_subtb_loss_applies_component_weights_after_ms_component_means() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(
            lambda_weight=1.0,
            normalize=True,
            root_loss_weight=2.0,
            pairwise_loss_weight=3.0,
            terminal_loss_weight=4.0,
        )
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([1.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[1.0]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        log_pb_steps=torch.tensor([[[0.0, 0.0]]], dtype=torch.float32),
        next_state_log_f_steps=torch.tensor([[[2.0, 7.0]]], dtype=torch.float32),
        terminal_num_steps=torch.tensor([[2]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[3.0]], dtype=torch.float32),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert loss_output.root_component_loss.item() == pytest.approx(0.0)
    assert loss_output.pairwise_component_loss.item() == pytest.approx(3.0)
    assert loss_output.terminal_component_loss.item() == pytest.approx(10.0)
    assert loss_output.loss.item() == pytest.approx(13.0)


def test_subtb_loss_keeps_terminal_components_per_rollout_before_batch_mean() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([0.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.0, 0.0]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 2, 2), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 2, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 2, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[1, 1]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[0.0, 2.0]], dtype=torch.float32),
        success_mask=torch.ones((1, 2), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert loss_output.root_component_loss.item() == pytest.approx(0.0)
    assert loss_output.pairwise_component_loss.item() == pytest.approx(0.0)
    assert loss_output.terminal_component_loss.item() == pytest.approx(2.0)
    assert loss_output.loss.item() == pytest.approx(2.0)


def test_subtb_loss_backward_is_autograd_safe() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    graph_log_z = torch.tensor([0.0], dtype=torch.float32, requires_grad=True)
    start_log_probs = torch.tensor([[0.5]], dtype=torch.float32, requires_grad=True)
    start_state_log_f = torch.tensor([[1.0]], dtype=torch.float32, requires_grad=True)
    log_pf_steps = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    next_state_log_f_steps = torch.tensor(
        [[[2.0, 7.0]]], dtype=torch.float32, requires_grad=True
    )
    terminal_log_rewards = torch.tensor(
        [[3.0]], dtype=torch.float32, requires_grad=True
    )
    sample_batch = _make_sample_batch(
        graph_log_z=graph_log_z,
        start_log_probs=start_log_probs,
        start_state_log_f=start_state_log_f,
        log_pf_steps=log_pf_steps,
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=next_state_log_f_steps,
        terminal_num_steps=torch.tensor([[2]], dtype=torch.long),
        terminal_log_rewards=terminal_log_rewards,
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))
    loss_output.loss.backward()

    assert graph_log_z.grad is not None and torch.isfinite(graph_log_z.grad).all()
    assert (
        start_log_probs.grad is not None and torch.isfinite(start_log_probs.grad).all()
    )
    assert (
        start_state_log_f.grad is not None
        and torch.isfinite(start_state_log_f.grad).all()
    )
    assert log_pf_steps.grad is not None and torch.isfinite(log_pf_steps.grad).all()
    assert (
        next_state_log_f_steps.grad is not None
        and torch.isfinite(next_state_log_f_steps.grad).all()
    )
    assert (
        terminal_log_rewards.grad is not None
        and torch.isfinite(terminal_log_rewards.grad).all()
    )


def test_subtb_root_metric_tracks_explicit_root_boundary_residual() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([0.2], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.1]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.5]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[0]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[0.5]], dtype=torch.float32),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert loss_output.root_abs.item() == pytest.approx(0.2)


def test_subtb_root_metric_subtracts_start_log_rewards() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = _make_sample_batch(
        graph_log_z=torch.tensor([0.2], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.1]], dtype=torch.float32),
        start_log_rewards=torch.tensor([[0.3]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.0]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[0]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[0.0]], dtype=torch.float32),
    )

    loss_output = loss_fn.compute(cast(Any, sample_batch))

    assert loss_output.root_abs.item() == pytest.approx(0.0)
    assert loss_output.root_component_loss.item() == pytest.approx(0.0)


def test_subtb_step_level_log_rewards_match_equivalent_terminal_bonus() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    terminal_bonus_batch = _make_sample_batch(
        graph_log_z=torch.tensor([0.4], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.4]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[-0.3, -0.2]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.tensor([[[0.1, 0.0]]], dtype=torch.float32),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        termination_action_steps=torch.tensor([[2]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[-0.1]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    process_reward_batch = _make_sample_batch(
        graph_log_z=torch.tensor([0.4], dtype=torch.float32),
        start_log_probs=torch.tensor([[0.0]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[0.4]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[-0.3, -0.2]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.tensor([[[-0.15, 0.0]]], dtype=torch.float32),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        termination_action_steps=torch.tensor([[2]], dtype=torch.long),
        log_reward_steps=torch.tensor([[[0.25, 0.0]]], dtype=torch.float32),
        terminal_log_rewards=torch.tensor([[-0.35]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    process_loss = loss_fn.compute(cast(Any, process_reward_batch))
    terminal_bonus_loss = loss_fn.compute(cast(Any, terminal_bonus_batch))

    assert process_loss.loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert terminal_bonus_loss.loss.item() == pytest.approx(0.0, abs=1.0e-6)
