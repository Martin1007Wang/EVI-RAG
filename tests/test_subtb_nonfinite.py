from __future__ import annotations

import pytest
import torch

from src.models.components.rollout_types import RolloutResult
from src.models.configs.objective import SubTBConfig
from src.models.metrics.subtb_loss import SubTrajectoryBalanceLoss


def _make_rollout(
    *,
    log_pf_steps: torch.Tensor,
    log_pb_steps: torch.Tensor,
    log_f_steps: torch.Tensor,
    num_moves: torch.Tensor,
    num_steps: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> RolloutResult:
    if num_steps is None:
        num_steps = num_moves
    device = num_moves.device
    batch_shape = tuple(num_moves.shape)
    return RolloutResult(
        log_pf_sum=torch.zeros(batch_shape, dtype=torch.float32, device=device),
        stop_nodes=torch.zeros(batch_shape, dtype=torch.long, device=device),
        num_moves=num_moves,
        num_steps=num_steps,
        stop_reason=torch.zeros(batch_shape, dtype=torch.long, device=device),
        log_pf_steps=log_pf_steps,
        log_pb_steps=log_pb_steps,
        log_f_steps=log_f_steps,
        valid_mask=valid_mask,
    )


def test_subtb_loss_masks_invalid_graph_nonfinite_values() -> None:
    loss_fn = SubTrajectoryBalanceLoss(config=SubTBConfig())
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.tensor([[[0.0, 0.0]], [[float("inf"), 0.0]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((2, 1, 2), dtype=torch.float32),
        log_f_steps=torch.tensor([[[0.0, 0.0]], [[float("inf"), 0.0]]], dtype=torch.float32),
        num_moves=torch.tensor([[1], [0]], dtype=torch.long),
        num_steps=torch.tensor([[1], [0]], dtype=torch.long),
        valid_mask=torch.tensor([[True], [False]], dtype=torch.bool),
    )
    rewards = torch.ones((2, 1), dtype=torch.float32)

    loss, metrics = loss_fn(fwd_rollout=fwd_rollout, rewards=rewards)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["subtb/loss"])


def test_subtb_loss_raises_when_valid_graph_is_nonfinite() -> None:
    loss_fn = SubTrajectoryBalanceLoss(config=SubTBConfig())
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.tensor([[[float("inf"), 0.0]]], dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        num_moves=torch.tensor([[2]], dtype=torch.long),
    )
    rewards = torch.ones((1, 1), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Non-finite loss detected in SubTB"):
        _ = loss_fn(fwd_rollout=fwd_rollout, rewards=rewards)


def test_subtb_miss_length_penalty_requires_explicit_hit_mask() -> None:
    loss_fn = SubTrajectoryBalanceLoss(config=SubTBConfig(miss_length_penalty=0.1))
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        num_moves=torch.tensor([[1]], dtype=torch.long),
    )
    rewards = torch.full((1, 1), 0.5, dtype=torch.float32)

    with pytest.raises(ValueError, match="requires explicit hit_mask"):
        _ = loss_fn(fwd_rollout=fwd_rollout, rewards=rewards, reward_beta=2.0)


def test_subtb_subtrajectory_loss_detects_mid_trajectory_drift() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTBConfig(
            lambda_weight=0.9,
            normalize=True,
            detach_end_flow=True,
            boundary_weight=0.0,
            miss_length_penalty=0.0,
        )
    )
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.zeros((1, 2, 2), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 2, 2), dtype=torch.float32),
        log_f_steps=torch.tensor([[[0.0, 2.0], [1.0, 1.0]]], dtype=torch.float32),
        num_moves=torch.tensor([[2, 2]], dtype=torch.long),
    )
    rewards = torch.ones((1, 2), dtype=torch.float32)
    hit_mask = torch.ones((1, 2), dtype=torch.bool)

    loss, metrics = loss_fn(
        fwd_rollout=fwd_rollout,
        rewards=rewards,
        reward_beta=1.0,
        hit_mask=hit_mask,
    )

    assert torch.isfinite(loss)
    assert metrics["subtb/var_loss"] > 0
    assert metrics["subtb/subtraj_residual_abs"] > 0


def test_subtb_reward_is_aligned_to_true_terminal_index_not_horizon_tail() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTBConfig(
            lambda_weight=1.0,
            normalize=False,
            detach_end_flow=False,
            boundary_weight=0.0,
            miss_length_penalty=0.0,
        )
    )
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_f_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        num_moves=torch.tensor([[2]], dtype=torch.long),  # early stop: L < max_steps
    )
    hit_mask = torch.ones((1, 1), dtype=torch.bool)

    loss_hit, _ = loss_fn(
        fwd_rollout=fwd_rollout,
        rewards=torch.tensor([[1.0]], dtype=torch.float32),
        reward_beta=1.0,
        hit_mask=hit_mask,
    )
    loss_miss, _ = loss_fn(
        fwd_rollout=fwd_rollout,
        rewards=torch.tensor([[1.0e-4]], dtype=torch.float32),
        reward_beta=1.0,
        hit_mask=hit_mask,
    )

    assert torch.isfinite(loss_hit)
    assert torch.isfinite(loss_miss)
    assert float(loss_miss.item()) > float(loss_hit.item())


def test_subtb_includes_explicit_stop_step_in_delta_prefix() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTBConfig(
            lambda_weight=1.0,
            normalize=False,
            detach_end_flow=False,
            boundary_weight=0.0,
            miss_length_penalty=0.0,
        )
    )
    log_pf_steps = torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True)
    fwd_rollout = _make_rollout(
        log_pf_steps=log_pf_steps,
        log_pb_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_f_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        num_moves=torch.tensor([[2]], dtype=torch.long),
        num_steps=torch.tensor([[3]], dtype=torch.long),  # includes stop decision at step index 2
    )
    loss, _ = loss_fn(
        fwd_rollout=fwd_rollout,
        rewards=torch.tensor([[1.0e-4]], dtype=torch.float32),
        reward_beta=1.0,
        hit_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert log_pf_steps.grad is not None
    assert float(log_pf_steps.grad[0, 0, 2].item()) != 0.0


def test_subtb_zero_step_rollout_still_anchors_start_flow_to_reward() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTBConfig(
            lambda_weight=1.0,
            normalize=False,
            detach_end_flow=False,
            boundary_weight=0.0,
            miss_length_penalty=0.0,
        )
    )
    log_f_steps = torch.zeros((1, 1, 4), dtype=torch.float32, requires_grad=True)
    fwd_rollout = _make_rollout(
        log_pf_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_pb_steps=torch.zeros((1, 1, 4), dtype=torch.float32),
        log_f_steps=log_f_steps,
        num_moves=torch.tensor([[0]], dtype=torch.long),
        num_steps=torch.tensor([[0]], dtype=torch.long),
    )
    loss, _ = loss_fn(
        fwd_rollout=fwd_rollout,
        rewards=torch.tensor([[1.0e-4]], dtype=torch.float32),
        reward_beta=1.0,
        hit_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert log_f_steps.grad is not None
    assert float(log_f_steps.grad[0, 0, 0].item()) != 0.0
