from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.weaver.loss import (
    SubTrajectoryBalanceLoss,
    stop_advantage_loss,
    stop_now_tb_loss,
)
from src.weaver.rollout.schema import RolloutBatch, RolloutStats, RolloutTraces


def _rollout_for_loss(
    *,
    state_log_flows: torch.Tensor,
    step_log_pf: torch.Tensor,
    terminal_log_reward: torch.Tensor,
    stop_log_pf: torch.Tensor,
    stop_now_log_reward: torch.Tensor,
    stop_tb_valid_mask: torch.Tensor,
    stop_adv_loss: torch.Tensor | None = None,
    stop_adv_target: torch.Tensor | None = None,
    stop_adv_valid_mask: torch.Tensor | None = None,
    stop_adv_continue_log_reward: torch.Tensor | None = None,
) -> RolloutBatch:
    batch_size, horizon = state_log_flows.shape
    zeros_b = torch.zeros(batch_size, dtype=torch.float32)
    zeros_bt = torch.zeros((batch_size, horizon), dtype=torch.float32)
    bool_bt = torch.zeros((batch_size, horizon), dtype=torch.bool)
    stop_mask = bool_bt.clone()
    stop_mask[:, -1] = True

    return RolloutBatch(
        stats=RolloutStats(
            root_log_z=zeros_b.clone(),
            trajectory_length=torch.full((batch_size,), horizon, dtype=torch.long),
            terminal_log_reward=terminal_log_reward,
            terminal_answer_f1=zeros_b.clone(),
            edge_action_entropy=zeros_b.clone(),
            edge_action_count=zeros_b.clone(),
        ),
        traces=RolloutTraces(
            state_log_flows=state_log_flows,
            log_pf=step_log_pf,
            log_pb=zeros_bt.clone(),
            action_type=torch.zeros((batch_size, horizon), dtype=torch.long),
            continue_mask=bool_bt.clone(),
            stop_mask=stop_mask,
            selected_edge_ids=torch.full((batch_size, horizon), -1, dtype=torch.long),
            stop_now_log_reward=stop_now_log_reward,
            stop_now_answer_f1=zeros_bt.clone(),
            stop_now_valid_mask=stop_tb_valid_mask.clone(),
            stop_log_pf=stop_log_pf,
            stop_tb_valid_mask=stop_tb_valid_mask,
            target_stop_prob=zeros_bt.clone(),
            target_continue_prob=zeros_bt.clone(),
            policy_action_valid_mask=bool_bt.clone(),
            edge_action_entropy=zeros_bt.clone(),
            edge_action_entropy_valid_mask=bool_bt.clone(),
            stop_adv_loss=stop_adv_loss,
            stop_adv_target=stop_adv_target,
            stop_adv_valid_mask=stop_adv_valid_mask,
            stop_adv_continue_log_reward=stop_adv_continue_log_reward,
        ),
    )


def test_stop_now_tb_loss_uses_only_valid_counterfactual_edges() -> None:
    loss = stop_now_tb_loss(
        state_log_flow=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        stop_log_pf=torch.tensor([[-0.2, -0.5]], dtype=torch.float32),
        stop_now_log_reward=torch.tensor([[0.3, 0.0]], dtype=torch.float32),
        valid_mask=torch.tensor([[True, False]], dtype=torch.bool),
    )

    assert loss.item() == pytest.approx((1.0 - 0.2 - 0.3) ** 2)


def test_stop_now_tb_loss_returns_zero_for_empty_mask() -> None:
    loss = stop_now_tb_loss(
        state_log_flow=torch.ones((1, 2), dtype=torch.float32),
        stop_log_pf=torch.zeros((1, 2), dtype=torch.float32),
        stop_now_log_reward=torch.zeros((1, 2), dtype=torch.float32),
        valid_mask=torch.zeros((1, 2), dtype=torch.bool),
    )

    assert loss.item() == pytest.approx(0.0)


def test_subtrajectory_balance_adds_weighted_stop_tb_loss() -> None:
    rollout = _rollout_for_loss(
        state_log_flows=torch.tensor([[0.0]], dtype=torch.float32),
        step_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        terminal_log_reward=torch.tensor([0.0], dtype=torch.float32),
        stop_log_pf=torch.tensor([[-0.25]], dtype=torch.float32),
        stop_now_log_reward=torch.tensor([[0.25]], dtype=torch.float32),
        stop_tb_valid_mask=torch.tensor([[True]], dtype=torch.bool),
    )
    loss_fn = SubTrajectoryBalanceLoss(max_trajectory_len=1, stop_tb_coef=0.5)

    output = loss_fn(rollout)

    assert output.metrics["loss/subtb"].item() == pytest.approx(0.0)
    assert output.metrics["loss/stop_tb"].item() == pytest.approx(0.25)
    assert output.metrics["loss/total"].item() == pytest.approx(0.125)
    assert output.loss.item() == pytest.approx(0.125)


def test_subtrajectory_balance_default_stop_tb_coef_includes_stop_tb_loss() -> None:
    rollout = _rollout_for_loss(
        state_log_flows=torch.tensor([[0.0]], dtype=torch.float32),
        step_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        terminal_log_reward=torch.tensor([0.0], dtype=torch.float32),
        stop_log_pf=torch.tensor([[-1.0]], dtype=torch.float32),
        stop_now_log_reward=torch.tensor([[1.0]], dtype=torch.float32),
        stop_tb_valid_mask=torch.tensor([[True]], dtype=torch.bool),
    )
    loss_fn = SubTrajectoryBalanceLoss(max_trajectory_len=1)

    output = loss_fn(rollout)

    assert output.metrics["loss/stop_tb"].item() == pytest.approx(4.0)
    assert output.metrics["loss/total"].item() == pytest.approx(
        output.metrics["loss/subtb"].item() + 4.0
    )


def test_subtrajectory_balance_adds_weighted_stop_adv_loss() -> None:
    rollout = _rollout_for_loss(
        state_log_flows=torch.tensor([[0.0]], dtype=torch.float32),
        step_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        terminal_log_reward=torch.tensor([0.0], dtype=torch.float32),
        stop_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        stop_now_log_reward=torch.tensor([[0.5]], dtype=torch.float32),
        stop_tb_valid_mask=torch.tensor([[False]], dtype=torch.bool),
        stop_adv_loss=torch.tensor([[2.0]], dtype=torch.float32),
        stop_adv_target=torch.tensor([[0.75]], dtype=torch.float32),
        stop_adv_valid_mask=torch.tensor([[True]], dtype=torch.bool),
        stop_adv_continue_log_reward=torch.tensor([[0.25]], dtype=torch.float32),
    )
    loss_fn = SubTrajectoryBalanceLoss(max_trajectory_len=1, stop_adv_coef=0.25)

    output = loss_fn(rollout)
    expected_stop_adv = stop_advantage_loss(
        stop_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        target=torch.tensor([[0.75]], dtype=torch.float32),
        valid_mask=torch.tensor([[True]], dtype=torch.bool),
    )

    assert output.metrics["loss/subtb"].item() == pytest.approx(0.0)
    assert output.metrics["loss/stop_adv"].item() == pytest.approx(
        expected_stop_adv.item()
    )
    assert output.metrics["loss/total"].item() == pytest.approx(
        0.25 * expected_stop_adv.item()
    )
    assert output.metrics["stop_adv/target_mean"].item() == pytest.approx(0.75)
    assert output.metrics["stop_adv/stop_now_better_ratio"].item() == pytest.approx(1.0)


def test_subtrajectory_balance_default_stop_adv_coef_keeps_auxiliary_optional() -> None:
    rollout = _rollout_for_loss(
        state_log_flows=torch.tensor([[0.0]], dtype=torch.float32),
        step_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        terminal_log_reward=torch.tensor([0.0], dtype=torch.float32),
        stop_log_pf=torch.tensor([[0.0]], dtype=torch.float32),
        stop_now_log_reward=torch.tensor([[0.5]], dtype=torch.float32),
        stop_tb_valid_mask=torch.tensor([[False]], dtype=torch.bool),
        stop_adv_target=torch.tensor([[0.75]], dtype=torch.float32),
        stop_adv_valid_mask=torch.tensor([[True]], dtype=torch.bool),
        stop_adv_continue_log_reward=torch.tensor([[0.25]], dtype=torch.float32),
    )

    loss_fn = SubTrajectoryBalanceLoss(max_trajectory_len=1)

    output = loss_fn(rollout)

    assert output.metrics["loss/stop_adv"].item() > 0.0
    assert output.metrics["loss/total"].item() == pytest.approx(0.0)
