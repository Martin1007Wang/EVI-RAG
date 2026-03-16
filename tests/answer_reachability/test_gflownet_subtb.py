from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.configs import SubTrajectoryBalanceConfig
from src.models.training.losses import SubTrajectoryBalanceLoss


def test_subtb_loss_zero_for_consistent_single_move_rollout() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = SimpleNamespace(
        graph_log_z=torch.tensor([0.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[-0.5]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[-0.5]], dtype=torch.float32),
        log_pf_steps=torch.tensor([[[-0.25, 0.0]]], dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[1]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[-0.75]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(sample_batch)

    assert torch.allclose(loss_output.loss, torch.tensor(0.0), atol=1.0e-6)
    assert torch.allclose(loss_output.subtb_loss, torch.tensor(0.0), atol=1.0e-6)


def test_subtb_loss_handles_zero_move_rollout_with_finite_anchor() -> None:
    loss_fn = SubTrajectoryBalanceLoss(
        config=SubTrajectoryBalanceConfig(lambda_weight=1.0, normalize=True)
    )
    sample_batch = SimpleNamespace(
        graph_log_z=torch.tensor([0.0], dtype=torch.float32),
        start_log_probs=torch.tensor([[-0.4]], dtype=torch.float32),
        start_state_log_f=torch.tensor([[-0.4]], dtype=torch.float32),
        log_pf_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        next_state_log_f_steps=torch.zeros((1, 1, 2), dtype=torch.float32),
        terminal_num_steps=torch.tensor([[0]], dtype=torch.long),
        terminal_log_rewards=torch.tensor([[-0.4]], dtype=torch.float32),
        success_mask=torch.ones((1, 1), dtype=torch.bool),
    )

    loss_output = loss_fn.compute(sample_batch)

    assert torch.isfinite(loss_output.loss)
    assert torch.allclose(loss_output.loss, torch.tensor(0.0), atol=1.0e-6)
