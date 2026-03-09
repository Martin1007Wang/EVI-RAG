from __future__ import annotations

import torch

from src.models.configs.trajectory_gfn import TrajectoryTrainingConfig
from src.models.trajectory_gfn.losses import StepwiseDetailedBalanceLoss
from src.models.trajectory_gfn.sampler import TrajectorySampleBatch


def test_db_identity_zero_residual() -> None:
    sample_batch = TrajectorySampleBatch(
        graph_log_z=torch.tensor([1.0]),
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        start_log_probs=torch.tensor([[0.5]]),
        start_state_log_f=torch.tensor([[1.5]]),
        log_pf_steps=torch.tensor([[[0.1, -0.2]]]),
        log_pb_steps=torch.tensor([[[0.8, 0.0]]]),
        state_log_f_steps=torch.tensor([[[2.0, 1.2]]]),
        next_state_log_f_steps=torch.tensor([[[1.3, 0.0]]]),
        chosen_edge_ids_steps=torch.tensor([[[0, -1]]], dtype=torch.long),
        active_steps=torch.tensor([[[True, True]]]),
        is_stop_steps=torch.tensor([[[False, True]]]),
        stop_nodes=torch.tensor([[2]], dtype=torch.long),
        hit_mask=torch.tensor([[True]]),
        terminal_rewards=torch.tensor([[1.0]]),
        terminal_log_rewards=torch.tensor([[1.0]]),
    )
    loss_fn = StepwiseDetailedBalanceLoss(config=TrajectoryTrainingConfig())
    out = loss_fn.compute(sample_batch)
    assert torch.allclose(out.loss, torch.tensor(0.0), atol=1.0e-7)
    assert torch.allclose(out.start_loss, torch.tensor(0.0), atol=1.0e-7)
    assert torch.allclose(out.move_loss, torch.tensor(0.0), atol=1.0e-7)
    assert torch.allclose(out.stop_loss, torch.tensor(0.0), atol=1.0e-7)
