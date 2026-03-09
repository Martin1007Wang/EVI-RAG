from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.configs.trajectory_gfn import TrajectoryTrainingConfig


@dataclass(frozen=True)
class DetailedBalanceLossOutput:
    loss: torch.Tensor
    start_loss: torch.Tensor
    move_loss: torch.Tensor
    stop_loss: torch.Tensor
    hit_rate: torch.Tensor


class StepwiseDetailedBalanceLoss:
    def __init__(self, *, config: TrajectoryTrainingConfig) -> None:
        self.config = config

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if int(mask.numel()) == 0 or not bool(mask.any().item()):
            return values.new_zeros(())
        return values[mask].mean()

    def compute(self, sample_batch) -> DetailedBalanceLossOutput:
        start_residual = (
            sample_batch.graph_log_z.unsqueeze(1)
            + sample_batch.start_log_probs
            - sample_batch.start_state_log_f
        )
        start_sq = start_residual.square()
        start_loss = start_sq.mean()

        move_mask = sample_batch.active_steps & (~sample_batch.is_stop_steps)
        move_residual = (
            sample_batch.state_log_f_steps
            + sample_batch.log_pf_steps
            - sample_batch.next_state_log_f_steps
            - sample_batch.log_pb_steps
        )
        move_loss = self._masked_mean(move_residual.square(), move_mask)

        stop_mask = sample_batch.active_steps & sample_batch.is_stop_steps
        stop_residual = (
            sample_batch.state_log_f_steps
            + sample_batch.log_pf_steps
            - sample_batch.terminal_log_rewards.unsqueeze(-1)
        )
        stop_loss = self._masked_mean(stop_residual.square(), stop_mask)

        loss = (
            self.config.lambda_start * start_loss
            + self.config.lambda_move * move_loss
            + self.config.lambda_stop * stop_loss
        )
        return DetailedBalanceLossOutput(
            loss=loss,
            start_loss=start_loss.detach(),
            move_loss=move_loss.detach(),
            stop_loss=stop_loss.detach(),
            hit_rate=sample_batch.hit_mask.to(dtype=torch.float32).mean().detach(),
        )
