from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from src.models.configs import SubTrajectoryBalanceConfig

if TYPE_CHECKING:
    from .sampler import TrajectoryGFNSampleBatch


@dataclass(frozen=True)
class SubTrajectoryBalanceLossOutput:
    loss: torch.Tensor
    subtb_loss: torch.Tensor
    residual_abs: torch.Tensor
    root_abs: torch.Tensor
    success_rate: torch.Tensor
    log_z_mean: torch.Tensor
    log_z_variance: torch.Tensor


class SubTrajectoryBalanceLoss:
    def __init__(
        self,
        *,
        config: SubTrajectoryBalanceConfig | None = None,
        root_weight: float | None = None,
        move_weight: float | None = None,
        terminal_weight: float | None = None,
    ) -> None:
        del root_weight, move_weight, terminal_weight
        self.config = config or SubTrajectoryBalanceConfig()

    @staticmethod
    def _weighted_mean(
        *,
        values: torch.Tensor,
        weights: torch.Tensor,
        normalize: bool,
    ) -> torch.Tensor:
        weighted = values * weights
        if normalize:
            denom = weights.sum(dim=(-2, -1)).clamp_min(1.0)
            return weighted.sum(dim=(-2, -1)) / denom
        return weighted.sum(dim=(-2, -1))

    @staticmethod
    def _build_prefix_terms(
        *,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_rollouts, max_steps = log_pf_steps.shape
        state_horizon = max_steps + 1
        forward_prefix = log_pf_steps.new_zeros(
            (batch_size, num_rollouts, state_horizon)
        )
        backward_prefix = torch.zeros_like(forward_prefix)
        if max_steps > 0:
            forward_prefix[:, :, 1:] = torch.cumsum(log_pf_steps, dim=-1)
            backward_prefix[:, :, 1:] = torch.cumsum(log_pb_steps, dim=-1)
        return forward_prefix, backward_prefix

    def compute(
        self, sample_batch: TrajectoryGFNSampleBatch
    ) -> SubTrajectoryBalanceLossOutput:
        log_pf_steps = sample_batch.log_pf_steps.to(dtype=torch.float32)
        log_pb_steps = sample_batch.log_pb_steps.to(dtype=torch.float32)
        if tuple(log_pb_steps.shape) != tuple(log_pf_steps.shape):
            raise ValueError(
                "log_pb_steps must match log_pf_steps shape for SubTB. "
                f"log_pb_steps={tuple(log_pb_steps.shape)} "
                f"log_pf_steps={tuple(log_pf_steps.shape)}."
            )

        batch_size, num_rollouts, max_steps = log_pf_steps.shape
        sequence_horizon = max_steps + 1
        graph_log_z_values = sample_batch.graph_log_z.to(dtype=torch.float32)

        state_values = log_pf_steps.new_zeros(
            (batch_size, num_rollouts, sequence_horizon)
        )
        state_values[:, :, 0] = sample_batch.start_state_log_f.to(dtype=torch.float32)
        if max_steps > 0:
            state_values[:, :, 1:] = sample_batch.next_state_log_f_steps.to(
                dtype=torch.float32
            )

        forward_prefix, backward_prefix = self._build_prefix_terms(
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
        )
        trajectory_values = state_values - forward_prefix + backward_prefix

        terminal_index = sample_batch.terminal_num_steps.to(dtype=torch.long)
        if bool((terminal_index < 0).any().item()) or bool(
            (terminal_index >= sequence_horizon).any().item()
        ):
            raise ValueError(
                "terminal_num_steps produced an out-of-range terminal state index for SubTB. "
                f"sequence_horizon={sequence_horizon}"
            )
        terminal_forward_prefix = forward_prefix.gather(
            -1, terminal_index.unsqueeze(-1)
        ).squeeze(-1)
        terminal_backward_prefix = backward_prefix.gather(
            -1, terminal_index.unsqueeze(-1)
        ).squeeze(-1)
        terminal_values = (
            sample_batch.terminal_log_rewards.to(dtype=torch.float32)
            - terminal_forward_prefix
            + terminal_backward_prefix
        )

        position_ids = torch.arange(sequence_horizon, device=trajectory_values.device)
        position_ids = position_ids.view(1, 1, -1)
        valid_positions = position_ids <= terminal_index.unsqueeze(-1)
        terminal_mask = position_ids == terminal_index.unsqueeze(-1)
        anchored_values = torch.where(
            terminal_mask,
            terminal_values.unsqueeze(-1),
            trajectory_values,
        )
        if not torch.isfinite(anchored_values[valid_positions]).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_z/log_pf/log_pb/log_reward."
            )

        start_positions = position_ids.unsqueeze(-1)
        end_positions = position_ids.unsqueeze(-2)
        pair_mask = (
            valid_positions.unsqueeze(-1)
            & valid_positions.unsqueeze(-2)
            & (start_positions < end_positions)
        )
        pair_lengths = (
            (end_positions - start_positions).clamp_min(0).to(dtype=torch.float32)
        )
        pairwise_residual = anchored_values.unsqueeze(-1) - anchored_values.unsqueeze(
            -2
        )

        lambda_weight = float(self.config.lambda_weight)
        if lambda_weight == 1.0:
            weights = torch.ones_like(pairwise_residual)
        else:
            weights = torch.pow(
                torch.full_like(pairwise_residual, fill_value=lambda_weight),
                (pair_lengths - 1.0).clamp_min(0.0),
            )
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))

        per_rollout_loss = self._weighted_mean(
            values=pairwise_residual.square(),
            weights=weights,
            normalize=bool(self.config.normalize),
        )
        loss = per_rollout_loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_z/log_pf/log_pb/log_reward."
            )

        terminal_state_values = anchored_values.gather(
            -1, terminal_index.unsqueeze(-1)
        ).squeeze(-1)
        root_residual = anchored_values[:, :, 0] - terminal_state_values
        if bool(pair_mask.any().item()):
            residual_abs = pairwise_residual[pair_mask].abs().mean()
        else:
            residual_abs = torch.zeros((), device=loss.device)

        return SubTrajectoryBalanceLossOutput(
            loss=loss,
            subtb_loss=loss.detach(),
            residual_abs=residual_abs.detach(),
            root_abs=root_residual.abs().mean().detach(),
            success_rate=sample_batch.success_mask.to(dtype=torch.float32)
            .mean()
            .detach(),
            log_z_mean=graph_log_z_values.mean().detach(),
            log_z_variance=graph_log_z_values.var(unbiased=False).detach(),
        )


__all__ = [
    "SubTrajectoryBalanceLoss",
    "SubTrajectoryBalanceLossOutput",
]
