from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.configs import SubTrajectoryBalanceConfig


@dataclass(frozen=True)
class SubTrajectoryBalanceLossOutput:
    loss: torch.Tensor
    subtb_loss: torch.Tensor
    residual_abs: torch.Tensor
    root_abs: torch.Tensor
    success_rate: torch.Tensor


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
            denom = weights.sum(dim=-1).clamp_min(1.0)
            return weighted.sum(dim=-1) / denom
        return weighted.sum(dim=-1)

    def compute(self, sample_batch) -> SubTrajectoryBalanceLossOutput:
        log_pf_steps = sample_batch.log_pf_steps.to(dtype=torch.float32)
        batch_size, num_rollouts, max_steps = log_pf_steps.shape
        sequence_horizon = max_steps + 2

        state_values = log_pf_steps.new_zeros(
            (batch_size, num_rollouts, sequence_horizon)
        )
        graph_log_z = sample_batch.graph_log_z.to(dtype=torch.float32).unsqueeze(1)
        state_values[:, :, 0] = graph_log_z.expand(-1, num_rollouts)
        state_values[:, :, 1] = sample_batch.start_state_log_f.to(dtype=torch.float32)
        state_values[:, :, 2:] = sample_batch.next_state_log_f_steps.to(
            dtype=torch.float32
        )

        prefix = log_pf_steps.new_zeros((batch_size, num_rollouts, sequence_horizon))
        start_log_probs = sample_batch.start_log_probs.to(dtype=torch.float32)
        prefix[:, :, 1] = start_log_probs
        prefix[:, :, 2:] = start_log_probs.unsqueeze(-1) + torch.cumsum(
            log_pf_steps, dim=-1
        )

        terminal_index = (
            sample_batch.terminal_num_steps.to(dtype=torch.long) + 1
        ).clamp(
            min=2,
            max=sequence_horizon - 1,
        )
        terminal_prefix = prefix.gather(-1, terminal_index.unsqueeze(-1)).squeeze(-1)
        terminal_anchor = (
            sample_batch.terminal_log_rewards.to(dtype=torch.float32) - terminal_prefix
        )

        x_values = state_values - prefix
        flat_x = x_values.view(-1, sequence_horizon)
        flat_terminal = terminal_index.view(-1)
        flat_anchor = terminal_anchor.view(-1)
        traj_ids = torch.arange(flat_x.size(0), device=flat_x.device, dtype=torch.long)
        flat_x[traj_ids, flat_terminal] = flat_anchor

        step_idx = torch.arange(sequence_horizon, device=flat_x.device).view(1, 1, -1)
        valid_positions = step_idx <= terminal_index.unsqueeze(-1)
        if not torch.isfinite(x_values[valid_positions]).all():
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_z/log_pf/log_reward."
            )

        x_end = x_values.gather(-1, terminal_index.unsqueeze(-1)).squeeze(-1)
        residual = torch.where(
            valid_positions,
            x_values - x_end.unsqueeze(-1),
            torch.zeros_like(x_values),
        )
        distance_to_end = (
            (terminal_index.unsqueeze(-1) - step_idx)
            .clamp(min=0)
            .to(dtype=torch.float32)
        )
        if float(self.config.lambda_weight) == 1.0:
            weights = torch.ones_like(residual)
        else:
            weights = torch.pow(
                torch.full_like(residual, fill_value=float(self.config.lambda_weight)),
                distance_to_end,
            )
        weights = torch.where(valid_positions, weights, torch.zeros_like(weights))

        per_rollout_loss = self._weighted_mean(
            values=residual.square(),
            weights=weights,
            normalize=bool(self.config.normalize),
        )
        loss = per_rollout_loss.mean()
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_z/log_pf/log_reward."
            )

        root_residual = x_values[:, :, 0] - x_end
        if bool(valid_positions.any().item()):
            residual_abs = residual[valid_positions].abs().mean()
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
        )


__all__ = [
    "SubTrajectoryBalanceLoss",
    "SubTrajectoryBalanceLossOutput",
]
