# src/models/metrics/subtb_loss.py
"""
[系统实体] Sub-Trajectory Balance Loss
线性时间版本，显式约束子轨迹残差。
"""
from __future__ import annotations
import torch
from torch import nn

from src.models.components.rollout_types import RolloutResult
from src.models.configs.objective import SubTBConfig


class SubTrajectoryBalanceLoss(nn.Module):
    """
    [系统实体] Sub-trajectory Balance Loss
    线性时间子轨迹残差形式，显式使用 log F(s) 与 log R
    """

    def __init__(self, config: SubTBConfig) -> None:
        super().__init__()
        self.config = config
        self.eps = 1.0e-8

    @staticmethod
    def _ensure_rollout_axis(value: torch.Tensor, *, name: str) -> torch.Tensor:
        if value.dim() == 3:
            return value
        if value.dim() == 2:
            return value.unsqueeze(1)
        raise ValueError(f"{name} must be 2D or 3D, got shape={tuple(value.shape)}")

    def _compute_prefix_pad(
        self,
        *,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        _, _, max_steps = log_pf_steps.shape
        step_idx = torch.arange(max_steps, device=log_pf_steps.device).view(1, 1, max_steps)
        step_mask = step_idx < lengths.unsqueeze(-1)
        delta = (log_pf_steps - log_pb_steps) * step_mask
        prefix = delta.cumsum(dim=-1)
        return torch.cat(
            [torch.zeros((*prefix.shape[:2], 1), device=prefix.device, dtype=prefix.dtype), prefix],
            dim=-1,
        )

    @staticmethod
    def _compute_x_values(
        *,
        log_f_steps: torch.Tensor,
        log_r: torch.Tensor,
        terminal_indices: torch.Tensor,
        prefix_pad: torch.Tensor,
    ) -> torch.Tensor:
        # Build [B, K, T+1] without appending reward to a fixed horizon slot.
        # Reward is aligned to each rollout's true terminal index L via advanced indexing.
        B, K, max_steps = log_f_steps.shape
        terminal_horizon = max_steps + 1
        log_f_all = log_f_steps.new_zeros((B, K, terminal_horizon))
        log_f_all[:, :, :max_steps] = log_f_steps

        flat = log_f_all.view(-1, terminal_horizon)
        flat_terminal_indices = terminal_indices.reshape(-1).clamp(min=0, max=max_steps)
        flat_log_r = log_r.reshape(-1).to(dtype=log_f_steps.dtype)
        traj_ids = torch.arange(flat.size(0), device=log_f_steps.device, dtype=torch.long)
        flat[traj_ids, flat_terminal_indices] = flat_log_r

        return log_f_all - prefix_pad

    @staticmethod
    def _compute_rollout_tb_mean(
        *,
        x_values: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.arange(x_values.size(-1), device=x_values.device).view(1, 1, -1)
        mask_x = idx <= lengths.unsqueeze(-1)
        sum_x = (x_values * mask_x).sum(dim=-1)
        n = (lengths + 1).to(dtype=x_values.dtype)
        return sum_x / n

    @staticmethod
    def _compute_subtrajectory_residual_loss(
        *,
        x_values: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
        lambda_weight: float,
        normalize: bool,
        detach_end_flow: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0.0 <= lambda_weight <= 1.0):
            raise ValueError(f"SubTB lambda_weight must be in [0, 1], got {lambda_weight}.")
        step_idx = torch.arange(x_values.size(-1), device=x_values.device).view(1, 1, -1)
        valid_x = step_idx <= lengths.unsqueeze(-1)
        valid_x = valid_x & valid_mask.unsqueeze(-1)
        end_idx = lengths.clamp(min=0, max=x_values.size(-1) - 1).unsqueeze(-1)
        x_end = x_values.gather(dim=-1, index=end_idx).squeeze(-1)
        x_end_anchor = x_end.detach() if detach_end_flow else x_end
        residual = x_values - x_end_anchor.unsqueeze(-1)
        residual = torch.where(valid_x, residual, torch.zeros_like(residual))

        distance_to_end = (lengths.unsqueeze(-1) - step_idx).clamp(min=0).to(dtype=x_values.dtype)
        if lambda_weight == 1.0:
            temporal_weights = torch.ones_like(residual)
        else:
            temporal_weights = torch.pow(torch.full_like(residual, fill_value=lambda_weight), distance_to_end)
        temporal_weights = torch.where(valid_x, temporal_weights, torch.zeros_like(temporal_weights))

        weighted_sq = residual.square() * temporal_weights
        if normalize:
            denom = temporal_weights.sum(dim=-1).clamp(min=1.0)
            per_rollout_loss = weighted_sq.sum(dim=-1) / denom
        else:
            per_rollout_loss = weighted_sq.sum(dim=-1)

        if bool(valid_mask.any().item()):
            subtraj_loss = per_rollout_loss[valid_mask].mean()
            residual_abs = residual[valid_x].abs().mean()
        else:
            subtraj_loss = x_values.new_zeros(())
            residual_abs = x_values.new_zeros(())
        return subtraj_loss, residual_abs

    def forward(
        self,
        fwd_rollout: RolloutResult,
        rewards: torch.Tensor,  # 注意：此时传入的应该是 raw_rewards (即 1.0 或 epsilon)
        reward_beta: float = 1.0,  # 【审计官添加】：直接接收当前 beta
        hit_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        计算 TB Loss
        Args:
            fwd_rollout: 前向推演结果
            env_state: 图环境上下文
            policy: 策略网络
            rewards: [B], 环境根据到达目标结算出的真实奖励 (R >= 0)
        """
        device = rewards.device
        log_pf_steps = fwd_rollout.log_pf_steps
        log_pb_steps = fwd_rollout.log_pb_steps
        log_f_steps = fwd_rollout.log_f_steps
        if log_pf_steps is None or log_pb_steps is None or log_f_steps is None:
            raise ValueError("SubTB requires log_pf_steps, log_pb_steps, and log_f_steps.")
        log_pf_steps = self._ensure_rollout_axis(log_pf_steps, name="log_pf_steps")
        log_pb_steps = self._ensure_rollout_axis(log_pb_steps, name="log_pb_steps")
        log_f_steps = self._ensure_rollout_axis(log_f_steps, name="log_f_steps")
        move_lengths = fwd_rollout.num_moves.to(device=device)
        if move_lengths.dim() == 1:
            move_lengths = move_lengths.unsqueeze(1)
        elif move_lengths.dim() != 2:
            raise ValueError(f"num_moves must be 1D or 2D, got shape={tuple(move_lengths.shape)}")

        max_steps = int(log_pf_steps.size(-1))
        lengths = fwd_rollout.num_steps.to(device=device)
        if lengths.dim() == 1:
            lengths = lengths.unsqueeze(1)
        elif lengths.dim() != 2:
            raise ValueError(f"num_steps must be 1D or 2D, got shape={tuple(lengths.shape)}")
        if tuple(lengths.shape) != tuple(move_lengths.shape):
            raise ValueError(
                f"num_steps shape mismatch: expected {tuple(move_lengths.shape)}, got {tuple(lengths.shape)}."
            )
        lengths = lengths.clamp(min=0, max=max_steps)
        terminal_indices = torch.where(lengths > 0, lengths, torch.ones_like(lengths)).clamp(
            min=0, max=max_steps
        )
        rollout_valid_mask = fwd_rollout.valid_mask
        rollout_valid_mask_2d: torch.Tensor | None = None
        if rollout_valid_mask is not None:
            if rollout_valid_mask.dim() == 1:
                rollout_valid_mask = rollout_valid_mask.unsqueeze(1)
            elif rollout_valid_mask.dim() != 2:
                raise ValueError(f"rollout.valid_mask must be 1D or 2D, got shape={tuple(rollout_valid_mask.shape)}")
            if tuple(rollout_valid_mask.shape) != tuple(lengths.shape):
                raise ValueError(
                    f"rollout.valid_mask shape mismatch: expected {tuple(lengths.shape)}, "
                    f"got {tuple(rollout_valid_mask.shape)}."
                )
            rollout_valid_mask_2d = rollout_valid_mask.to(device=device, dtype=torch.bool)

        valid_mask = terminal_indices >= 0
        if rollout_valid_mask_2d is not None:
            valid_mask = valid_mask & rollout_valid_mask_2d
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {"subtb/loss": 0.0}
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        elif rewards.dim() != 2:
            raise ValueError(f"rewards must be 1D or 2D, got shape={tuple(rewards.shape)}")
        hit_mask_2d: torch.Tensor | None = None
        if hit_mask is not None:
            if hit_mask.dim() == 1:
                hit_mask = hit_mask.unsqueeze(1)
            elif hit_mask.dim() != 2:
                raise ValueError(f"hit_mask must be 1D or 2D, got shape={tuple(hit_mask.shape)}")
            if tuple(hit_mask.shape) != tuple(rewards.shape):
                raise ValueError(
                    f"hit_mask shape mismatch: expected {tuple(rewards.shape)}, got {tuple(hit_mask.shape)}."
                )
            hit_mask_2d = hit_mask.to(device=device, dtype=torch.bool)
        safe_raw_rewards = rewards.clamp(min=1e-7)
        log_r_raw = torch.log(safe_raw_rewards)
        log_r = log_r_raw * float(reward_beta)
        miss_length_penalty = float(self.config.miss_length_penalty)
        if miss_length_penalty > 0.0:
            if hit_mask_2d is None:
                raise ValueError("miss_length_penalty > 0 requires explicit hit_mask to avoid reward-threshold ambiguity.")
            is_miss = (~hit_mask_2d) & valid_mask
            wasted_steps = (max_steps - lengths).clamp(min=0).to(dtype=log_r.dtype)
            penalty = is_miss.to(dtype=log_r.dtype) * wasted_steps * miss_length_penalty
            log_r = log_r - penalty

        prefix_pad = self._compute_prefix_pad(
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            lengths=lengths,
        )
        x_values = self._compute_x_values(
            log_f_steps=log_f_steps,
            log_r=log_r,
            terminal_indices=terminal_indices,
            prefix_pad=prefix_pad,
        )
        rollout_tb_mean = self._compute_rollout_tb_mean(x_values=x_values, lengths=terminal_indices)
        lambda_weight = float(self.config.lambda_weight)
        normalize = bool(self.config.normalize)
        detach_end_flow = bool(self.config.detach_end_flow)
        var_loss, residual_abs = self._compute_subtrajectory_residual_loss(
            x_values=x_values,
            lengths=terminal_indices,
            valid_mask=valid_mask,
            lambda_weight=lambda_weight,
            normalize=normalize,
            detach_end_flow=detach_end_flow,
        )

        if bool(valid_mask.any().item()):
            std_x = rollout_tb_mean[valid_mask].std(unbiased=False)
        else:
            std_x = rollout_tb_mean.new_zeros(())

        loss = var_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected in SubTB. Check log_f/log_pf/log_pb.")
        metrics = {
            "subtb/loss": loss.detach(),
            "subtb/var_loss": var_loss.detach(),
            "subtb/subtraj_residual_abs": residual_abs.detach(),
            "subtb/log_r_mean": log_r[valid_mask].mean().detach(),
            "subtb/tb_diff_mean": rollout_tb_mean[valid_mask].mean().detach(),
            "subtb/tb_diff_std": std_x.detach(),
            "rollout/num_moves_mean": move_lengths[valid_mask].float().mean().detach(),
        }
        return loss, metrics


__all__ = ["SubTrajectoryBalanceLoss"]
