from __future__ import annotations
import torch
from torch import nn

from src.models.rollout import RolloutResult
from src.models.configs.objective import SubTBConfig


class SubTrajectoryBalanceLoss(nn.Module):
    """
    [系统实体] Forward-Looking SubTB
    对每个前缀状态 s_t 约束：
        log F(s_t) + sum_{i=t}^{L-1}(log P_F(i) - log P_B(i)) = log R
    在当前路径前缀 MDP 契约中 log P_B == 0，但这里仍保留一般形式以便审计。
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

    def _compute_subtrajectory_residual_loss(
        self,
        *,
        residual: torch.Tensor,
        valid_state_mask: torch.Tensor,
        effective_state_len: torch.Tensor,
        step_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        lambda_weight: float,
        normalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0.0 <= lambda_weight <= 1.0):
            raise ValueError(
                f"SubTB lambda_weight must be in [0, 1], got {lambda_weight}."
            )
        distance_to_end = (
            (effective_state_len.unsqueeze(-1) - 1 - step_idx)
            .clamp(min=0)
            .to(dtype=residual.dtype)
        )
        if lambda_weight == 1.0:
            temporal_weights = torch.ones_like(residual)
        else:
            temporal_weights = torch.pow(
                torch.full_like(residual, fill_value=lambda_weight), distance_to_end
            )
        temporal_weights = torch.where(
            valid_state_mask, temporal_weights, torch.zeros_like(temporal_weights)
        )

        weighted_sq = residual.square() * temporal_weights
        if normalize:
            denom = temporal_weights.sum(dim=-1).clamp(min=1.0)
            per_rollout_loss = weighted_sq.sum(dim=-1) / denom
        else:
            per_rollout_loss = weighted_sq.sum(dim=-1)

        if bool(valid_mask.any().item()):
            subtraj_loss = per_rollout_loss[valid_mask].mean()
            residual_abs = residual[valid_state_mask].abs().mean()
        else:
            subtraj_loss = residual.new_zeros(())
            residual_abs = residual.new_zeros(())
        return subtraj_loss, residual_abs

    @staticmethod
    def _compute_forward_looking_residual(
        *,
        log_pf_steps: torch.Tensor,
        log_pb_steps: torch.Tensor,
        log_f_steps: torch.Tensor,
        log_r: torch.Tensor,
        lengths: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_steps = int(log_pf_steps.size(-1))
        step_idx = torch.arange(
            max_steps, device=log_pf_steps.device, dtype=torch.long
        ).view(1, 1, max_steps)
        valid_rollout = valid_mask.unsqueeze(-1)
        transition_mask = (step_idx < lengths.unsqueeze(-1)) & valid_rollout
        effective_state_len = torch.where(
            lengths > 0, lengths, torch.ones_like(lengths)
        ).clamp(min=1, max=max_steps)
        valid_state_mask = (
            step_idx < effective_state_len.unsqueeze(-1)
        ) & valid_rollout

        safe_log_pf = torch.where(
            valid_rollout, log_pf_steps, torch.zeros_like(log_pf_steps)
        )
        safe_log_pb = torch.where(
            valid_rollout, log_pb_steps, torch.zeros_like(log_pb_steps)
        )
        safe_log_f = torch.where(
            valid_rollout, log_f_steps, torch.zeros_like(log_f_steps)
        )

        # suffix_delta[t] = sum_{i=t}^{L-1} (logPF_i - logPB_i)
        delta = torch.where(
            transition_mask, safe_log_pf - safe_log_pb, torch.zeros_like(safe_log_pf)
        )
        suffix_delta = torch.flip(
            torch.cumsum(torch.flip(delta, dims=[-1]), dim=-1), dims=[-1]
        )
        residual_raw = (
            safe_log_f + suffix_delta - log_r.unsqueeze(-1).to(dtype=safe_log_f.dtype)
        )
        residual = torch.where(
            valid_state_mask, residual_raw, torch.zeros_like(residual_raw)
        )
        return residual, valid_state_mask, step_idx

    def forward(
        self,
        fwd_rollout: RolloutResult,
        rewards: torch.Tensor,
        reward_beta: float = 1.0,
        hit_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if bool(self.config.detach_end_flow):
            raise ValueError(
                "subtb.detach_end_flow=True is incompatible with Forward-Looking SubTB. "
                "Set detach_end_flow=false."
            )
        device = rewards.device
        log_pf_steps = fwd_rollout.log_pf_steps
        log_pb_steps = fwd_rollout.log_pb_steps
        log_f_steps = fwd_rollout.log_f_steps
        if log_pf_steps is None or log_pb_steps is None or log_f_steps is None:
            raise ValueError(
                "SubTB requires log_pf_steps, log_pb_steps, and log_f_steps."
            )
        log_pf_steps = self._ensure_rollout_axis(log_pf_steps, name="log_pf_steps")
        log_pb_steps = self._ensure_rollout_axis(log_pb_steps, name="log_pb_steps")
        log_f_steps = self._ensure_rollout_axis(log_f_steps, name="log_f_steps")
        if tuple(log_pb_steps.shape) != tuple(log_pf_steps.shape):
            raise ValueError(
                "log_pb_steps shape mismatch with log_pf_steps: "
                f"log_pb_steps={tuple(log_pb_steps.shape)}, log_pf_steps={tuple(log_pf_steps.shape)}."
            )
        if tuple(log_f_steps.shape) != tuple(log_pf_steps.shape):
            raise ValueError(
                "log_f_steps shape mismatch with log_pf_steps: "
                f"log_f_steps={tuple(log_f_steps.shape)}, log_pf_steps={tuple(log_pf_steps.shape)}."
            )
        move_lengths = fwd_rollout.num_moves.to(device=device)
        if move_lengths.dim() == 1:
            move_lengths = move_lengths.unsqueeze(1)
        elif move_lengths.dim() != 2:
            raise ValueError(
                f"num_moves must be 1D or 2D, got shape={tuple(move_lengths.shape)}"
            )

        max_steps = int(log_pf_steps.size(-1))
        lengths = fwd_rollout.num_steps.to(device=device)
        if lengths.dim() == 1:
            lengths = lengths.unsqueeze(1)
        elif lengths.dim() != 2:
            raise ValueError(
                f"num_steps must be 1D or 2D, got shape={tuple(lengths.shape)}"
            )
        if tuple(lengths.shape) != tuple(move_lengths.shape):
            raise ValueError(
                f"num_steps shape mismatch: expected {tuple(move_lengths.shape)}, got {tuple(lengths.shape)}."
            )
        lengths = lengths.clamp(min=0, max=max_steps)
        rollout_valid_mask = fwd_rollout.valid_mask
        rollout_valid_mask_2d: torch.Tensor | None = None
        if rollout_valid_mask is not None:
            if rollout_valid_mask.dim() == 1:
                rollout_valid_mask = rollout_valid_mask.unsqueeze(1)
            elif rollout_valid_mask.dim() != 2:
                raise ValueError(
                    f"rollout.valid_mask must be 1D or 2D, got shape={tuple(rollout_valid_mask.shape)}"
                )
            if tuple(rollout_valid_mask.shape) != tuple(lengths.shape):
                raise ValueError(
                    f"rollout.valid_mask shape mismatch: expected {tuple(lengths.shape)}, "
                    f"got {tuple(rollout_valid_mask.shape)}."
                )
            rollout_valid_mask_2d = rollout_valid_mask.to(
                device=device, dtype=torch.bool
            )

        valid_mask = lengths >= 0
        if rollout_valid_mask_2d is not None:
            valid_mask = valid_mask & rollout_valid_mask_2d
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True), {
                "subtb/loss": 0.0
            }
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        elif rewards.dim() != 2:
            raise ValueError(
                f"rewards must be 1D or 2D, got shape={tuple(rewards.shape)}"
            )
        if tuple(rewards.shape) != tuple(lengths.shape):
            raise ValueError(
                f"rewards shape mismatch with rollout axis: expected {tuple(lengths.shape)}, got {tuple(rewards.shape)}."
            )
        hit_mask_2d: torch.Tensor | None = None
        if hit_mask is not None:
            if hit_mask.dim() == 1:
                hit_mask = hit_mask.unsqueeze(1)
            elif hit_mask.dim() != 2:
                raise ValueError(
                    f"hit_mask must be 1D or 2D, got shape={tuple(hit_mask.shape)}"
                )
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
                raise ValueError(
                    "miss_length_penalty > 0 requires explicit hit_mask to avoid reward-threshold ambiguity."
                )
            is_miss = (~hit_mask_2d) & valid_mask
            wasted_steps = (max_steps - lengths).clamp(min=0).to(dtype=log_r.dtype)
            penalty = is_miss.to(dtype=log_r.dtype) * wasted_steps * miss_length_penalty
            log_r = log_r - penalty

        residual, valid_state_mask, step_idx = self._compute_forward_looking_residual(
            log_pf_steps=log_pf_steps,
            log_pb_steps=log_pb_steps,
            log_f_steps=log_f_steps,
            log_r=log_r,
            lengths=lengths,
            valid_mask=valid_mask,
        )
        effective_state_len = torch.where(
            lengths > 0, lengths, torch.ones_like(lengths)
        ).clamp(min=1, max=max_steps)
        state_count = valid_state_mask.sum(dim=-1).clamp(min=1).to(dtype=residual.dtype)
        rollout_tb_mean = residual.sum(dim=-1) / state_count
        lambda_weight = float(self.config.lambda_weight)
        normalize = bool(self.config.normalize)
        var_loss, residual_abs = self._compute_subtrajectory_residual_loss(
            residual=residual,
            valid_state_mask=valid_state_mask,
            effective_state_len=effective_state_len,
            step_idx=step_idx,
            valid_mask=valid_mask,
            lambda_weight=lambda_weight,
            normalize=normalize,
        )

        if bool(valid_mask.any().item()):
            std_x = rollout_tb_mean[valid_mask].std(unbiased=False)
        else:
            std_x = rollout_tb_mean.new_zeros(())

        loss = var_loss
        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss detected in SubTB. Check log_f/log_pf/log_pb."
            )
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
