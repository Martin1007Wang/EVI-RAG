from __future__ import annotations

from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .rollout import RolloutBatch


def _zero() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


@dataclass(frozen=True)
class TrajectoryBalanceLossOutput:
    """
    TB loss 的所有输出量。

    核心量：
        loss            — 用于反向传播的标量
        tb_loss         — 与 loss 相同（预留给未来加正则项时区分）
        residual_abs    — |residual| 均值，直观反映 TB 违约程度
        residual_variance — residual 方差，衡量不同轨迹间的不一致性

    监控量（detach，不参与梯度）：
        log_z_mean / log_z_variance — Z 的估计稳定性
        log_reward_mean             — reward 分布
        trajectory_length_mean      — 平均轨迹长度
    """

    loss: torch.Tensor
    tb_loss: torch.Tensor
    residual_abs: torch.Tensor
    residual_variance: torch.Tensor = field(default_factory=_zero)
    log_z_mean: torch.Tensor = field(default_factory=_zero)
    log_z_variance: torch.Tensor = field(default_factory=_zero)
    log_reward_mean: torch.Tensor = field(default_factory=_zero)
    trajectory_length_mean: torch.Tensor = field(default_factory=_zero)


class TrajectoryBalanceLoss(nn.Module):
    """
    Trajectory Balance Loss (Malkin et al., 2022)。

    TB 目标：
        log Z + Σ log P_F(aₜ|sₜ) = log R(s_T) + Σ log P_B(sₜ|sₜ₊₁)

    residual = log Z + Σ log P_F − log R − Σ log P_B
    L_TB     = E_τ[residual²]
    """

    def __init__(self, log_reward_clip_min: float = -100.0) -> None:
        """
        Args:
            log_reward_clip_min: 对 log_reward 做下截断，防止 reward=0 时
                                 出现 -inf 导致 loss=nan。默认 -100 对应
                                 e^{-100} ≈ 0 的极小 reward，实践中足够安全。
        """
        super().__init__()
        self.log_reward_clip_min = log_reward_clip_min

    def forward(self, rollout_batch: RolloutBatch) -> TrajectoryBalanceLossOutput:
        trajectory_log_pf = rollout_batch.trajectory_log_pf.float()  # (B,)
        trajectory_log_pb = rollout_batch.trajectory_log_pb.float()  # (B,)
        log_reward = rollout_batch.terminal_log_rewards.float()  # (B,)
        log_z = rollout_batch.root_log_z.float()  # (B,)
        trajectory_lengths = rollout_batch.termination_action_steps.float()

        # ── 防御性检查：捕获上游 rollout 的静默错误 ──
        if not trajectory_lengths.gt(0).any():
            # 所有轨迹都没有有效步，通常意味着 rollout 引擎有 bug
            raise ValueError(
                "termination_action_steps are all zero: no valid steps in any trajectory. "
                "Check rollout engine for off-by-one errors around Stop action masking."
            )

        # ── 截断 log_reward，防止 reward=0 导致 -inf 污染 loss ──
        log_reward_safe = log_reward.clamp(min=self.log_reward_clip_min)

        # ── 核心 TB residual ──
        residual = (
            log_z + trajectory_log_pf - log_reward_safe - trajectory_log_pb
        )  # (B,)
        tb_loss = residual.square().mean()

        # ── 辅助指标（全部 detach，不污染计算图）──
        with torch.no_grad():
            residual_abs = residual.abs().mean()
            residual_variance = (
                residual.var(unbiased=False)  # 与 tb_loss 语义一致，有偏估计
                if residual.numel() > 1
                else torch.zeros((), device=trajectory_log_pf.device)
            )
            log_z_variance = (
                log_z.var(unbiased=False)
                if log_z.numel() > 1
                else torch.zeros((), device=log_z.device)  # 与上面统一用有偏版本
            )

        return TrajectoryBalanceLossOutput(
            loss=tb_loss,
            tb_loss=tb_loss,
            residual_abs=residual_abs.detach(),
            residual_variance=residual_variance.detach(),
            log_z_mean=log_z.mean().detach(),
            log_z_variance=log_z_variance.detach(),
            log_reward_mean=log_reward.mean().detach(),  # 用原始值监控，不用 clamp 后的
            trajectory_length_mean=trajectory_lengths.mean().detach(),
        )


__all__ = [
    "TrajectoryBalanceLoss",
    "TrajectoryBalanceLossOutput",
]
