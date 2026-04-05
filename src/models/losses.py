from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from .rollout import RolloutBatch


def _zero_metric_tensor() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


@dataclass(frozen=True)
class DetailedBalanceLossOutput:
    loss: torch.Tensor
    db_loss: torch.Tensor
    residual_abs: torch.Tensor
    success_rate: torch.Tensor
    average_terminal_answer_candidate_count: torch.Tensor
    average_terminal_gold_answer_count: torch.Tensor
    average_terminal_component_count: torch.Tensor
    residual_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)
    root_abs: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_mean: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)


class DetailedBalanceLoss(nn.Module):
    def __init__(self, **_: object) -> None:
        super().__init__()

    # [修改2] 参数类型和名称改为 rollout_batch: "RolloutBatch"
    def compute(self, rollout_batch: "RolloutBatch") -> DetailedBalanceLossOutput:
        # [修改3] 内部所有调用统一替换为 rollout_batch
        log_pf = rollout_batch.log_pf_actions.float()
        log_pb = rollout_batch.log_pb_actions.float()
        log_reward = rollout_batch.log_reward_actions.float()
        state_log_flow = rollout_batch.state_log_flows.float()
        action_mask = rollout_batch.action_mask.bool()
        termination = rollout_batch.termination_action_steps.long()
        max_actions = log_pf.size(-1)

        step_indices = torch.arange(1, max_actions + 1, device=log_pf.device).view(1, 1, -1)
        is_terminal = action_mask & (termination.unsqueeze(-1) == step_indices)
        is_nonterminal = action_mask & ~is_terminal

        next_state_log_flow = torch.zeros_like(state_log_flow)
        if max_actions > 1:
            next_state_log_flow[..., :-1] = state_log_flow[..., 1:]

        # --- DB Loss 核心向量化计算 ---
        forward_term = state_log_flow + log_pf
        backward_term_nonterminal = next_state_log_flow + log_pb
        backward_term_terminal = log_reward

        target_backward = torch.where(is_terminal, backward_term_terminal, backward_term_nonterminal)
        residual = forward_term - target_backward

        valid_residuals = residual[action_mask]

        if valid_residuals.numel() <= 0:
            db_loss = torch.tensor(0.0, device=log_pf.device, requires_grad=True)
            residual_abs = torch.tensor(0.0, device=log_pf.device)
            residual_variance = torch.tensor(0.0, device=log_pf.device)
        else:
            db_loss = valid_residuals.square().mean()
            residual_abs = valid_residuals.abs().mean()
            residual_variance = (
                valid_residuals.var(unbiased=False) if valid_residuals.numel() > 1 else torch.tensor(0.0, device=log_pf.device)
            )

        root_log_flows = state_log_flow[:, :, 0]
        centered_root = root_log_flows - root_log_flows.mean()

        return DetailedBalanceLossOutput(
            loss=db_loss,
            db_loss=db_loss,
            residual_abs=residual_abs,
            residual_variance=residual_variance,
            root_abs=root_log_flows.abs().mean(),
            success_rate=rollout_batch.success_rate.float(),
            log_z_mean=root_log_flows.mean(),
            log_z_variance=centered_root.square().mean(),
            average_terminal_answer_candidate_count=rollout_batch.terminal_answer_candidate_counts.float().mean(),
            average_terminal_gold_answer_count=rollout_batch.terminal_gold_answer_counts.float().mean(),
            average_terminal_component_count=rollout_batch.terminal_component_counts.float().mean(),
        )


__all__ = [
    "DetailedBalanceLoss",
    "DetailedBalanceLossOutput",
]
