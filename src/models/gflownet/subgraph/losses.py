from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .sampler import SubgraphTrajectorySampleBatch


def _zero_metric_tensor() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


@dataclass(frozen=True)
class SubgraphSubTrajectoryBalanceLossOutput:
    loss: torch.Tensor
    subtb_loss: torch.Tensor
    residual_abs: torch.Tensor
    success_rate: torch.Tensor
    average_terminal_answer_count: torch.Tensor
    average_terminal_component_count: torch.Tensor
    residual_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)
    root_abs: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_mean: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)
    root_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    pairwise_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    terminal_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    answer_quotient_component_loss: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )
    answer_quotient_residual_abs: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )
    answer_quotient_observed_sink_count: torch.Tensor = field(
        default_factory=_zero_metric_tensor
    )


class SubgraphSubTrajectoryBalanceLoss:
    def __init__(self, *, config: Any) -> None:
        self.config = config

    def compute(
        self, sample_batch: "SubgraphTrajectorySampleBatch"
    ) -> SubgraphSubTrajectoryBalanceLossOutput:
        log_pf = sample_batch.log_pf_actions.to(dtype=torch.float32)
        log_reward = sample_batch.log_reward_actions.to(dtype=torch.float32)
        state_log_flow = sample_batch.state_log_flows.to(dtype=torch.float32)
        action_mask = sample_batch.action_mask.to(dtype=torch.bool)
        termination = sample_batch.termination_action_steps.to(dtype=torch.long)
        max_actions = int(log_pf.size(-1))
        step_delta = log_pf - log_reward
        prefix_delta = torch.cumsum(step_delta, dim=-1)
        total_loss = torch.zeros((), device=log_pf.device, dtype=torch.float32)
        total_abs = torch.zeros_like(total_loss)
        total_weight = torch.zeros_like(total_loss)
        lambda_weight = float(self.config.lambda_weight)
        for start in range(max_actions):
            valid_start = termination > int(start)
            if not bool(valid_start.any().item()):
                continue
            start_flow = state_log_flow[:, :, start]
            start_prefix = (
                prefix_delta[:, :, start - 1]
                if start > 0
                else torch.zeros_like(start_flow, dtype=torch.float32)
            )
            for end in range(start, max_actions):
                valid = valid_start & action_mask[:, :, end]
                if not bool(valid.any().item()):
                    continue
                delta_sum = prefix_delta[:, :, end] - start_prefix
                next_flow = torch.zeros_like(start_flow, dtype=torch.float32)
                nonterminal = valid & (termination > int(end + 1))
                if bool(nonterminal.any().item()) and (end + 1) < max_actions:
                    next_flow[nonterminal] = state_log_flow[:, :, end + 1][nonterminal]
                residual = start_flow + delta_sum - next_flow
                weight_value = float(lambda_weight) ** int(end - start)
                weight = valid.to(dtype=torch.float32) * float(weight_value)
                total_loss = total_loss + (residual.square() * weight).sum()
                total_abs = total_abs + (residual.abs() * weight).sum()
                total_weight = total_weight + weight.sum()
        subtb_loss = total_loss / total_weight.clamp_min(1.0)
        residual_abs = total_abs / total_weight.clamp_min(1.0)
        root_log_flows = sample_batch.state_log_flows[:, :, 0].to(dtype=torch.float32)
        centered_root = root_log_flows - root_log_flows.mean()
        return SubgraphSubTrajectoryBalanceLossOutput(
            loss=subtb_loss,
            subtb_loss=subtb_loss,
            residual_abs=residual_abs,
            residual_variance=subtb_loss,
            root_abs=root_log_flows.abs().mean(),
            success_rate=sample_batch.success_rate.to(dtype=torch.float32),
            log_z_mean=root_log_flows.mean(),
            log_z_variance=centered_root.square().mean(),
            average_terminal_answer_count=sample_batch.terminal_answer_counts.to(
                dtype=torch.float32
            ).mean(),
            average_terminal_component_count=sample_batch.terminal_component_counts.to(
                dtype=torch.float32
            ).mean(),
        )


__all__ = [
    "SubgraphSubTrajectoryBalanceLoss",
    "SubgraphSubTrajectoryBalanceLossOutput",
]
