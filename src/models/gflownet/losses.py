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
    average_terminal_commit_candidate_count: torch.Tensor
    average_terminal_gold_answer_count: torch.Tensor
    average_terminal_component_count: torch.Tensor
    average_topology_weight: torch.Tensor = field(default_factory=_zero_metric_tensor)
    residual_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)
    root_abs: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_mean: torch.Tensor = field(default_factory=_zero_metric_tensor)
    log_z_variance: torch.Tensor = field(default_factory=_zero_metric_tensor)
    root_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    pairwise_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)
    terminal_component_loss: torch.Tensor = field(default_factory=_zero_metric_tensor)


class SubgraphSubTrajectoryBalanceLoss:
    def __init__(
        self,
        *,
        lambda_weight: float = 1.0,
        topology_weight_alpha: float = 0.0,
    ) -> None:
        self.lambda_weight = float(lambda_weight)
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("training.subtb.lambda_weight must be in [0, 1].")
        self.topology_weight_alpha = float(topology_weight_alpha)
        if self.topology_weight_alpha < 0.0:
            raise ValueError("training.subtb.topology_weight_alpha must be >= 0.")

    def compute(
        self, sample_batch: "SubgraphTrajectorySampleBatch"
    ) -> SubgraphSubTrajectoryBalanceLossOutput:
        log_pf = sample_batch.log_pf_actions.to(dtype=torch.float32)
        log_pb = sample_batch.log_pb_actions.to(dtype=torch.float32)
        log_reward = sample_batch.log_reward_actions.to(dtype=torch.float32)
        state_log_flow = sample_batch.state_log_flows.to(dtype=torch.float32)
        action_mask = sample_batch.action_mask.to(dtype=torch.bool)
        termination = sample_batch.termination_action_steps.to(dtype=torch.long)
        max_actions = int(log_pf.size(-1))
        step_delta = log_pf - log_pb - log_reward
        prefix_delta = torch.cumsum(step_delta, dim=-1)
        total_loss = torch.zeros((), device=log_pf.device, dtype=torch.float32)
        total_abs = torch.zeros_like(total_loss)
        total_residual = torch.zeros_like(total_loss)
        total_weight = torch.zeros_like(total_loss)
        total_topology_weight = torch.zeros_like(total_loss)
        total_valid_pairs = torch.zeros_like(total_loss)
        lambda_weight = float(self.lambda_weight)
        component_counts = None
        if sample_batch.state_component_counts is not None:
            component_counts = sample_batch.state_component_counts.to(
                dtype=torch.float32
            )
        terminal_component_counts = sample_batch.terminal_component_counts.to(
            dtype=torch.float32
        )
        topology_alpha = float(self.topology_weight_alpha)
        for start in range(max_actions):
            valid_start = termination > int(start)
            if not bool(valid_start.any().item()):
                continue
            start_flow = state_log_flow[:, :, start]
            start_components = None
            if component_counts is not None:
                start_components = component_counts[:, :, start]
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
                topology_weight = torch.ones_like(start_flow, dtype=torch.float32)
                if component_counts is not None and topology_alpha > 0.0:
                    if start_components is None:
                        raise RuntimeError(
                            "Missing start component counts for topology-weighted SubTB."
                        )
                    if (end + 1) < max_actions:
                        next_components = torch.where(
                            termination > int(end + 1),
                            component_counts[:, :, end + 1],
                            terminal_component_counts,
                        )
                    else:
                        next_components = terminal_component_counts
                    topology_weight = torch.exp(
                        float(topology_alpha)
                        * (start_components.to(dtype=torch.float32) - next_components)
                    )
                weight = (
                    valid.to(dtype=torch.float32)
                    * float(weight_value)
                    * topology_weight.to(dtype=torch.float32)
                )
                total_loss = total_loss + (residual.square() * weight).sum()
                total_abs = total_abs + (residual.abs() * weight).sum()
                total_residual = total_residual + (residual * weight).sum()
                total_weight = total_weight + weight.sum()
                total_topology_weight = (
                    total_topology_weight
                    + (
                        topology_weight.to(dtype=torch.float32)
                        * valid.to(dtype=torch.float32)
                    ).sum()
                )
                total_valid_pairs = (
                    total_valid_pairs + valid.to(dtype=torch.float32).sum()
                )
        subtb_loss = total_loss / total_weight.clamp_min(1.0)
        residual_abs = total_abs / total_weight.clamp_min(1.0)
        residual_mean = total_residual / total_weight.clamp_min(1.0)
        residual_variance = (subtb_loss - residual_mean.square()).clamp_min(0.0)
        root_log_flows = sample_batch.state_log_flows[:, :, 0].to(dtype=torch.float32)
        centered_root = root_log_flows - root_log_flows.mean()
        return SubgraphSubTrajectoryBalanceLossOutput(
            loss=subtb_loss,
            subtb_loss=subtb_loss,
            residual_abs=residual_abs,
            residual_variance=residual_variance,
            root_abs=root_log_flows.abs().mean(),
            success_rate=sample_batch.success_rate.to(dtype=torch.float32),
            log_z_mean=root_log_flows.mean(),
            log_z_variance=centered_root.square().mean(),
            average_terminal_commit_candidate_count=sample_batch.terminal_commit_candidate_counts.to(
                dtype=torch.float32
            ).mean(),
            average_terminal_gold_answer_count=sample_batch.terminal_gold_answer_counts.to(
                dtype=torch.float32
            ).mean(),
            average_terminal_component_count=sample_batch.terminal_component_counts.to(
                dtype=torch.float32
            ).mean(),
            average_topology_weight=total_topology_weight
            / total_valid_pairs.clamp_min(1.0),
        )


__all__ = [
    "SubgraphSubTrajectoryBalanceLoss",
    "SubgraphSubTrajectoryBalanceLossOutput",
]
