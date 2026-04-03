from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .sampler import SubgraphTrajectorySampleBatch


def _zero_metric_tensor() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


@dataclass(frozen=True)
class SubgraphDetailedBalanceLossOutput:
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


class SubgraphDetailedBalanceLoss:
    def __init__(self, **_: object) -> None:
        pass

    def compute(
        self, sample_batch: "SubgraphTrajectorySampleBatch"
    ) -> SubgraphDetailedBalanceLossOutput:
        log_pf = sample_batch.log_pf_actions.to(dtype=torch.float32)
        log_pb = sample_batch.log_pb_actions.to(dtype=torch.float32)
        log_reward = sample_batch.log_reward_actions.to(dtype=torch.float32)
        state_log_flow = sample_batch.state_log_flows.to(dtype=torch.float32)
        action_mask = sample_batch.action_mask.to(dtype=torch.bool)
        termination = sample_batch.termination_action_steps.to(dtype=torch.long)
        max_actions = int(log_pf.size(-1))

        residual = torch.zeros_like(log_pf, dtype=torch.float32)
        for action_step in range(max_actions):
            valid = action_mask[:, :, action_step]
            if not bool(valid.any().item()):
                continue
            current_flow = state_log_flow[:, :, action_step]
            terminal = valid & (termination == int(action_step + 1))
            if bool(terminal.any().item()):
                residual[:, :, action_step] = torch.where(
                    terminal,
                    current_flow
                    + log_pf[:, :, action_step]
                    - log_reward[:, :, action_step],
                    residual[:, :, action_step],
                )
            nonterminal = valid & ~terminal
            if bool(nonterminal.any().item()):
                if (action_step + 1) >= max_actions:
                    raise RuntimeError(
                        "Detailed Balance requires next-state flow for nonterminal actions."
                    )
                residual[:, :, action_step] = torch.where(
                    nonterminal,
                    current_flow
                    + log_pf[:, :, action_step]
                    - state_log_flow[:, :, action_step + 1]
                    - log_pb[:, :, action_step],
                    residual[:, :, action_step],
                )

        valid_residuals = residual[action_mask]
        if int(valid_residuals.numel()) <= 0:
            db_loss = residual.new_zeros((), dtype=torch.float32)
            residual_abs = residual.new_zeros((), dtype=torch.float32)
            residual_variance = residual.new_zeros((), dtype=torch.float32)
        else:
            db_loss = valid_residuals.square().mean()
            residual_abs = valid_residuals.abs().mean()
            residual_variance = valid_residuals.var(unbiased=False)

        root_log_flows = sample_batch.state_log_flows[:, :, 0].to(dtype=torch.float32)
        centered_root = root_log_flows - root_log_flows.mean()
        return SubgraphDetailedBalanceLossOutput(
            loss=db_loss,
            db_loss=db_loss,
            residual_abs=residual_abs,
            residual_variance=residual_variance,
            root_abs=root_log_flows.abs().mean(),
            success_rate=sample_batch.success_rate.to(dtype=torch.float32),
            log_z_mean=root_log_flows.mean(),
            log_z_variance=centered_root.square().mean(),
            average_terminal_answer_candidate_count=sample_batch.terminal_answer_candidate_counts.to(
                dtype=torch.float32
            ).mean(),
            average_terminal_gold_answer_count=sample_batch.terminal_gold_answer_counts.to(
                dtype=torch.float32
            ).mean(),
            average_terminal_component_count=sample_batch.terminal_component_counts.to(
                dtype=torch.float32
            ).mean(),
        )


__all__ = [
    "SubgraphDetailedBalanceLoss",
    "SubgraphDetailedBalanceLossOutput",
]
