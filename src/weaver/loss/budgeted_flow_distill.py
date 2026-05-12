from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.weaver.rollout.schema import RolloutBatch

from .schema import LossOutput


class BudgetedFlowDistillLoss(nn.Module):
    """
    Distill exact budgeted lexicographic flow targets into the joint policy.
    """

    requires_budgeted_flow_trace = True

    def __init__(
        self,
        *,
        policy_kl_weight: float = 1.0,
        terminal_weight: float = 1.0,
        value_weight: float = 0.5,
        **_: object,
    ) -> None:
        super().__init__()
        self.policy_kl_weight = float(policy_kl_weight)
        self.terminal_weight = float(terminal_weight)
        self.value_weight = float(value_weight)

    def forward(self, rollout: RolloutBatch) -> LossOutput:
        traces = rollout.traces
        required = (
            traces.budgeted_policy_kl,
            traces.budgeted_terminal_loss,
            traces.budgeted_value_loss,
            traces.budgeted_valid_mask,
        )
        if any(value is None for value in required):
            raise ValueError("BudgetedFlowDistillLoss requires budgeted-flow traces.")
        assert traces.budgeted_policy_kl is not None
        assert traces.budgeted_terminal_loss is not None
        assert traces.budgeted_value_loss is not None
        assert traces.budgeted_valid_mask is not None
        valid = traces.budgeted_valid_mask.bool()
        policy = _masked_mean(traces.budgeted_policy_kl, valid)
        terminal = _masked_mean(traces.budgeted_terminal_loss, valid)
        value = _masked_mean(traces.budgeted_value_loss, valid)
        loss = (
            self.policy_kl_weight * policy
            + self.terminal_weight * terminal
            + self.value_weight * value
        )
        metrics = {
            "loss/total": loss.detach(),
            "loss/budgeted_flow_distill": loss.detach(),
            "loss/policy_kl": policy.detach(),
            "loss/terminal_huber": terminal.detach(),
            "loss/value_huber": value.detach(),
        }
        optional = {
            "oracle/V_star_mean": traces.oracle_v_star,
            "oracle/terminal_J_mean": traces.oracle_terminal_j,
            "oracle/oracle_stop_prob_mean": traces.oracle_stop_prob,
            "oracle/oracle_edge_entropy": traces.oracle_edge_entropy,
            "oracle/oracle_policy_kl": traces.budgeted_policy_kl,
            "policy/model_stop_prob": traces.model_stop_prob,
            "policy/budgeted_oracle_good_edge_policy_mass": traces.budgeted_oracle_good_edge_policy_mass,
            "policy/sampled_oracle_good_edge_rate": traces.sampled_oracle_good_edge_rate,
        }
        for name, values in optional.items():
            if values is not None:
                metrics[name] = _masked_mean(values, valid).detach()
        return LossOutput(loss=loss, metrics=metrics, per_trajectory_loss=None)


def budgeted_flow_state_loss(
    *,
    model_terminal: torch.Tensor,
    model_value: torch.Tensor,
    model_edge_logprobs: torch.Tensor,
    oracle_terminal: torch.Tensor,
    oracle_value: torch.Tensor,
    oracle_stop_prob: torch.Tensor,
    oracle_edge_probs: torch.Tensor,
    frontier_batch_ids: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rows = int(model_terminal.numel())
    device = model_terminal.device
    dtype = model_terminal.dtype
    row_ids = frontier_batch_ids.to(device=device, dtype=torch.long).view(-1)
    edge_probs = oracle_edge_probs.to(device=device, dtype=dtype).view(-1)
    edge_logp = model_edge_logprobs.to(device=device, dtype=dtype).view(-1)

    kl = oracle_stop_prob.to(device=device, dtype=dtype) * (
        oracle_stop_prob.to(device=device, dtype=dtype).clamp_min(1.0e-12).log()
        - (model_terminal - model_value)
    )
    if edge_probs.numel() > 0:
        edge_terms = edge_probs * (edge_probs.clamp_min(1.0e-12).log() - edge_logp)
        kl.scatter_add_(0, row_ids, edge_terms)
    terminal_loss = F.huber_loss(
        model_terminal,
        oracle_terminal.to(device=device, dtype=dtype),
        reduction="none",
    )
    value_loss = F.huber_loss(
        model_value,
        oracle_value.to(device=device, dtype=dtype),
        reduction="none",
    )
    return {
        "policy_kl": torch.where(valid_mask.to(device=device, dtype=torch.bool), kl, torch.zeros_like(kl)),
        "terminal_loss": torch.where(valid_mask.to(device=device, dtype=torch.bool), terminal_loss, torch.zeros_like(terminal_loss)),
        "value_loss": torch.where(valid_mask.to(device=device, dtype=torch.bool), value_loss, torch.zeros_like(value_loss)),
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any()):
        return values.sum() * 0.0
    return values[mask].mean()


__all__ = ["BudgetedFlowDistillLoss", "budgeted_flow_state_loss"]
