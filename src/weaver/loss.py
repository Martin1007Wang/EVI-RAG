from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    num_states: int
    per_unit_loss: torch.Tensor | None = None

    @staticmethod
    def aggregate(outputs: Sequence[LossOutput]) -> LossOutput:
        if not outputs:
            raise ValueError("Cannot aggregate an empty LossOutput sequence.")
        if len(outputs) == 1:
            return outputs[0]
        loss = torch.stack([x.loss for x in outputs]).mean()
        keys = set(outputs[0].metrics)
        for output in outputs[1:]:
            keys &= set(output.metrics)
        metrics = {
            key: torch.stack(
                [x.metrics[key].to(device=loss.device, dtype=loss.dtype) for x in outputs]
            ).mean()
            for key in sorted(keys)
        }
        parts = [
            x.per_unit_loss.detach().reshape(-1).to(device=loss.device)
            for x in outputs
            if x.per_unit_loss is not None
        ]
        return LossOutput(
            loss=loss,
            metrics=metrics,
            num_states=sum(int(x.num_states) for x in outputs),
            per_unit_loss=None if not parts else torch.cat(parts, dim=0),
        )


class ProbabilityDBLoss(nn.Module):
    def forward(
        self,
        *,
        parent_log_reward: torch.Tensor,
        child_log_reward: torch.Tensor,
        log_backward_prob: torch.Tensor,
        parent_stop_log_prob: torch.Tensor,
        parent_continue_log_prob: torch.Tensor,
        parent_edge_log_prob: torch.Tensor,
        child_stop_log_prob: torch.Tensor,
    ) -> LossOutput:
        residual = (
            child_log_reward
            + log_backward_prob
            + parent_stop_log_prob
            - parent_continue_log_prob
            - parent_edge_log_prob
            - parent_log_reward
            - child_stop_log_prob
        )
        per_unit_loss = residual.square()
        loss = per_unit_loss.mean() if per_unit_loss.numel() > 0 else residual.sum() * 0.0
        metrics = {
            "objective/total": loss.detach(),
            "db/residual_mean": _safe_mean(residual).detach(),
            "db/residual_abs_mean": _safe_mean(residual.abs()).detach(),
            "db/residual_sq_mean": _safe_mean(per_unit_loss).detach(),
        }
        return LossOutput(
            loss=loss,
            metrics=metrics,
            num_states=int(residual.numel()),
            per_unit_loss=per_unit_loss.detach(),
        )


def _safe_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.float().mean()


__all__ = [
    "LossOutput",
    "ProbabilityDBLoss",
]
