from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn


def _safe_f1(precision: torch.Tensor, recall: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return 2 * precision * recall / (precision + recall + eps)


@dataclass
class RewardOutput:
    reward: torch.Tensor
    log_reward: torch.Tensor
    success: torch.Tensor
    pos_precision: torch.Tensor
    pos_recall: torch.Tensor
    pos_f1: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class GFlowNetReward(nn.Module):
    """Unified reward: constant success/failure with optional edge-set F1 shaping."""

    def __init__(
        self,
        *,
        success_reward: float = 1.0,
        failure_reward: float = 0.01,
        shaping_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.shaping_coef = float(shaping_coef)
        if self.success_reward <= 0.0:
            raise ValueError(f"success_reward must be positive; got {self.success_reward}.")
        if self.failure_reward <= 0.0:
            raise ValueError(f"failure_reward must be positive; got {self.failure_reward}.")
        if self.success_reward <= self.failure_reward:
            raise ValueError(
                f"success_reward must be greater than failure_reward; got {self.success_reward} <= {self.failure_reward}."
            )
        if self.shaping_coef < 0.0:
            raise ValueError(f"shaping_coef must be >= 0; got {self.shaping_coef}.")
        self.log_success = float(math.log(self.success_reward))
        self.log_failure = float(math.log(self.failure_reward))

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        answer_hit: torch.Tensor,  # [B]
        **_,
    ) -> RewardOutput:
        device = selected_mask.device
        answer_hit = answer_hit.to(device=device)
        num_graphs = int(answer_hit.numel())
        if num_graphs <= 0:
            empty = torch.zeros(0, device=device, dtype=torch.float32)
            return RewardOutput(
                reward=empty,
                log_reward=empty,
                success=empty,
                pos_precision=empty,
                pos_recall=empty,
                pos_f1=empty,
            )

        edge_batch = edge_batch.to(device=device, dtype=torch.long)
        pred = selected_mask.to(device=device, dtype=torch.float32)
        target = (edge_labels.to(device=device) > 0.5).to(dtype=torch.float32)

        zeros = torch.zeros(num_graphs, device=device, dtype=torch.float32)
        tp = zeros.clone()
        pred_sum = zeros.clone()
        target_sum = zeros.clone()
        if pred.numel() > 0:
            tp.index_add_(0, edge_batch, pred * target)
            pred_sum.index_add_(0, edge_batch, pred)
            target_sum.index_add_(0, edge_batch, target)

        precision = torch.where(pred_sum > 0, tp / pred_sum.clamp(min=1.0), zeros)
        recall = torch.where(target_sum > 0, tp / target_sum.clamp(min=1.0), zeros)
        f1 = _safe_f1(precision, recall)

        log_reward = torch.where(
            answer_hit.to(dtype=torch.bool),
            torch.full((num_graphs,), self.log_success, device=device, dtype=torch.float32),
            torch.full((num_graphs,), self.log_failure, device=device, dtype=torch.float32),
        )
        if self.shaping_coef > 0.0:
            log_reward = log_reward + self.shaping_coef * f1.to(dtype=log_reward.dtype)

        reward = torch.exp(log_reward)
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=answer_hit.to(dtype=torch.float32),
            pos_precision=precision,
            pos_recall=recall,
            pos_f1=f1,
        )


__all__ = ["RewardOutput", "GFlowNetReward"]
