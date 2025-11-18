from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.models.outputs import DeterministicOutput

from .base import ContrastiveLoss, DeterministicLoss
from .output import LossOutput
from .registry import register_loss

logger = logging.getLogger(__name__)

try:  # Optional dependency used by InfoNCE reduction
    import torch_scatter  # type: ignore
except Exception:  # pragma: no cover
    torch_scatter = None


@register_loss(
    "det_bce",
    description="Sigmoid + BCE on deterministic logits",
    family="deterministic",
)
class DeterministicBCELoss(DeterministicLoss):
    """Standard BCE-with-logits loss for deterministic retrievers."""

    def _compute_base_loss(
        self,
        model_output: DeterministicOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(model_output.raw_logits, targets, reduction="none")


@register_loss(
    "det_mse",
    description="MSE on sigmoid probabilities",
    family="deterministic",
)
class DeterministicMSELoss(DeterministicLoss):
    """Mean squared error in probability space."""

    def _compute_pointwise(
        self,
        model_output: DeterministicOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]]]:
        probs = torch.sigmoid(model_output.raw_logits)
        losses = (probs - targets).pow(2)
        components = {"prediction": float(losses.detach().mean().item())}
        return losses, probs, {"components": components, "metrics": {}}


@register_loss(
    "det_infonce",
    description="InfoNCE contrastive objective with optional hard negative mining",
    family="deterministic",
)
class DeterministicInfoNCELoss(ContrastiveLoss):
    """InfoNCE loss operating on deterministic scores."""

    def __init__(
        self,
        temperature: float = 0.1,
        *,
        hard_negative_mining: bool = True,
        hard_negative_ratio: float = 0.5,
    ) -> None:
        super().__init__(temperature)
        self.hard_negative_mining = hard_negative_mining
        self.hard_negative_ratio = hard_negative_ratio
        if self.hard_negative_mining and torch_scatter is None:
            logger.warning("torch_scatter not available; hard-negative mining will be disabled")
            self.hard_negative_mining = False

    def forward(self, model_output: DeterministicOutput, targets: torch.Tensor, query_ids=None, **_) -> LossOutput:
        if torch_scatter is None:
            raise RuntimeError("det_infonce requires torch_scatter to be installed")

        scores = model_output.scores
        device = scores.device
        pos_mask = targets == 1
        if not pos_mask.any():
            metrics = {
                "dataset/pos_ratio": float(pos_mask.float().mean().item()),
            }
            return LossOutput(loss=torch.tensor(0.0, device=device, requires_grad=True), components={}, metrics=metrics)

        if self.hard_negative_mining:
            before_neg = (~pos_mask).sum().item()
            scores, targets, query_ids = self._mine_hard_negatives(scores, targets, query_ids)
            pos_mask = targets == 1
            after_neg = (~pos_mask).sum().item()
            if not pos_mask.any():
                metrics = {
                    "dataset/pos_ratio": 0.0,
                }
                return LossOutput(loss=torch.tensor(0.0, device=device, requires_grad=True), components={}, metrics=metrics)

        logits = scores / self.temperature
        query_logsumexp = torch_scatter.scatter_logsumexp(logits, query_ids, dim=0)
        pos_logits = logits[pos_mask]
        pos_query_ids = query_ids[pos_mask]
        pos_query_logsumexp = query_logsumexp[pos_query_ids]
        pos_losses = -pos_logits + pos_query_logsumexp

        loss = pos_losses.mean()
        metrics = {
            "dataset/pos_ratio": float(pos_mask.float().mean().item()),
        }
        components = {"prediction": float(loss.detach().item())}
        return LossOutput(loss=loss, components=components, metrics=metrics)

    def _mine_hard_negatives(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_mask = labels == 1
        neg_mask = ~pos_mask
        if not neg_mask.any():
            return scores, labels, query_ids

        queries_with_pos = torch.unique(query_ids[pos_mask])
        valid_neg_mask = neg_mask & torch.isin(query_ids, queries_with_pos)
        if not valid_neg_mask.any():
            keep_mask = pos_mask
            return scores[keep_mask], labels[keep_mask], query_ids[keep_mask]

        neg_scores = scores[valid_neg_mask]
        num_keep = max(1, int(neg_scores.numel() * self.hard_negative_ratio))
        if num_keep >= neg_scores.numel():
            keep_mask = pos_mask | valid_neg_mask
            return scores[keep_mask], labels[keep_mask], query_ids[keep_mask]

        _, topk_idx = torch.topk(neg_scores, k=num_keep, largest=True)
        valid_neg_indices = torch.where(valid_neg_mask)[0]
        hard_neg_indices = valid_neg_indices[topk_idx.to(valid_neg_indices.device)]

        keep_mask = torch.zeros_like(scores, dtype=torch.bool)
        keep_mask[pos_mask] = True
        keep_mask[hard_neg_indices] = True
        return scores[keep_mask], labels[keep_mask], query_ids[keep_mask]


__all__ = ["DeterministicBCELoss", "DeterministicMSELoss", "DeterministicInfoNCELoss"]
