from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.outputs import BaseModelOutput, DeterministicOutput, ProbabilisticOutput

from .output import LossOutput


class BaseLoss(nn.Module, ABC):
    """Base interface for all retriever losses."""

    @abstractmethod
    def forward(self, model_output: BaseModelOutput, targets: torch.Tensor, **kwargs) -> LossOutput:
        raise NotImplementedError


class PointwiseLoss(BaseLoss):
    """Binary classification style loss that evaluates each edge independently."""

    _EPS = 1e-6

    def forward(self, model_output: BaseModelOutput, targets: torch.Tensor, **kwargs) -> LossOutput:
        compute_result = self._compute_pointwise(model_output, targets, **kwargs)
        info: Dict[str, Dict[str, float]] = {}
        if isinstance(compute_result, tuple) and len(compute_result) == 3:
            losses, probs, info = compute_result
        else:
            losses, probs = compute_result  # type: ignore[misc]

        components = dict(info.get("components", {}))
        metrics = dict(info.get("metrics", {}))
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.any():
            metrics.setdefault("prob/pos_mean", float(probs[pos_mask].mean().item()))
        if neg_mask.any():
            metrics.setdefault("prob/neg_mean", float(probs[neg_mask].mean().item()))
        if pos_mask.any() and neg_mask.any():
            separation = metrics["prob/pos_mean"] - metrics["prob/neg_mean"]
            metrics.setdefault("prob/separation", separation)
        metrics.setdefault("dataset/pos_ratio", float(pos_mask.float().mean().item()) if targets.numel() else 0.0)

        loss = losses.mean()
        return LossOutput(loss=loss, components=components, metrics=metrics)

    @abstractmethod
    def _compute_pointwise(
        self,
        model_output: BaseModelOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DeterministicLoss(PointwiseLoss):
    """Base class for pointwise losses that operate on deterministic logits."""

    def _compute_pointwise(
        self,
        model_output: DeterministicOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Dict[str, float]]]:
        losses = self._compute_base_loss(model_output, targets, **kwargs)
        probs = torch.sigmoid(model_output.raw_logits)
        components = {"prediction": float(losses.detach().mean().item())}
        return losses, probs, {"components": components, "metrics": {}}

    @abstractmethod
    def _compute_base_loss(
        self,
        model_output: DeterministicOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class ProbabilisticLoss(PointwiseLoss):
    """Base class for evidential (Beta) losses."""

    def _compute_pointwise(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model_output.ensure_moments(self._EPS)
        alpha = torch.clamp(model_output.alpha, min=self._EPS)
        beta = torch.clamp(model_output.beta, min=self._EPS)
        probs = torch.clamp(alpha / (alpha + beta), self._EPS, 1.0 - self._EPS)
        losses = F.binary_cross_entropy(probs, targets, reduction="none")
        return losses, probs


class ContrastiveLoss(BaseLoss, ABC):
    """Base class for contrastive objectives."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = float(temperature)

    @abstractmethod
    def forward(self, model_output: BaseModelOutput, targets: torch.Tensor, **kwargs) -> LossOutput:
        raise NotImplementedError


__all__ = [
    "BaseLoss",
    "PointwiseLoss",
    "DeterministicLoss",
    "ProbabilisticLoss",
    "ContrastiveLoss",
]
