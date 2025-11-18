"""Evidential Beta losses used by evidential retrievers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.special import digamma

from src.models.outputs import ProbabilisticOutput

from .base import PointwiseLoss
from .registry import register_loss


def _extract_alpha_beta(model_output: ProbabilisticOutput, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_output.alpha is not None and model_output.beta is not None:
        alpha = torch.clamp(model_output.alpha, min=eps)
        beta = torch.clamp(model_output.beta, min=eps)
    else:
        if model_output.evidence_logits is None and model_output.evidence_logit is None:
            raise ValueError("ProbabilisticOutput requires either alpha/beta or evidence_logits")
        evid = model_output.evidence_logits or model_output.evidence_logit  # type: ignore[operator]
        v_pos, v_neg = evid.unbind(dim=-1)  # type: ignore[union-attr]
        s_pos = F.softplus(v_pos)
        s_neg = F.softplus(v_neg)
        lam_pos = float(getattr(model_output, "lambda_prior_pos", getattr(model_output, "lambda_prior", 1.0)))
        lam_neg = float(getattr(model_output, "lambda_prior_neg", getattr(model_output, "lambda_prior", 1.0)))
        alpha = torch.clamp(lam_pos + s_pos, min=eps)
        beta = torch.clamp(lam_neg + s_neg, min=eps)

    total = torch.clamp(alpha + beta, min=eps)
    mu = torch.clamp(alpha / total, eps, 1.0 - eps)
    return mu, alpha, beta


def _kl_beta(alpha: torch.Tensor, beta: torch.Tensor, a0: float, b0: float) -> torch.Tensor:
    a0_t = torch.tensor(a0, dtype=alpha.dtype, device=alpha.device)
    b0_t = torch.tensor(b0, dtype=alpha.dtype, device=alpha.device)
    log_B_ab = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    log_B_a0b0 = torch.lgamma(a0_t) + torch.lgamma(b0_t) - torch.lgamma(a0_t + b0_t)
    psi_ab = digamma(alpha + beta)
    term1 = log_B_a0b0 - log_B_ab
    term2 = (alpha - a0_t) * (digamma(alpha) - psi_ab)
    term3 = (beta - b0_t) * (digamma(beta) - psi_ab)
    return torch.clamp(term1 + term2 + term3, min=0.0)


@register_loss(
    "edl",
    description="Evidential regression (MSE + variance + KL)",
    family="evidential",
)
class EDLLoss(PointwiseLoss):
    def __init__(
        self,
        *,
        kl_weight: float = 0.1,
        clamp_min: float = 1e-6,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.kl_weight = float(max(kl_weight, 0.0))
        self._eps = float(max(clamp_min, 0.0))
        self._prior_alpha = float(max(prior_alpha, self._eps))
        self._prior_beta = float(max(prior_beta, self._eps))

    def _compute_pointwise(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        targets = targets.to(model_output.scores.dtype)
        mu, alpha, beta = _extract_alpha_beta(model_output, self._eps)
        mse_term = (targets - mu).pow(2)
        variance_term = (mu * (1.0 - mu)) / torch.clamp(alpha + beta + 1.0, min=1.0)
        kl = _kl_beta(alpha, beta, self._prior_alpha, self._prior_beta)
        loss = mse_term + variance_term + self.kl_weight * kl

        components = {
            "prediction": float(mse_term.mean().item()),
            "variance": float(variance_term.mean().item()),
            "kl": float(kl.mean().item()),
        }
        return loss, mu, {"components": components, "metrics": {}}


@register_loss(
    "redl",
    description="Residual evidential loss (MSE only)",
    family="evidential",
)
class REDLLoss(PointwiseLoss):
    def __init__(self, *, clamp_min: float = 1e-6) -> None:
        super().__init__()
        self._eps = float(max(clamp_min, 0.0))

    def _compute_pointwise(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        targets = targets.to(model_output.scores.dtype)
        mu, alpha, beta = _extract_alpha_beta(model_output, self._eps)
        mse_term = (targets - mu).pow(2)

        components = {"prediction": float(mse_term.mean().item())}
        return mse_term, mu, {"components": components, "metrics": {}}


@register_loss(
    "type2",
    description="Type-II marginal likelihood objective",
    family="evidential",
)
class Type2Loss(PointwiseLoss):
    def __init__(self, *, clamp_min: float = 1e-6) -> None:
        super().__init__()
        self._eps = float(max(clamp_min, 0.0))

    def _compute_pointwise(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        targets = targets.to(model_output.scores.dtype)
        mu, alpha, beta = _extract_alpha_beta(model_output, self._eps)
        total = torch.clamp(alpha + beta, min=self._eps)
        log_total = torch.log(total)
        log_alpha = torch.log(torch.clamp(alpha, min=self._eps))
        log_beta = torch.log(torch.clamp(beta, min=self._eps))
        loss = targets * (log_total - log_alpha) + (1.0 - targets) * (log_total - log_beta)

        components = {"prediction": float(loss.mean().item())}
        return loss, mu, {"components": components, "metrics": {}}


@register_loss(
    "vera_lite",
    description="Class-weighted evidential loss with KL regularization (Phoenix-Lite)",
    family="evidential",
)
class VERALiteLoss(PointwiseLoss):
    """Simplified evidential loss with static class weights and KL regularization.

    This loss is intentionally minimal:
    - prediction term: weighted BCE in probability space using Beta mean mu
    - regularization: KL(Beta(alpha,beta) || Beta(prior_alpha, prior_beta))

    It is designed to be paired with a single evidential head where mu is used
    both for ranking and for probability/calibration.
    """

    def __init__(
        self,
        *,
        w_pos: float = 1.0,
        w_neg: float = 1.0,
        kl_weight: float = 0.1,
        clamp_min: float = 1e-6,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_pos = float(max(w_pos, 0.0))
        self.w_neg = float(max(w_neg, 0.0))
        self.kl_weight = float(max(kl_weight, 0.0))
        self._eps = float(max(clamp_min, 0.0))
        self._prior_alpha = float(max(prior_alpha, self._eps))
        self._prior_beta = float(max(prior_beta, self._eps))

    def _compute_pointwise(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        **_: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        # targets in {0,1}, cast to match dtype
        targets = targets.to(model_output.scores.dtype)
        mu, alpha, beta = _extract_alpha_beta(model_output, self._eps)

        mu = torch.clamp(mu, self._eps, 1.0 - self._eps)
        pos_term = -torch.log(mu)
        neg_term = -torch.log(1.0 - mu)

        losses_pred = self.w_pos * targets * pos_term + self.w_neg * (1.0 - targets) * neg_term

        if self.kl_weight > 0.0:
            kl = _kl_beta(alpha, beta, self._prior_alpha, self._prior_beta)
            losses = losses_pred + self.kl_weight * kl
        else:
            kl = torch.zeros_like(losses_pred)
            losses = losses_pred

        components = {
            "prediction": float(losses_pred.mean().item()),
            "kl": float(kl.mean().item()),
        }
        return losses, mu, {"components": components, "metrics": {}}


__all__ = ["EDLLoss", "REDLLoss", "Type2Loss", "VERALiteLoss"]
