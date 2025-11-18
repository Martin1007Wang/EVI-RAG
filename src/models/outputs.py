from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

DEFAULT_EPS = 1e-12


def bernoulli_entropy(probs: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """Entropy of a Bernoulli with success probability `probs`."""
    probs = torch.clamp(probs, eps, 1.0 - eps)
    return -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))


def beta_expected_entropy(
    alpha: torch.Tensor, beta: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """E[H(Bernoulli(p))] where p ~ Beta(alpha, beta)."""
    alpha = torch.clamp(alpha, min=eps)
    beta = torch.clamp(beta, min=eps)
    total = torch.clamp(alpha + beta, min=eps)

    digamma_total = torch.digamma(total + 1.0)
    alpha_term = (alpha / total) * (torch.digamma(alpha + 1.0) - digamma_total)
    beta_term = (beta / total) * (torch.digamma(beta + 1.0) - digamma_total)
    entropy = -(alpha_term + beta_term)
    return torch.clamp(entropy, min=0.0)


def beta_mutual_information(
    alpha: torch.Tensor, beta: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Mutual information between label and model parameters under Beta-Bernoulli."""
    total = torch.clamp(alpha + beta, min=eps)
    predictive = bernoulli_entropy(torch.clamp(alpha / total, eps, 1.0 - eps), eps=eps)
    aleatoric = beta_expected_entropy(alpha, beta, eps=eps)
    return torch.clamp(predictive - aleatoric, min=0.0)


@dataclass
class RetrieverOutput:
    """Unified forward output for retriever models.

    scores: probability-like scores used for ranking (shape aligns with query_ids).
    logits: raw logit scores if available (deterministic models).
    alpha/beta/aleatoric/epistemic: evidential parameters and uncertainties (evidential models).
    evidence_logits: raw evidential logits (two channels), optional for debugging/diagnostics.
    """

    scores: torch.Tensor
    query_ids: torch.Tensor
    logits: torch.Tensor | None = None

    alpha: torch.Tensor | None = None
    beta: torch.Tensor | None = None
    aleatoric: torch.Tensor | None = None
    epistemic: torch.Tensor | None = None
    evidence_logits: torch.Tensor | None = None

    # Evidential priors (kept for loss functions)
    lambda_prior: float = 1.0
    lambda_prior_pos: float = 1.0
    lambda_prior_neg: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Return a minimal dict representation with non-None fields."""
        fields = [
            "scores",
            "query_ids",
            "logits",
            "alpha",
            "beta",
            "aleatoric",
            "epistemic",
            "evidence_logits",
            "lambda_prior",
            "lambda_prior_pos",
            "lambda_prior_neg",
        ]
        out: Dict[str, Any] = {}
        for name in fields:
            value = getattr(self, name, None)
            if value is not None:
                out[name] = value
        return out


class DeterministicOutput(RetrieverOutput):
    """Retriever output for deterministic scorer."""

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.logits is None:
            raise ValueError("DeterministicOutput requires logits to be provided.")
        # Ensure probability scores align with logits if not explicitly set.
        if self.scores is None:
            self.scores = torch.sigmoid(self.logits)
        # Deterministic models don't carry evidential fields.
        self.alpha = None
        self.beta = None
        self.aleatoric = None
        self.epistemic = None
        self.evidence_logits = None

    @property
    def raw_logits(self) -> torch.Tensor:
        return self.logits  # type: ignore[return-value]


@dataclass
class ProbabilisticOutput(RetrieverOutput):
    """Retriever output for evidential scorers."""

    posterior_mean: torch.Tensor | None = None

    def ensure_moments(self, eps: float = 1e-6) -> None:
        """Populate alpha/beta/mu/uncertainties while preserving existing ranking scores."""
        if (
            self.alpha is not None
            and self.beta is not None
            and self.aleatoric is not None
            and self.epistemic is not None
            and self.posterior_mean is not None
        ):
            return

        if self.evidence_logits is None and (self.alpha is None or self.beta is None):
            raise ValueError("ProbabilisticOutput requires evidence_logits or alpha/beta.")

        if self.alpha is None or self.beta is None:
            v_pos, v_neg = self.evidence_logits.unbind(dim=-1)  # type: ignore[union-attr]
            s_pos = F.softplus(v_pos)
            s_neg = F.softplus(v_neg)
            lam_pos = float(self.lambda_prior_pos or self.lambda_prior)
            lam_neg = float(self.lambda_prior_neg or self.lambda_prior)
            alpha = torch.clamp(lam_pos + s_pos, min=eps)
            beta = torch.clamp(lam_neg + s_neg, min=eps)
        else:
            alpha = torch.clamp(self.alpha, min=eps)
            beta = torch.clamp(self.beta, min=eps)

        total = torch.clamp(alpha + beta, min=eps)
        mu = torch.clamp(alpha / total, eps, 1.0 - eps)
        aleatoric = beta_expected_entropy(alpha, beta)
        epistemic = beta_mutual_information(alpha, beta)

        self.alpha = alpha
        self.beta = beta
        self.posterior_mean = mu
        self.aleatoric = aleatoric
        self.epistemic = epistemic
        # Evidential models do not produce logits by default.
        if self.logits is None and self.evidence_logits is not None:
            self.logits = None

    def to_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        self.ensure_moments()
        payload = super().to_dict()
        if self.posterior_mean is not None:
            payload["posterior_mean"] = self.posterior_mean
        return payload

    @property
    def evidence_logit(self) -> torch.Tensor | None:
        return self.evidence_logits


# Backwards compatibility aliases
BaseModelOutput = RetrieverOutput

__all__ = [
    "bernoulli_entropy",
    "beta_expected_entropy",
    "beta_mutual_information",
    "RetrieverOutput",
    "DeterministicOutput",
    "ProbabilisticOutput",
    "BaseModelOutput",
]
