from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReachabilityAnalysis:
    """Shared reachability result container used by evaluation backends."""

    terminal_mass: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    gold_total_mass: float
    answer_prob_ci_low: torch.Tensor | None = None
    answer_prob_ci_high: torch.Tensor | None = None
    gold_total_mass_ci_low: float | None = None
    gold_total_mass_ci_high: float | None = None
    ci_confidence_level: float | None = None
    retrieval_answer_entity_ids: torch.Tensor | None = None
    retrieval_answer_probs: torch.Tensor | None = None
    success_by_step: torch.Tensor | None = None
    log_terminal_mass: torch.Tensor | None = None
    log_answer_probs: torch.Tensor | None = None
    log_gold_total_mass: float | None = None
    log_retrieval_answer_probs: torch.Tensor | None = None
    log_success_by_step: torch.Tensor | None = None
    inference_mode: str | None = None
    probe_count: int | None = None
    remaining_mass_upper: float | None = None
    stop_reason: str | None = None
    coverage_certified: bool | None = None


@dataclass(frozen=True)
class EdgeSupportAnalysis:
    edge_success_mass: torch.Tensor
    edge_conditional_success_prob: torch.Tensor
    gold_mass: float


__all__ = ["EdgeSupportAnalysis", "ReachabilityAnalysis"]
