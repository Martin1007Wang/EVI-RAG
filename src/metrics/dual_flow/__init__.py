from __future__ import annotations

from .answer_metrics import (
    extract_sample_ids,
    extract_answer_entity_ids,
    compute_answer_recall,
    compute_answer_hit,
)
from .config import CompositeScoreConfig, resolve_composite_score_cfg
from .export import DualFlowRolloutExporter, GraphExportInputs, RolloutExportInputs
from .rollout_metrics import DualFlowRolloutMetrics
from .rollout_ops import (
    reduce_rollout_metrics,
    stack_rollout_metrics,
    finalize_rollout_metrics,
    compute_terminal_hits,
    compute_terminal_hit_prefixes,
    compute_composite_score,
    compute_reward_gap,
    compute_diag_metrics,
    build_potential_metrics,
)

__all__ = [
    "extract_sample_ids",
    "extract_answer_entity_ids",
    "compute_answer_recall",
    "compute_answer_hit",
    "CompositeScoreConfig",
    "resolve_composite_score_cfg",
    "DualFlowRolloutExporter",
    "GraphExportInputs",
    "RolloutExportInputs",
    "DualFlowRolloutMetrics",
    "reduce_rollout_metrics",
    "stack_rollout_metrics",
    "finalize_rollout_metrics",
    "compute_terminal_hits",
    "compute_terminal_hit_prefixes",
    "compute_composite_score",
    "compute_reward_gap",
    "compute_diag_metrics",
    "build_potential_metrics",
]
