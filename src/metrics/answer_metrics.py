from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

import torch
from torchmetrics import MeanMetric, SumMetric

from src.graph_runtime import TrajectoryBatch
from src.models.configs import SearchEvalConfig
from src.models.gflownet import (
    PreparedSearchBatch,
    SearchPolicyProtocol,
    StartDistributionError,
    TrajectorySamplerProtocol,
)

from .base import BaseMetricRuntime
from .prediction_io import (
    PredictionCodecProtocol,
    iter_jsonl_records,
    jsonl_has_records,
)
from .protocol import MetricEvaluationOutput
from .ranking_metrics import compute_topk_set_metrics, mean_metric_dicts

UNSPECIFIED_INFERENCE_MODE = "unspecified"
UNSPECIFIED_MASS_REFERENCE = "unspecified"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReachabilityAnalysis:
    terminal_mass: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    gold_answer_mass: float
    answer_prob_ci_low: torch.Tensor | None = None
    answer_prob_ci_high: torch.Tensor | None = None
    gold_answer_mass_ci_low: float | None = None
    gold_answer_mass_ci_high: float | None = None
    ci_confidence_level: float | None = None


@dataclass(frozen=True)
class ReachabilityRanking:
    answer_entity_ids: torch.Tensor
    answer_probs: torch.Tensor
    answer_prob_ci_low: torch.Tensor | None = None
    answer_prob_ci_high: torch.Tensor | None = None


@dataclass(frozen=True)
class SearchDiagnostics:
    inference_mode: str
    probe_count: int
    remaining_mass_upper: float
    stop_reason: str
    coverage_certified: bool
    covered_mass_ci_low: float | None = None
    covered_mass_ci_high: float | None = None
    ci_confidence_level: float | None = None


@dataclass(frozen=True)
class EdgeSupportAnalysis:
    edge_success_mass: torch.Tensor
    edge_conditional_success_prob: torch.Tensor
    success_rollout_mass: float


@dataclass(frozen=True)
class EdgeRecord:
    edge_id: int
    src_entity_id: int
    relation_id: int
    dst_entity_id: int


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    path_rank: int
    log_prob: float
    prob: float
    cumulative_mass: float
    terminal_entity_id: int
    is_gold: bool
    edges: list[EdgeRecord]
    start_entity_id: int | None = None
    answer_rank: int = 0
    support_rank: int = 0
    conditional_prob: float = 0.0
    conditional_cumulative_mass: float = 0.0


@dataclass(frozen=True)
class AnswerPosteriorRecord:
    answer_entity_id: int
    prob: float
    cumulative_mass: float
    is_gold: bool
    is_selected: bool = False
    support_mass: float = 0.0
    support_conditioned_mass: float = 0.0
    support_path_count: int = 0
    prob_ci_low: float = 0.0
    prob_ci_high: float = 0.0


@dataclass(frozen=True)
class AnswerSupportRecord:
    answer_entity_id: int
    answer_rank: int
    prob: float
    cumulative_mass: float
    is_gold: bool
    is_selected: bool
    support_mass: float
    support_conditioned_mass: float
    support_path_count: int
    trajectories: list[TrajectoryRecord]
    prob_ci_low: float = 0.0
    prob_ci_high: float = 0.0


@dataclass(frozen=True)
class SupportWindowResult:
    """Artifact-facing evaluation output.

    Backend-specific metadata should be filled explicitly by the evaluator.
    The schema defaults stay backend-neutral so ad hoc constructions in tests or
    tools do not silently claim Monte Carlo provenance.
    """

    sample_id: str
    dataset_scope: str
    mass_threshold: float
    window_size: int
    covered_mass: float
    residual_mass: float
    gold_answer_mass: float
    covered_gold_answer_mass: float
    missed_gold_answer_mass: float
    unique_answer_count: int
    unique_path_count: int
    gold_answer_entity_ids: list[int]
    start_entity_ids: list[int]
    trajectories: list[TrajectoryRecord]
    inference_mode: str = UNSPECIFIED_INFERENCE_MODE
    ci_confidence_level: float | None = None
    covered_mass_ci_low: float | None = None
    covered_mass_ci_high: float | None = None
    gold_answer_mass_ci_low: float | None = None
    gold_answer_mass_ci_high: float | None = None
    answer_mass_threshold: float = 1.0
    support_mass_threshold: float = 1.0
    probe_count: int = 0
    emit_path_count: int = 0
    remaining_mass_upper: float = 0.0
    stop_reason: str = ""
    coverage_certified: bool = False
    answer_mass_reference: str = UNSPECIFIED_MASS_REFERENCE
    support_mass_reference: str = UNSPECIFIED_MASS_REFERENCE
    selected_answer_ids: list[int] = field(default_factory=list)
    answer_posterior: list[AnswerPosteriorRecord] = field(default_factory=list)
    answer_support: list[AnswerSupportRecord] = field(default_factory=list)


@dataclass(frozen=True)
class SupportWindowLabelRecord:
    sample_id: str
    question: str
    start_entity_ids: list[int]
    answer_entity_ids: list[int]
    a_entity_in_graph: bool


_MASS_TOLERANCE = 1.0e-6


# ---------------------------------------------------------------------------
# Ranking and aggregate metrics
# ---------------------------------------------------------------------------


def ranking_from_analysis(analysis: ReachabilityAnalysis) -> ReachabilityRanking:
    return ReachabilityRanking(
        answer_entity_ids=analysis.answer_entity_ids,
        answer_probs=analysis.answer_probs,
        answer_prob_ci_low=analysis.answer_prob_ci_low,
        answer_prob_ci_high=analysis.answer_prob_ci_high,
    )


def build_answer_posterior(
    *,
    ranking: ReachabilityRanking,
    gold_answers: set[int],
    answer_mass_threshold: float,
    total_mass_reference: float | None = None,
) -> tuple[list[AnswerPosteriorRecord], list[int]]:
    answer_ids = [int(value) for value in ranking.answer_entity_ids.tolist()]
    answer_probs = [float(value) for value in ranking.answer_probs.tolist()]
    answer_ci_low = (
        [float(value) for value in ranking.answer_prob_ci_low.tolist()]
        if ranking.answer_prob_ci_low is not None
        and int(ranking.answer_prob_ci_low.numel()) == len(answer_ids)
        else [0.0 for _ in answer_ids]
    )
    answer_ci_high = (
        [float(value) for value in ranking.answer_prob_ci_high.tolist()]
        if ranking.answer_prob_ci_high is not None
        and int(ranking.answer_prob_ci_high.numel()) == len(answer_ids)
        else [0.0 for _ in answer_ids]
    )
    order = sorted(
        range(len(answer_ids)),
        key=lambda idx: (-answer_probs[idx], answer_ids[idx]),
    )
    total_mass = float(math.fsum(answer_probs))
    reference_mass = (
        float(total_mass_reference)
        if total_mass_reference is not None
        else float(total_mass)
    )
    cumulative = 0.0
    records: list[AnswerPosteriorRecord] = []
    selected_answer_ids: list[int] = []
    threshold_mass = float(answer_mass_threshold) * max(reference_mass, 0.0)
    for idx in order:
        answer_id = answer_ids[idx]
        prob = answer_probs[idx]
        previous_cumulative = cumulative
        cumulative += prob
        is_selected = prob > 0.0 and (
            not selected_answer_ids
            or previous_cumulative < threshold_mass - _MASS_TOLERANCE
        )
        if is_selected:
            selected_answer_ids.append(answer_id)
        records.append(
            AnswerPosteriorRecord(
                answer_entity_id=answer_id,
                prob=prob,
                cumulative_mass=min(cumulative, 1.0),
                is_gold=answer_id in gold_answers,
                is_selected=is_selected,
                prob_ci_low=max(float(answer_ci_low[idx]), 0.0),
                prob_ci_high=max(float(answer_ci_high[idx]), 0.0),
            )
        )
    return records, selected_answer_ids


def compute_rank_metrics(
    *,
    answer_records: list[AnswerPosteriorRecord],
    answer_top_ks: tuple[int, ...],
    gold_answer_mass: float | None = None,
) -> dict[str, float]:
    gold_answers = {
        int(record.answer_entity_id)
        for record in answer_records
        if bool(record.is_gold)
    }
    ordered_answer_ids = [int(record.answer_entity_id) for record in answer_records]
    metrics: dict[str, float] = {
        "answer/gold_answer_mass": (
            float(gold_answer_mass)
            if gold_answer_mass is not None
            else float(
                sum(
                    float(record.prob)
                    for record in answer_records
                    if bool(record.is_gold)
                )
            )
        )
    }
    metrics.update(
        compute_topk_set_metrics(
            ranked_ids=ordered_answer_ids,
            relevant_ids=gold_answers,
            top_ks=answer_top_ks,
            prefix="answer",
            include_precision=False,
            include_f1=False,
        )
    )
    return metrics


def aggregate_rank_metrics(
    *,
    results: list[SupportWindowResult],
    answer_top_ks: tuple[int, ...],
) -> dict[str, float]:
    if not results:
        return {}
    return mean_metric_dicts(
        [
            compute_rank_metrics(
                answer_records=result.answer_posterior,
                answer_top_ks=answer_top_ks,
                gold_answer_mass=float(result.gold_answer_mass),
            )
            for result in results
        ]
    )


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _context_stats(
    result: SupportWindowResult,
    *,
    trajectories: list[TrajectoryRecord] | None = None,
) -> tuple[float, float]:
    answer_set = set(result.gold_answer_entity_ids)
    if not answer_set:
        return 0.0, 0.0
    start_set = set(result.start_entity_ids)
    context_nodes = set(start_set)
    path_records = result.trajectories if trajectories is None else trajectories
    for trajectory in path_records:
        context_nodes.add(int(trajectory.terminal_entity_id))
        for edge in trajectory.edges:
            context_nodes.add(int(edge.src_entity_id))
            context_nodes.add(int(edge.dst_entity_id))
    recall = float(len(context_nodes & answer_set)) / float(len(answer_set))
    hit = 1.0 if recall > 0.0 else 0.0
    return recall, hit


def _support_path_diversity(result: SupportWindowResult) -> float:
    per_answer_paths: dict[int, list[set[int]]] = {}
    for trajectory in result.trajectories:
        per_answer_paths.setdefault(int(trajectory.terminal_entity_id), []).append(
            {int(edge.edge_id) for edge in trajectory.edges}
        )
    if not per_answer_paths:
        return 0.0
    per_answer_diversity: list[float] = []
    for edge_sets in per_answer_paths.values():
        if len(edge_sets) <= 1:
            per_answer_diversity.append(1.0)
            continue
        pairwise_overlap: list[float] = []
        for left_idx in range(len(edge_sets)):
            for right_idx in range(left_idx + 1, len(edge_sets)):
                left = edge_sets[left_idx]
                right = edge_sets[right_idx]
                denom = max(len(left), len(right), 1)
                pairwise_overlap.append(float(len(left & right)) / float(denom))
        per_answer_diversity.append(1.0 - _safe_mean(pairwise_overlap))
    return _safe_mean(per_answer_diversity)


def compute_support_metrics(results: list[SupportWindowResult]) -> dict[str, float]:
    if not results:
        return {}

    metric_values: dict[str, list[float]] = {
        "support/hit": [],
        "support/recall": [],
        "support/path_count": [],
        "support/path_mass": [],
        "support/covered_gold_answer_mass": [],
        "support/missed_gold_answer_mass": [],
        "support/diversity": [],
        "search/probe_count": [],
        "search/remaining_mass_upper": [],
        "search/coverage_rate": [],
    }

    for result in results:
        recall, hit = _context_stats(result)
        metric_values["support/hit"].append(hit)
        metric_values["support/recall"].append(recall)
        metric_values["support/path_count"].append(float(result.window_size))
        metric_values["support/path_mass"].append(float(result.covered_mass))
        metric_values["support/covered_gold_answer_mass"].append(
            float(result.covered_gold_answer_mass)
        )
        metric_values["support/missed_gold_answer_mass"].append(
            float(result.missed_gold_answer_mass)
        )
        metric_values["support/diversity"].append(_support_path_diversity(result))
        metric_values["search/probe_count"].append(float(result.probe_count))
        metric_values["search/remaining_mass_upper"].append(
            float(result.remaining_mass_upper)
        )
        metric_values["search/coverage_rate"].append(
            1.0 if result.coverage_certified else 0.0
        )

    return {name: _safe_mean(values) for name, values in metric_values.items()}


def summarize_reachability_metrics(
    *,
    results: list[SupportWindowResult],
    answer_top_ks: tuple[int, ...],
    metrics_profile: str,
    invalid_start_reason: str = "invalid_start_candidates",
) -> tuple[dict[str, float], dict[str, float]]:
    rank_metrics = aggregate_rank_metrics(
        results=results,
        answer_top_ks=answer_top_ks,
    )
    diagnostics: dict[str, float] = {}
    if metrics_profile != "rank_only":
        diagnostics.update(compute_support_metrics(results))
    invalid_start_count = sum(
        1 for result in results if result.stop_reason == invalid_start_reason
    )
    if results:
        diagnostics["invalid_start_count"] = float(invalid_start_count)
        diagnostics["invalid_start_rate"] = float(invalid_start_count) / float(
            len(results)
        )
    return rank_metrics, diagnostics


# ---------------------------------------------------------------------------
# Support window construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredTrajectory:
    start_node: int
    terminal_node: int
    answer_entity_id: int
    edge_ids: tuple[int, ...]
    log_prob: float
    is_gold: bool

    @property
    def prob(self) -> float:
        return float(math.exp(self.log_prob))


def graph_gold_answers(*, batch: TrajectoryBatch) -> set[int]:
    return {int(value) for value in batch.answer_entity_ids.tolist()}


def graph_start_entity_ids(*, batch: TrajectoryBatch) -> list[int]:
    return [
        int(batch.node_global_ids[node_idx].item())
        for node_idx in batch.q_local_indices.tolist()
    ]


def build_edge_records(
    *, batch: TrajectoryBatch, edge_ids: tuple[int, ...]
) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    for edge_id in edge_ids:
        src = int(batch.edge_index[0, edge_id].item())
        dst = int(batch.edge_index[1, edge_id].item())
        records.append(
            EdgeRecord(
                edge_id=int(edge_id),
                src_entity_id=int(batch.node_global_ids[src].item()),
                relation_id=int(batch.edge_rel_global[edge_id].item()),
                dst_entity_id=int(batch.node_global_ids[dst].item()),
            )
        )
    return records


def _path_edge_overlap(lhs: DiscoveredTrajectory, rhs: DiscoveredTrajectory) -> float:
    if lhs.edge_ids == rhs.edge_ids:
        return 1.0
    if not lhs.edge_ids or not rhs.edge_ids:
        return 0.0
    lhs_edges = set(lhs.edge_ids)
    rhs_edges = set(rhs.edge_ids)
    overlap = len(lhs_edges & rhs_edges)
    if overlap == 0:
        return 0.0
    denom = max(len(lhs_edges), len(rhs_edges))
    return float(overlap) / float(denom)


def _select_support_paths_for_answer(
    *,
    answer_paths: list[DiscoveredTrajectory],
    target_mass: float,
    overlap_penalty: float,
) -> tuple[list[DiscoveredTrajectory], float]:
    if target_mass <= 0.0 or not answer_paths:
        return [], 0.0
    selected: list[DiscoveredTrajectory] = []
    remaining = list(answer_paths)
    accumulated_mass = 0.0
    while remaining and accumulated_mass + _MASS_TOLERANCE < target_mass:
        if not selected or overlap_penalty <= 0.0:
            best_idx = max(
                range(len(remaining)),
                key=lambda idx: (remaining[idx].prob, -len(remaining[idx].edge_ids)),
            )
        else:
            best_idx = max(
                range(len(remaining)),
                key=lambda idx: (
                    math.log(max(remaining[idx].prob, _MASS_TOLERANCE))
                    - overlap_penalty
                    * max(
                        _path_edge_overlap(remaining[idx], chosen)
                        for chosen in selected
                    ),
                    remaining[idx].prob,
                    -len(remaining[idx].edge_ids),
                ),
            )
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        accumulated_mass += chosen.prob
    return selected, accumulated_mass


def _build_support_outputs(
    *,
    batch: TrajectoryBatch,
    discovered_paths: list[DiscoveredTrajectory],
    answer_records: list[AnswerPosteriorRecord],
    selected_set: set[int],
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
    support_answer_upper_bounds: dict[int, float] | None = None,
) -> tuple[list[TrajectoryRecord], list[AnswerPosteriorRecord]]:
    answer_mass = {
        record.answer_entity_id: float(record.prob) for record in answer_records
    }
    support_paths: list[TrajectoryRecord] = []
    support_summary: dict[int, tuple[float, int]] = {}
    answer_rank = {
        record.answer_entity_id: idx
        for idx, record in enumerate(answer_records, start=1)
    }
    for answer_id in selected_set:
        answer_paths = [
            path for path in discovered_paths if path.answer_entity_id == answer_id
        ]
        answer_paths.sort(
            key=lambda item: (-item.prob, item.answer_entity_id, item.edge_ids)
        )
        target_mass = float(support_mass_threshold) * (
            float(support_answer_upper_bounds[answer_id])
            if support_answer_upper_bounds is not None
            and answer_id in support_answer_upper_bounds
            else answer_mass.get(answer_id, 0.0)
        )
        selected_paths, cumulative_mass = _select_support_paths_for_answer(
            answer_paths=answer_paths,
            target_mass=target_mass,
            overlap_penalty=float(support_path_overlap_penalty),
        )
        kept_count = 0
        cumulative_selected_mass = 0.0
        for support_rank, path in enumerate(selected_paths, start=1):
            cumulative_selected_mass += path.prob
            support_paths.append(
                TrajectoryRecord(
                    sample_id=batch.sample_ids[0],
                    path_rank=0,
                    log_prob=float(path.log_prob),
                    prob=float(path.prob),
                    cumulative_mass=0.0,
                    terminal_entity_id=int(path.answer_entity_id),
                    is_gold=bool(path.is_gold),
                    edges=build_edge_records(batch=batch, edge_ids=path.edge_ids),
                    start_entity_id=int(batch.node_global_ids[path.start_node].item()),
                    answer_rank=int(answer_rank.get(answer_id, 0)),
                    support_rank=int(support_rank),
                    conditional_prob=(
                        float(path.prob) / answer_mass[answer_id]
                        if answer_mass.get(answer_id, 0.0) > 0.0
                        else 0.0
                    ),
                    conditional_cumulative_mass=(
                        cumulative_selected_mass / answer_mass[answer_id]
                        if answer_mass.get(answer_id, 0.0) > 0.0
                        else 0.0
                    ),
                )
            )
            kept_count = support_rank
        support_summary[answer_id] = (cumulative_mass, kept_count)
    support_paths.sort(
        key=lambda record: (
            -float(record.prob),
            int(record.answer_rank),
            int(record.support_rank),
            int(record.terminal_entity_id),
        )
    )
    cumulative_global_mass = 0.0
    ranked_support_paths: list[TrajectoryRecord] = []
    for path_rank, record in enumerate(support_paths, start=1):
        cumulative_global_mass += float(record.prob)
        ranked_support_paths.append(
            TrajectoryRecord(
                sample_id=record.sample_id,
                path_rank=int(path_rank),
                log_prob=float(record.log_prob),
                prob=float(record.prob),
                cumulative_mass=float(cumulative_global_mass),
                terminal_entity_id=int(record.terminal_entity_id),
                is_gold=bool(record.is_gold),
                edges=list(record.edges),
                start_entity_id=record.start_entity_id,
                answer_rank=int(record.answer_rank),
                support_rank=int(record.support_rank),
                conditional_prob=float(record.conditional_prob),
                conditional_cumulative_mass=float(record.conditional_cumulative_mass),
            )
        )
    updated_answers: list[AnswerPosteriorRecord] = []
    for record in answer_records:
        support_mass, support_path_count = support_summary.get(
            record.answer_entity_id, (0.0, 0)
        )
        conditioned_mass = (
            support_mass / float(record.prob) if float(record.prob) > 0.0 else 0.0
        )
        updated_answers.append(
            AnswerPosteriorRecord(
                answer_entity_id=int(record.answer_entity_id),
                prob=float(record.prob),
                cumulative_mass=float(record.cumulative_mass),
                is_gold=bool(record.is_gold),
                is_selected=bool(record.answer_entity_id in selected_set),
                support_mass=float(support_mass),
                support_conditioned_mass=float(conditioned_mass),
                support_path_count=int(support_path_count),
                prob_ci_low=float(record.prob_ci_low),
                prob_ci_high=float(record.prob_ci_high),
            )
        )
    return ranked_support_paths, updated_answers


def _build_answer_support_records(
    *,
    answer_records: list[AnswerPosteriorRecord],
    emitted_paths: list[TrajectoryRecord],
) -> list[AnswerSupportRecord]:
    answer_rank = {
        int(record.answer_entity_id): idx
        for idx, record in enumerate(answer_records, start=1)
    }
    grouped_paths: dict[int, list[TrajectoryRecord]] = {}
    for path in emitted_paths:
        grouped_paths.setdefault(int(path.terminal_entity_id), []).append(path)
    support_records: list[AnswerSupportRecord] = []
    for record in answer_records:
        answer_id = int(record.answer_entity_id)
        support_records.append(
            AnswerSupportRecord(
                answer_entity_id=answer_id,
                answer_rank=int(answer_rank.get(answer_id, 0)),
                prob=float(record.prob),
                cumulative_mass=float(record.cumulative_mass),
                is_gold=bool(record.is_gold),
                is_selected=bool(record.is_selected),
                support_mass=float(record.support_mass),
                support_conditioned_mass=float(record.support_conditioned_mass),
                support_path_count=int(record.support_path_count),
                trajectories=list(grouped_paths.get(answer_id, [])),
                prob_ci_low=float(record.prob_ci_low),
                prob_ci_high=float(record.prob_ci_high),
            )
        )
    return support_records


def _resolve_gold_interval(
    analysis: ReachabilityAnalysis,
) -> tuple[float, float]:
    gold_low = (
        float(analysis.gold_answer_mass_ci_low)
        if analysis.gold_answer_mass_ci_low is not None
        else float(analysis.gold_answer_mass)
    )
    gold_high = (
        float(analysis.gold_answer_mass_ci_high)
        if analysis.gold_answer_mass_ci_high is not None
        else float(analysis.gold_answer_mass)
    )
    return gold_low, gold_high


def build_window_result(
    *,
    batch: TrajectoryBatch,
    analysis: ReachabilityAnalysis,
    diagnostics: SearchDiagnostics,
    discovered_paths: list[DiscoveredTrajectory],
    answer_mass_threshold: float,
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
    answer_mass_reference: str,
    support_mass_reference: str,
    answer_mass_reference_total: float | None = None,
    support_answer_upper_bounds: dict[int, float] | None = None,
    include_answer_support: bool = True,
) -> SupportWindowResult:
    gold_answers = graph_gold_answers(batch=batch)
    answer_records, selected_answer_ids = build_answer_posterior(
        ranking=ranking_from_analysis(analysis),
        gold_answers=gold_answers,
        answer_mass_threshold=answer_mass_threshold,
        total_mass_reference=answer_mass_reference_total,
    )
    emitted_paths, updated_answers = _build_support_outputs(
        batch=batch,
        discovered_paths=discovered_paths,
        answer_records=answer_records,
        selected_set=set(selected_answer_ids),
        support_mass_threshold=support_mass_threshold,
        support_path_overlap_penalty=support_path_overlap_penalty,
        support_answer_upper_bounds=support_answer_upper_bounds,
    )
    answer_support = (
        _build_answer_support_records(
            answer_records=updated_answers,
            emitted_paths=emitted_paths,
        )
        if include_answer_support
        else []
    )
    covered_mass = sum(path.prob for path in emitted_paths)
    covered_gold_answer_mass = sum(path.prob for path in emitted_paths if path.is_gold)
    gold_low, gold_high = _resolve_gold_interval(analysis)
    return SupportWindowResult(
        sample_id=batch.sample_ids[0],
        dataset_scope=batch.dataset_scope,
        mass_threshold=float(support_mass_threshold),
        window_size=len(emitted_paths),
        covered_mass=covered_mass,
        residual_mass=max(1.0 - covered_mass, 0.0),
        gold_answer_mass=float(analysis.gold_answer_mass),
        covered_gold_answer_mass=covered_gold_answer_mass,
        missed_gold_answer_mass=max(
            float(analysis.gold_answer_mass) - covered_gold_answer_mass,
            0.0,
        ),
        unique_answer_count=len({path.terminal_entity_id for path in emitted_paths}),
        unique_path_count=len(emitted_paths),
        gold_answer_entity_ids=sorted(gold_answers),
        start_entity_ids=graph_start_entity_ids(batch=batch),
        trajectories=emitted_paths,
        inference_mode=str(diagnostics.inference_mode),
        ci_confidence_level=(
            diagnostics.ci_confidence_level
            if diagnostics.ci_confidence_level is not None
            else analysis.ci_confidence_level
        ),
        covered_mass_ci_low=(
            float(diagnostics.covered_mass_ci_low)
            if diagnostics.covered_mass_ci_low is not None
            else float(covered_mass)
        ),
        covered_mass_ci_high=(
            float(diagnostics.covered_mass_ci_high)
            if diagnostics.covered_mass_ci_high is not None
            else float(covered_mass)
        ),
        gold_answer_mass_ci_low=gold_low,
        gold_answer_mass_ci_high=gold_high,
        answer_mass_threshold=float(answer_mass_threshold),
        support_mass_threshold=float(support_mass_threshold),
        probe_count=int(diagnostics.probe_count),
        emit_path_count=len(emitted_paths),
        remaining_mass_upper=max(float(diagnostics.remaining_mass_upper), 0.0),
        stop_reason=str(diagnostics.stop_reason),
        coverage_certified=bool(diagnostics.coverage_certified),
        answer_mass_reference=str(answer_mass_reference),
        support_mass_reference=str(support_mass_reference),
        selected_answer_ids=selected_answer_ids,
        answer_posterior=updated_answers,
        answer_support=answer_support,
    )


def build_rank_only_result(
    *,
    batch: TrajectoryBatch,
    analysis: ReachabilityAnalysis,
    ranking: ReachabilityRanking,
    diagnostics: SearchDiagnostics,
    answer_mass_threshold: float,
    support_mass_threshold: float,
    answer_mass_reference: str,
    answer_mass_reference_total: float | None = None,
) -> SupportWindowResult:
    gold_answers = graph_gold_answers(batch=batch)
    answer_records, selected_answer_ids = build_answer_posterior(
        ranking=ranking,
        gold_answers=gold_answers,
        answer_mass_threshold=answer_mass_threshold,
        total_mass_reference=answer_mass_reference_total,
    )
    gold_low, gold_high = _resolve_gold_interval(analysis)
    return SupportWindowResult(
        sample_id=batch.sample_ids[0],
        dataset_scope=batch.dataset_scope,
        mass_threshold=float(support_mass_threshold),
        window_size=0,
        covered_mass=0.0,
        residual_mass=1.0,
        gold_answer_mass=float(analysis.gold_answer_mass),
        covered_gold_answer_mass=0.0,
        missed_gold_answer_mass=float(analysis.gold_answer_mass),
        unique_answer_count=0,
        unique_path_count=0,
        gold_answer_entity_ids=sorted(gold_answers),
        start_entity_ids=graph_start_entity_ids(batch=batch),
        trajectories=[],
        inference_mode=str(diagnostics.inference_mode),
        ci_confidence_level=(
            diagnostics.ci_confidence_level
            if diagnostics.ci_confidence_level is not None
            else analysis.ci_confidence_level
        ),
        covered_mass_ci_low=0.0,
        covered_mass_ci_high=0.0,
        gold_answer_mass_ci_low=gold_low,
        gold_answer_mass_ci_high=gold_high,
        answer_mass_threshold=float(answer_mass_threshold),
        support_mass_threshold=float(support_mass_threshold),
        probe_count=int(diagnostics.probe_count),
        emit_path_count=0,
        remaining_mass_upper=max(float(diagnostics.remaining_mass_upper), 0.0),
        stop_reason=str(diagnostics.stop_reason),
        coverage_certified=bool(diagnostics.coverage_certified),
        answer_mass_reference=str(answer_mass_reference),
        support_mass_reference="skipped",
        selected_answer_ids=selected_answer_ids,
        answer_posterior=answer_records,
        answer_support=[],
    )


class ReachabilityBackendProtocol(Protocol):
    inference_mode: str

    def evaluate_graph(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        metrics_profile: str,
        include_answer_support: bool,
    ) -> SupportWindowResult: ...

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
        metrics_profile: str,
        include_answer_support: bool,
    ) -> list[SupportWindowResult]: ...


INVALID_START_REASON = "invalid_start_candidates"


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


class AnswerReachabilityEvaluator:
    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
        backend: ReachabilityBackendProtocol,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.backend = backend

    @staticmethod
    def empty_reachability_analysis(batch: TrajectoryBatch) -> ReachabilityAnalysis:
        device = batch.node_ptr.device
        return ReachabilityAnalysis(
            terminal_mass=torch.zeros(
                (batch.num_nodes_total,), device=device, dtype=torch.float32
            ),
            answer_entity_ids=torch.empty((0,), device=device, dtype=torch.long),
            answer_probs=torch.empty((0,), device=device, dtype=torch.float32),
            gold_answer_mass=0.0,
        )

    def build_invalid_start_result(self, batch: TrajectoryBatch) -> SupportWindowResult:
        return build_window_result(
            batch=batch,
            analysis=self.empty_reachability_analysis(batch),
            diagnostics=SearchDiagnostics(
                inference_mode=self.backend.inference_mode,
                probe_count=0,
                remaining_mass_upper=1.0,
                stop_reason=INVALID_START_REASON,
                coverage_certified=False,
            ),
            discovered_paths=[],
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            answer_mass_reference="none",
            support_mass_reference="none",
            answer_mass_reference_total=0.0,
            include_answer_support=False,
        )

    def _prepare_batch(self, batch: TrajectoryBatch) -> PreparedSearchBatch:
        return self.policy.prepare_batch(batch)

    def _evaluate_individual_graphs(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None,
    ) -> list[SupportWindowResult]:
        results: list[SupportWindowResult] = []
        for graph_idx in range(batch.num_graphs):
            graph_batch = batch.select_graph(graph_idx, validate=False)
            prepared_graph = self._prepare_batch(graph_batch)
            try:
                results.append(
                    self.backend.evaluate_graph(
                        batch=graph_batch,
                        policy=self.policy,
                        prepared_batch=prepared_graph,
                        metrics_profile=metrics_profile,
                        include_answer_support=include_answer_support,
                    )
                )
            except StartDistributionError:
                if on_invalid_start is not None:
                    on_invalid_start(graph_batch)
                results.append(self.build_invalid_start_result(graph_batch))
        return results

    def _build_window_results(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None,
    ) -> list[SupportWindowResult]:
        prepared_batch = self._prepare_batch(batch)
        try:
            return self.backend.evaluate_batch(
                batch=batch,
                policy=self.policy,
                prepared_batch=prepared_batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
            )
        except StartDistributionError:
            return self._evaluate_individual_graphs(
                batch=batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
                on_invalid_start=on_invalid_start,
            )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        with torch.no_grad():
            window_results = self._build_window_results(
                batch=batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
                on_invalid_start=on_invalid_start,
            )
        metrics, diagnostics = summarize_reachability_metrics(
            results=window_results,
            answer_top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
            metrics_profile=metrics_profile,
            invalid_start_reason=INVALID_START_REASON,
        )
        return MetricEvaluationOutput(
            model_metrics={},
            primary_metrics=metrics,
            secondary_metrics=diagnostics,
            results=window_results,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[SupportWindowResult]:
        with torch.no_grad():
            return self._build_window_results(
                batch=batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
                on_invalid_start=on_invalid_start,
            )

    @staticmethod
    def build_predict_labels(
        batch: TrajectoryBatch,
        outputs: list[SupportWindowResult],
    ) -> list[SupportWindowLabelRecord]:
        if len(outputs) != batch.num_graphs:
            raise ValueError(
                "Predict outputs must align with TrajectoryBatch graph count. "
                f"outputs={len(outputs)} num_graphs={batch.num_graphs}."
            )
        labels: list[SupportWindowLabelRecord] = []
        a_counts = batch.a_ptr[1:] - batch.a_ptr[:-1]
        for graph_idx, result in enumerate(outputs):
            answer_start = int(batch.answer_ptr[graph_idx].item())
            answer_end = int(batch.answer_ptr[graph_idx + 1].item())
            labels.append(
                SupportWindowLabelRecord(
                    sample_id=result.sample_id,
                    question=batch.questions[graph_idx],
                    start_entity_ids=list(result.start_entity_ids),
                    answer_entity_ids=[
                        int(value)
                        for value in batch.answer_entity_ids[
                            answer_start:answer_end
                        ].tolist()
                    ],
                    a_entity_in_graph=bool(int(a_counts[graph_idx].item()) > 0),
                )
            )
        return labels


# ---------------------------------------------------------------------------
# JSONL serialization
# ---------------------------------------------------------------------------


def load_edge_record(record: dict[str, Any]) -> EdgeRecord:
    return EdgeRecord(
        edge_id=int(record.get("edge_id", 0)),
        src_entity_id=int(record.get("src_entity_id", 0)),
        relation_id=int(record.get("relation_id", 0)),
        dst_entity_id=int(record.get("dst_entity_id", 0)),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_value(record: dict[str, Any], key: str, default: int = 0) -> int:
    value = record.get(key, default)
    return int(default if value is None else value)


def _float_value(record: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = record.get(key, default)
    return float(default if value is None else value)


def load_trajectory_record(record: dict[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord(
        sample_id=str(record.get("sample_id", "")),
        path_rank=_int_value(record, "path_rank"),
        log_prob=_float_value(record, "log_prob"),
        prob=_float_value(record, "prob"),
        cumulative_mass=_float_value(record, "cumulative_mass"),
        terminal_entity_id=_int_value(record, "terminal_entity_id"),
        is_gold=bool(record.get("is_gold", False)),
        edges=[load_edge_record(edge) for edge in record.get("edges") or []],
        start_entity_id=_optional_int(record.get("start_entity_id")),
        answer_rank=_int_value(record, "answer_rank"),
        support_rank=_int_value(record, "support_rank"),
        conditional_prob=_float_value(record, "conditional_prob"),
        conditional_cumulative_mass=_float_value(record, "conditional_cumulative_mass"),
    )


def load_answer_posterior_record(record: dict[str, Any]) -> AnswerPosteriorRecord:
    return AnswerPosteriorRecord(
        answer_entity_id=_int_value(record, "answer_entity_id"),
        prob=_float_value(record, "prob"),
        cumulative_mass=_float_value(record, "cumulative_mass"),
        is_gold=bool(record.get("is_gold", False)),
        is_selected=bool(record.get("is_selected", False)),
        support_mass=_float_value(record, "support_mass"),
        support_conditioned_mass=_float_value(record, "support_conditioned_mass"),
        support_path_count=_int_value(record, "support_path_count"),
        prob_ci_low=_float_value(record, "prob_ci_low"),
        prob_ci_high=_float_value(record, "prob_ci_high"),
    )


def load_answer_support_record(record: dict[str, Any]) -> AnswerSupportRecord:
    return AnswerSupportRecord(
        answer_entity_id=_int_value(record, "answer_entity_id"),
        answer_rank=_int_value(record, "answer_rank"),
        prob=_float_value(record, "prob"),
        cumulative_mass=_float_value(record, "cumulative_mass"),
        is_gold=bool(record.get("is_gold", False)),
        is_selected=bool(record.get("is_selected", False)),
        support_mass=_float_value(record, "support_mass"),
        support_conditioned_mass=_float_value(record, "support_conditioned_mass"),
        support_path_count=_int_value(record, "support_path_count"),
        trajectories=[
            load_trajectory_record(entry) for entry in record.get("trajectories") or []
        ],
        prob_ci_low=_float_value(record, "prob_ci_low"),
        prob_ci_high=_float_value(record, "prob_ci_high"),
    )


def load_support_window_result(record: dict[str, Any]) -> SupportWindowResult:
    return SupportWindowResult(
        sample_id=str(record.get("sample_id", "")),
        dataset_scope=str(record.get("dataset_scope", "")),
        mass_threshold=_float_value(record, "mass_threshold"),
        window_size=_int_value(record, "window_size"),
        covered_mass=_float_value(record, "covered_mass"),
        residual_mass=_float_value(record, "residual_mass"),
        gold_answer_mass=_float_value(record, "gold_answer_mass"),
        covered_gold_answer_mass=_float_value(record, "covered_gold_answer_mass"),
        missed_gold_answer_mass=_float_value(record, "missed_gold_answer_mass"),
        unique_answer_count=_int_value(record, "unique_answer_count"),
        unique_path_count=_int_value(record, "unique_path_count"),
        gold_answer_entity_ids=[
            int(v) for v in record.get("gold_answer_entity_ids") or []
        ],
        start_entity_ids=[int(v) for v in record.get("start_entity_ids") or []],
        trajectories=[
            load_trajectory_record(entry) for entry in record.get("trajectories") or []
        ],
        answer_posterior=[
            load_answer_posterior_record(entry)
            for entry in record.get("answer_posterior") or []
        ],
        answer_support=[
            load_answer_support_record(entry)
            for entry in record.get("answer_support") or []
        ],
        covered_mass_ci_low=_optional_float(record.get("covered_mass_ci_low")),
        covered_mass_ci_high=_optional_float(record.get("covered_mass_ci_high")),
        gold_answer_mass_ci_low=_optional_float(record.get("gold_answer_mass_ci_low")),
        gold_answer_mass_ci_high=_optional_float(
            record.get("gold_answer_mass_ci_high")
        ),
        ci_confidence_level=_optional_float(record.get("ci_confidence_level")),
        answer_mass_threshold=_float_value(
            record, "answer_mass_threshold", _float_value(record, "mass_threshold")
        ),
        support_mass_threshold=_float_value(
            record, "support_mass_threshold", _float_value(record, "mass_threshold")
        ),
        probe_count=_int_value(record, "probe_count"),
        emit_path_count=_int_value(record, "emit_path_count"),
        remaining_mass_upper=_float_value(record, "remaining_mass_upper"),
        stop_reason=str(record.get("stop_reason", "")),
        coverage_certified=bool(record.get("coverage_certified", False)),
        inference_mode=str(record.get("inference_mode", UNSPECIFIED_INFERENCE_MODE)),
        answer_mass_reference=str(
            record.get("answer_mass_reference", UNSPECIFIED_MASS_REFERENCE)
        ),
        support_mass_reference=str(
            record.get("support_mass_reference", UNSPECIFIED_MASS_REFERENCE)
        ),
        selected_answer_ids=[int(v) for v in record.get("selected_answer_ids") or []],
    )


def load_support_window_label(record: dict[str, Any]) -> SupportWindowLabelRecord:
    return SupportWindowLabelRecord(
        sample_id=str(record.get("sample_id", "")),
        question=str(record.get("question", "")),
        start_entity_ids=[int(v) for v in record.get("start_entity_ids") or []],
        answer_entity_ids=[int(v) for v in record.get("answer_entity_ids") or []],
        a_entity_in_graph=bool(record.get("a_entity_in_graph", False)),
    )


class SupportWindowPredictionCodec(PredictionCodecProtocol):
    kind = "support_window"

    def serialize_result(self, result: SupportWindowResult) -> dict[str, Any]:
        return asdict(result)

    def deserialize_result(self, record: dict[str, Any]) -> SupportWindowResult:
        return load_support_window_result(record)

    def serialize_label(self, label: SupportWindowLabelRecord) -> dict[str, Any]:
        return asdict(label)

    def deserialize_label(self, record: dict[str, Any]) -> SupportWindowLabelRecord:
        return load_support_window_label(record)


# ---------------------------------------------------------------------------
# Artifact export
# ---------------------------------------------------------------------------


def _edge_to_dict(edge: EdgeRecord) -> dict[str, Any]:
    return {
        "edge_id": int(edge.edge_id),
        "src_entity_id": int(edge.src_entity_id),
        "relation_id": int(edge.relation_id),
        "dst_entity_id": int(edge.dst_entity_id),
    }


def _trajectory_text(trajectory: TrajectoryRecord) -> str:
    if not trajectory.edges:
        return str(trajectory.terminal_entity_id)
    return " ; ".join(
        f"{edge.src_entity_id} --{edge.relation_id}--> {edge.dst_entity_id}"
        for edge in trajectory.edges
    )


def _trajectory_to_prompt_record(trajectory: TrajectoryRecord) -> dict[str, Any]:
    return {
        "path_rank": int(trajectory.path_rank),
        "log_prob": float(trajectory.log_prob),
        "prob": float(trajectory.prob),
        "cumulative_mass": float(trajectory.cumulative_mass),
        "terminal_entity_id": int(trajectory.terminal_entity_id),
        "start_entity_id": _optional_int(trajectory.start_entity_id),
        "answer_rank": int(trajectory.answer_rank),
        "support_rank": int(trajectory.support_rank),
        "conditional_prob": float(trajectory.conditional_prob),
        "conditional_cumulative_mass": float(trajectory.conditional_cumulative_mass),
        "edges": [_edge_to_dict(edge) for edge in trajectory.edges],
        "trajectory_text": _trajectory_text(trajectory),
    }


def _answer_to_dict(answer: AnswerPosteriorRecord) -> dict[str, Any]:
    return {
        "answer_entity_id": int(answer.answer_entity_id),
        "prob": float(answer.prob),
        "prob_ci_low": float(answer.prob_ci_low),
        "prob_ci_high": float(answer.prob_ci_high),
        "cumulative_mass": float(answer.cumulative_mass),
        "is_gold": bool(answer.is_gold),
        "is_selected": bool(answer.is_selected),
        "support_mass": float(answer.support_mass),
        "support_conditioned_mass": float(answer.support_conditioned_mass),
        "support_path_count": int(answer.support_path_count),
    }


def _answer_support_to_dict(answer: AnswerSupportRecord) -> dict[str, Any]:
    return {
        "answer_entity_id": int(answer.answer_entity_id),
        "answer_rank": int(answer.answer_rank),
        "prob": float(answer.prob),
        "prob_ci_low": float(answer.prob_ci_low),
        "prob_ci_high": float(answer.prob_ci_high),
        "cumulative_mass": float(answer.cumulative_mass),
        "is_gold": bool(answer.is_gold),
        "is_selected": bool(answer.is_selected),
        "support_mass": float(answer.support_mass),
        "support_conditioned_mass": float(answer.support_conditioned_mass),
        "support_path_count": int(answer.support_path_count),
        "trajectories": [
            _trajectory_to_prompt_record(trajectory)
            for trajectory in answer.trajectories
        ],
    }


class SupportWindowArtifactWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str,
        artifact_name: str = "rankflow",
        schema_version: int = 1,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        del (
            artifact_name,
            schema_version,
            entity_vocab_path,
            relation_vocab_path,
            questions_path,
        )
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.overwrite = bool(overwrite)

    @property
    def prompt_path(self) -> Path:
        return self.output_dir / f"{self.split}.jsonl"

    @property
    def labels_path(self) -> Path:
        return self.output_dir / f"{self.split}.labels.jsonl"

    def _ensure_output_paths(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.overwrite:
            return
        for path in (self.prompt_path, self.labels_path):
            if path.exists():
                raise FileExistsError(f"Artifact already exists: {path}")

    def _build_prompt_record(
        self,
        result: SupportWindowResult,
        label: SupportWindowLabelRecord | None,
    ) -> dict[str, Any]:
        question = "" if label is None else str(label.question)
        return {
            "sample_id": result.sample_id,
            "question": question,
            "dataset_scope": result.dataset_scope,
            "mass_threshold": float(result.mass_threshold),
            "window_size": int(result.window_size),
            "covered_mass": float(result.covered_mass),
            "residual_mass": float(result.residual_mass),
            "gold_answer_mass": float(result.gold_answer_mass),
            "covered_gold_answer_mass": float(result.covered_gold_answer_mass),
            "missed_gold_answer_mass": float(result.missed_gold_answer_mass),
            "unique_answer_count": int(result.unique_answer_count),
            "unique_path_count": int(result.unique_path_count),
            "gold_answer_entity_ids": list(result.gold_answer_entity_ids),
            "start_entity_ids": list(result.start_entity_ids),
            "trajectories": [
                _trajectory_to_prompt_record(trajectory)
                for trajectory in result.trajectories
            ],
            "inference_mode": str(result.inference_mode or UNSPECIFIED_INFERENCE_MODE),
            "answer_mass_threshold": float(result.answer_mass_threshold),
            "support_mass_threshold": float(result.support_mass_threshold),
            "probe_count": int(result.probe_count),
            "emit_path_count": int(result.emit_path_count),
            "remaining_mass_upper": float(result.remaining_mass_upper),
            "covered_mass_ci_low": _optional_float(result.covered_mass_ci_low),
            "covered_mass_ci_high": _optional_float(result.covered_mass_ci_high),
            "gold_answer_mass_ci_low": _optional_float(result.gold_answer_mass_ci_low),
            "gold_answer_mass_ci_high": _optional_float(
                result.gold_answer_mass_ci_high
            ),
            "ci_confidence_level": _optional_float(result.ci_confidence_level),
            "coverage_certified": bool(result.coverage_certified),
            "answer_mass_reference": str(
                result.answer_mass_reference or UNSPECIFIED_MASS_REFERENCE
            ),
            "support_mass_reference": str(
                result.support_mass_reference or UNSPECIFIED_MASS_REFERENCE
            ),
            "selected_answer_ids": list(result.selected_answer_ids),
            "answer_posterior": [
                _answer_to_dict(answer) for answer in result.answer_posterior
            ],
            "answer_support": [
                _answer_support_to_dict(answer) for answer in result.answer_support
            ],
        }

    @staticmethod
    def _build_label_record(
        label: SupportWindowLabelRecord,
    ) -> dict[str, Any]:
        return {
            "sample_id": label.sample_id,
            "question": label.question,
            "start_entity_ids": list(label.start_entity_ids),
            "answer_entity_ids": list(label.answer_entity_ids),
            "answer_texts": [],
            "a_entity_in_graph": bool(label.a_entity_in_graph),
        }

    @staticmethod
    def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    def write(
        self,
        *,
        results: Sequence[SupportWindowResult],
        labels: Sequence[SupportWindowLabelRecord],
    ) -> dict[str, Path]:
        self._ensure_output_paths()
        label_by_id = {label.sample_id: label for label in labels}
        prompt_records = [
            self._build_prompt_record(result, label_by_id.get(result.sample_id))
            for result in results
        ]
        label_records = [self._build_label_record(label) for label in labels]
        self._write_jsonl(self.prompt_path, prompt_records)
        self._write_jsonl(self.labels_path, label_records)
        return {"prompt_path": self.prompt_path, "labels_path": self.labels_path}

    def write_from_jsonl(
        self,
        *,
        results_path: str | Path,
        labels_path: str | Path,
    ) -> dict[str, Path]:
        self._ensure_output_paths()
        labels = [
            load_support_window_label(record)
            for record in iter_jsonl_records(labels_path)
        ]
        label_by_id = {label.sample_id: label for label in labels}
        prompt_records = (
            self._build_prompt_record(
                result,
                label_by_id.get(result.sample_id),
            )
            for result in (
                load_support_window_result(record)
                for record in iter_jsonl_records(results_path)
            )
        )
        label_records = (self._build_label_record(label) for label in labels)
        self._write_jsonl(self.prompt_path, prompt_records)
        self._write_jsonl(self.labels_path, label_records)
        return {"prompt_path": self.prompt_path, "labels_path": self.labels_path}


# ---------------------------------------------------------------------------
# Predict-time metric aggregation
# ---------------------------------------------------------------------------


@dataclass
class SupportWindowMetricsAccumulator:
    metrics: dict[str, MeanMetric] = field(default_factory=dict)
    invalid_start_count: SumMetric = field(default_factory=SumMetric)
    sample_count: SumMetric = field(default_factory=SumMetric)


def initialize_predict_metrics_accumulator() -> SupportWindowMetricsAccumulator:
    return SupportWindowMetricsAccumulator()


def update_predict_metrics_accumulator(
    *,
    accumulator: SupportWindowMetricsAccumulator,
    eval_cfg: SearchEvalConfig,
    predict_results: list[SupportWindowResult],
    metrics_profile: str,
) -> None:
    if not predict_results:
        return
    metrics, diagnostics = summarize_reachability_metrics(
        results=predict_results,
        answer_top_ks=tuple(int(k) for k in eval_cfg.answer_top_ks),
        metrics_profile=metrics_profile,
        invalid_start_reason=INVALID_START_REASON,
    )
    batch_weight = torch.tensor(float(len(predict_results)), dtype=torch.float32)
    for metric_values in (metrics, diagnostics):
        for name, value in metric_values.items():
            if name in {"invalid_start_count", "invalid_start_rate"}:
                continue
            metric = accumulator.metrics.get(name)
            if metric is None:
                metric = MeanMetric()
                accumulator.metrics[name] = metric
            metric.update(
                torch.tensor(float(value), dtype=torch.float32),
                weight=batch_weight,
            )
    accumulator.invalid_start_count.update(
        torch.tensor(
            float(diagnostics.get("invalid_start_count", 0.0)),
            dtype=torch.float32,
        )
    )
    accumulator.sample_count.update(batch_weight)


def finalize_predict_metrics_accumulator(
    *,
    accumulator: SupportWindowMetricsAccumulator,
) -> dict[str, float]:
    count = float(accumulator.sample_count.compute().item())
    if count < 1.0:
        return {}
    metrics = {
        name: float(metric.compute().item())
        for name, metric in accumulator.metrics.items()
    }
    invalid_start_count = float(accumulator.invalid_start_count.compute().item())
    metrics["invalid_start_count"] = invalid_start_count
    metrics["invalid_start_rate"] = invalid_start_count / count
    return metrics


def summarize_predict_results(
    *,
    eval_cfg: SearchEvalConfig,
    predict_results: list[SupportWindowResult],
    metrics_profile: str,
) -> dict[str, float]:
    if not predict_results:
        return {}
    metrics, diagnostics = summarize_reachability_metrics(
        results=predict_results,
        answer_top_ks=tuple(int(k) for k in eval_cfg.answer_top_ks),
        metrics_profile=metrics_profile,
        invalid_start_reason=INVALID_START_REASON,
    )
    return {**metrics, **diagnostics}


def summarize_predict_results_from_jsonl(
    *,
    predict_results_path: str | Path,
    eval_cfg: SearchEvalConfig,
    metrics_profile: str,
) -> dict[str, float]:
    if not jsonl_has_records(predict_results_path):
        return {}
    accumulator = initialize_predict_metrics_accumulator()
    batch_results: list[SupportWindowResult] = []
    for record in iter_jsonl_records(predict_results_path):
        batch_results.append(load_support_window_result(record))
        if len(batch_results) >= 256:
            update_predict_metrics_accumulator(
                accumulator=accumulator,
                eval_cfg=eval_cfg,
                predict_results=batch_results,
                metrics_profile=metrics_profile,
            )
            batch_results.clear()
    if batch_results:
        update_predict_metrics_accumulator(
            accumulator=accumulator,
            eval_cfg=eval_cfg,
            predict_results=batch_results,
            metrics_profile=metrics_profile,
        )
    return finalize_predict_metrics_accumulator(accumulator=accumulator)


# ---------------------------------------------------------------------------
# Runtime adapter
# ---------------------------------------------------------------------------


class AnswerReachabilityRuntime(BaseMetricRuntime):
    sampler: TrajectorySamplerProtocol | None

    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        evaluator: AnswerReachabilityEvaluator,
        sampler: TrajectorySamplerProtocol,
        search_backend: Any,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.evaluator = evaluator
        self._prediction_codec = SupportWindowPredictionCodec()
        self.sampler = sampler
        self.search = search_backend

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        return self.evaluator.evaluate_batch(
            batch=batch,
            metrics_profile=metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=on_invalid_start,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]:
        return self.evaluator.predict_batch(
            batch=batch,
            metrics_profile=metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=on_invalid_start,
        )

    def build_predict_labels(
        self, batch: TrajectoryBatch, outputs: list[Any]
    ) -> list[Any]:
        return self.evaluator.build_predict_labels(batch, outputs)

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> dict[str, float]:
        return summarize_predict_results(
            eval_cfg=self.eval_cfg,
            predict_results=predict_results,
            metrics_profile=metrics_profile,
        )

    def write_prediction_artifacts(
        self,
        *,
        results: list[Any],
        labels: list[Any],
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        if not results:
            return None
        writer = SupportWindowArtifactWriter(
            output_dir=output_dir,
            split=split,
            artifact_name=artifact_name,
            schema_version=schema_version,
            entity_vocab_path=entity_vocab_path,
            relation_vocab_path=relation_vocab_path,
            questions_path=questions_path,
            overwrite=overwrite,
        )
        return writer.write(results=results, labels=labels)

    def initialize_predict_metrics_accumulator(
        self,
        *,
        metrics_profile: str,
    ) -> SupportWindowMetricsAccumulator:
        del metrics_profile
        return initialize_predict_metrics_accumulator()

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: SupportWindowMetricsAccumulator,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> None:
        update_predict_metrics_accumulator(
            accumulator=accumulator,
            eval_cfg=self.eval_cfg,
            predict_results=predict_results,
            metrics_profile=metrics_profile,
        )

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: SupportWindowMetricsAccumulator,
        metrics_profile: str,
    ) -> dict[str, float]:
        del metrics_profile
        return finalize_predict_metrics_accumulator(accumulator=accumulator)

    def summarize_predict_epoch_from_jsonl(
        self,
        *,
        predict_results_path: str | Path,
        metrics_profile: str,
    ) -> dict[str, float]:
        return summarize_predict_results_from_jsonl(
            predict_results_path=predict_results_path,
            eval_cfg=self.eval_cfg,
            metrics_profile=metrics_profile,
        )
