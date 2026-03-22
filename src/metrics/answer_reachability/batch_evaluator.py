from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast, Protocol

import torch

from .analysis import ReachabilityAnalysis
from src.models.configs import SearchEvalConfig
from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import StartDistributionError
from src.models.gflownet import PreparedSearchBatch, SearchPolicyProtocol

from .metrics import compute_support_metrics
from .posterior import (
    aggregate_rank_metrics,
    build_rank_only_result,
    build_window_result,
)
from .schema import (
    SupportWindowEvalBatch,
    SupportWindowLabelRecord,
    SupportWindowResult,
)
from .support_search import SupportSearchProtocol

INVALID_START_REASON = "invalid_start_candidates"


class ReachabilityAnalyzerProtocol(Protocol):
    def analyze(
        self,
        *,
        batch: TrajectoryBatch,
        policy: SearchPolicyProtocol,
        prepared_batch: PreparedSearchBatch,
    ) -> ReachabilityAnalysis: ...


@dataclass(frozen=True)
class PreparedSingleGraphEvaluation:
    batch: TrajectoryBatch
    prepared_batch: PreparedSearchBatch
    analysis: ReachabilityAnalysis | None
    invalid_start: bool = False


@dataclass(frozen=True)
class ReachabilityBatchOutput:
    support_metrics: dict[str, float]
    window_results: list[SupportWindowResult]
    model_metrics: dict[str, float]
    rank_metrics: dict[str, float]


class ReachabilityBatchEvaluator:
    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
        analyzer: ReachabilityAnalyzerProtocol,
        support_search: SupportSearchProtocol,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.analyzer = analyzer
        self.support_search = support_search

    def _inference_mode(self) -> str:
        return str(self.eval_cfg.support_search_method)

    @staticmethod
    def empty_reachability_analysis(
        batch: TrajectoryBatch,
    ) -> ReachabilityAnalysis:
        device = batch.node_ptr.device
        return ReachabilityAnalysis(
            terminal_mass=torch.zeros(
                (batch.num_nodes_total,), device=device, dtype=torch.float32
            ),
            answer_entity_ids=torch.empty((0,), device=device, dtype=torch.long),
            answer_probs=torch.empty((0,), device=device, dtype=torch.float32),
            gold_total_mass=0.0,
        )

    def build_invalid_start_result(
        self,
        batch: TrajectoryBatch,
    ) -> SupportWindowResult:
        return build_window_result(
            batch=batch,
            discovered_paths=[],
            analysis=self.empty_reachability_analysis(batch),
            inference_mode=self._inference_mode(),
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            support_path_overlap_penalty=float(
                self.eval_cfg.support_path_overlap_penalty
            ),
            probe_count=0,
            remaining_mass_upper=1.0,
            stop_reason=INVALID_START_REASON,
            coverage_certified=False,
            answer_mass_reference="none",
            support_mass_reference="none",
            answer_mass_reference_total=0.0,
        )

    def _prepare_graph_with_fallback(
        self,
        batch: TrajectoryBatch,
        *,
        require_analysis: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> PreparedSingleGraphEvaluation:
        prepared_batch = self.policy.prepare_batch(batch)
        try:
            analysis = None
            if require_analysis:
                analysis = self.analyzer.analyze(
                    batch=batch,
                    policy=self.policy,
                    prepared_batch=prepared_batch,
                )
            else:
                self.policy.compute_root_action_distribution(prepared_batch)
        except StartDistributionError:
            if on_invalid_start is not None:
                on_invalid_start(batch)
            return PreparedSingleGraphEvaluation(
                batch=batch,
                prepared_batch=prepared_batch,
                analysis=self.empty_reachability_analysis(batch),
                invalid_start=True,
            )
        return PreparedSingleGraphEvaluation(
            batch=batch,
            prepared_batch=prepared_batch,
            analysis=analysis,
        )

    def prepare_evaluation_graphs(
        self,
        batch: TrajectoryBatch,
        *,
        metrics_profile: str,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[PreparedSingleGraphEvaluation]:
        prepared_graphs: list[PreparedSingleGraphEvaluation] = []
        require_analysis = bool(metrics_profile == "rank_only") or bool(
            self.support_search.requires_analysis
        )
        for graph_idx in range(batch.num_graphs):
            prepared_graph = self._prepare_graph_with_fallback(
                batch.select_graph(graph_idx, validate=False),
                require_analysis=require_analysis,
                on_invalid_start=on_invalid_start,
            )
            prepared_graphs.append(prepared_graph)
        return prepared_graphs

    def _build_rank_only_result(
        self,
        *,
        batch: TrajectoryBatch,
        analysis: ReachabilityAnalysis,
    ) -> SupportWindowResult:
        return build_rank_only_result(
            batch=batch,
            analysis=analysis,
            inference_mode=(
                str(analysis.inference_mode)
                if analysis.inference_mode is not None
                else self._inference_mode()
            ),
            answer_mass_threshold=float(self.eval_cfg.answer_mass_threshold),
            support_mass_threshold=float(self.eval_cfg.support_mass_threshold),
            probe_count=(
                int(analysis.probe_count)
                if analysis.probe_count is not None
                else int(self.eval_cfg.monte_carlo_rollouts)
                if self.eval_cfg.support_search_method == "monte_carlo"
                else 0
            ),
            remaining_mass_upper=(
                float(analysis.remaining_mass_upper)
                if analysis.remaining_mass_upper is not None
                else 0.0
            ),
            stop_reason=(
                str(analysis.stop_reason)
                if analysis.stop_reason is not None
                else "rank_only_monte_carlo"
                if self.eval_cfg.support_search_method == "monte_carlo"
                else "rank_only_flow_frontier"
            ),
            coverage_certified=(
                bool(analysis.coverage_certified)
                if analysis.coverage_certified is not None
                else self.eval_cfg.support_search_method != "monte_carlo"
            ),
            answer_mass_reference=(
                str(analysis.inference_mode)
                if analysis.inference_mode is not None
                else self._inference_mode()
            ),
            answer_mass_reference_total=1.0,
            ci_confidence_level=(
                float(analysis.ci_confidence_level)
                if analysis.ci_confidence_level is not None
                else float(self.eval_cfg.monte_carlo_confidence)
                if self.eval_cfg.support_search_method == "monte_carlo"
                else None
            ),
            gold_total_mass_ci_low=analysis.gold_total_mass_ci_low,
            gold_total_mass_ci_high=analysis.gold_total_mass_ci_high,
        )

    def _build_rank_only_window_results_batched(
        self,
        *,
        batch: TrajectoryBatch,
    ) -> list[SupportWindowResult] | None:
        analyze_batch = getattr(self.analyzer, "analyze_batch", None)
        if not callable(analyze_batch):
            return None
        prepared_batch = self.policy.prepare_batch(batch)
        try:
            analyses = cast(
                list[ReachabilityAnalysis],
                analyze_batch(
                    batch=batch,
                    policy=self.policy,
                    prepared_batch=prepared_batch,
                ),
            )
        except StartDistributionError:
            return None
        if len(analyses) != batch.num_graphs:
            raise RuntimeError(
                "Batched rank-only analyzer must return one analysis per graph. "
                f"analyses={len(analyses)} num_graphs={batch.num_graphs}."
            )
        return [
            self._build_rank_only_result(
                batch=batch.select_graph(graph_idx, validate=False),
                analysis=analysis,
            )
            for graph_idx, analysis in enumerate(analyses)
        ]

    def _build_support_window_results_batched(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> list[SupportWindowResult] | None:
        generate_windows_batch = getattr(
            self.support_search, "generate_windows_batch", None
        )
        if not callable(generate_windows_batch):
            return None
        prepared_batch = self.policy.prepare_batch(batch)
        try:
            window_results = cast(
                list[SupportWindowResult],
                generate_windows_batch(
                    batch=batch,
                    policy=self.policy,
                    prepared_batch=prepared_batch,
                    analysis=None,
                    include_answer_support=include_answer_support,
                ),
            )
        except StartDistributionError:
            return None
        if len(window_results) != batch.num_graphs:
            raise RuntimeError(
                "Batched support search must return one result per graph. "
                f"results={len(window_results)} num_graphs={batch.num_graphs}."
            )
        return window_results

    def _build_window_results(
        self,
        *,
        prepared_graphs: list[PreparedSingleGraphEvaluation],
        metrics_profile: str,
        include_answer_support: bool,
    ) -> list[SupportWindowResult]:
        window_results: list[SupportWindowResult] = []
        for graph in prepared_graphs:
            if graph.invalid_start:
                window_results.append(self.build_invalid_start_result(graph.batch))
                continue
            if metrics_profile == "rank_only":
                if graph.analysis is None:
                    raise RuntimeError(
                        "Rank-only evaluation requires a reachability analysis."
                    )
                window_results.append(
                    self._build_rank_only_result(
                        batch=graph.batch,
                        analysis=graph.analysis,
                    )
                )
                continue
            window_results.append(
                self.support_search.generate_window(
                    batch=graph.batch,
                    policy=self.policy,
                    prepared_batch=graph.prepared_batch,
                    analysis=graph.analysis,
                    include_answer_support=include_answer_support,
                )
            )
        return window_results

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> ReachabilityBatchOutput:
        with torch.no_grad():
            if metrics_profile == "rank_only":
                window_results = self._build_rank_only_window_results_batched(
                    batch=batch
                )
            else:
                window_results = self._build_support_window_results_batched(
                    batch=batch,
                    include_answer_support=include_answer_support,
                )
            if window_results is None:
                prepared_graphs = self.prepare_evaluation_graphs(
                    batch,
                    metrics_profile=metrics_profile,
                    on_invalid_start=on_invalid_start,
                )
                window_results = self._build_window_results(
                    prepared_graphs=prepared_graphs,
                    metrics_profile=metrics_profile,
                    include_answer_support=include_answer_support,
                )
        support_metrics = (
            {}
            if metrics_profile == "rank_only"
            else compute_support_metrics(
                SupportWindowEvalBatch(
                    dataset_scope=batch.dataset_scope,
                    mass_threshold=float(self.eval_cfg.support_mass_threshold),
                    results=window_results,
                    window_top_ks=tuple(int(k) for k in self.eval_cfg.window_top_ks),
                )
            )
        )
        rank_metrics = aggregate_rank_metrics(
            results=window_results,
            answer_top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
        )
        return ReachabilityBatchOutput(
            support_metrics=support_metrics,
            window_results=window_results,
            model_metrics={},
            rank_metrics=rank_metrics,
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
            if metrics_profile == "rank_only":
                window_results = self._build_rank_only_window_results_batched(
                    batch=batch
                )
            else:
                window_results = self._build_support_window_results_batched(
                    batch=batch,
                    include_answer_support=include_answer_support,
                )
            if window_results is not None:
                return window_results
            prepared_graphs = self.prepare_evaluation_graphs(
                batch,
                metrics_profile=metrics_profile,
                on_invalid_start=on_invalid_start,
            )
            return self._build_window_results(
                prepared_graphs=prepared_graphs,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
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


__all__ = [
    "INVALID_START_REASON",
    "PreparedSingleGraphEvaluation",
    "ReachabilityBatchEvaluator",
    "ReachabilityBatchOutput",
]
