from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .exact_analysis import (
    ExactReachabilityAnalysis,
    ExactReachabilityAnalyzer,
)
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
from .support_search import ExactSupportSearch

INVALID_START_REASON = "invalid_start_candidates"


@dataclass(frozen=True)
class PreparedSingleGraphEvaluation:
    batch: TrajectoryBatch
    prepared_batch: PreparedSearchBatch
    analysis: ExactReachabilityAnalysis
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
        analyzer: ExactReachabilityAnalyzer,
        support_search: ExactSupportSearch,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.policy = policy
        self.analyzer = analyzer
        self.support_search = support_search

    @staticmethod
    def empty_reachability_analysis(
        batch: TrajectoryBatch,
    ) -> ExactReachabilityAnalysis:
        device = batch.node_ptr.device
        return ExactReachabilityAnalysis(
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
            inference_mode="exact",
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
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> PreparedSingleGraphEvaluation:
        prepared_batch = self.policy.prepare_batch(batch)
        try:
            analysis = self.analyzer.analyze(
                batch=batch,
                policy=self.policy,
                prepared_batch=prepared_batch,
            )
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
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[PreparedSingleGraphEvaluation]:
        prepared_graphs: list[PreparedSingleGraphEvaluation] = []
        for graph_idx in range(batch.num_graphs):
            prepared_graph = self._prepare_graph_with_fallback(
                batch.select_graph(graph_idx),
                on_invalid_start=on_invalid_start,
            )
            prepared_graphs.append(prepared_graph)
        return prepared_graphs

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
                window_results.append(
                    build_rank_only_result(
                        batch=graph.batch,
                        analysis=graph.analysis,
                        inference_mode="exact",
                        answer_mass_threshold=float(
                            self.eval_cfg.answer_mass_threshold
                        ),
                        support_mass_threshold=float(
                            self.eval_cfg.support_mass_threshold
                        ),
                        probe_count=0,
                        remaining_mass_upper=0.0,
                        stop_reason="rank_only_exact",
                        coverage_certified=True,
                        answer_mass_reference="exact",
                        answer_mass_reference_total=1.0,
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
            prepared_graphs = self.prepare_evaluation_graphs(
                batch,
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
        prepared_graphs = self.prepare_evaluation_graphs(
            batch,
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
