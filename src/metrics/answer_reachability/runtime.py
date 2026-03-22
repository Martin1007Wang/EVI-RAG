from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.graph_runtime import TrajectoryBatch
from src.models.configs import (
    SearchEvalConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
)
from src.models.gflownet import (
    AnswerReachabilityTrajectorySupervisor,
    ForwardTrajectoryGFNSampler,
    TrajectorySamplerProtocol,
)
from src.models.gflownet import SearchPolicyProtocol
from src.metrics.base import BaseMetricRuntime
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeProtocol

from .artifacts import SupportWindowArtifactWriter
from .batch_evaluator import (
    INVALID_START_REASON,
    ReachabilityBatchEvaluator,
)
from .edge_eval import EdgeRetrievalEvaluator, compute_edge_metrics
from .flow_frontier import (
    FlowFrontierReachabilityAnalyzer,
    FlowFrontierSupportSearch,
)
from .metrics import compute_support_metrics
from .monte_carlo import MonteCarloReachabilityAnalyzer, MonteCarloSupportSearch
from .posterior import aggregate_rank_metrics
from .prediction_io import (
    iter_jsonl_records,
    jsonl_has_records,
    load_edge_retrieval_result,
    load_support_window_result,
)
from .schema import SupportWindowEvalBatch
from .support_search import SupportSearchProtocol


@dataclass
class _OnlinePredictMetricsAccumulator:
    metric_sums: dict[str, float] = field(default_factory=dict)
    count: int = 0
    invalid_start_count: int = 0


class SearchMetricRuntime(BaseMetricRuntime):
    sampler: TrajectorySamplerProtocol | None

    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        reachability_evaluator: ReachabilityBatchEvaluator,
        edge_evaluator: EdgeRetrievalEvaluator,
        sampler: TrajectorySamplerProtocol,
        support_search: SupportSearchProtocol,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.reachability_evaluator = reachability_evaluator
        self.edge_evaluator = edge_evaluator
        self.sampler = sampler
        self.support_search = support_search

    @property
    def search(self) -> SupportSearchProtocol:
        return self.support_search

    def _uses_edge_retrieval_task(self) -> bool:
        return self.eval_cfg.task == "edge_retrieval"

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.evaluate_batch(
                batch=batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
                on_invalid_start=on_invalid_start,
            )
        output = self.reachability_evaluator.evaluate_batch(
            batch=batch,
            metrics_profile=metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=on_invalid_start,
        )
        return MetricEvaluationOutput(
            model_metrics=output.model_metrics,
            primary_metrics=output.support_metrics,
            secondary_metrics=output.rank_metrics,
            results=output.window_results,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]:
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.predict_batch(
                batch=batch,
                metrics_profile=metrics_profile,
                include_answer_support=include_answer_support,
                on_invalid_start=on_invalid_start,
            )
        return self.reachability_evaluator.predict_batch(
            batch=batch,
            metrics_profile=metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=on_invalid_start,
        )

    def build_predict_labels(
        self,
        batch: TrajectoryBatch,
        outputs: list[Any],
    ) -> list[Any]:
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.build_predict_labels(batch, outputs)
        return self.reachability_evaluator.build_predict_labels(batch, outputs)

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> dict[str, float]:
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.summarize_predict_epoch(
                predict_results=predict_results,
                metrics_profile=metrics_profile,
            )
        if not predict_results:
            return {}
        invalid_start_count = sum(
            1
            for result in predict_results
            if result.stop_reason == INVALID_START_REASON
        )
        support_metrics = (
            {}
            if metrics_profile == "rank_only"
            else compute_support_metrics(
                SupportWindowEvalBatch(
                    dataset_scope=predict_results[0].dataset_scope,
                    mass_threshold=float(self.eval_cfg.support_mass_threshold),
                    results=predict_results,
                    window_top_ks=tuple(int(k) for k in self.eval_cfg.window_top_ks),
                )
            )
        )
        rank_metrics = aggregate_rank_metrics(
            results=predict_results,
            answer_top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
        )
        return {
            **rank_metrics,
            **support_metrics,
            "invalid_start_count": invalid_start_count,
            "invalid_start_rate": float(invalid_start_count)
            / float(len(predict_results)),
        }

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
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.write_prediction_artifacts(
                results=results,
                labels=labels,
                output_dir=output_dir,
                split=split,
                artifact_name=artifact_name,
                schema_version=schema_version,
                entity_vocab_path=entity_vocab_path,
                relation_vocab_path=relation_vocab_path,
                questions_path=questions_path,
                overwrite=overwrite,
            )
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

    @staticmethod
    def initialize_predict_metrics_accumulator(
        *,
        metrics_profile: str,
    ) -> _OnlinePredictMetricsAccumulator:
        del metrics_profile
        return _OnlinePredictMetricsAccumulator()

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: _OnlinePredictMetricsAccumulator,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> None:
        if self._uses_edge_retrieval_task():
            for result in predict_results:
                metrics = compute_edge_metrics(
                    results=[result],
                    edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
                )
                for name, value in metrics.items():
                    accumulator.metric_sums[name] = accumulator.metric_sums.get(
                        name, 0.0
                    ) + float(value)
                accumulator.count += 1
            return

        for result in predict_results:
            if result.stop_reason == INVALID_START_REASON:
                accumulator.invalid_start_count += 1
            rank_metrics = aggregate_rank_metrics(
                results=[result],
                answer_top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
            )
            for name, value in rank_metrics.items():
                accumulator.metric_sums[name] = accumulator.metric_sums.get(
                    name, 0.0
                ) + float(value)
            if metrics_profile != "rank_only":
                support_metrics = compute_support_metrics(
                    SupportWindowEvalBatch(
                        dataset_scope=result.dataset_scope,
                        mass_threshold=float(self.eval_cfg.support_mass_threshold),
                        results=[result],
                        window_top_ks=tuple(
                            int(k) for k in self.eval_cfg.window_top_ks
                        ),
                    )
                )
                for name, value in support_metrics.items():
                    if name == "meta/num_samples":
                        continue
                    accumulator.metric_sums[name] = accumulator.metric_sums.get(
                        name, 0.0
                    ) + float(value)
            accumulator.count += 1

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: _OnlinePredictMetricsAccumulator,
        metrics_profile: str,
    ) -> dict[str, float]:
        if accumulator.count < 1:
            return {}
        metrics = {
            name: value / float(accumulator.count)
            for name, value in accumulator.metric_sums.items()
        }
        if not self._uses_edge_retrieval_task():
            if metrics_profile != "rank_only":
                metrics["meta/num_samples"] = float(accumulator.count)
            metrics["invalid_start_count"] = float(accumulator.invalid_start_count)
            metrics["invalid_start_rate"] = float(
                accumulator.invalid_start_count
            ) / float(accumulator.count)
        return metrics

    def summarize_predict_epoch_from_jsonl(
        self,
        *,
        predict_results_path: str | Path,
        metrics_profile: str,
    ) -> dict[str, float]:
        if not jsonl_has_records(predict_results_path):
            return {}
        if self._uses_edge_retrieval_task():
            metric_sums: dict[str, float] = {}
            count = 0
            for record in iter_jsonl_records(predict_results_path):
                result = load_edge_retrieval_result(record)
                metrics = compute_edge_metrics(
                    results=[result],
                    edge_top_ks=tuple(int(k) for k in self.eval_cfg.edge_top_ks),
                )
                for name, value in metrics.items():
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                count += 1
            if count < 1:
                return {}
            return {name: value / float(count) for name, value in metric_sums.items()}

        rank_metric_sums: dict[str, float] = {}
        support_metric_sums: dict[str, float] = {}
        count = 0
        invalid_start_count = 0
        for record in iter_jsonl_records(predict_results_path):
            result = load_support_window_result(record)
            if result.stop_reason == INVALID_START_REASON:
                invalid_start_count += 1
            rank_metrics = aggregate_rank_metrics(
                results=[result],
                answer_top_ks=tuple(int(k) for k in self.eval_cfg.answer_top_ks),
            )
            for name, value in rank_metrics.items():
                rank_metric_sums[name] = rank_metric_sums.get(name, 0.0) + float(value)
            if metrics_profile != "rank_only":
                support_metrics = compute_support_metrics(
                    SupportWindowEvalBatch(
                        dataset_scope=result.dataset_scope,
                        mass_threshold=float(self.eval_cfg.support_mass_threshold),
                        results=[result],
                        window_top_ks=tuple(
                            int(k) for k in self.eval_cfg.window_top_ks
                        ),
                    )
                )
                for name, value in support_metrics.items():
                    if name == "meta/num_samples":
                        continue
                    support_metric_sums[name] = support_metric_sums.get(
                        name, 0.0
                    ) + float(value)
            count += 1
        if count < 1:
            return {}
        metrics = {
            name: value / float(count)
            for name, value in {**rank_metric_sums, **support_metric_sums}.items()
        }
        if metrics_profile != "rank_only":
            metrics["meta/num_samples"] = float(count)
        metrics["invalid_start_count"] = float(invalid_start_count)
        metrics["invalid_start_rate"] = float(invalid_start_count) / float(count)
        return metrics

    def write_prediction_artifacts_from_jsonl(
        self,
        *,
        results_path: str | Path,
        labels_path: str | Path,
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        if not jsonl_has_records(results_path):
            return None
        if self._uses_edge_retrieval_task():
            return self.edge_evaluator.write_prediction_artifacts_from_jsonl(
                results_path=results_path,
                labels_path=labels_path,
                output_dir=output_dir,
                split=split,
                artifact_name=artifact_name,
                overwrite=overwrite,
            )
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
        return writer.write_from_jsonl(
            results_path=results_path,
            labels_path=labels_path,
        )


class SearchMetricRuntimeFactory:
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
    ) -> MetricRuntimeProtocol:
        if eval_cfg.support_search_method == "flow_frontier":
            analyzer = FlowFrontierReachabilityAnalyzer(
                max_steps=int(horizon_cfg.max_steps),
                eval_cfg=eval_cfg,
            )
            support_search = FlowFrontierSupportSearch(
                horizon_cfg=horizon_cfg,
                eval_cfg=eval_cfg,
            )
        else:
            analyzer = MonteCarloReachabilityAnalyzer(
                max_steps=int(horizon_cfg.max_steps),
                eval_cfg=eval_cfg,
            )
            support_search = MonteCarloSupportSearch(
                horizon_cfg=horizon_cfg,
                eval_cfg=eval_cfg,
            )
        edge_analyzer = MonteCarloReachabilityAnalyzer(
            max_steps=int(horizon_cfg.max_steps),
            eval_cfg=eval_cfg,
        )
        trajectory_supervisor = AnswerReachabilityTrajectorySupervisor()
        sampler = ForwardTrajectoryGFNSampler(
            max_steps=int(horizon_cfg.max_steps),
            trajectory_supervisor=trajectory_supervisor,
            force_stop_on_answer_hit=bool(training_cfg.force_stop_on_answer_hit),
        )
        reachability_evaluator = ReachabilityBatchEvaluator(
            eval_cfg=eval_cfg,
            policy=policy,
            analyzer=analyzer,
            support_search=support_search,
        )
        edge_evaluator = EdgeRetrievalEvaluator(
            eval_cfg=eval_cfg,
            policy=policy,
            analyzer=edge_analyzer,
        )
        return SearchMetricRuntime(
            eval_cfg=eval_cfg,
            reachability_evaluator=reachability_evaluator,
            edge_evaluator=edge_evaluator,
            sampler=sampler,
            support_search=support_search,
        )


__all__ = [
    "SearchMetricRuntime",
    "SearchMetricRuntimeFactory",
]
