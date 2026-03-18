from __future__ import annotations

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
from .edge_eval import EdgeRetrievalEvaluator
from .exact_analysis import ExactReachabilityAnalyzer
from .metrics import compute_support_metrics
from .posterior import aggregate_rank_metrics
from .schema import SupportWindowEvalBatch
from .support_search import ExactSupportSearch


class SearchMetricRuntime(BaseMetricRuntime):
    sampler: TrajectorySamplerProtocol | None

    def __init__(
        self,
        *,
        eval_cfg: SearchEvalConfig,
        reachability_evaluator: ReachabilityBatchEvaluator,
        edge_evaluator: EdgeRetrievalEvaluator,
        sampler: TrajectorySamplerProtocol,
        support_search: ExactSupportSearch,
    ) -> None:
        self.eval_cfg = eval_cfg
        self.reachability_evaluator = reachability_evaluator
        self.edge_evaluator = edge_evaluator
        self.sampler = sampler
        self.support_search = support_search

    @property
    def search(self) -> ExactSupportSearch:
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


class SearchMetricRuntimeFactory:
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
    ) -> MetricRuntimeProtocol:
        analyzer = ExactReachabilityAnalyzer(max_steps=int(horizon_cfg.max_steps))
        support_search = ExactSupportSearch(
            horizon_cfg=horizon_cfg,
            eval_cfg=eval_cfg,
            analyzer=analyzer,
        )
        trajectory_supervisor = AnswerReachabilityTrajectorySupervisor(
            epsilon=float(training_cfg.reward_epsilon),
            failure_reward_mode=str(training_cfg.failure_reward_mode),
        )
        sampler = ForwardTrajectoryGFNSampler(
            max_steps=int(horizon_cfg.max_steps),
            trajectory_supervisor=trajectory_supervisor,
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
            analyzer=analyzer,
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
