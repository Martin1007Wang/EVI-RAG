from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.graph_runtime import TrajectoryBatch
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeProtocol


@dataclass
class PredictionEpochState:
    results: list[Any] = field(default_factory=list)
    labels: list[Any] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.results.clear()
        self.labels.clear()
        self.metrics.clear()

    def record_batch(self, *, results: list[Any], labels: list[Any]) -> None:
        self.results.extend(results)
        self.labels.extend(labels)

    def finalize(self, metrics: dict[str, Any]) -> None:
        self.metrics = dict(metrics)


class MetricRuntimeController:
    def __init__(
        self,
        *,
        metric_runtime: MetricRuntimeProtocol,
        metrics_profile: str,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> None:
        self.metric_runtime = metric_runtime
        self.metrics_profile = str(metrics_profile)
        self.on_invalid_start = on_invalid_start
        self.prediction_state = PredictionEpochState()

    @property
    def sampler(self) -> Any:
        return self.metric_runtime.sampler

    @property
    def search(self) -> Any:
        return self.metric_runtime.search

    def evaluate_batch_output(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool = False,
    ) -> MetricEvaluationOutput:
        return self.metric_runtime.evaluate_batch(
            batch=batch,
            metrics_profile=self.metrics_profile,
            include_answer_support=include_answer_support,
            on_invalid_start=self.on_invalid_start,
        )

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool = False,
    ) -> tuple[dict[str, float], list[Any], dict[str, Any], dict[str, float]]:
        outputs = self.evaluate_batch_output(
            batch=batch,
            include_answer_support=include_answer_support,
        )
        return (
            outputs.primary_metrics,
            outputs.results,
            outputs.model_metrics,
            outputs.secondary_metrics,
        )

    def reset_prediction_state(self) -> None:
        self.prediction_state.reset()

    def predict_batch(self, *, batch: TrajectoryBatch) -> list[Any]:
        return self.metric_runtime.predict_batch(
            batch=batch,
            metrics_profile=self.metrics_profile,
            include_answer_support=self.metrics_profile != "rank_only",
            on_invalid_start=self.on_invalid_start,
        )

    def record_prediction_batch(
        self,
        *,
        batch: TrajectoryBatch,
        outputs: list[Any] | None,
    ) -> None:
        if not outputs:
            return
        self.prediction_state.record_batch(
            results=list(outputs),
            labels=self.metric_runtime.build_predict_labels(batch, outputs),
        )

    def finalize_prediction_epoch(self) -> None:
        self.prediction_state.finalize(
            self.metric_runtime.summarize_predict_epoch(
                predict_results=self.prediction_state.results,
                metrics_profile=self.metrics_profile,
            )
        )

    def get_predict_metrics(self) -> dict[str, Any]:
        return dict(self.prediction_state.metrics)

    def write_prediction_artifacts(
        self,
        *,
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        return self.metric_runtime.write_prediction_artifacts(
            results=self.prediction_state.results,
            labels=self.prediction_state.labels,
            output_dir=output_dir,
            split=split,
            artifact_name=artifact_name,
            schema_version=schema_version,
            entity_vocab_path=entity_vocab_path,
            relation_vocab_path=relation_vocab_path,
            questions_path=questions_path,
            overwrite=overwrite,
        )


__all__ = ["MetricRuntimeController", "PredictionEpochState"]
