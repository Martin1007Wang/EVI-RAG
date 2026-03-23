from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any, Callable, cast

from src.graph import TrajectoryBatch
from src.metrics.prediction_io import (
    PredictionCodecProtocol,
    append_jsonl_records,
    iter_jsonl_records,
    jsonl_has_records,
)
from src.metrics.protocol import MetricEvaluationOutput, MetricRuntimeProtocol
from src.models.gflownet import TrajectorySamplerProtocol


PredictionResult = Any
PredictionLabel = Any


@dataclass
class PredictionEpochState:
    results: list[PredictionResult] = field(default_factory=list)
    labels: list[PredictionLabel] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    metrics_accumulator: Any | None = None
    prediction_codec: PredictionCodecProtocol | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)
    results_jsonl_path: Path | None = None
    labels_jsonl_path: Path | None = None

    def reset(self) -> None:
        self._clear_file_cache()
        self.results.clear()
        self.labels.clear()
        self.metrics.clear()
        self.metrics_accumulator = None
        self.prediction_codec = None

    def record_batch(
        self,
        *,
        results: list[PredictionResult],
        labels: list[PredictionLabel],
        codec: PredictionCodecProtocol,
    ) -> None:
        if not results:
            return
        if self.prediction_codec is None:
            self.prediction_codec = codec
        elif self.prediction_codec.kind != codec.kind:
            raise ValueError(
                "Prediction epoch state cannot mix different prediction result types."
            )
        self.results.clear()
        self.labels.clear()
        results_path, labels_path = self._ensure_file_cache()
        append_jsonl_records(
            results_path,
            records=[codec.serialize_result(result) for result in results],
        )
        append_jsonl_records(
            labels_path,
            records=[codec.serialize_label(label) for label in labels],
        )

    def finalize(self, metrics: dict[str, float]) -> None:
        self.metrics = dict(metrics)

    def replace(
        self,
        *,
        results: list[PredictionResult] | None = None,
        labels: list[PredictionLabel] | None = None,
        metrics: dict[str, float] | None = None,
        codec: PredictionCodecProtocol | None = None,
    ) -> None:
        if results is not None or labels is not None:
            self._clear_file_cache()
            self.results.clear()
            self.labels.clear()
            self.metrics_accumulator = None
            self.prediction_codec = codec
        if results is not None:
            self.results = list(results)
        if labels is not None:
            self.labels = list(labels)
        if metrics is not None:
            self.metrics = dict(metrics)

    def get_results(self) -> list[PredictionResult]:
        if self.results:
            return list(self.results)
        if self.prediction_codec is None or self.results_jsonl_path is None:
            return []
        return [
            self.prediction_codec.deserialize_result(record)
            for record in iter_jsonl_records(self.results_jsonl_path)
        ]

    def get_labels(self) -> list[PredictionLabel]:
        if self.labels:
            return list(self.labels)
        if self.prediction_codec is None or self.labels_jsonl_path is None:
            return []
        return [
            self.prediction_codec.deserialize_label(record)
            for record in iter_jsonl_records(self.labels_jsonl_path)
        ]

    def has_file_cache(self) -> bool:
        return jsonl_has_records(self.results_jsonl_path)

    def _ensure_file_cache(self) -> tuple[Path, Path]:
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="rankflow_predict_")
            temp_root = Path(self.temp_dir.name)
            self.results_jsonl_path = temp_root / "results.jsonl"
            self.labels_jsonl_path = temp_root / "labels.jsonl"
        assert self.results_jsonl_path is not None
        assert self.labels_jsonl_path is not None
        return self.results_jsonl_path, self.labels_jsonl_path

    def _clear_file_cache(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
        self.temp_dir = None
        self.results_jsonl_path = None
        self.labels_jsonl_path = None


@dataclass(frozen=True)
class PredictionArtifactWriteConfig:
    output_dir: str | Path
    split: str
    artifact_name: str = "rankflow"
    schema_version: int = 1
    entity_vocab_path: str | Path | None = None
    relation_vocab_path: str | Path | None = None
    questions_path: str | Path | None = None
    overwrite: bool = True


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
        self._prediction_state = PredictionEpochState()

    @property
    def sampler(self) -> TrajectorySamplerProtocol | None:
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
    ) -> tuple[
        dict[str, float],
        list[PredictionResult],
        dict[str, float],
        dict[str, float],
    ]:
        outputs = self.evaluate_batch_output(
            batch=batch,
            include_answer_support=include_answer_support,
        )
        return (
            outputs.metrics,
            cast(list[PredictionResult], outputs.results),
            outputs.model_metrics,
            outputs.diagnostics,
        )

    def reset_prediction_state(self) -> None:
        self._prediction_state.reset()

    def replace_prediction_state(
        self,
        *,
        results: list[PredictionResult] | None = None,
        labels: list[PredictionLabel] | None = None,
        metrics: dict[str, float] | None = None,
    ) -> None:
        codec = self.metric_runtime.prediction_codec if results or labels else None
        self._prediction_state.replace(
            results=results,
            labels=labels,
            metrics=metrics,
            codec=codec,
        )

    def get_predict_results(self) -> list[PredictionResult]:
        return self._prediction_state.get_results()

    def get_predict_labels(self) -> list[PredictionLabel]:
        return self._prediction_state.get_labels()

    def predict_batch(self, *, batch: TrajectoryBatch) -> list[PredictionResult]:
        return cast(
            list[PredictionResult],
            self.metric_runtime.predict_batch(
                batch=batch,
                metrics_profile=self.metrics_profile,
                include_answer_support=self.metrics_profile != "rank_only",
                on_invalid_start=self.on_invalid_start,
            ),
        )

    def record_prediction_batch(
        self,
        *,
        batch: TrajectoryBatch,
        outputs: list[PredictionResult] | None,
    ) -> None:
        if not outputs:
            return
        if self._prediction_state.metrics_accumulator is None:
            self._prediction_state.metrics_accumulator = (
                self.metric_runtime.initialize_predict_metrics_accumulator(
                    metrics_profile=self.metrics_profile
                )
            )
        self.metric_runtime.update_predict_metrics_accumulator(
            accumulator=self._prediction_state.metrics_accumulator,
            predict_results=cast(list[Any], outputs),
            metrics_profile=self.metrics_profile,
        )
        self._prediction_state.record_batch(
            results=list(outputs),
            labels=cast(
                list[PredictionLabel],
                self.metric_runtime.build_predict_labels(batch, outputs),
            ),
            codec=self.metric_runtime.prediction_codec,
        )

    def finalize_prediction_epoch(self) -> None:
        if self._prediction_state.metrics_accumulator is not None:
            metrics = self.metric_runtime.finalize_predict_metrics_accumulator(
                accumulator=self._prediction_state.metrics_accumulator,
                metrics_profile=self.metrics_profile,
            )
        elif self._prediction_state.has_file_cache():
            assert self._prediction_state.results_jsonl_path is not None
            metrics = self.metric_runtime.summarize_predict_epoch_from_jsonl(
                predict_results_path=self._prediction_state.results_jsonl_path,
                metrics_profile=self.metrics_profile,
            )
        else:
            metrics = self.metric_runtime.summarize_predict_epoch(
                predict_results=self._prediction_state.get_results(),
                metrics_profile=self.metrics_profile,
            )
        self._prediction_state.finalize(cast(dict[str, float], metrics))

    def get_predict_metrics(self) -> dict[str, float]:
        return dict(self._prediction_state.metrics)

    def write_prediction_artifacts(
        self,
        *,
        settings: PredictionArtifactWriteConfig,
    ) -> dict[str, Path] | None:
        if self._prediction_state.has_file_cache():
            assert self._prediction_state.results_jsonl_path is not None
            assert self._prediction_state.labels_jsonl_path is not None
            return cast(
                dict[str, Path] | None,
                self.metric_runtime.write_prediction_artifacts_from_jsonl(
                    results_path=self._prediction_state.results_jsonl_path,
                    labels_path=self._prediction_state.labels_jsonl_path,
                    output_dir=settings.output_dir,
                    split=settings.split,
                    artifact_name=settings.artifact_name,
                    schema_version=settings.schema_version,
                    entity_vocab_path=settings.entity_vocab_path,
                    relation_vocab_path=settings.relation_vocab_path,
                    questions_path=settings.questions_path,
                    overwrite=settings.overwrite,
                ),
            )
        return self.metric_runtime.write_prediction_artifacts(
            results=self._prediction_state.get_results(),
            labels=self._prediction_state.get_labels(),
            output_dir=settings.output_dir,
            split=settings.split,
            artifact_name=settings.artifact_name,
            schema_version=settings.schema_version,
            entity_vocab_path=settings.entity_vocab_path,
            relation_vocab_path=settings.relation_vocab_path,
            questions_path=settings.questions_path,
            overwrite=settings.overwrite,
        )


__all__ = [
    "MetricRuntimeController",
    "PredictionArtifactWriteConfig",
    "PredictionEpochState",
    "PredictionLabel",
    "PredictionResult",
]
