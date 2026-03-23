from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import TrajectorySamplerProtocol

from .prediction_io import (
    PredictionCodecProtocol,
    iter_jsonl_records,
    jsonl_has_records,
)
from .protocol import MetricEvaluationOutput


class BaseMetricRuntime(ABC):
    _prediction_codec: PredictionCodecProtocol
    sampler: TrajectorySamplerProtocol | None = None
    search: Any = None

    @property
    def prediction_codec(self) -> PredictionCodecProtocol:
        return self._prediction_codec

    @abstractmethod
    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]:
        raise NotImplementedError

    @abstractmethod
    def build_predict_labels(
        self,
        batch: TrajectoryBatch,
        outputs: list[Any],
    ) -> list[Any]:
        raise NotImplementedError

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> dict[str, float]:
        del predict_results, metrics_profile
        return {}

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
        del (
            results,
            labels,
            output_dir,
            split,
            artifact_name,
            schema_version,
            entity_vocab_path,
            relation_vocab_path,
            questions_path,
            overwrite,
        )
        return None

    def initialize_predict_metrics_accumulator(
        self,
        *,
        metrics_profile: str,
    ) -> Any:
        del metrics_profile
        return None

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: Any,
        predict_results: list[Any],
        metrics_profile: str,
    ) -> None:
        del accumulator, predict_results, metrics_profile

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: Any,
        metrics_profile: str,
    ) -> dict[str, float]:
        del accumulator, metrics_profile
        return {}

    def summarize_predict_epoch_from_jsonl(
        self,
        *,
        predict_results_path: str | Path,
        metrics_profile: str,
    ) -> dict[str, float]:
        if not jsonl_has_records(predict_results_path):
            return {}
        results = [
            self.prediction_codec.deserialize_result(record)
            for record in iter_jsonl_records(predict_results_path)
        ]
        return self.summarize_predict_epoch(
            predict_results=results,
            metrics_profile=metrics_profile,
        )

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
        results = [
            self.prediction_codec.deserialize_result(record)
            for record in iter_jsonl_records(results_path)
        ]
        labels = [
            self.prediction_codec.deserialize_label(record)
            for record in iter_jsonl_records(labels_path)
        ]
        return self.write_prediction_artifacts(
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


__all__ = ["BaseMetricRuntime"]
