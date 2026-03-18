from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from src.graph_runtime import TrajectoryBatch
from src.models.gflownet import TrajectorySamplerProtocol

from .protocol import MetricEvaluationOutput


class BaseMetricRuntime(ABC):
    sampler: TrajectorySamplerProtocol | None = None
    search: Any = None

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


__all__ = ["BaseMetricRuntime"]
