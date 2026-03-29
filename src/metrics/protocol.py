from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from src.graph import TrajectoryBatch
from src.models.configs import GFlowNetTrainingConfig, HorizonConfig, SearchEvalConfig

from .prediction_io import PredictionCodecProtocol


@dataclass(frozen=True)
class MetricEvaluationOutput:
    model_metrics: dict[str, float] = field(default_factory=dict)
    primary_metrics: dict[str, float] = field(default_factory=dict)
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    results: list[Any] = field(default_factory=list)

    @property
    def metrics(self) -> dict[str, float]:
        return self.primary_metrics

    @property
    def diagnostics(self) -> dict[str, float]:
        return self.secondary_metrics


class MetricRuntimeProtocol(Protocol):
    @property
    def prediction_codec(self) -> PredictionCodecProtocol: ...

    sampler: Any
    search: Any

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput: ...

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]: ...

    def build_predict_labels(
        self,
        batch: TrajectoryBatch,
        outputs: list[Any],
    ) -> list[Any]: ...

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        report_profile: str,
    ) -> dict[str, float]: ...

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
    ) -> dict[str, Path] | None: ...

    def initialize_predict_metrics_accumulator(
        self,
        *,
        report_profile: str,
    ) -> Any: ...

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: Any,
        predict_results: list[Any],
        report_profile: str,
    ) -> None: ...

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: Any,
        report_profile: str,
    ) -> dict[str, float]: ...

    def summarize_predict_epoch_from_jsonl(
        self,
        *,
        predict_results_path: str | Path,
        report_profile: str,
    ) -> dict[str, float]: ...

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
    ) -> dict[str, Path] | None: ...


class MetricRuntimeFactoryProtocol(Protocol):
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: Any,
    ) -> MetricRuntimeProtocol: ...


__all__ = [
    "MetricEvaluationOutput",
    "MetricRuntimeFactoryProtocol",
    "MetricRuntimeProtocol",
]
