from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from src.graph_runtime import TrajectoryBatch
from src.models.configs import GFlowNetTrainingConfig, HorizonConfig, SearchEvalConfig
from src.models.gflownet import TrajectorySamplerProtocol
from src.models.gflownet import SearchPolicyProtocol


@dataclass(frozen=True)
class MetricEvaluationOutput:
    model_metrics: dict[str, float] = field(default_factory=dict)
    primary_metrics: dict[str, float] = field(default_factory=dict)
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    results: list[Any] = field(default_factory=list)


class MetricRuntimeProtocol(Protocol):
    sampler: TrajectorySamplerProtocol | None
    search: Any

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput: ...

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        metrics_profile: str,
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
        metrics_profile: str,
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


class MetricRuntimeFactoryProtocol(Protocol):
    def build_runtime(
        self,
        *,
        horizon_cfg: HorizonConfig,
        training_cfg: GFlowNetTrainingConfig,
        eval_cfg: SearchEvalConfig,
        policy: SearchPolicyProtocol,
    ) -> MetricRuntimeProtocol: ...


__all__ = [
    "MetricEvaluationOutput",
    "MetricRuntimeFactoryProtocol",
    "MetricRuntimeProtocol",
]
