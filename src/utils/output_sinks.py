from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .metrics_io import to_serializable, write_metrics_json, write_metrics_jsonl

_DEFAULT_STAGE_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "predict": "predict.jsonl",
}


def normalize_optional_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


@dataclass(frozen=True)
class MetricsSnapshotSettings:
    output_dir: str | Path
    filename: str = "metrics.json"

    def resolve_path(self) -> Path:
        return Path(str(self.output_dir)) / str(self.filename)


@dataclass(frozen=True)
class StageMetricsSettings:
    output_dir: str | Path
    stage: str
    step: int
    epoch: int | None = None
    metadata: dict[str, Any] | None = None
    file_name: str | None = None

    def resolve_path(self) -> Path:
        file_name = self.file_name or _DEFAULT_STAGE_FILES.get(
            self.stage, f"{self.stage}.jsonl"
        )
        return Path(str(self.output_dir)) / file_name


@dataclass(frozen=True)
class PredictionArtifactSettings:
    enabled: bool = False
    execution_mode: str = "predict"
    output_root: str | Path | None = None
    artifact_subdir: str = "rankflow"
    artifact_name: str = "rankflow"
    schema_version: int = 1
    split: str = "test"
    dataset_scope: str | None = None
    dataset_variant: str | None = None
    entity_vocab_path: str | Path | None = None
    relation_vocab_path: str | Path | None = None
    questions_path: str | Path | None = None
    dataset_out_dir: str | Path | None = None
    overwrite: bool = True

    def resolve_output_dir(self) -> Path:
        output_root = normalize_optional_path(self.output_root)
        if output_root is None:
            raise ValueError(
                "Prediction artifact writing requires dataset.artifact_dir when enabled."
            )
        dataset_label = self.dataset_variant or self.dataset_scope
        if dataset_label in (None, ""):
            raise ValueError(
                "Prediction artifact writing requires dataset_scope or run.dataset_variant when enabled."
            )
        return output_root / str(self.artifact_subdir) / str(dataset_label)

    def resolve_questions_path(self) -> Path | None:
        questions_path = normalize_optional_path(self.questions_path)
        if questions_path is not None:
            return questions_path if questions_path.exists() else None
        dataset_out_dir = normalize_optional_path(self.dataset_out_dir)
        if dataset_out_dir is None:
            return None
        candidate = dataset_out_dir / "questions.parquet"
        return candidate if candidate.exists() else None


def write_metrics_snapshot(
    *,
    metrics: Mapping[str, Any],
    settings: MetricsSnapshotSettings,
) -> Path:
    return write_metrics_json(path=settings.resolve_path(), metrics=metrics)


def append_stage_metrics(
    *,
    metrics: Mapping[str, Any],
    settings: StageMetricsSettings,
) -> Path | None:
    if not metrics:
        return None
    payload = {str(name): to_serializable(value) for name, value in metrics.items()}
    path = settings.resolve_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_jsonl(
        path=path,
        stage=str(settings.stage),
        metrics=payload,
        step=int(settings.step),
        epoch=settings.epoch,
        metadata=settings.metadata,
    )
    return path


def write_prediction_artifacts(
    model: Any,
    *,
    settings: PredictionArtifactSettings,
) -> dict[str, Path] | None:
    existing_paths = getattr(model, "predict_artifact_paths", None)
    if isinstance(existing_paths, dict) and existing_paths:
        return existing_paths
    if not settings.enabled or settings.execution_mode != "predict":
        return None
    write_fn = getattr(model, "write_prediction_artifacts", None)
    if not callable(write_fn):
        return None
    paths = write_fn(
        output_dir=settings.resolve_output_dir(),
        split=str(settings.split),
        artifact_name=str(settings.artifact_name),
        schema_version=int(settings.schema_version),
        entity_vocab_path=normalize_optional_path(settings.entity_vocab_path),
        relation_vocab_path=normalize_optional_path(settings.relation_vocab_path),
        questions_path=settings.resolve_questions_path(),
        overwrite=bool(settings.overwrite),
    )
    if not isinstance(paths, dict):
        return None
    setattr(model, "predict_artifact_paths", paths)
    return paths


__all__ = [
    "MetricsSnapshotSettings",
    "PredictionArtifactSettings",
    "StageMetricsSettings",
    "append_stage_metrics",
    "normalize_optional_path",
    "write_metrics_snapshot",
    "write_prediction_artifacts",
]
