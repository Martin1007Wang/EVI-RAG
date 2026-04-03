from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.metrics.serialization import write_metrics_json, write_metrics_jsonl


_DEFAULT_STAGE_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "predict": "predict.jsonl",
}


def _optional_path(value: str | Path | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))


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
        output_root = _optional_path(self.output_root)
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
        questions_path = _optional_path(self.questions_path)
        if questions_path is not None:
            return questions_path if questions_path.exists() else None
        dataset_out_dir = _optional_path(self.dataset_out_dir)
        if dataset_out_dir is None:
            return None
        candidate = dataset_out_dir / "questions.parquet"
        return candidate if candidate.exists() else None

    def cache_key(self) -> tuple[Any, ...]:
        return (
            bool(self.enabled),
            str(self.execution_mode),
            _optional_path(self.output_root),
            str(self.artifact_subdir),
            str(self.artifact_name),
            int(self.schema_version),
            str(self.split),
            None if self.dataset_scope in (None, "") else str(self.dataset_scope),
            None if self.dataset_variant in (None, "") else str(self.dataset_variant),
            _optional_path(self.entity_vocab_path),
            _optional_path(self.relation_vocab_path),
            _optional_path(self.questions_path),
            _optional_path(self.dataset_out_dir),
            bool(self.overwrite),
        )


def write_metrics_snapshot(
    *,
    output_dir: str | Path,
    metrics: Mapping[str, Any],
    filename: str = "metrics.json",
) -> Path:
    return write_metrics_json(
        path=Path(str(output_dir)) / str(filename), metrics=metrics
    )


def append_stage_metrics(
    *,
    output_dir: str | Path,
    stage: str,
    step: int,
    metrics: Mapping[str, Any],
    epoch: int | None = None,
    record_kind: str | None = None,
    metadata: dict[str, Any] | None = None,
    file_name: str | None = None,
) -> Path | None:
    if not metrics:
        return None
    resolved_file_name = file_name or _DEFAULT_STAGE_FILES.get(stage, f"{stage}.jsonl")
    path = Path(str(output_dir)) / resolved_file_name
    write_metrics_jsonl(
        path=path,
        stage=str(stage),
        metrics=metrics,
        step=int(step),
        epoch=epoch,
        record_kind=record_kind,
        metadata=metadata,
    )
    return path


def write_prediction_artifacts(
    model: Any,
    *,
    settings: PredictionArtifactSettings,
) -> dict[str, Path] | None:
    if not settings.enabled or settings.execution_mode != "predict":
        return None

    cache_key = settings.cache_key()
    existing_paths = getattr(model, "predict_artifact_paths", None)
    existing_cache_key = getattr(model, "predict_artifact_settings_cache_key", None)
    if (
        isinstance(existing_paths, dict)
        and existing_paths
        and existing_cache_key == cache_key
    ):
        return existing_paths

    write_fn = getattr(model, "write_prediction_artifacts", None)
    if not callable(write_fn):
        return None

    paths = write_fn(
        output_dir=settings.resolve_output_dir(),
        split=str(settings.split),
        artifact_name=str(settings.artifact_name),
        schema_version=int(settings.schema_version),
        entity_vocab_path=_optional_path(settings.entity_vocab_path),
        relation_vocab_path=_optional_path(settings.relation_vocab_path),
        questions_path=settings.resolve_questions_path(),
        overwrite=bool(settings.overwrite),
    )
    if not isinstance(paths, dict):
        return None

    setattr(model, "predict_artifact_paths", paths)
    setattr(model, "predict_artifact_settings_cache_key", cache_key)
    return paths


__all__ = [
    "PredictionArtifactSettings",
    "append_stage_metrics",
    "write_metrics_snapshot",
    "write_prediction_artifacts",
]
