from __future__ import annotations

from pathlib import Path

from lightning.pytorch.callbacks import Callback

from src.utils.output_sinks import (
    PredictionArtifactSettings,
    normalize_optional_path,
    write_prediction_artifacts,
)
from src.utils.logging_utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


def _resolve_execution_mode(
    *,
    execution_mode: str | None,
) -> str:
    mode = str(execution_mode or "predict").strip().lower()
    if mode not in {"predict", "test"}:
        raise ValueError(
            "PredictionArtifactsWriter execution mode must be one of {'predict', 'test'}."
        )
    return mode


class PredictionArtifactsWriter(Callback):
    """Persist prediction artifacts for models that expose a writer hook."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        execution_mode: str = "predict",
        output_root: str | Path | None = None,
        artifact_subdir: str = "rankflow",
        artifact_name: str = "rankflow",
        schema_version: int = 1,
        split: str = "test",
        dataset_scope: str | None = None,
        dataset_variant: str | None = None,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        dataset_out_dir: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.execution_mode = _resolve_execution_mode(execution_mode=execution_mode)
        self.output_root = normalize_optional_path(output_root)
        self.artifact_subdir = str(artifact_subdir)
        self.artifact_name = str(artifact_name)
        self.schema_version = int(schema_version)
        self.split = str(split)
        self.dataset_scope = None if dataset_scope in (None, "") else str(dataset_scope)
        self.dataset_variant = (
            None if dataset_variant in (None, "") else str(dataset_variant)
        )
        self.entity_vocab_path = normalize_optional_path(entity_vocab_path)
        self.relation_vocab_path = normalize_optional_path(relation_vocab_path)
        self.questions_path = normalize_optional_path(questions_path)
        self.dataset_out_dir = normalize_optional_path(dataset_out_dir)
        self.overwrite = bool(overwrite)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or self.execution_mode != "predict":
            return
        if not getattr(trainer, "is_global_zero", True):
            return
        paths = write_prediction_artifacts(
            pl_module,
            settings=PredictionArtifactSettings(
                enabled=self.enabled,
                execution_mode=self.execution_mode,
                output_root=self.output_root,
                artifact_subdir=self.artifact_subdir,
                artifact_name=self.artifact_name,
                schema_version=self.schema_version,
                split=self.split,
                dataset_scope=self.dataset_scope,
                dataset_variant=self.dataset_variant,
                entity_vocab_path=self.entity_vocab_path,
                relation_vocab_path=self.relation_vocab_path,
                questions_path=self.questions_path,
                dataset_out_dir=self.dataset_out_dir,
                overwrite=self.overwrite,
            ),
        )
        if not isinstance(paths, dict) or not paths:
            return
        log_path = (
            paths.get("prompt_path")
            or paths.get("results_path")
            or next(iter(paths.values()))
        )
        log.info("Artifacts written to %s", log_path)


__all__ = ["PredictionArtifactsWriter"]
