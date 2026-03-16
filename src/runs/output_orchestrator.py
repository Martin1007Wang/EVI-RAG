from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig

from src.utils.output_sinks import (
    PredictionArtifactSettings,
    write_prediction_artifacts,
)


@dataclass(frozen=True)
class RunOutputResult:
    metrics: dict[str, Any]
    metrics_path: Path | None = None
    artifact_paths: dict[str, Path] | None = None


class RunOutputOrchestrator:
    def __init__(
        self,
        *,
        collect_metrics: Callable[[dict[str, Any], Any], dict[str, Any]],
        resolve_metrics_filename: Callable[[DictConfig], str | None],
        save_metrics: Callable[[DictConfig, dict[str, Any], str], Path],
        build_artifact_settings: Callable[[DictConfig], PredictionArtifactSettings],
        empty_metrics_warning: str = "No metrics were produced; skipping metrics.json.",
    ) -> None:
        self.collect_metrics = collect_metrics
        self.resolve_metrics_filename = resolve_metrics_filename
        self.save_metrics = save_metrics
        self.build_artifact_settings = build_artifact_settings
        self.empty_metrics_warning = str(empty_metrics_warning)

    def persist(
        self,
        *,
        cfg: DictConfig,
        callback_metrics: dict[str, Any],
        model: Any,
        log: Any,
    ) -> RunOutputResult:
        metrics = self.collect_metrics(callback_metrics, model)
        metrics_path: Path | None = None
        if metrics:
            metrics_filename = self.resolve_metrics_filename(cfg)
            if metrics_filename not in (None, ""):
                metrics_path = self.save_metrics(cfg, metrics, str(metrics_filename))
                log.info("Metrics saved to %s", metrics_path)
        else:
            log.warning(self.empty_metrics_warning)

        artifact_paths = write_prediction_artifacts(
            model,
            settings=self.build_artifact_settings(cfg),
        )
        if isinstance(artifact_paths, dict) and artifact_paths:
            log_path = (
                artifact_paths.get("prompt_path")
                or artifact_paths.get("results_path")
                or next(iter(artifact_paths.values()))
            )
            log.info("Artifacts available at %s", log_path)
        else:
            artifact_paths = None
        return RunOutputResult(
            metrics=metrics,
            metrics_path=metrics_path,
            artifact_paths=artifact_paths,
        )


__all__ = ["RunOutputOrchestrator", "RunOutputResult"]
