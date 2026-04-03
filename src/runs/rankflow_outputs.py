from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from src.runs.common import normalize_dataset_scope, resolve_execution_mode
from src.runs.output import (
    PredictionArtifactSettings,
    write_metrics_snapshot,
    write_prediction_artifacts,
)


def collect_rankflow_metrics(
    *, callback_metrics: dict[str, Any], model: Any
) -> dict[str, Any]:
    if callback_metrics:
        return callback_metrics
    get_predict_metrics = getattr(model, "get_predict_metrics", None)
    if callable(get_predict_metrics):
        metrics = get_predict_metrics()
        if isinstance(metrics, dict):
            return metrics
    metrics_from_model = getattr(model, "predict_metrics", None)
    return (
        metrics_from_model if isinstance(metrics_from_model, dict) else callback_metrics
    )


def resolve_metrics_filename(
    *,
    run_cfg: DictConfig | dict[str, Any],
    dataset_cfg: DictConfig | dict[str, Any],
) -> str:
    metrics_filename = "metrics.json"
    split = run_cfg.get("split") if hasattr(run_cfg, "get") else None
    dataset_variant = (
        run_cfg.get("dataset_variant") if hasattr(run_cfg, "get") else None
    )
    scope = None
    if dataset_variant:
        scope = normalize_dataset_scope(dataset_cfg)
        metrics_filename = f"metrics_{scope}.json"
    if bool(run_cfg.get("run_all_splits", False)) and split not in (None, ""):
        prefix = f"metrics_{scope}_" if scope else "metrics_"
        metrics_filename = f"{prefix}{split}.json"
    return metrics_filename


def build_artifact_settings(
    cfg: DictConfig, *, default_name: str
) -> PredictionArtifactSettings:
    run_cfg = cfg.get("run") or {}
    dataset_cfg = cfg.get("dataset") or {}
    dataset_paths = dataset_cfg.get("paths") or {}
    return PredictionArtifactSettings(
        enabled=bool(run_cfg.get("write_artifacts", False)),
        execution_mode=resolve_execution_mode(run_cfg),
        output_root=dataset_cfg.get("artifact_dir"),
        artifact_subdir=str(run_cfg.get("artifact_subdir") or default_name),
        split=str(run_cfg.get("split") or "test"),
        dataset_scope=normalize_dataset_scope(dataset_cfg),
        dataset_variant=run_cfg.get("dataset_variant"),
        entity_vocab_path=dataset_paths.get("entity_vocab"),
        relation_vocab_path=dataset_paths.get("relation_vocab"),
        dataset_out_dir=dataset_cfg.get("out_dir"),
        overwrite=bool(run_cfg.get("artifact_overwrite", True)),
    )


def persist_rankflow_outputs(
    *,
    cfg: DictConfig,
    callback_metrics: dict[str, Any],
    model: Any,
    log: Any,
    default_name: str,
) -> dict[str, Any]:
    metrics = collect_rankflow_metrics(callback_metrics=callback_metrics, model=model)
    if metrics:
        metrics_path = write_metrics_snapshot(
            output_dir=cfg.paths.output_dir,
            metrics=metrics,
            filename=resolve_metrics_filename(
                run_cfg=cfg.get("run") or {},
                dataset_cfg=cfg.dataset,
            ),
        )
        log.info("Metrics saved to %s", metrics_path)
    else:
        log.warning("No metrics were produced; skipping metrics.json.")

    artifact_paths = write_prediction_artifacts(
        model,
        settings=build_artifact_settings(cfg, default_name=default_name),
    )
    if isinstance(artifact_paths, dict) and artifact_paths:
        log_path = (
            artifact_paths.get("prompt_path")
            or artifact_paths.get("results_path")
            or next(iter(artifact_paths.values()))
        )
        log.info("Artifacts available at %s", log_path)
    return metrics


__all__ = [
    "build_artifact_settings",
    "collect_rankflow_metrics",
    "persist_rankflow_outputs",
    "resolve_metrics_filename",
]
