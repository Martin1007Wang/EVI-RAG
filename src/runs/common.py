from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.output_sinks import MetricsSnapshotSettings, write_metrics_snapshot


_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
_DATASET_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "dataset"
_MISSING_MODE_VALUES = {None, "", "null", "None"}


@dataclass(frozen=True)
class DatasetVariantSpec:
    label: str
    dataset_name: str
    dataset_cfg: DictConfig
    run_overrides: dict[str, Any] = field(default_factory=dict)


def resolve_execution_mode(run_cfg: dict[str, Any] | Any) -> str:
    raw_execution_mode = (
        run_cfg.get("execution_mode") if hasattr(run_cfg, "get") else None
    )
    raw = None if raw_execution_mode in _MISSING_MODE_VALUES else raw_execution_mode
    mode = str(raw or "predict").strip().lower()
    if mode in {"predict", "test"}:
        return mode
    raise ValueError("run.execution_mode must be one of {'predict', 'test'}.")


def resolve_splits(raw_splits: Sequence[str]) -> list[str]:
    split_list = [str(split) for split in raw_splits]
    if not split_list:
        raise ValueError(
            "run.splits must be a non-empty list when run.run_all_splits=true."
        )
    return split_list


def normalize_dataset_scope(dataset_cfg: DictConfig | dict[str, Any]) -> str:
    scope_raw = (
        dataset_cfg.get("dataset_scope") if hasattr(dataset_cfg, "get") else None
    )
    scope = str(scope_raw or "").strip().lower()
    if scope in {"full", "sub"}:
        return scope
    name_raw = dataset_cfg.get("name") if hasattr(dataset_cfg, "get") else ""
    name = str(name_raw or "")
    return "sub" if name.endswith("-sub") else "full"


def _clone_cfg_node(node: Any) -> Any:
    if node is None:
        return None
    if OmegaConf.is_config(node):
        return OmegaConf.create(OmegaConf.to_container(node, resolve=False))
    return OmegaConf.create(node)


@contextmanager
def _hydra_compose_context():
    if GlobalHydra.instance().is_initialized():
        yield
        return
    with hydra.initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        yield


def _normalize_override_list(raw: Any, *, field_name: str) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        return [raw]
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        overrides = [str(item).strip() for item in list(raw) if str(item).strip()]
        return overrides
    raise TypeError(f"{field_name} must be a string or list of strings.")


def _normalize_override_mapping(raw: Any, *, field_name: str) -> DictConfig:
    if raw in (None, ""):
        return OmegaConf.create({})
    cfg = OmegaConf.create(raw)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{field_name} must be a mapping/object.")
    return cfg


def load_dataset_config_by_name(
    name: str,
    paths_cfg: DictConfig,
    *,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    path = _DATASET_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    data_dir = paths_cfg.get("data_dir") if hasattr(paths_cfg, "get") else None
    if data_dir in (None, ""):
        raise ValueError("paths.data_dir is required to compose dataset configs.")
    compose_overrides = [f"dataset={name}", f"paths.data_dir={data_dir}"]
    compose_overrides.extend(str(item) for item in (overrides or []))
    with _hydra_compose_context():
        cfg = hydra.compose(
            config_name="eval.yaml",
            overrides=compose_overrides,
        )
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    return OmegaConf.create(dataset_cfg)


def compose_config(
    *,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    with _hydra_compose_context():
        return hydra.compose(
            config_name=config_name,
            overrides=[str(item) for item in (overrides or [])],
        )


def resolve_dataset_variants(cfg: DictConfig) -> list[DatasetVariantSpec]:
    run_cfg = cfg.get("run") or {}
    raw_variants = run_cfg.get("dataset_variants")
    if not raw_variants:
        return []
    if OmegaConf.is_list(raw_variants) or isinstance(raw_variants, (list, tuple)):
        items = list(raw_variants)
    else:
        items = [raw_variants]

    variants: list[DatasetVariantSpec] = []
    for item in items:
        if hasattr(item, "get"):
            if item.get("name") not in (None, ""):
                raise ValueError(
                    "dataset_variants entries must use `dataset`, not the removed `name` key."
                )
            if item.get("compose_overrides") not in (None, []):
                raise ValueError(
                    "dataset_variants entries must use `overrides`, not the removed `compose_overrides` key."
                )
            dataset_name = str(item.get("dataset") or "").strip()
            label = str(item.get("label") or dataset_name).strip()
            compose_overrides = _normalize_override_list(
                item.get("overrides"),
                field_name="dataset_variants.overrides",
            )
            dataset_overrides = _normalize_override_mapping(
                item.get("dataset_overrides"),
                field_name="dataset_variants.dataset_overrides",
            )
            run_overrides = _normalize_override_mapping(
                item.get("run_overrides"),
                field_name="dataset_variants.run_overrides",
            )
        else:
            dataset_name = str(item).strip()
            label = dataset_name
            compose_overrides = []
            dataset_overrides = OmegaConf.create({})
            run_overrides = OmegaConf.create({})
        if not dataset_name:
            raise ValueError("dataset_variants entries must define a dataset name.")
        dataset_cfg = load_dataset_config_by_name(
            dataset_name,
            cfg.paths,
            overrides=compose_overrides,
        )
        if dataset_overrides:
            dataset_cfg = OmegaConf.merge(dataset_cfg, dataset_overrides)
            OmegaConf.resolve(dataset_cfg)
        variants.append(
            DatasetVariantSpec(
                label=label,
                dataset_name=dataset_name,
                dataset_cfg=dataset_cfg,
                run_overrides=OmegaConf.to_container(
                    run_overrides,
                    resolve=True,
                )
                if run_overrides
                else {},
            )
        )
    return variants


@contextmanager
def temporary_cfg_overrides(
    cfg: DictConfig,
    *,
    dataset_cfg: DictConfig | dict[str, Any] | None = None,
    run_overrides: dict[str, Any] | DictConfig | None = None,
    paths_overrides: dict[str, Any] | DictConfig | None = None,
):
    original_dataset = _clone_cfg_node(cfg.get("dataset"))
    original_run = _clone_cfg_node(cfg.get("run"))
    original_paths = _clone_cfg_node(cfg.get("paths"))
    with open_dict(cfg):
        if dataset_cfg is not None:
            cfg.dataset = _clone_cfg_node(dataset_cfg)
        if run_overrides:
            cfg.run = OmegaConf.merge(cfg.run, OmegaConf.create(run_overrides))
        if paths_overrides:
            cfg.paths = OmegaConf.merge(cfg.paths, OmegaConf.create(paths_overrides))
    try:
        yield cfg
    finally:
        with open_dict(cfg):
            cfg.dataset = original_dataset
            cfg.run = original_run
            cfg.paths = original_paths


def collect_model_metrics(
    *, callback_metrics: dict[str, Any], model: Any
) -> dict[str, Any]:
    if callback_metrics:
        return callback_metrics
    if hasattr(model, "get_predict_metrics"):
        metrics = model.get_predict_metrics()
        if isinstance(metrics, dict):
            return metrics
    metrics_from_model = getattr(model, "predict_metrics", None)
    return (
        metrics_from_model if isinstance(metrics_from_model, dict) else callback_metrics
    )


def save_metrics_snapshot(
    *,
    output_dir: str | Path,
    metrics: dict[str, Any],
    filename: str = "metrics.json",
) -> Path:
    return write_metrics_snapshot(
        metrics=metrics,
        settings=MetricsSnapshotSettings(output_dir=output_dir, filename=filename),
    )


__all__ = [
    "compose_config",
    "collect_model_metrics",
    "DatasetVariantSpec",
    "load_dataset_config_by_name",
    "normalize_dataset_scope",
    "resolve_dataset_variants",
    "resolve_execution_mode",
    "resolve_splits",
    "save_metrics_snapshot",
    "temporary_cfg_overrides",
]
