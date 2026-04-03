from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from src.runs.hydra import compose_config


_DATASET_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "dataset"


@dataclass(frozen=True)
class DatasetVariantSpec:
    label: str
    dataset_cfg: DictConfig
    run_overrides: dict[str, Any] = field(default_factory=dict)
    set_dataset_variant: bool = True


def _normalize_override_list(raw: Any, *, field_name: str) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        return [raw]
    if OmegaConf.is_list(raw) or isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in list(raw) if str(item).strip()]
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
    cfg = compose_config(
        config_name="eval.yaml",
        overrides=[
            f"dataset={name}",
            f"paths.data_dir={data_dir}",
            *(str(item) for item in (overrides or [])),
        ],
    )
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    return OmegaConf.create(dataset_cfg)


def resolve_dataset_variants(cfg: DictConfig) -> list[DatasetVariantSpec]:
    run_cfg = cfg.get("run") or {}
    raw_variants = run_cfg.get("dataset_variants")
    if not raw_variants:
        return []
    items = (
        list(raw_variants)
        if OmegaConf.is_list(raw_variants) or isinstance(raw_variants, (list, tuple))
        else [raw_variants]
    )

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
                dataset_cfg=dataset_cfg,
                run_overrides=(
                    OmegaConf.to_container(run_overrides, resolve=True)
                    if run_overrides
                    else {}
                ),
            )
        )
    return variants


__all__ = [
    "DatasetVariantSpec",
    "load_dataset_config_by_name",
    "resolve_dataset_variants",
]
