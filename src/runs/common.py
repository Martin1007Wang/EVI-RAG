from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict

_MISSING_MODE_VALUES = {None, "", "null", "None"}


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


__all__ = [
    "normalize_dataset_scope",
    "resolve_execution_mode",
    "resolve_splits",
    "temporary_cfg_overrides",
]
