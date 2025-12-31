from __future__ import annotations

from typing import Optional

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict


def resolve_run_name(cfg: DictConfig) -> str:
    run_name = cfg.get("task_name", "train")
    try:
        hydra_cfg = HydraConfig.get()
        experiment_choice: Optional[str] = None
        dataset_choice: Optional[str] = None
        if hydra_cfg is not None:
            runtime = getattr(hydra_cfg, "runtime", None)
            if runtime is not None:
                experiment_choice = runtime.choices.get("experiment")  # type: ignore[attr-defined]
                dataset_choice = runtime.choices.get("dataset")  # type: ignore[attr-defined]
        if experiment_choice:
            run_name = experiment_choice
        dataset_cfg = cfg.get("dataset") if hasattr(cfg, "get") else None
        dataset_cfg_name = ""
        if dataset_cfg is not None and hasattr(dataset_cfg, "get"):
            dataset_cfg_name = str(dataset_cfg.get("name", "") or "")
        dataset_name = str(dataset_choice or dataset_cfg_name or "").strip()
        if dataset_name:
            run_name = f"{run_name}_{dataset_name}"
    except Exception:
        pass
    return run_name.replace("evidential", "evi")


def apply_run_name(cfg: DictConfig) -> str:
    run_name = resolve_run_name(cfg)
    with open_dict(cfg):
        cfg.run_name = run_name
    logger_cfg = cfg.get("logger")
    if isinstance(logger_cfg, DictConfig):
        for _, lg_conf in logger_cfg.items():
            target = str(lg_conf.get("_target_", "")).lower()
            if "wandb" in target:
                with open_dict(lg_conf):
                    lg_conf["name"] = run_name
    return run_name
