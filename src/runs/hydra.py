from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.logging_utils import RankedLogger

if TYPE_CHECKING:  # pragma: no cover
    from lightning import Callback
    from lightning.pytorch.loggers import Logger


log = RankedLogger(__name__, rank_zero_only=True)


def _instantiate_config_group(
    group_cfg: DictConfig | None,
    *,
    group_name: str,
    item_name: str,
) -> list[Any]:
    if not group_cfg:
        log.warning("No %s configs found! Skipping...", group_name)
        return []
    if not isinstance(group_cfg, DictConfig):
        raise TypeError(f"{item_name.capitalize()} config must be a DictConfig!")

    instances: list[Any] = []
    for item_cfg in group_cfg.values():
        if isinstance(item_cfg, DictConfig) and "_target_" in item_cfg:
            log.info("Instantiating %s <%s>", item_name, item_cfg._target_)
            instances.append(hydra.utils.instantiate(item_cfg))
    return instances


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list["Callback"]:
    return cast(
        list["Callback"],
        _instantiate_config_group(
            callbacks_cfg,
            group_name="callback",
            item_name="callback",
        ),
    )


def instantiate_loggers(logger_cfg: DictConfig | None) -> list["Logger"]:
    return cast(
        list["Logger"],
        _instantiate_config_group(
            logger_cfg,
            group_name="logger",
            item_name="logger",
        ),
    )


def _require_rich(entrypoint: str):
    try:
        import rich
        import rich.syntax
        import rich.tree
        from rich.prompt import Prompt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"rich is required to call {entrypoint}. Install rich or disable Rich-dependent features."
        ) from exc
    return rich, Prompt


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    rich, _prompt = _require_rich("print_config_tree")

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue: list[str] = []
    for field in print_order:
        if field in cfg:
            queue.append(field)
            continue
        log.warning(
            "Field %r not found in config. Skipping %r config printing...",
            field,
            field,
        )
    for field in cfg:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)
    if save_to_file:
        with Path(cfg.paths.output_dir, "config_tree.log").open(
            "w", encoding="utf-8"
        ) as handle:
            rich.print(tree, file=handle)


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    rich, Prompt = _require_rich("enforce_tags")

    if not cfg.get("tags"):
        hydra_cfg = HydraConfig.get()
        if "id" in hydra_cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        resolved_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        with open_dict(cfg):
            cfg.tags = resolved_tags
        log.info("Tags: %s", cfg.tags)

    if save_to_file:
        with Path(cfg.paths.output_dir, "tags.log").open(
            "w", encoding="utf-8"
        ) as handle:
            rich.print(cfg.tags, file=handle)


def _configure_hf_cache(cfg: DictConfig) -> None:
    paths = cfg.get("paths")
    if paths is None:
        return
    hf_home = paths.get("hf_home")
    hf_datasets_cache = paths.get("hf_datasets_cache")
    if hf_home:
        os.environ["HF_HOME"] = str(hf_home)
    if hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)
    if "TRANSFORMERS_CACHE" in os.environ:
        os.environ.pop("TRANSFORMERS_CACHE", None)
        log.info(
            "Unset TRANSFORMERS_CACHE; using HF_HOME=%s", os.environ.get("HF_HOME")
        )


def extras(cfg: DictConfig) -> None:
    _configure_hf_cache(cfg)
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for extras torch settings."
        ) from exc

    precision = cfg.extras.get("torch_float32_matmul_precision")
    if precision:
        torch.set_float32_matmul_precision(str(precision))
        log.info("torch.set_float32_matmul_precision(%s)", precision)

    allow_tf32_matmul = cfg.extras.get("allow_tf32_matmul")
    if allow_tf32_matmul is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_matmul)
            log.info(
                "torch.backends.cuda.matmul.allow_tf32=%s",
                bool(allow_tf32_matmul),
            )
        except AttributeError:
            log.warning(
                "torch.backends.cuda.matmul.allow_tf32 not available in this build."
            )

    allow_tf32_cudnn = cfg.extras.get("allow_tf32_cudnn")
    if allow_tf32_cudnn is not None:
        try:
            torch.backends.cudnn.allow_tf32 = bool(allow_tf32_cudnn)
            log.info("torch.backends.cudnn.allow_tf32=%s", bool(allow_tf32_cudnn))
        except AttributeError:
            log.warning("torch.backends.cudnn.allow_tf32 not available in this build.")

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def resolve_run_name(cfg: DictConfig) -> str:
    run_name = cfg.get("task_name", "train")
    try:
        hydra_cfg = HydraConfig.get()
        experiment_choice: Optional[str] = None
        dataset_choice: Optional[str] = None
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
        for logger_item_cfg in logger_cfg.values():
            target = str(logger_item_cfg.get("_target_", "")).lower()
            if "wandb" in target:
                with open_dict(logger_item_cfg):
                    logger_item_cfg["name"] = run_name
    return run_name


def compose_config(
    *,
    config_name: str,
    overrides: Sequence[str] | None = None,
) -> DictConfig:
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    compose_overrides = [str(item) for item in (overrides or [])]

    if GlobalHydra.instance().is_initialized():
        cfg = hydra.compose(config_name=config_name, overrides=compose_overrides)
    else:
        with hydra.initialize_config_dir(
            version_base="1.3", config_dir=str(config_dir)
        ):
            cfg = hydra.compose(config_name=config_name, overrides=compose_overrides)

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Hydra compose returned {type(cfg)!r}, expected DictConfig for {config_name!r}."
        )
    return cfg


__all__ = [
    "apply_run_name",
    "compose_config",
    "enforce_tags",
    "extras",
    "instantiate_callbacks",
    "instantiate_loggers",
    "print_config_tree",
    "resolve_run_name",
]
