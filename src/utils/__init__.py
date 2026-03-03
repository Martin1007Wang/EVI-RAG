"""Public utils API (runtime helpers)."""

from __future__ import annotations

import os
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, TYPE_CHECKING

from .logging_utils import RankedLogger, log_hyperparameters, log_metric

if TYPE_CHECKING:  # pragma: no cover
    from lightning import Callback
    from lightning.pytorch.loggers import Logger
    from omegaconf import DictConfig

log = RankedLogger(__name__, rank_zero_only=True)


def _require_hydra(entrypoint: str):
    try:
        import hydra  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"hydra is required to call {entrypoint}. Install hydra-core or disable Hydra-dependent features."
        ) from exc
    return hydra


def _require_hydra_config(entrypoint: str):
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"HydraConfig is required for {entrypoint}; install hydra-core."
        ) from exc
    return HydraConfig


def _require_rich(entrypoint: str):
    try:
        import rich  # type: ignore
        import rich.syntax  # type: ignore
        import rich.tree  # type: ignore
        from rich.prompt import Prompt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"rich is required to call {entrypoint}. Install rich or disable Rich-dependent features."
        ) from exc
    return rich, Prompt


def _require_omegaconf(entrypoint: str):
    try:
        from omegaconf import DictConfig, OmegaConf, open_dict  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            f"omegaconf is required to call {entrypoint}. Install omegaconf or disable Hydra-dependent features."
        ) from exc
    return DictConfig, OmegaConf, open_dict


def instantiate_callbacks(callbacks_cfg: "DictConfig") -> list["Callback"]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    DictConfig, _OmegaConf, _open_dict = _require_omegaconf("instantiate_callbacks")
    hydra = _require_hydra("instantiate_callbacks")
    callbacks: list["Callback"] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: "DictConfig") -> list["Logger"]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    DictConfig, _OmegaConf, _open_dict = _require_omegaconf("instantiate_loggers")
    hydra = _require_hydra("instantiate_loggers")
    logger: list["Logger"] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def print_config_tree(
    cfg: "DictConfig",
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
    """Prints the contents of a DictConfig as a tree structure using the Rich library."""
    DictConfig, OmegaConf, _open_dict = _require_omegaconf("print_config_tree")
    rich, _Prompt = _require_rich("print_config_tree")

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def enforce_tags(cfg: "DictConfig", save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""
    DictConfig, _OmegaConf, open_dict = _require_omegaconf("enforce_tags")
    HydraConfig = _require_hydra_config("enforce_tags")
    rich, Prompt = _require_rich("enforce_tags")

    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


def extras(cfg: "DictConfig") -> None:
    """Applies optional utilities before the task is started."""
    _configure_hf_cache(cfg)

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "torch is required for extras torch settings."
        ) from exc

    # allow opt-in/opt-out Tensor Core friendly matmul precision for FP32
    precision = cfg.extras.get("torch_float32_matmul_precision")
    if precision:
        torch.set_float32_matmul_precision(str(precision))
        log.info("torch.set_float32_matmul_precision(%s)", precision)
    allow_tf32_matmul = cfg.extras.get("allow_tf32_matmul")
    if allow_tf32_matmul is not None:
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32_matmul)
            log.info(
                "torch.backends.cuda.matmul.allow_tf32=%s", bool(allow_tf32_matmul)
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

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def _configure_hf_cache(cfg: "DictConfig") -> None:
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


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function."""

    def wrap(cfg: "DictConfig") -> Tuple[dict[str, Any], dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(
    metric_dict: dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule."""
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].detach().tolist()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def resolve_run_name(cfg: "DictConfig") -> str:
    run_name = cfg.get("task_name", "train")
    try:
        HydraConfig = _require_hydra_config("resolve_run_name")
    except ModuleNotFoundError:
        return run_name
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


def apply_run_name(cfg: "DictConfig") -> str:
    DictConfig, _OmegaConf, open_dict = _require_omegaconf("apply_run_name")
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


__all__ = [
    "instantiate_callbacks",
    "instantiate_loggers",
    "log_hyperparameters",
    "log_metric",
    "RankedLogger",
    "enforce_tags",
    "print_config_tree",
    "extras",
    "get_metric_value",
    "task_wrapper",
    "resolve_run_name",
    "apply_run_name",
]
