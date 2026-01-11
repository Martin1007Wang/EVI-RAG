from __future__ import annotations

import os
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

try:  # pragma: no cover - optional dependency guard
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None

try:  # pragma: no cover - optional dependency guard
    from hydra.core.hydra_config import HydraConfig
except Exception:  # pragma: no cover
    HydraConfig = None

from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

try:  # pragma: no cover - optional dependency guard
    import rich
    import rich.syntax
    import rich.tree
    from rich.prompt import Prompt
except ModuleNotFoundError:  # pragma: no cover
    rich = None
    Prompt = None

from .logging_utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    _ensure_hydra_available("instantiate_callbacks")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    _ensure_hydra_available("instantiate_loggers")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


def _ensure_hydra_available(entrypoint: str) -> None:
    if hydra is None:
        raise ModuleNotFoundError(
            f"hydra is required to call {entrypoint}. Install hydra-core or disable Hydra-dependent features."
        )


def _ensure_rich_available(entrypoint: str) -> None:
    if rich is None or Prompt is None:
        raise ModuleNotFoundError(
            f"rich is required to call {entrypoint}. Install rich or disable Rich-dependent features."
        )


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
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    _ensure_rich_available("print_config_tree")

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


def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    _ensure_rich_available("enforce_tags")
    if HydraConfig is None:
        raise ModuleNotFoundError("HydraConfig is required for enforce_tags; install hydra-core.")
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


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
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

    # allow opt-in/opt-out Tensor Core friendly matmul precision for FP32
    precision = cfg.extras.get("torch_float32_matmul_precision")
    if precision:
        torch.set_float32_matmul_precision(str(precision))
        log.info("torch.set_float32_matmul_precision(%s)", precision)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


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
        log.info("Unset TRANSFORMERS_CACHE; using HF_HOME=%s", os.environ.get("HF_HOME"))


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.

    :param metric_dict: A dict containing metric values.
    :param metric_name: If provided, the name of the metric to retrieve.
    :return: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


def resolve_run_name(cfg: DictConfig) -> str:
    run_name = cfg.get("task_name", "train")
    if HydraConfig is None:
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


__all__ = [
    "instantiate_callbacks",
    "instantiate_loggers",
    "print_config_tree",
    "enforce_tags",
    "extras",
    "task_wrapper",
    "get_metric_value",
    "resolve_run_name",
    "apply_run_name",
]
