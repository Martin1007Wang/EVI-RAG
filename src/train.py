from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch

from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# --------------------------------------------------------------------------
# GFlowNet 专用 debug 日志：独立文件，避免和 tqdm/CLI 混杂。
# --------------------------------------------------------------------------
import logging
from pathlib import Path


def _setup_gflownet_debug_logging(cfg) -> None:
    enable = cfg.get("debug", False)
    if not enable:
        return

    log_paths = []
    out_dir = Path(cfg.paths.output_dir)
    run_log_dir = out_dir / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    log_paths.append(run_log_dir / "gflownet_debug.log")

    debug_log_path = getattr(cfg.paths, "debug_log_path", None)
    if debug_log_path:
        debug_log = Path(debug_log_path).expanduser()
        debug_log.parent.mkdir(parents=True, exist_ok=True)
        log_paths.append(debug_log)

    base_logger = logging.getLogger("gflownet.debug")
    base_logger.setLevel(logging.INFO)

    # 防止重复添加 handler
    for path in log_paths:
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == Path(path) for h in base_logger.handlers):
            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            base_logger.addHandler(handler)
    base_logger.propagate = False


from src.utils.run_context import apply_run_name

log = RankedLogger(__name__, rank_zero_only=True)


def _run_data_loading_preflight(cfg: DictConfig) -> None:
    """Run a minimal fast_dev_run to surface dataloader/worker crashes early."""
    log.info("Running data-loading preflight with fast_dev_run=1 batch...")
    preflight_datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    preflight_model: LightningModule = hydra.utils.instantiate(cfg.model)
    preflight_trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[],
        logger=False,
        enable_checkpointing=False,
        fast_dev_run=True,
    )
    preflight_trainer.fit(model=preflight_model, datamodule=preflight_datamodule, ckpt_path=None)
    log.info("Preflight passed; proceeding to full training run.")


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # GFlowNet 深度调试日志重定向（独立文件）。
    _setup_gflownet_debug_logging(cfg)

    if cfg.get("debug_data_loading"):
        _run_data_loading_preflight(cfg)

    resolved_run_name = apply_run_name(cfg)
    log.info(f"Resolved run name: {resolved_run_name}")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        test_ckpt_path: Optional[str] = cfg.get("test_ckpt_path")
        if test_ckpt_path not in (None, ""):
            ckpt_path = test_ckpt_path
        else:
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback is None:
                raise RuntimeError(
                    "Testing requested but no checkpoint callback is configured. "
                    "Provide `test_ckpt_path` or enable a checkpoint callback."
                )
            ckpt_path = checkpoint_callback.best_model_path
            if ckpt_path == "":
                if cfg.get("allow_test_without_checkpoint", False):
                    log.warning("Best ckpt not found! Using current weights for testing...")
                    ckpt_path = None
                else:
                    raise RuntimeError(
                        "Best checkpoint path is empty. Set `allow_test_without_checkpoint=True` "
                        "or provide `test_ckpt_path` to proceed explicitly."
                    )
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
