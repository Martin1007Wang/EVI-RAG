from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils

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

from src.utils.run_context import apply_run_name

log = RankedLogger(__name__, rank_zero_only=True)

def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return True
    return False


def _validate_gflownet_required_args(cfg: DictConfig) -> None:
    model_cfg = cfg.get("model")
    data_cfg = cfg.get("data")
    model_target = model_cfg.get("_target_") if model_cfg else ""
    data_target = data_cfg.get("_target_") if data_cfg else ""
    is_gflownet = (
        model_target == "src.models.gflownet_module.GFlowNetModule"
        or data_target == "src.data.g_agent_datamodule.GAgentDataModule"
    )
    if not is_gflownet:
        return

    missing = []
    if cfg.get("dataset") is None:
        missing.append("dataset")

    embedder_cfg = model_cfg.get("embedder_cfg") if model_cfg else None
    allow_deferred = bool(embedder_cfg.get("allow_deferred_init", False)) if embedder_cfg else False
    ckpt_cfg = cfg.get("ckpt") or {}
    retriever_ckpt = ckpt_cfg.get("retriever") if hasattr(ckpt_cfg, "get") else None
    if not allow_deferred and _is_missing_value(retriever_ckpt):
        missing.append("ckpt.retriever")

    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Missing required GFlowNet inputs: "
            f"{missing_str}. Please specify both `dataset=<name>` and "
            "`ckpt.retriever=/path/to/retriever.ckpt` for GFlowNet training. "
            "Example: python src/train.py experiment=train_gflownet "
            "dataset=webqsp ckpt.retriever=/path/to/epoch_003.ckpt"
        )


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
    _validate_gflownet_required_args(cfg)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    return get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
