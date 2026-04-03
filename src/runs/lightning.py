from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Callable

import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.runs.hydra import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters


@dataclass(frozen=True)
class LightningTaskObjects:
    cfg: DictConfig
    datamodule: Any
    model: Any
    callbacks: list[Any]
    logger: list[Any]
    trainer: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg,
            "datamodule": self.datamodule,
            "model": self.model,
            "callbacks": self.callbacks,
            "logger": self.logger,
            "trainer": self.trainer,
        }


def resolve_instantiate_config(cfg_node: DictConfig) -> DictConfig:
    if not isinstance(cfg_node, DictConfig):
        return cfg_node
    container = OmegaConf.to_container(cfg_node, resolve=True)
    if not isinstance(container, dict):
        return cfg_node
    return OmegaConf.create(container)


def _normalize_trainer_cfg(cfg_node: DictConfig) -> DictConfig:
    trainer_cfg = resolve_instantiate_config(cfg_node)
    container = OmegaConf.to_container(trainer_cfg, resolve=True)
    if not isinstance(container, dict):
        return trainer_cfg
    if container.get("max_steps") is None:
        container["max_steps"] = -1
    return OmegaConf.create(container)


def _filter_trainer_callbacks(
    *, callbacks: list[Any], trainer_cfg: DictConfig
) -> list[Any]:
    if bool(trainer_cfg.get("enable_checkpointing", True)):
        return callbacks
    return [
        callback for callback in callbacks if not isinstance(callback, ModelCheckpoint)
    ]


def instantiate_lightning_task_objects(
    cfg: DictConfig,
    *,
    log: Any,
    on_datamodule_instantiated: Callable[[Any], None] | None = None,
    on_model_instantiated: Callable[[Any], None] | None = None,
) -> LightningTaskObjects:
    data_cfg = resolve_instantiate_config(cfg.data)
    log.info("Instantiating datamodule <%s>", data_cfg._target_)
    datamodule = hydra.utils.instantiate(data_cfg)
    if on_datamodule_instantiated is not None:
        on_datamodule_instantiated(datamodule)

    model_cfg = resolve_instantiate_config(cfg.model)
    log.info("Instantiating model <%s>", model_cfg._target_)
    model = hydra.utils.instantiate(model_cfg)
    if on_model_instantiated is not None:
        on_model_instantiated(model)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    trainer_cfg = _normalize_trainer_cfg(cfg.trainer)
    callbacks = _filter_trainer_callbacks(callbacks=callbacks, trainer_cfg=trainer_cfg)
    log.info("Instantiating trainer <%s>", trainer_cfg._target_)
    trainer = hydra.utils.instantiate(trainer_cfg, callbacks=callbacks, logger=logger)

    objects = LightningTaskObjects(
        cfg=cfg,
        datamodule=datamodule,
        model=model,
        callbacks=callbacks,
        logger=logger,
        trainer=trainer,
    )
    if objects.logger:
        log.info("Logging hyperparameters...")
        log_hyperparameters(objects.as_dict())
    return objects


def seed_everything_if_needed(cfg: DictConfig) -> None:
    seed = cfg.get("seed")
    if seed is None:
        return
    L.seed_everything(int(seed), workers=True)


def finalize_task(*, cfg: DictConfig, log: Any) -> None:
    log.info("Output dir: %s", cfg.paths.output_dir)
    if not find_spec("wandb"):
        return
    import wandb

    if not wandb.run:
        return

    from lightning_utilities.core.rank_zero import rank_zero_only

    @rank_zero_only
    def _finish() -> None:
        log.info("Closing wandb!")
        wandb.finish()

    _finish()


def select_logged_metrics(trainer: Any, *, prefix: str | None = None) -> dict[str, Any]:
    callback_metrics = dict(getattr(trainer, "callback_metrics", {}) or {})
    if not prefix:
        return callback_metrics
    return {
        key: value
        for key, value in callback_metrics.items()
        if str(key).startswith(prefix)
    }


__all__ = [
    "finalize_task",
    "LightningTaskObjects",
    "instantiate_lightning_task_objects",
    "resolve_instantiate_config",
    "seed_everything_if_needed",
    "select_logged_metrics",
]
