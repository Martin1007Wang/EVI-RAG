from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import hydra
from omegaconf import DictConfig, OmegaConf

from .hydra_utils import instantiate_callbacks, instantiate_loggers
from .logging_utils import log_hyperparameters


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


def _strip_instantiate_metadata(cfg_node: DictConfig) -> DictConfig:
    if not isinstance(cfg_node, DictConfig):
        return cfg_node
    # Keep root-level interpolations like `${dataset}` and `${run.split}` intact by
    # resolving while the node is still attached to the composed config tree.
    container = OmegaConf.to_container(cfg_node, resolve=True)
    if not isinstance(container, dict):
        return cfg_node
    container.pop("contract", None)
    return OmegaConf.create(container)


def require_run_target_config(
    cfg: DictConfig,
    *,
    missing_run_message: str,
    missing_target_message: str,
) -> DictConfig:
    run_cfg = cfg.get("run")
    if run_cfg is None:
        raise ValueError(missing_run_message)
    run_target = str(run_cfg.get("_target_", "") or "").strip()
    if run_target in {"", "null", "None"}:
        raise ValueError(missing_target_message)
    return run_cfg


def instantiate_task_runner(
    run_cfg: DictConfig,
    *,
    run_signature: str,
) -> Any:
    runner = hydra.utils.instantiate(run_cfg, _recursive_=False)
    if not callable(getattr(runner, "validate", None)) or not callable(
        getattr(runner, "run", None)
    ):
        raise TypeError(
            "run._target_ must instantiate an object exposing `validate(cfg)` and "
            f"`{run_signature}`."
        )
    return runner


def instantiate_lightning_task_objects(
    cfg: DictConfig,
    *,
    log: Any,
    on_datamodule_instantiated: Callable[[Any], None] | None = None,
    on_model_instantiated: Callable[[Any], None] | None = None,
) -> LightningTaskObjects:
    data_cfg = _strip_instantiate_metadata(cfg.data)

    log.info(f"Instantiating datamodule <{data_cfg._target_}>")
    datamodule = hydra.utils.instantiate(data_cfg)
    if on_datamodule_instantiated is not None:
        on_datamodule_instantiated(datamodule)

    model_cfg = _strip_instantiate_metadata(cfg.model)
    log.info(f"Instantiating model <{model_cfg._target_}>")
    model = hydra.utils.instantiate(model_cfg)
    if on_model_instantiated is not None:
        on_model_instantiated(model)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

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


__all__ = [
    "LightningTaskObjects",
    "instantiate_lightning_task_objects",
    "instantiate_task_runner",
    "require_run_target_config",
]
