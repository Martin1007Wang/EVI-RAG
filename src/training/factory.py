from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig


def require_cfg(cfg: DictConfig, key: str) -> Any:
    value = cfg.get(key, None)
    if value is None:
        raise KeyError(f"Missing required config section: {key!r}.")
    return value


def instantiate_many(config: Any | None) -> list[Any]:
    """
    Instantiate Hydra object collections.

    Supported:
    - None
    - single config with _target_
    - list/tuple of configs
    - mapping name -> config

    Null entries are skipped.
    """
    if config is None:
        return []

    if isinstance(config, (list, tuple)):
        return [hydra.utils.instantiate(item) for item in config if item is not None]

    if isinstance(config, Mapping):
        if "_target_" in config:
            return [hydra.utils.instantiate(config)]

        objects: list[Any] = []
        for item in config.values():
            if item is not None:
                objects.append(hydra.utils.instantiate(item))
        return objects

    raise TypeError(
        "Expected None, a Hydra target config, a sequence, or a mapping; "
        f"got {type(config)!r}."
    )


def build_datamodule(cfg: DictConfig) -> LightningDataModule:
    datamodule = hydra.utils.instantiate(require_cfg(cfg, "datamodule"))

    if not isinstance(datamodule, LightningDataModule):
        raise TypeError(
            "cfg.datamodule must instantiate LightningDataModule, "
            f"got {type(datamodule)!r}."
        )

    return datamodule


def build_model(cfg: DictConfig, resources: Any) -> LightningModule:
    model = hydra.utils.instantiate(
        require_cfg(cfg, "model"),
        entity_text_embeddings=resources.entity_text_embeddings,
        entity_embedding_map=resources.entity_embedding_map,
        relation_embeddings=resources.relation_embeddings,
    )

    if not isinstance(model, LightningModule):
        raise TypeError(
            "cfg.model must instantiate LightningModule, " f"got {type(model)!r}."
        )

    return model


def build_trainer(cfg: DictConfig) -> Trainer:
    callbacks = instantiate_many(cfg.get("callbacks", None))
    loggers = instantiate_many(cfg.get("logger", None))

    if len(loggers) == 0:
        logger_arg: bool | Any | list[Any] = False
    elif len(loggers) == 1:
        logger_arg = loggers[0]
    else:
        logger_arg = loggers

    trainer = hydra.utils.instantiate(
        require_cfg(cfg, "trainer"),
        callbacks=callbacks,
        logger=logger_arg,
    )

    if not isinstance(trainer, Trainer):
        raise TypeError(
            "cfg.trainer must instantiate lightning.Trainer, " f"got {type(trainer)!r}."
        )

    return trainer
