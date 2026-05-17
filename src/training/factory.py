from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeVar, cast

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import build_training_data_config

T = TypeVar("T")


def setup_datamodule(
    datamodule: LightningDataModule,
    stage: str = "fit",
) -> Any:
    datamodule.prepare_data()
    datamodule.setup(stage)

    if not hasattr(datamodule, "model_resources"):
        raise AttributeError(
            f"Datamodule must expose `model_resources` after setup({stage!r})."
        )
    resources = getattr(datamodule, "model_resources", None)
    return resources


def build_datamodule(cfg: DictConfig) -> LightningDataModule:
    datamodule_cfg = OmegaConf.to_container(
        cfg.datamodule,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(datamodule_cfg, dict):
        raise TypeError(
            f"cfg.datamodule must resolve to dict, got {type(datamodule_cfg).__name__}."
        )

    target = datamodule_cfg.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("cfg.datamodule._target_ must be a non-empty string.")

    datamodule_cfg = {
        "_target_": target,
        "_convert_": "object",
        "data_config": build_training_data_config(cfg),
    }

    datamodule = hydra.utils.instantiate(datamodule_cfg)
    return require_type(datamodule, LightningDataModule, "cfg.datamodule")


def build_model(
    cfg: DictConfig,
    resources: Any,
) -> LightningModule:
    model_cfg = OmegaConf.to_container(
        cfg.model,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(model_cfg, dict):
        raise TypeError(
            f"cfg.model must resolve to dict, got {type(model_cfg).__name__}."
        )

    model_cfg = deepcopy(model_cfg)
    _inject_model_resources(model_cfg, resources)

    model = hydra.utils.instantiate(model_cfg)
    return require_type(model, LightningModule, "cfg.model")


def build_trainer(cfg: DictConfig) -> Trainer:
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=instantiate_list(cfg.callbacks, "cfg.callbacks"),
        logger=trainer_logger(cfg.logger),
    )

    return require_type(trainer, Trainer, "cfg.trainer")


def instantiate_list(
    config: ListConfig | DictConfig | None,
    name: str,
) -> list[Any]:
    if config is None:
        return []

    if isinstance(config, ListConfig):
        return [hydra.utils.instantiate(item) for item in config if item is not None]

    if isinstance(config, DictConfig):
        if "_target_" in config:
            return [hydra.utils.instantiate(config)]

        return [
            hydra.utils.instantiate(item)
            for item in config.values()
            if item is not None
        ]

    raise TypeError(
        f"{name} must be a ListConfig, DictConfig, or null, got {type(config).__name__}."
    )


def trainer_logger(config: ListConfig | DictConfig | None) -> bool | Any | list[Any]:
    loggers = instantiate_list(config, "cfg.logger")

    if len(loggers) == 0:
        return False

    if len(loggers) == 1:
        return loggers[0]

    return loggers


def model_resource_kwargs(resources: Any) -> dict[str, torch.Tensor]:
    return {
        "entity_text_embeddings": require_attr(
            resources,
            "entity_text_embeddings",
            torch.Tensor,
        ),
        "entity_embedding_map": require_attr(
            resources,
            "entity_embedding_map",
            torch.Tensor,
        ),
        "relation_embeddings": require_attr(
            resources,
            "relation_embeddings",
            torch.Tensor,
        ),
    }


def _inject_model_resources(
    model_cfg: dict[str, Any],
    resources: Any,
) -> None:
    model_cfg.pop("hidden_dim", None)

    feature_encoder_cfg = _require_mapping(
        model_cfg.get("feature_encoder"),
        "cfg.model.feature_encoder",
    )

    resource_kwargs = model_resource_kwargs(resources)
    feature_encoder_cfg.update(resource_kwargs)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must resolve to dict, got {type(value).__name__}.")
    return value


def require_attr(
    obj: Any,
    name: str,
    expected_type: type[T],
) -> T:
    if not hasattr(obj, name):
        raise AttributeError(f"resources.{name} is required.")

    value = getattr(obj, name)

    if not isinstance(value, expected_type):
        raise TypeError(
            f"resources.{name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )

    return cast(T, value)


def require_type(
    value: Any,
    expected_type: type[T],
    name: str,
) -> T:
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{name} must instantiate {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )

    return cast(T, value)


__all__ = [
    "build_datamodule",
    "build_model",
    "build_trainer",
    "instantiate_list",
    "model_resource_kwargs",
    "require_attr",
    "require_type",
    "setup_datamodule",
    "trainer_logger",
]
