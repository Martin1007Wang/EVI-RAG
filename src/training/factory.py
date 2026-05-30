from __future__ import annotations

from typing import Any, TypeVar, cast

import hydra
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import (
    ModelResources,
    RetrievalDataConfig,
    build_retrieval_data_config,
    load_model_resources,
    validate_retrieval_data_config,
)
from src.weaver.feature import FeatureEncoder
from src.weaver.policy import ForwardPolicy

T = TypeVar("T")


def build_datamodule(
    cfg: DictConfig,
    data_config: RetrievalDataConfig,
) -> LightningDataModule:
    datamodule_cfg = OmegaConf.to_container(
        cfg.datamodule,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(datamodule_cfg, dict):
        raise TypeError(f"cfg.datamodule must resolve to dict, got {type(datamodule_cfg).__name__}.")

    target = datamodule_cfg.get("_target_")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("cfg.datamodule._target_ must be a non-empty string.")

    datamodule_cfg = {
        "_target_": target,
        "_convert_": "object",
        "data_config": data_config,
    }

    datamodule = hydra.utils.instantiate(datamodule_cfg)
    return require_type(datamodule, LightningDataModule, "cfg.datamodule")


def prepare_training_components(
    cfg: DictConfig,
    *,
    stage: str | tuple[str, ...],
) -> tuple[LightningDataModule, ModelResources]:
    data_config = build_retrieval_data_config(cfg)
    stages = _normalized_stages(stage)
    validate_retrieval_data_config(
        data_config,
        splits=_split_names_for_stages(data_config, stages),
    )
    datamodule = build_datamodule(cfg, data_config)
    datamodule.prepare_data()
    datamodule.setup(stages[0])
    resources = load_model_resources(data_config.materialization)
    return datamodule, resources


def build_model(
    cfg: DictConfig,
    resources: Any,
) -> LightningModule:
    feature_encoder_obj = hydra.utils.instantiate(
        cfg.model.feature_encoder,
        entity_text_semantic_table=resources.entity_text_semantic_table,
        text_row_by_entity_id=resources.text_row_by_entity_id,
        relation_semantic_table=resources.relation_semantic_table,
    )
    feature_encoder = require_type(
        feature_encoder_obj,
        FeatureEncoder,
        "cfg.model.feature_encoder",
    )

    policy = _build_policy(cfg.model.policy)

    model_obj = hydra.utils.instantiate(
        cfg.model,
        feature_encoder=feature_encoder,
        policy=policy,
    )
    module = require_type(model_obj, LightningModule, "cfg.model")

    debug_lookup = getattr(resources, "debug_lookup", None)
    if debug_lookup is not None:
        module.debug_lookup = debug_lookup

    return module


def _split_names_for_stages(
    data_config: RetrievalDataConfig,
    stages: tuple[str, ...],
) -> tuple[str, ...]:
    split_plan = data_config.splits
    splits: list[str] = []

    for stage in stages:
        if stage == "fit":
            splits.extend((split_plan.train, split_plan.validation))
            continue
        if stage == "validate":
            splits.append(split_plan.validation)
            continue
        if stage == "test":
            splits.append(split_plan.test)
            continue
        raise ValueError(f"Unsupported stage: {stage!r}.")

    return tuple(dict.fromkeys(splits))


def _normalized_stages(stage: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(stage, str):
        stages = (stage,)
    else:
        stages = tuple(stage)

    if not stages:
        raise ValueError("stage must contain at least one entry.")
    return stages


def build_trainer(cfg: DictConfig) -> Trainer:
    profiler = (
        hydra.utils.instantiate(cfg.profiler)
        if OmegaConf.is_missing(cfg, "profiler") is False and cfg.profiler is not None and "_target_" in cfg.profiler
        else None
    )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=instantiate_list(cfg.callbacks, "cfg.callbacks"),
        logger=trainer_logger(cfg.logger),
        profiler=profiler,
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

        return [hydra.utils.instantiate(item) for item in config.values() if item is not None]

    raise TypeError(f"{name} must be a ListConfig, DictConfig, or null, got {type(config).__name__}.")


def trainer_logger(config: ListConfig | DictConfig | None) -> bool | Any | list[Any]:
    loggers = instantiate_list(config, "cfg.logger")

    if len(loggers) == 0:
        return False

    if len(loggers) == 1:
        return loggers[0]

    return loggers


def model_resource_kwargs(resources: Any) -> dict[str, Any]:
    return {
        "entity_text_semantic_table": require_attr(
            resources,
            "entity_text_semantic_table",
            torch.Tensor,
        ),
        "text_row_by_entity_id": require_attr(
            resources,
            "text_row_by_entity_id",
            torch.Tensor,
        ),
        "relation_semantic_table": require_attr(
            resources,
            "relation_semantic_table",
            torch.Tensor,
        ),
    }


def require_attr(
    obj: Any,
    name: str,
    expected_type: type[T],
) -> T:
    if not hasattr(obj, name):
        raise AttributeError(f"resources.{name} is required.")

    value = getattr(obj, name)

    if not isinstance(value, expected_type):
        raise TypeError(f"resources.{name} must be {expected_type.__name__}, " f"got {type(value).__name__}.")

    return cast(T, value)


def require_type(
    value: Any,
    expected_type: type[T],
    name: str,
) -> T:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must instantiate {expected_type.__name__}, " f"got {type(value).__name__}.")

    return cast(T, value)


def _build_policy(policy_cfg: DictConfig) -> ForwardPolicy:
    interaction = hydra.utils.instantiate(policy_cfg.stop_head.interaction)
    stop_head = hydra.utils.instantiate(
        policy_cfg.stop_head,
        interaction=interaction,
    )
    edge_head = hydra.utils.instantiate(
        policy_cfg.edge_head,
        interaction=interaction,
    )
    cache_builder = hydra.utils.instantiate(policy_cfg.cache_builder)
    state_encoder = hydra.utils.instantiate(policy_cfg.state_encoder)
    policy = ForwardPolicy(
        cache_builder=cache_builder,
        state_encoder=state_encoder,
        stop_head=stop_head,
        edge_head=edge_head,
    )
    return require_type(policy, ForwardPolicy, "cfg.model.policy")


__all__ = [
    "build_datamodule",
    "build_model",
    "build_trainer",
    "instantiate_list",
    "model_resource_kwargs",
    "prepare_training_components",
    "require_attr",
    "require_type",
    "trainer_logger",
]
