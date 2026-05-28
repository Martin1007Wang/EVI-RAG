from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

SchedulerInterval = Literal["step", "epoch"]


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    no_decay_on_bias_and_norm: bool


@dataclass(frozen=True, slots=True)
class SchedulerSpec:
    type: str
    interval: SchedulerInterval
    warmup_ratio: float
    eta_min: float


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec | None


def configure_optimization(
    *,
    modules: Sequence[nn.Module],
    cfg: DictConfig | Mapping[str, Any],
    trainer: object,
    explicit_t_max: int | None = None,
) -> OptimizerLRScheduler:
    spec = parse_optimization_config(cfg)

    optimizer = build_optimizer(
        modules=modules,
        cfg=spec.optimizer,
    )

    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=spec.scheduler,
        trainer=trainer,
        base_lr=spec.optimizer.lr,
        explicit_t_max=explicit_t_max,
    )

    if scheduler is None:
        return optimizer

    if spec.scheduler is None:
        raise RuntimeError("build_scheduler returned a scheduler even though scheduler spec is None.")

    config: OptimizerLRSchedulerConfig = {
        "optimizer": optimizer,
        "lr_scheduler": build_lightning_scheduler_config(
            scheduler=scheduler,
            interval=spec.scheduler.interval,
        ),
    }
    return config


def parse_optimization_config(
    cfg: DictConfig | Mapping[str, Any],
) -> OptimizationSpec:
    node = to_plain_mapping(cfg, name="optimization")

    optimizer_node = require_mapping(
        node,
        key="optimizer",
        owner="optimization",
    )

    scheduler_node = node.get("scheduler", None)

    return OptimizationSpec(
        optimizer=parse_optimizer_spec(optimizer_node),
        scheduler=parse_scheduler_spec(scheduler_node),
    )


def parse_optimizer_spec(
    node: Mapping[str, Any],
) -> OptimizerSpec:
    optimizer_type = str(node.get("type", "adamw")).lower()
    if optimizer_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {optimizer_type!r}.")

    return OptimizerSpec(
        lr=positive_float(node, "lr", owner="optimizer"),
        betas=parse_betas(node.get("betas", (0.9, 0.999))),
        weight_decay=non_negative_float(
            node,
            "weight_decay",
            owner="optimizer",
            default=0.0,
        ),
        no_decay_on_bias_and_norm=bool(node.get("no_decay_on_bias_and_norm", True)),
    )


def parse_scheduler_spec(
    node: Any,
) -> SchedulerSpec | None:
    if node is None:
        return None

    if isinstance(node, DictConfig):
        node = to_plain_mapping(node, name="scheduler")

    if not isinstance(node, Mapping):
        raise TypeError("optimization.scheduler must be null or a mapping; " f"got {type(node)!r}.")

    scheduler_type = str(node.get("type", "none")).lower()
    if scheduler_type == "none":
        return None

    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type!r}.")

    return SchedulerSpec(
        type=scheduler_type,
        interval=parse_interval(node.get("interval", "step")),
        warmup_ratio=bounded_float(
            node,
            "warmup_ratio",
            owner="scheduler",
            lower=0.0,
            upper=1.0,
            include_upper=False,
        ),
        eta_min=non_negative_float(
            node,
            "eta_min",
            owner="scheduler",
            default=0.0,
        ),
    )


def build_optimizer(
    *,
    modules: Sequence[nn.Module],
    cfg: OptimizerSpec,
) -> torch.optim.Optimizer:
    groups = parameter_groups(
        modules=modules,
        weight_decay=cfg.weight_decay,
        no_decay_on_bias_and_norm=cfg.no_decay_on_bias_and_norm,
    )

    if not groups:
        raise RuntimeError("No trainable parameters found. " "Expected at least one trainable parameter in the provided modules.")

    return AdamW(
        groups,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerSpec | None,
    trainer: object,
    base_lr: float,
    explicit_t_max: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg is None:
        return None

    if cfg.type != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler type: {cfg.type!r}.")

    base_lr = float(base_lr)
    if base_lr <= 0.0:
        raise ValueError(f"base_lr must be positive, got {base_lr}.")

    if cfg.eta_min > base_lr:
        raise ValueError(f"eta_min must be <= base_lr for cosine decay; " f"got eta_min={cfg.eta_min}, base_lr={base_lr}.")

    horizon = resolve_scheduler_horizon(
        trainer=trainer,
        explicit_t_max=explicit_t_max,
        interval=cfg.interval,
    )
    if horizon <= 0:
        raise RuntimeError("Could not resolve a positive scheduler horizon. " "Set trainer.max_steps / trainer.max_epochs, or pass explicit_t_max.")

    warmup_steps = int(float(horizon) * cfg.warmup_ratio)
    min_factor = cfg.eta_min / base_lr

    return LambdaLR(
        optimizer,
        lr_lambda=build_warmup_cosine_lambda(
            horizon=horizon,
            warmup_steps=warmup_steps,
            min_factor=min_factor,
        ),
    )


def build_lightning_scheduler_config(
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    interval: SchedulerInterval,
) -> LRSchedulerConfigType:
    return {
        "scheduler": scheduler,
        "interval": interval,
    }


def build_warmup_cosine_lambda(
    *,
    horizon: int,
    warmup_steps: int,
    min_factor: float,
) -> Callable[[int], float]:
    horizon = int(horizon)
    warmup_steps = int(warmup_steps)
    min_factor = float(min_factor)

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")

    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    if warmup_steps >= horizon:
        raise ValueError(f"warmup_steps must be smaller than horizon; " f"got warmup_steps={warmup_steps}, horizon={horizon}.")

    if min_factor < 0.0 or min_factor > 1.0:
        raise ValueError(f"min_factor must be in [0, 1], got {min_factor}.")

    cosine_steps = horizon - warmup_steps

    def lr_lambda(step: int) -> float:
        step = int(step)

        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step + 1) / float(warmup_steps)
            return max(1.0e-8, progress)

        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return lr_lambda


def resolve_scheduler_horizon(
    *,
    trainer: object,
    explicit_t_max: int | None,
    interval: SchedulerInterval,
) -> int:
    if explicit_t_max is not None:
        horizon = int(explicit_t_max)
        if horizon <= 0:
            raise ValueError(f"explicit_t_max must be positive, got {horizon}.")
        return horizon

    if interval == "step":
        max_steps = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return int(max_steps)

        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, int) and estimated > 0:
            return int(estimated)

        return 0

    if interval == "epoch":
        max_epochs = getattr(trainer, "max_epochs", None)
        if isinstance(max_epochs, int) and max_epochs > 0:
            return int(max_epochs)

        return 0

    raise ValueError(f"Unsupported scheduler interval: {interval!r}.")


def parameter_groups(
    *,
    modules: Sequence[nn.Module],
    weight_decay: float,
    no_decay_on_bias_and_norm: bool,
) -> list[dict[str, Any]]:
    if not no_decay_on_bias_and_norm:
        params = unique_trainable_parameters(modules)
        if not params:
            return []

        return [
            {
                "params": params,
                "weight_decay": float(weight_decay),
            }
        ]

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, parameter in named_unique_trainable_parameters(modules):
        if is_no_decay_parameter(name, parameter):
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    groups: list[dict[str, Any]] = []

    if decay:
        groups.append(
            {
                "params": decay,
                "weight_decay": float(weight_decay),
            }
        )

    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "weight_decay": 0.0,
            }
        )

    return groups


def is_no_decay_parameter(
    name: str,
    parameter: nn.Parameter,
) -> bool:
    return name.endswith(".bias") or parameter.ndim <= 1


def unique_trainable_parameters(
    modules: Sequence[nn.Module],
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue

            parameter_id = id(parameter)
            if parameter_id in seen:
                continue

            seen.add(parameter_id)
            params.append(parameter)

    return params


def named_unique_trainable_parameters(
    modules: Sequence[nn.Module],
) -> list[tuple[str, nn.Parameter]]:
    params: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()

    for module_id, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue

            parameter_id = id(parameter)
            if parameter_id in seen:
                continue

            seen.add(parameter_id)
            params.append((f"module_{module_id}.{name}", parameter))

    return params


def parse_interval(value: Any) -> SchedulerInterval:
    interval = str(value)
    if interval == "step":
        return "step"

    if interval == "epoch":
        return "epoch"

    raise ValueError(f"scheduler.interval must be 'step' or 'epoch', got {interval!r}.")


def parse_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("AdamW betas must be a sequence of two floats; " f"got {type(value)!r}.")

    betas = tuple(float(x) for x in value)
    if len(betas) != 2:
        raise ValueError(f"AdamW betas must contain two values, got {betas}.")

    beta1, beta2 = betas
    if not (0.0 <= beta1 < 1.0):
        raise ValueError(f"AdamW beta1 must be in [0, 1), got {beta1}.")

    if not (0.0 <= beta2 < 1.0):
        raise ValueError(f"AdamW beta2 must be in [0, 1), got {beta2}.")

    return beta1, beta2


def to_plain_mapping(
    cfg: DictConfig | Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(
            cfg,
            resolve=True,
            throw_on_missing=True,
        )
    else:
        container = dict(cfg)

    if not isinstance(container, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(container)!r}.")

    return container


def require_mapping(
    node: Mapping[str, Any],
    *,
    key: str,
    owner: str,
) -> Mapping[str, Any]:
    if key not in node:
        raise KeyError(f"{owner}.{key} is required.")

    value = node[key]
    if isinstance(value, DictConfig):
        return to_plain_mapping(value, name=f"{owner}.{key}")

    if not isinstance(value, Mapping):
        raise TypeError(f"{owner}.{key} must be a mapping, got {type(value)!r}.")

    return value


def positive_float(
    node: Mapping[str, Any],
    key: str,
    *,
    owner: str,
) -> float:
    if key not in node:
        raise KeyError(f"{owner}.{key} is required.")

    value = float(node[key])
    if value <= 0.0:
        raise ValueError(f"{owner}.{key} must be positive, got {value}.")

    return value


def non_negative_float(
    node: Mapping[str, Any],
    key: str,
    *,
    owner: str,
    default: float,
) -> float:
    value = float(node.get(key, default))
    if value < 0.0:
        raise ValueError(f"{owner}.{key} must be non-negative, got {value}.")

    return value


def bounded_float(
    node: Mapping[str, Any],
    key: str,
    *,
    owner: str,
    lower: float,
    upper: float,
    include_upper: bool,
) -> float:
    if key not in node:
        raise KeyError(f"{owner}.{key} is required.")

    value = float(node[key])

    lower_ok = value >= lower
    upper_ok = value <= upper if include_upper else value < upper

    if not lower_ok or not upper_ok:
        upper_op = "<=" if include_upper else "<"
        raise ValueError(f"{owner}.{key} must satisfy {lower} <= value {upper_op} {upper}; " f"got {value}.")

    return value


@contextmanager
def freeze_params(
    module: nn.Module,
) -> Iterator[None]:
    requires_grad = [parameter.requires_grad for parameter in module.parameters()]

    for parameter in module.parameters():
        parameter.requires_grad_(False)

    try:
        yield
    finally:
        for parameter, old_value in zip(module.parameters(), requires_grad):
            parameter.requires_grad_(old_value)


__all__ = [
    "OptimizerSpec",
    "OptimizationSpec",
    "SchedulerInterval",
    "SchedulerSpec",
    "build_lightning_scheduler_config",
    "build_optimizer",
    "build_scheduler",
    "build_warmup_cosine_lambda",
    "configure_optimization",
    "freeze_params",
    "named_unique_trainable_parameters",
    "parameter_groups",
    "parse_optimization_config",
    "resolve_scheduler_horizon",
    "unique_trainable_parameters",
]
