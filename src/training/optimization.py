from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, Protocol

import torch
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

SchedulerInterval = Literal["step", "epoch"]


class OptimizerConfig(Protocol):
    @property
    def lr(self) -> float: ...

    @property
    def betas(self) -> tuple[float, float]: ...

    @property
    def weight_decay(self) -> float: ...

    @property
    def no_decay_on_bias_and_norm(self) -> bool: ...


class SchedulerConfig(Protocol):
    @property
    def interval(self) -> SchedulerInterval: ...

    @property
    def warmup_ratio(self) -> float: ...

    @property
    def eta_min(self) -> float: ...


def build_optimizer(
    *,
    modules: Sequence[nn.Module],
    cfg: OptimizerConfig,
) -> torch.optim.Optimizer:
    params = parameter_groups(
        modules=modules,
        weight_decay=cfg.weight_decay,
        no_decay_on_bias_and_norm=cfg.no_decay_on_bias_and_norm,
    )

    return AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    cfg: SchedulerConfig | None,
    trainer: object,
    base_lr: float,
    explicit_t_max: int | None = None,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg is None:
        return None
    if str(cfg.type) == "none":
        return None

    horizon = resolve_scheduler_horizon(
        trainer=trainer,
        explicit_t_max=explicit_t_max,
        interval=cfg.interval,
    )
    if horizon <= 0:
        raise RuntimeError("Could not resolve a positive scheduler horizon. " "Set trainer.max_steps / trainer.max_epochs, or pass explicit_t_max.")

    warmup_steps = int(float(horizon) * float(cfg.warmup_ratio))
    min_factor = float(cfg.eta_min) / float(base_lr)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_warmup_cosine_lambda(
            horizon=horizon,
            warmup_steps=warmup_steps,
            min_factor=min_factor,
        ),
    )

    return scheduler


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
    if min_factor < 0.0:
        raise ValueError(f"min_factor must be non-negative, got {min_factor}.")

    cosine_steps = max(1, horizon - warmup_steps)

    def lr_lambda(step: int) -> float:
        step = int(step)

        if warmup_steps > 0 and step <= warmup_steps:
            progress = float(step) / float(warmup_steps)
            return 1.0e-8 + (1.0 - 1.0e-8) * progress

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
        return int(explicit_t_max)

    if interval == "step":
        max_steps = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return int(max_steps)

        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, int) and estimated > 0:
            return int(estimated)

        return 0

    max_epochs = getattr(trainer, "max_epochs", None)
    if isinstance(max_epochs, int) and max_epochs > 0:
        return int(max_epochs)

    return 0


def parameter_groups(
    *,
    modules: Sequence[nn.Module],
    weight_decay: float,
    no_decay_on_bias_and_norm: bool,
) -> list[nn.Parameter] | list[dict[str, Any]]:
    if not no_decay_on_bias_and_norm:
        return unique_parameters(modules)

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []

    for name, parameter in named_unique_parameters(modules):
        if name.endswith(".bias") or parameter.ndim <= 1:
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


def unique_parameters(
    modules: Sequence[nn.Module],
) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in modules:
        for parameter in module.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue

            seen.add(parameter_id)
            params.append(parameter)

    return params


def named_unique_parameters(
    modules: Sequence[nn.Module],
) -> list[tuple[str, nn.Parameter]]:
    params: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()

    for module_id, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            parameter_id = id(parameter)
            if parameter_id in seen:
                continue

            seen.add(parameter_id)
            params.append((f"module_{module_id}.{name}", parameter))

    return params


@contextmanager
def freeze_params(
    module: nn.Module,
) -> Iterator[None]:
    """
    Temporarily disable gradients for a module.

    Use this when actor loss reads critic / utility outputs but must not update
    critic parameters.
    """

    requires_grad = [parameter.requires_grad for parameter in module.parameters()]

    for parameter in module.parameters():
        parameter.requires_grad_(False)

    try:
        yield
    finally:
        for parameter, old_value in zip(module.parameters(), requires_grad):
            parameter.requires_grad_(old_value)


__all__ = [
    "OptimizerConfig",
    "SchedulerConfig",
    "SchedulerInterval",
    "build_lightning_scheduler_config",
    "build_optimizer",
    "build_scheduler",
    "build_warmup_cosine_lambda",
    "freeze_params",
    "named_unique_parameters",
    "parameter_groups",
    "resolve_scheduler_horizon",
    "unique_parameters",
]
