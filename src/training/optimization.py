from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


SchedulerInterval = Literal["step", "epoch"]
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_WARMUP_RATIO = 0.2


def build_optimizer_and_scheduler(
    module: LightningModule,
    *,
    optimizer_cfg: dict[str, object],
    scheduler_cfg: dict[str, object] | None = None,
) -> OptimizerLRScheduler:
    """
    Build the optimizer and optional LR scheduler for Lightning.

    This function belongs in src.training, not src.utils, because it depends on:
    - LightningModule;
    - module.trainer;
    - Lightning's configure_optimizers return format.
    """
    del optimizer_cfg, scheduler_cfg

    optimizer = AdamW(
        module.parameters(),
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    horizon = resolve_scheduler_horizon(
        module,
        explicit_t_max=None,
        interval="step",
    )

    if horizon <= 0:
        raise RuntimeError(
            "Could not resolve a positive scheduler horizon. "
            "Make sure Trainer has max_steps or estimated_stepping_batches "
            "available."
        )

    warmup_steps = int(horizon * DEFAULT_WARMUP_RATIO)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=build_warmup_cosine_lambda(
            horizon=horizon,
            warmup_steps=warmup_steps,
        ),
    )

    return cast(
        OptimizerLRScheduler,
        {
            "optimizer": optimizer,
            "lr_scheduler": build_lightning_scheduler_config(
                scheduler=scheduler,
            ),
        },
    )


def resolve_scheduler_horizon(
    module: LightningModule,
    *,
    explicit_t_max: int | None,
    interval: SchedulerInterval,
) -> int:
    if explicit_t_max is not None:
        return explicit_t_max

    trainer = module.trainer

    if interval == "step":
        max_steps = getattr(trainer, "max_steps", None)
        if isinstance(max_steps, int) and max_steps > 0:
            return max_steps

        estimated = getattr(trainer, "estimated_stepping_batches", None)
        if isinstance(estimated, int) and estimated > 0:
            return estimated

        return 0

    max_epochs = getattr(trainer, "max_epochs", None)
    if isinstance(max_epochs, int) and max_epochs > 0:
        return max_epochs

    return 0


def build_lightning_scheduler_config(
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> LRSchedulerConfig:
    config_dict: dict[str, Any] = {
        "scheduler": scheduler,
        "interval": "step",
    }

    return cast(LRSchedulerConfig, config_dict)


def build_warmup_cosine_lambda(
    *,
    horizon: int,
    warmup_steps: int,
) -> Callable[[int], float]:
    horizon = int(horizon)
    warmup_steps = int(warmup_steps)
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}.")

    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step <= warmup_steps:
            progress = float(step) / float(warmup_steps)
            return 1.0e-8 + (1.0 - 1.0e-8) * progress

        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(horizon)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


__all__ = [
    "DEFAULT_LR",
    "DEFAULT_WARMUP_RATIO",
    "DEFAULT_WEIGHT_DECAY",
    "build_optimizer_and_scheduler",
    "build_lightning_scheduler_config",
    "build_warmup_cosine_lambda",
    "resolve_scheduler_horizon",
]
