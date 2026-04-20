"""Optimizer and LR-scheduler construction for PyTorch Lightning.

Design notes
------------
``configure_optimizers`` in Lightning receives ``self`` (the LightningModule),
so all trainer state — ``estimated_stepping_batches``, ``max_steps``,
``max_epochs`` — is available directly via ``self.trainer``.  There is no need
to serialise these into a separate dataclass before calling this helper.
"""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
)
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, LRSchedulerConfig

# ---------------------------------------------------------------------------
# Parameter-group helpers
# ---------------------------------------------------------------------------


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Bias and 1-D params (LayerNorm weight, embedding) get zero weight decay."""
    return name.endswith(".bias") or param.ndim <= 1


def _is_log_z_head(name: str) -> bool:
    return (
        "root_flow_" in name
        or ".z_head." in name
        or name.startswith("policy.z_head.")
        or ".flow_head." in name
        or name.startswith("policy.flow_head.")
    )


def build_param_groups(
    module: torch.nn.Module,
    *,
    base_lr: float,
    log_z_lr_multiplier: float,
    weight_decay: float,
    no_decay_on_bias_and_norm: bool,
) -> list[dict[str, Any]]:
    """
    Split trainable parameters into up to four AdamW param groups.
    """
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = no_decay_on_bias_and_norm and _is_no_decay(name, param)
        key = (_is_log_z_head(name), no_decay)
        groups.setdefault(key, []).append(param)

    if not groups:
        raise RuntimeError("No trainable parameters found.")

    _names = {
        (False, False): "decay",
        (False, True): "no_decay",
        (True, False): "log_z_head_decay",
        (True, True): "log_z_head_no_decay",
    }

    return [
        {
            "params": params,
            "lr": base_lr * (log_z_lr_multiplier if is_log_z else 1.0),
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": _names[(is_log_z, no_decay)],
        }
        for (is_log_z, no_decay), params in groups.items()
    ]


# ---------------------------------------------------------------------------
# Horizon resolution
# ---------------------------------------------------------------------------


def _resolve_horizon(
    module: LightningModule,
    *,
    explicit_t_max: int | None,
    interval: Literal["step", "epoch"],
) -> int | None:
    if explicit_t_max is not None:
        if explicit_t_max <= 0:
            raise ValueError(f"scheduler.t_max must be > 0, got {explicit_t_max}.")
        return explicit_t_max

    trainer = module.trainer
    if interval == "step":
        if trainer.max_steps and trainer.max_steps > 0:
            return trainer.max_steps
        esb = trainer.estimated_stepping_batches
        if isinstance(esb, int) and esb > 0:
            return esb
        return None
    else:
        return (
            trainer.max_epochs
            if trainer.max_epochs and trainer.max_epochs > 0
            else None
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_optimizer_and_scheduler(
    module: LightningModule,
    *,
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any] | None = None,
) -> OptimizerLRScheduler:
    """
    Build an AdamW optimizer + optional LR scheduler and return the structure
    that Lightning's ``configure_optimizers`` expects (OptimizerLRScheduler).
    """
    # --- validate optimizer type early ---
    opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(
            f"Unsupported optimizer type: {opt_type!r}. Only 'adamw' is supported."
        )

    # --- parse optimizer hyper-parameters ---
    base_lr = float(optimizer_cfg.get("lr", 1e-4))
    if base_lr <= 0.0:
        raise ValueError("optimizer.lr must be > 0.")

    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-4))
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be >= 0.")

    raw_betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    if len(raw_betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values.")
    beta1, beta2 = float(raw_betas[0]), float(raw_betas[1])

    log_z_mult = float(optimizer_cfg.get("log_z_head_lr_multiplier", 5.0))
    no_decay_split = bool(optimizer_cfg.get("no_decay_on_bias_and_norm", True))

    # --- build optimizer ---
    param_groups = build_param_groups(
        module,
        base_lr=base_lr,
        log_z_lr_multiplier=log_z_mult,
        weight_decay=weight_decay,
        no_decay_on_bias_and_norm=no_decay_split,
    )
    optimizer = AdamW(
        param_groups, lr=base_lr, weight_decay=weight_decay, betas=(beta1, beta2)
    )

    if not scheduler_cfg:
        return optimizer

    # --- build scheduler ---
    raw_interval = str(scheduler_cfg.get("interval", "step")).lower()
    if raw_interval not in {"step", "epoch"}:
        raise ValueError(
            f"scheduler.interval must be 'step' or 'epoch', got {raw_interval!r}."
        )
    interval: Literal["step", "epoch"] = raw_interval

    explicit_t_max = (
        int(scheduler_cfg["t_max"]) if scheduler_cfg.get("t_max") is not None else None
    )
    horizon = _resolve_horizon(module, explicit_t_max=explicit_t_max, interval=interval)

    if horizon is None:
        return optimizer

    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    eta_min = float(scheduler_cfg.get("eta_min", 1e-6))

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=horizon, eta_min=eta_min)

    elif scheduler_type == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=horizon,
            T_mult=int(scheduler_cfg.get("t_mult", 1)),
            eta_min=eta_min,
        )

    elif scheduler_type == "onecycle":
        if interval != "step":
            raise ValueError("OneCycleLR requires scheduler.interval='step'.")
        esb = module.trainer.estimated_stepping_batches
        if isinstance(esb, int) and esb > 0 and horizon < esb:
            raise ValueError("OneCycleLR would exhaust before training ends.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(scheduler_cfg.get("lr", base_lr)),
            total_steps=horizon,
            pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
            anneal_strategy=scheduler_cfg.get("anneal", "cos"),
        )

    # 🚨 新增：支持 Hugging Face 风格的 cosine_with_warmup 🚨
    elif scheduler_type == "cosine_with_warmup":
        if interval != "step":
            raise ValueError("cosine_with_warmup requires scheduler.interval='step'.")

        num_warmup_steps = int(scheduler_cfg.get("num_warmup_steps", 500))

        def lr_lambda(current_step: int) -> float:
            # 1. 线性预热阶段
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            # 2. 余弦衰减阶段
            progress = float(current_step - num_warmup_steps) / float(
                max(1, horizon - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type!r}. "
            "Expected 'cosine', 'cosine_warm_restarts', 'onecycle', or 'cosine_with_warmup'."
        )

    config_dict: dict[str, Any] = {
        "scheduler": scheduler,
        "interval": interval,
    }
    for key in ["monitor", "frequency", "strict"]:
        if key in scheduler_cfg:
            config_dict[key] = scheduler_cfg[key]

    lr_scheduler_config = cast(LRSchedulerConfig, config_dict)

    return cast(
        OptimizerLRScheduler,
        {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        },
    )
