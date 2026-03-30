from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from .training_schedules import TrainingScheduleContext, normalize_scheduler_interval


def use_zero_weight_decay(*, name: str, parameter: torch.nn.Parameter) -> bool:
    if name.endswith(".bias"):
        return True
    return parameter.ndim <= 1


def is_log_z_head_parameter(*, name: str) -> bool:
    return "root_flow_" in name


def _resolve_optimizer_hyperparameters(
    optimizer_cfg: dict[str, Any],
) -> tuple[float, float, float, tuple[float, float], bool]:
    base_lr = float(optimizer_cfg.get("lr", 1.0e-4))
    if base_lr <= 0.0:
        raise ValueError("optimizer.lr must be > 0.")
    log_z_head_lr_multiplier = float(optimizer_cfg.get("log_z_head_lr_multiplier", 5.0))
    if log_z_head_lr_multiplier <= 0.0:
        raise ValueError("optimizer.log_z_head_lr_multiplier must be > 0.")
    base_weight_decay = float(optimizer_cfg.get("weight_decay", 1.0e-4))
    if base_weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be >= 0.")
    raw_betas = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))
    if len(raw_betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values.")
    beta1, beta2 = (float(raw_betas[0]), float(raw_betas[1]))
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("optimizer.betas[0] must be in [0, 1).")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("optimizer.betas[1] must be in [0, 1).")
    return (
        base_lr,
        log_z_head_lr_multiplier,
        base_weight_decay,
        (beta1, beta2),
        bool(optimizer_cfg.get("no_decay_on_bias_and_norm", True)),
    )


def build_optimizer_param_groups(
    *,
    model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    trainable_named_params = [
        (name, parameter)
        for name, parameter in model_parameters
        if parameter.requires_grad
    ]
    if not trainable_named_params:
        raise RuntimeError("No trainable parameters found in model.")

    (
        base_lr,
        log_z_head_lr_multiplier,
        base_weight_decay,
        _,
        use_no_decay_split,
    ) = _resolve_optimizer_hyperparameters(optimizer_cfg)
    grouped_params: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in trainable_named_params:
        has_zero_weight_decay = use_no_decay_split and use_zero_weight_decay(
            name=name, parameter=parameter
        )
        group_key = (is_log_z_head_parameter(name=name), has_zero_weight_decay)
        grouped_params.setdefault(group_key, []).append(parameter)

    param_groups: list[dict[str, Any]] = []
    for (is_log_z_head, has_zero_weight_decay), params in grouped_params.items():
        group_lr = base_lr * (log_z_head_lr_multiplier if is_log_z_head else 1.0)
        group_weight_decay = 0.0 if has_zero_weight_decay else base_weight_decay
        if is_log_z_head and has_zero_weight_decay:
            group_name = "log_z_head_no_decay"
        elif is_log_z_head:
            group_name = "log_z_head_decay"
        elif has_zero_weight_decay:
            group_name = "no_decay"
        else:
            group_name = "decay"
        param_groups.append(
            {
                "params": params,
                "lr": group_lr,
                "weight_decay": group_weight_decay,
                "group_name": group_name,
            }
        )
    return param_groups


def build_optimizer_and_scheduler(
    *,
    model_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    schedule_context: TrainingScheduleContext,
) -> dict[str, Any]:
    optimizer_param_groups = build_optimizer_param_groups(
        model_parameters=model_parameters,
        optimizer_cfg=optimizer_cfg,
    )
    base_lr, _, base_weight_decay, betas, _ = _resolve_optimizer_hyperparameters(
        optimizer_cfg
    )

    opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
    optimizer = AdamW(
        optimizer_param_groups,
        lr=base_lr,
        weight_decay=base_weight_decay,
        betas=betas,
    )
    scheduler = None
    scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
    interval = normalize_scheduler_interval(scheduler_cfg)
    explicit_t_max = (
        int(scheduler_cfg["t_max"]) if scheduler_cfg.get("t_max") is not None else None
    )
    schedule_horizon = schedule_context.resolve_horizon(
        explicit_horizon=explicit_t_max,
        interval=interval,
    )
    if schedule_horizon is not None:
        eta_min = float(scheduler_cfg.get("eta_min", 1.0e-6))
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=schedule_horizon,
                eta_min=eta_min,
            )
        elif scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=schedule_horizon,
                T_mult=int(scheduler_cfg.get("t_mult", 1)),
                eta_min=eta_min,
            )
        elif scheduler_type == "onecycle":
            if interval != "step":
                raise ValueError(
                    "onecycle scheduler requires interval='step' because it must advance per optimizer step."
                )
            configured_training_steps = schedule_context.configured_training_steps()
            if (
                configured_training_steps is not None
                and schedule_horizon < configured_training_steps
            ):
                raise ValueError(
                    "onecycle scheduler would exhaust before training ends: "
                    f"t_max={schedule_horizon} configured_steps={configured_training_steps}. "
                    "Set trainer.max_steps and scheduler t_max consistently."
                )
            scheduler_lr = scheduler_cfg.get("lr", optimizer_cfg.get("lr", 1.0e-4))
            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(scheduler_lr),
                total_steps=schedule_horizon,
                pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
                anneal_strategy=str(scheduler_cfg.get("anneal", "cos")),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    result: dict[str, Any] = {"optimizer": optimizer}
    if scheduler is not None:
        result["lr_scheduler"] = {"scheduler": scheduler, "interval": interval}
    return result


__all__ = [
    "build_optimizer_and_scheduler",
    "build_optimizer_param_groups",
    "is_log_z_head_parameter",
    "use_zero_weight_decay",
]
