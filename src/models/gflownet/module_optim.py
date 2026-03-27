from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from .schedules import TrainingScheduleContext, normalize_scheduler_interval


def cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Expected dataclass or dict config, got {type(cfg)!r}.")


def use_zero_weight_decay(*, name: str, parameter: torch.nn.Parameter) -> bool:
    if name.endswith(".bias"):
        return True
    return parameter.ndim <= 1


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

    base_lr = float(optimizer_cfg.get("lr", 1.0e-4))
    base_weight_decay = float(optimizer_cfg.get("weight_decay", 0.01))
    if not bool(optimizer_cfg.get("no_decay_on_bias_and_norm", True)):
        return [
            {
                "params": [parameter for _, parameter in trainable_named_params],
                "lr": base_lr,
                "weight_decay": base_weight_decay,
                "group_name": "default",
            }
        ]

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, parameter in trainable_named_params:
        if use_zero_weight_decay(name=name, parameter=parameter):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append(
            {
                "params": decay_params,
                "lr": base_lr,
                "weight_decay": base_weight_decay,
                "group_name": "decay",
            }
        )
    if no_decay_params:
        param_groups.append(
            {
                "params": no_decay_params,
                "lr": base_lr,
                "weight_decay": 0.0,
                "group_name": "no_decay",
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

    opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
    optimizer = AdamW(
        optimizer_param_groups,
        lr=float(optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
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
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))
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
    "cfg_to_dict",
    "use_zero_weight_decay",
]
