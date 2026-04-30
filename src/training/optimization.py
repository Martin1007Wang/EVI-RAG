from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, cast

import torch
from lightning import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


SchedulerInterval = Literal["step", "epoch"]
SchedulerType = Literal["none", "constant", "cosine", "cosine_with_warmup"]


@dataclass(frozen=True)
class AdamWConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    flow_scalar_head_lr_multiplier: float = 1.0
    no_decay_on_bias_and_norm: bool = True

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> AdamWConfig:
        opt_type = str(cfg.get("type", "adamw")).lower()
        if opt_type != "adamw":
            raise ValueError(f"Unsupported optimizer type: {opt_type!r}. " "Only 'adamw' is supported.")

        lr = float(cfg.get("lr", cls.lr))
        if lr <= 0.0:
            raise ValueError(f"optimizer.lr must be > 0, got {lr}.")

        weight_decay = float(cfg.get("weight_decay", cls.weight_decay))
        if weight_decay < 0.0:
            raise ValueError(f"optimizer.weight_decay must be >= 0, got {weight_decay}.")

        raw_betas = tuple(cfg.get("betas", cls.betas))
        if len(raw_betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values, " f"got {raw_betas}.")

        beta1 = float(raw_betas[0])
        beta2 = float(raw_betas[1])
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"optimizer.betas[0] must be in [0, 1), got {beta1}.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"optimizer.betas[1] must be in [0, 1), got {beta2}.")

        # Canonical name.
        if "flow_scalar_head_lr_multiplier" in cfg:
            multiplier = float(cfg["flow_scalar_head_lr_multiplier"])
        # Backward-compatible alias for existing configs.
        elif "log_z_head_lr_multiplier" in cfg:
            multiplier = float(cfg["log_z_head_lr_multiplier"])
        else:
            multiplier = cls.flow_scalar_head_lr_multiplier

        if multiplier <= 0.0:
            raise ValueError("optimizer.flow_scalar_head_lr_multiplier must be > 0, " f"got {multiplier}.")

        return cls(
            lr=lr,
            weight_decay=weight_decay,
            betas=(beta1, beta2),
            flow_scalar_head_lr_multiplier=multiplier,
            no_decay_on_bias_and_norm=bool(
                cfg.get(
                    "no_decay_on_bias_and_norm",
                    cls.no_decay_on_bias_and_norm,
                )
            ),
        )


@dataclass(frozen=True)
class SchedulerConfig:
    type: SchedulerType = "cosine_with_warmup"
    interval: SchedulerInterval = "step"
    num_warmup_steps: int = 200
    t_max: int | None = None
    eta_min: float = 0.0
    min_lr_ratio: float = 0.0
    monitor: str | None = None
    frequency: int | None = None
    strict: bool | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> SchedulerConfig:
        scheduler_type = str(cfg.get("type", cls.type)).lower()
        allowed_types = {"none", "constant", "cosine", "cosine_with_warmup"}
        if scheduler_type not in allowed_types:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type!r}. " f"Expected one of {sorted(allowed_types)}.")

        raw_interval = str(cfg.get("interval", cls.interval)).lower()
        if raw_interval not in {"step", "epoch"}:
            raise ValueError(f"scheduler.interval must be 'step' or 'epoch', got {raw_interval!r}.")

        num_warmup_steps = int(cfg.get("num_warmup_steps", cls.num_warmup_steps))
        if num_warmup_steps < 0:
            raise ValueError(f"scheduler.num_warmup_steps must be >= 0, got {num_warmup_steps}.")

        t_max = cfg.get("t_max", None)
        explicit_t_max = int(t_max) if t_max is not None else None
        if explicit_t_max is not None and explicit_t_max <= 0:
            raise ValueError(f"scheduler.t_max must be > 0, got {explicit_t_max}.")

        eta_min = float(cfg.get("eta_min", cls.eta_min))
        if eta_min < 0.0:
            raise ValueError(f"scheduler.eta_min must be >= 0, got {eta_min}.")

        min_lr_ratio = float(cfg.get("min_lr_ratio", cls.min_lr_ratio))
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("scheduler.min_lr_ratio must be in [0, 1], " f"got {min_lr_ratio}.")

        frequency = cfg.get("frequency", None)
        parsed_frequency = int(frequency) if frequency is not None else None
        if parsed_frequency is not None and parsed_frequency <= 0:
            raise ValueError(f"scheduler.frequency must be > 0, got {parsed_frequency}.")

        strict = cfg.get("strict", None)

        return cls(
            type=cast(SchedulerType, scheduler_type),
            interval=cast(SchedulerInterval, raw_interval),
            num_warmup_steps=num_warmup_steps,
            t_max=explicit_t_max,
            eta_min=eta_min,
            min_lr_ratio=min_lr_ratio,
            monitor=cfg.get("monitor", None),
            frequency=parsed_frequency,
            strict=bool(strict) if strict is not None else None,
        )


def build_optimizer_and_scheduler(
    module: LightningModule,
    *,
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any] | None = None,
) -> OptimizerLRScheduler:
    """
    Build the optimizer and optional LR scheduler for Lightning.

    This function belongs in src.training, not src.utils, because it depends on:
    - LightningModule;
    - module.trainer;
    - Lightning's configure_optimizers return format.
    """
    opt_cfg = AdamWConfig.from_dict(optimizer_cfg)

    optimizer = AdamW(
        build_param_groups(module, cfg=opt_cfg),
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        weight_decay=opt_cfg.weight_decay,
    )

    if not scheduler_cfg:
        return optimizer

    sch_cfg = SchedulerConfig.from_dict(scheduler_cfg)

    if sch_cfg.type in {"none", "constant"}:
        return optimizer

    horizon = resolve_scheduler_horizon(
        module,
        explicit_t_max=sch_cfg.t_max,
        interval=sch_cfg.interval,
    )

    if horizon <= 0:
        raise RuntimeError(
            "Could not resolve a positive scheduler horizon. "
            "Set scheduler_cfg.t_max explicitly, or make sure Trainer has "
            "max_steps / estimated_stepping_batches / max_epochs available."
        )

    if sch_cfg.type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=horizon,
            eta_min=sch_cfg.eta_min,
        )

    elif sch_cfg.type == "cosine_with_warmup":
        if sch_cfg.interval != "step":
            raise ValueError("scheduler.type='cosine_with_warmup' requires interval='step'.")

        if sch_cfg.num_warmup_steps >= horizon:
            raise ValueError(
                "scheduler.num_warmup_steps must be smaller than the scheduler " f"horizon. Got warmup={sch_cfg.num_warmup_steps}, horizon={horizon}."
            )

        scheduler = LambdaLR(
            optimizer,
            lr_lambda=build_cosine_warmup_lambda(
                num_warmup_steps=sch_cfg.num_warmup_steps,
                num_training_steps=horizon,
                min_lr_ratio=sch_cfg.min_lr_ratio,
            ),
        )

    else:
        raise AssertionError(f"Unhandled scheduler type: {sch_cfg.type!r}.")

    return cast(
        OptimizerLRScheduler,
        {
            "optimizer": optimizer,
            "lr_scheduler": build_lightning_scheduler_config(
                scheduler=scheduler,
                cfg=sch_cfg,
            ),
        },
    )


def build_param_groups(
    module: nn.Module,
    *,
    cfg: AdamWConfig,
) -> list[dict[str, Any]]:
    """
    Create deterministic AdamW parameter groups.

    Groups:
    - main_decay
    - main_no_decay
    - flow_scalar_decay
    - flow_scalar_no_decay

    "flow scalar" means z_head and flow_head. This is deliberately not called
    log_z_head because flow_head controls log F(s), not only log Z.
    """
    grouped: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }

    param_names: dict[tuple[bool, bool], list[str]] = {
        (False, False): [],
        (False, True): [],
        (True, False): [],
        (True, True): [],
    }

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue

        is_flow_scalar = is_flow_scalar_head(name)
        no_decay = cfg.no_decay_on_bias_and_norm and is_no_decay_parameter(
            name,
            param,
        )

        key = (is_flow_scalar, no_decay)
        grouped[key].append(param)
        param_names[key].append(name)

    group_names = {
        (False, False): "main_decay",
        (False, True): "main_no_decay",
        (True, False): "flow_scalar_decay",
        (True, True): "flow_scalar_no_decay",
    }

    param_groups: list[dict[str, Any]] = []
    for key in ((False, False), (False, True), (True, False), (True, True)):
        params = grouped[key]
        if not params:
            continue

        is_flow_scalar, no_decay = key
        lr = cfg.lr * (cfg.flow_scalar_head_lr_multiplier if is_flow_scalar else 1.0)

        param_groups.append(
            {
                "params": params,
                "lr": lr,
                "weight_decay": 0.0 if no_decay else cfg.weight_decay,
                "name": group_names[key],
                # Extra optimizer group metadata. PyTorch optimizers preserve
                # unknown keys; this is useful for debugging and checkpoint
                # introspection.
                "param_names": tuple(param_names[key]),
            }
        )

    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    return param_groups


def is_no_decay_parameter(name: str, param: nn.Parameter) -> bool:
    """
    Standard AdamW rule:
    - no weight decay for bias;
    - no weight decay for 1-D parameters such as LayerNorm scales and embeddings.
    """
    return name.endswith(".bias") or param.ndim <= 1


def is_flow_scalar_head(name: str) -> bool:
    """
    Parameters of scalar flow heads.

    Includes:
    - z_head: root log partition log Z(q);
    - flow_head: state log-flow log F(s | q).

    This deliberately does not include the edge scorer or action head.
    """
    return (
        ".z_head." in name
        or name.startswith("policy.z_head.")
        or ".flow_head." in name
        or name.startswith("policy.flow_head.")
        or "root_flow_" in name
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


def build_cosine_warmup_lambda(
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> Any:
    """
    Return LambdaLR multiplier function.

    During warmup:
        lr_ratio = step / warmup

    After warmup:
        lr_ratio = min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be positive, got {num_training_steps}.")
    if num_warmup_steps < 0:
        raise ValueError(f"num_warmup_steps must be non-negative, got {num_warmup_steps}.")
    if num_warmup_steps >= num_training_steps:
        raise ValueError("num_warmup_steps must be smaller than num_training_steps, " f"got warmup={num_warmup_steps}, total={num_training_steps}.")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}.")

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return lr_lambda


def build_lightning_scheduler_config(
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg: SchedulerConfig,
) -> LRSchedulerConfig:
    config_dict: dict[str, Any] = {
        "scheduler": scheduler,
        "interval": cfg.interval,
    }

    if cfg.monitor is not None:
        config_dict["monitor"] = cfg.monitor
    if cfg.frequency is not None:
        config_dict["frequency"] = cfg.frequency
    if cfg.strict is not None:
        config_dict["strict"] = cfg.strict

    return cast(LRSchedulerConfig, config_dict)


__all__ = [
    "AdamWConfig",
    "SchedulerConfig",
    "build_optimizer_and_scheduler",
    "build_param_groups",
    "build_cosine_warmup_lambda",
    "build_lightning_scheduler_config",
    "is_flow_scalar_head",
    "is_no_decay_parameter",
    "resolve_scheduler_horizon",
]
