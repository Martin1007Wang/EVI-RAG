"""Optimizer and LR-scheduler construction for PyTorch Lightning.

Design notes
------------
``configure_optimizers`` in Lightning receives ``self`` (the LightningModule),
so all trainer state — ``estimated_stepping_batches``, ``max_steps``,
``max_epochs`` — is available directly via ``self.trainer``.  There is no need
to serialise these into a separate dataclass before calling this helper.

Usage inside a LightningModule::

    from .optimization import build_optimizer_and_scheduler

    class MyModel(LightningModule):
        def configure_optimizers(self):
            return build_optimizer_and_scheduler(
                module=self,
                optimizer_cfg=self.hparams.optimizer,
                scheduler_cfg=self.hparams.scheduler,
            )

The returned dict is exactly the format Lightning's ``configure_optimizers``
expects: ``{"optimizer": ..., "lr_scheduler": {"scheduler": ..., "interval": ...}}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

if TYPE_CHECKING:
    from lightning import LightningModule


# ---------------------------------------------------------------------------
# Parameter-group helpers
# ---------------------------------------------------------------------------


def _is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """Bias and 1-D params (LayerNorm weight, embedding) get zero weight decay."""
    return name.endswith(".bias") or param.ndim <= 1


def _is_log_z_head(name: str) -> bool:
    return "root_flow_" in name


def build_param_groups(
    module: torch.nn.Module,
    *,
    base_lr: float,
    log_z_lr_multiplier: float,
    weight_decay: float,
    no_decay_on_bias_and_norm: bool,
) -> list[dict[str, Any]]:
    """
    Split trainable parameters into up to four AdamW param groups:

    - ``decay``             — normal lr, normal wd
    - ``no_decay``          — normal lr, wd=0
    - ``log_z_head_decay``  — boosted lr, normal wd
    - ``log_z_head_no_decay`` — boosted lr, wd=0
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
# Horizon resolution — uses the trainer directly
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
        # Lightning returns ``float("inf")`` when it cannot estimate; treat as unknown.
        if isinstance(esb, int) and esb > 0:
            return esb
        return None
    else:
        return trainer.max_epochs if trainer.max_epochs and trainer.max_epochs > 0 else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_optimizer_and_scheduler(
    module: LightningModule,
    *,
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
) -> dict[str, Any]:
    """
    Build an AdamW optimizer + optional LR scheduler and return the dict that
    Lightning's ``configure_optimizers`` expects.

    Supported scheduler types
    -------------------------
    ``cosine``               → ``CosineAnnealingLR``
    ``cosine_warm_restarts`` → ``CosineAnnealingWarmRestarts``
    ``onecycle``             → ``OneCycleLR``  (interval must be ``"step"``)

    All scheduler types require a resolvable horizon (``trainer.max_steps``,
    ``trainer.estimated_stepping_batches``, or an explicit ``t_max`` in the
    scheduler config).  If no horizon can be resolved the scheduler is skipped
    and only the optimizer is returned.
    """
    # --- validate optimizer type early ---
    opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_type!r}.  Only 'adamw' is supported.")

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
    if not 0.0 <= beta1 < 1.0:
        raise ValueError("optimizer.betas[0] must be in [0, 1).")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("optimizer.betas[1] must be in [0, 1).")

    log_z_mult = float(optimizer_cfg.get("log_z_head_lr_multiplier", 5.0))
    if log_z_mult <= 0.0:
        raise ValueError("optimizer.log_z_head_lr_multiplier must be > 0.")

    no_decay_split = bool(optimizer_cfg.get("no_decay_on_bias_and_norm", True))

    # --- build optimizer ---
    param_groups = build_param_groups(
        module,
        base_lr=base_lr,
        log_z_lr_multiplier=log_z_mult,
        weight_decay=weight_decay,
        no_decay_on_bias_and_norm=no_decay_split,
    )
    optimizer = AdamW(param_groups, lr=base_lr, weight_decay=weight_decay, betas=(beta1, beta2))

    # --- build scheduler ---
    raw_interval = str(scheduler_cfg.get("interval", "step")).lower()
    if raw_interval not in {"step", "epoch"}:
        raise ValueError(f"scheduler.interval must be 'step' or 'epoch', got {raw_interval!r}.")
    interval: Literal["step", "epoch"] = raw_interval  # type: ignore[assignment]

    explicit_t_max = int(scheduler_cfg["t_max"]) if scheduler_cfg.get("t_max") is not None else None
    horizon = _resolve_horizon(module, explicit_t_max=explicit_t_max, interval=interval)

    if horizon is None:
        # No horizon available — return optimizer only.
        return {"optimizer": optimizer}

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
        # Guard against t_max being shorter than the full training run.
        esb = module.trainer.estimated_stepping_batches
        if isinstance(esb, int) and esb > 0 and horizon < esb:
            raise ValueError(
                f"OneCycleLR would exhaust before training ends: "
                f"t_max={horizon}, estimated_stepping_batches={esb}.  "
                "Set trainer.max_steps and scheduler.t_max consistently."
            )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=float(scheduler_cfg.get("lr", base_lr)),
            total_steps=horizon,
            pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
            anneal_strategy=scheduler_cfg.get("anneal", "cos"),
        )

    else:
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type!r}.  " "Expected one of 'cosine', 'cosine_warm_restarts', 'onecycle'."
        )

    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": interval},
    }
