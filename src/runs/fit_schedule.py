from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict


@dataclass(frozen=True)
class PassFitScheduleConfig:
    max_passes: float
    val_every_passes: float
    early_stopping_patience_passes: float

    def __post_init__(self) -> None:
        if self.max_passes <= 0.0:
            raise ValueError("fit_schedule.max_passes must be > 0.")
        if self.val_every_passes <= 0.0:
            raise ValueError("fit_schedule.val_every_passes must be > 0.")
        if self.early_stopping_patience_passes <= 0.0:
            raise ValueError("fit_schedule.early_stopping_patience_passes must be > 0.")

    @classmethod
    def from_config(
        cls, cfg: DictConfig | dict[str, Any] | None
    ) -> PassFitScheduleConfig:
        if cfg is None:
            raise ValueError(
                "Missing required config group: `fit_schedule`. "
                "Fix: use the pass-based default or pass `fit_schedule=pass`."
            )
        container = (
            OmegaConf.to_container(cfg, resolve=True)
            if isinstance(cfg, DictConfig)
            else dict(cfg)
        )
        if not isinstance(container, dict):
            raise TypeError("fit_schedule must resolve to a mapping.")
        mode = str(container.get("mode") or "pass_based").strip().lower()
        if mode != "pass_based":
            raise ValueError(
                "fit_schedule.mode must be 'pass_based'. "
                f"Got {container.get('mode')!r}."
            )
        return cls(
            max_passes=_resolve_required_float(container, field_name="max_passes"),
            val_every_passes=_resolve_required_float(
                container,
                field_name="val_every_passes",
            ),
            early_stopping_patience_passes=_resolve_required_float(
                container,
                field_name="early_stopping_patience_passes",
            ),
        )


@dataclass(frozen=True)
class ResolvedPassFitSchedule:
    max_passes: float
    val_every_passes: float
    early_stopping_patience_passes: float
    train_size: int
    per_device_batch_size: int
    data_parallel_size: int
    global_batch_size: int
    accumulate_grad_batches: int
    examples_per_optimizer_step: int
    train_batches_per_pass: float
    optimizer_steps_per_pass: float
    max_steps: int
    val_check_interval_batches: int
    early_stopping_patience_checks: int

    def effective_pass(self, *, global_step: int) -> float:
        return (
            float(max(global_step, 0)) * float(self.examples_per_optimizer_step)
        ) / float(self.train_size)


def _resolve_positive_int(value: Any, *, field_name: str) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{field_name} must be >= 1.")
    return resolved


def _resolve_required_float(container: dict[str, Any], *, field_name: str) -> float:
    value = container.get(field_name)
    if value is None:
        raise ValueError(f"fit_schedule.{field_name} must be provided.")
    return float(value)


def _resolve_configured_device_count(trainer_cfg: DictConfig | dict[str, Any]) -> int:
    raw_devices = trainer_cfg.get("devices") if hasattr(trainer_cfg, "get") else None
    if raw_devices in (None, "", "auto"):
        return 1
    if isinstance(raw_devices, int):
        return max(int(raw_devices), 1)
    if isinstance(raw_devices, str):
        tokens = [token.strip() for token in raw_devices.split(",") if token.strip()]
        if not tokens:
            return 1
        if len(tokens) == 1 and tokens[0].isdigit():
            return max(int(tokens[0]), 1)
        return len(tokens)
    if OmegaConf.is_list(raw_devices) or isinstance(raw_devices, (list, tuple)):
        return max(len(list(raw_devices)), 1)
    return 1


def resolve_pass_fit_schedule(
    *,
    fit_schedule_cfg: DictConfig | dict[str, Any],
    trainer_cfg: DictConfig | dict[str, Any],
    train_size: int,
    per_device_batch_size: int,
) -> ResolvedPassFitSchedule:
    schedule_cfg = PassFitScheduleConfig.from_config(fit_schedule_cfg)
    if train_size < 1:
        raise ValueError(
            "Resolved train split is empty; cannot derive pass-based schedule."
        )
    batch_size = _resolve_positive_int(
        per_device_batch_size,
        field_name="data.batch_size",
    )
    data_parallel_size = _resolve_configured_device_count(trainer_cfg)
    accumulate_grad_batches = _resolve_positive_int(
        (
            trainer_cfg.get("accumulate_grad_batches")
            if hasattr(trainer_cfg, "get")
            else 1
        )
        or 1,
        field_name="trainer.accumulate_grad_batches",
    )
    global_batch_size = batch_size * data_parallel_size
    examples_per_optimizer_step = global_batch_size * accumulate_grad_batches
    train_batches_per_pass = float(train_size) / float(global_batch_size)
    optimizer_steps_per_pass = float(train_size) / float(examples_per_optimizer_step)
    max_steps = max(1, int(ceil(schedule_cfg.max_passes * optimizer_steps_per_pass)))
    val_check_interval_batches = max(
        1,
        int(ceil(schedule_cfg.val_every_passes * train_batches_per_pass)),
    )
    early_stopping_patience_checks = max(
        1,
        int(
            ceil(
                schedule_cfg.early_stopping_patience_passes
                / schedule_cfg.val_every_passes
            )
        ),
    )
    return ResolvedPassFitSchedule(
        max_passes=schedule_cfg.max_passes,
        val_every_passes=schedule_cfg.val_every_passes,
        early_stopping_patience_passes=schedule_cfg.early_stopping_patience_passes,
        train_size=int(train_size),
        per_device_batch_size=batch_size,
        data_parallel_size=data_parallel_size,
        global_batch_size=global_batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        examples_per_optimizer_step=examples_per_optimizer_step,
        train_batches_per_pass=train_batches_per_pass,
        optimizer_steps_per_pass=optimizer_steps_per_pass,
        max_steps=max_steps,
        val_check_interval_batches=val_check_interval_batches,
        early_stopping_patience_checks=early_stopping_patience_checks,
    )


def apply_resolved_pass_fit_schedule(
    cfg: DictConfig,
    resolved: ResolvedPassFitSchedule,
) -> None:
    with open_dict(cfg):
        cfg.trainer.max_steps = int(resolved.max_steps)
        cfg.trainer.val_check_interval = int(resolved.val_check_interval_batches)
        cfg.trainer.check_val_every_n_epoch = None
        callbacks_cfg = cfg.get("callbacks")
        if (
            callbacks_cfg is not None
            and callbacks_cfg.get("early_stopping") is not None
        ):
            cfg.callbacks.early_stopping.patience = int(
                resolved.early_stopping_patience_checks
            )


__all__ = [
    "PassFitScheduleConfig",
    "ResolvedPassFitSchedule",
    "apply_resolved_pass_fit_schedule",
    "resolve_pass_fit_schedule",
]
