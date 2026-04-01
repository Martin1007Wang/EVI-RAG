from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def normalize_scheduler_interval(scheduler_cfg: dict[str, Any]) -> str:
    interval = str(scheduler_cfg.get("interval", "step")).lower()
    if interval not in {"step", "epoch"}:
        raise ValueError(
            f"Unsupported scheduler interval: {interval!r}. Expected 'step' or 'epoch'."
        )
    return interval


@dataclass(frozen=True)
class TrainingScheduleContext:
    estimated_stepping_batches: int | None
    trainer_max_steps: int | None = None
    trainer_max_epochs: int | None = None

    @staticmethod
    def _normalize_explicit_horizon(explicit_horizon: int | None) -> int | None:
        if explicit_horizon is None:
            return None
        horizon = int(explicit_horizon)
        if horizon <= 0:
            raise ValueError(f"scheduler requires t_max > 0, got {horizon}.")
        return horizon

    def resolve_horizon(
        self,
        *,
        explicit_horizon: int | None,
        interval: str,
    ) -> int | None:
        horizon = self._normalize_explicit_horizon(explicit_horizon)
        if horizon is None:
            if interval == "step":
                horizon = self.configured_training_steps()
            else:
                horizon = self.trainer_max_epochs
        if horizon is None:
            return None
        if int(horizon) <= 0:
            raise ValueError(f"Scheduler horizon must be > 0, got {horizon}.")
        return int(horizon)

    def configured_training_steps(self) -> int | None:
        if self.trainer_max_steps is not None:
            return int(self.trainer_max_steps)
        if self.estimated_stepping_batches is None:
            return None
        return int(self.estimated_stepping_batches)


class SamplingTemperatureScheduler:
    def __init__(
        self,
        *,
        base_temperature: float,
        type: str = "constant",
        initial_temperature: float | None = None,
        final_temperature: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        self.base_temperature = float(base_temperature)
        schedule_type = str(type)
        _validate_schedule_type(
            schedule_type=schedule_type,
            field_name="training.sampling_temperature_schedule",
        )
        if initial_temperature is not None and float(initial_temperature) <= 0.0:
            raise ValueError(
                "training.sampling_temperature_schedule.initial_temperature must be > 0."
            )
        if final_temperature is not None and float(final_temperature) <= 0.0:
            raise ValueError(
                "training.sampling_temperature_schedule.final_temperature must be > 0."
            )
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError(
                "training.sampling_temperature_schedule.total_steps must be >= 1."
            )
        if int(hold_steps) < 0:
            raise ValueError(
                "training.sampling_temperature_schedule.hold_steps must be >= 0."
            )
        if schedule_type != "constant" and final_temperature is None:
            raise ValueError(
                "training.sampling_temperature_schedule.final_temperature must be set "
                "for annealed schedules."
            )
        self.schedule_type = schedule_type
        self.initial_temperature = (
            None if initial_temperature is None else float(initial_temperature)
        )
        self.final_temperature = (
            None if final_temperature is None else float(final_temperature)
        )
        self.total_steps = None if total_steps is None else int(total_steps)
        self.hold_steps = int(hold_steps)

    def value(
        self,
        *,
        global_step: int,
        schedule_context: TrainingScheduleContext,
    ) -> float:
        initial_temperature = (
            self.base_temperature
            if self.initial_temperature is None
            else float(self.initial_temperature)
        )
        if self.schedule_type == "constant":
            return initial_temperature

        total_steps = schedule_context.resolve_horizon(
            explicit_horizon=self.total_steps,
            interval="step",
        )
        if total_steps is None:
            raise RuntimeError(
                "sampling temperature schedule requires a known step horizon. "
                "Set trainer.max_steps, ensure estimated_stepping_batches is available, "
                "or configure training.sampling_temperature_schedule.total_steps explicitly."
            )

        progress = self._progress(
            global_step=global_step,
            total_steps=total_steps,
            hold_steps=self.hold_steps,
        )
        final_temperature_value = self.final_temperature
        if final_temperature_value is None:
            raise RuntimeError(
                "sampling temperature schedule requires final_temperature for annealed "
                "schedules."
            )
        return _interpolate_scalar_schedule(
            schedule_type=self.schedule_type,
            initial_value=initial_temperature,
            final_value=float(final_temperature_value),
            progress=progress,
        )

    @staticmethod
    def _progress(*, global_step: int, total_steps: int, hold_steps: int = 0) -> float:
        if total_steps <= 1:
            return 1.0
        clipped_step = min(max(int(global_step), 0), int(total_steps) - 1)
        effective_hold_steps = min(max(int(hold_steps), 0), int(total_steps) - 1)
        if effective_hold_steps == 0:
            return float(clipped_step) / float(int(total_steps) - 1)
        if clipped_step < effective_hold_steps:
            return 0.0
        anneal_steps = int(total_steps) - effective_hold_steps
        return float(clipped_step - effective_hold_steps + 1) / float(anneal_steps)


def _interpolate_scalar_schedule(
    *,
    schedule_type: str,
    initial_value: float,
    final_value: float,
    progress: float,
) -> float:
    if schedule_type == "linear":
        return initial_value + (final_value - initial_value) * progress
    if schedule_type == "cosine":
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_value + (initial_value - final_value) * cosine_weight
    raise ValueError(f"Unsupported scalar schedule type: {schedule_type!r}.")


class ProposalBiasScheduler:
    def __init__(
        self,
        *,
        base_scale: float,
        type: str = "constant",
        initial_scale: float | None = None,
        final_scale: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        self.base_scale = float(base_scale)
        schedule_type = str(type)
        _validate_schedule_type(
            schedule_type=schedule_type,
            field_name="training.auxiliary.proposal.schedule",
        )
        if initial_scale is not None and float(initial_scale) < 0.0:
            raise ValueError(
                "training.auxiliary.proposal.schedule.initial_scale must be >= 0."
            )
        if final_scale is not None and float(final_scale) < 0.0:
            raise ValueError(
                "training.auxiliary.proposal.schedule.final_scale must be >= 0."
            )
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError(
                "training.auxiliary.proposal.schedule.total_steps must be >= 1."
            )
        if int(hold_steps) < 0:
            raise ValueError(
                "training.auxiliary.proposal.schedule.hold_steps must be >= 0."
            )
        if schedule_type != "constant" and final_scale is None:
            raise ValueError(
                "training.auxiliary.proposal.schedule.final_scale must be set for annealed schedules."
            )
        self.schedule_type = schedule_type
        self.initial_scale = None if initial_scale is None else float(initial_scale)
        self.final_scale = None if final_scale is None else float(final_scale)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.hold_steps = int(hold_steps)

    def value(
        self,
        *,
        global_step: int,
        schedule_context: TrainingScheduleContext,
    ) -> float:
        initial_scale = (
            self.base_scale if self.initial_scale is None else float(self.initial_scale)
        )
        if self.schedule_type == "constant":
            return initial_scale

        total_steps = schedule_context.resolve_horizon(
            explicit_horizon=self.total_steps,
            interval="step",
        )
        if total_steps is None:
            raise RuntimeError(
                "proposal-bias schedule requires a known step horizon. Set trainer.max_steps, "
                "ensure estimated_stepping_batches is available, or configure "
                "training.auxiliary.proposal.schedule.total_steps explicitly."
            )

        final_scale_value = self.final_scale
        if final_scale_value is None:
            raise RuntimeError(
                "proposal-bias schedule requires final_scale for annealed schedules."
            )
        progress = SamplingTemperatureScheduler._progress(
            global_step=global_step,
            total_steps=total_steps,
            hold_steps=self.hold_steps,
        )
        return _interpolate_scalar_schedule(
            schedule_type=self.schedule_type,
            initial_value=initial_scale,
            final_value=float(final_scale_value),
            progress=progress,
        )


class ReplayMixScheduler:
    def __init__(
        self,
        *,
        base_alpha: float,
        type: str = "constant",
        initial_alpha: float | None = None,
        final_alpha: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        self.base_alpha = float(base_alpha)
        schedule_type = str(type)
        _validate_schedule_type(
            schedule_type=schedule_type,
            field_name="training.auxiliary.replay.schedule",
        )
        if initial_alpha is not None and not 0.0 <= float(initial_alpha) < 1.0:
            raise ValueError(
                "training.auxiliary.replay.schedule.initial_alpha must be in [0, 1)."
            )
        if final_alpha is not None and not 0.0 <= float(final_alpha) < 1.0:
            raise ValueError(
                "training.auxiliary.replay.schedule.final_alpha must be in [0, 1)."
            )
        if total_steps is not None and int(total_steps) < 1:
            raise ValueError(
                "training.auxiliary.replay.schedule.total_steps must be >= 1."
            )
        if int(hold_steps) < 0:
            raise ValueError(
                "training.auxiliary.replay.schedule.hold_steps must be >= 0."
            )
        if schedule_type != "constant" and final_alpha is None:
            raise ValueError(
                "training.auxiliary.replay.schedule.final_alpha must be set for annealed schedules."
            )
        self.schedule_type = schedule_type
        self.initial_alpha = None if initial_alpha is None else float(initial_alpha)
        self.final_alpha = None if final_alpha is None else float(final_alpha)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.hold_steps = int(hold_steps)

    def value(
        self,
        *,
        global_step: int,
        schedule_context: TrainingScheduleContext,
    ) -> float:
        initial_alpha = (
            self.base_alpha if self.initial_alpha is None else float(self.initial_alpha)
        )
        if self.schedule_type == "constant":
            return initial_alpha

        total_steps = schedule_context.resolve_horizon(
            explicit_horizon=self.total_steps,
            interval="step",
        )
        if total_steps is None:
            raise RuntimeError(
                "replay-mix schedule requires a known step horizon. Set trainer.max_steps, "
                "ensure estimated_stepping_batches is available, or configure "
                "training.auxiliary.replay.schedule.total_steps explicitly."
            )

        final_alpha_value = self.final_alpha
        if final_alpha_value is None:
            raise RuntimeError(
                "replay-mix schedule requires final_alpha for annealed schedules."
            )
        progress = SamplingTemperatureScheduler._progress(
            global_step=global_step,
            total_steps=total_steps,
            hold_steps=self.hold_steps,
        )
        return _interpolate_scalar_schedule(
            schedule_type=self.schedule_type,
            initial_value=initial_alpha,
            final_value=float(final_alpha_value),
            progress=progress,
        )


def _validate_schedule_type(*, schedule_type: str, field_name: str) -> None:
    if schedule_type not in {"constant", "linear", "cosine"}:
        raise ValueError(
            f"{field_name}.type must be one of {{'constant', 'linear', 'cosine'}}."
        )


__all__ = [
    "ProposalBiasScheduler",
    "ReplayMixScheduler",
    "SamplingTemperatureScheduler",
    "TrainingScheduleContext",
    "normalize_scheduler_interval",
]
