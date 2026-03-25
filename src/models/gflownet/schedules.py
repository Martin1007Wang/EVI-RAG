from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.models.configs import SamplingTemperatureScheduleConfig


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
        config: SamplingTemperatureScheduleConfig,
    ) -> None:
        self.base_temperature = float(base_temperature)
        self.config = config

    def value(
        self,
        *,
        global_step: int,
        schedule_context: TrainingScheduleContext,
    ) -> float:
        initial_temperature = (
            self.base_temperature
            if self.config.initial_temperature is None
            else float(self.config.initial_temperature)
        )
        if self.config.type == "constant":
            return initial_temperature

        total_steps = schedule_context.resolve_horizon(
            explicit_horizon=self.config.total_steps,
            interval="step",
        )
        if total_steps is None:
            raise RuntimeError(
                "sampling temperature schedule requires a known step horizon. "
                "Set trainer.max_steps, ensure estimated_stepping_batches is available, "
                "or configure training.sampling_temperature_schedule.total_steps explicitly."
            )

        progress = self._progress(global_step=global_step, total_steps=total_steps)
        final_temperature_value = self.config.final_temperature
        if final_temperature_value is None:
            raise RuntimeError(
                "sampling temperature schedule requires final_temperature for annealed "
                "schedules."
            )
        final_temperature = float(final_temperature_value)
        if self.config.type == "linear":
            return (
                initial_temperature
                + (final_temperature - initial_temperature) * progress
            )
        if self.config.type == "cosine":
            cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                final_temperature
                + (initial_temperature - final_temperature) * cosine_weight
            )
        raise ValueError(
            f"Unsupported sampling temperature schedule type: {self.config.type!r}."
        )

    @staticmethod
    def _progress(*, global_step: int, total_steps: int) -> float:
        if total_steps <= 1:
            return 1.0
        clipped_step = min(max(int(global_step), 0), int(total_steps) - 1)
        return float(clipped_step) / float(int(total_steps) - 1)


__all__ = [
    "SamplingTemperatureScheduler",
    "TrainingScheduleContext",
    "normalize_scheduler_interval",
]
