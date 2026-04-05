"""Scalar value schedules (temperature, replay-mix alpha, etc.)

Design notes
------------
A single ``ScalarScheduler`` handles all scalar annealing patterns.
``SamplingTemperatureScheduler`` and ``ReplayMixScheduler`` are thin,
validated wrappers that enforce domain-specific constraints (e.g. temperature
must be > 0, alpha must be in [0, 1)) before delegating to it.

Integration with PyTorch Lightning
-----------------------------------
Schedules are queried inside ``LightningModule`` callbacks.  The calling
module is responsible for passing the correct ``global_step`` and
``total_steps`` — typically read directly from ``self.global_step`` and
``self.trainer.estimated_stepping_batches``.  No trainer-state dataclass
is needed.
"""

from __future__ import annotations

import math
from typing import Literal, cast

ScheduleType = Literal["constant", "linear", "cosine"]


def _validate_schedule_type(value: str, field: str) -> ScheduleType:
    if value not in {"constant", "linear", "cosine"}:
        raise ValueError(f"{field} must be one of {{'constant', 'linear', 'cosine'}}, got {value!r}.")
    return value  # type: ignore[return-value]


def _resolve_progress(global_step: int, total_steps: int, hold_steps: int = 0) -> float:
    """Map (global_step, total_steps, hold_steps) → progress in [0, 1]."""
    if total_steps <= 1:
        return 1.0
    step = min(max(global_step, 0), total_steps - 1)
    hold = min(max(hold_steps, 0), total_steps - 1)
    if step < hold:
        return 0.0
    anneal_steps = total_steps - hold
    return (step - hold + 1) / anneal_steps


def _interpolate(
    schedule_type: ScheduleType,
    initial: float,
    final: float,
    progress: float,
) -> float:
    if schedule_type == "linear":
        return initial + (final - initial) * progress
    # cosine
    w = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (initial - final) * w


class ScalarScheduler:
    """
    General-purpose scalar annealing schedule.

    Parameters
    ----------
    initial_value:
        Value at step 0 (or after the hold phase ends).
    schedule_type:
        One of ``"constant"``, ``"linear"``, ``"cosine"``.
    final_value:
        Target value at the end of the schedule.  Required unless
        ``schedule_type="constant"``.
    total_steps:
        Override for the schedule horizon.  When ``None`` the caller must
        supply ``total_steps`` to :meth:`value`.
    hold_steps:
        Number of steps to keep the value fixed at ``initial_value`` before
        annealing begins.
    """

    def __init__(
        self,
        *,
        initial_value: float,
        schedule_type: ScheduleType = "constant",
        final_value: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
        field_name: str = "schedule",
    ) -> None:
        self.initial_value = float(initial_value)
        self.schedule_type = _validate_schedule_type(schedule_type, field_name)
        self.final_value = None if final_value is None else float(final_value)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.hold_steps = int(hold_steps)

        if self.schedule_type != "constant" and self.final_value is None:
            raise ValueError(f"{field_name}: final_value is required for schedule_type={schedule_type!r}.")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError(f"{field_name}: total_steps must be >= 1.")
        if self.hold_steps < 0:
            raise ValueError(f"{field_name}: hold_steps must be >= 0.")

    def value(self, global_step: int, total_steps: int | None = None) -> float:
        """
        Return the scheduled value at ``global_step``.

        Parameters
        ----------
        global_step:
            Current optimizer step (``LightningModule.global_step``).
        total_steps:
            Fallback horizon used only when ``self.total_steps`` is ``None``.
            Typically ``trainer.estimated_stepping_batches`` or
            ``trainer.max_steps``.
        """
        if self.schedule_type == "constant":
            return self.initial_value

        resolved = self.total_steps if self.total_steps is not None else total_steps
        if resolved is None:
            raise RuntimeError(
                "ScalarScheduler requires a known total_steps.  "
                "Pass total_steps to ScalarScheduler.__init__ or to .value(), "
                "or set trainer.max_steps."
            )
        assert self.final_value is not None  # guaranteed by __init__
        progress = _resolve_progress(global_step, resolved, self.hold_steps)
        return _interpolate(self.schedule_type, self.initial_value, self.final_value, progress)


class SamplingTemperatureScheduler(ScalarScheduler):
    """
    Annealing schedule for GFlowNet sampling temperature.

    Temperature must always be > 0.

    Example (Hydra / OmegaConf config)::

        sampling_temperature:
          initial_value: 2.0
          schedule_type: cosine
          final_value: 0.5
          hold_steps: 500
    """

    def __init__(
        self,
        *,
        initial_value: float,
        schedule_type: ScheduleType = "constant",
        final_value: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        if initial_value <= 0.0:
            raise ValueError("SamplingTemperatureScheduler: initial_value must be > 0.")
        if final_value is not None and final_value <= 0.0:
            raise ValueError("SamplingTemperatureScheduler: final_value must be > 0.")
        super().__init__(
            initial_value=initial_value,
            schedule_type=schedule_type,
            final_value=final_value,
            total_steps=total_steps,
            hold_steps=hold_steps,
            field_name="sampling_temperature",
        )


class ReplayMixScheduler(ScalarScheduler):
    """
    Annealing schedule for the replay buffer mixing ratio α ∈ [0, 1).

    Example (Hydra / OmegaConf config)::

        replay_mix:
          initial_value: 0.5
          schedule_type: linear
          final_value: 0.1
    """

    def __init__(
        self,
        *,
        initial_value: float,
        schedule_type: ScheduleType = "constant",
        final_value: float | None = None,
        total_steps: int | None = None,
        hold_steps: int = 0,
    ) -> None:
        if not 0.0 <= initial_value < 1.0:
            raise ValueError("ReplayMixScheduler: initial_value must be in [0, 1).")
        if final_value is not None and not 0.0 <= final_value < 1.0:
            raise ValueError("ReplayMixScheduler: final_value must be in [0, 1).")
        super().__init__(
            initial_value=initial_value,
            schedule_type=schedule_type,
            final_value=final_value,
            total_steps=total_steps,
            hold_steps=hold_steps,
            field_name="replay_mix",
        )


__all__ = [
    "ScalarScheduler",
    "SamplingTemperatureScheduler",
    "ReplayMixScheduler",
]
