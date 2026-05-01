from __future__ import annotations

from typing import Any

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class TemperatureSchedule:
    """
    Manages the training rollout temperature warm-up schedule.

    Temperature starts at `temperature_start` and linearly increases
    to `temperature_end` over `temperature_warmup_steps` global steps.
    """

    def __init__(
        self,
        *,
        temperature: float,
        eval_temperature: float,
        cfg: dict[str, Any] | None = None,
    ) -> None:
        cfg = dict(cfg or {})

        self.eval_temperature = eval_temperature
        self.temperature_start = float(cfg.pop("temperature_start", temperature))
        self.temperature_end = float(cfg.pop("temperature_end", temperature))
        self.temperature_warmup_steps = int(cfg.pop("temperature_warmup_steps", 0))

        if self.temperature_start <= 0.0:
            raise ValueError(
                f"temperature_start must be positive, got {self.temperature_start}."
            )
        if self.temperature_end <= 0.0:
            raise ValueError(
                f"temperature_end must be positive, got {self.temperature_end}."
            )
        if self.temperature_warmup_steps < 0:
            raise ValueError(
                f"temperature_warmup_steps must be >= 0, got {self.temperature_warmup_steps}."
            )
        if eval_temperature <= 0.0:
            raise ValueError(
                f"eval_temperature must be positive, got {eval_temperature}."
            )
        if cfg:
            log.warning("Unused temperature_cfg keys: %s", sorted(cfg))

    def current(self, global_step: int) -> float:
        """Return the rollout temperature for the given global step."""
        if self.temperature_warmup_steps <= 0:
            return self.temperature_end

        progress = min(
            max(float(global_step), 0.0) / float(self.temperature_warmup_steps), 1.0
        )
        return (
            self.temperature_start
            + (self.temperature_end - self.temperature_start) * progress
        )
