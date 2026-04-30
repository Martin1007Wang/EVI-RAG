from __future__ import annotations

import math
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
            raise ValueError(f"temperature_start must be positive, got {self.temperature_start}.")
        if self.temperature_end <= 0.0:
            raise ValueError(f"temperature_end must be positive, got {self.temperature_end}.")
        if self.temperature_warmup_steps < 0:
            raise ValueError(f"temperature_warmup_steps must be >= 0, got {self.temperature_warmup_steps}.")
        if eval_temperature <= 0.0:
            raise ValueError(f"eval_temperature must be positive, got {eval_temperature}.")
        if cfg:
            log.warning("Unused temperature_cfg keys: %s", sorted(cfg))

    def current(self, global_step: int) -> float:
        """Return the rollout temperature for the given global step."""
        if self.temperature_warmup_steps <= 0:
            return self.temperature_end

        progress = min(max(float(global_step), 0.0) / float(self.temperature_warmup_steps), 1.0)
        return self.temperature_start + (self.temperature_end - self.temperature_start) * progress


class ProposalSchedule:
    """
    Manages the proposal intervention probability cosine-decay schedule.

    The clean training regime is:

        coverage proposal warmup / decay → strict target-policy fine-tuning

    `proposal_final_prob` defaults to 0.0. A non-zero final probability is
    only allowed when `allow_nonzero_final_prob=True` is set explicitly for
    ablation experiments.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})

        self.enabled = bool(cfg.pop("enabled", True))
        self.warmup_steps = int(cfg.pop("warmup_steps", 0))
        self.decay_steps = int(cfg.pop("decay_steps", 0))
        self.initial_prob = float(cfg.pop("initial_prob", 0.95))
        self.final_prob = float(cfg.pop("final_prob", 0.0))
        allow_nonzero_final = bool(cfg.pop("allow_nonzero_final_prob", False))

        if self.warmup_steps < 0:
            raise ValueError(f"proposal warmup_steps must be >= 0, got {self.warmup_steps}.")
        if self.decay_steps < 0:
            raise ValueError(f"proposal decay_steps must be >= 0, got {self.decay_steps}.")
        if not 0.0 <= self.initial_prob <= 1.0:
            raise ValueError(f"proposal initial_prob must be in [0, 1], got {self.initial_prob}.")
        if not 0.0 <= self.final_prob <= 1.0:
            raise ValueError(f"proposal final_prob must be in [0, 1], got {self.final_prob}.")
        if self.final_prob != 0.0 and not allow_nonzero_final:
            raise ValueError(
                "proposal final_prob must be 0.0 for the clean "
                "coverage-warmup -> target-policy-finetune regime. Set "
                "allow_nonzero_final_prob=True only for explicit ablations."
            )
        if not self.enabled:
            self.initial_prob = 0.0
            self.final_prob = 0.0
        if cfg:
            log.warning("Unused proposal_cfg keys: %s", sorted(cfg))

    def current(self, global_step: int) -> float:
        """Return the proposal intervention probability for the given global step."""
        if not self.enabled:
            return 0.0

        step = max(float(global_step), 0.0)

        if step < self.warmup_steps:
            return self.initial_prob

        if self.decay_steps <= 0:
            return self.final_prob

        progress = min((step - self.warmup_steps) / float(self.decay_steps), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.final_prob + (self.initial_prob - self.final_prob) * cosine_decay
