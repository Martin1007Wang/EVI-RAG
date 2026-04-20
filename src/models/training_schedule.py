from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumScheduleConfig:
    warmup_steps: int = 0
    decay_steps: int = 0
    initial_teacher_prob: float = 1.0
    final_teacher_prob: float = 0.0

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError(
                f"schedule.warmup_steps must be >= 0, got {self.warmup_steps}."
            )
        if self.decay_steps < 0:
            raise ValueError(
                f"schedule.decay_steps must be >= 0, got {self.decay_steps}."
            )
        for name, value in (
            ("initial_teacher_prob", self.initial_teacher_prob),
            ("final_teacher_prob", self.final_teacher_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"schedule.{name} must be in [0, 1], got {value}.")
        if self.final_teacher_prob > self.initial_teacher_prob:
            raise ValueError(
                "schedule.final_teacher_prob must be <= initial_teacher_prob, got "
                f"{self.final_teacher_prob} > {self.initial_teacher_prob}."
            )


class CurriculumSchedule:
    def __init__(
        self,
        *,
        warmup_steps: int,
        decay_steps: int,
        initial_teacher_prob: float,
        final_teacher_prob: float,
    ) -> None:
        self.warmup_steps = int(warmup_steps)
        self.decay_steps = int(decay_steps)
        self.initial_teacher_prob = float(initial_teacher_prob)
        self.final_teacher_prob = float(final_teacher_prob)

    def teacher_force_prob(self, global_step: int) -> float:
        step = max(int(global_step), 0)
        if step < self.warmup_steps:
            return self.initial_teacher_prob
        if self.decay_steps == 0:
            return self.final_teacher_prob
        progress = min(float(step - self.warmup_steps) / float(self.decay_steps), 1.0)
        return self.initial_teacher_prob + (
            self.final_teacher_prob - self.initial_teacher_prob
        ) * progress

    def phase(self, global_step: int) -> str:
        step = max(int(global_step), 0)
        if step < self.warmup_steps and self.initial_teacher_prob > 0.0:
            return "warmup"
        if self.teacher_force_prob(step) > 0.0:
            return "mix"
        return "online"


__all__ = ["CurriculumSchedule", "CurriculumScheduleConfig"]
