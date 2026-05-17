from __future__ import annotations

class TemperatureSchedule:
    """
    Manages the training rollout temperature warm-up schedule.

    The mainline uses a fixed training temperature and a fixed eval
    temperature. Schedules are intentionally not exposed in Hydra.
    """

    def __init__(
        self,
        *,
        temperature: float,
        eval_temperature: float,
    ) -> None:
        self.temperature = float(temperature)
        self.eval_temperature = float(eval_temperature)
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}.")
        if self.eval_temperature <= 0.0:
            raise ValueError(
                f"eval_temperature must be positive, got {self.eval_temperature}."
            )

    def current(self, global_step: int) -> float:
        del global_step
        return self.temperature
