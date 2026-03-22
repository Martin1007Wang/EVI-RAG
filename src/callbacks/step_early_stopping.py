from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from lightning.pytorch.callbacks import EarlyStopping


class LoggedMetricEarlyStopping(EarlyStopping):
    """Use logged metrics as a fallback when step-driven validation skips callback reductions."""

    def on_validation_end(self, trainer, pl_module) -> None:
        monitor = self.monitor
        callback_metrics = getattr(trainer, "callback_metrics", None)
        logged_metrics = getattr(trainer, "logged_metrics", None)
        if (
            isinstance(callback_metrics, MutableMapping)
            and monitor not in callback_metrics
            and isinstance(logged_metrics, Mapping)
        ):
            step_value = logged_metrics.get(monitor)
            if step_value is not None:
                callback_metrics[monitor] = step_value
        super().on_validation_end(trainer, pl_module)


StepEarlyStopping = LoggedMetricEarlyStopping


__all__ = ["LoggedMetricEarlyStopping", "StepEarlyStopping"]
