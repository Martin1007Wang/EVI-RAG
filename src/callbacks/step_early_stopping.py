from __future__ import annotations

from lightning.pytorch.callbacks import EarlyStopping


class StepEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module) -> None:
        monitor = self.monitor
        if monitor not in trainer.callback_metrics:
            step_value = trainer.logged_metrics.get(monitor)
            if step_value is not None:
                trainer.callback_metrics[monitor] = step_value
        super().on_validation_end(trainer, pl_module)


__all__ = ["StepEarlyStopping"]
