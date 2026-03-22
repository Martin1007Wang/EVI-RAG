from __future__ import annotations

from types import SimpleNamespace

from lightning.pytorch.callbacks import EarlyStopping

from src.callbacks.step_early_stopping import (
    LoggedMetricEarlyStopping,
    StepEarlyStopping,
)


def test_logged_metric_early_stopping_promotes_logged_metric(monkeypatch) -> None:
    observed: dict[str, float] = {}

    def _capture_super(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        del self, pl_module
        observed.update(trainer.callback_metrics)

    monkeypatch.setattr(EarlyStopping, "on_validation_end", _capture_super)
    callback = LoggedMetricEarlyStopping(monitor="val/sub/answer/recall@10", mode="max")
    trainer = SimpleNamespace(
        callback_metrics={},
        logged_metrics={"val/sub/answer/recall@10": 0.5},
    )

    callback.on_validation_end(trainer, object())

    assert trainer.callback_metrics["val/sub/answer/recall@10"] == 0.5
    assert observed["val/sub/answer/recall@10"] == 0.5


def test_step_early_stopping_alias_preserves_existing_callback_metric(
    monkeypatch,
) -> None:
    observed: dict[str, float] = {}

    def _capture_super(self, trainer, pl_module) -> None:  # type: ignore[no-untyped-def]
        del self, pl_module
        observed.update(trainer.callback_metrics)

    monkeypatch.setattr(EarlyStopping, "on_validation_end", _capture_super)
    callback = StepEarlyStopping(monitor="val/sub/answer/recall@10", mode="max")
    trainer = SimpleNamespace(
        callback_metrics={"val/sub/answer/recall@10": 0.8},
        logged_metrics={"val/sub/answer/recall@10": 0.5},
    )

    callback.on_validation_end(trainer, object())

    assert trainer.callback_metrics["val/sub/answer/recall@10"] == 0.8
    assert observed["val/sub/answer/recall@10"] == 0.8


def test_step_early_stopping_alias_points_to_logged_metric_class() -> None:
    assert StepEarlyStopping is LoggedMetricEarlyStopping
