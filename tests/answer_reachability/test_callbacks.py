from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar

from src.callbacks.local_metrics_writer import LocalMetricsWriter
from src.callbacks.step_early_stopping import (
    LoggedMetricEarlyStopping,
    StepEarlyStopping,
)
from src.runs.hydra import instantiate_callbacks


_HYDRA_TEST_OVERRIDES = [
    "logger=none",
    "hydra/job_logging=stdout",
    "hydra/hydra_logging=none",
    "extras.enforce_tags=false",
    "extras.print_config=false",
]


def test_train_rankflow_callbacks_instantiate_expected_runtime_bundle(
    tmp_path: Path,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=train_rankflow",
                "dataset=webqsp-sub",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    cfg.paths.output_dir = str(tmp_path / "train")
    callbacks = instantiate_callbacks(cfg.callbacks)

    assert any(isinstance(callback, ModelCheckpoint) for callback in callbacks)
    assert any(
        isinstance(callback, LoggedMetricEarlyStopping) for callback in callbacks
    )
    assert any(isinstance(callback, LocalMetricsWriter) for callback in callbacks)
    assert any(isinstance(callback, RichProgressBar) for callback in callbacks)


def test_rankflow_eval_callbacks_exclude_training_only_callbacks(
    tmp_path: Path,
) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=eval_rankflow",
                "ckpt.gflownet=/tmp/model.ckpt",
                *_HYDRA_TEST_OVERRIDES,
            ],
        )

    cfg.paths.output_dir = str(tmp_path / "eval")
    callbacks = instantiate_callbacks(cfg.callbacks)

    assert any(isinstance(callback, LocalMetricsWriter) for callback in callbacks)
    assert any(isinstance(callback, RichProgressBar) for callback in callbacks)
    assert not any(isinstance(callback, ModelCheckpoint) for callback in callbacks)
    assert not any(
        isinstance(callback, EarlyStopping)
        and not isinstance(callback, RichProgressBar)
        for callback in callbacks
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
