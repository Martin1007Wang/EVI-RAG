from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_default_training_callbacks_use_logged_metric_early_stopping() -> None:
    callbacks_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "configs" / "callbacks" / "default.yaml"
    )

    assert (
        callbacks_cfg.early_stopping._target_
        == "src.callbacks.step_early_stopping.LoggedMetricEarlyStopping"
    )
    assert "train_window_size" not in callbacks_cfg.local_metrics_writer


def test_eval_config_uses_eval_callback_bundle() -> None:
    config_dir = Path(__file__).resolve().parents[2] / "configs"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="eval.yaml",
            overrides=[
                "experiment=rankflow",
                "ckpt.gflownet=/tmp/model.ckpt",
                "logger=none",
                "hydra/job_logging=stdout",
                "hydra/hydra_logging=none",
                "extras.enforce_tags=false",
                "extras.print_config=false",
            ],
        )

    assert sorted(cfg.callbacks.keys()) == ["local_metrics_writer", "rich_progress_bar"]
    assert "train_window_size" not in cfg.callbacks.local_metrics_writer
