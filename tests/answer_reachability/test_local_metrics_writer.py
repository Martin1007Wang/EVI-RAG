from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.callbacks.local_metrics_writer import LocalMetricsWriter


def test_local_metrics_writer_uses_model_predict_metrics(tmp_path: Path) -> None:
    writer = LocalMetricsWriter(output_dir=tmp_path, enabled=True)
    trainer = SimpleNamespace(global_step=7, is_global_zero=True)
    model = SimpleNamespace(get_predict_metrics=lambda: {"answer/hit@1": 0.75})

    writer.on_predict_end(trainer, model)

    record = json.loads(
        (tmp_path / "predict.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["stage"] == "predict"
    assert record["step"] == 7
    assert record["metrics"] == {"answer/hit@1": 0.75}


def test_local_metrics_writer_skips_non_global_zero_predict_writes(
    tmp_path: Path,
) -> None:
    writer = LocalMetricsWriter(output_dir=tmp_path, enabled=True)
    trainer = SimpleNamespace(global_step=7, is_global_zero=False)
    model = SimpleNamespace(get_predict_metrics=lambda: {"answer/hit@1": 0.75})

    writer.on_predict_end(trainer, model)

    assert not (tmp_path / "predict.jsonl").exists()


def test_local_metrics_writer_logs_current_batch_and_flushes_tail(
    tmp_path: Path,
) -> None:
    writer = LocalMetricsWriter(output_dir=tmp_path, enabled=True, train_window_size=2)
    trainer = SimpleNamespace(
        global_step=0,
        current_epoch=3,
        log_every_n_steps=2,
        is_global_zero=True,
        callback_metrics={},
        logged_metrics={},
    )

    writer.on_fit_start(trainer, object())

    trainer.global_step = 1
    trainer.callback_metrics = {"train/loss": 4.0}
    writer.on_train_batch_end(trainer, object(), None, None, 0)

    trainer.global_step = 2
    trainer.callback_metrics = {"train/loss": 2.0}
    writer.on_train_batch_end(trainer, object(), None, None, 1)

    trainer.global_step = 3
    trainer.callback_metrics = {"train/loss": 1.0}
    writer.on_train_batch_end(trainer, object(), None, None, 2)
    writer.on_train_end(trainer, object())

    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "stage": "train",
            "epoch": 3,
            "step": 2,
            "timestamp": records[0]["timestamp"],
            "metrics": {"train/loss": 3.0},
        },
        {
            "stage": "train",
            "epoch": 3,
            "step": 3,
            "timestamp": records[1]["timestamp"],
            "metrics": {"train/loss": 1.5},
        },
    ]
