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


def test_local_metrics_writer_logs_raw_snapshots_at_logger_cadence(
    tmp_path: Path,
) -> None:
    writer = LocalMetricsWriter(output_dir=tmp_path, enabled=True)
    trainer = SimpleNamespace(
        global_step=0,
        current_epoch=3,
        is_global_zero=True,
        log_every_n_steps=2,
    )
    records = [
        {"train/loss": 4.0, "train/effective_pass": 0.5},
        {"train/loss": 2.0, "train/effective_pass": 1.0},
        {"train/loss": 1.0, "train/effective_pass": 1.5},
    ]

    def _pop_latest_train_metrics() -> dict[str, float] | None:
        return records.pop(0) if records else None

    model = SimpleNamespace(pop_latest_train_metrics=_pop_latest_train_metrics)

    writer.on_fit_start(trainer, model)
    for batch_idx, global_step in enumerate((1, 2, 3)):
        trainer.global_step = global_step
        writer.on_train_batch_end(trainer, model, None, object(), batch_idx)
    writer.on_train_end(trainer, model)

    written_records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert written_records == [
        {
            "stage": "train",
            "epoch": 3,
            "step": 2,
            "timestamp": written_records[0]["timestamp"],
            "record_kind": "train_step",
            "metrics": {"train/effective_pass": 1.0, "train/loss": 2.0},
            "metadata": {"batch_idx": 1},
        },
        {
            "stage": "train",
            "epoch": 3,
            "step": 3,
            "timestamp": written_records[1]["timestamp"],
            "record_kind": "train_step",
            "metrics": {"train/effective_pass": 1.5, "train/loss": 1.0},
            "metadata": {"batch_idx": 2},
        },
    ]
    assert not (tmp_path / "train.summary.jsonl").exists()


def test_local_metrics_writer_deduplicates_repeated_global_steps(
    tmp_path: Path,
) -> None:
    writer = LocalMetricsWriter(output_dir=tmp_path, enabled=True)
    trainer = SimpleNamespace(
        global_step=0,
        current_epoch=1,
        is_global_zero=True,
        log_every_n_steps=1,
    )
    records = [
        {"train/loss": 4.0},
        {"train/loss": 2.0},
        {"train/loss": 1.0},
    ]

    def _pop_latest_train_metrics() -> dict[str, float] | None:
        return records.pop(0) if records else None

    model = SimpleNamespace(pop_latest_train_metrics=_pop_latest_train_metrics)

    writer.on_fit_start(trainer, model)
    trainer.global_step = 1
    writer.on_train_batch_end(trainer, model, None, object(), 0)
    trainer.global_step = 1
    writer.on_train_batch_end(trainer, model, None, object(), 1)
    trainer.global_step = 2
    writer.on_train_batch_end(trainer, model, None, object(), 2)
    writer.on_train_end(trainer, model)

    written_records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in written_records] == [1, 2]
    assert [record["metrics"]["train/loss"] for record in written_records] == [2.0, 1.0]
    assert [record["metadata"]["batch_idx"] for record in written_records] == [1, 2]
