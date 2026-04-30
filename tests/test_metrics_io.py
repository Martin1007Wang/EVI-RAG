from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.io.metrics_writer import append_stage_metrics
from src.io.serialization import write_metrics_json, write_metrics_jsonl


def test_write_metrics_json_and_jsonl(tmp_path) -> None:
    json_path = tmp_path / "metrics.json"
    jsonl_path = tmp_path / "metrics.jsonl"

    write_metrics_json(path=json_path, metrics={"score": 1.5})
    write_metrics_jsonl(
        path=jsonl_path,
        stage="test",
        metrics={"test/score": 1.5},
        step=3,
        epoch=1,
        metadata={"split": "dev"},
    )

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"score": 1.5}

    payload = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert payload == {
        "epoch": 1,
        "metadata": {"split": "dev"},
        "metrics": {"test/score": 1.5},
        "record_kind": "metrics",
        "stage": "test",
        "step": 3,
    }


def test_append_stage_metrics_uses_stage_default_file(tmp_path) -> None:
    append_stage_metrics(
        tmp_path,
        stage="train",
        step=7,
        epoch=2,
        metrics={"train/loss/total": 0.25},
        record_kind="train_step",
        metadata={"batch_idx": 4},
    )

    payload = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").strip())
    assert payload["stage"] == "train"
    assert payload["record_kind"] == "train_step"
    assert payload["metrics"] == {"train/loss/total": 0.25}
    assert payload["metadata"] == {"batch_idx": 4}
