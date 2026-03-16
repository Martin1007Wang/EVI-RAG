from __future__ import annotations

import json
from pathlib import Path

import torch

from src.utils.metrics_io import write_metrics_json, write_metrics_jsonl


def test_write_metrics_json_serializes_nested_tensor_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    written_path = write_metrics_json(
        path=path,
        metrics={
            "scalar": torch.tensor(1.5),
            "vector": torch.tensor([1, 2]),
            "nested": {"items": (torch.tensor(3.0), "ok")},
        },
    )

    assert written_path == path
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "scalar": 1.5,
        "vector": [1, 2],
        "nested": {"items": [3.0, "ok"]},
    }


def test_write_metrics_jsonl_serializes_metrics_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"

    write_metrics_jsonl(
        path=path,
        stage="predict",
        metrics={"predict/hit@1": torch.tensor(0.5)},
        step=7,
        metadata={
            "source": Path("artifact.json"),
            "flags": [True, torch.tensor(2.0)],
        },
    )

    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 1
    assert records[0]["stage"] == "predict"
    assert records[0]["step"] == 7
    assert records[0]["metrics"] == {"predict/hit@1": 0.5}
    assert records[0]["metadata"] == {
        "source": "artifact.json",
        "flags": [True, 2.0],
    }
