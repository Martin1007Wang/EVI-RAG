from __future__ import annotations

import json
from pathlib import Path

import torch

from src.metrics.serialization import write_metrics_json, write_metrics_jsonl
from src.runs.output import (
    PredictionArtifactSettings,
    append_stage_metrics,
    write_metrics_snapshot,
    write_prediction_artifacts,
)


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


def test_write_metrics_snapshot_uses_resolved_sink_path(tmp_path: Path) -> None:
    path = write_metrics_snapshot(
        output_dir=tmp_path,
        metrics={"answer/hit@1": 0.75},
        filename="metrics_sub.json",
    )

    assert path == tmp_path / "metrics_sub.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"answer/hit@1": 0.75}


def test_append_stage_metrics_can_write_custom_train_record_kind(
    tmp_path: Path,
) -> None:
    path = append_stage_metrics(
        output_dir=tmp_path,
        stage="train",
        step=12,
        metrics={"train/loss": 1.5},
        record_kind="train_batch",
        metadata={"train_batch_index": 3},
    )

    assert path is not None
    assert path == tmp_path / "train.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["record_kind"] == "train_batch"
    assert record["metadata"] == {"train_batch_index": 3}


def test_write_prediction_artifacts_reuses_existing_model_paths(tmp_path: Path) -> None:
    model = type("_Model", (), {})()
    existing_paths = {"prompt_path": tmp_path / "existing.jsonl"}
    settings = PredictionArtifactSettings(
        enabled=True,
        output_root=tmp_path,
        dataset_scope="sub",
    )
    setattr(model, "predict_artifact_paths", existing_paths)
    setattr(model, "predict_artifact_settings_cache_key", settings.cache_key())

    paths = write_prediction_artifacts(
        model,
        settings=settings,
    )

    assert paths == existing_paths


def test_write_prediction_artifacts_refreshes_cache_when_settings_change(
    tmp_path: Path,
) -> None:
    calls: list[Path] = []

    class _Model:
        def write_prediction_artifacts(self, **kwargs):  # type: ignore[no-untyped-def]
            output_dir = Path(str(kwargs["output_dir"]))
            calls.append(output_dir)
            return {"prompt_path": output_dir / "fresh.jsonl"}

    model = _Model()
    old_settings = PredictionArtifactSettings(
        enabled=True,
        output_root=tmp_path,
        dataset_scope="full",
    )
    setattr(model, "predict_artifact_paths", {"prompt_path": tmp_path / "old.jsonl"})
    setattr(model, "predict_artifact_settings_cache_key", old_settings.cache_key())

    new_settings = PredictionArtifactSettings(
        enabled=True,
        output_root=tmp_path,
        dataset_scope="sub",
    )
    paths = write_prediction_artifacts(model, settings=new_settings)

    assert calls == [tmp_path / "rankflow" / "sub"]
    assert paths == {"prompt_path": tmp_path / "rankflow" / "sub" / "fresh.jsonl"}
    assert (
        getattr(model, "predict_artifact_settings_cache_key")
        == new_settings.cache_key()
    )
