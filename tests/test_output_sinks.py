from __future__ import annotations

import json
from pathlib import Path

from src.utils.output_sinks import (
    MetricsSnapshotSettings,
    PredictionArtifactSettings,
    StageMetricsSettings,
    append_stage_metrics,
    write_metrics_snapshot,
    write_prediction_artifacts,
)


def test_write_metrics_snapshot_uses_resolved_sink_path(tmp_path: Path) -> None:
    path = write_metrics_snapshot(
        metrics={"answer/hit@1": 0.75},
        settings=MetricsSnapshotSettings(
            output_dir=tmp_path, filename="metrics_sub.json"
        ),
    )

    assert path == tmp_path / "metrics_sub.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"answer/hit@1": 0.75}


def test_append_stage_metrics_uses_default_stage_file(tmp_path: Path) -> None:
    path = append_stage_metrics(
        metrics={"answer/hit@1": 0.5},
        settings=StageMetricsSettings(output_dir=tmp_path, stage="predict", step=7),
    )

    assert path == tmp_path / "predict.jsonl"
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["stage"] == "predict"
    assert record["step"] == 7
    assert record["metrics"] == {"answer/hit@1": 0.5}


def test_write_prediction_artifacts_reuses_existing_model_paths(tmp_path: Path) -> None:
    model = type("_Model", (), {})()
    existing_paths = {"prompt_path": tmp_path / "existing.jsonl"}
    setattr(model, "predict_artifact_paths", existing_paths)

    paths = write_prediction_artifacts(
        model,
        settings=PredictionArtifactSettings(enabled=True, output_root=tmp_path),
    )

    assert paths == existing_paths
