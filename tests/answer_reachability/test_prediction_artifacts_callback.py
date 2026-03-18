from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.callbacks.local_metrics_writer import LocalMetricsWriter
from src.callbacks.prediction_artifacts_writer import PredictionArtifactsWriter


def test_prediction_artifacts_writer_delegates_to_model_method(tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    questions_path = tmp_path / "questions.parquet"
    questions_path.write_text("placeholder", encoding="utf-8")

    class _DummyModel:
        def write_prediction_artifacts(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            output_dir = Path(str(kwargs["output_dir"]))
            return {"prompt_path": output_dir / "test.jsonl"}

    callback = PredictionArtifactsWriter(
        enabled=True,
        execution_mode="predict",
        output_root=tmp_path,
        artifact_subdir="rankflow",
        artifact_name="rankflow",
        schema_version=1,
        split="test",
        dataset_scope="sub",
        dataset_out_dir=tmp_path,
        overwrite=True,
    )
    model = _DummyModel()

    callback.on_predict_end(SimpleNamespace(is_global_zero=True), model)

    assert captured["split"] == "test"
    assert captured["output_dir"] == tmp_path / "rankflow" / "sub"
    assert captured["questions_path"] == questions_path
    assert getattr(model, "predict_artifact_paths") == {
        "prompt_path": tmp_path / "rankflow" / "sub" / "test.jsonl"
    }


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


def test_prediction_artifacts_writer_rejects_unknown_execution_mode(
    tmp_path: Path,
) -> None:
    try:
        PredictionArtifactsWriter(
            enabled=True,
            execution_mode="invalid",
            output_root=tmp_path,
        )
    except ValueError as exc:
        assert "execution mode" in str(exc)
    else:
        raise AssertionError("expected invalid execution mode to be rejected")
