from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.runs.rankflow import persist_outputs


def test_eval_reporter_serializes_tensor_metrics(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {"name": "webqsp-sub", "dataset_scope": "sub"},
            "run": {"name": "rankflow", "split": "test"},
        }
    )
    model = SimpleNamespace(
        get_predict_metrics=lambda: {
            "answer/hit@1": torch.tensor(0.75),
            "answer/topk": torch.tensor([1.0, 2.0]),
        }
    )

    persist_outputs(
        cfg=cfg,
        callback_metrics={},
        model=model,
        log=SimpleNamespace(
            info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None
        ),
    )

    assert json.loads((tmp_path / "metrics.json").read_text()) == {
        "answer/hit@1": 0.75,
        "answer/topk": [1.0, 2.0],
    }


def test_eval_reporter_writes_prediction_artifacts_when_enabled(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    dataset_out_dir = tmp_path / "normalized"
    dataset_out_dir.mkdir()
    questions_path = dataset_out_dir / "questions.parquet"
    questions_path.write_text("placeholder", encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "paths": {"output_dir": str(tmp_path)},
            "dataset": {
                "name": "webqsp-sub",
                "dataset_scope": "sub",
                "artifact_dir": str(artifact_root),
                "out_dir": str(dataset_out_dir),
                "paths": {
                    "entity_vocab": str(tmp_path / "entity_vocab.parquet"),
                    "relation_vocab": str(tmp_path / "relation_vocab.parquet"),
                },
            },
            "run": {
                "name": "rankflow",
                "split": "test",
                "execution_mode": "predict",
                "write_artifacts": True,
                "artifact_subdir": "rankflow",
                "artifact_name": "rankflow",
            },
        }
    )
    captured: dict[str, object] = {}

    class _DummyModel:
        def get_predict_metrics(self) -> dict[str, float]:
            return {"answer/hit@1": 0.75}

        def write_prediction_artifacts(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            output_dir = Path(str(kwargs["output_dir"]))
            return {"prompt_path": output_dir / "test.jsonl"}

    model = _DummyModel()

    persist_outputs(
        cfg=cfg,
        callback_metrics={},
        model=model,
        log=SimpleNamespace(
            info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None
        ),
    )

    assert captured["output_dir"] == artifact_root / "rankflow" / "sub"
    assert captured["questions_path"] == questions_path
    assert getattr(model, "predict_artifact_paths") == {
        "prompt_path": artifact_root / "rankflow" / "sub" / "test.jsonl"
    }
