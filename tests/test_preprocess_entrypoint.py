from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.preprocess.config import build_preprocess_filters
from src.preprocess import _run_preprocess


def _import_run_preprocess_pipeline():
    pytest.importorskip("transformers")
    from src.data.preprocess.main import run_preprocess_pipeline

    return run_preprocess_pipeline


def test_run_preprocess_requires_dataset_group() -> None:
    cfg = OmegaConf.create({"pipeline_stage": "parquet"})

    with pytest.raises(ValueError, match="Missing required config group: `dataset`"):
        _run_preprocess(cfg)


def test_run_preprocess_pipeline_rejects_removed_preprocess_keys() -> None:
    run_preprocess_pipeline = _import_run_preprocess_pipeline()
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp"},
            "pipeline_stage": "all",
            "filter": {"train": {"skip_no_question_entity": True}},
        }
    )

    with pytest.raises(ValueError, match="Removed preprocess config keys detected"):
        run_preprocess_pipeline(cfg)


def test_run_preprocess_pipeline_rejects_removed_dataset_preprocess_keys() -> None:
    run_preprocess_pipeline = _import_run_preprocess_pipeline()
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "webqsp",
                "time_relation_mode": "drop",
            },
            "pipeline_stage": "all",
        }
    )

    with pytest.raises(ValueError, match="Removed dataset preprocess config keys"):
        run_preprocess_pipeline(cfg)


def test_build_preprocess_filters_rejects_legacy_skip_no_topic_key() -> None:
    cfg = OmegaConf.create({"preprocess_filter": {"train": {"skip_no_topic": True}}})

    with pytest.raises(ValueError, match="skip_no_topic->skip_no_question_entity"):
        build_preprocess_filters(cfg)


@pytest.mark.parametrize(
    ("stage", "expected_calls"),
    [
        ("parquet", ["parquet"]),
        ("lmdb", ["lmdb"]),
        ("all", ["parquet", "lmdb"]),
    ],
)
def test_run_preprocess_pipeline_uses_pipeline_stage(
    tmp_path: Path,
    monkeypatch,
    stage: str,
    expected_calls: list[str],
) -> None:
    run_preprocess_pipeline = _import_run_preprocess_pipeline()
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp"},
            "pipeline_stage": stage,
            "out_dir": str(tmp_path / "normalized"),
            "output_dir": str(tmp_path / "materialized"),
            "embeddings_out_dir": str(tmp_path / "embeddings"),
        }
    )
    calls: list[str] = []

    monkeypatch.setattr(
        "src.data.preprocess.main.preprocess",
        lambda ctx: calls.append("parquet"),
    )
    monkeypatch.setattr(
        "src.data.preprocess.main.build_dataset",
        lambda ctx: calls.append("lmdb"),
    )

    run_preprocess_pipeline(cfg)

    assert calls == expected_calls
