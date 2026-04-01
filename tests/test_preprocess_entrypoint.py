from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.preprocess.config import build_preprocess_filters
from src.data.preprocess.main import run_preprocess_pipeline
from src.preprocess import _run_preprocess


def test_run_preprocess_requires_dataset_group() -> None:
    cfg = OmegaConf.create({"pipeline_stage": "parquet"})

    with pytest.raises(ValueError, match="Missing required config group: `dataset`"):
        _run_preprocess(cfg)


def test_run_preprocess_delegates_to_pipeline(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {"name": "webqsp"},
            "pipeline_stage": "all",
        }
    )
    seen: dict[str, object] = {}

    def _fake_run_preprocess_pipeline(current_cfg):  # type: ignore[no-untyped-def]
        seen["cfg"] = current_cfg

    monkeypatch.setattr(
        "src.preprocess._get_preprocess_runner", lambda: _fake_run_preprocess_pipeline
    )

    _run_preprocess(cfg)

    assert seen == {"cfg": cfg}


def test_build_retrieval_pipeline_uses_hydra_logging_defaults() -> None:
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "build_retrieval_pipeline.yaml"
    )
    pipeline_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs" / "pipeline" / "default.yaml"
    )
    defaults = OmegaConf.to_container(cfg.defaults, resolve=False)

    assert {"hydra": "default"} in defaults
    assert cfg.task_name == "preprocess"
    assert cfg.hf_offline is False
    assert cfg.get("filter") is None
    assert pipeline_cfg.get("preprocess_filter") is not None
    assert pipeline_cfg.overwrite_lmdb is False
    assert cfg.get("keep_start_adjacent_edges") is None
    assert cfg.get("canonicalize_relations") is None
    assert cfg.get("skip_parquet_stage") is None


def test_run_preprocess_pipeline_rejects_removed_preprocess_keys() -> None:
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
