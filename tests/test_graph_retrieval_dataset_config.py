from __future__ import annotations

from pathlib import Path

import pytest

from src.datasets import graph_retrieval_dataset as dataset_module


def _base_dataset_cfg(tmp_path: Path) -> dict[str, object]:
    embeddings_dir = tmp_path / "embeddings"
    processed_dir = tmp_path / "processed"
    entity_vocab = tmp_path / "entity_vocab.parquet"
    embeddings_dir.mkdir()
    processed_dir.mkdir()
    entity_vocab.write_text("entity", encoding="utf-8")
    return {
        "paths": {
            "embeddings": str(embeddings_dir),
            "processed": str(processed_dir),
            "entity_vocab": str(entity_vocab),
        },
    }


def test_create_graph_retrieval_dataset_uses_runtime_filter_missing_keys(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _base_dataset_cfg(tmp_path)
    processed_dir = Path(cfg["paths"]["processed"])  # type: ignore[index]
    (processed_dir / "filter_missing_start.json").write_text("{}", encoding="utf-8")
    cfg["runtime_filter_missing_start"] = {"train": True}
    cfg["runtime_filter_missing_answer"] = {"train": False}
    captured: dict[str, object] = {}

    class _DummyDataset:
        def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
            captured.update(kwargs)
            self.sample_ids: list[str] = []

    monkeypatch.setattr(dataset_module, "GraphRetrievalDataset", _DummyDataset)

    dataset_module.create_graph_retrieval_dataset(cfg, "train")

    sample_filter_path = captured["sample_filter_path"]
    assert sample_filter_path == [processed_dir / "filter_missing_start.json"]


def test_create_graph_retrieval_dataset_rejects_renamed_filter_keys(
    tmp_path: Path,
) -> None:
    cfg = _base_dataset_cfg(tmp_path)
    cfg["filter_missing_start"] = {"train": True}

    with pytest.raises(ValueError, match="Renamed dataset config keys detected"):
        dataset_module.create_graph_retrieval_dataset(cfg, "train")
