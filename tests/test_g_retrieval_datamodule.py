from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.datasets.g_retrieval_datamodule import GRetrievalDataModule


class _DummyDataset:
    def __init__(self, split: str) -> None:
        self.split = str(split)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _write_dataset_paths(tmp_path: Path) -> dict[str, object]:
    entity_vocab = tmp_path / "entity_vocab.parquet"
    relation_vocab = tmp_path / "relation_vocab.parquet"
    embeddings_dir = tmp_path / "embeddings"
    entity_vocab.write_text("entity", encoding="utf-8")
    relation_vocab.write_text("relation", encoding="utf-8")
    embeddings_dir.mkdir()
    return {
        "name": "webqsp-sub",
        "dataset_scope": "sub",
        "paths": {
            "entity_vocab": str(entity_vocab),
            "relation_vocab": str(relation_vocab),
            "embeddings": str(embeddings_dir),
        },
    }


def test_g_retrieval_datamodule_predict_stage_uses_requested_eval_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_splits: list[str] = []

    monkeypatch.setattr(
        "src.datasets.g_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.g_retrieval_datamodule.create_g_retrieval_dataset",
        lambda **kwargs: created_splits.append(str(kwargs["split_name"]))
        or _DummyDataset(str(kwargs["split_name"])),
    )
    monkeypatch.setattr(
        "src.datasets.g_retrieval_datamodule.build_retrieval_dataloader",
        lambda dataset, **kwargs: {
            "split": dataset.split,
            "shuffle": kwargs["shuffle"],
        },
    )

    datamodule = GRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=False,
        prefetch_factor=None,
        persistent_workers=False,
        eval_split="val",
    )

    datamodule.setup(stage="predict")
    loader = datamodule.predict_dataloader()

    assert created_splits == ["validation"]
    assert loader == {"split": "validation", "shuffle": False}

    datamodule.teardown()
    assert datamodule.eval_dataset is None
