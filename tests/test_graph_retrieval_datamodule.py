from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.datasets.graph_retrieval_datamodule import (
    BudgetBatchSampler,
    GraphRetrievalDataModule,
    StepDrivenTrainSampler,
)
from src.graph_runtime import TrajectoryBatch


class _DummyDataset:
    def __init__(self, split: str, size: int = 5) -> None:
        self.split = str(split)
        self.size = int(size)
        self.closed = False

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        self.closed = True

    def get_sample_stats(self, idx: int):
        del idx
        return SimpleNamespace(num_nodes=4, num_edges=6, question_tokens=8)


def _make_runtime_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        num_graphs=1,
        node_ptr=torch.tensor([0, 2], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        edge_batch=torch.tensor([0], dtype=torch.long),
        node_batch=torch.tensor([0, 0], dtype=torch.long),
        node_embeddings=torch.randn((2, 4), dtype=torch.float32),
        edge_embeddings=torch.randn((1, 4), dtype=torch.float32),
        question_emb=torch.randn((1, 4), dtype=torch.float32),
        question_ctx=torch.randn((1, 2, 4), dtype=torch.float32),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_ids=["sample"],
        questions=["q"],
        dataset_scope="sub",
    )


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


def test_graph_retrieval_datamodule_predict_stage_uses_requested_eval_split(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created_splits: list[str] = []

    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.create_graph_retrieval_dataset",
        lambda **kwargs: created_splits.append(str(kwargs["split_name"]))
        or _DummyDataset(str(kwargs["split_name"])),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.build_retrieval_dataloader",
        lambda dataset, **kwargs: {
            "split": dataset.split,
            "shuffle": kwargs["shuffle"],
            "batch_size": kwargs["batch_size"],
        },
    )

    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        eval_batch_size=7,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=False,
        prefetch_factor=None,
        persistent_workers=False,
        eval_split="validation",
    )

    datamodule.setup(stage="predict")
    loader = datamodule.predict_dataloader()

    assert created_splits == ["validation"]
    assert loader == {"split": "validation", "shuffle": False, "batch_size": 7}

    datamodule.teardown()
    assert datamodule.eval_dataset is None


def test_step_driven_train_sampler_cycles_without_epoch_boundary() -> None:
    sampler = StepDrivenTrainSampler(
        dataset_size=3,
        num_samples=7,
        shuffle=False,
        seed=13,
    )

    assert list(sampler) == [0, 1, 2, 0, 1, 2, 0]


def test_graph_retrieval_datamodule_rejects_removed_val_alias(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="eval_split must be one of"):
        GraphRetrievalDataModule(
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


def test_graph_retrieval_datamodule_train_loader_uses_step_driven_sampler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.create_graph_retrieval_dataset",
        lambda **kwargs: _DummyDataset(str(kwargs["split_name"]), size=4),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.build_retrieval_dataloader",
        lambda dataset, **kwargs: captured.update(
            {"dataset": dataset, "kwargs": kwargs}
        )
        or {"sampler": kwargs.get("sampler"), "shuffle": kwargs["shuffle"]},
    )

    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=True,
        train_num_samples=17,
        prefetch_factor=None,
        persistent_workers=False,
        eval_split="validation",
    )

    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    assert loader["shuffle"] is False
    assert captured["kwargs"]["batch_size"] == 2
    sampler = loader["sampler"]
    assert isinstance(sampler, StepDrivenTrainSampler)
    assert sampler.dataset_size == 4
    assert len(sampler) == 17


def test_graph_retrieval_datamodule_train_loader_uses_budget_batch_sampler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.create_graph_retrieval_dataset",
        lambda **kwargs: _DummyDataset(str(kwargs["split_name"]), size=4),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.build_retrieval_dataloader",
        lambda dataset, **kwargs: captured.update(
            {"dataset": dataset, "kwargs": kwargs}
        )
        or {"batch_sampler": kwargs.get("batch_sampler")},
    )

    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=True,
        train_num_samples=17,
        prefetch_factor=None,
        persistent_workers=False,
        train_max_edges_per_batch=12,
    )

    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    batch_sampler = loader["batch_sampler"]
    assert isinstance(batch_sampler, BudgetBatchSampler)
    assert captured["kwargs"]["sampler"] is not None


def test_transfer_batch_to_device_casts_runtime_features_when_configured(
    tmp_path: Path,
) -> None:
    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=False,
        prefetch_factor=None,
        persistent_workers=False,
        eval_feature_dtype="bf16",
    )

    transferred = datamodule.transfer_batch_to_device(
        _make_runtime_batch(),
        device=torch.device("cpu"),
        dataloader_idx=0,
    )

    assert transferred.node_embeddings.dtype == torch.bfloat16
    assert transferred.edge_embeddings.dtype == torch.bfloat16
    assert transferred.question_emb.dtype == torch.bfloat16
    assert transferred.question_ctx.dtype == torch.bfloat16
    assert transferred.question_ctx_mask.dtype == torch.bool
