from __future__ import annotations

from itertools import islice
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.datasets.graph_retrieval_datamodule import (
    BudgetBatchSampler,
    GraphRetrievalDataModule,
    StepDrivenTrainSampler,
    _resolve_embedding_attachment_device,
)
from src.graph import TrajectoryBatch


class _DummyDataset:
    def __init__(
        self,
        split: str,
        size: int = 5,
        *,
        stats_by_idx: dict[int, SimpleNamespace] | None = None,
    ) -> None:
        self.split = str(split)
        self.size = int(size)
        self.closed = False
        self._stats_by_idx = stats_by_idx or {}

    def __len__(self) -> int:
        return self.size

    def close(self) -> None:
        self.closed = True

    def get_sample_stats(self, idx: int):
        return self._stats_by_idx.get(
            int(idx),
            SimpleNamespace(num_nodes=4, num_edges=6, question_tokens=8),
        )


def _make_runtime_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        num_graphs=1,
        node_ptr=torch.tensor([0, 2], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        edge_batch=torch.tensor([0], dtype=torch.long),
        node_batch=torch.tensor([0, 0], dtype=torch.long),
        node_embeddings=torch.randn((2, 4), dtype=torch.float32),
        edge_embeddings=None,
        question_emb=torch.randn((1, 4), dtype=torch.float32),
        question_ctx=torch.randn((1, 2, 4), dtype=torch.float32),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_ids=["sample"],
        questions=["q"],
        dataset_scope="sub",
        relation_embeddings=torch.randn((1, 4), dtype=torch.float32),
        edge_rel_local=torch.tensor([0], dtype=torch.long),
    )


def _make_pyg_batch_without_edge_batch() -> SimpleNamespace:
    return SimpleNamespace(
        num_graphs=1,
        ptr=torch.tensor([0, 2], dtype=torch.long),
        node_ptr=torch.tensor([0, 2], dtype=torch.long),
        batch=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([0], dtype=torch.long),
        node_embeddings=torch.randn((2, 4), dtype=torch.float32),
        relation_embeddings=torch.randn((1, 4), dtype=torch.float32),
        edge_rel_local=torch.tensor([0], dtype=torch.long),
        question_emb=torch.randn((1, 4), dtype=torch.float32),
        question_ctx=torch.randn((1, 2, 4), dtype=torch.float32),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id=["sample"],
        question=["q"],
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
        shuffle=False,
        seed=13,
    )

    assert list(islice(iter(sampler), 7)) == [0, 1, 2, 0, 1, 2, 0]
    with pytest.raises(TypeError, match="intentionally unsized"):
        len(sampler)


def test_step_driven_train_sampler_supports_explicit_finite_budget() -> None:
    sampler = StepDrivenTrainSampler(
        dataset_size=3,
        num_samples=7,
        shuffle=False,
        seed=13,
    )

    assert len(sampler) == 7
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
    with pytest.raises(TypeError, match="intentionally unsized"):
        len(sampler)


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
        prefetch_factor=None,
        persistent_workers=False,
        train_max_edges_per_batch=12,
    )

    datamodule.setup(stage="fit")
    loader = datamodule.train_dataloader()

    batch_sampler = loader["batch_sampler"]
    assert isinstance(batch_sampler, BudgetBatchSampler)
    assert captured["kwargs"]["sampler"] is not None
    with pytest.raises(TypeError, match="unsized sampler"):
        len(batch_sampler)


def test_graph_retrieval_datamodule_train_loader_is_unsized_but_eval_loader_is_sized(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.create_graph_retrieval_dataset",
        lambda **kwargs: _DummyDataset(str(kwargs["split_name"]), size=4),
    )

    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=True,
        prefetch_factor=None,
        persistent_workers=False,
        train_max_edges_per_batch=12,
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    with pytest.raises(TypeError, match="unsized sampler"):
        len(train_loader)
    assert len(val_loader) == 2


def test_budget_batch_sampler_rejects_oversized_singleton() -> None:
    dataset = _DummyDataset(
        "train",
        size=1,
        stats_by_idx={0: SimpleNamespace(num_nodes=32, num_edges=4, question_tokens=2)},
    )
    sampler = BudgetBatchSampler(
        sampler=[0],
        dataset=dataset,
        max_graphs_per_batch=4,
        max_nodes_per_batch=16,
        max_edges_per_batch=None,
        max_question_tokens_per_batch=None,
        drop_last=False,
    )

    with pytest.raises(ValueError, match="Single sample exceeds batch budget"):
        list(sampler)


def test_budget_batch_sampler_len_respects_active_edge_budget() -> None:
    dataset = _DummyDataset(
        "train",
        size=3,
        stats_by_idx={
            0: SimpleNamespace(num_nodes=4, num_edges=6, question_tokens=2),
            1: SimpleNamespace(num_nodes=4, num_edges=6, question_tokens=2),
            2: SimpleNamespace(num_nodes=4, num_edges=1, question_tokens=2),
        },
    )
    sampler = BudgetBatchSampler(
        sampler=[0, 1, 2],
        dataset=dataset,
        max_graphs_per_batch=10,
        max_nodes_per_batch=None,
        max_edges_per_batch=10,
        max_question_tokens_per_batch=None,
        drop_last=False,
    )

    assert len(sampler) == 2
    assert list(sampler) == [[0], [1, 2]]


def test_graph_retrieval_datamodule_set_eval_split_invalidates_cached_dataset(
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
    datamodule.set_eval_split("test")
    loader = datamodule.predict_dataloader()

    assert created_splits == ["validation", "test"]
    assert loader == {"split": "test", "shuffle": False, "batch_size": 7}


def test_graph_retrieval_datamodule_uses_eval_worker_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_calls: list[dict[str, object]] = []
    supported_methods = [method.lower() for method in mp.get_all_start_methods()]
    train_context = next(
        (method for method in supported_methods if method != "spawn"),
        supported_methods[0],
    )
    eval_context = "spawn" if "spawn" in supported_methods else supported_methods[0]

    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.SharedDataResources",
        lambda **kwargs: SimpleNamespace(clear=lambda: None, **kwargs),
    )
    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.create_graph_retrieval_dataset",
        lambda **kwargs: _DummyDataset(str(kwargs["split_name"]), size=4),
    )

    def _build_loader(dataset, **kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append({"dataset": dataset, **kwargs})
        return {"split": dataset.split, "num_workers": kwargs["num_workers"]}

    monkeypatch.setattr(
        "src.datasets.graph_retrieval_datamodule.build_retrieval_dataloader",
        _build_loader,
    )

    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        eval_batch_size=7,
        num_workers=8,
        eval_num_workers=3,
        pin_memory=False,
        drop_last=False,
        train_shuffle=True,
        prefetch_factor=4,
        eval_prefetch_factor=2,
        persistent_workers=True,
        eval_persistent_workers=True,
        multiprocessing_context=train_context,
        eval_multiprocessing_context=eval_context,
        eval_split="validation",
    )

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    assert train_loader == {"split": "train", "num_workers": 8}
    assert val_loader == {"split": "validation", "num_workers": 3}
    assert captured_calls[0]["prefetch_factor"] == 4
    assert captured_calls[0]["persistent_workers"] is True
    assert captured_calls[0]["multiprocessing_context"] == train_context
    assert captured_calls[1]["prefetch_factor"] == 2
    assert captured_calls[1]["persistent_workers"] is True
    assert captured_calls[1]["multiprocessing_context"] == eval_context


def test_graph_retrieval_datamodule_disables_eval_context_when_workers_zero(
    tmp_path: Path,
) -> None:
    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=4,
        eval_num_workers=0,
        prefetch_factor=4,
        eval_prefetch_factor=2,
        persistent_workers=True,
        eval_persistent_workers=True,
        multiprocessing_context="spawn",
        eval_multiprocessing_context="spawn",
    )

    assert datamodule.eval_prefetch_factor is None
    assert datamodule.eval_persistent_workers is False
    assert datamodule.eval_multiprocessing_context is None


def test_graph_retrieval_datamodule_rejects_unknown_multiprocessing_context(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="multiprocessing_context"):
        GraphRetrievalDataModule(
            dataset_cfg=_write_dataset_paths(tmp_path),
            batch_size=2,
            num_workers=1,
            multiprocessing_context="definitely-not-a-start-method",
        )


def test_on_after_batch_transfer_computes_missing_edge_batch(tmp_path: Path) -> None:
    datamodule = GraphRetrievalDataModule(
        dataset_cfg=_write_dataset_paths(tmp_path),
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        train_shuffle=False,
        prefetch_factor=None,
        persistent_workers=False,
        precompute_edge_batch=False,
    )

    runtime_batch = datamodule.on_after_batch_transfer(
        _make_pyg_batch_without_edge_batch(),
        dataloader_idx=0,
    )

    assert isinstance(runtime_batch, TrajectoryBatch)
    assert torch.equal(runtime_batch.edge_batch, torch.tensor([0], dtype=torch.long))
    assert runtime_batch.edge_embeddings is None
    assert runtime_batch.relation_embeddings is not None
    assert torch.equal(
        runtime_batch.edge_rel_local, torch.tensor([0], dtype=torch.long)
    )


def test_on_before_batch_transfer_attaches_relation_table_and_casts_features(
    tmp_path: Path,
) -> None:
    class _DummyGlobalEmbeddings:
        def __init__(self) -> None:
            self.entity_embeddings = torch.arange(12, dtype=torch.float32).view(3, 4)
            self.relation_embeddings = torch.arange(8, dtype=torch.float32).view(2, 4)

        def get_entity_embeddings(self, ids, *, device=None, dtype=None):
            out = self.entity_embeddings.index_select(0, ids.to(dtype=torch.long))
            if dtype is not None:
                out = out.to(dtype=dtype)
            if device is not None:
                out = out.to(device=device)
            return out

        def get_relation_embeddings(self, ids, *, device=None, dtype=None):
            out = self.relation_embeddings.index_select(0, ids.to(dtype=torch.long))
            if dtype is not None:
                out = out.to(dtype=dtype)
            if device is not None:
                out = out.to(device=device)
            return out

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
    datamodule._shared_resources = SimpleNamespace(
        global_embeddings=_DummyGlobalEmbeddings()
    )
    batch = SimpleNamespace(
        node_embedding_ids=torch.tensor([0, 2], dtype=torch.long),
        edge_attr=torch.tensor([1, 1, 0], dtype=torch.long),
        question_emb=torch.randn((1, 4), dtype=torch.float32),
        question_ctx=torch.randn((1, 2, 4), dtype=torch.float32),
        heuristic_log_v=torch.randn((2,), dtype=torch.float32),
    )

    attached = datamodule.on_before_batch_transfer(batch, dataloader_idx=0)

    assert attached is batch
    assert attached.node_embeddings.dtype == torch.bfloat16
    assert attached.relation_embeddings.dtype == torch.bfloat16
    assert torch.equal(
        attached.edge_rel_local, torch.tensor([1, 1, 0], dtype=torch.long)
    )
    assert attached.edge_embeddings is None
    assert attached.question_emb.dtype == torch.bfloat16
    assert attached.question_ctx.dtype == torch.bfloat16
    assert attached.heuristic_log_v.dtype == torch.bfloat16


def test_resolve_embedding_attachment_device_prefers_trainer_cuda_root() -> None:
    trainer = SimpleNamespace(
        strategy=SimpleNamespace(root_device=torch.device("cuda", 1)),
        lightning_module=None,
    )

    assert _resolve_embedding_attachment_device(None, trainer=trainer) == torch.device(
        "cuda", 1
    )
    assert _resolve_embedding_attachment_device("cpu", trainer=trainer) == torch.device(
        "cpu"
    )


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
    assert transferred.edge_embeddings is None
    assert transferred.relation_embeddings is not None
    assert transferred.relation_embeddings.dtype == torch.bfloat16
    assert transferred.question_emb.dtype == torch.bfloat16
    assert transferred.question_ctx.dtype == torch.bfloat16
    assert transferred.question_ctx_mask.dtype == torch.bool
