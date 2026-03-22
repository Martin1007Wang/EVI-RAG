from __future__ import annotations

from pathlib import Path

import torch

from src.data.io.runtime_sample_metadata import save_runtime_sample_metadata
from src.datasets.graph_retrieval_dataset import GraphRetrievalDataset


def _build_dataset_shell(tmp_path: Path) -> GraphRetrievalDataset:
    dataset = GraphRetrievalDataset.__new__(GraphRetrievalDataset)
    dataset.split = "train"
    dataset._shared_resources = None
    dataset._heuristic_log_v_path = None
    dataset._heuristic_log_v = None
    dataset._sample_stats_cache = {}
    dataset._runtime_sample_metadata = None
    dataset._runtime_sample_metadata_path = tmp_path / "train.metadata.pt"
    dataset.sample_ids = ["unit/train/q1"]
    return dataset


def test_build_data_synthesizes_question_context_from_question_embedding(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset_shell(tmp_path)
    raw = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "num_nodes": torch.tensor(2, dtype=torch.long),
        "node_global_ids": torch.tensor([0, 1], dtype=torch.long),
        "node_embedding_ids": torch.tensor([0, 0], dtype=torch.long),
        "question_emb": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        "q_local_indices": torch.tensor([0], dtype=torch.long),
        "a_local_indices": torch.tensor([1], dtype=torch.long),
        "answer_entity_ids": torch.tensor([1], dtype=torch.long),
    }

    data = dataset._build_data(raw, "unit/train/q1")

    assert tuple(data.question_ctx.shape) == (1, 1, 3)
    assert torch.equal(data.question_ctx[:, 0, :], data.question_emb)
    assert torch.equal(data.question_ctx_mask, torch.tensor([[True]], dtype=torch.bool))


def test_get_sample_stats_uses_synthesized_question_context_mask(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset_shell(tmp_path)
    raw = {
        "edge_attr": torch.tensor([0, 1], dtype=torch.long),
        "num_nodes": torch.tensor(2, dtype=torch.long),
        "question_emb": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
    }
    dataset._load_raw_sample = lambda sample_id: raw  # type: ignore[method-assign]

    stats = dataset.get_sample_stats(0)

    assert stats.num_nodes == 2
    assert stats.num_edges == 2
    assert stats.question_tokens == 1


def test_get_sample_stats_prefers_runtime_sample_metadata(tmp_path: Path) -> None:
    dataset = _build_dataset_shell(tmp_path)
    save_runtime_sample_metadata(
        dataset._runtime_sample_metadata_path,
        split="train",
        sample_ids=["unit/train/q1"],
        questions=["Who wrote it?"],
        num_nodes=[7],
        num_edges=[11],
        question_tokens=[5],
    )
    dataset._load_raw_sample = lambda sample_id: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError(f"unexpected LMDB read for {sample_id}")
    )

    stats = dataset.get_sample_stats(0)

    assert stats.num_nodes == 7
    assert stats.num_edges == 11
    assert stats.question_tokens == 5


def test_build_data_uses_runtime_sample_metadata_question_text(tmp_path: Path) -> None:
    dataset = _build_dataset_shell(tmp_path)
    save_runtime_sample_metadata(
        dataset._runtime_sample_metadata_path,
        split="train",
        sample_ids=["unit/train/q1"],
        questions=["What is the question text?"],
        num_nodes=[2],
        num_edges=[1],
        question_tokens=[1],
    )
    raw = {
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "edge_attr": torch.tensor([0], dtype=torch.long),
        "num_nodes": torch.tensor(2, dtype=torch.long),
        "node_global_ids": torch.tensor([0, 1], dtype=torch.long),
        "node_embedding_ids": torch.tensor([0, 0], dtype=torch.long),
        "question_emb": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        "q_local_indices": torch.tensor([0], dtype=torch.long),
        "a_local_indices": torch.tensor([1], dtype=torch.long),
        "answer_entity_ids": torch.tensor([1], dtype=torch.long),
    }

    data = dataset._build_data(raw, "unit/train/q1")

    assert data.question == "What is the question text?"
