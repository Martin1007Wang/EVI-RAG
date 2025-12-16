from pathlib import Path
import pickle

import lmdb
import pytest
import torch
from torch_geometric.data import Batch, Data

from src.data.components.embedding_store import EmbeddingStore
from src.data.components.g_agent_builder import GAgentBuilder, GAgentSettings
from src.data.g_agent_dataset import load_g_agent_samples
from src.models.components.retriever import RetrieverOutput


def _write_lmdb(path: Path, entries: dict[str, dict]) -> None:
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            for key, value in entries.items():
                txn.put(key.encode("utf-8"), pickle.dumps(value))
    finally:
        env.close()


def test_g_agent_builder_roundtrip(tmp_path: Path) -> None:
    sample_id = "sample-1"
    node_global_ids = torch.tensor([10, 11], dtype=torch.long)

    data = Data(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([0], dtype=torch.long),
        labels=torch.tensor([1.0], dtype=torch.float32),
        node_global_ids=node_global_ids,
        node_embedding_ids=torch.tensor([10, 11], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        sample_id=sample_id,
    )
    batch = Batch.from_data_list([data])

    lmdb_path = tmp_path / "train.lmdb"
    raw = {
        "question_emb": torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32),
        "question": "mock question",
        "seed_entity_ids": torch.tensor([10], dtype=torch.long),
        "answer_entity_ids": torch.tensor([11], dtype=torch.long),
        "gt_path_edge_indices": [0],
    }
    _write_lmdb(lmdb_path, {sample_id: raw})

    store = EmbeddingStore(lmdb_path)
    try:
        settings = GAgentSettings(
            enabled=True,
            anchor_top_k=1,
            output_path=tmp_path / "g_agent_samples.pt",
            force_include_gt=False,
        )
        builder = GAgentBuilder(settings, embedding_store=store)
        output = RetrieverOutput(
            scores=torch.tensor([0.9], dtype=torch.float32),
            logits=torch.tensor([2.0], dtype=torch.float32),
            query_ids=torch.zeros(1, dtype=torch.long),
        )
        builder.process_batch(batch, output)
        assert len(builder.samples) == 1

        builder.save(settings.output_path)
        loaded = load_g_agent_samples(settings.output_path, drop_unreachable=False)
        assert len(loaded) == 1
        sample = loaded[0]
        assert sample.sample_id == sample_id
        assert sample.is_answer_reachable == (sample.answer_node_locals.numel() > 0)
    finally:
        store.close()


def test_g_agent_builder_keeps_unreachable_answers(tmp_path: Path) -> None:
    sample_id = "sample-unreachable"
    node_global_ids = torch.tensor([10, 11], dtype=torch.long)

    data = Data(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.tensor([0], dtype=torch.long),
        labels=torch.tensor([1.0], dtype=torch.float32),
        node_global_ids=node_global_ids,
        node_embedding_ids=torch.tensor([10, 11], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        sample_id=sample_id,
    )
    batch = Batch.from_data_list([data])

    lmdb_path = tmp_path / "train.lmdb"
    raw = {
        "question_emb": torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32),
        "question": "mock question unreachable",
        "seed_entity_ids": torch.tensor([10], dtype=torch.long),
        # Answer entity is not present in the selected subgraph nodes.
        "answer_entity_ids": torch.tensor([999], dtype=torch.long),
        "gt_path_edge_indices": [],
    }
    _write_lmdb(lmdb_path, {sample_id: raw})

    store = EmbeddingStore(lmdb_path)
    try:
        settings = GAgentSettings(
            enabled=True,
            anchor_top_k=1,
            output_path=tmp_path / "g_agent_samples_unreachable.pt",
            force_include_gt=False,
        )
        builder = GAgentBuilder(settings, embedding_store=store)
        output = RetrieverOutput(
            scores=torch.tensor([0.9], dtype=torch.float32),
            logits=torch.tensor([2.0], dtype=torch.float32),
            query_ids=torch.zeros(1, dtype=torch.long),
        )
        builder.process_batch(batch, output)
        assert len(builder.samples) == 1

        builder.save(settings.output_path)
        loaded = load_g_agent_samples(settings.output_path, drop_unreachable=False)
        assert len(loaded) == 1
        sample = loaded[0]
        assert sample.answer_entity_ids.numel() == 1
        assert sample.answer_node_locals.numel() == 0
        assert sample.is_answer_reachable is False

        with pytest.raises(ValueError):
            load_g_agent_samples(settings.output_path, drop_unreachable=True)
    finally:
        store.close()
