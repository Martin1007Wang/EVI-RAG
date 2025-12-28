import os
import pickle
from pathlib import Path

import lmdb
import pytest
import torch
from torch.testing import assert_close

pytest.importorskip("lightning")

from src.data import GRetrievalDataModule
from src.models.components.retriever import Retriever


def _write_lmdb(path: Path, entries) -> None:
    env = lmdb.open(str(path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            for key, value in entries.items():
                txn.put(key.encode("utf-8"), pickle.dumps(value))
    finally:
        env.close()


def _mock_sample() -> dict:
    node_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    return {
        "edge_index": torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
        "edge_attr": torch.zeros(2, dtype=torch.long),
        "labels": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "num_nodes": node_ids.numel(),
        "node_global_ids": node_ids,
        "node_embedding_ids": node_ids,
        "topic_one_hot": torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        "question_emb": torch.tensor([[0.25, 0.5, 0.75, 1.0]], dtype=torch.float32),
        "question": "mock question",
        "q_local_indices": torch.tensor([0], dtype=torch.long),
        "a_local_indices": torch.tensor([1], dtype=torch.long),
        "answer_entity_ids": torch.tensor([1], dtype=torch.long),
        "answer_entity_ids_len": torch.tensor([1], dtype=torch.long),
        "pair_start_node_locals": torch.empty(0, dtype=torch.long),
        "pair_answer_node_locals": torch.empty(0, dtype=torch.long),
        "pair_edge_local_ids": torch.empty(0, dtype=torch.long),
        "pair_edge_counts": torch.empty(0, dtype=torch.long),
    }


def _setup_mock_retrieval_root(tmp_path: Path):
    root = tmp_path / "mock_retrieval"
    vocab_dir = root / "vocabulary"
    vocab_dir.mkdir(parents=True)
    vocab_path = vocab_dir / "vocabulary.lmdb"

    entity_mapping = {"node_a": 0, "node_b": 1, "node_c": 2}
    relation_mapping = {"linked": 0}
    env = lmdb.open(str(vocab_path), map_size=1 << 20)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"entity_to_id", pickle.dumps(entity_mapping))
            txn.put(b"id_to_entity", pickle.dumps({v: k for k, v in entity_mapping.items()}))
            txn.put(b"relation_to_id", pickle.dumps(relation_mapping))
            txn.put(b"id_to_relation", pickle.dumps({v: k for k, v in relation_mapping.items()}))
    finally:
        env.close()

    embeddings_dir = root / "embeddings"
    embeddings_dir.mkdir(parents=True)
    entity_embeddings = torch.arange(12, dtype=torch.float32).view(3, 4)
    relation_embeddings = torch.arange(4, dtype=torch.float32).view(1, 4)
    torch.save(entity_embeddings, embeddings_dir / "entity_embeddings.pt")
    torch.save(relation_embeddings, embeddings_dir / "relation_embeddings.pt")

    for split in ("train", "validation", "test"):
        lmdb_path = embeddings_dir / f"{split}.lmdb"
        _write_lmdb(lmdb_path, {f"{split}_sample": _mock_sample()})

    return root, entity_embeddings, relation_embeddings


def test_retrieval_datamodule_preserves_indices(tmp_path) -> None:
    root, entity_emb, relation_emb = _setup_mock_retrieval_root(tmp_path)
    cfg = {
        "name": "mock_retrieval",
        "paths": {
            "vocabulary": str(root / "vocabulary" / "vocabulary.lmdb"),
            "embeddings": str(root / "embeddings"),
        },
    }
    dm = GRetrievalDataModule(
        dataset_cfg=cfg,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    # Node/edge embeddings must match the referenced global IDs exactly.
    assert torch.equal(batch.node_embeddings, entity_emb[batch.node_global_ids])
    assert torch.equal(batch.edge_embeddings, relation_emb[batch.edge_attr])

    head_global = batch.node_global_ids[batch.edge_index[0]]
    tail_global = batch.node_global_ids[batch.edge_index[1]]
    assert torch.equal(head_global, torch.tensor([0, 2], dtype=torch.long))
    assert torch.equal(tail_global, torch.tensor([1, 0], dtype=torch.long))
    assert hasattr(batch, "edge_batch")
    expected_edge_batch = batch.batch[batch.edge_index[0]]
    assert torch.equal(batch.edge_batch, expected_edge_batch)


def test_retrieval_datamodule_real_dataset(tmp_path) -> None:
    real_root_env = os.environ.get("REAL_RETRIEVAL_DATA_ROOT") or os.environ.get("RETRIEVE_DATA_ROOT")
    if not real_root_env:
        pytest.skip("REAL_RETRIEVAL_DATA_ROOT not configured")

    root = Path(real_root_env).expanduser()
    if not root.exists():
        pytest.skip(f"Dataset root {root} missing")

    vocab_path = root / "vocabulary" / "vocabulary.lmdb"
    embeddings_dir = root / "embeddings"
    if not vocab_path.exists() or not embeddings_dir.exists():
        pytest.skip("Real dataset root missing vocabulary/embeddings")

    cfg = {
        "name": root.name,
        "paths": {
            "vocabulary": str(root / "vocabulary" / "vocabulary.lmdb"),
            "embeddings": str(root / "embeddings"),
        },
        "sample_limit": {"train": 2},
    }
    dm = GRetrievalDataModule(
        dataset_cfg=cfg,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    dm.prepare_data()
    dm.setup()

    train_ds = dm.train_dataset
    assert train_ds is not None

    batch = next(iter(dm.train_dataloader()))

    ref_nodes = train_ds.global_embeddings.get_entity_embeddings(batch.node_global_ids.cpu())
    ref_edges = train_ds.global_embeddings.get_relation_embeddings(batch.edge_attr.cpu())

    assert_close(batch.node_embeddings.cpu(), ref_nodes)
    assert_close(batch.edge_embeddings.cpu(), ref_edges)


def test_retriever_forward_smoke(tmp_path) -> None:
    root, _, _ = _setup_mock_retrieval_root(tmp_path)
    cfg = {
        "name": "mock_retrieval",
        "paths": {
            "vocabulary": str(root / "vocabulary" / "vocabulary.lmdb"),
            "embeddings": str(root / "embeddings"),
        },
    }
    dm = GRetrievalDataModule(
        dataset_cfg=cfg,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    dm.prepare_data()
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    model = Retriever(
        emb_dim=int(batch.question_emb.shape[-1]),
        hidden_dim=8,
        topic_pe=True,
        num_topics=2,
        dde_cfg={"num_rounds": 0, "num_reverse_rounds": 0},
        dropout_p=0.0,
    ).eval()
    with torch.no_grad():
        output = model(batch)
    scores = torch.sigmoid(output.logits)
    assert scores.numel() == batch.edge_attr.numel()
    assert output.logits.shape == scores.shape
    assert output.query_ids.numel() == scores.numel()


def test_retrieval_datamodule_requires_paths_or_data_dir() -> None:
    cfg = {"name": "mock"}
    with pytest.raises(ValueError, match="paths"):
        GRetrievalDataModule(
            dataset_cfg=cfg,
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )
