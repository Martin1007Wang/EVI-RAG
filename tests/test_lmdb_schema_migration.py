from __future__ import annotations

from pathlib import Path

import lmdb
import torch

from src.data.io.lmdb_utils import _deserialize_sample
from src.data.io.lmdb_utils import _serialize_sample
from src.data.io.lmdb_utils import migrate_legacy_node_entity_ids_lmdb


def _load_single_sample(lmdb_path: Path, sample_id: str) -> dict[str, torch.Tensor]:
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        subdir=lmdb_path.is_dir(),
    )
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(sample_id.encode("utf-8"))
            assert payload is not None
            return _deserialize_sample(payload)
    finally:
        env.close()


def test_migrate_legacy_node_entity_ids_lmdb_renames_sample_field(
    tmp_path: Path,
) -> None:
    lmdb_path = tmp_path / "train.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(
                b"unit/train/q1",
                _serialize_sample(
                    {
                        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
                        "edge_attr": torch.tensor([0], dtype=torch.long),
                        "num_nodes": torch.tensor(2, dtype=torch.long),
                        "node_global_ids": torch.tensor([0, 1], dtype=torch.long),
                        "node_embedding_ids": torch.tensor([0, 0], dtype=torch.long),
                        "question_emb": torch.tensor(
                            [[0.1, 0.2, 0.3]], dtype=torch.float32
                        ),
                        "anchor_local_indices": torch.tensor([0], dtype=torch.long),
                        "a_local_indices": torch.tensor([1], dtype=torch.long),
                        "answer_entity_ids": torch.tensor([1], dtype=torch.long),
                    }
                ),
            )
    finally:
        env.close()

    stats = migrate_legacy_node_entity_ids_lmdb(lmdb_path)
    migrated = _load_single_sample(lmdb_path, "unit/train/q1")

    assert stats == {"total_samples": 1, "migrated_samples": 1}
    assert "node_global_ids" not in migrated
    assert torch.equal(
        migrated["node_entity_ids"], torch.tensor([0, 1], dtype=torch.long)
    )
