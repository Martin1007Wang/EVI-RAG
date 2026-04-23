from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from safetensors.torch import save

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.lmdb_utils import deserialize_sample, serialize_sample


def test_serialize_sample_clones_shared_storage_aliases() -> None:
    shared_mask = torch.tensor([True, False, True], dtype=torch.bool)
    shared_ids = torch.tensor([2, 5], dtype=torch.long)
    sample = {
        "train_target_mask": shared_mask,
        "is_target_mask": shared_mask,
        "reachable_target_node_ids": shared_ids,
        "train_target_node_ids": shared_ids,
    }

    with pytest.raises(RuntimeError, match="share memory"):
        save(sample)

    payload = serialize_sample(sample)
    restored = deserialize_sample(payload)

    assert torch.equal(restored["train_target_mask"], shared_mask)
    assert torch.equal(restored["is_target_mask"], shared_mask)
    assert torch.equal(restored["reachable_target_node_ids"], shared_ids)
    assert torch.equal(restored["train_target_node_ids"], shared_ids)
