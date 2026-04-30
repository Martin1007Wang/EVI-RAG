from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List

import lmdb
import torch

from src.data.schema import StorageSchema


# =========================
# serialization
# =========================


def serialize_sample(sample: Dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    for k, v in sample.items():
        if not (torch.is_tensor(v) or isinstance(v, int)):
            raise TypeError(f"{k} must be tensor or int, got {type(v)}")

    return save(sample)


def deserialize_sample(payload: bytes) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load

    data = load(payload)
    StorageSchema.validate(data)
    return data


# =========================
# LMDB paths
# =========================


def resolve_lmdb_paths(root: Path, split: str) -> List[Path]:
    base = root / f"{split}.lmdb"
    if base.exists():
        return [base]

    shards = sorted(root.glob(f"{split}.shard*.lmdb"))
    if not shards:
        raise FileNotFoundError(f"No LMDB found for {split}")

    return shards


def load_sample_ids_from_paths(paths: list[Path]) -> list[str]:
    sample_ids: list[str] = []
    for path in paths:
        sample_ids.extend(get_all_keys_from_lmdb(path))
    return sorted(sample_ids)


# =========================
# sharding
# =========================


def assign_lmdb_shard(sample_key: str | bytes, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    if isinstance(sample_key, str):
        sample_key = sample_key.encode("utf-8")
    return int(zlib.crc32(sample_key) % num_shards)


# =========================
# key scan (optional)
# =========================


def get_all_keys_from_lmdb(path: Path) -> list[str]:
    env = lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    try:
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            out: list[str] = []
            for item in cursor:
                key = item[0] if isinstance(item, tuple) else item
                out.append(bytes(key).decode("utf-8"))
            return out
    finally:
        env.close()
