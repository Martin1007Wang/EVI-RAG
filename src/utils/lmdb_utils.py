from __future__ import annotations

import zlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import lmdb
import torch

# =========================
# serialization
# =========================


SampleValue = torch.Tensor | int


def serialize_sample(sample: Dict[str, torch.Tensor]) -> bytes:
    from safetensors.torch import save

    return save(_normalize_sample_for_safetensors(sample))


def _normalize_sample_for_safetensors(
    sample: Dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for k, v in sample.items():
        if not (torch.is_tensor(v) or isinstance(v, int)):
            raise TypeError(f"{k} must be tensor or int, got {type(v)}")
        tensor = torch.as_tensor(v) if isinstance(v, int) else v
        normalized[k] = tensor.detach()

    for key, tensor in normalized.items():
        if not tensor.is_contiguous():
            normalized[key] = tensor.contiguous()

    for key in _overlapping_storage_keys(normalized):
        normalized[key] = normalized[key].clone().contiguous()

    return normalized


def _overlapping_storage_keys(sample: Dict[str, torch.Tensor]) -> set[str]:
    storage_groups: dict[tuple[torch.device, int, int], list[tuple[int, int, str]]] = defaultdict(list)
    for key, tensor in sample.items():
        if tensor.device.type == "meta" or tensor.untyped_storage().nbytes() == 0:
            continue
        storage = tensor.untyped_storage()
        storage_groups[(tensor.device, storage.data_ptr(), storage.nbytes())].append((_tensor_start_ptr(tensor), _tensor_end_ptr(tensor), key))

    overlapping: set[str] = set()
    for spans in storage_groups.values():
        if len(spans) < 2:
            continue
        spans.sort()
        _, last_end, last_key = spans[0]
        for start, end, key in spans[1:]:
            if start < last_end:
                overlapping.add(last_key)
                overlapping.add(key)
            if end > last_end:
                last_end = end
                last_key = key
    return overlapping


def _tensor_start_ptr(tensor: torch.Tensor) -> int:
    return tensor.data_ptr()


def _tensor_end_ptr(tensor: torch.Tensor) -> int:
    return tensor.data_ptr() + tensor.nelement() * tensor.element_size()


def deserialize_sample(payload: bytes) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load

    return load(payload)


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


def load_row_ids_from_paths(paths: list[Path]) -> list[int]:
    row_ids: list[int] = []
    for path in paths:
        row_ids.extend(get_all_row_ids_from_lmdb(path))
    return sorted(row_ids)


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


def get_all_row_ids_from_lmdb(path: Path) -> list[int]:
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
            out: list[int] = []
            for item in cursor:
                key = item[0] if isinstance(item, tuple) else item
                out.append(int.from_bytes(bytes(key), byteorder="big", signed=False))
            return out
    finally:
        env.close()
