from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lmdb
import torch
from omegaconf import DictConfig

from src.data.schema.constants import (
    _BYTES_PER_GB,
    _DEFAULT_LMDB_MAP_GROWTH_FACTOR,
    _DEFAULT_LMDB_MAP_GROWTH_GB,
    _LMDB_GROWTH_FACTOR_MIN,
    _LMDB_GROWTH_GB_MIN,
    _LMDB_MAP_SIZE_GB_MIN,
    _LMDB_SHARDS_MIN,
    _ONE,
    _ZERO,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _serialize_sample(sample: dict) -> bytes:
    from safetensors.torch import save

    tensors: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not torch.is_tensor(value):
            raise TypeError(f"LMDB sample field {key!r} must be a torch.Tensor, got {type(value)!r}")
        tensor = value.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.to(device="cpu")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        tensors[str(key)] = tensor
    return save(tensors)


def _deserialize_sample(payload: bytes) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load

    data = load(payload)
    if not isinstance(data, dict):
        raise ValueError("LMDB sample payload must decode to a dict.")
    return data


def _write_sample(txn: lmdb.Transaction, sample_key: bytes, payload: bytes) -> None:
    txn.put(sample_key, payload)


def _local_indices(node_ids: Sequence[int], targets: Sequence[int]) -> List[int]:
    position = {nid: idx for idx, nid in enumerate(node_ids)}
    return [position[t] for t in targets if t in position]


def _resolve_lmdb_map_config(cfg: DictConfig) -> Tuple[int, int, float, Optional[int]]:
    map_size_gb_cfg = cfg.get("map_size_gb")
    if map_size_gb_cfg is None:
        raise ValueError("map_size_gb must be set for LMDB materialization.")
    map_size_gb = int(map_size_gb_cfg)
    growth_gb_cfg = cfg.get("map_growth_gb")
    growth_gb = _DEFAULT_LMDB_MAP_GROWTH_GB if growth_gb_cfg is None else int(growth_gb_cfg)
    growth_factor_cfg = cfg.get("map_growth_factor")
    growth_factor = _DEFAULT_LMDB_MAP_GROWTH_FACTOR if growth_factor_cfg is None else float(growth_factor_cfg)
    max_gb_cfg = cfg.get("map_max_gb")
    max_map_size_bytes = None if max_gb_cfg is None else int(max_gb_cfg) * _BYTES_PER_GB
    if map_size_gb < _LMDB_MAP_SIZE_GB_MIN:
        raise ValueError(f"map_size_gb must be >= {_LMDB_MAP_SIZE_GB_MIN}, got {map_size_gb}")
    if growth_gb <= _LMDB_GROWTH_GB_MIN and growth_factor <= _LMDB_GROWTH_FACTOR_MIN:
        raise ValueError("LMDB map growth disabled; set map_growth_gb > 0 or map_growth_factor > 1.0.")
    map_size_bytes = map_size_gb * _BYTES_PER_GB
    growth_bytes = growth_gb * _BYTES_PER_GB
    if max_map_size_bytes is not None and max_map_size_bytes < map_size_bytes:
        raise ValueError("map_max_gb must be >= map_size_gb when set.")
    return map_size_bytes, growth_bytes, growth_factor, max_map_size_bytes


def _resolve_lmdb_shards(cfg: DictConfig) -> int:
    shards = int(cfg.get("lmdb_shards", _LMDB_SHARDS_MIN))
    if shards < _LMDB_SHARDS_MIN:
        raise ValueError(f"lmdb_shards must be >= {_LMDB_SHARDS_MIN}, got {shards}")
    return shards


def _lmdb_shard_suffix(shard_id: int, num_shards: int) -> str:
    if num_shards <= 1:
        return ""
    return f".shard{int(shard_id):03d}"


def _format_lmdb_path(base_dir: Path, split: str, shard_id: int, num_shards: int, *, suffix: str) -> Path:
    shard_suffix = _lmdb_shard_suffix(shard_id, num_shards)
    return base_dir / f"{split}{shard_suffix}{suffix}"


def _resolve_core_lmdb_paths(embeddings_dir: Path, split: str) -> List[Path]:
    split = str(split)
    base_path = embeddings_dir / f"{split}.lmdb"
    shard_paths = sorted(embeddings_dir.glob(f"{split}.shard*.lmdb"))
    if base_path.exists():
        if shard_paths:
            raise ValueError(f"Both sharded and unsharded LMDBs found under {embeddings_dir}.")
        return [base_path]
    if not shard_paths:
        raise FileNotFoundError(f"Core LMDB not found for split={split} under {embeddings_dir}")
    shard_map: Dict[int, Path] = {}
    token = f"{split}.shard"
    for path in shard_paths:
        stem = path.stem
        if not stem.startswith(token):
            raise ValueError(f"Unexpected LMDB shard name: {path.name}")
        shard_text = stem[len(token):]
        if not shard_text.isdigit():
            raise ValueError(f"Invalid shard id in LMDB shard {path.name}")
        shard_id = int(shard_text)
        if shard_id in shard_map:
            raise ValueError(f"Duplicate LMDB shard id {shard_id} for {split}.")
        shard_map[shard_id] = path
    shard_ids = sorted(shard_map)
    if not shard_ids:
        raise FileNotFoundError(f"No LMDB shards found for {split}.")
    expected = list(range(shard_ids[-1] + _ONE))
    if shard_ids != expected:
        raise ValueError(f"LMDB shards must be contiguous from 0; found {shard_ids}.")
    return [shard_map[shard_id] for shard_id in expected]


def _assign_lmdb_shard(sample_key: str | bytes, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    import zlib

    if isinstance(sample_key, str):
        sample_key = sample_key.encode("utf-8")
    return int(zlib.crc32(sample_key) % num_shards)


def _next_lmdb_map_size_bytes(
    current_size: int,
    growth_bytes: int,
    growth_factor: float,
    max_size: Optional[int],
) -> int:
    next_size = max(current_size + growth_bytes, int(current_size * growth_factor))
    if max_size is not None:
        next_size = min(next_size, max_size)
    if next_size <= current_size:
        raise RuntimeError("LMDB map size limit reached; increase map_max_gb to continue.")
    return next_size


def _replay_pending_with_growth(
    *,
    env: lmdb.Environment,
    pending_payloads: List[Tuple[bytes, bytes]],
    map_size_bytes: int,
    growth_bytes: int,
    growth_factor: float,
    max_size_bytes: Optional[int],
) -> Tuple[lmdb.Transaction, int]:
    while True:
        txn = env.begin(write=True)
        try:
            for key, payload in pending_payloads:
                txn.put(key, payload)
            return txn, map_size_bytes
        except lmdb.MapFullError:
            txn.abort()
            map_size_bytes = _next_lmdb_map_size_bytes(
                current_size=map_size_bytes,
                growth_bytes=growth_bytes,
                growth_factor=growth_factor,
                max_size=max_size_bytes,
            )
            env.set_mapsize(map_size_bytes)


def _commit_pending_with_growth(
    *,
    env: lmdb.Environment,
    txn: lmdb.Transaction,
    pending_payloads: List[Tuple[bytes, bytes]],
    map_size_bytes: int,
    growth_bytes: int,
    growth_factor: float,
    max_size_bytes: Optional[int],
) -> Tuple[lmdb.Transaction, int]:
    while True:
        try:
            txn.commit()
            return env.begin(write=True), map_size_bytes
        except lmdb.MapFullError:
            txn.abort()
            map_size_bytes = _next_lmdb_map_size_bytes(
                current_size=map_size_bytes,
                growth_bytes=growth_bytes,
                growth_factor=growth_factor,
                max_size=max_size_bytes,
            )
            env.set_mapsize(map_size_bytes)
            txn = env.begin(write=True)
            for key, payload in pending_payloads:
                txn.put(key, payload)


def _prepare_lmdb_dir(path: Path, *, overwrite: bool) -> Path:
    """Prepare a clean temporary LMDB directory and return its path.

    We always write into a sibling ``*.tmp`` directory and atomically swap on success.
    This prevents stale keys from older builds from leaking into the current dataset.
    """
    tmp_path = Path(str(path) + ".tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"LMDB already exists at {path}; set overwrite_lmdb=true to rebuild deterministically.")
    ensure_dir(tmp_path)
    return tmp_path


def _finalize_lmdb_dir(*, tmp_path: Path, final_path: Path, overwrite: bool) -> None:
    if not tmp_path.exists():
        raise FileNotFoundError(f"Temporary LMDB dir missing: {tmp_path}")
    if final_path.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing LMDB at {final_path}")
        shutil.rmtree(final_path)
    tmp_path.rename(final_path)


def _load_filter_ids_from_path(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Filter file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    import json

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Filter JSON must be list or dict: {path}") from exc
    if isinstance(data, list):
        return set(map(str, data))
    if isinstance(data, dict):
        if "sample_ids" not in data:
            raise ValueError(f"Filter JSON dict missing 'sample_ids': {path}")
        return set(map(str, data["sample_ids"]))
    raise ValueError(f"Filter JSON must be list or dict: {path}")


def _apply_filter_intersection(sample_ids: Sequence[str], filter_paths: Sequence[Path]) -> List[str]:
    if not filter_paths:
        return list(sample_ids)
    keep_ids: Optional[set[str]] = None
    for path in filter_paths:
        ids = _load_filter_ids_from_path(path)
        if keep_ids is None:
            keep_ids = ids
        else:
            keep_ids &= ids
    if keep_ids is None:
        return list(sample_ids)
    return [sid for sid in sample_ids if sid in keep_ids]
