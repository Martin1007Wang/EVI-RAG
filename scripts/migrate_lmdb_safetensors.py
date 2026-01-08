#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import lmdb
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.io.lmdb_utils import _serialize_sample

_LMDB_SUFFIX = ".lmdb"
_MAP_GROWTH_FACTOR_DEFAULT = 2.0
_TXN_SIZE_DEFAULT = 4096
_MAX_READERS = 1
_JSON_ENCODING = "utf-8"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate LMDB payloads to safetensors.")
    parser.add_argument("--input", required=True, help="LMDB file or directory containing *.lmdb")
    parser.add_argument("--output", required=True, help="Output directory for migrated LMDBs")
    parser.add_argument("--txn-size", type=int, default=_TXN_SIZE_DEFAULT, help="LMDB transaction batch size")
    parser.add_argument(
        "--map-growth-factor",
        type=float,
        default=_MAP_GROWTH_FACTOR_DEFAULT,
        help="Map size growth factor on MapFullError",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output LMDB paths if they exist")
    return parser.parse_args()


def _resolve_inputs(path: Path) -> List[Path]:
    if path.is_dir() and path.suffix == _LMDB_SUFFIX:
        return [path]
    if path.is_file() and path.suffix == _LMDB_SUFFIX:
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob(f"*{_LMDB_SUFFIX}") if p.is_dir())
    raise FileNotFoundError(f"LMDB input not found: {path}")


def _open_read_env(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=_MAX_READERS,
    )


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _open_write_env(path: Path, *, map_size: int, overwrite: bool) -> lmdb.Environment:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output LMDB already exists: {path}")
        import shutil

        shutil.rmtree(path)
    _ensure_output_dir(path)
    return lmdb.open(str(path), map_size=map_size, subdir=True, create=True, max_readers=_MAX_READERS)


def _grow_map(env: lmdb.Environment, factor: float) -> None:
    current = int(env.info().get("map_size", 0))
    next_size = int(current * factor)
    if next_size <= current:
        raise RuntimeError("LMDB map size did not grow; adjust map-growth-factor.")
    env.set_mapsize(next_size)


def _write_with_growth(
    env: lmdb.Environment,
    txn: lmdb.Transaction,
    key: bytes,
    payload: bytes,
    *,
    growth_factor: float,
) -> lmdb.Transaction:
    while True:
        try:
            txn.put(key, payload)
            return txn
        except lmdb.MapFullError:
            txn.abort()
            _grow_map(env, growth_factor)
            txn = env.begin(write=True)


def _load_pickled_sample(payload: bytes) -> Dict[str, object]:
    sample = pickle.loads(payload)
    if not isinstance(sample, dict):
        raise TypeError(f"Expected dict payload, got {type(sample)!r}.")
    return sample


def _coerce_value(value: object, key: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value)
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value)
    raise TypeError(f"Unsupported value for key {key!r}: {type(value)!r}")


def _coerce_sample(sample: Dict[str, object]) -> Dict[str, torch.Tensor]:
    return {str(key): _coerce_value(value, str(key)) for key, value in sample.items()}


def _iter_lmdb_items(env: lmdb.Environment) -> Iterator[Tuple[bytes, bytes]]:
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            yield key, value


def _read_vocab_labels(txn: lmdb.Transaction) -> Tuple[List[str], List[str]]:
    entity_labels = txn.get(b"entity_labels")
    relation_labels = txn.get(b"relation_labels")
    if entity_labels and relation_labels:
        return json.loads(entity_labels.decode(_JSON_ENCODING)), json.loads(relation_labels.decode(_JSON_ENCODING))
    entity_to_id = txn.get(b"entity_to_id")
    relation_to_id = txn.get(b"relation_to_id")
    if entity_to_id and relation_to_id:
        return _labels_from_mapping(pickle.loads(entity_to_id)), _labels_from_mapping(pickle.loads(relation_to_id))
    raise ValueError("Vocabulary LMDB missing expected keys.")


def _labels_from_mapping(mapping: Dict[str, int]) -> List[str]:
    if not mapping:
        return []
    max_id = max(int(idx) for idx in mapping.values())
    labels: List[str] = [""] * (max_id + 1)
    for label, idx in mapping.items():
        index = int(idx)
        if labels[index]:
            raise ValueError(f"Duplicate vocab id {index} for label {label!r}")
        labels[index] = str(label)
    if any(label == "" for label in labels):
        raise ValueError("Vocab ids are not contiguous; cannot rebuild label list.")
    return labels


def _convert_vocab_lmdb(input_path: Path, output_path: Path, *, overwrite: bool) -> None:
    in_env = _open_read_env(input_path)
    map_size = int(in_env.info().get("map_size", 0))
    out_env = _open_write_env(output_path, map_size=map_size, overwrite=overwrite)
    try:
        with in_env.begin(write=False) as in_txn, out_env.begin(write=True) as out_txn:
            entity_labels, relation_labels = _read_vocab_labels(in_txn)
            entity_payload = json.dumps(entity_labels, ensure_ascii=True).encode(_JSON_ENCODING)
            relation_payload = json.dumps(relation_labels, ensure_ascii=True).encode(_JSON_ENCODING)
            out_txn.put(b"entity_labels", entity_payload)
            out_txn.put(b"relation_labels", relation_payload)
    finally:
        in_env.close()
        out_env.close()


def _convert_core_lmdb(
    input_path: Path,
    output_path: Path,
    *,
    txn_size: int,
    growth_factor: float,
    overwrite: bool,
) -> None:
    in_env = _open_read_env(input_path)
    map_size = int(in_env.info().get("map_size", 0))
    out_env = _open_write_env(output_path, map_size=map_size, overwrite=overwrite)
    try:
        processed = 0
        txn = out_env.begin(write=True)
        for key, payload in _iter_lmdb_items(in_env):
            sample = _load_pickled_sample(payload)
            tensor_sample = _coerce_sample(sample)
            serialized = _serialize_sample(tensor_sample)
            txn = _write_with_growth(out_env, txn, key, serialized, growth_factor=growth_factor)
            processed += 1
            if txn_size > 0 and processed % txn_size == 0:
                txn.commit()
                txn = out_env.begin(write=True)
        txn.commit()
    finally:
        in_env.close()
        out_env.close()


def _is_vocab_lmdb(path: Path) -> bool:
    env = _open_read_env(path)
    try:
        with env.begin(write=False) as txn:
            return txn.get(b"entity_labels") is not None or txn.get(b"entity_to_id") is not None
    finally:
        env.close()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    _ensure_output_dir(output_dir)
    inputs = _resolve_inputs(input_path)
    for path in inputs:
        out_path = output_dir / path.name
        if _is_vocab_lmdb(path):
            _convert_vocab_lmdb(path, out_path, overwrite=args.overwrite)
        else:
            _convert_core_lmdb(
                path,
                out_path,
                txn_size=int(args.txn_size),
                growth_factor=float(args.map_growth_factor),
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
