#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from array import array
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import lmdb

_ONE = 1
_ZERO = 0

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(_ZERO, str(_REPO_ROOT))

from src.data.io.lmdb_utils import _deserialize_sample

from src.data.schema.constants import (
    _DISTANCE_BYTES_PER_INT16,
    _DISTANCE_BYTES_PER_INT8,
    _DIST_UNREACHABLE,
)

_DATA_DIR_DEFAULT = Path("/mnt/data/retrieval_dataset")
_OUTPUT_DIR_DEFAULT = Path("outputs")
_DOCS_PATH_DEFAULT = Path("docs/cwq_dataset_stats.md")
_ANSWER_JSON_DEFAULT = _OUTPUT_DIR_DEFAULT / "cwq_answer_stats.json"
_DISTANCE_JSON_DEFAULT = _OUTPUT_DIR_DEFAULT / "cwq_distance_stats.json"
_DIRECTED_EXAMPLES_JSON_DEFAULT = _OUTPUT_DIR_DEFAULT / "cwq_directed_examples.json"
_PATHS_CONFIG = Path("configs/paths/default.yaml")
_CWQ_DIR = "cwq"
_NORMALIZED_DIR = "normalized"
_MATERIALIZED_DIR = "materialized"
_QUESTIONS_FILENAME = "questions.parquet"
_SUB_FILTER_FILENAME = "sub_filter.json"
_TRAIN_FILTER_FILENAME = "train_length_rel_filter.json"
_EMBEDDINGS_DIR = "embeddings"
_DISTANCES_DIR = "distances"
_SPLITS = ("train", "validation", "test")
_PERCENTILES = (50, 90, 95, 99)
_FLOAT_PRECISION = 3
_PROGRESS_INTERVAL_DEFAULT = 5000
_DIRECTED_PROGRESS_INTERVAL_DEFAULT = 1000
_LMDB_MAX_READERS = 256
_ENV_THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_ENV_SHM_DISABLE = "KMP_SHM_DISABLE"
_DIRECTED_MAX_DEPTH_DEFAULT = 6
_DIRECTED_MAX_EXAMPLES_DEFAULT = 20
_DIRECTED_GT2_THRESHOLD = 2
_UNDIRECTED_PATH_MAX_DEPTH = 2


def _read_data_dir_from_yaml(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("data_dir:"):
            _, value = stripped.split(":", maxsplit=_ONE)
            value = value.strip()
            if value:
                return Path(value)
    return None


def _resolve_data_dir(value: Optional[str]) -> Path:
    if value:
        return Path(value)
    from_cfg = _read_data_dir_from_yaml(_PATHS_CONFIG)
    return from_cfg or _DATA_DIR_DEFAULT


def _load_filter_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "sample_ids" in payload:
        return set(payload["sample_ids"])
    if isinstance(payload, list):
        return set(payload)
    raise ValueError(f"Unexpected filter format: {path}")


def _percentile(sorted_vals: Sequence[int], percentile: int) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == _ONE:
        return float(sorted_vals[_ZERO])
    k = (len(sorted_vals) - _ONE) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def _summarize_counts(counts: Sequence[int]) -> Dict[str, Optional[float]]:
    if not counts:
        return {
            "n": _ZERO,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    sorted_vals = sorted(counts)
    total = sum(sorted_vals)
    n = len(sorted_vals)
    stats = {
        "n": n,
        "min": int(sorted_vals[_ZERO]),
        "max": int(sorted_vals[-_ONE]),
        "mean": float(total / n),
    }
    for pct in _PERCENTILES:
        stats[f"p{pct}"] = _percentile(sorted_vals, pct)
    return stats


def _make_dist(counts: Sequence[int]) -> OrderedDict[int, int]:
    counter = Counter(counts)
    return OrderedDict(sorted(counter.items(), key=lambda item: item[_ZERO]))


def _collect_answer_counts(question_path: Path) -> Tuple[List[str], List[str], List[int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        question_path,
        columns=["question_uid", "answer_entity_ids", "split"],
        use_threads=False,
    )
    question_ids = table["question_uid"].to_pylist()
    answer_lists = table["answer_entity_ids"].to_pylist()
    splits = table["split"].to_pylist()
    counts = [len(items) if items is not None else _ZERO for items in answer_lists]
    return question_ids, splits, counts


def _partition_counts(
    question_ids: Sequence[str],
    splits: Sequence[str],
    counts: Sequence[int],
    sub_train_ids: set[str],
    sub_ids: set[str],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    full_by_split = {split: [] for split in _SPLITS}
    sub_by_split = {split: [] for split in _SPLITS}
    for sample_id, split, count in zip(question_ids, splits, counts):
        if split in full_by_split:
            full_by_split[split].append(int(count))
        if split == "train":
            if sample_id in sub_train_ids:
                sub_by_split["train"].append(int(count))
        elif split in ("validation", "test"):
            if sample_id in sub_ids:
                sub_by_split[split].append(int(count))
    return full_by_split, sub_by_split


def _answer_stats_from_counts(counts_by_split: Dict[str, List[int]]) -> Dict[str, Any]:
    overall = []
    for split in _SPLITS:
        overall.extend(counts_by_split[split])
    return {
        "overall": _summarize_counts(overall),
        "by_split": {split: _summarize_counts(counts_by_split[split]) for split in _SPLITS},
        "dist": {
            "overall": _make_dist(overall),
            "by_split": {split: _make_dist(counts_by_split[split]) for split in _SPLITS},
        },
    }


def compute_answer_stats(question_path: Path, sub_filter_path: Path, train_filter_path: Path) -> Dict[str, Any]:
    question_ids, splits, counts = _collect_answer_counts(question_path)
    sub_ids = _load_filter_ids(sub_filter_path)
    train_ids = _load_filter_ids(train_filter_path)
    full_by_split, sub_by_split = _partition_counts(question_ids, splits, counts, train_ids, sub_ids)
    return {
        "full": _answer_stats_from_counts(full_by_split),
        "sub": _answer_stats_from_counts(sub_by_split),
    }


def _open_lmdb(path: Path) -> lmdb.Environment:
    return lmdb.open(
        str(path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=_LMDB_MAX_READERS,
    )


def _load_sample_ids(path: Path) -> List[str]:
    env = _open_lmdb(path)
    keys: List[str] = []
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key in cursor.iternext(values=False):
            keys.append(key.decode("utf-8"))
    env.close()
    return keys


def _decode_distance_payload(payload: bytes, num_nodes: int) -> array:
    byte_len = len(payload)
    expected_int8 = num_nodes * _DISTANCE_BYTES_PER_INT8
    expected_int16 = num_nodes * _DISTANCE_BYTES_PER_INT16
    if byte_len == expected_int8:
        values = array("b")
    elif byte_len == expected_int16:
        values = array("h")
    else:
        raise ValueError(
            f"Distance cache length mismatch: bytes={byte_len} expected={expected_int8}/{expected_int16}."
        )
    values.frombytes(payload)
    return values


def _extract_min_distance(raw: Dict[str, Any], dist_payload: bytes) -> Tuple[Optional[int], str]:
    q_local = raw.get("q_local_indices")
    a_local = raw.get("a_local_indices")
    if q_local is None or a_local is None:
        return None, "missing"
    q_local = q_local.view(-1)
    a_local = a_local.view(-1)
    if q_local.numel() == _ZERO or a_local.numel() == _ZERO:
        return None, "missing"
    num_nodes = int(raw["num_nodes"])
    node_min_dists = _decode_distance_payload(dist_payload, num_nodes)
    min_dist = None
    for idx in q_local.tolist():
        value = node_min_dists[idx]
        if min_dist is None or value < min_dist:
            min_dist = value
    if min_dist is None:
        return None, "missing"
    if min_dist < _DIST_UNREACHABLE + _ONE:
        return None, "unreachable"
    return int(min_dist), "ok"


def _build_directed_adjacency(
    edge_index: Any, edge_attr: Any, num_nodes: int
) -> List[List[Tuple[int, int]]]:
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    rel_list = edge_attr.tolist()
    for src, dst, rel in zip(src_list, dst_list, rel_list):
        adjacency[src].append((int(dst), int(rel)))
    return adjacency


def _build_undirected_adjacency(
    edge_index: Any, edge_attr: Any, num_nodes: int
) -> List[List[Tuple[int, int, int]]]:
    adjacency: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_nodes)]
    src_list = edge_index[0].tolist()
    dst_list = edge_index[1].tolist()
    rel_list = edge_attr.tolist()
    for src, dst, rel in zip(src_list, dst_list, rel_list):
        adjacency[src].append((int(dst), int(rel), _ONE))
        adjacency[dst].append((int(src), int(rel), -_ONE))
    return adjacency


def _bfs_directed_shortest_path(
    *,
    adjacency: List[List[Tuple[int, int]]],
    sources: Sequence[int],
    targets: set[int],
    max_depth: int,
) -> Tuple[Optional[int], List[int], List[Optional[int]], Optional[int]]:
    from collections import deque

    num_nodes = len(adjacency)
    dist = [-_ONE] * num_nodes
    parent = [-_ONE] * num_nodes
    parent_rel: List[Optional[int]] = [None] * num_nodes
    queue = deque()
    for src in sources:
        if dist[src] != -_ONE:
            continue
        dist[src] = _ZERO
        queue.append(src)
    while queue:
        node = queue.popleft()
        depth = dist[node]
        if node in targets:
            return depth, parent, parent_rel, node
        if depth >= max_depth:
            continue
        for nbr, rel in adjacency[node]:
            if dist[nbr] != -_ONE:
                continue
            dist[nbr] = depth + _ONE
            parent[nbr] = node
            parent_rel[nbr] = rel
            queue.append(nbr)
    return None, parent, parent_rel, None


def _bfs_undirected_shortest_path(
    *,
    adjacency: List[List[Tuple[int, int, int]]],
    sources: Sequence[int],
    targets: set[int],
    max_depth: int,
) -> Tuple[Optional[int], List[int], List[Optional[int]], List[Optional[int]], Optional[int]]:
    from collections import deque

    num_nodes = len(adjacency)
    dist = [-_ONE] * num_nodes
    parent = [-_ONE] * num_nodes
    parent_rel: List[Optional[int]] = [None] * num_nodes
    parent_dir: List[Optional[int]] = [None] * num_nodes
    queue = deque()
    for src in sources:
        if dist[src] != -_ONE:
            continue
        dist[src] = _ZERO
        queue.append(src)
    while queue:
        node = queue.popleft()
        depth = dist[node]
        if node in targets:
            return depth, parent, parent_rel, parent_dir, node
        if depth >= max_depth:
            continue
        for nbr, rel, direction in adjacency[node]:
            if dist[nbr] != -_ONE:
                continue
            dist[nbr] = depth + _ONE
            parent[nbr] = node
            parent_rel[nbr] = rel
            parent_dir[nbr] = direction
            queue.append(nbr)
    return None, parent, parent_rel, parent_dir, None


def _reconstruct_path(
    parent: Sequence[int],
    parent_rel: Sequence[Optional[int]],
    target: int,
) -> Tuple[List[int], List[int]]:
    nodes: List[int] = [target]
    rels: List[int] = []
    curr = target
    while parent[curr] != -_ONE:
        rel = parent_rel[curr]
        if rel is None:
            break
        rels.append(int(rel))
        curr = parent[curr]
        nodes.append(curr)
    nodes.reverse()
    rels.reverse()
    return nodes, rels


def _reconstruct_undirected_path(
    parent: Sequence[int],
    parent_rel: Sequence[Optional[int]],
    parent_dir: Sequence[Optional[int]],
    target: int,
) -> Tuple[List[int], List[int], List[int]]:
    nodes: List[int] = [target]
    rels: List[int] = []
    dirs: List[int] = []
    curr = target
    while parent[curr] != -_ONE:
        rel = parent_rel[curr]
        direction = parent_dir[curr]
        if rel is None or direction is None:
            break
        rels.append(int(rel))
        dirs.append(int(direction))
        curr = parent[curr]
        nodes.append(curr)
    nodes.reverse()
    rels.reverse()
    dirs.reverse()
    return nodes, rels, dirs


def _collect_path_globals(node_global_ids: Any, path_nodes: Sequence[int]) -> List[int]:
    return [int(node_global_ids[idx].item()) for idx in path_nodes]


def _collect_global_ids(node_global_ids: Any, indices: Sequence[int]) -> List[int]:
    return [int(node_global_ids[idx].item()) for idx in indices]


def _as_index_list(values: Any) -> List[int]:
    return [int(val) for val in values.view(-1).tolist()]


def _should_collect_example(
    undirected_min: int,
    directed_min: Optional[int],
    max_depth: int,
) -> str:
    if directed_min is None:
        return "unreachable"
    if directed_min > _DIRECTED_GT2_THRESHOLD:
        return "gt2"
    if directed_min >= max_depth:
        return "gt_max_depth"
    return "skip"


def _compute_directed_path(
    raw: Dict[str, Any],
    sources: Sequence[int],
    targets: set[int],
    max_depth: int,
) -> Tuple[Optional[int], Optional[Tuple[List[int], List[int]]]]:
    adjacency = _build_directed_adjacency(
        raw["edge_index"], raw["edge_attr"], int(raw["num_nodes"])
    )
    directed_min, parent, parent_rel, target = _bfs_directed_shortest_path(
        adjacency=adjacency,
        sources=sources,
        targets=targets,
        max_depth=max_depth,
    )
    if directed_min is None or target is None:
        return directed_min, None
    return directed_min, _reconstruct_path(parent, parent_rel, target)


def _compute_undirected_path(
    raw: Dict[str, Any],
    sources: Sequence[int],
    targets: set[int],
    max_depth: int,
) -> Optional[Tuple[List[int], List[int], List[int]]]:
    undirected_adj = _build_undirected_adjacency(
        raw["edge_index"], raw["edge_attr"], int(raw["num_nodes"])
    )
    u_dist, u_parent, u_rel, u_dir, u_target = _bfs_undirected_shortest_path(
        adjacency=undirected_adj,
        sources=sources,
        targets=targets,
        max_depth=max_depth,
    )
    if u_dist is None or u_target is None:
        return None
    return _reconstruct_undirected_path(u_parent, u_rel, u_dir, u_target)


def _process_directed_sample(
    *,
    sample_id: str,
    split: str,
    raw: Dict[str, Any],
    dist_payload: bytes,
    max_depth: int,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    undirected_min, status = _extract_min_distance(raw, dist_payload)
    if status != "ok" or undirected_min is None:
        return "skip", None
    sources = _as_index_list(raw["q_local_indices"])
    targets = set(_as_index_list(raw["a_local_indices"]))
    directed_min, directed_path = _compute_directed_path(raw, sources, targets, max_depth)
    category = _should_collect_example(undirected_min, directed_min, max_depth)
    if category == "skip":
        return category, None
    undirected_path = None
    if undirected_min <= _UNDIRECTED_PATH_MAX_DEPTH:
        undirected_path = _compute_undirected_path(raw, sources, targets, _UNDIRECTED_PATH_MAX_DEPTH)
    payload = _build_example_payload(
        sample_id=sample_id,
        split=split,
        undirected_min=undirected_min,
        directed_min=directed_min,
        raw=raw,
        directed_path=directed_path,
        undirected_path=undirected_path,
    )
    return category, payload


def _build_example_payload(
    *,
    sample_id: str,
    split: str,
    undirected_min: int,
    directed_min: Optional[int],
    raw: Dict[str, Any],
    directed_path: Optional[Tuple[List[int], List[int]]],
    undirected_path: Optional[Tuple[List[int], List[int], List[int]]],
) -> Dict[str, Any]:
    q_local = _as_index_list(raw["q_local_indices"])
    a_local = _as_index_list(raw["a_local_indices"])
    node_global_ids = raw["node_global_ids"]
    payload = {
        "sample_id": sample_id,
        "split": split,
        "undirected_min": undirected_min,
        "directed_min": directed_min,
        "q_local_indices": q_local,
        "a_local_indices": a_local,
        "q_global_ids": _collect_global_ids(node_global_ids, q_local),
        "a_global_ids": _collect_global_ids(node_global_ids, a_local),
    }
    if directed_path is not None:
        nodes, rels = directed_path
        payload["directed_path_nodes"] = nodes
        payload["directed_path_rel_ids"] = rels
        payload["directed_path_global_ids"] = _collect_path_globals(node_global_ids, nodes)
    if undirected_path is not None:
        nodes, rels, dirs = undirected_path
        payload["undirected_path_nodes"] = nodes
        payload["undirected_path_rel_ids"] = rels
        payload["undirected_path_dirs"] = dirs
        payload["undirected_path_global_ids"] = _collect_path_globals(node_global_ids, nodes)
    return payload


def _collect_directed_examples_for_split(
    *,
    split: str,
    sample_ids: Sequence[str],
    emb_path: Path,
    dist_path: Path,
    max_depth: int,
    max_examples: int,
    progress_interval: int,
) -> Dict[str, Any]:
    examples_gt2: List[Dict[str, Any]] = []
    examples_unreachable: List[Dict[str, Any]] = []
    processed = _ZERO
    emb_env = _open_lmdb(emb_path)
    dist_env = _open_lmdb(dist_path)
    with emb_env.begin(write=False) as emb_txn, dist_env.begin(write=False) as dist_txn:
        for sample_id in sample_ids:
            processed += _ONE
            if progress_interval > _ZERO and processed % progress_interval == _ZERO:
                print(f"{split}: processed {processed}", flush=True)
            key = sample_id.encode("utf-8")
            payload = emb_txn.get(key)
            if payload is None:
                continue
            dist_payload = dist_txn.get(key)
            if dist_payload is None:
                continue
            raw = _deserialize_sample(payload)
            category, example = _process_directed_sample(
                sample_id=sample_id,
                split=split,
                raw=raw,
                dist_payload=dist_payload,
                max_depth=max_depth,
            )
            if category == "skip" or example is None:
                continue
            if category == "gt2":
                examples_gt2.append(example)
            else:
                examples_unreachable.append(example)
            if len(examples_gt2) >= max_examples and len(examples_unreachable) >= max_examples:
                break
    emb_env.close()
    dist_env.close()
    return {
        "processed": processed,
        "gt2": examples_gt2,
        "unreachable_or_gt_max": examples_unreachable,
    }


def compute_directed_examples(
    emb_root: Path,
    dist_root: Path,
    sub_filter_path: Path,
    train_filter_path: Path,
    scope: str,
    max_depth: int,
    max_examples: int,
    progress_interval: int,
) -> Dict[str, Any]:
    import torch

    torch.set_num_threads(_ONE)
    torch.set_num_interop_threads(_ONE)
    sample_ids_by_split = {split: _load_sample_ids(emb_root / f"{split}.lmdb") for split in _SPLITS}
    if scope == "sub":
        sub_ids = _load_filter_ids(sub_filter_path)
        train_ids = _load_filter_ids(train_filter_path)
        sample_ids_by_split = {
            "train": [sid for sid in sample_ids_by_split["train"] if sid in train_ids],
            "validation": [sid for sid in sample_ids_by_split["validation"] if sid in sub_ids],
            "test": [sid for sid in sample_ids_by_split["test"] if sid in sub_ids],
        }
    results = {}
    for split in _SPLITS:
        results[split] = _collect_directed_examples_for_split(
            split=split,
            sample_ids=sample_ids_by_split[split],
            emb_path=emb_root / f"{split}.lmdb",
            dist_path=dist_root / f"{split}.lmdb",
            max_depth=max_depth,
            max_examples=max_examples,
            progress_interval=progress_interval,
        )
    return {
        "scope": scope,
        "max_depth": max_depth,
        "max_examples_per_kind": max_examples,
        "splits": results,
    }


def _compute_distance_split(
    sample_ids: Sequence[str],
    emb_path: Path,
    dist_path: Path,
    progress_interval: int,
) -> Tuple[List[int], Dict[str, int]]:
    lengths: List[int] = []
    missing_samples = _ZERO
    unreachable_samples = _ZERO
    total_samples = _ZERO
    emb_env = _open_lmdb(emb_path)
    dist_env = _open_lmdb(dist_path)
    with emb_env.begin(write=False) as emb_txn, dist_env.begin(write=False) as dist_txn:
        for sample_id in sample_ids:
            total_samples += _ONE
            if progress_interval > _ZERO and total_samples % progress_interval == _ZERO:
                print(f"{emb_path.stem}: processed {total_samples}", flush=True)
            key = sample_id.encode("utf-8")
            payload = emb_txn.get(key)
            if payload is None:
                missing_samples += _ONE
                continue
            dist_payload = dist_txn.get(key)
            if dist_payload is None:
                missing_samples += _ONE
                continue
            raw = _deserialize_sample(payload)
            min_dist, status = _extract_min_distance(raw, dist_payload)
            if status == "ok" and min_dist is not None:
                lengths.append(min_dist)
            elif status == "unreachable":
                unreachable_samples += _ONE
            else:
                missing_samples += _ONE
    emb_env.close()
    dist_env.close()
    meta = {
        "total_samples": total_samples,
        "missing_samples": missing_samples,
        "unreachable_samples": unreachable_samples,
        "valid_samples": len(lengths),
    }
    return lengths, meta


def _distance_stats_from_lengths(
    lengths_by_split: Dict[str, List[int]],
    meta_by_split: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    overall_lengths: List[int] = []
    for split in _SPLITS:
        overall_lengths.extend(lengths_by_split[split])
    overall_meta = {
        "total_samples": sum(meta_by_split[split]["total_samples"] for split in _SPLITS),
        "missing_samples": sum(meta_by_split[split]["missing_samples"] for split in _SPLITS),
        "unreachable_samples": sum(meta_by_split[split]["unreachable_samples"] for split in _SPLITS),
        "valid_samples": len(overall_lengths),
    }
    return {
        "overall": _summarize_counts(overall_lengths),
        "by_split": {split: _summarize_counts(lengths_by_split[split]) for split in _SPLITS},
        "dist": {
            "overall": _make_dist(overall_lengths),
            "by_split": {split: _make_dist(lengths_by_split[split]) for split in _SPLITS},
        },
        "meta": {"overall": overall_meta, "by_split": meta_by_split},
    }


def compute_distance_stats(
    emb_root: Path,
    dist_root: Path,
    sub_filter_path: Path,
    train_filter_path: Path,
    progress_interval: int,
) -> Dict[str, Any]:
    import torch

    torch.set_num_threads(_ONE)
    torch.set_num_interop_threads(_ONE)

    sample_ids_by_split = {split: _load_sample_ids(emb_root / f"{split}.lmdb") for split in _SPLITS}
    sub_ids = _load_filter_ids(sub_filter_path)
    train_ids = _load_filter_ids(train_filter_path)
    sub_sample_ids_by_split = {
        "train": [sid for sid in sample_ids_by_split["train"] if sid in train_ids],
        "validation": [sid for sid in sample_ids_by_split["validation"] if sid in sub_ids],
        "test": [sid for sid in sample_ids_by_split["test"] if sid in sub_ids],
    }
    full_lengths = {}
    full_meta = {}
    sub_lengths = {}
    sub_meta = {}
    for split in _SPLITS:
        lengths, meta = _compute_distance_split(
            sample_ids_by_split[split],
            emb_root / f"{split}.lmdb",
            dist_root / f"{split}.lmdb",
            progress_interval,
        )
        full_lengths[split] = lengths
        full_meta[split] = meta
    for split in _SPLITS:
        lengths, meta = _compute_distance_split(
            sub_sample_ids_by_split[split],
            emb_root / f"{split}.lmdb",
            dist_root / f"{split}.lmdb",
            progress_interval,
        )
        sub_lengths[split] = lengths
        sub_meta[split] = meta
    return {
        "full": _distance_stats_from_lengths(full_lengths, full_meta),
        "sub": _distance_stats_from_lengths(sub_lengths, sub_meta),
    }


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{_FLOAT_PRECISION}f}"
    return str(value)


def _build_summary_table(title: str, stats: Dict[str, Any]) -> str:
    lines = [f"#### {title}", ""]
    lines.append("| split | n | min | max | mean | p50 | p90 | p95 | p99 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    rows = [("overall", stats["overall"])]
    rows.extend((split, stats["by_split"][split]) for split in _SPLITS)
    for split, row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(row.get("n", _ZERO)),
                    _format_value(row.get("min")),
                    _format_value(row.get("max")),
                    _format_value(row.get("mean")),
                    _format_value(row.get("p50")),
                    _format_value(row.get("p90")),
                    _format_value(row.get("p95")),
                    _format_value(row.get("p99")),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_meta_table(title: str, meta: Dict[str, Any]) -> str:
    lines = [f"#### {title}", ""]
    lines.append("| split | total_samples | valid_samples | missing_samples | unreachable_samples |")
    lines.append("| --- | --- | --- | --- | --- |")
    rows = [("overall", meta["overall"])]
    rows.extend((split, meta["by_split"][split]) for split in _SPLITS)
    for split, row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(row.get("total_samples", _ZERO)),
                    str(row.get("valid_samples", _ZERO)),
                    str(row.get("missing_samples", _ZERO)),
                    str(row.get("unreachable_samples", _ZERO)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _build_distribution_table(title: str, dist: Dict[str, Any]) -> str:
    lines = [f"#### {title}", ""]
    lines.append("| value | overall | train | validation | test |")
    lines.append("| --- | --- | --- | --- | --- |")
    overall = _normalize_dist_keys(dist["overall"])
    by_split = {split: _normalize_dist_keys(dist["by_split"][split]) for split in _SPLITS}
    all_keys = set(overall.keys())
    for split in _SPLITS:
        all_keys.update(by_split[split].keys())
    for key in sorted(all_keys, key=_dist_sort_key):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(key),
                    str(overall.get(key, _ZERO)),
                    str(by_split["train"].get(key, _ZERO)),
                    str(by_split["validation"].get(key, _ZERO)),
                    str(by_split["test"].get(key, _ZERO)),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _normalize_dist_keys(values: Dict[Any, int]) -> Dict[Any, int]:
    normalized: Dict[Any, int] = {}
    for key, value in values.items():
        if isinstance(key, int):
            normalized[key] = value
            continue
        if isinstance(key, str) and key.lstrip("-").isdigit():
            normalized[int(key)] = value
            continue
        normalized[key] = value
    return normalized


def _dist_sort_key(value: Any) -> Tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    return (1, str(value))


def render_markdown(answer_stats: Dict[str, Any], distance_stats: Dict[str, Any]) -> str:
    lines = [
        "# CWQ Dataset Statistics",
        "",
        "## Answer Count Distribution",
        "",
        "Source: `normalized/questions.parquet` (Gold answer_entity_ids).",
        "",
        "### Full",
        "",
        _build_summary_table("Answer Count Summary", answer_stats["full"]),
        _build_distribution_table("Answer Count Distribution", answer_stats["full"]["dist"]),
        "### Sub",
        "",
        _build_summary_table("Answer Count Summary", answer_stats["sub"]),
        _build_distribution_table("Answer Count Distribution", answer_stats["sub"]["dist"]),
        "## Shortest Distance (q -> a)",
        "",
        "Source: `materialized/embeddings/*.lmdb` + `materialized/distances/*.pt`.",
        "",
        "Definition: min distance from any q_local_indices to any a_local_indices in g_retrieval.",
        "Unreachable samples have distance = -1 and are counted in coverage.",
        "",
        "### Full",
        "",
        _build_meta_table("Coverage", distance_stats["full"]["meta"]),
        _build_summary_table("Shortest Distance Summary", distance_stats["full"]),
        _build_distribution_table("Shortest Distance Distribution", distance_stats["full"]["dist"]),
        "### Sub",
        "",
        _build_meta_table("Coverage", distance_stats["sub"]["meta"]),
        _build_summary_table("Shortest Distance Summary", distance_stats["sub"]),
        _build_distribution_table("Shortest Distance Distribution", distance_stats["sub"]["dist"]),
        "",
    ]
    return "\n".join(lines)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _apply_env_defaults(env: Dict[str, str]) -> Dict[str, str]:
    updated = env.copy()
    updated.setdefault(_ENV_SHM_DISABLE, "1")
    for name in _ENV_THREADS:
        updated.setdefault(name, "1")
    return updated


def run_answer(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    cwq_dir = data_dir / _CWQ_DIR / _NORMALIZED_DIR
    question_path = cwq_dir / _QUESTIONS_FILENAME
    sub_filter_path = cwq_dir / _SUB_FILTER_FILENAME
    train_filter_path = cwq_dir / _TRAIN_FILTER_FILENAME
    stats = compute_answer_stats(question_path, sub_filter_path, train_filter_path)
    payload = {
        "dataset": "cwq",
        "source": str(question_path),
        "filters": {
            "sub_filter": str(sub_filter_path),
            "train_filter": str(train_filter_path),
        },
        **stats,
    }
    out_path = Path(args.out) if args.out else _ANSWER_JSON_DEFAULT
    _write_json(out_path, payload)
    print(f"Wrote answer stats to {out_path}")


def run_distance(args: argparse.Namespace) -> None:
    os.environ.setdefault(_ENV_SHM_DISABLE, "1")
    for name in _ENV_THREADS:
        os.environ.setdefault(name, "1")
    data_dir = _resolve_data_dir(args.data_dir)
    cwq_dir = data_dir / _CWQ_DIR
    emb_root = cwq_dir / _MATERIALIZED_DIR / _EMBEDDINGS_DIR
    dist_root = cwq_dir / _MATERIALIZED_DIR / _DISTANCES_DIR
    sub_filter_path = cwq_dir / _NORMALIZED_DIR / _SUB_FILTER_FILENAME
    train_filter_path = cwq_dir / _NORMALIZED_DIR / _TRAIN_FILTER_FILENAME
    stats = compute_distance_stats(
        emb_root,
        dist_root,
        sub_filter_path,
        train_filter_path,
        int(args.progress_interval),
    )
    payload = {
        "dataset": "cwq",
        "source": {"embeddings": str(emb_root), "distances": str(dist_root)},
        "filters": {
            "sub_filter": str(sub_filter_path),
            "train_filter": str(train_filter_path),
        },
        **stats,
    }
    out_path = Path(args.out) if args.out else _DISTANCE_JSON_DEFAULT
    _write_json(out_path, payload)
    print(f"Wrote distance stats to {out_path}")


def run_render(args: argparse.Namespace) -> None:
    answer_stats = _read_json(Path(args.answer_json))
    distance_stats = _read_json(Path(args.distance_json))
    markdown = render_markdown(answer_stats, distance_stats)
    docs_path = Path(args.docs_path)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(markdown)
    print(f"Wrote markdown to {docs_path}")


def run_directed_examples(args: argparse.Namespace) -> None:
    os.environ.setdefault(_ENV_SHM_DISABLE, "1")
    for name in _ENV_THREADS:
        os.environ.setdefault(name, "1")
    data_dir = _resolve_data_dir(args.data_dir)
    cwq_dir = data_dir / _CWQ_DIR
    emb_root = cwq_dir / _MATERIALIZED_DIR / _EMBEDDINGS_DIR
    dist_root = cwq_dir / _MATERIALIZED_DIR / _DISTANCES_DIR
    sub_filter_path = cwq_dir / _NORMALIZED_DIR / _SUB_FILTER_FILENAME
    train_filter_path = cwq_dir / _NORMALIZED_DIR / _TRAIN_FILTER_FILENAME
    stats = compute_directed_examples(
        emb_root=emb_root,
        dist_root=dist_root,
        sub_filter_path=sub_filter_path,
        train_filter_path=train_filter_path,
        scope=args.scope,
        max_depth=int(args.max_depth),
        max_examples=int(args.max_examples),
        progress_interval=int(args.directed_progress_interval),
    )
    payload = {
        "dataset": "cwq",
        "source": {"embeddings": str(emb_root), "distances": str(dist_root)},
        "filters": {"sub_filter": str(sub_filter_path), "train_filter": str(train_filter_path)},
        **stats,
    }
    out_path = Path(args.examples_json)
    _write_json(out_path, payload)
    print(f"Wrote directed examples to {out_path}")


def run_all(args: argparse.Namespace) -> None:
    data_dir = _resolve_data_dir(args.data_dir)
    answer_json = Path(args.answer_json)
    distance_json = Path(args.distance_json)
    docs_path = Path(args.docs_path)
    env = _apply_env_defaults(os.environ)
    base_cmd = [sys.executable, str(Path(__file__).resolve())]
    subprocess.check_call(
        base_cmd
        + [
            "--step",
            "answer",
            "--data-dir",
            str(data_dir),
            "--out",
            str(answer_json),
        ],
        env=env,
    )
    subprocess.check_call(
        base_cmd
        + [
            "--step",
            "distance",
            "--data-dir",
            str(data_dir),
            "--out",
            str(distance_json),
            "--progress-interval",
            str(args.progress_interval),
        ],
        env=env,
    )
    subprocess.check_call(
        base_cmd
        + [
            "--step",
            "render",
            "--answer-json",
            str(answer_json),
            "--distance-json",
            str(distance_json),
            "--docs-path",
            str(docs_path),
        ],
        env=env,
    )


def _add_base_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--step",
        choices=("answer", "distance", "render", "directed-examples", "all"),
        required=True,
        help="Which step to run.",
    )
    parser.add_argument("--data-dir", default=None, help="Data directory root.")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument(
        "--answer-json",
        default=str(_ANSWER_JSON_DEFAULT),
        help="Answer stats JSON (for render/all).",
    )
    parser.add_argument(
        "--distance-json",
        default=str(_DISTANCE_JSON_DEFAULT),
        help="Distance stats JSON (for render/all).",
    )
    parser.add_argument(
        "--docs-path",
        default=str(_DOCS_PATH_DEFAULT),
        help="Docs markdown output path.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=_PROGRESS_INTERVAL_DEFAULT,
        help="Progress logging interval (distance step).",
    )


def _add_directed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--directed-progress-interval",
        type=int,
        default=_DIRECTED_PROGRESS_INTERVAL_DEFAULT,
        help="Progress logging interval (directed examples step).",
    )
    parser.add_argument(
        "--examples-json",
        default=str(_DIRECTED_EXAMPLES_JSON_DEFAULT),
        help="Output JSON for directed examples.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=_DIRECTED_MAX_DEPTH_DEFAULT,
        help="Max depth for directed BFS.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=_DIRECTED_MAX_EXAMPLES_DEFAULT,
        help="Max examples per category (gt2/unreachable).",
    )
    parser.add_argument(
        "--scope",
        choices=("full", "sub"),
        default="full",
        help="Dataset scope for directed example search.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CWQ dataset stats to markdown.")
    _add_base_args(parser)
    _add_directed_args(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.step == "answer":
        run_answer(args)
        return
    if args.step == "distance":
        run_distance(args)
        return
    if args.step == "render":
        run_render(args)
        return
    if args.step == "directed-examples":
        run_directed_examples(args)
        return
    if args.step == "all":
        run_all(args)
        return
    raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()
