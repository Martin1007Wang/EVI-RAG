#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lmdb

_REPO_PARENT_LEVELS = 1
_REPO_ROOT = Path(__file__).resolve().parents[_REPO_PARENT_LEVELS]
sys.path.insert(0, str(_REPO_ROOT))

from src.data.io.lmdb_utils import _deserialize_sample

_DEFAULT_DATA_DIR = Path("/mnt/data/retrieval_dataset")
_DEFAULT_OUTPUT_DIR = Path("outputs")
_DEFAULT_DATASET = "cwq"
_DEFAULT_SPLIT = "train"
_DEFAULT_TOP_K = 50
_DEFAULT_PROGRESS_EVERY = 1000
_DEFAULT_BATCH_SIZE = 65536
_LMDB_MAX_READERS = 256
_DIST_UNREACHABLE = -1
_ZERO = 0
_ONE = 1
_DEFAULT_ALLOW_INVALID = False

_ENV_LIMITS = {
    "KMP_SHM_DISABLE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def _set_env_limits() -> None:
    for key, value in _ENV_LIMITS.items():
        os.environ[key] = value


_set_env_limits()

import torch
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.components.distance_store import DistancePTStore


def _resolve_lmdb_path(data_dir: Path, dataset: str, split: str) -> Path:
    return data_dir / dataset / "materialized" / "embeddings" / f"{split}.lmdb"


def _resolve_distance_path(data_dir: Path, dataset: str, split: str, distances_dir: Optional[Path]) -> Path:
    base = distances_dir if distances_dir is not None else data_dir / dataset / "materialized" / "distances"
    return base / f"{split}.pt"


def _resolve_vocab_paths(data_dir: Path, dataset: str) -> Tuple[Path, Path]:
    base = data_dir / dataset / "normalized"
    return base / "entity_vocab.parquet", base / "relation_vocab.parquet"


def _load_max_ids(entity_vocab: Path, relation_vocab: Path) -> Tuple[int, int]:
    entity_col = pq.read_table(entity_vocab, columns=["entity_id"]).column(0)
    relation_col = pq.read_table(relation_vocab, columns=["relation_id"]).column(0)
    return int(pc.max(entity_col).as_py()), int(pc.max(relation_col).as_py())


def _compute_bits(max_value: int) -> int:
    return max(max_value.bit_length(), 1)


def _pack_edge(src: int, dst: int, rel: int, shift_src: int, shift_dst: int) -> int:
    return (src << shift_src) | (dst << shift_dst) | rel


def _unpack_edge(packed: int, shift_src: int, shift_dst: int, node_mask: int, rel_mask: int) -> Tuple[int, int, int]:
    src = packed >> shift_src
    dst = (packed >> shift_dst) & node_mask
    rel = packed & rel_mask
    return src, dst, rel


@dataclass(frozen=True)
class Packing:
    node_bits: int
    rel_bits: int
    shift_dst: int
    shift_src: int
    node_mask: int
    rel_mask: int


@dataclass(frozen=True)
class HubStats:
    processed: int
    unique_edges: int
    unique_nodes: int
    edges: set[int]
    indeg: Dict[int, int]
    outdeg: Dict[int, int]
    hub_nodes: List[Tuple[int, int]]
    in_rel: Dict[int, Counter]
    out_rel: Dict[int, Counter]


@dataclass(frozen=True)
class DistanceStats:
    processed: int
    invalid: int
    invalid_reasons: Dict[str, int]
    reachable: int
    unreachable: int
    min_distance: Optional[int]
    max_distance: Optional[int]
    mean_distance: Optional[float]
    dist: Dict[int, int]


@dataclass
class DistanceAccumulator:
    dist_counter: Counter
    invalid_reasons: Counter
    processed: int = _ZERO
    invalid: int = _ZERO
    reachable: int = _ZERO
    unreachable: int = _ZERO
    sum_dist: int = _ZERO
    min_dist: Optional[int] = None
    max_dist: Optional[int] = None

    def record_invalid(self, reason: str) -> None:
        self.invalid += _ONE
        self.invalid_reasons[reason] += _ONE

    def update(self, min_distance: int) -> None:
        self.dist_counter[min_distance] += _ONE
        if min_distance == _DIST_UNREACHABLE:
            self.unreachable += _ONE
            return
        self.reachable += _ONE
        self.sum_dist += min_distance
        self.min_dist = min_distance if self.min_dist is None else min(self.min_dist, min_distance)
        self.max_dist = min_distance if self.max_dist is None else max(self.max_dist, min_distance)

    def to_stats(self) -> DistanceStats:
        mean_dist = float(self.sum_dist) / float(self.reachable) if self.reachable > _ZERO else None
        return DistanceStats(
            processed=self.processed,
            invalid=self.invalid,
            invalid_reasons=dict(self.invalid_reasons),
            reachable=self.reachable,
            unreachable=self.unreachable,
            min_distance=self.min_dist,
            max_distance=self.max_dist,
            mean_distance=mean_dist,
            dist=dict(self.dist_counter),
        )


def _pack_edges_from_raw(raw: dict, shift_src: int, shift_dst: int) -> set[int]:
    edge_index = raw["edge_index"]
    edge_attr = raw["edge_attr"]
    node_global = raw["node_global_ids"]
    src_globals = node_global[edge_index[0]].tolist()
    dst_globals = node_global[edge_index[1]].tolist()
    rel_ids = edge_attr.tolist()
    return set(_pack_edge(s, d, r, shift_src, shift_dst) for s, d, r in zip(src_globals, dst_globals, rel_ids))


def _apply_new_edges(
    edges: set[int],
    indeg: Dict[int, int],
    outdeg: Dict[int, int],
    packed_edges: set[int],
    *,
    shift_src: int,
    shift_dst: int,
    node_mask: int,
) -> None:
    if not packed_edges:
        return
    new_edges = packed_edges.difference(edges)
    if not new_edges:
        return
    edges.update(new_edges)
    for packed in new_edges:
        src = packed >> shift_src
        dst = (packed >> shift_dst) & node_mask
        outdeg[src] += 1
        indeg[dst] += 1


def _iter_lmdb_edges(
    lmdb_path: Path,
    *,
    shift_src: int,
    shift_dst: int,
    node_mask: int,
    progress_every: int,
) -> Tuple[int, int, set[int], Dict[int, int], Dict[int, int]]:
    edges: set[int] = set()
    indeg: Dict[int, int] = defaultdict(int)
    outdeg: Dict[int, int] = defaultdict(int)
    processed = 0
    start = time.time()
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=_LMDB_MAX_READERS,
    )
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for _, val in cursor:
            raw = _deserialize_sample(val)
            edges_local = _pack_edges_from_raw(raw, shift_src, shift_dst)
            _apply_new_edges(edges, indeg, outdeg, edges_local, shift_src=shift_src, shift_dst=shift_dst, node_mask=node_mask)
            processed += 1
            if progress_every and processed % progress_every == 0:
                elapsed = time.time() - start
                print(
                    f"processed {processed} samples | unique_edges={len(edges)} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
    env.close()
    return processed, len(edges), edges, indeg, outdeg


def _decode_sample_id(key: bytes) -> str:
    try:
        return key.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("LMDB keys must be ASCII sample_ids for distance lookup.") from exc


def _extract_num_nodes(raw: dict, sample_id: str) -> int:
    num_nodes = raw.get("num_nodes")
    if num_nodes is None:
        node_ids = raw.get("node_global_ids")
        if node_ids is None:
            raise KeyError(f"node_global_ids missing for {sample_id}.")
        num_nodes = int(node_ids.numel() if torch.is_tensor(node_ids) else len(node_ids))
    elif torch.is_tensor(num_nodes):
        num_nodes = int(num_nodes.item())
    else:
        num_nodes = int(num_nodes)
    if num_nodes < _ZERO:
        raise ValueError(f"num_nodes must be >= {_ZERO}, got {num_nodes} for {sample_id}.")
    return num_nodes


def _to_long_tensor(value: object, name: str, sample_id: str) -> torch.Tensor:
    if value is None:
        raise KeyError(f"{name} missing for {sample_id}.")
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if tensor.numel() == _ZERO:
        return tensor.to(dtype=torch.long).reshape(-1)
    if tensor.dtype != torch.long:
        tensor = tensor.to(dtype=torch.long)
    return tensor.reshape(-1)


def _update_entity_counts(counter: Counter, ids: torch.Tensor) -> None:
    if ids.numel() == _ZERO:
        return
    unique_ids, counts = torch.unique(ids, return_counts=True)
    counter.update(dict(zip(unique_ids.tolist(), counts.tolist())))


def _get_distance_sample_data(
    raw: dict,
    sample_id: str,
    distance_store: DistancePTStore,
    allow_invalid: bool,
    acc: DistanceAccumulator,
) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
    num_nodes = _extract_num_nodes(raw, sample_id)
    q_local = _to_long_tensor(raw.get("q_local_indices"), "q_local_indices", sample_id)
    if q_local.numel() == _ZERO:
        acc.record_invalid("q_local_empty")
        return None
    q_min = int(q_local.min().item())
    q_max = int(q_local.max().item())
    if q_min < _ZERO or q_max >= num_nodes:
        if allow_invalid:
            acc.record_invalid("q_local_out_of_range")
            return None
        raise ValueError(f"q_local_indices out of range for {sample_id}: [{q_min}, {q_max}] vs {num_nodes}.")
    a_local = _to_long_tensor(raw.get("a_local_indices"), "a_local_indices", sample_id)
    if a_local.numel() == _ZERO:
        acc.record_invalid("a_local_empty")
        return None
    a_min = int(a_local.min().item())
    a_max = int(a_local.max().item())
    if a_min < _ZERO or a_max >= num_nodes:
        if allow_invalid:
            acc.record_invalid("a_local_out_of_range")
            return None
        raise ValueError(f"a_local_indices out of range for {sample_id}: [{a_min}, {a_max}] vs {num_nodes}.")
    node_global_ids = _to_long_tensor(raw.get("node_global_ids"), "node_global_ids", sample_id)
    if int(node_global_ids.numel()) != num_nodes:
        raise ValueError(f"node_global_ids length mismatch for {sample_id}: {node_global_ids.numel()} vs {num_nodes}.")
    start_globals = node_global_ids.index_select(0, q_local)
    answer_globals = node_global_ids.index_select(0, a_local)
    dist = distance_store.load(sample_id, num_nodes)
    min_distance = int(dist.index_select(0, q_local).min().item())
    return min_distance, start_globals, answer_globals


def _compute_distance_stats(
    lmdb_path: Path,
    distance_path: Path,
    *,
    progress_every: int,
    allow_invalid: bool,
) -> Tuple[DistanceStats, Counter, Counter]:
    if not distance_path.exists():
        raise FileNotFoundError(f"Distance PT not found: {distance_path}")
    distance_store = DistancePTStore(distance_path)
    start_counts: Counter = Counter()
    answer_counts: Counter = Counter()
    acc = DistanceAccumulator(Counter(), Counter())
    start_time = time.time()
    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=_LMDB_MAX_READERS,
    )
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            acc.processed += _ONE
            sample_id = _decode_sample_id(key)
            raw = _deserialize_sample(val)
            sample = _get_distance_sample_data(raw, sample_id, distance_store, allow_invalid, acc)
            if sample is None:
                continue
            min_distance, start_globals, answer_globals = sample
            _update_entity_counts(start_counts, start_globals)
            _update_entity_counts(answer_counts, answer_globals)
            acc.update(min_distance)
            if progress_every and acc.processed % progress_every == _ZERO:
                elapsed = time.time() - start_time
                print(
                    f"distance processed {acc.processed} samples | unreachable={acc.unreachable} | elapsed={elapsed:.1f}s",
                    flush=True,
                )
    env.close()
    return acc.to_stats(), start_counts, answer_counts


def _top_k_hubs(indeg: Dict[int, int], outdeg: Dict[int, int], top_k: int) -> List[Tuple[int, int]]:
    nodes = set(indeg) | set(outdeg)
    total_deg = {node: indeg.get(node, 0) + outdeg.get(node, 0) for node in nodes}
    return sorted(total_deg.items(), key=lambda item: item[1], reverse=True)[:top_k]


def _count_relations(
    edges: Iterable[int],
    hub_ids: set[int],
    *,
    shift_src: int,
    shift_dst: int,
    node_mask: int,
    rel_mask: int,
) -> Tuple[Dict[int, Counter], Dict[int, Counter]]:
    in_rel = {node: Counter() for node in hub_ids}
    out_rel = {node: Counter() for node in hub_ids}
    for packed in edges:
        src, dst, rel = _unpack_edge(packed, shift_src, shift_dst, node_mask, rel_mask)
        if src in hub_ids:
            out_rel[src][rel] += 1
        if dst in hub_ids:
            in_rel[dst][rel] += 1
    return in_rel, out_rel


def _load_relation_labels(relation_vocab: Path) -> Dict[int, str]:
    table = pq.read_table(relation_vocab, columns=["relation_id", "label"])
    rel_ids = table.column(0).to_pylist()
    labels = table.column(1).to_pylist()
    return {int(rid): str(label) for rid, label in zip(rel_ids, labels)}


def _load_entity_labels(entity_vocab: Path, target_ids: Sequence[int], batch_size: int) -> Dict[int, str]:
    targets = sorted(set(target_ids))
    if not targets:
        return {}
    target_set = pa.array(targets, type=pa.int64())
    lookup = pc.SetLookupOptions(value_set=target_set)
    mapping: Dict[int, str] = {}
    parquet = pq.ParquetFile(entity_vocab)
    for batch in parquet.iter_batches(columns=["entity_id", "label"], batch_size=batch_size):
        mask = pc.is_in(batch.column(0), options=lookup)
        if pc.any(mask).as_py():
            filtered = batch.filter(mask)
            ids = filtered.column(0).to_pylist()
            labels = filtered.column(1).to_pylist()
            for eid, label in zip(ids, labels):
                mapping[int(eid)] = str(label)
        if len(mapping) >= len(targets):
            break
    return mapping


def _build_node_entries(
    hub_nodes: List[Tuple[int, int]],
    indeg: Dict[int, int],
    outdeg: Dict[int, int],
    in_rel: Dict[int, Counter],
    out_rel: Dict[int, Counter],
    entity_labels: Optional[Dict[int, str]],
    relation_labels: Optional[Dict[int, str]],
    start_counts: Optional[Counter],
    answer_counts: Optional[Counter],
) -> List[Dict[str, object]]:
    nodes_out: List[Dict[str, object]] = []
    for node, total in hub_nodes:
        in_list = [
            {"relation_id": int(rid), "count": int(cnt), "label": relation_labels.get(rid, str(rid)) if relation_labels else None}
            for rid, cnt in in_rel[node].most_common()
        ]
        out_list = [
            {"relation_id": int(rid), "count": int(cnt), "label": relation_labels.get(rid, str(rid)) if relation_labels else None}
            for rid, cnt in out_rel[node].most_common()
        ]
        start_count = int(start_counts.get(node, _ZERO)) if start_counts is not None else _ZERO
        answer_count = int(answer_counts.get(node, _ZERO)) if answer_counts is not None else _ZERO
        nodes_out.append(
            {
                "entity_id": int(node),
                "label": entity_labels.get(node, str(node)) if entity_labels else None,
                "total_degree": int(total),
                "in_degree": int(indeg.get(node, 0)),
                "out_degree": int(outdeg.get(node, 0)),
                "start_count": start_count,
                "answer_count": answer_count,
                "in_relations": in_list,
                "out_relations": out_list,
            }
        )
    return nodes_out


def _build_payload(
    *,
    dataset: str,
    split: str,
    lmdb_path: Path,
    distance_path: Path,
    entity_vocab: Optional[Path],
    relation_vocab: Optional[Path],
    top_k: int,
    processed: int,
    unique_edges: int,
    unique_nodes: int,
    nodes_out: List[Dict[str, object]],
    node_bits: int,
    rel_bits: int,
    distance_stats: DistanceStats,
) -> Dict[str, object]:
    return {
        "dataset": dataset,
        "split": split,
        "graph": {
            "mode": "union",
            "dedupe": "edge(src_global_id, dst_global_id, relation_id)",
            "degree": "in+out",
            "top_k": top_k,
            "packing": {"node_bits": node_bits, "relation_bits": rel_bits},
        },
        "source": {
            "lmdb": str(lmdb_path),
            "distances_pt": str(distance_path),
            "entity_vocab": str(entity_vocab) if entity_vocab else None,
            "relation_vocab": str(relation_vocab) if relation_vocab else None,
        },
        "stats": {
            "samples_processed": processed,
            "unique_edges": unique_edges,
            "unique_nodes": unique_nodes,
        },
        "distance": {
            "processed": distance_stats.processed,
            "invalid": distance_stats.invalid,
            "invalid_reasons": distance_stats.invalid_reasons,
            "reachable": distance_stats.reachable,
            "unreachable": distance_stats.unreachable,
            "min": distance_stats.min_distance,
            "max": distance_stats.max_distance,
            "mean": distance_stats.mean_distance,
            "dist": _format_distance_dist(distance_stats.dist),
        },
        "nodes": nodes_out,
    }


def _format_distance_dist(dist: Dict[int, int]) -> Dict[str, int]:
    return {str(key): int(dist[key]) for key in sorted(dist)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute hub-node relation stats and QA distance distribution.")
    parser.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR)
    parser.add_argument("--dataset", type=str, default=_DEFAULT_DATASET)
    parser.add_argument("--split", type=str, default=_DEFAULT_SPLIT)
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("--progress-every", type=int, default=_DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument("--with-labels", action="store_true", default=False)
    parser.add_argument("--distances-dir", type=Path, default=None)
    parser.add_argument("--allow-invalid", action="store_true", default=_DEFAULT_ALLOW_INVALID)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _resolve_packing(entity_vocab: Path, relation_vocab: Path) -> Packing:
    max_entity_id, max_rel_id = _load_max_ids(entity_vocab, relation_vocab)
    node_bits = _compute_bits(max_entity_id)
    rel_bits = _compute_bits(max_rel_id)
    shift_dst = rel_bits
    shift_src = rel_bits + node_bits
    node_mask = (1 << node_bits) - 1
    rel_mask = (1 << rel_bits) - 1
    return Packing(
        node_bits=node_bits,
        rel_bits=rel_bits,
        shift_dst=shift_dst,
        shift_src=shift_src,
        node_mask=node_mask,
        rel_mask=rel_mask,
    )


def _compute_hub_stats(lmdb_path: Path, packing: Packing, top_k: int, progress_every: int) -> HubStats:
    processed, unique_edges, edges, indeg, outdeg = _iter_lmdb_edges(
        lmdb_path,
        shift_src=packing.shift_src,
        shift_dst=packing.shift_dst,
        node_mask=packing.node_mask,
        progress_every=progress_every,
    )
    hub_nodes = _top_k_hubs(indeg, outdeg, top_k)
    hub_ids = {node for node, _ in hub_nodes}
    in_rel, out_rel = _count_relations(
        edges,
        hub_ids,
        shift_src=packing.shift_src,
        shift_dst=packing.shift_dst,
        node_mask=packing.node_mask,
        rel_mask=packing.rel_mask,
    )
    unique_nodes = len(set(indeg) | set(outdeg))
    return HubStats(
        processed=processed,
        unique_edges=unique_edges,
        unique_nodes=unique_nodes,
        edges=edges,
        indeg=indeg,
        outdeg=outdeg,
        hub_nodes=hub_nodes,
        in_rel=in_rel,
        out_rel=out_rel,
    )


def _maybe_load_labels(
    with_labels: bool,
    entity_vocab: Path,
    relation_vocab: Path,
    hub_nodes: List[Tuple[int, int]],
    batch_size: int,
) -> Tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
    if not with_labels:
        return None, None
    relation_labels = _load_relation_labels(relation_vocab)
    hub_ids = [node for node, _ in hub_nodes]
    entity_labels = _load_entity_labels(entity_vocab, hub_ids, batch_size)
    return entity_labels, relation_labels


def _write_payload(out_dir: Path, dataset: str, split: str, with_labels: bool, payload: Dict[str, object]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "json" if with_labels else "ids.json"
    out_path = out_dir / f"{dataset}_{split}_hub_nodes_relations_{suffix}"
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    _set_env_limits()
    args = _parse_args()

    lmdb_path = _resolve_lmdb_path(args.data_dir, args.dataset, args.split)
    distance_path = _resolve_distance_path(args.data_dir, args.dataset, args.split, args.distances_dir)
    entity_vocab, relation_vocab = _resolve_vocab_paths(args.data_dir, args.dataset)
    if not lmdb_path.exists():
        raise FileNotFoundError(f"LMDB not found: {lmdb_path}")

    packing = _resolve_packing(entity_vocab, relation_vocab)
    stats = _compute_hub_stats(lmdb_path, packing, args.top_k, args.progress_every)
    distance_stats, start_counts, answer_counts = _compute_distance_stats(
        lmdb_path,
        distance_path,
        progress_every=args.progress_every,
        allow_invalid=args.allow_invalid,
    )
    entity_labels, relation_labels = _maybe_load_labels(
        args.with_labels,
        entity_vocab,
        relation_vocab,
        stats.hub_nodes,
        args.batch_size,
    )
    nodes_out = _build_node_entries(
        stats.hub_nodes,
        stats.indeg,
        stats.outdeg,
        stats.in_rel,
        stats.out_rel,
        entity_labels,
        relation_labels,
        start_counts,
        answer_counts,
    )
    payload = _build_payload(
        dataset=args.dataset,
        split=args.split,
        lmdb_path=lmdb_path,
        distance_path=distance_path,
        entity_vocab=entity_vocab if args.with_labels else None,
        relation_vocab=relation_vocab if args.with_labels else None,
        top_k=args.top_k,
        processed=stats.processed,
        unique_edges=stats.unique_edges,
        unique_nodes=stats.unique_nodes,
        nodes_out=nodes_out,
        node_bits=packing.node_bits,
        rel_bits=packing.rel_bits,
        distance_stats=distance_stats,
    )
    out_path = _write_payload(args.output_dir, args.dataset, args.split, args.with_labels, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
