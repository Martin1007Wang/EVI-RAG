#!/usr/bin/env python3
"""Build a structured sample filter with path dedup + length stratification."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch

from src.data.components.lmdb_store import EmbeddingStore
from src.data.io.lmdb_utils import _apply_filter_intersection, _assign_lmdb_shard, _resolve_core_lmdb_paths
from src.data.schema.constants import _PATH_MODE_QA_DIRECTED, _PATH_MODE_UNDIRECTED
from src.data.utils.connectivity import shortest_path_edge_indices_undirected
from src.utils.logging_utils import get_logger, log_event

LOGGER = get_logger(__name__)


def _parse_bins(text: str) -> List[Tuple[int, Optional[int], str]]:
    bins: List[Tuple[int, Optional[int], str]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("+"):
            start = int(token[:-1])
            bins.append((start, None, f"{start}+"))
        else:
            val = int(token)
            bins.append((val, val, str(val)))
    if not bins:
        raise ValueError("bins must be non-empty, e.g. '0,1,2,3,4,5+'")
    return bins


def _assign_bin(length: int, bins: Sequence[Tuple[int, Optional[int], str]]) -> str:
    for start, end, label in bins:
        if end is None:
            if length >= start:
                return label
        elif start <= length <= end:
            return label
    return "other"


def _unique_ints(values: Iterable[int]) -> List[int]:
    return sorted({int(val) for val in values if int(val) >= 0})


def _shortest_path_directed(
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    sources: Sequence[int],
    targets: Sequence[int],
) -> Tuple[List[int], List[int]]:
    if not sources or not targets or num_nodes <= 0:
        return [], []
    adjacency: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for idx, (u_raw, v_raw) in enumerate(zip(edge_src, edge_dst)):
        u = int(u_raw)
        v = int(v_raw)
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append((v, idx))
    for nbrs in adjacency:
        nbrs.sort(key=lambda item: (item[0], item[1]))
    sources_unique = _unique_ints(sources)
    targets_unique = _unique_ints(targets)
    if not sources_unique or not targets_unique:
        return [], []

    from collections import deque

    dist = [-1] * num_nodes
    parent = [-1] * num_nodes
    parent_edge = [-1] * num_nodes
    q: deque[int] = deque()
    for s in sources_unique:
        dist[s] = 0
        q.append(s)
    while q:
        cur = q.popleft()
        next_dist = dist[cur] + 1
        for nb, e_idx in adjacency[cur]:
            if dist[nb] != -1:
                continue
            dist[nb] = next_dist
            parent[nb] = cur
            parent_edge[nb] = int(e_idx)
            q.append(nb)

    best_target = None
    best_dist = None
    for tgt in targets_unique:
        if dist[tgt] < 0:
            continue
        if best_dist is None or dist[tgt] < best_dist or (dist[tgt] == best_dist and tgt < best_target):
            best_target = tgt
            best_dist = dist[tgt]
    if best_target is None:
        return [], []

    nodes_rev: List[int] = [int(best_target)]
    edges_rev: List[int] = []
    cur = int(best_target)
    sources_set = set(sources_unique)
    while cur not in sources_set:
        prev = int(parent[cur])
        edge = int(parent_edge[cur])
        if prev < 0 or edge < 0:
            return [], []
        edges_rev.append(edge)
        nodes_rev.append(prev)
        cur = prev
    edges = list(reversed(edges_rev))
    nodes = list(reversed(nodes_rev))
    if not edges:
        return [], nodes
    return edges, nodes


def _shortest_path_edges(
    *,
    num_nodes: int,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    seeds: Sequence[int],
    answers: Sequence[int],
    path_mode: str,
) -> Tuple[List[int], List[int]]:
    if path_mode == _PATH_MODE_QA_DIRECTED:
        return _shortest_path_directed(num_nodes, edge_src, edge_dst, seeds, answers)
    return shortest_path_edge_indices_undirected(num_nodes, edge_src, edge_dst, seeds, answers)


def _edge_path_signature(edge_rel_ids: Sequence[int], edge_ids: Sequence[int]) -> str:
    rels = [str(int(edge_rel_ids[idx])) for idx in edge_ids]
    length = len(rels)
    return f"{length}|" + ",".join(rels)


def _resolve_num_nodes(raw: Dict[str, torch.Tensor]) -> int:
    num_nodes = raw.get("num_nodes")
    if num_nodes is not None:
        if torch.is_tensor(num_nodes):
            return int(num_nodes.view(-1)[0].detach().tolist())
        return int(num_nodes)
    node_ids = raw.get("node_global_ids")
    if node_ids is not None and torch.is_tensor(node_ids):
        return int(node_ids.numel())
    return 0


def _compute_sample_signature(
    raw: Dict[str, torch.Tensor],
    *,
    path_mode: str,
) -> Optional[Tuple[int, str]]:
    edge_index = raw.get("edge_index")
    edge_rel = raw.get("edge_attr")
    if edge_index is None or edge_rel is None:
        return None
    q_local = raw.get("q_local_indices")
    a_local = raw.get("a_local_indices")
    if q_local is None or a_local is None:
        return None
    edge_index = edge_index.to(dtype=torch.long, device="cpu")
    edge_rel = edge_rel.to(dtype=torch.long, device="cpu")
    if edge_index.numel() == 0 or edge_rel.numel() == 0:
        return None
    seeds = _unique_ints(q_local.view(-1).tolist())
    answers = _unique_ints(a_local.view(-1).tolist())
    if not seeds or not answers:
        return None
    num_nodes = _resolve_num_nodes(raw)
    if num_nodes <= 0:
        return None
    edge_src = edge_index[0].tolist()
    edge_dst = edge_index[1].tolist()
    edge_ids, nodes = _shortest_path_edges(
        num_nodes=num_nodes,
        edge_src=edge_src,
        edge_dst=edge_dst,
        seeds=seeds,
        answers=answers,
        path_mode=path_mode,
    )
    if edge_ids is None or not nodes:
        return None
    length = len(edge_ids)
    signature = _edge_path_signature(edge_rel.tolist(), edge_ids)
    return length, signature


def _load_sample_ids(paths: Sequence[Path]) -> List[str]:
    sample_ids: List[str] = []
    for path in paths:
        store = EmbeddingStore(path)
        try:
            sample_ids.extend(store.get_sample_ids())
        finally:
            store.close()
    return sample_ids


def _build_stores(paths: Sequence[Path]) -> Dict[int, EmbeddingStore]:
    stores: Dict[int, EmbeddingStore] = {}
    for idx, path in enumerate(paths):
        stores[idx] = EmbeddingStore(path)
    return stores


def _close_stores(stores: Dict[int, EmbeddingStore]) -> None:
    for store in stores.values():
        store.close()


def _select_from_bin(
    sig_map: Dict[str, List[str]],
    *,
    target: int,
    max_per_signature: int,
) -> List[str]:
    if target <= 0:
        return []
    selected: List[str] = []
    for signature in sorted(sig_map):
        if len(selected) >= target:
            break
        ids = sorted(sig_map[signature])
        take = min(len(ids), max_per_signature, target - len(selected))
        selected.extend(ids[:take])
    return selected


def _alloc_targets(
    bins: Sequence[str],
    *,
    total: int,
    availability: Dict[str, int],
) -> Dict[str, int]:
    total_available = sum(availability.values())
    if total > total_available:
        total = total_available
    if total <= 0 or not bins:
        return {label: 0 for label in bins}
    base = total // len(bins)
    rem = total % len(bins)
    targets: Dict[str, int] = {}
    for idx, label in enumerate(bins):
        want = base + (1 if idx < rem else 0)
        targets[label] = min(want, availability.get(label, 0))
    remaining = total - sum(targets.values())
    if remaining <= 0:
        return targets
    for label in bins:
        if remaining <= 0:
            break
        capacity = availability.get(label, 0) - targets[label]
        if capacity <= 0:
            continue
        take = min(capacity, remaining)
        targets[label] += take
        remaining -= take
    return targets


def _resolve_target(args: argparse.Namespace, split: str) -> Optional[int]:
    if split == "train" and args.target_train is not None:
        return int(args.target_train)
    if split == "validation" and args.target_validation is not None:
        return int(args.target_validation)
    if split == "test" and args.target_test is not None:
        return int(args.target_test)
    if args.target is None:
        return None
    return int(args.target)


def _write_filter(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def build_filter_for_split(
    *,
    dataset_name: str,
    split: str,
    embeddings_dir: Path,
    bins: Sequence[Tuple[int, Optional[int], str]],
    path_mode: str,
    base_filters: Sequence[Path],
    max_per_signature: int,
    max_path_length: Optional[int],
    target: Optional[int],
    output_dir: Path,
    output_prefix: str,
) -> Path:
    lmdb_paths = _resolve_core_lmdb_paths(embeddings_dir, split)
    sample_ids = _load_sample_ids(lmdb_paths)
    if base_filters:
        sample_ids = _apply_filter_intersection(sample_ids, base_filters)

    stores = _build_stores(lmdb_paths)
    bin_map: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    stats = {
        "total_candidates": 0,
        "no_path": 0,
        "too_long": 0,
        "selected": 0,
        "bins": {},
    }
    try:
        for sample_id in sample_ids:
            stats["total_candidates"] += 1
            shard_id = _assign_lmdb_shard(sample_id, len(lmdb_paths))
            raw = stores[shard_id].load_sample(sample_id)
            sig = _compute_sample_signature(raw, path_mode=path_mode)
            if sig is None:
                stats["no_path"] += 1
                continue
            length, signature = sig
            if max_path_length is not None and length > max_path_length:
                stats["too_long"] += 1
                continue
            label = _assign_bin(length, bins)
            bin_map[label][signature].append(sample_id)
    finally:
        _close_stores(stores)

    bin_labels = [label for _, _, label in bins]
    availability = {
        label: sum(min(len(ids), max_per_signature) for ids in sig_map.values())
        for label, sig_map in bin_map.items()
    }
    if target is None:
        targets = {label: availability.get(label, 0) for label in bin_labels}
    else:
        targets = _alloc_targets(bin_labels, total=int(target), availability=availability)

    selected: List[str] = []
    for label in bin_labels:
        sig_map = bin_map.get(label, {})
        chosen = _select_from_bin(sig_map, target=targets.get(label, 0), max_per_signature=max_per_signature)
        selected.extend(chosen)
        stats["bins"][label] = {
            "unique_signatures": len(sig_map),
            "available": availability.get(label, 0),
            "selected": len(chosen),
        }
    stats["selected"] = len(selected)

    payload = {
        "dataset": dataset_name,
        "split": split,
        "sample_ids": sorted(selected),
        "criteria": {
            "path_mode": path_mode,
            "bins": [label for _, _, label in bins],
            "max_per_signature": max_per_signature,
            "max_path_length": max_path_length,
            "target": target,
            "base_filters": [str(p) for p in base_filters],
        },
        "stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_prefix}_{split}.json"
    _write_filter(output_path, payload)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a structured sample filter.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. cwq.")
    parser.add_argument(
        "--dataset-root",
        default="/mnt/data/retrieval_dataset",
        help="Root directory containing datasets.",
    )
    parser.add_argument(
        "--splits",
        default="train,validation,test",
        help="Comma-separated splits to process.",
    )
    parser.add_argument(
        "--bins",
        default="0,1,2,3,4,5+",
        help="Comma-separated path length bins, e.g. '0,1,2,3,4,5+'.",
    )
    parser.add_argument(
        "--path-mode",
        default=_PATH_MODE_UNDIRECTED,
        choices=(_PATH_MODE_UNDIRECTED, _PATH_MODE_QA_DIRECTED),
        help="Shortest-path traversal mode.",
    )
    parser.add_argument(
        "--base-filter",
        default=None,
        help="Optional base filter (e.g., sub_filter.json) applied before sampling.",
    )
    parser.add_argument(
        "--max-per-signature",
        type=int,
        default=1,
        help="Max samples per unique relation-path signature.",
    )
    parser.add_argument(
        "--max-path-length",
        type=int,
        default=None,
        help="Drop samples with shortest path length > max-path-length.",
    )
    parser.add_argument("--target", type=int, default=None, help="Target samples per split (uniform).")
    parser.add_argument("--target-train", type=int, default=None, help="Target samples for train split.")
    parser.add_argument("--target-validation", type=int, default=None, help="Target samples for validation split.")
    parser.add_argument("--target-test", type=int, default=None, help="Target samples for test split.")
    parser.add_argument(
        "--output-prefix",
        default="structured_filter",
        help="Output filter file prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root)
    dataset_dir = dataset_root / args.dataset
    embeddings_dir = dataset_dir / "materialized" / "embeddings"
    output_dir = dataset_dir / "normalized"
    bins = _parse_bins(args.bins)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    base_filters: List[Path] = []
    if args.base_filter:
        base_filters.append(Path(args.base_filter))
    else:
        default_filter = output_dir / "sub_filter.json"
        if default_filter.exists():
            base_filters.append(default_filter)

    for split in splits:
        target = _resolve_target(args, split)
        log_event(
            LOGGER,
            "structured_filter_start",
            dataset=args.dataset,
            split=split,
            target=target,
        )
        output_path = build_filter_for_split(
            dataset_name=args.dataset,
            split=split,
            embeddings_dir=embeddings_dir,
            bins=bins,
            path_mode=args.path_mode,
            base_filters=base_filters,
            max_per_signature=args.max_per_signature,
            max_path_length=args.max_path_length,
            target=target,
            output_dir=output_dir,
            output_prefix=args.output_prefix,
        )
        log_event(LOGGER, "structured_filter_done", split=split, path=str(output_path))


if __name__ == "__main__":
    main()
