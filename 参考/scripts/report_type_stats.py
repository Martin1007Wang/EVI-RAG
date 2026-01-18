#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq

from src.utils.logging_utils import get_logger, init_logging, log_event

LOGGER = get_logger(__name__)
_ZERO = 0
_ONE = 1
_DEFAULT_BATCH_SIZE = 256
_DEFAULT_PROGRESS_INTERVAL = 0


@dataclass
class SplitStats:
    graphs: int = _ZERO
    nodes: int = _ZERO
    type_nodes: int = _ZERO
    type_ids: int = _ZERO
    graphs_no_type: int = _ZERO
    hist: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.hist is None:
            self.hist = {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report node type (type feature) statistics.")
    parser.add_argument("--graphs-path", type=str, required=True, help="Path to graphs.parquet.")
    parser.add_argument("--questions-path", type=str, default=None, help="Optional questions.parquet for split labels.")
    parser.add_argument(
        "--relation-vocab-path",
        type=str,
        default=None,
        help="relation_vocab.parquet (required when node_type_counts is absent).",
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, help="Row batch size.")
    parser.add_argument("--progress-interval", type=int, default=_DEFAULT_PROGRESS_INTERVAL, help="Log every N graphs.")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--log-path", type=str, default=None, help="Optional log file path.")
    return parser.parse_args()


def _load_split_lookup(questions_path: Optional[Path]) -> Optional[Dict[str, str]]:
    if questions_path is None:
        return None
    table = pq.read_table(questions_path, columns=["graph_id", "split"])
    graph_ids = table.column("graph_id").to_pylist()
    splits = table.column("split").to_pylist()
    lookup: Dict[str, str] = {}
    for graph_id, split in zip(graph_ids, splits):
        if graph_id in lookup:
            continue
        lookup[str(graph_id)] = str(split)
    return lookup


def _iter_graph_batches(graphs_path: Path, batch_size: int, columns: List[str]) -> Iterable[dict]:
    parquet = pq.ParquetFile(graphs_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pydict()


def _update_hist(hist: Dict[int, int], counts: np.ndarray) -> None:
    if counts.size == 0:
        return
    unique_vals, freq = np.unique(counts, return_counts=True)
    for val, cnt in zip(unique_vals.tolist(), freq.tolist()):
        hist[val] = hist.get(int(val), _ZERO) + int(cnt)


def _compute_percentile(hist: Dict[int, int], percentile: int) -> int:
    if not hist:
        return _ZERO
    total = sum(hist.values())
    if total <= _ZERO:
        return _ZERO
    threshold = int(np.ceil(total * (percentile / 100.0)))
    running = _ZERO
    for val in sorted(hist):
        running += hist[val]
        if running >= threshold:
            return int(val)
    return int(max(hist))


def _summarize_split(stats: SplitStats) -> Dict[str, object]:
    avg_nodes = stats.nodes / stats.graphs if stats.graphs > _ZERO else 0.0
    avg_type_nodes = stats.type_nodes / stats.graphs if stats.graphs > _ZERO else 0.0
    avg_type_ids = stats.type_ids / stats.graphs if stats.graphs > _ZERO else 0.0
    type_node_ratio = stats.type_nodes / stats.nodes if stats.nodes > _ZERO else 0.0
    avg_types_per_node = stats.type_ids / stats.nodes if stats.nodes > _ZERO else 0.0
    avg_types_per_type_node = stats.type_ids / stats.type_nodes if stats.type_nodes > _ZERO else 0.0
    non_zero_hist = {k: v for k, v in stats.hist.items() if k > _ZERO}
    summary = {
        "graphs": stats.graphs,
        "nodes": stats.nodes,
        "type_nodes": stats.type_nodes,
        "type_ids": stats.type_ids,
        "graphs_no_type": stats.graphs_no_type,
        "type_node_ratio": round(type_node_ratio, 6),
        "avg_nodes_per_graph": round(avg_nodes, 4),
        "avg_type_nodes_per_graph": round(avg_type_nodes, 4),
        "avg_type_ids_per_graph": round(avg_type_ids, 4),
        "avg_types_per_node": round(avg_types_per_node, 6),
        "avg_types_per_type_node": round(avg_types_per_type_node, 6),
        "type_count_p50": _compute_percentile(non_zero_hist, 50),
        "type_count_p90": _compute_percentile(non_zero_hist, 90),
        "type_count_p99": _compute_percentile(non_zero_hist, 99),
        "type_count_max": max(non_zero_hist) if non_zero_hist else _ZERO,
    }
    return summary


def _load_relation_vocab(path: Path) -> Tuple[np.ndarray, List[str]]:
    table = pq.read_table(path, columns=["relation_id", "label"])
    rel_ids = np.asarray(table.column("relation_id").to_numpy(), dtype=np.int64)
    labels = [str(label) for label in table.column("label").to_pylist()]
    return rel_ids, labels


def _resolve_type_relation_mask(relation_vocab_path: Path) -> np.ndarray:
    from src.data.relation_cleaning_rules import (
        DEFAULT_RELATION_CLEANING_RULES,
        RELATION_ACTION_TYPE,
        relation_action,
    )

    rel_ids, labels = _load_relation_vocab(relation_vocab_path)
    max_rel_id = int(rel_ids.max()) if rel_ids.size > 0 else _ZERO
    mask = np.zeros((max_rel_id + _ONE,), dtype=bool)
    for rel_id, label in zip(rel_ids.tolist(), labels):
        action = relation_action(label, DEFAULT_RELATION_CLEANING_RULES, enabled=True)
        if action == RELATION_ACTION_TYPE:
            mask[int(rel_id)] = True
    return mask


def _counts_from_edges(
    *,
    node_entity_ids: List[int],
    edge_src: List[int],
    edge_dst: List[int],
    edge_rel: List[int],
    type_rel_mask: np.ndarray,
) -> np.ndarray:
    num_nodes = len(node_entity_ids)
    if num_nodes <= _ZERO or not edge_src:
        return np.zeros((num_nodes,), dtype=np.int64)
    rel = np.asarray(edge_rel, dtype=np.int64)
    src = np.asarray(edge_src, dtype=np.int64)
    dst = np.asarray(edge_dst, dtype=np.int64)
    if rel.size == 0 or src.size == 0 or dst.size == 0:
        return np.zeros((num_nodes,), dtype=np.int64)
    if rel.max(initial=_ZERO) >= type_rel_mask.size:
        raise ValueError("relation_vocab does not cover edge_relation_ids.")
    type_edge_mask = type_rel_mask[rel]
    if not np.any(type_edge_mask):
        return np.zeros((num_nodes,), dtype=np.int64)
    src = src[type_edge_mask]
    dst = dst[type_edge_mask]
    nodes = np.asarray(node_entity_ids, dtype=np.int64)
    tail_entity_ids = nodes[dst]
    pairs = np.stack([src, tail_entity_ids], axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    if unique_pairs.size == 0:
        return np.zeros((num_nodes,), dtype=np.int64)
    return np.bincount(unique_pairs[:, 0], minlength=num_nodes)


def main() -> None:
    args = _parse_args()
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else None
    init_logging(log_path=log_path)
    graphs_path = Path(args.graphs_path)
    questions_path = Path(args.questions_path) if args.questions_path else None
    relation_vocab_path = Path(args.relation_vocab_path) if args.relation_vocab_path else None
    split_lookup = _load_split_lookup(questions_path)
    parquet = pq.ParquetFile(graphs_path)
    has_node_type = "node_type_counts" in parquet.schema.names
    type_rel_mask = None
    columns = ["graph_id", "node_type_counts"] if has_node_type else [
        "graph_id",
        "node_entity_ids",
        "edge_src",
        "edge_dst",
        "edge_relation_ids",
    ]
    if not has_node_type:
        if relation_vocab_path is None:
            raise ValueError("relation_vocab_path required when node_type_counts is absent.")
        type_rel_mask = _resolve_type_relation_mask(relation_vocab_path)
        log_event(LOGGER, "type_stats_fallback", mode="from_edges", relation_vocab=str(relation_vocab_path))
    log_event(
        LOGGER,
        "type_stats_start",
        graphs_path=str(graphs_path),
        questions_path=str(questions_path) if questions_path else None,
        batch_size=args.batch_size,
        progress_interval=args.progress_interval,
    )
    stats_by_split: Dict[str, SplitStats] = {}
    processed_graphs = _ZERO
    for batch in _iter_graph_batches(graphs_path, args.batch_size, columns):
        graph_ids = batch["graph_id"]
        if has_node_type:
            type_counts_list = batch["node_type_counts"]
        for idx, graph_id in enumerate(graph_ids):
            if has_node_type:
                counts = type_counts_list[idx]
            else:
                counts = _counts_from_edges(
                    node_entity_ids=batch["node_entity_ids"][idx],
                    edge_src=batch["edge_src"][idx],
                    edge_dst=batch["edge_dst"][idx],
                    edge_rel=batch["edge_relation_ids"][idx],
                    type_rel_mask=type_rel_mask,
                )
            split = "all"
            if split_lookup is not None:
                split = split_lookup.get(str(graph_id), "unknown")
            stats = stats_by_split.get(split)
            if stats is None:
                stats = SplitStats()
                stats_by_split[split] = stats
            counts_np = np.asarray(counts, dtype=np.int64)
            stats.graphs += _ONE
            stats.nodes += int(counts_np.size)
            stats.type_nodes += int(np.count_nonzero(counts_np))
            stats.type_ids += int(counts_np.sum())
            if counts_np.size == 0 or counts_np.sum() == _ZERO:
                stats.graphs_no_type += _ONE
            _update_hist(stats.hist, counts_np)
            processed_graphs += _ONE
            if args.progress_interval > _ZERO and processed_graphs % args.progress_interval == 0:
                log_event(LOGGER, "type_stats_progress", graphs=processed_graphs)

    summaries = {split: _summarize_split(stats) for split, stats in stats_by_split.items()}
    log_event(LOGGER, "type_stats_done", splits=list(summaries.keys()))
    output = {"summaries": summaries}
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=True), encoding="utf-8")
        log_event(LOGGER, "type_stats_written", path=str(output_path))
    else:
        log_event(LOGGER, "type_stats_result", summaries=summaries)


if __name__ == "__main__":
    main()
