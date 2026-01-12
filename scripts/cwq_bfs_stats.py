#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

_ZERO = 0
_ONE = 1
_TWO = 2
_NEG_ONE = -1

_TRIPLE_LEN = 3
_HEAD_IDX = 0
_TAIL_IDX = 2

_DIST_UNREACHABLE = _NEG_ONE

_PERCENTILES = (50, 90, 95, 99)
_PROGRESS_INTERVAL_DEFAULT = 5000
_PROGRESS_DISABLED = 0

_DATA_DIR_DEFAULT = Path("/mnt/data/retrieval_dataset")
_HF_ROOT_DEFAULT = Path("/mnt/data/huggingface/datasets")
_HF_DATASET_NAME = "rmanluo/RoG-cwq"
_HF_DATASET_DIRNAME = "rmanluo___ro_g-cwq"
_HF_ARROW_PREFIX = "ro_g-cwq-"
_HF_ARROW_EXT = ".arrow"

_CWQ_DIR = "cwq"
_NORMALIZED_DIR = "normalized"
_QUESTIONS_FILENAME = "questions.parquet"
_GRAPHS_FILENAME = "graphs.parquet"
_PATHS_CONFIG = Path("configs/paths/default.yaml")


class SampleOutcome(NamedTuple):
    missing_start: bool
    missing_answer: bool
    start_not_in_graph: bool
    answer_not_in_graph: bool
    distance: Optional[int]


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


def _resolve_hf_root(value: Optional[str]) -> Path:
    if value:
        return Path(value)
    env_cache = os.environ.get("HF_DATASETS_CACHE")
    if env_cache:
        return Path(env_cache)
    env_home = os.environ.get("HF_HOME")
    if env_home:
        return Path(env_home) / "datasets"
    return _HF_ROOT_DEFAULT


def _find_hf_dataset_root(hf_root: Path) -> Path:
    direct = hf_root / _HF_DATASET_DIRNAME
    if direct.exists():
        return direct
    matches = list(hf_root.rglob(_HF_DATASET_DIRNAME))
    if matches:
        return matches[_ZERO]
    raise FileNotFoundError(f"Missing cached dataset under {hf_root} for {_HF_DATASET_NAME}")


def _split_from_arrow_name(name: str) -> Optional[str]:
    if not name.startswith(_HF_ARROW_PREFIX) or not name.endswith(_HF_ARROW_EXT):
        return None
    trimmed = name[len(_HF_ARROW_PREFIX) : -len(_HF_ARROW_EXT)]
    parts = trimmed.split("-")
    if not parts:
        return None
    return parts[_ZERO]


def _collect_arrow_files(dataset_root: Path) -> Dict[str, List[Path]]:
    files = sorted(dataset_root.rglob(f"{_HF_ARROW_PREFIX}*{_HF_ARROW_EXT}"))
    if not files:
        raise FileNotFoundError(f"No arrow files found under {dataset_root}")
    by_split: Dict[str, List[Path]] = defaultdict(list)
    for path in files:
        split = _split_from_arrow_name(path.name)
        if split:
            by_split[split].append(path)
    return by_split


def _import_pyarrow_ipc():
    try:
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read HF arrow files.") from exc
    return ipc


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read parquet files.") from exc
    return pq


def _build_adjacency_from_triples(
    triples: Sequence[Sequence[str]],
) -> Tuple[Dict[str, int], List[List[int]]]:
    node_to_idx: Dict[str, int] = {}
    adjacency: List[List[int]] = []
    for triple in triples:
        if not triple or len(triple) < _TRIPLE_LEN:
            continue
        head = triple[_HEAD_IDX]
        tail = triple[_TAIL_IDX]
        head_idx = node_to_idx.get(head)
        if head_idx is None:
            head_idx = len(adjacency)
            node_to_idx[head] = head_idx
            adjacency.append([])
        tail_idx = node_to_idx.get(tail)
        if tail_idx is None:
            tail_idx = len(adjacency)
            node_to_idx[tail] = tail_idx
            adjacency.append([])
        adjacency[head_idx].append(tail_idx)
    return node_to_idx, adjacency


def _build_adjacency_from_edge_index(
    num_nodes: int, edge_src: Sequence[int], edge_dst: Sequence[int]
) -> List[List[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for src, dst in zip(edge_src, edge_dst):
        adjacency[src].append(dst)
    return adjacency


def _filter_entities_in_graph(
    entities: Sequence[str], node_to_idx: Dict[str, int]
) -> List[int]:
    if not entities:
        return []
    return [node_to_idx[ent] for ent in entities if ent in node_to_idx]


def _bfs_shortest_distance(
    adjacency: List[List[int]], sources: Sequence[int], targets: Sequence[int]
) -> int:
    if not sources or not targets:
        return _DIST_UNREACHABLE
    target_set = set(targets)
    for src in sources:
        if src in target_set:
            return _ZERO
    distances = [_DIST_UNREACHABLE] * len(adjacency)
    queue = deque()
    for src in sources:
        if distances[src] == _DIST_UNREACHABLE:
            distances[src] = _ZERO
            queue.append(src)
    while queue:
        node = queue.popleft()
        next_dist = distances[node] + _ONE
        for nxt in adjacency[node]:
            if distances[nxt] == _DIST_UNREACHABLE:
                if nxt in target_set:
                    return next_dist
                distances[nxt] = next_dist
                queue.append(nxt)
    return _DIST_UNREACHABLE


def _compute_outcome_from_triples(
    q_entities: Sequence[str], a_entities: Sequence[str], triples: Sequence[Sequence[str]]
) -> SampleOutcome:
    missing_start = not q_entities
    missing_answer = not a_entities
    node_to_idx, adjacency = _build_adjacency_from_triples(triples)
    q_local = _filter_entities_in_graph(q_entities, node_to_idx)
    a_local = _filter_entities_in_graph(a_entities, node_to_idx)
    start_not_in_graph = bool(q_entities) and not q_local
    answer_not_in_graph = bool(a_entities) and not a_local
    distance = None
    if q_local and a_local:
        distance = _bfs_shortest_distance(adjacency, q_local, a_local)
    return SampleOutcome(
        missing_start=missing_start,
        missing_answer=missing_answer,
        start_not_in_graph=start_not_in_graph,
        answer_not_in_graph=answer_not_in_graph,
        distance=distance,
    )


def _compute_outcome_from_edge_index(
    seed_ids: Sequence[int],
    answer_ids: Sequence[int],
    node_entity_ids: Sequence[int],
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
) -> SampleOutcome:
    missing_start = not seed_ids
    missing_answer = not answer_ids
    id_to_idx = {entity_id: idx for idx, entity_id in enumerate(node_entity_ids)}
    seed_local = [id_to_idx[eid] for eid in seed_ids if eid in id_to_idx]
    answer_local = [id_to_idx[eid] for eid in answer_ids if eid in id_to_idx]
    start_not_in_graph = bool(seed_ids) and not seed_local
    answer_not_in_graph = bool(answer_ids) and not answer_local
    distance = None
    if seed_local and answer_local:
        adjacency = _build_adjacency_from_edge_index(len(node_entity_ids), edge_src, edge_dst)
        distance = _bfs_shortest_distance(adjacency, seed_local, answer_local)
    return SampleOutcome(
        missing_start=missing_start,
        missing_answer=missing_answer,
        start_not_in_graph=start_not_in_graph,
        answer_not_in_graph=answer_not_in_graph,
        distance=distance,
    )


def _init_counter() -> Dict[str, int]:
    return {
        "total": _ZERO,
        "missing_start": _ZERO,
        "missing_answer": _ZERO,
        "start_not_in_graph": _ZERO,
        "answer_not_in_graph": _ZERO,
        "bfs_attempted": _ZERO,
        "connected": _ZERO,
        "disconnected": _ZERO,
    }


def _init_distance_bucket() -> Dict[str, object]:
    return {"reachable": [], "unreachable": _ZERO}


def _update_stats(
    counts: Dict[str, int], distances: Dict[str, object], outcome: SampleOutcome
) -> None:
    counts["total"] += _ONE
    if outcome.missing_start:
        counts["missing_start"] += _ONE
    if outcome.missing_answer:
        counts["missing_answer"] += _ONE
    if outcome.start_not_in_graph:
        counts["start_not_in_graph"] += _ONE
    if outcome.answer_not_in_graph:
        counts["answer_not_in_graph"] += _ONE
    if outcome.distance is None:
        return
    counts["bfs_attempted"] += _ONE
    if outcome.distance == _DIST_UNREACHABLE:
        counts["disconnected"] += _ONE
        distances["unreachable"] += _ONE
        return
    counts["connected"] += _ONE
    distances["reachable"].append(outcome.distance)


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


def _build_distance_report(distances: Dict[str, object]) -> Dict[str, object]:
    reachable = list(distances["reachable"])
    return {
        "reachable": len(reachable),
        "unreachable": int(distances["unreachable"]),
        "summary": _summarize_counts(reachable),
        "dist": _make_dist(reachable),
    }


def _build_report(
    counts: Dict[str, int], distances: Dict[str, object]
) -> Dict[str, object]:
    return {"counts": counts, "distance": _build_distance_report(distances)}


def _should_log(progress_interval: int, processed: int) -> bool:
    if progress_interval == _PROGRESS_DISABLED:
        return False
    return processed % progress_interval == _ZERO


def _consume_hf_batch(
    batch_dict: Dict[str, list],
    split_counts: Dict[str, int],
    split_distances: Dict[str, object],
    overall_counts: Dict[str, int],
    overall_distances: Dict[str, object],
    progress: Dict[str, int],
    max_samples: Optional[int],
    progress_interval: int,
) -> bool:
    total = len(batch_dict["id"])
    for idx in range(total):
        q_entities = batch_dict["q_entity"][idx] or []
        a_entities = batch_dict["a_entity"][idx] or []
        triples = batch_dict["graph"][idx] or []
        outcome = _compute_outcome_from_triples(q_entities, a_entities, triples)
        _update_stats(split_counts, split_distances, outcome)
        _update_stats(overall_counts, overall_distances, outcome)
        progress["processed"] += _ONE
        if max_samples and progress["processed"] >= max_samples:
            return True
        if _should_log(progress_interval, progress["processed"]):
            print(f"processed={progress['processed']}", file=sys.stderr)
    return False


def _process_hf_dataset(
    files_by_split: Dict[str, List[Path]],
    progress_interval: int,
    max_samples: Optional[int],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, object]]]:
    ipc = _import_pyarrow_ipc()
    counts_by_split: Dict[str, Dict[str, int]] = defaultdict(_init_counter)
    dist_by_split: Dict[str, Dict[str, object]] = defaultdict(_init_distance_bucket)
    overall_counts = _init_counter()
    overall_distances = _init_distance_bucket()
    progress = {"processed": _ZERO}
    for split, files in sorted(files_by_split.items()):
        for path in files:
            with path.open("rb") as handle:
                reader = ipc.open_stream(handle)
                for batch in reader:
                    stop = _consume_hf_batch(
                        batch.to_pydict(),
                        counts_by_split[split],
                        dist_by_split[split],
                        overall_counts,
                        overall_distances,
                        progress,
                        max_samples,
                        progress_interval,
                    )
                    if stop:
                        counts_by_split["overall"] = overall_counts
                        dist_by_split["overall"] = overall_distances
                        return counts_by_split, dist_by_split
    counts_by_split["overall"] = overall_counts
    dist_by_split["overall"] = overall_distances
    return counts_by_split, dist_by_split


def _align_graph_ids(
    q_graph_ids: Sequence[str], g_graph_ids: Sequence[str]
) -> Optional[Dict[str, int]]:
    if list(q_graph_ids) == list(g_graph_ids):
        return None
    return {gid: idx for idx, gid in enumerate(g_graph_ids)}


def _consume_normalized_row_group(
    qtable,
    gtable,
    counts_by_split: Dict[str, Dict[str, int]],
    dist_by_split: Dict[str, Dict[str, object]],
    overall_counts: Dict[str, int],
    overall_distances: Dict[str, object],
    progress: Dict[str, int],
    max_samples: Optional[int],
    progress_interval: int,
) -> bool:
    q_graph_ids = qtable["graph_id"].to_pylist()
    g_graph_ids = gtable["graph_id"].to_pylist()
    g_index = _align_graph_ids(q_graph_ids, g_graph_ids)
    total = qtable.num_rows
    for idx in range(total):
        graph_idx = idx if g_index is None else g_index[q_graph_ids[idx]]
        split = qtable["split"][idx].as_py()
        seed_ids = qtable["seed_entity_ids"][idx].as_py() or []
        answer_ids = qtable["answer_entity_ids"][idx].as_py() or []
        node_entity_ids = gtable["node_entity_ids"][graph_idx].as_py() or []
        edge_src = gtable["edge_src"][graph_idx].as_py() or []
        edge_dst = gtable["edge_dst"][graph_idx].as_py() or []
        outcome = _compute_outcome_from_edge_index(
            seed_ids, answer_ids, node_entity_ids, edge_src, edge_dst
        )
        split_counts = counts_by_split[split]
        split_distances = dist_by_split[split]
        _update_stats(split_counts, split_distances, outcome)
        _update_stats(overall_counts, overall_distances, outcome)
        progress["processed"] += _ONE
        if max_samples and progress["processed"] >= max_samples:
            return True
        if _should_log(progress_interval, progress["processed"]):
            print(f"processed={progress['processed']}", file=sys.stderr)
    return False


def _process_normalized_dataset(
    question_path: Path,
    graph_path: Path,
    progress_interval: int,
    max_samples: Optional[int],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, object]]]:
    pq = _import_pyarrow_parquet()
    qpf = pq.ParquetFile(question_path)
    gpf = pq.ParquetFile(graph_path)
    if qpf.metadata.num_rows != gpf.metadata.num_rows:
        raise ValueError("questions.parquet and graphs.parquet row counts differ.")
    counts_by_split: Dict[str, Dict[str, int]] = defaultdict(_init_counter)
    dist_by_split: Dict[str, Dict[str, object]] = defaultdict(_init_distance_bucket)
    overall_counts = _init_counter()
    overall_distances = _init_distance_bucket()
    progress = {"processed": _ZERO}
    for rg in range(qpf.num_row_groups):
        qtable = qpf.read_row_group(
            rg, columns=["graph_id", "split", "seed_entity_ids", "answer_entity_ids"]
        )
        gtable = gpf.read_row_group(
            rg, columns=["graph_id", "node_entity_ids", "edge_src", "edge_dst"]
        )
        stop = _consume_normalized_row_group(
            qtable,
            gtable,
            counts_by_split,
            dist_by_split,
            overall_counts,
            overall_distances,
            progress,
            max_samples,
            progress_interval,
        )
        if stop:
            counts_by_split["overall"] = overall_counts
            dist_by_split["overall"] = overall_distances
            return counts_by_split, dist_by_split
    counts_by_split["overall"] = overall_counts
    dist_by_split["overall"] = overall_distances
    return counts_by_split, dist_by_split


def _print_report(title: str, report: Dict[str, object]) -> None:
    print(title)
    counts = report["counts"]
    for key in (
        "total",
        "missing_start",
        "missing_answer",
        "start_not_in_graph",
        "answer_not_in_graph",
        "bfs_attempted",
        "connected",
        "disconnected",
    ):
        print(f"  {key}: {counts[key]}")
    dist = report["distance"]
    print("  distance:")
    print(f"    reachable: {dist['reachable']}")
    print(f"    unreachable: {dist['unreachable']}")
    summary = dist["summary"]
    print(f"    summary: {json.dumps(summary, ensure_ascii=True)}")
    print(f"    dist: {json.dumps(dist['dist'], ensure_ascii=True)}")


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=_TWO)


def _resolve_source(
    source: str, data_dir: Path, hf_root: Path
) -> Tuple[str, Optional[Path], Optional[Path], Optional[Path]]:
    normalized_dir = data_dir / _CWQ_DIR / _NORMALIZED_DIR
    question_path = normalized_dir / _QUESTIONS_FILENAME
    graph_path = normalized_dir / _GRAPHS_FILENAME
    if source == "normalized":
        if not question_path.exists() or not graph_path.exists():
            raise FileNotFoundError("Missing normalized cwq parquet files.")
        return "normalized", question_path, graph_path, None
    if source == "hf":
        dataset_root = _find_hf_dataset_root(hf_root)
        return "hf", None, None, dataset_root
    if question_path.exists() and graph_path.exists():
        return "normalized", question_path, graph_path, None
    dataset_root = _find_hf_dataset_root(hf_root)
    return "hf", None, None, dataset_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CWQ BFS connectivity and distance stats.")
    parser.add_argument(
        "--source",
        choices=("auto", "normalized", "hf"),
        default="auto",
        help="Data source: normalized parquet or HF RoG-cwq cache.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base data dir containing cwq/normalized (optional).",
    )
    parser.add_argument(
        "--hf-root",
        type=str,
        default=None,
        help="HF datasets cache root (optional).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=_PROGRESS_INTERVAL_DEFAULT,
        help="Log progress every N samples (0 disables).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = _resolve_data_dir(args.data_dir)
    hf_root = _resolve_hf_root(args.hf_root)
    source, question_path, graph_path, dataset_root = _resolve_source(
        args.source, data_dir, hf_root
    )
    if source == "normalized":
        counts_by_split, dist_by_split = _process_normalized_dataset(
            question_path,
            graph_path,
            args.progress_interval,
            args.max_samples,
        )
    else:
        files_by_split = _collect_arrow_files(dataset_root)
        counts_by_split, dist_by_split = _process_hf_dataset(
            files_by_split,
            args.progress_interval,
            args.max_samples,
        )
    report = {
        "source": source,
        "overall": _build_report(
            counts_by_split["overall"], dist_by_split["overall"]
        ),
        "by_split": {
            split: _build_report(counts, dist_by_split[split])
            for split, counts in counts_by_split.items()
            if split != "overall"
        },
    }
    _print_report("overall", report["overall"])
    for split in sorted(report["by_split"].keys()):
        _print_report(f"split={split}", report["by_split"][split])
    if args.output_json:
        _write_json(Path(args.output_json), report)


if __name__ == "__main__":
    main()
