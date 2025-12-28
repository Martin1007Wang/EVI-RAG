"""Compute descriptive statistics for g_agent caches (g_agent_samples.pt).

Usage:
    python scripts/g_agent_stats.py --pos-threshold 0.5 --max-samples 1000 path1.pt [path2.pt ...]

Metrics definitions (per split):
  - num_samples: total samples.
  - avg_selected_edges / median_selected_edges / max_selected_edges / total_selected_edges: size of selected_edges.
  - pos_selected_edges_ratio: fraction of selected_edges with label > pos_threshold.
  - gt_path_exists_ratio: fraction of samples with gt_path_exists True (or non-empty path ids).
  - gt_path_len_{min,max,mean}: stats over ground-truth path length (>0 only).
  - gt_path_edges_selected_ratio: total GT path edges / total_selected_edges.
  - gt_path_edges_pos_ratio: within GT path edges, fraction with label > pos_threshold.
  - gt_path_edges_in_selected_ratio: fraction of GT path edges present in selected_edges.
  - gt_path_edges_in_top_ratio: fraction of GT path edges present in top_edge_local_indices.
  - coverage_selected_full_ratio: samples where all GT path edges are in selected_edges.
  - coverage_selected_partial_ratio: samples where some but not all GT path edges are in selected_edges.
  - coverage_top_full_ratio / coverage_top_partial_ratio: same, but w.r.t. top_edge_local_indices.
  - retrieval_failed_ratio: fraction with retrieval_failed flag.
  - score_mean_on_path / score_mean_off_path / score_delta: retrieval score signal for GT vs non-GT edges.
"""

from __future__ import annotations

import argparse
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_HW_SUBSET", "1t")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import numpy as np
import torch

torch.set_num_threads(1)


def _safe_ratio(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else 0.0


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if "samples" not in payload:
        raise RuntimeError(f"Missing 'samples' in {path}")
    samples = payload["samples"]
    if not isinstance(samples, list) or not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def _maybe_subsample(samples: Sequence[Dict[str, Any]], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(samples):
        return list(samples)
    idxs = np.random.choice(len(samples), max_samples, replace=False)
    return [samples[int(i)] for i in idxs]


def _compute_stats(
    samples: Iterable[Dict[str, Any]],
    pos_threshold: float,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    samples_list = _maybe_subsample(list(samples), max_samples)
    edge_counts: List[int] = []
    path_lengths: List[int] = []
    gt_triples_len: List[int] = []

    total_edges = 0
    pos_edges = 0
    path_edges_total = 0
    path_edges_pos = 0
    path_edges_in_selected = 0
    path_edges_in_top = 0

    with_gt_path = 0
    coverage_sel_full = 0
    coverage_sel_partial = 0
    coverage_top_full = 0
    coverage_top_partial = 0
    gt_path_exists = 0
    retrieval_failed = 0
    gt_triples_nonempty = 0

    scores_on_path: List[float] = []
    scores_off_path: List[float] = []

    for sample in samples_list:
        if "selected_edges" not in sample:
            raise KeyError("selected_edges missing from sample")
        if "top_edge_local_indices" not in sample:
            raise KeyError("top_edge_local_indices missing from sample")
        edges = sample["selected_edges"]
        top_list = sample["top_edge_local_indices"]
        selected_ids: Set[int] = {int(e["local_index"]) for e in edges}
        top_ids: Set[int] = {int(x) for x in top_list}
        score_by_loc = {int(e["local_index"]): float(e["score"]) for e in edges}

        edge_counts.append(len(edges))
        total_edges += len(edges)

        for e in edges:
            if float(e["label"]) > pos_threshold:
                pos_edges += 1

        if "gt_path_edge_local_indices" not in sample:
            raise KeyError("gt_path_edge_local_indices missing from sample")
        path_ids: Set[int] = {int(x) for x in sample["gt_path_edge_local_indices"]}
        if path_ids:
            with_gt_path += 1
            path_lengths.append(len(path_ids))
            present_sel = sum(1 for pid in path_ids if pid in selected_ids)
            present_top = sum(1 for pid in path_ids if pid in top_ids)
            if present_sel == len(path_ids):
                coverage_sel_full += 1
            elif present_sel > 0:
                coverage_sel_partial += 1
            if present_top == len(path_ids):
                coverage_top_full += 1
            elif present_top > 0:
                coverage_top_partial += 1
            path_edges_total += len(path_ids)
            path_edges_in_selected += present_sel
            path_edges_in_top += present_top
            for e in edges:
                loc = int(e["local_index"])
                if loc in path_ids and float(e["label"]) > pos_threshold:
                    path_edges_pos += 1
            # Score signal split
            for pid in path_ids:
                if pid in score_by_loc:
                    scores_on_path.append(score_by_loc[pid])
            # Down-sample negatives to similar scale to avoid memory blow-up.
            neg_candidates = [score for loc, score in score_by_loc.items() if loc not in path_ids]
            if neg_candidates:
                k = min(len(neg_candidates), max(len(path_ids), 1))
                perm = np.random.permutation(len(neg_candidates))[:k]
                scores_off_path.extend([neg_candidates[j] for j in perm])

        if "gt_path_exists" not in sample:
            raise KeyError("gt_path_exists missing from sample")
        if bool(sample["gt_path_exists"]):
            gt_path_exists += 1
        if "retrieval_failed" not in sample:
            raise KeyError("retrieval_failed missing from sample")
        if bool(sample["retrieval_failed"]):
            retrieval_failed += 1
        if "gt_paths_triples" not in sample:
            raise KeyError("gt_paths_triples missing from sample")
        triples = sample["gt_paths_triples"]
        if triples:
            gt_triples_nonempty += 1
            gt_triples_len.append(len(triples[0]) if triples and len(triples) > 0 else 0)

    stats: Dict[str, Any] = {
        "num_samples": len(samples_list),
        "avg_selected_edges": statistics.mean(edge_counts) if edge_counts else 0.0,
        "median_selected_edges": statistics.median(edge_counts) if edge_counts else 0.0,
        "max_selected_edges": max(edge_counts) if edge_counts else 0,
        "total_selected_edges": total_edges,
        "pos_selected_edges_ratio": _safe_ratio(pos_edges, total_edges),
        "gt_path_exists_ratio": _safe_ratio(gt_path_exists, len(samples_list)),
        "gt_path_len_min": min(path_lengths) if path_lengths else 0,
        "gt_path_len_max": max(path_lengths) if path_lengths else 0,
        "gt_path_len_mean": statistics.mean(path_lengths) if path_lengths else 0.0,
        "gt_path_edges_selected_ratio": _safe_ratio(path_edges_total, total_edges),
        "gt_path_edges_pos_ratio": _safe_ratio(path_edges_pos, path_edges_total),
        "gt_path_edges_in_selected_ratio": _safe_ratio(path_edges_in_selected, path_edges_total),
        "gt_path_edges_in_top_ratio": _safe_ratio(path_edges_in_top, path_edges_total),
        "coverage_selected_full_ratio": _safe_ratio(coverage_sel_full, with_gt_path),
        "coverage_selected_partial_ratio": _safe_ratio(coverage_sel_partial, with_gt_path),
        "coverage_top_full_ratio": _safe_ratio(coverage_top_full, with_gt_path),
        "coverage_top_partial_ratio": _safe_ratio(coverage_top_partial, with_gt_path),
        "retrieval_failed_ratio": _safe_ratio(retrieval_failed, len(samples_list)),
        "score_mean_on_path": statistics.mean(scores_on_path) if scores_on_path else 0.0,
        "score_mean_off_path": statistics.mean(scores_off_path) if scores_off_path else 0.0,
        "score_delta": (statistics.mean(scores_on_path) - statistics.mean(scores_off_path))
        if scores_on_path and scores_off_path
        else 0.0,
        "gt_triples_nonempty_ratio": _safe_ratio(gt_triples_nonempty, len(samples_list)),
        "gt_triples_len_min": min(gt_triples_len) if gt_triples_len else 0,
        "gt_triples_len_max": max(gt_triples_len) if gt_triples_len else 0,
        "gt_triples_len_mean": statistics.mean(gt_triples_len) if gt_triples_len else 0.0,
    }
    return stats


def _print_stats(path: Path, stats: Dict[str, Any], pos_threshold: float) -> None:
    print(f"== {path} ==")
    print(f"pos_threshold: {pos_threshold}")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.6f}")
        else:
            print(f"{key:30s}: {value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize g_agent cache statistics.")
    parser.add_argument("paths", nargs="+", help="Paths to g_agent .pt files")
    parser.add_argument("--pos-threshold", type=float, default=0.5, help="Positive label threshold")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to audit (random subset)",
    )
    args = parser.parse_args()

    for raw_path in args.paths:
        path = Path(raw_path).expanduser().resolve()
        samples = _load_samples(path)
        stats = _compute_stats(
            samples,
            pos_threshold=args.pos_threshold,
            max_samples=args.max_samples,
        )
        _print_stats(path, stats, args.pos_threshold)


if __name__ == "__main__":
    main()
import os
import sys
import types
