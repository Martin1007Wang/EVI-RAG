#!/usr/bin/env python
"""
Compute one-hop edge statistics for seed entities in g_retrieval LMDBs.

Metrics per seed:
  - one-hop edge count: edges incident to the seed (head or tail)
  - positive ratio: fraction of those edges with label > 0.5

Percentiles are computed over all seeds (positive ratio excludes zero-edge seeds).
"""
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow.dataset as ds
import torch


from src.data.components.embedding_store import EmbeddingStore

PCTS = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]


def _percentile(sorted_vals: List[float], p: float) -> float | None:
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    rank = (p / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    w = rank - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _percentiles(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(values)
    return {str(p): _percentile(vals, p) for p in PCTS if vals}


def _load_graph_ids_by_split(parquet_path: Path, limit: int) -> Dict[str, List[str]]:
    dataset = ds.dataset(str(parquet_path), format="parquet")
    if "graph_id" not in dataset.schema.names or "split" not in dataset.schema.names:
        raise ValueError(f"questions.parquet missing graph_id/split: {parquet_path}")
    splits: Dict[str, List[str]] = defaultdict(list)
    seen = 0
    scanner = dataset.scanner(columns=["graph_id", "split"], batch_size=65536)
    for batch in scanner.to_batches():
        graph_ids = batch.column(0).to_pylist()
        split_vals = batch.column(1).to_pylist()
        for gid, sp in zip(graph_ids, split_vals):
            splits[str(sp)].append(str(gid))
            seen += 1
            if limit > 0 and seen >= limit:
                return splits
    return splits


def _compute_stats_for_root(root: Path, *, limit: int, progress_every: int) -> Dict[str, object]:
    parquet_path = root / "normalized" / "questions.parquet"
    splits = _load_graph_ids_by_split(parquet_path, limit=limit)
    all_counts: List[int] = []
    all_ratios: List[float] = []
    zero_edge_seeds = 0
    total_seeds = 0

    for split, graph_ids in splits.items():
        lmdb_path = root / "materialized" / "embeddings" / f"{split}.lmdb"
        if not lmdb_path.exists():
            continue
        store = EmbeddingStore(lmdb_path)
        try:
            for idx, gid in enumerate(graph_ids, start=1):
                raw = store.load_sample(gid)
                edge_index = raw["edge_index"]
                labels = raw["labels"]
                q_local_indices = raw["q_local_indices"]
                num_nodes = int(raw["num_nodes"])
                if edge_index.numel() == 0 or q_local_indices.numel() == 0:
                    continue
                heads = edge_index[0].to(dtype=torch.long, device="cpu")
                tails = edge_index[1].to(dtype=torch.long, device="cpu")
                edge_counts = torch.bincount(heads, minlength=num_nodes) + torch.bincount(tails, minlength=num_nodes)
                pos_mask = labels.view(-1) > 0.5
                if pos_mask.any():
                    pos_heads = heads[pos_mask]
                    pos_tails = tails[pos_mask]
                    pos_counts = torch.bincount(pos_heads, minlength=num_nodes) + torch.bincount(pos_tails, minlength=num_nodes)
                else:
                    pos_counts = torch.zeros(num_nodes, dtype=edge_counts.dtype)
                seeds = torch.unique(q_local_indices.view(-1).to(dtype=torch.long, device="cpu"))
                for s in seeds.tolist():
                    total_seeds += 1
                    if s < 0 or s >= num_nodes:
                        continue
                    count = int(edge_counts[s].item())
                    all_counts.append(count)
                    if count <= 0:
                        zero_edge_seeds += 1
                        continue
                    pos = int(pos_counts[s].item())
                    all_ratios.append(pos / count if count > 0 else 0.0)
                if progress_every > 0 and idx % progress_every == 0:
                    print(f"[{root.name}] {split}: processed {idx}/{len(graph_ids)} samples")
        finally:
            store.close()

    return {
        "dataset": root.name,
        "total_seeds": total_seeds,
        "zero_edge_seeds": zero_edge_seeds,
        "count_percentiles": _percentiles(all_counts),
        "ratio_percentiles": _percentiles(all_ratios),
        "sample_limit": limit,
    }


def main() -> None:
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="Seed one-hop statistics for g_retrieval LMDBs.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["/mnt/data/retrieval_dataset/cwq", "/mnt/data/retrieval_dataset/webqsp"],
        help="Dataset roots (each containing normalized/ and materialized/).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on total samples scanned per dataset (0 = no limit).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print progress every N samples (0 disables).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for JSON results.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for root_str in args.roots:
        root = Path(root_str).expanduser().resolve()
        stats = _compute_stats_for_root(root, limit=int(args.limit), progress_every=int(args.progress_every))
        out_path = out_dir / f"seed_onehop_stats_{root.name}.json"
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[{root.name}] wrote {out_path}")


if __name__ == "__main__":
    main()
