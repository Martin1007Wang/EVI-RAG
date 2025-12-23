#!/usr/bin/env python3
"""Report top-K relation bigram transitions from a precomputed matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq


def _load_relation_map(vocab_path: Path) -> Dict[int, str]:
    table = pq.read_table(vocab_path, columns=["relation_id", "kg_id", "label"])
    rel_ids = table.column("relation_id").to_numpy()
    kg_ids = table.column("kg_id").to_pylist()
    labels = table.column("label").to_pylist()
    mapping: Dict[int, str] = {}
    for rid, kg_id, label in zip(rel_ids, kg_ids, labels):
        text = label if label is not None and str(label).strip() else kg_id
        mapping[int(rid)] = str(text)
    return mapping


def _topk(flat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = int(min(k, flat.size))
    if k <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=flat.dtype)
    idx = np.argpartition(flat, -k)[-k:]
    vals = flat[idx]
    order = np.argsort(-vals)
    return idx[order], vals[order]


def main() -> None:
    parser = argparse.ArgumentParser(description="Report top-K relation transitions.")
    parser.add_argument("--bigram", type=Path, required=True)
    parser.add_argument("--relation-vocab", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--top-k-extra", type=int, default=100)
    args = parser.parse_args()

    bigram = np.load(args.bigram, mmap_mode="r")
    if bigram.ndim != 2 or bigram.shape[0] != bigram.shape[1]:
        raise ValueError(f"Expected square matrix, got {bigram.shape}")
    num_rel = bigram.shape[0]

    rel_map = _load_relation_map(args.relation_vocab)
    if len(rel_map) != num_rel:
        raise ValueError(
            f"relation_vocab size mismatch: vocab={len(rel_map)} vs matrix={num_rel}"
        )

    flat = bigram.reshape(-1)
    total = int(flat.sum())
    nnz = int(np.count_nonzero(flat))

    idx, vals = _topk(flat, args.top_k)
    idx_extra, vals_extra = _topk(flat, args.top_k_extra)
    share_top_k = float(vals.sum() / total) if total > 0 else 0.0
    share_top_extra = float(vals_extra.sum() / total) if total > 0 else 0.0

    row_sum = bigram.sum(axis=1)
    row_max = bigram.max(axis=1)
    row_mask = row_sum > 0
    row_max_ratio = row_max[row_mask] / row_sum[row_mask]
    avg_row_max = float(row_max_ratio.mean()) if row_max_ratio.size else 0.0
    med_row_max = float(np.median(row_max_ratio)) if row_max_ratio.size else 0.0
    strong_rows = int(np.count_nonzero(row_max_ratio >= 0.5))
    strong_ratio = float(strong_rows / row_max_ratio.size) if row_max_ratio.size else 0.0

    print(f"[stats] total_transitions={total}")
    print(f"[stats] nonzero_pairs={nnz} sparsity={nnz / (num_rel * num_rel):.6f}")
    print(f"[stats] top_{args.top_k}_share={share_top_k:.4f}")
    print(f"[stats] top_{args.top_k_extra}_share={share_top_extra:.4f}")
    print(f"[stats] row_max_ratio_avg={avg_row_max:.4f} median={med_row_max:.4f} >=0.5_ratio={strong_ratio:.4f}")
    print("")
    print("rank\tcount\tshare\trel_in_id\trel_in\t->\trel_out_id\trel_out\tP(out|in)\tP(in|out)")

    row_sum = row_sum.astype(np.float64)
    col_sum = bigram.sum(axis=0).astype(np.float64)
    for rank, (flat_idx, count) in enumerate(zip(idx, vals), start=1):
        r_in = int(flat_idx // num_rel)
        r_out = int(flat_idx % num_rel)
        share = float(count / total) if total > 0 else 0.0
        p_out_in = float(count / row_sum[r_in]) if row_sum[r_in] > 0 else 0.0
        p_in_out = float(count / col_sum[r_out]) if col_sum[r_out] > 0 else 0.0
        rel_in = rel_map.get(r_in, str(r_in))
        rel_out = rel_map.get(r_out, str(r_out))
        print(
            f"{rank}\t{int(count)}\t{share:.6f}\t{r_in}\t{rel_in}\t->\t"
            f"{r_out}\t{rel_out}\t{p_out_in:.4f}\t{p_in_out:.4f}"
        )


if __name__ == "__main__":
    main()
