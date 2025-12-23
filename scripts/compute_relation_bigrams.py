#!/usr/bin/env python3
"""Compute relation bigram transition matrix from shortest-path edge masks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _load_num_relations(vocab_path: Path) -> int:
    table = pq.read_table(vocab_path, columns=["relation_id"])
    if table.num_rows == 0:
        raise ValueError(f"Empty relation vocab: {vocab_path}")
    rel_ids = table.column("relation_id").to_numpy()
    max_id = int(rel_ids.max())
    num_rel = max_id + 1
    if num_rel != table.num_rows:
        raise ValueError(
            f"Non-contiguous relation IDs in {vocab_path}: max_id={max_id}, rows={table.num_rows}"
        )
    return num_rel


def _resolve_mask_field(schema_names: Iterable[str], requested: str, fallback: str | None) -> str:
    names = set(schema_names)
    if requested in names:
        return requested
    if fallback and fallback in names:
        print(f"[info] mask field '{requested}' missing; falling back to '{fallback}'.")
        return fallback
    raise ValueError(f"Mask field '{requested}' not found in schema: {sorted(names)}")


def _coerce_mask(mask: Sequence[object], num_edges: int) -> np.ndarray:
    mask_arr = np.asarray(mask)
    if mask_arr.size == 0:
        return np.zeros(num_edges, dtype=bool)
    if mask_arr.dtype == np.bool_:
        if mask_arr.size != num_edges:
            raise ValueError(f"Mask length {mask_arr.size} != num_edges {num_edges}")
        return mask_arr
    if mask_arr.ndim != 1:
        raise ValueError("Mask must be a 1D array.")
    if mask_arr.size == num_edges:
        return mask_arr.astype(bool)
    # Interpret as a list of edge indices.
    if int(mask_arr.min()) < 0 or int(mask_arr.max()) >= num_edges:
        raise ValueError("Mask indices out of range.")
    out = np.zeros(num_edges, dtype=bool)
    out[mask_arr.astype(np.int64)] = True
    return out


def _update_bigram(
    bigram: np.ndarray,
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    edge_rel: Sequence[int],
    mask: Sequence[object],
    num_nodes: int,
) -> Tuple[int, int]:
    if num_nodes <= 0:
        return 0, 0
    edge_src_arr = np.asarray(edge_src, dtype=np.int64)
    edge_dst_arr = np.asarray(edge_dst, dtype=np.int64)
    edge_rel_arr = np.asarray(edge_rel, dtype=np.int64)
    if edge_src_arr.size == 0:
        return 0, 0
    if edge_src_arr.size != edge_dst_arr.size or edge_src_arr.size != edge_rel_arr.size:
        raise ValueError("edge_src/edge_dst/edge_rel length mismatch.")

    mask_arr = _coerce_mask(mask, edge_src_arr.size)
    if not mask_arr.any():
        return int(edge_src_arr.size), 0

    src = edge_src_arr[mask_arr]
    dst = edge_dst_arr[mask_arr]
    rel = edge_rel_arr[mask_arr]
    if src.size == 0:
        return int(edge_src_arr.size), 0

    rel_unique, rel_inv = np.unique(rel, return_inverse=True)
    rel_unique = rel_unique.astype(np.int64, copy=False)
    local_rel = int(rel_unique.size)
    idx_out = src * local_rel + rel_inv
    idx_in = dst * local_rel + rel_inv

    counts_out = np.bincount(idx_out, minlength=num_nodes * local_rel).reshape(num_nodes, local_rel)
    counts_in = np.bincount(idx_in, minlength=num_nodes * local_rel).reshape(num_nodes, local_rel)
    local = counts_in.T @ counts_out

    bigram[np.ix_(rel_unique, rel_unique)] += local
    return int(edge_src_arr.size), int(src.size)


def _iter_batches(graphs_path: Path, columns: Sequence[str], batch_size: int) -> Iterable[dict]:
    pf = pq.ParquetFile(graphs_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pydict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute relation bigram transition matrix.")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data/retrieval_dataset/webqsp"))
    parser.add_argument("--graphs-path", type=Path, default=None)
    parser.add_argument("--relation-vocab", type=Path, default=None)
    parser.add_argument("--mask-field", type=str, default="gt_shortest_edge_mask")
    parser.add_argument("--fallback-mask-field", type=str, default="positive_triple_mask")
    parser.add_argument("--output", type=Path, default=Path("relation_bigrams.npy"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    graphs_path = args.graphs_path or (args.data_root / "normalized" / "graphs.parquet")
    relation_vocab = args.relation_vocab or (args.data_root / "normalized" / "relation_vocab.parquet")
    if not graphs_path.exists():
        raise FileNotFoundError(f"graphs.parquet not found: {graphs_path}")
    if not relation_vocab.exists():
        raise FileNotFoundError(f"relation_vocab.parquet not found: {relation_vocab}")

    num_rel = _load_num_relations(relation_vocab)
    print(f"[info] num_relations={num_rel}")

    schema_names = ds.dataset(graphs_path, format="parquet").schema.names
    mask_field = _resolve_mask_field(schema_names, args.mask_field, args.fallback_mask_field)
    columns = ["edge_src", "edge_dst", "edge_relation_ids", mask_field, "node_entity_ids"]

    output_path = args.output
    if output_path.suffix != ".npy":
        raise ValueError("Output must be a .npy file for memmap-friendly writing.")
    bigram = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int64, shape=(num_rel, num_rel))
    bigram[:] = 0

    total_rows = pq.ParquetFile(graphs_path).metadata.num_rows
    pbar = tqdm(total=total_rows, desc="Scanning graphs") if tqdm else None

    total_edges = 0
    total_masked = 0
    processed = 0
    for batch in _iter_batches(graphs_path, columns, args.batch_size):
        rows = len(batch["edge_src"])
        for edge_src, edge_dst, edge_rel, mask, node_ids in zip(
            batch["edge_src"],
            batch["edge_dst"],
            batch["edge_relation_ids"],
            batch[mask_field],
            batch["node_entity_ids"],
        ):
            num_nodes = len(node_ids)
            edges, masked = _update_bigram(
                bigram=bigram,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_rel=edge_rel,
                mask=mask,
                num_nodes=num_nodes,
            )
            total_edges += edges
            total_masked += masked
        processed += rows
        if pbar:
            pbar.update(rows)

    if pbar:
        pbar.close()

    bigram.flush()
    print(f"[done] graphs={processed} edges={total_edges} masked_edges={total_masked}")
    print(f"[done] saved matrix: {output_path}")


if __name__ == "__main__":
    main()
