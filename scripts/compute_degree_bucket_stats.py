#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import lmdb
import torch
import pyarrow.parquet as pq
from safetensors.torch import load

_INVERSE_SUFFIX = "__inv"
_SELF_RELATION_ID = -1


def _load_inverse_mask(relation_vocab_path: Path) -> torch.Tensor:
    table = pq.read_table(relation_vocab_path, columns=["relation_id", "kg_id"])
    rel_ids = table.column("relation_id").to_numpy()
    kg_ids = table.column("kg_id").to_pylist()
    max_rel_id = int(rel_ids.max())
    inv_mask = torch.zeros((max_rel_id + 1,), dtype=torch.bool)
    for rid, kg in zip(rel_ids, kg_ids):
        if str(kg).endswith(_INVERSE_SUFFIX):
            inv_mask[int(rid)] = True
    return inv_mask


def _iter_lmdb_paths(embeddings_dir: Path, split: str) -> Iterable[Path]:
    base = embeddings_dir / f"{split}.lmdb"
    if base.exists():
        yield base
        return
    shards = sorted(embeddings_dir.glob(f"{split}.shard*.lmdb"))
    for shard in shards:
        yield shard


def _max_log_degree_for_dataset(dataset_dir: Path) -> float:
    relation_vocab_path = dataset_dir / "normalized" / "relation_vocab.parquet"
    embeddings_dir = dataset_dir / "materialized" / "embeddings"
    if not relation_vocab_path.exists():
        raise FileNotFoundError(f"relation_vocab.parquet not found: {relation_vocab_path}")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"embeddings dir not found: {embeddings_dir}")

    inv_mask = _load_inverse_mask(relation_vocab_path)
    max_log_deg = 0.0
    splits = ("train", "validation", "test")
    for split in splits:
        lmdb_paths = list(_iter_lmdb_paths(embeddings_dir, split))
        if not lmdb_paths:
            continue
        for lmdb_path in lmdb_paths:
            env = lmdb.open(
                str(lmdb_path),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
                max_readers=256,
            )
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                for _, payload in cursor:
                    data = load(payload)
                    edge_index = data["edge_index"]
                    edge_attr = data["edge_attr"].view(-1)
                    if edge_index.numel() == 0:
                        continue
                    num_nodes = int(data["num_nodes"].item())
                    valid = edge_attr >= 0
                    edge_is_inverse = torch.zeros_like(edge_attr, dtype=torch.bool)
                    if valid.any():
                        edge_is_inverse[valid] = inv_mask.index_select(0, edge_attr[valid])
                    self_loop = edge_attr == _SELF_RELATION_ID
                    edge_mask_fwd = (~edge_is_inverse) | self_loop
                    if not edge_mask_fwd.any():
                        continue
                    heads = edge_index[0][edge_mask_fwd]
                    tails = edge_index[1][edge_mask_fwd]
                    out_deg = torch.bincount(heads, minlength=num_nodes)
                    in_deg = torch.bincount(tails, minlength=num_nodes)
                    log_out = torch.log(out_deg.to(torch.float32) + 1.0)
                    log_in = torch.log(in_deg.to(torch.float32) + 1.0)
                    local_max = torch.max(torch.cat([log_out, log_in])).item()
                    if local_max > max_log_deg:
                        max_log_deg = local_max
            env.close()
    return max_log_deg


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute max log degree for degree bucketization.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/mnt/data/retrieval_dataset"),
        help="Root directory containing dataset subdirs (e.g., webqsp, cwq).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["webqsp", "cwq"],
        help="Dataset names to scan.",
    )
    args = parser.parse_args()

    max_values = {}
    for name in args.datasets:
        dataset_dir = args.data_dir / name
        if not dataset_dir.exists():
            print(f"[skip] {name}: {dataset_dir} not found")
            continue
        max_values[name] = _max_log_degree_for_dataset(dataset_dir)
        print(f"{name}: max_log_deg={max_values[name]:.6f}")

    if max_values:
        unified = max(max_values.values())
        print(f"unified_max_log_deg={unified:.6f}")


if __name__ == "__main__":
    main()
