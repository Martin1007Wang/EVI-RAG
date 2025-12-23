#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


log = logging.getLogger("compute_gpt_difficulty")


def load_gpt_triples(base_dir: Path, dataset: str) -> pd.DataFrame:
    """Load gpt_triples.pth for a given dataset and compute basic stats.

    Each row corresponds to one question / sample_id.
    """
    path = base_dir / dataset / "gpt_triples.pth"
    if not path.exists():
        raise FileNotFoundError(f"gpt_triples file not found: {path}")

    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict from {path}, got {type(data)}")

    rows: List[Dict[str, object]] = []
    for sample_id, triples in data.items():
        if triples is None:
            triples = []
        if not isinstance(triples, Iterable):
            log.warning("Sample %s has non-iterable triples=%r; skipping", sample_id, type(triples))
            continue

        triples_list = list(triples)
        n_triples = len(triples_list)
        entities = set()
        relations = set()
        for t in triples_list:
            if not isinstance(t, (list, tuple)) or len(t) != 3:
                continue
            h, r, v = t
            entities.add(str(h))
            entities.add(str(v))
            relations.add(str(r))

        n_entities = len(entities)
        n_relations = len(relations)
        branching = n_triples / (n_entities + 1e-6) if n_entities > 0 else 0.0

        rows.append(
            {
                "sample_id": str(sample_id),
                "num_triples": int(n_triples),
                "num_entities": int(n_entities),
                "num_relations": int(n_relations),
                "branching": float(branching),
                "dataset": dataset,
            }
        )

    df = pd.DataFrame(rows)
    log.info("Loaded %d samples for dataset=%s", len(df), dataset)
    return df


def compute_global_thresholds(df: pd.DataFrame) -> Tuple[float, float, float, float]:
    """Compute global num_triples quantiles (20/40/60/80%)."""
    t_all = df["num_triples"].to_numpy()
    q20, q40, q60, q80 = np.quantile(t_all, [0.2, 0.4, 0.6, 0.8])
    log.info(
        "Global num_triples quantiles (20/40/60/80%%): %.3f, %.3f, %.3f, %.3f",
        q20,
        q40,
        q60,
        q80,
    )
    return q20, q40, q60, q80


def assign_difficulty_levels(df: pd.DataFrame, thresholds: Tuple[float, float, float, float]) -> pd.DataFrame:
    """Assign difficulty_level in {1..5} based on global num_triples quantiles."""
    q20, q40, q60, q80 = thresholds

    def difficulty_from_triples(t: int) -> int:
        if t <= q20:
            return 1
        if t <= q40:
            return 2
        if t <= q60:
            return 3
        if t <= q80:
            return 4
        return 5

    df = df.copy()
    df["difficulty_level"] = df["num_triples"].apply(difficulty_from_triples)
    counts = df["difficulty_level"].value_counts().sort_index()
    log.info("Difficulty level distribution:\n%s", counts.to_string())
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Compute per-question GPT evidence complexity statistics and difficulty levels " "from gpt_triples.pth.")
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/data/retrieve",
        help="Base directory containing dataset subfolders (default: /mnt/data/retrieve).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["webqsp", "cwq"],
        help="Dataset names to process (subdirectories under base-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/gpt_difficulty",
        help="Directory where CSV files will be written (relative to CWD).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []
    for name in args.datasets:
        df_ds = load_gpt_triples(base_dir, name)
        all_dfs.append(df_ds)

    if not all_dfs:
        log.error("No datasets loaded. Check --datasets and --base-dir.")
        return

    df_all = pd.concat(all_dfs, ignore_index=True)
    thresholds = compute_global_thresholds(df_all)

    # Assign difficulty using global thresholds for all datasets
    df_all = assign_difficulty_levels(df_all, thresholds)

    # Save per-dataset CSVs
    for name in args.datasets:
        df_ds = df_all[df_all["dataset"] == name].reset_index(drop=True)
        out_path = output_dir / f"gpt_difficulty_{name}.csv"
        df_ds.to_csv(out_path, index=False)
        log.info("Wrote %d rows to %s", len(df_ds), out_path)

    # Also save a combined CSV if multiple datasets are processed
    if len(args.datasets) > 1:
        combined_path = output_dir / "gpt_difficulty_all.csv"
        df_all.to_csv(combined_path, index=False)
        log.info("Wrote combined CSV with %d rows to %s", len(df_all), combined_path)


if __name__ == "__main__":
    main()
