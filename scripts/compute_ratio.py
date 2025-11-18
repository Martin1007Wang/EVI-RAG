#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from typing import Any, Dict

import torch

from src.data.dataset import RetrievalDataset

log = logging.getLogger("compute_ratio")


def _build_dataset_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "name": args.name,
        "data_dir": args.data_dir,
        "split": args.split,
    }
    if args.sample_limit is not None:
        cfg["sample_limit"] = {args.split: args.sample_limit}
    return cfg


def compute_ratio(dataset: RetrievalDataset, max_samples: int | None = None) -> float:
    total_pos = 0
    total = 0
    limit = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    log.info("Scanning %s samples from split=%s", limit, dataset.split)
    for idx in range(limit):
        sample = dataset.get(idx)
        labels = getattr(sample, "labels", None)
        if labels is None:
            continue
        labels = labels.long().view(-1)
        total_pos += (labels == 1).sum().item()
        total += int(labels.numel())
    if total == 0:
        return 0.0
    return total_pos / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute positive ratio for retrieval datasets.")
    parser.add_argument("--name", required=True, help="Dataset name (used for logging only).")
    parser.add_argument("--data-dir", required=True, help="Dataset root directory.")
    parser.add_argument("--split", default="train", help="Dataset split to inspect.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples to scan.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional sample_limit override.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    cfg = _build_dataset_config(args)
    dataset = RetrievalDataset(cfg)
    ratio = compute_ratio(dataset, max_samples=args.max_samples)
    log.info("Positive ratio: %.4f", ratio)


if __name__ == "__main__":
    main()
