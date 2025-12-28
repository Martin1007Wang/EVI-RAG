"""Analyze pair_shortest_lengths distribution in g_agent caches.

Usage:
  python scripts/analyze_gflownet_pairs.py --max-steps 3 /path/train_g_agent.pt /path/val_g_agent.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def _load_g_agent_module() -> object:
    repo_root = Path(__file__).resolve().parent.parent
    module_path = repo_root / "src" / "data" / "g_agent_dataset.py"
    spec = importlib.util.spec_from_file_location("g_agent_dataset_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["g_agent_dataset_local"] = module
    spec.loader.exec_module(module)
    return module


def _iter_indices(total: int, max_samples: int | None, seed: int) -> Iterable[int]:
    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return range(total)
    rng = random.Random(seed)
    return rng.sample(range(total), max_samples)


def _ratio(num: int, denom: int) -> float:
    return float(num / denom) if denom > 0 else 0.0


def _quantiles(values: List[int], qs: Iterable[float]) -> Dict[float, float]:
    if not values:
        return {float(q): 0.0 for q in qs}
    t = torch.tensor(values, dtype=torch.float32)
    return {float(q): float(torch.quantile(t, float(q)).item()) for q in qs}


def analyze_path(
    path: Path,
    *,
    max_steps: int | None,
    max_samples: int | None,
    max_pairs_sample: int,
    hist_max: int,
    drop_unreachable: bool,
    prefer_jsonl: bool,
    convert_pt_to_jsonl: bool,
    seed: int,
) -> None:
    module = _load_g_agent_module()
    DatasetCls = getattr(module, "GAgentPyGDataset")
    dataset = DatasetCls(
        path,
        drop_unreachable=drop_unreachable,
        prefer_jsonl=prefer_jsonl,
        convert_pt_to_jsonl=convert_pt_to_jsonl,
    )

    total_samples = len(dataset)
    indices = list(_iter_indices(total_samples, max_samples, seed))

    samples_with_pairs = 0
    samples_without_pairs = 0
    samples_with_any_valid = 0
    samples_with_only_long = 0
    samples_unreachable = 0

    pair_counts: List[int] = []
    length_samples: List[int] = []
    hist: Dict[int, int] = {}
    hist_over = 0

    pairs_total = 0
    pairs_zero = 0
    pairs_le_max = 0
    pairs_gt_max = 0
    mismatched_pairs = 0

    rng = random.Random(seed)
    seen_pairs = 0

    for idx in indices:
        data = dataset[idx]
        pair_lengths = data.pair_shortest_lengths.view(-1).to(torch.long)
        pair_count = int(pair_lengths.numel())
        pair_counts.append(pair_count)

        is_reachable = bool(getattr(data, "is_answer_reachable", torch.tensor([True])).view(-1)[0].item())
        if not is_reachable:
            samples_unreachable += 1

        if pair_count == 0:
            samples_without_pairs += 1
            continue

        samples_with_pairs += 1
        if max_steps is not None:
            valid_mask = pair_lengths <= int(max_steps)
            if bool(valid_mask.any().item()):
                samples_with_any_valid += 1
            else:
                samples_with_only_long += 1

        pair_start = data.pair_start_node_locals.view(-1)
        if pair_start.numel() != pair_lengths.numel():
            mismatched_pairs += 1

        for length in pair_lengths.tolist():
            length_int = int(length)
            pairs_total += 1
            if length_int == 0:
                pairs_zero += 1
            if max_steps is not None:
                if length_int <= int(max_steps):
                    pairs_le_max += 1
                else:
                    pairs_gt_max += 1
            if length_int <= hist_max:
                hist[length_int] = hist.get(length_int, 0) + 1
            else:
                hist_over += 1

            seen_pairs += 1
            if max_pairs_sample > 0:
                if len(length_samples) < max_pairs_sample:
                    length_samples.append(length_int)
                else:
                    j = rng.randrange(seen_pairs)
                    if j < max_pairs_sample:
                        length_samples[j] = length_int

    total_seen = len(indices)
    print(f"\n== {path} ==")
    print(f"samples_total: {total_seen}/{total_samples}")
    print(f"samples_with_pairs: {samples_with_pairs} ({_ratio(samples_with_pairs, total_seen):.4f})")
    print(f"samples_without_pairs: {samples_without_pairs} ({_ratio(samples_without_pairs, total_seen):.4f})")
    print(f"samples_unreachable: {samples_unreachable} ({_ratio(samples_unreachable, total_seen):.4f})")
    print(f"pairs_total: {pairs_total}")
    if mismatched_pairs > 0:
        print(f"pair_count_mismatch: {mismatched_pairs}")

    if pairs_total > 0:
        q = _quantiles(length_samples, [0.5, 0.9, 0.95, 0.99])
        print(
            "pair_len_stats: "
            f"min={min(length_samples) if length_samples else 0} "
            f"p50={q[0.5]:.2f} p90={q[0.9]:.2f} p95={q[0.95]:.2f} p99={q[0.99]:.2f} "
            f"max={max(length_samples) if length_samples else 0} "
            f"(sampled={len(length_samples)}/{pairs_total})"
        )
        print(f"pair_len_zero: {pairs_zero} ({_ratio(pairs_zero, pairs_total):.4f})")

    if max_steps is not None:
        print(f"max_steps: {max_steps}")
        print(f"pairs_len<=max_steps: {pairs_le_max} ({_ratio(pairs_le_max, pairs_total):.4f})")
        print(f"pairs_len>max_steps: {pairs_gt_max} ({_ratio(pairs_gt_max, pairs_total):.4f})")
        print(
            "samples_with_any_valid_pair: "
            f"{samples_with_any_valid} ({_ratio(samples_with_any_valid, max(1, samples_with_pairs)):.4f})"
        )
        print(
            "samples_with_only_long_pairs: "
            f"{samples_with_only_long} ({_ratio(samples_with_only_long, max(1, samples_with_pairs)):.4f})"
        )

    if pair_counts:
        pair_counts_t = torch.tensor(pair_counts, dtype=torch.float32)
        print(
            "pairs_per_sample: "
            f"mean={pair_counts_t.mean().item():.2f} "
            f"median={pair_counts_t.median().item():.2f} "
            f"max={int(pair_counts_t.max().item())}"
        )

    if hist:
        keys = sorted(hist.keys())
        hist_items = [f"{k}:{hist[k]}" for k in keys]
        if hist_over > 0:
            hist_items.append(f">{hist_max}:{hist_over}")
        print("pair_len_hist: " + ", ".join(hist_items))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pair_shortest_lengths in g_agent caches.")
    parser.add_argument("paths", nargs="+", help="Paths to g_agent cache files (.pt or .jsonl).")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps to evaluate pair validity.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples per cache.")
    parser.add_argument("--max-pairs-sample", type=int, default=200000, help="Reservoir size for quantiles.")
    parser.add_argument("--hist-max", type=int, default=10, help="Max length to show in histogram buckets.")
    parser.add_argument("--drop-unreachable", action="store_true", help="Drop unreachable samples.")
    parser.add_argument("--prefer-jsonl", action="store_true", help="Prefer .jsonl if present.")
    parser.add_argument("--no-prefer-jsonl", action="store_true", help="Disable .jsonl preference.")
    parser.add_argument("--convert-pt-to-jsonl", action="store_true", help="Convert .pt to .jsonl for streaming.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for sampling.")
    args = parser.parse_args()

    prefer_jsonl = True
    if args.no_prefer_jsonl:
        prefer_jsonl = False
    if args.prefer_jsonl:
        prefer_jsonl = True

    for raw_path in args.paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"g_agent cache not found: {path}")
        analyze_path(
            path,
            max_steps=args.max_steps,
            max_samples=args.max_samples,
            max_pairs_sample=args.max_pairs_sample,
            hist_max=args.hist_max,
            drop_unreachable=args.drop_unreachable,
            prefer_jsonl=prefer_jsonl,
            convert_pt_to_jsonl=args.convert_pt_to_jsonl,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
