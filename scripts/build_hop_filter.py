from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import rootutils
import torch
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.preprocess.labels.edge_retrieval import (  # noqa: E402
    resolve_forward_shortest_path_trajectory,
)
from src.datasets.graph_retrieval_dataset import create_graph_retrieval_dataset  # noqa: E402
from src.runs.common import compose_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sample-id filter for answer-committed RankFlow probes above a "
            "hop threshold."
        )
    )
    parser.add_argument(
        "--dataset", required=True, help="Dataset config name, e.g. webqsp-sub"
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=("train", "validation", "test"),
        help="Which LMDB split to scan.",
    )
    parser.add_argument(
        "--min-hop",
        type=int,
        default=2,
        help="Keep samples whose shortest reachable answer path has hop >= min-hop.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to <dataset.out_dir>/<split>_hop_ge_<min-hop>.json.",
    )
    return parser.parse_args()


def _resolve_num_nodes(raw_sample: dict[str, Any]) -> int:
    raw_num_nodes = raw_sample.get("num_nodes")
    if raw_num_nodes is not None:
        return int(torch.as_tensor(raw_num_nodes).view(-1)[0].item())
    return int(torch.as_tensor(raw_sample["node_entity_ids"]).numel())


def _default_output_path(*, dataset_out_dir: Path, split: str, min_hop: int) -> Path:
    return dataset_out_dir / f"{split}_hop_ge_{int(min_hop)}.json"


def main() -> None:
    args = parse_args()
    if int(args.min_hop) < 0:
        raise ValueError("--min-hop must be >= 0.")

    cfg = compose_config(
        config_name="train.yaml",
        overrides=[
            f"dataset={args.dataset}",
            "extras.print_config=false",
            "extras.enforce_tags=false",
        ],
    )
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    if not isinstance(dataset_cfg, dict):
        raise TypeError(
            f"Expected dataset config to resolve to a mapping, got {type(dataset_cfg)!r}."
        )
    dataset_out_dir = Path(str(dataset_cfg["out_dir"]))
    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else _default_output_path(
            dataset_out_dir=dataset_out_dir,
            split=str(args.split),
            min_hop=int(args.min_hop),
        )
    )

    dataset = create_graph_retrieval_dataset(
        cfg=dataset_cfg, split_name=str(args.split)
    )
    kept_sample_ids: list[str] = []
    hop_histogram: Counter[str] = Counter()
    unreachable_count = 0
    total_samples = len(dataset.sample_ids)

    try:
        for sample_id in dataset.sample_ids:
            raw_sample = dataset._load_raw_sample(sample_id)
            shortest_path = resolve_forward_shortest_path_trajectory(
                edge_index=torch.as_tensor(raw_sample["edge_index"], dtype=torch.long),
                anchor_local_indices=torch.as_tensor(
                    raw_sample["anchor_local_indices"], dtype=torch.long
                ),
                a_local_indices=torch.as_tensor(
                    raw_sample["a_local_indices"], dtype=torch.long
                ),
                num_nodes=_resolve_num_nodes(raw_sample),
            )
            if shortest_path is None:
                unreachable_count += 1
                hop_histogram["unreachable"] += 1
                continue
            hop_length = int(shortest_path.hop_length)
            hop_histogram[str(hop_length)] += 1
            if hop_length >= int(args.min_hop):
                kept_sample_ids.append(str(sample_id))
    finally:
        dataset.close()

    output = {
        "dataset": str(args.dataset),
        "split": str(args.split),
        "min_hop": int(args.min_hop),
        "total_samples": int(total_samples),
        "kept_samples": int(len(kept_sample_ids)),
        "unreachable_samples": int(unreachable_count),
        "hop_histogram": dict(sorted(hop_histogram.items(), key=lambda item: item[0])),
        "sample_ids": kept_sample_ids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output_path": str(output_path), **output}, ensure_ascii=True, indent=2
        )
    )


if __name__ == "__main__":
    main()
