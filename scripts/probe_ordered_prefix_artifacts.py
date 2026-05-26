from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.trajectory import TrajectoryBatch, trajectory_logp
from src.weaver.state import StateBatch


@dataclass(frozen=True, slots=True)
class PrefixRecord:
    ordered_key: tuple[int, ...]
    unordered_key: frozenset[int]
    reward: float
    trajectory_logp: float


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt.resolve() if args.ckpt is not None else run_dir / "checkpoints" / "last.ckpt"

    cfg = load_run_config(
        run_dir=run_dir,
        data_dir=str(args.data_dir),
        dataset=args.dataset,
    )
    datamodule, resources = prepare_training_components(cfg, stage="fit")
    dataset = {
        "train": datamodule.train_dataset,
        "validation": datamodule.val_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }[args.split]
    if dataset is None:
        raise RuntimeError(f"split {args.split!r} is not initialized.")

    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(ckpt_path), strict=False)
    device = resolve_device(args.device)
    model.to(device)
    model.eval()

    end = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.start_idx) + int(args.max_samples))
    indices = list(range(int(args.start_idx), end))

    detail_path = Path(args.output_jsonl) if args.output_jsonl else None
    summary = RunningSummary()

    detail_fh = detail_path.open("w", encoding="utf-8") if detail_path is not None else None
    try:
        with torch.no_grad():
            for start in range(0, len(indices), int(args.batch_size)):
                batch_indices = indices[start : start + int(args.batch_size)]
                samples = [dataset[idx] for idx in batch_indices]
                batch = datamodule.collator(samples).to(device)
                ctx = model.batch_context(batch)
                trajectories = model.runner.eval_rollouts(
                    policy=model.policy,
                    context=ctx.graph,
                    features=ctx.features,
                    num_rollouts=int(args.rollouts),
                )
                records_by_graph = terminal_records_by_graph(
                    trajectories=trajectories,
                    graph=ctx.graph,
                    target=ctx.target,
                    reward_model=model.reward_model,
                )

                for graph_id in range(int(ctx.graph.num_graphs)):
                    sample_id = sample_id_for(batch, graph_id)
                    result = analyze_question(
                        dataset_name=str(cfg.dataset.name),
                        split=str(args.split),
                        sample_id=sample_id,
                        graph_id=int(graph_id),
                        records=records_by_graph.get(graph_id, []),
                        reward_similarity_tol=float(args.reward_similarity_tol),
                    )
                    summary.add(result)
                    if detail_fh is not None:
                        detail_fh.write(json.dumps(result, sort_keys=True) + "\n")
    finally:
        if detail_fh is not None:
            detail_fh.close()

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    report = summary.finalize(
        dataset_name=str(cfg.dataset.name),
        split=str(args.split),
        rollouts=int(args.rollouts),
        checkpoint=str(ckpt_path),
        epoch=checkpoint.get("epoch"),
        global_step=checkpoint.get("global_step"),
        missing=len(missing),
        unexpected=len(unexpected),
        details=str(detail_path) if detail_path is not None else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe whether ordered-prefix state multiplicity creates KGQA sampling artifacts.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/data/retrieval"))
    parser.add_argument("--dataset", choices=("webqsp", "cwq"), default=None, help="Override cfg.dataset while reusing the run model config.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means the full split.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--reward-similarity-tol", type=float, default=1.0e-5)
    parser.add_argument("--output-jsonl", default="", help="Optional per-question JSONL output path.")
    return parser.parse_args()


def load_run_config(*, run_dir: Path, data_dir: str, dataset: str | None):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = str(data_dir)
        if dataset is not None:
            apply_dataset_override(cfg, dataset=dataset, data_dir=data_dir)

        runner = cfg.model.runner
        if "replay_source" in runner:
            runner.weak_replay_source = runner.pop("replay_source")
        if "train_replay_rollouts" in runner:
            runner.pop("train_replay_rollouts")
        if "weak_replay_loss" not in cfg.model:
            cfg.model.weak_replay_loss = None
    return cfg


def apply_dataset_override(cfg: Any, *, dataset: str, data_dir: str) -> None:
    root = Path(data_dir) / dataset
    cfg.dataset.name = dataset
    cfg.dataset.dataset_scope = dataset
    cfg.dataset.root_dir = str(root)
    cfg.dataset.artifact_dir = str(root / "artifacts")
    cfg.dataset.paths.raw_dir = str(root / "raw")
    cfg.dataset.paths.metadata_dir = str(root / "metadata")
    cfg.dataset.paths.embeddings_dir = str(root / "embeddings")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(value)


def terminal_records_by_graph(
    *,
    trajectories: TrajectoryBatch,
    graph: GraphContext,
    target: TargetContext,
    reward_model,
) -> dict[int, list[PrefixRecord]]:
    if trajectories.num_trajectories == 0:
        return {}

    terminal_state = StateBatch(
        graph_ids=trajectories.graph_ids,
        edge_ids=trajectories.edge_ids,
        edge_count=trajectories.edge_count,
        budget=trajectories.budget,
    )
    rewards = reward_model(
        state=terminal_state,
        graph_context=graph,
        target_context=target,
    ).log_reward.detach().cpu()
    traj_logp = trajectory_logp(trajectories).detach().cpu()
    edge_ids = trajectories.edge_ids.detach().cpu()
    edge_count = trajectories.edge_count.detach().cpu()
    graph_ids = trajectories.graph_ids.detach().cpu()

    by_graph: dict[int, list[PrefixRecord]] = defaultdict(list)
    for row in range(int(trajectories.num_trajectories)):
        count = int(edge_count[row].item())
        ordered = tuple(int(x) for x in edge_ids[row, :count].tolist())
        unordered = frozenset(ordered)
        by_graph[int(graph_ids[row].item())].append(
            PrefixRecord(
                ordered_key=ordered,
                unordered_key=unordered,
                reward=float(rewards[row].item()),
                trajectory_logp=float(traj_logp[row].item()),
            )
        )
    return by_graph


def analyze_question(
    *,
    dataset_name: str,
    split: str,
    sample_id: str,
    graph_id: int,
    records: list[PrefixRecord],
    reward_similarity_tol: float,
) -> dict[str, Any]:
    ordered_counts: Counter[tuple[int, ...]] = Counter(record.ordered_key for record in records)
    unordered_counts: Counter[frozenset[int]] = Counter(record.unordered_key for record in records)

    ordered_rewards: dict[tuple[int, ...], list[float]] = defaultdict(list)
    unordered_rewards: dict[frozenset[int], list[float]] = defaultdict(list)
    unordered_ordered_keys: dict[frozenset[int], set[tuple[int, ...]]] = defaultdict(set)
    for record in records:
        ordered_rewards[record.ordered_key].append(record.reward)
        unordered_rewards[record.unordered_key].append(record.reward)
        unordered_ordered_keys[record.unordered_key].add(record.ordered_key)

    unordered_rows = []
    for key, sample_frequency in unordered_counts.items():
        rewards = unordered_rewards[key]
        duplicate_factor = len(unordered_ordered_keys[key])
        unordered_rows.append(
            {
                "unordered_key": sorted(int(x) for x in key),
                "sample_frequency": int(sample_frequency),
                "duplicate_factor": int(duplicate_factor),
                "reward_mean": mean(rewards),
                "reward_std": std(rewards),
                "ordered_prefix_count": int(duplicate_factor),
            }
        )

    duplicate_factors = [float(row["duplicate_factor"]) for row in unordered_rows]
    sample_frequencies = [float(row["sample_frequency"]) for row in unordered_rows]
    reward_means = [float(row["reward_mean"]) for row in unordered_rows]
    multi_prefix = [row for row in unordered_rows if int(row["duplicate_factor"]) > 1]
    similar_multi_prefix = [
        row
        for row in multi_prefix
        if float(row["reward_std"]) <= float(reward_similarity_tol)
    ]

    num_ordered = int(len(records))
    num_unordered = int(len(unordered_counts))
    dup_ratio = 0.0 if num_ordered == 0 else 1.0 - (float(num_unordered) / float(num_ordered))

    return {
        "dataset": dataset_name,
        "split": split,
        "sample_id": sample_id,
        "graph_id": int(graph_id),
        "num_ordered": num_ordered,
        "num_unique_ordered": int(len(ordered_counts)),
        "num_unordered": num_unordered,
        "dup_ratio": dup_ratio,
        "correlation_duplicate_factor_sample_frequency": pearson(duplicate_factors, sample_frequencies),
        "correlation_duplicate_factor_reward": pearson(duplicate_factors, reward_means),
        "correlation_sample_frequency_reward": pearson(sample_frequencies, reward_means),
        "max_duplicate_factor": max((int(row["duplicate_factor"]) for row in unordered_rows), default=0),
        "max_sample_frequency": max((int(row["sample_frequency"]) for row in unordered_rows), default=0),
        "multi_prefix_unordered_sets": len(multi_prefix),
        "similar_reward_multi_prefix_sets": len(similar_multi_prefix),
        "similar_reward_multi_prefix_rate": safe_div(len(similar_multi_prefix), len(multi_prefix)),
        "mean_reward_std_multi_prefix": mean([float(row["reward_std"]) for row in multi_prefix]),
        "reward_by_ordered_prefix": [
            {
                "ordered_key": list(key),
                "sample_frequency": int(ordered_counts[key]),
                "reward_mean": mean(values),
                "reward_std": std(values),
            }
            for key, values in sorted(ordered_rewards.items(), key=lambda item: (-ordered_counts[item[0]], item[0]))
        ],
        "reward_by_unordered_set": sorted(
            unordered_rows,
            key=lambda row: (-int(row["sample_frequency"]), -int(row["duplicate_factor"]), row["unordered_key"]),
        ),
    }


class RunningSummary:
    def __init__(self) -> None:
        self.question_count = 0
        self.total_ordered = 0
        self.total_unordered = 0
        self.dup_ratios: list[float] = []
        self.corr_dup_freq: list[float] = []
        self.corr_dup_reward: list[float] = []
        self.max_duplicate_factor = 0
        self.max_sample_frequency = 0
        self.multi_prefix_sets = 0
        self.similar_multi_prefix_sets = 0
        self.pooled_duplicate_factor: list[float] = []
        self.pooled_sample_frequency: list[float] = []
        self.pooled_reward: list[float] = []

    def add(self, result: dict[str, Any]) -> None:
        self.question_count += 1
        self.total_ordered += int(result["num_ordered"])
        self.total_unordered += int(result["num_unordered"])
        self.dup_ratios.append(float(result["dup_ratio"]))
        add_finite(self.corr_dup_freq, result["correlation_duplicate_factor_sample_frequency"])
        add_finite(self.corr_dup_reward, result["correlation_duplicate_factor_reward"])
        self.max_duplicate_factor = max(self.max_duplicate_factor, int(result["max_duplicate_factor"]))
        self.max_sample_frequency = max(self.max_sample_frequency, int(result["max_sample_frequency"]))
        self.multi_prefix_sets += int(result["multi_prefix_unordered_sets"])
        self.similar_multi_prefix_sets += int(result["similar_reward_multi_prefix_sets"])
        for row in result["reward_by_unordered_set"]:
            self.pooled_duplicate_factor.append(float(row["duplicate_factor"]))
            self.pooled_sample_frequency.append(float(row["sample_frequency"]))
            self.pooled_reward.append(float(row["reward_mean"]))

    def finalize(
        self,
        *,
        dataset_name: str,
        split: str,
        rollouts: int,
        checkpoint: str,
        epoch: Any,
        global_step: Any,
        missing: int,
        unexpected: int,
        details: str | None,
    ) -> dict[str, Any]:
        weighted_dup_ratio = 0.0 if self.total_ordered == 0 else 1.0 - float(self.total_unordered) / float(self.total_ordered)
        return {
            "dataset": dataset_name,
            "split": split,
            "rollouts_per_question": int(rollouts),
            "questions": int(self.question_count),
            "checkpoint": checkpoint,
            "checkpoint_epoch": epoch,
            "checkpoint_global_step": global_step,
            "checkpoint_missing_keys": int(missing),
            "checkpoint_unexpected_keys": int(unexpected),
            "details_jsonl": details,
            "mean_dup_ratio": mean(self.dup_ratios),
            "weighted_dup_ratio": weighted_dup_ratio,
            "mean_question_corr_duplicate_factor_sample_frequency": mean(self.corr_dup_freq),
            "mean_question_corr_duplicate_factor_reward": mean(self.corr_dup_reward),
            "pooled_corr_duplicate_factor_sample_frequency": pearson(self.pooled_duplicate_factor, self.pooled_sample_frequency),
            "pooled_corr_duplicate_factor_reward": pearson(self.pooled_duplicate_factor, self.pooled_reward),
            "max_duplicate_factor": int(self.max_duplicate_factor),
            "max_sample_frequency": int(self.max_sample_frequency),
            "multi_prefix_unordered_sets": int(self.multi_prefix_sets),
            "similar_reward_multi_prefix_sets": int(self.similar_multi_prefix_sets),
            "similar_reward_multi_prefix_rate": safe_div(self.similar_multi_prefix_sets, self.multi_prefix_sets),
            "verdict": verdict(weighted_dup_ratio, pearson(self.pooled_duplicate_factor, self.pooled_sample_frequency)),
        }


def verdict(dup_ratio: float, corr_dup_freq: float | None) -> str:
    if dup_ratio < 0.1:
        return "ordered-prefix likely harmless by dup_ratio<0.1"
    if dup_ratio > 0.3 and corr_dup_freq is not None and corr_dup_freq >= 0.5:
        return "ordering artifact likely: dup_ratio>0.3 and duplicate_factor/sample_frequency correlation is high"
    return "inconclusive"


def sample_id_for(batch: Any, graph_id: int) -> str:
    sample_ids = getattr(batch, "sample_id", None)
    if isinstance(sample_ids, (list, tuple)) and graph_id < len(sample_ids):
        return str(sample_ids[graph_id])
    return str(graph_id)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_vals = [x for x, _ in pairs]
    y_vals = [y for _, y in pairs]
    x_mean = mean(x_vals)
    y_mean = mean(y_vals)
    x_centered = [x - x_mean for x in x_vals]
    y_centered = [y - y_mean for y in y_vals]
    x_var = sum(x * x for x in x_centered)
    y_var = sum(y * y for y in y_centered)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    return float(sum(x * y for x, y in zip(x_centered, y_centered)) / math.sqrt(x_var * y_var))


def mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    return float(sum(finite) / len(finite))


def std(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return 0.0
    mu = mean(finite)
    return float(math.sqrt(sum((value - mu) ** 2 for value in finite) / len(finite)))


def safe_div(numerator: int | float, denominator: int | float) -> float:
    denominator = float(denominator)
    if denominator == 0.0:
        return 0.0
    return float(numerator) / denominator


def add_finite(values: list[float], value: Any) -> None:
    if value is None:
        return
    value = float(value)
    if math.isfinite(value):
        values.append(value)


if __name__ == "__main__":
    main()
