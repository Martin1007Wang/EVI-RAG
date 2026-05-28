from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.segments import segment_logsumexp
from src.eval.rollout import (
    evaluate_rollout_samples,
    rollout_eval_tensors,
    terminal_state_from_trajectories,
)
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.replay import WeakTransitionSource, initial_replay_state_batch
from src.weaver.rollout.trajectory import BUDGET, NO_FRONTIER, POLICY_STOP
from src.weaver.state import ExpansionBatch, StateBatch


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_dir: Path
    source_ckpt: Path
    frozen_ckpt: Path
    out_dir: Path


class ScalarStats:
    def __init__(self) -> None:
        self.values: list[float] = []
        self.weights: list[float] = []

    def add(self, value: float | int | torch.Tensor | None, *, weight: float = 1.0) -> None:
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            value = float(value.detach().cpu().item())
        value = float(value)
        if math.isfinite(value):
            self.values.append(value)
            self.weights.append(float(weight))

    def mean(self) -> float:
        if not self.values:
            return math.nan
        denom = sum(self.weights)
        if denom <= 0.0:
            return math.nan
        return float(sum(value * weight for value, weight in zip(self.values, self.weights, strict=True)) / denom)

    def total(self) -> float:
        return float(sum(value * weight for value, weight in zip(self.values, self.weights, strict=True)))

    def count(self) -> int:
        return len(self.values)

    def weight_sum(self) -> float:
        return float(sum(self.weights))

    def p50(self) -> float:
        return quantile(self.values, 0.50)

    def p90(self) -> float:
        return quantile(self.values, 0.90)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = resolve_out_dir(args.out_dir, run_dir=run_dir)
    paths = freeze_checkpoint(
        run_dir=run_dir,
        ckpt_arg=args.ckpt,
        out_dir=out_dir,
    )

    cfg = load_run_config(run_dir=run_dir, data_dir=args.data_dir)
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
    missing, unexpected = load_checkpoint_weights(
        model,
        str(paths.frozen_ckpt),
        strict=False,
    )
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    oracle_by_row = load_oracle_curve(
        args.oracle_curve,
        split=args.split,
        budget=int(args.oracle_budget),
    )

    all_metrics: dict[str, ScalarStats] = defaultdict(ScalarStats)
    edge_stats: dict[tuple[str, str], ScalarStats] = defaultdict(ScalarStats)
    stop_stats: dict[tuple[str, int, str], ScalarStats] = defaultdict(ScalarStats)
    per_sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, limit, int(args.batch_size)):
            end = min(limit, start + int(args.batch_size))
            samples = [dataset[idx] for idx in range(start, end)]
            batch = datamodule.collator(samples).to(device)
            graph = GraphContext.from_batch(batch)
            target = TargetContext.from_batch(batch=batch, graph_context=graph)
            features = model.feature_encoder(batch)

            trajectories = model.runner.eval_rollouts(
                policy=model.policy,
                context=graph,
                features=features,
                budget=int(model.budget),
                num_rollouts=int(args.num_rollouts),
            )
            metrics = evaluate_rollout_samples(
                trajectories=trajectories,
                batch=batch,
                context=graph,
                exclude_anchors_from_retrieved=model.evaluation.exclude_anchors_from_retrieved,
                use_reachable_targets=model.evaluation.use_reachable_targets,
                k_windows=model.evaluation.k_windows,
                enable_terminal_diagnostics=True,
            )
            for name, value in metrics.items():
                all_metrics[name].add(value, weight=float(batch.num_graphs_total))

            tensors = rollout_eval_tensors(
                trajectories=trajectories,
                batch=batch,
                context=graph,
                exclude_anchors_from_retrieved=model.evaluation.exclude_anchors_from_retrieved,
                use_reachable_targets=model.evaluation.use_reachable_targets,
            )
            add_per_sample_rows(
                rows=per_sample_rows,
                batch=batch,
                tensors=tensors,
                global_start=start,
                oracle_by_row=oracle_by_row,
            )

            initial = StateBatch.initial(
                graph_ids=torch.arange(int(graph.num_graphs), dtype=torch.long, device=device),
                budget=int(model.budget),
            )
            collect_edge_score_stats(
                model=model,
                features=features,
                graph=graph,
                target=target,
                state=initial,
                label="initial",
                stats=edge_stats,
            )
            collect_stop_stats(
                model=model,
                features=features,
                graph=graph,
                target=target,
                state=initial,
                label="initial",
                stats=stop_stats,
            )

            weak = WeakTransitionSource(
                max_depth=int(model.budget),
                max_states_per_graph=int(args.states_per_graph),
                max_positive_edges_per_state=int(args.branch_per_state),
            ).collect(
                graph_context=graph,
                target_context=target,
                initial_state=initial_replay_state_batch(
                    graph_context=graph,
                    target_context=target,
                    budget=int(model.budget),
                ),
            )
            if weak.nonterminal is not None and int(weak.nonterminal.parent_state.num_states) > 0:
                collect_edge_score_stats(
                    model=model,
                    features=features,
                    graph=graph,
                    target=target,
                    state=weak.nonterminal.parent_state,
                    label="oracle_prefix",
                    stats=edge_stats,
                )
                collect_stop_stats(
                    model=model,
                    features=features,
                    graph=graph,
                    target=target,
                    state=weak.nonterminal.parent_state,
                    label="oracle_prefix",
                    stats=stop_stats,
                )

            policy_terminal = terminal_state_from_trajectories(trajectories)
            collect_stop_stats(
                model=model,
                features=features,
                graph=graph,
                target=target,
                state=policy_terminal,
                label="policy_terminal",
                stats=stop_stats,
            )

            parent_prefix = sampled_parent_prefixes(trajectories)
            if parent_prefix.num_states > 0:
                collect_edge_score_stats(
                    model=model,
                    features=features,
                    graph=graph,
                    target=target,
                    state=parent_prefix,
                    label="policy_prefix",
                    stats=edge_stats,
                )
                collect_stop_stats(
                    model=model,
                    features=features,
                    graph=graph,
                    target=target,
                    state=parent_prefix,
                    label="policy_prefix",
                    stats=stop_stats,
                )

    manifest = build_manifest(
        args=args,
        paths=paths,
        missing=missing,
        unexpected=unexpected,
        sample_count=limit,
    )
    write_json(paths.out_dir / "manifest.json", manifest)
    write_metric_summary(paths.out_dir / "rollout_summary.csv", all_metrics)
    write_edge_score_stats(paths.out_dir / "edge_score_stats.csv", edge_stats)
    write_stop_gate_stats(paths.out_dir / "stop_gate_by_depth.csv", stop_stats)
    write_csv(paths.out_dir / "per_sample_gap.csv", per_sample_rows)
    write_report(
        paths.out_dir / "report.md",
        manifest=manifest,
        metrics=all_metrics,
        edge_stats=edge_stats,
        stop_stats=stop_stats,
    )
    print(f"diagnostic_out={paths.out_dir}")
    print(f"frozen_ckpt={paths.frozen_ckpt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose rollout recall gap, edge scoring, and stop-gate behavior.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint to freeze. Defaults to monitor best from run/checkpoints/last.ckpt.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--states-per-graph", type=int, default=8)
    parser.add_argument("--branch-per-state", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--oracle-curve", type=Path, default=ROOT / "outputs" / "analysis" / "budget_recall_oracle" / "webqsp" / "per_sample_budget_curve.csv")
    parser.add_argument("--oracle-budget", type=int, default=8)
    return parser.parse_args()


def resolve_out_dir(out_dir: Path | None, *, run_dir: Path) -> Path:
    if out_dir is not None:
        return out_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (ROOT / "outputs" / "diagnostics" / "webqsp_valfit_gap" / stamp).resolve()


def freeze_checkpoint(*, run_dir: Path, ckpt_arg: Path | None, out_dir: Path) -> RunPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ckpt_arg.resolve() if ckpt_arg is not None else best_checkpoint_from_last(run_dir / "checkpoints" / "last.ckpt")
    frozen = out_dir / "current_best.ckpt"
    if source.resolve() != frozen.resolve():
        shutil.copy2(source, frozen)
    shutil.copy2(run_dir / ".hydra" / "config.yaml", out_dir / "config.yaml")
    val_metrics = run_dir / "artifacts" / "metrics" / "val.jsonl"
    if val_metrics.exists():
        shutil.copy2(val_metrics, out_dir / "val.jsonl")
    train_metrics = run_dir / "artifacts" / "metrics" / "train.jsonl"
    if train_metrics.exists():
        shutil.copy2(train_metrics, out_dir / "train.jsonl")
    return RunPaths(run_dir=run_dir, source_ckpt=source, frozen_ckpt=frozen, out_dir=out_dir)


def best_checkpoint_from_last(last_ckpt: Path) -> Path:
    payload = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    callbacks = payload.get("callbacks", {})
    for state in callbacks.values():
        if isinstance(state, dict) and state.get("best_model_path"):
            return Path(str(state["best_model_path"])).resolve()
    raise RuntimeError(f"Could not find best_model_path in {last_ckpt}.")


def load_run_config(*, run_dir: Path, data_dir: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = str(data_dir)
        cfg.logger = None
        cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.enable_checkpointing = False
        cfg.model.evaluation.enable_terminal_diagnostics = True
    return cfg


def sampled_parent_prefixes(trajectories) -> StateBatch:
    from src.weaver.state import cat_state_batches

    device = trajectories.device
    budget = int(trajectories.budget)
    state = StateBatch.initial(graph_ids=trajectories.graph_ids, budget=budget)
    parents: list[StateBatch] = []
    for step in range(budget):
        rows = trajectories.edge_count.gt(step).nonzero(as_tuple=False).flatten()
        if int(rows.numel()) == 0:
            continue
        parents.append(state.take(rows))
        edge_ids = trajectories.edge_ids.index_select(0, rows)[:, step]
        state = state.advance(
            ExpansionBatch(
                state_ids=rows,
                edge_ids=edge_ids,
            )
        )
    if not parents:
        return StateBatch.initial(graph_ids=torch.empty(0, dtype=torch.long, device=device), budget=budget)

    return cat_state_batches(parents)


def collect_edge_score_stats(
    *,
    model,
    features,
    graph: GraphContext,
    target: TargetContext,
    state: StateBatch,
    label: str,
    stats: dict[tuple[str, str], ScalarStats],
) -> None:
    action_space = state.action_space(graph)
    if state.num_states == 0:
        return
    policy_out = model.policy(features=features, state=state, context=graph, action_space=action_space)
    semantic, residual = score_components(model=model, features=features, graph=graph, state=state, action_space=action_space)
    positive_all = target.shortest_path_edge_mask.index_select(0, action_space.expand_edge_ids).detach().cpu() if action_space.num_expansions else torch.empty(0, dtype=torch.bool)
    raw = policy_out.edge_raw_score.detach().cpu()
    edge_prob = policy_out.edge_log_prob.detach().cpu().exp()
    cond_prob = policy_out.conditional_edge_log_prob.detach().cpu().exp()
    semantic = semantic.detach().cpu()
    residual = residual.detach().cpu()
    stop_prob = policy_out.stop_log_prob.detach().cpu().exp()
    frontier = action_space.expand_count.detach().cpu()
    ptr = action_space.expand_ptr.detach().cpu()

    for row in range(int(state.num_states)):
        add(stats, label, "state_count", 1.0)
        add(stats, label, "frontier", int(frontier[row].item()))
        add(stats, label, "stop_prob", float(stop_prob[row].item()))
        start = int(ptr[row].item())
        end = int(ptr[row + 1].item())
        if end <= start:
            continue
        positive = positive_all[start:end]
        add(stats, label, "frontier_positive_count", int(positive.sum().item()))
        row_raw = raw[start:end]
        row_semantic = semantic[start:end]
        row_residual = residual[start:end]
        add(stats, label, "raw_std", float(row_raw.std(unbiased=False).item()))
        add(stats, label, "semantic_std", float(row_semantic.std(unbiased=False).item()))
        add(stats, label, "residual_std", float(row_residual.std(unbiased=False).item()))
        add(stats, label, "conditional_entropy", entropy(cond_prob[start:end]))
        if not bool(positive.any()):
            continue
        add(stats, label, "positive_state_count", 1.0)
        add(stats, label, "positive_action_mass", float(edge_prob[start:end][positive].sum().item()))
        add(stats, label, "positive_conditional_mass", float(cond_prob[start:end][positive].sum().item()))
        for name, scores in (("raw", row_raw), ("semantic", row_semantic), ("residual", row_residual)):
            rank, top1 = best_positive_rank(scores=scores, positive=positive)
            add(stats, label, f"{name}_rank", rank)
            add(stats, label, f"{name}_top1", 1.0 if top1 else 0.0)
            negative = ~positive
            if bool(negative.any()):
                gap = float(scores[positive].mean().item() - scores[negative].mean().item())
                add(stats, label, f"{name}_gap", gap)


def collect_stop_stats(
    *,
    model,
    features,
    graph: GraphContext,
    target: TargetContext,
    state: StateBatch,
    label: str,
    stats: dict[tuple[str, int, str], ScalarStats],
) -> None:
    if state.num_states == 0:
        return
    action_space = state.action_space(graph)
    policy_out = model.policy(features=features, state=state, context=graph, action_space=action_space)
    reward = model.reward_model(state=state, graph_context=graph, target_context=target)
    reward_dominance = prefix_reward_stop_dominance(
        model=model,
        graph=graph,
        target=target,
        state=state,
        action_space=action_space,
        reward=reward,
    ).detach().cpu()
    stop_prob = policy_out.stop_log_prob.detach().cpu().exp()
    continue_prob = policy_out.continue_log_prob.detach().cpu().exp()
    margin = (policy_out.stop_log_flow - policy_out.continue_log_flow).detach().cpu()
    frontier = action_space.expand_count.detach().cpu()
    depth = state.edge_count.detach().cpu()
    recall = reward.target_recall.detach().cpu()
    hit = reward.answer_count.detach().cpu().gt(0)

    for row in range(int(state.num_states)):
        d = int(depth[row].item())
        add3(stats, label, d, "state_count", 1.0)
        add3(stats, label, d, "frontier", int(frontier[row].item()))
        add3(stats, label, d, "stop_prob", float(stop_prob[row].item()))
        add3(stats, label, d, "continue_prob", float(continue_prob[row].item()))
        if math.isfinite(float(margin[row].item())):
            add3(stats, label, d, "stop_continue_log_flow_margin", float(margin[row].item()))
        add3(stats, label, d, "target_recall", float(recall[row].item()))
        add3(stats, label, d, "hit_rate", 1.0 if bool(hit[row].item()) else 0.0)
        add3(stats, label, d, "full_cover_rate", 1.0 if float(recall[row].item()) >= 1.0 - 1e-8 else 0.0)
        if math.isfinite(float(reward_dominance[row].item())):
            add3(stats, label, d, "reward_stop_dominance", float(reward_dominance[row].item()))


def prefix_reward_stop_dominance(
    *,
    model,
    graph: GraphContext,
    target: TargetContext,
    state: StateBatch,
    action_space,
    reward,
) -> torch.Tensor:
    if action_space.num_expansions <= 0:
        return reward.log_reward.new_full((state.num_states,), float("-inf"))

    child = state.branch(
        ExpansionBatch(
            state_ids=action_space.expand_state_ids,
            edge_ids=action_space.expand_edge_ids,
        )
    )
    child_reward = model.reward_model(state=child, graph_context=graph, target_context=target)
    child_log_reward = segment_logsumexp(
        values=child_reward.log_reward.float(),
        segment_ids=action_space.expand_state_ids,
        num_segments=int(state.num_states),
    )
    return reward.log_reward.float() - child_log_reward


def score_components(*, model, features, graph: GraphContext, state: StateBatch, action_space) -> tuple[torch.Tensor, torch.Tensor]:
    if action_space.num_expansions <= 0:
        empty = state.graph_ids.new_empty((0,), dtype=torch.float32)
        return empty, empty
    state_repr = model.policy.encode_state(
        features=features,
        state=state,
        context=graph,
    )
    frontier = model.policy.encode_frontier(
        features=features,
        context=graph,
        action_space=action_space,
    )
    semantic = model.policy.semantic_prior(
        state_repr=state_repr,
        frontier=frontier,
    )
    residual = model.policy.edge_residual_scorer(
        state_repr=state_repr,
        frontier=frontier,
    )
    return semantic.float(), residual.float()


def add(stats: dict[tuple[str, str], ScalarStats], label: str, name: str, value: float | int) -> None:
    stats[(label, name)].add(value)


def add3(stats: dict[tuple[str, int, str], ScalarStats], label: str, depth: int, name: str, value: float | int) -> None:
    stats[(label, int(depth), name)].add(value)


def best_positive_rank(*, scores: torch.Tensor, positive: torch.Tensor) -> tuple[int, bool]:
    best_positive = scores[positive].max()
    rank = int(scores.gt(best_positive).sum().item()) + 1
    top1 = bool(positive[int(torch.argmax(scores).item())].item())
    return rank, top1


def entropy(prob: torch.Tensor) -> float:
    prob = prob.float()
    prob = prob[prob.gt(0)]
    if int(prob.numel()) == 0:
        return 0.0
    return float((-prob * prob.log()).sum().item())


def add_per_sample_rows(*, rows: list[dict[str, Any]], batch, tensors, global_start: int, oracle_by_row: dict[int, dict[str, Any]]) -> None:
    valid = tensors.valid_graph_mask.detach().cpu()
    recall = tensors.recall.detach().cpu()
    edge_count = tensors.edge_count.detach().cpu()
    union8 = tensors.node_masks[:8].any(dim=0, keepdim=True)
    from src.eval.rollout import retrieval_from_masks

    _, union_recall, _, _ = retrieval_from_masks(
        node_masks=union8,
        batch=batch.cpu(),
        exclude_anchors_from_retrieved=True,
        use_reachable_targets=True,
    )
    target_counts = torch.bincount(
        batch.cpu().batch.index_select(0, batch.cpu().reachable_target_node_ids.to(dtype=torch.long)),
        minlength=int(batch.num_graphs_total),
    )
    for graph_id in range(int(batch.num_graphs_total)):
        if not bool(valid[graph_id].item()):
            continue
        oracle = oracle_by_row.get(int(global_start + graph_id), {})
        oracle_recall = oracle.get("oracle_recall")
        gap = None if oracle_recall is None else float(oracle_recall) - float(union_recall[0, graph_id].item())
        rows.append(
            {
                "sample_index": int(global_start + graph_id),
                "sample_id": oracle.get("sample_id"),
                "single_rollout_recall_mean": float(recall[:, graph_id].mean().item()),
                "single_rollout_edge_count_mean": float(edge_count[:, graph_id].float().mean().item()),
                "union8_recall": float(union_recall[0, graph_id].item()),
                "oracle8_recall": oracle_recall,
                "oracle8_used_edges": oracle.get("oracle_used_edges"),
                "oracle8_minus_union8": gap,
                "target_count": int(target_counts[graph_id].item()),
            }
        )


def load_oracle_curve(path: Path, *, split: str, budget: int) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("split") not in {split, "validation" if split == "val" else split}:
                continue
            if int(row.get("budget", -1)) != int(budget):
                continue
            row_idx = int(row["row_idx"])
            out[row_idx] = {
                "sample_id": row.get("sample_id"),
                "oracle_recall": float(row["recall"]),
                "oracle_used_edges": int(row["used_edges"]),
            }
    return out


def build_manifest(*, args: argparse.Namespace, paths: RunPaths, missing: list[str], unexpected: list[str], sample_count: int) -> dict[str, Any]:
    return {
        "run_dir": str(paths.run_dir),
        "source_ckpt": str(paths.source_ckpt),
        "frozen_ckpt": str(paths.frozen_ckpt),
        "frozen_ckpt_sha256": sha256_file(paths.frozen_ckpt),
        "sample_count": int(sample_count),
        "args": vars(args),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "git_status": command_text(["git", "status", "--short"]),
    }


def write_metric_summary(path: Path, metrics: dict[str, ScalarStats]) -> None:
    rows = [
        {
            "metric": name,
            "count": stat.count(),
            "weight_sum": stat.weight_sum(),
            "mean": stat.mean(),
            "p50": stat.p50(),
            "p90": stat.p90(),
            "sum": stat.total(),
        }
        for name, stat in sorted(metrics.items())
    ]
    write_csv(path, rows)


def write_edge_score_stats(path: Path, stats: dict[tuple[str, str], ScalarStats]) -> None:
    labels = sorted({label for label, _ in stats})
    names = sorted({name for _, name in stats})
    rows = []
    for label in labels:
        row: dict[str, Any] = {"prefix": label}
        for name in names:
            stat = stats.get((label, name))
            row[name] = None if stat is None else stat.mean()
            row[f"{name}_n"] = 0 if stat is None else stat.count()
        rows.append(row)
    write_csv(path, rows)


def write_stop_gate_stats(path: Path, stats: dict[tuple[str, int, str], ScalarStats]) -> None:
    labels_depths = sorted({(label, depth) for label, depth, _ in stats})
    names = sorted({name for _, _, name in stats})
    rows = []
    for label, depth in labels_depths:
        row: dict[str, Any] = {"prefix": label, "depth": depth}
        for name in names:
            stat = stats.get((label, depth, name))
            row[name] = None if stat is None else stat.mean()
            row[f"{name}_n"] = 0 if stat is None else stat.count()
        rows.append(row)
    write_csv(path, rows)


def write_report(path: Path, *, manifest: dict[str, Any], metrics: dict[str, ScalarStats], edge_stats: dict[tuple[str, str], ScalarStats], stop_stats: dict[tuple[str, int, str], ScalarStats]) -> None:
    def m(name: str) -> float:
        return metrics[name].mean() if name in metrics else math.nan

    def e(label: str, name: str) -> float:
        stat = edge_stats.get((label, name))
        return math.nan if stat is None else stat.mean()

    def s(label: str, depth: int, name: str) -> float:
        stat = stop_stats.get((label, depth, name))
        return math.nan if stat is None else stat.mean()

    text = f"""# Rollout Gap Diagnostic

## Run
- run_dir: `{manifest['run_dir']}`
- frozen_ckpt: `{manifest['frozen_ckpt']}`
- source_ckpt: `{manifest['source_ckpt']}`
- samples: {manifest['sample_count']}
- oracle curve: `{manifest['args'].get('oracle_curve')}`

## Rollout Summary
- union@8 recall: {m('rollout_union@8/recall'):.6f}
- union@8 edges: {m('rollout_union@8/edges'):.6f}
- union@8 redundancy: {m('rollout_union@8/redundancy'):.6f}
- single-rollout recall: {m('single_rollout/mean_recall'):.6f}
- edge budget full rate: {m('rollout/edge_budget_full_rate'):.6f}
- policy stop rate: {m('terminal/policy_stop_rate'):.6f}
- budget boundary rate: {m('terminal/budget_boundary_rate'):.6f}
- hit-then-continue rate: {m('terminal/hit_then_continue_rate'):.6f}

## Edge Scoring
- initial raw top1: {e('initial', 'raw_top1'):.6f}; raw gap: {e('initial', 'raw_gap'):.6f}; positive mass: {e('initial', 'positive_action_mass'):.6f}
- oracle-prefix raw top1: {e('oracle_prefix', 'raw_top1'):.6f}; raw gap: {e('oracle_prefix', 'raw_gap'):.6f}; positive mass: {e('oracle_prefix', 'positive_action_mass'):.6f}
- policy-prefix raw top1: {e('policy_prefix', 'raw_top1'):.6f}; raw gap: {e('policy_prefix', 'raw_gap'):.6f}; positive mass: {e('policy_prefix', 'positive_action_mass'):.6f}

## Stop Gate By Representative Depth
- initial depth 0 stop_prob: {s('initial', 0, 'stop_prob'):.6f}
- policy prefix depth 1 stop_prob: {s('policy_prefix', 1, 'stop_prob'):.6f}
- policy prefix depth 4 stop_prob: {s('policy_prefix', 4, 'stop_prob'):.6f}
- policy prefix depth 7 stop_prob: {s('policy_prefix', 7, 'stop_prob'):.6f}
- policy terminal depth 8 stop_prob: {s('policy_terminal', 8, 'stop_prob'):.6f}

## Initial Interpretation
- If `edge_budget_full_rate` is near 1 and `policy_stop_rate` is near 0, rollout termination is dominated by budget exhaustion rather than learned early STOP.
- If positive action mass/top1 is low on oracle prefixes, the policy is not ranking shortest-path frontier edges strongly enough even on supervised prefixes.
- If stop probability stays low after hit/full-cover states, the stop gate is not acting as an evidence-sufficiency decision.
"""
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(tensor.quantile(float(q)).item())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"<failed: {exc}>"


if __name__ == "__main__":
    main()
