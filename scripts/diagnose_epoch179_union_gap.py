from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf, open_dict
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.aggregation import grouped_sample_ids
from src.eval.rollout import rollout_eval_tensors
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.feature import FeaturePack
from src.weaver.context import GraphContext
from src.weaver.rollout.trajectory import BUDGET, NO_FRONTIER, POLICY_STOP, TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch


@dataclass(frozen=True, slots=True)
class RunInputs:
    run_dir: Path
    ckpt_path: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose union-recall gap for the epoch 179 checkpoint with read-only rollout audits.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/debug_valfit/2026-05-29/11-38-27"),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("outputs/debug_valfit/2026-05-29/11-38-27/checkpoints/epoch_epoch=179-step_step=0010620.ckpt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/diagnostics/epoch179_union_gap"),
    )
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument(
        "--oracle-curve",
        type=Path,
        default=Path("outputs/analysis/budget_recall_oracle/webqsp/per_sample_budget_curve.csv"),
    )
    parser.add_argument("--oracle-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = RunInputs(
        run_dir=args.run_dir.resolve(),
        ckpt_path=args.ckpt.resolve(),
        out_dir=args.out_dir.resolve(),
    )
    inputs.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_run_config(run_dir=inputs.run_dir, data_dir=args.data_dir)
    datamodule, resources = prepare_training_components(cfg, stage="fit")
    dataset = {
        "train": datamodule.train_dataset,
        "validation": datamodule.val_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }[args.split]
    if dataset is None:
        raise RuntimeError(f"split {args.split!r} is not initialized")

    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(inputs.ckpt_path), strict=False)
    device = torch.device(args.device)
    cpu_feature_encoder = copy.deepcopy(model.feature_encoder).to(torch.device("cpu"))
    if device.type == "cuda":
        model.policy = model.policy.to(device)
    else:
        model = model.to(device)
    model.eval()

    oracle_budget_recall = load_oracle_budget_curve(args.oracle_curve, budget=int(args.oracle_budget))

    manifest = {
        "run_dir": str(inputs.run_dir),
        "checkpoint": str(inputs.ckpt_path),
        "split": args.split,
        "batch_size": int(args.batch_size),
        "num_rollouts": int(args.num_rollouts),
        "oracle_budget": int(args.oracle_budget),
        "device": str(device),
        "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected,
        "evaluation": {
            "exclude_anchors_from_retrieved": bool(cfg.model.evaluation.exclude_anchors_from_retrieved),
            "use_reachable_targets": bool(cfg.model.evaluation.use_reachable_targets),
            "budget": int(cfg.model.budget),
        },
    }

    state_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []

    policy_expected_recall: list[float] = []
    policy_best_recall: list[float] = []
    greedy_policy_recall: list[float] = []
    greedy_oracle_recall: list[float] = []
    oracle_budget8_recall: list[float] = []

    depth_buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    replay_buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    logit_buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    limit = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    start_time = time.time()

    with torch.no_grad():
        for start in range(0, limit, int(args.batch_size)):
            end = min(limit, start + int(args.batch_size))
            samples = [dataset[idx] for idx in range(start, end)]
            cpu_batch = datamodule.collator(samples)
            batch = datamodule.collator(samples).to(device)
            graph = GraphContext.from_batch(batch)
            features = feature_pack_to_device(cpu_feature_encoder(cpu_batch), device=device)
            eval_views = build_eval_views(batch=batch, device=device)
            trajectories = model.runner.eval_rollouts(
                policy=model.policy,
                context=graph,
                features=features,
                budget=int(model.budget),
                num_rollouts=int(args.num_rollouts),
            )

            eval_tensors = rollout_eval_tensors(
                trajectories=trajectories,
                batch=batch,
                context=graph,
                exclude_anchors_from_retrieved=bool(cfg.model.evaluation.exclude_anchors_from_retrieved),
                use_reachable_targets=bool(cfg.model.evaluation.use_reachable_targets),
            )
            sample_ids = grouped_sample_ids(trajectories, num_graphs=int(graph.num_graphs)).cpu()
            recall_matrix = eval_tensors.recall.cpu()
            best_recall_by_graph = recall_matrix.max(dim=0).values
            expected_recall_by_graph = recall_matrix.mean(dim=0)

            for local_graph_id in range(int(graph.num_graphs)):
                global_idx = start + local_graph_id
                sample_id = str(batch.sample_id[local_graph_id])
                oracle_val = float(
                    oracle_budget_recall.get((args.split, global_idx), math.nan)
                )
                oracle_budget8_recall.append(oracle_val)
                expected_val = float(expected_recall_by_graph[local_graph_id].item())
                best_val = float(best_recall_by_graph[local_graph_id].item())
                policy_expected_recall.append(expected_val)
                policy_best_recall.append(best_val)

                greedy_policy_state = run_greedy_policy_single(
                    model=model,
                    features=features,
                    graph=graph,
                    graph_id=local_graph_id,
                    budget=int(model.budget),
                )
                greedy_policy_val = state_recall(
                    state=greedy_policy_state,
                    eval_views=eval_views,
                    graph_id=local_graph_id,
                )
                greedy_policy_recall.append(greedy_policy_val)

                greedy_oracle_state = run_greedy_oracle_single(
                    graph=graph,
                    eval_views=eval_views,
                    graph_id=local_graph_id,
                    budget=int(model.budget),
                )
                greedy_oracle_val = state_recall(
                    state=greedy_oracle_state,
                    eval_views=eval_views,
                    graph_id=local_graph_id,
                )
                greedy_oracle_recall.append(greedy_oracle_val)

                per_sample_rows.append(
                    {
                        "row_idx": global_idx,
                        "sample_id": sample_id,
                        "policy_expected_recall": expected_val,
                        "policy_best_of_8_recall": best_val,
                        "greedy_policy_recall": greedy_policy_val,
                        "greedy_oracle_recall": greedy_oracle_val,
                        "oracle_budget8_recall": oracle_val,
                        "sampling_gap": best_val - expected_val,
                        "ranking_gap": greedy_oracle_val - greedy_policy_val,
                        "support_proxy_gap": oracle_val - best_val if math.isfinite(oracle_val) else math.nan,
                        "total_gap": oracle_val - expected_val if math.isfinite(oracle_val) else math.nan,
                    }
                )

            frontier_rows_batch, state_rows_batch = collect_policy_frontier_diagnostics(
                model=model,
                features=features,
                graph=graph,
                batch=batch,
                eval_views=eval_views,
                trajectories=trajectories,
                sample_start=start,
            )
            frontier_rows.extend(frontier_rows_batch)
            state_rows.extend(state_rows_batch)
            elapsed = time.time() - start_time
            print(
                f"progress samples={end}/{limit} "
                f"state_rows={len(state_rows)} frontier_rows={len(frontier_rows)} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    aggregate_depth_buckets(
        state_rows=state_rows,
        frontier_rows=frontier_rows,
        depth_buckets=depth_buckets,
        replay_buckets=replay_buckets,
        logit_buckets=logit_buckets,
    )

    summary = {
        "sample_count": int(limit),
        "policy_expected_recall_mean": safe_mean(policy_expected_recall),
        "policy_best_of_8_recall_mean": safe_mean(policy_best_recall),
        "greedy_policy_recall_mean": safe_mean(greedy_policy_recall),
        "greedy_oracle_recall_mean": safe_mean(greedy_oracle_recall),
        "oracle_budget8_recall_mean": safe_mean(oracle_budget8_recall),
        "gap_sampling": safe_mean(policy_best_recall) - safe_mean(policy_expected_recall),
        "gap_ranking": safe_mean(greedy_oracle_recall) - safe_mean(greedy_policy_recall),
        "gap_support_proxy": safe_mean(oracle_budget8_recall) - safe_mean(policy_best_recall),
        "gap_total": safe_mean(oracle_budget8_recall) - safe_mean(policy_expected_recall),
    }

    write_json(inputs.out_dir / "manifest.json", manifest)
    write_json(inputs.out_dir / "rollout_summary.json", summary)
    write_csv(inputs.out_dir / "state_summary.csv", state_rows)
    write_csv(inputs.out_dir / "frontier_edges.csv", frontier_rows)
    write_csv(inputs.out_dir / "per_sample_gap.csv", per_sample_rows)
    write_csv(inputs.out_dir / "depth_buckets.csv", flatten_bucket_map(depth_buckets, "depth"))
    write_csv(inputs.out_dir / "replay_audit.csv", flatten_bucket_map(replay_buckets, "depth"))
    write_csv(inputs.out_dir / "logit_margin.csv", flatten_bucket_map(logit_buckets, "depth"))
    write_report(inputs.out_dir / "report.md", summary=summary, manifest=manifest)

    print(f"diagnostic_out={inputs.out_dir}")


def load_run_config(*, run_dir: Path, data_dir: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = str(data_dir)
        cfg.logger = None
        cfg.trainer.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.enable_checkpointing = False
    return cfg


def load_oracle_budget_curve(path: Path, *, budget: int) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["budget"]) != int(budget):
                continue
            out[(str(row["split"]), int(row["row_idx"]))] = float(row["recall"])
    return out


def feature_pack_to_device(features: FeaturePack, *, device: torch.device) -> FeaturePack:
    return FeaturePack(
        query_sem_h=features.query_sem_h.to(device),
        node_sem_h=features.node_sem_h.to(device),
        rel_sem_h=features.rel_sem_h.to(device),
        query_h=features.query_h.to(device),
        node_h=features.node_h.to(device),
        node_has_text=features.node_has_text.to(device),
        node_graph_ids=features.node_graph_ids.to(device),
        anchor_node_ids=features.anchor_node_ids.to(device),
        anchor_graph_ids=features.anchor_graph_ids.to(device),
        edge_h=features.edge_h.to(device),
        edge_src=features.edge_src.to(device),
        edge_dst=features.edge_dst.to(device),
        edge_graph_ids=features.edge_graph_ids.to(device),
        device=device,
    )


@dataclass(frozen=True, slots=True)
class EvalViews:
    node_batch: torch.Tensor
    target_mask: torch.Tensor
    anchor_mask: torch.Tensor
    gold_count: torch.Tensor
    valid_graph_mask: torch.Tensor
    num_graphs: int


def build_eval_views(*, batch, device: torch.device) -> EvalViews:
    node_batch = batch.batch.to(device=device, dtype=torch.long)
    target_ids = batch.reachable_target_node_ids.to(device=device, dtype=torch.long)
    target_mask = torch.zeros(int(batch.num_nodes_total), dtype=torch.bool, device=device)
    if int(target_ids.numel()) > 0:
        target_mask[target_ids] = True
    anchor_ids = batch.anchor_node_ids.to(device=device, dtype=torch.long)
    anchor_mask = torch.zeros(int(batch.num_nodes_total), dtype=torch.bool, device=device)
    if int(anchor_ids.numel()) > 0:
        anchor_mask[anchor_ids] = True
    gold_count = torch.bincount(
        node_batch[target_mask],
        minlength=int(batch.num_graphs_total),
    ).to(dtype=torch.float32)
    valid_graph_mask = gold_count.gt(0.0)
    return EvalViews(
        node_batch=node_batch,
        target_mask=target_mask,
        anchor_mask=anchor_mask,
        gold_count=gold_count,
        valid_graph_mask=valid_graph_mask,
        num_graphs=int(batch.num_graphs_total),
    )


def batch_state_recall(*, states: StateBatch, eval_views: EvalViews) -> torch.Tensor:
    node_masks = states.node_mask
    retrieved = node_masks & ~eval_views.anchor_mask.view(1, -1)
    hit = retrieved & eval_views.target_mask.view(1, -1)
    hits = torch.matmul(hit.float(), F.one_hot(eval_views.node_batch, num_classes=eval_views.num_graphs).float())
    recall = torch.where(
        eval_views.gold_count.view(1, -1).gt(0.0),
        hits / eval_views.gold_count.view(1, -1).clamp_min(1e-8),
        torch.zeros_like(hits),
    )
    return recall


def collect_policy_frontier_diagnostics(
    *,
    model,
    features,
    graph: GraphContext,
    batch,
    eval_views: EvalViews,
    trajectories: TrajectoryBatch,
    sample_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frontier_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    sample_ids = grouped_sample_ids(trajectories, num_graphs=int(graph.num_graphs)).cpu()

    for traj_row in range(int(trajectories.num_trajectories)):
        graph_id = int(trajectories.graph_ids[traj_row].item())
        sample_id = str(batch.sample_id[graph_id])
        sample_group = int(sample_ids[traj_row].item())
        global_row_idx = sample_start + graph_id
        state = StateBatch.initial(
            graph_ids=torch.tensor([graph_id], dtype=torch.long, device=graph.device),
            budget=int(trajectories.budget),
            graph_context=graph,
        )
        edge_count = int(trajectories.edge_count[traj_row].item())

        for depth in range(edge_count + 1):
            out = model.policy(state=state, features=features)
            recall_before = state_recall(state=state, eval_views=eval_views, graph_id=graph_id)
            stop_prob = float((out.stop_log_flow - out.state_log_flow)[0].exp().item())
            continue_prob = float((out.continue_log_flow - out.state_log_flow)[0].exp().item()) if math.isfinite(float(out.continue_log_flow[0].item())) else 0.0

            chosen_edge = None
            stop_taken = depth == edge_count
            stop_reason = int(trajectories.stop_reason[traj_row].item()) if stop_taken else None
            if depth < edge_count:
                chosen_edge = int(trajectories.edge_ids[traj_row, depth].item())

            row_frontier_positions = out.frontier.row_ids.eq(0).nonzero(as_tuple=False).flatten()
            gain_exists = False
            choose_gain = False
            gain_mass = 0.0
            gain_count = 0
            replay_gain_count = 0
            replay_gain_zero_count = 0
            best_gain_flow = float("-inf")
            best_zero_flow = float("-inf")
            best_gain_rank = math.nan
            chosen_rank = math.nan
            child_recall_by_edge: dict[int, float] = {}

            row_edge_ids = out.frontier.edge_ids.index_select(0, row_frontier_positions) if int(row_frontier_positions.numel()) > 0 else torch.empty(0, dtype=torch.long, device=graph.device)
            if int(row_edge_ids.numel()) > 0:
                child_states = state.branch(
                    ExpansionBatch(
                        state_ids=torch.zeros(int(row_edge_ids.numel()), dtype=torch.long, device=graph.device),
                        edge_ids=row_edge_ids,
                    ),
                    graph_context=graph,
                )
                child_recall = batch_state_recall(states=child_states, eval_views=eval_views)[:, graph_id]
                for edge_id, recall_after in zip(row_edge_ids.tolist(), child_recall.tolist(), strict=True):
                    child_recall_by_edge[int(edge_id)] = float(recall_after)

            for rank_idx, pos in enumerate(
                torch.argsort(
                    out.edge_log_flow.index_select(0, row_frontier_positions),
                    descending=True,
                ).tolist(),
                start=1,
            ):
                absolute_pos = int(row_frontier_positions[pos].item())
                edge_id = int(out.frontier.edge_ids[absolute_pos].item())
                recall_after = child_recall_by_edge[edge_id]
                delta_recall = recall_after - recall_before
                edge_prob = float((out.edge_log_flow[absolute_pos] - out.state_log_flow[0]).exp().item())
                is_replay = bool(batch.weak_replay_edge_ids_batch.eq(graph_id).logical_and(batch.weak_replay_edge_ids.eq(edge_id)).any().item())
                replay_weight = replay_weight_for_edge(batch=batch, graph_id=graph_id, edge_id=edge_id)
                is_gain = delta_recall > 1e-8
                gain_exists = gain_exists or is_gain
                if is_gain:
                    gain_mass += edge_prob
                    gain_count += 1
                    best_gain_flow = max(best_gain_flow, float(out.edge_log_flow[absolute_pos].item()))
                    if math.isnan(best_gain_rank):
                        best_gain_rank = float(rank_idx)
                    if is_replay:
                        replay_gain_count += 1
                else:
                    best_zero_flow = max(best_zero_flow, float(out.edge_log_flow[absolute_pos].item()))
                    if is_replay and abs(delta_recall) <= 1e-8:
                        replay_gain_zero_count += 1

                if chosen_edge is not None and edge_id == chosen_edge:
                    choose_gain = is_gain
                    chosen_rank = float(rank_idx)

                frontier_rows.append(
                    {
                        "row_idx": global_row_idx,
                        "sample_id": sample_id,
                        "graph_id": graph_id,
                        "trajectory_id": traj_row,
                        "sample_group": sample_group,
                        "depth": depth,
                        "selected_edge_count": int(state.edge_count[0].item()),
                        "frontier_size": int(row_frontier_positions.numel()),
                        "edge_id": edge_id,
                        "edge_log_flow": float(out.edge_log_flow[absolute_pos].item()),
                        "edge_prob": edge_prob,
                        "stop_log_flow": float(out.stop_log_flow[0].item()),
                        "continue_log_flow": float(out.continue_log_flow[0].item()),
                        "state_log_flow": float(out.state_log_flow[0].item()),
                        "chosen_edge_id": chosen_edge,
                        "recall_before": recall_before,
                        "recall_after": recall_after,
                        "delta_recall": delta_recall,
                        "is_replay_edge": int(is_replay),
                        "replay_weight": replay_weight,
                        "is_gain": int(is_gain),
                        "is_zero_gain": int(abs(delta_recall) <= 1e-8),
                    }
                )

            state_rows.append(
                {
                    "row_idx": global_row_idx,
                    "sample_id": sample_id,
                    "graph_id": graph_id,
                    "trajectory_id": traj_row,
                    "sample_group": sample_group,
                    "depth": depth,
                    "selected_edge_count": int(state.edge_count[0].item()),
                    "frontier_size": int(row_frontier_positions.numel()),
                    "chosen_edge_id": chosen_edge,
                    "stop_taken": int(stop_taken),
                    "stop_reason": stop_reason_name(stop_reason) if stop_reason is not None else "",
                    "stop_prob": stop_prob,
                    "continue_prob": continue_prob,
                    "stop_log_flow": float(out.stop_log_flow[0].item()),
                    "continue_log_flow": float(out.continue_log_flow[0].item()),
                    "state_log_flow": float(out.state_log_flow[0].item()),
                    "recall_before": recall_before,
                    "gain_exists": int(gain_exists),
                    "choose_gain": int(choose_gain),
                    "mass_on_gain": gain_mass,
                    "gain_count": gain_count,
                    "best_gain_flow": best_gain_flow if gain_exists else math.nan,
                    "best_zero_flow": best_zero_flow if best_zero_flow > float("-inf") else math.nan,
                    "gain_margin": (
                        best_gain_flow - best_zero_flow
                        if gain_exists and best_zero_flow > float("-inf")
                        else math.nan
                    ),
                    "best_gain_rank": best_gain_rank,
                    "chosen_rank": chosen_rank,
                    "replay_gain_count": replay_gain_count,
                    "replay_zero_gain_count": replay_gain_zero_count,
                }
            )

            if depth < edge_count:
                next_edge = int(trajectories.edge_ids[traj_row, depth].item())
                state = state.advance(
                    ExpansionBatch(
                        state_ids=torch.tensor([0], dtype=torch.long, device=graph.device),
                        edge_ids=torch.tensor([next_edge], dtype=torch.long, device=graph.device),
                    ),
                    graph_context=graph,
                )

    return frontier_rows, state_rows


def run_greedy_policy_single(*, model, features, graph: GraphContext, graph_id: int, budget: int) -> StateBatch:
    state = StateBatch.initial(
        graph_ids=torch.tensor([graph_id], dtype=torch.long, device=graph.device),
        budget=budget,
        graph_context=graph,
    )
    for _ in range(budget):
        out = model.policy(state=state, features=features)
        row_pos = out.frontier.row_ids.eq(0).nonzero(as_tuple=False).flatten()
        best_edge_flow = float("-inf")
        best_edge_id = None
        for pos in row_pos.tolist():
            flow = float(out.edge_log_flow[pos].item())
            if flow > best_edge_flow:
                best_edge_flow = flow
                best_edge_id = int(out.frontier.edge_ids[pos].item())
        if best_edge_id is None or float(out.stop_log_flow[0].item()) >= best_edge_flow:
            break
        state = state.advance(
            ExpansionBatch(
                state_ids=torch.tensor([0], dtype=torch.long, device=graph.device),
                edge_ids=torch.tensor([best_edge_id], dtype=torch.long, device=graph.device),
            ),
            graph_context=graph,
        )
    return state


def run_greedy_oracle_single(
    *,
    graph: GraphContext,
    eval_views: EvalViews,
    graph_id: int,
    budget: int,
) -> StateBatch:
    state = StateBatch.initial(
        graph_ids=torch.tensor([graph_id], dtype=torch.long, device=graph.device),
        budget=budget,
        graph_context=graph,
    )
    for _ in range(budget):
        frontier = state.frontier(
            edge_src=graph.edge_src,
            edge_dst=graph.edge_dst,
            remaining_budget=state.budget_left,
        )
        row_pos = frontier.row_ids.eq(0).nonzero(as_tuple=False).flatten()
        if int(row_pos.numel()) == 0:
            break
        recall_before = state_recall(state=state, eval_views=eval_views, graph_id=graph_id)
        best_delta = 0.0
        best_edge_id = None
        row_edge_ids = frontier.edge_ids.index_select(0, row_pos)
        child_states = state.branch(
            ExpansionBatch(
                state_ids=torch.zeros(int(row_edge_ids.numel()), dtype=torch.long, device=graph.device),
                edge_ids=row_edge_ids,
            ),
            graph_context=graph,
        )
        child_recall = batch_state_recall(states=child_states, eval_views=eval_views)[:, graph_id]
        for edge_id, recall_after in zip(row_edge_ids.tolist(), child_recall.tolist(), strict=True):
            delta = float(recall_after) - recall_before
            if delta > best_delta + 1e-8:
                best_delta = delta
                best_edge_id = int(edge_id)
        if best_edge_id is None:
            break
        state = state.advance(
            ExpansionBatch(
                state_ids=torch.tensor([0], dtype=torch.long, device=graph.device),
                edge_ids=torch.tensor([best_edge_id], dtype=torch.long, device=graph.device),
            ),
            graph_context=graph,
        )
    return state


def state_recall(
    *,
    state: StateBatch,
    eval_views: EvalViews,
    graph_id: int,
) -> float:
    recall = batch_state_recall(states=state, eval_views=eval_views)
    if not bool(eval_views.valid_graph_mask[graph_id].item()):
        return 0.0
    return float(recall[0, graph_id].item())


def replay_weight_for_edge(*, batch, graph_id: int, edge_id: int) -> float:
    mask = batch.weak_replay_edge_ids_batch.eq(graph_id) & batch.weak_replay_edge_ids.eq(edge_id)
    if not bool(mask.any().item()):
        return 0.0
    return float(batch.weak_replay_edge_weight[mask][0].item())


def aggregate_depth_buckets(
    *,
    state_rows: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    depth_buckets: dict[int, dict[str, list[float]]],
    replay_buckets: dict[int, dict[str, list[float]]],
    logit_buckets: dict[int, dict[str, list[float]]],
) -> None:
    for row in state_rows:
        depth = int(row["depth"])
        depth_buckets[depth]["state_count"].append(1.0)
        depth_buckets[depth]["gain_exists_rate"].append(float(row["gain_exists"]))
        depth_buckets[depth]["choose_gain_rate"].append(float(row["choose_gain"]))
        depth_buckets[depth]["mass_on_gain_mean"].append(float(row["mass_on_gain"]))
        depth_buckets[depth]["frontier_size_mean"].append(float(row["frontier_size"]))
        depth_buckets[depth]["stop_prob_mean"].append(float(row["stop_prob"]))
        if not math.isnan(float(row["gain_margin"])):
            logit_buckets[depth]["gain_margin_mean"].append(float(row["gain_margin"]))
        if not math.isnan(float(row["best_gain_rank"])):
            logit_buckets[depth]["best_gain_rank_mean"].append(float(row["best_gain_rank"]))
        if not math.isnan(float(row["chosen_rank"])):
            logit_buckets[depth]["chosen_rank_mean"].append(float(row["chosen_rank"]))

    for row in frontier_rows:
        depth = int(row["depth"])
        if int(row["is_replay_edge"]) != 1:
            continue
        replay_buckets[depth]["replay_edge_count"].append(1.0)
        replay_buckets[depth]["p_delta_positive"].append(float(row["is_gain"]))
        replay_buckets[depth]["p_delta_zero"].append(float(row["is_zero_gain"]))
        replay_buckets[depth]["delta_recall_mean"].append(float(row["delta_recall"]))

    positives_by_state: dict[tuple[int, int, int], tuple[int, int]] = {}
    for row in frontier_rows:
        key = (int(row["trajectory_id"]), int(row["sample_group"]), int(row["depth"]))
        total, covered = positives_by_state.get(key, (0, 0))
        if int(row["is_gain"]) == 1:
            total += 1
            if int(row["is_replay_edge"]) == 1:
                covered += 1
        positives_by_state[key] = (total, covered)
    for (_, _, depth), (total, covered) in positives_by_state.items():
        if total <= 0:
            continue
        replay_buckets[depth]["gain_state_count"].append(1.0)
        replay_buckets[depth]["oracle_positive_edges"].append(float(total))
        replay_buckets[depth]["replay_covered_positive_edges"].append(float(covered))
        replay_buckets[depth]["coverage_mean"].append(float(covered) / float(total))


def flatten_bucket_map(bucket_map: dict[int, dict[str, list[float]]], depth_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for depth in sorted(bucket_map):
        row: dict[str, Any] = {depth_key: depth}
        for metric, values in sorted(bucket_map[depth].items()):
            row[metric] = safe_mean(values)
        rows.append(row)
    return rows


def stop_reason_name(code: int | None) -> str:
    if code is None:
        return ""
    if int(code) == POLICY_STOP:
        return "policy_stop"
    if int(code) == NO_FRONTIER:
        return "no_frontier"
    if int(code) == BUDGET:
        return "budget"
    return f"unknown_{code}"


def safe_mean(values: list[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return math.nan
    return float(sum(finite) / len(finite))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_report(path: Path, *, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    lines = [
        "# Epoch 179 Union-Gap Report",
        "",
        f"- Checkpoint: `{manifest['checkpoint']}`",
        f"- Split: `{manifest['split']}`",
        f"- Samples: `{summary['sample_count']}`",
        "",
        "## Summary",
        "",
        f"- Policy expected recall: `{summary['policy_expected_recall_mean']:.6f}`",
        f"- Policy best-of-8 recall: `{summary['policy_best_of_8_recall_mean']:.6f}`",
        f"- Greedy policy recall: `{summary['greedy_policy_recall_mean']:.6f}`",
        f"- Greedy oracle recall: `{summary['greedy_oracle_recall_mean']:.6f}`",
        f"- Oracle budget-8 recall: `{summary['oracle_budget8_recall_mean']:.6f}`",
        f"- Sampling gap: `{summary['gap_sampling']:.6f}`",
        f"- Ranking gap: `{summary['gap_ranking']:.6f}`",
        f"- Support proxy gap: `{summary['gap_support_proxy']:.6f}`",
        f"- Total gap: `{summary['gap_total']:.6f}`",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
