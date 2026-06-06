from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.rollout import evaluate_rollout_samples, rollout_eval_tensors
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.context import GraphContext, TargetContext
from src.weaver.objectives.subtb.batch import prepare_subtb_batch
from src.weaver.objectives.subtb.scoring import score_subtb_batch
from src.weaver.policy import STOP_EDGE_ID
from src.weaver.rollout.trajectory import BUDGET_TRUNCATED, POLICY_STOP, TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch

DEFAULT_RUN_DIR = Path("outputs/valfit/2026-06-06/10-10-40")
DEFAULT_CKPT = DEFAULT_RUN_DIR / "checkpoints" / "last_copy.ckpt"
DEFAULT_OUTPUT = Path("outputs/diagnostics/weaver_checkpoint/2026-06-06_10-10-40_last_copy")
DEFAULT_METADATA_DIR = Path("/mnt/data/retrieval/webqsp/metadata")
UNREACHABLE_DISTANCE = 1_000_000_000


@dataclass(frozen=True, slots=True)
class RankStats:
    hit1: int = 0
    hit3: int = 0
    hit_any: int = 0
    count: int = 0
    reciprocal_rank_sum: float = 0.0
    rank_sum: float = 0.0
    stop_before_gold: int = 0


@dataclass(frozen=True, slots=True)
class StopResidualStats:
    count: int
    mean: float
    abs_mean: float
    min: float
    max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Weaver checkpoint rollout, STOP calibration, and first-action ranking.",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_RUN_DIR / ".hydra" / "config.yaml")
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means full split.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config_path)
    cfg.dataset.paths.metadata_dir = str(args.metadata_dir)
    cfg.datamodule.splits.validation = str(args.split)
    cfg.datamodule.eval_num_workers = 0
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = 2
    cfg.model.budget = int(args.budget)
    cfg.model.runner.eval_rollouts = int(args.num_rollouts)
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    cfg.trainer.precision = "32-true"
    cfg.logger = None
    cfg.callbacks = None

    device = resolve_device(str(args.device))
    datamodule, resources = prepare_training_components(cfg, stage="validate")
    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(args.ckpt_path), strict=False)
    model.to(device)
    model.eval()

    loader = datamodule.val_dataloader()
    accum = DiagnosticAccumulator()
    per_sample_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if int(args.max_batches) > 0 and batch_idx >= int(args.max_batches):
                break
            batch = batch.to(device)
            result = diagnose_batch(
                model=model,
                batch=batch,
                budget=int(args.budget),
                num_rollouts=int(args.num_rollouts),
            )
            accum.add(result.summary)
            per_sample_rows.extend(result.per_sample_rows)
            prefix_rows.extend(result.prefix_rows)
            print(f"diagnosed batch {batch_idx + 1}")

    summary = accum.finalize()
    summary["checkpoint"] = str(args.ckpt_path)
    summary["config_path"] = str(args.config_path)
    summary["split"] = str(args.split)
    summary["budget"] = int(args.budget)
    summary["num_rollouts"] = int(args.num_rollouts)
    summary["load_missing_keys"] = missing
    summary["load_unexpected_keys"] = unexpected

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_sample.csv", per_sample_rows)
    write_csv(output_dir / "prefix_steps.csv", prefix_rows)
    print(f"wrote {output_dir / 'summary.json'}")
    print(f"wrote {output_dir / 'per_sample.csv'}")
    print(f"wrote {output_dir / 'prefix_steps.csv'}")


@dataclass(frozen=True, slots=True)
class BatchDiagnostic:
    summary: dict[str, float]
    per_sample_rows: list[dict[str, Any]]
    prefix_rows: list[dict[str, Any]]


def diagnose_batch(
    *,
    model,
    batch,
    budget: int,
    num_rollouts: int,
) -> BatchDiagnostic:
    contexts = model._build_step_contexts(batch)
    inputs = model._build_policy_inputs_from_graph(batch=batch, graph=contexts.graph)
    trajectories = model.runner.eval_rollouts(
        policy=model.policy,
        context=contexts.graph,
        features=inputs.features,
        policy_input=inputs.policy_input,
        budget=int(budget),
        num_rollouts=int(num_rollouts),
    )
    rollout_metrics = evaluate_rollout_samples(
        trajectories=trajectories,
        batch=batch,
        context=contexts.graph,
        exclude_anchors_from_retrieved=bool(model.evaluation.exclude_anchors_from_retrieved),
        use_reachable_targets=bool(model.evaluation.use_reachable_targets),
        k_windows=tuple(model.evaluation.k_windows),
        enable_terminal_diagnostics=True,
    )
    tensors = rollout_eval_tensors(
        trajectories=trajectories,
        batch=batch,
        context=contexts.graph,
        exclude_anchors_from_retrieved=bool(model.evaluation.exclude_anchors_from_retrieved),
        use_reachable_targets=bool(model.evaluation.use_reachable_targets),
    )

    oracle = oracle_truncate_after_hit(
        trajectories=trajectories,
        batch=batch,
        context=contexts.graph,
        budget=int(budget),
    )
    oracle_tensors = rollout_eval_tensors(
        trajectories=oracle,
        batch=batch,
        context=contexts.graph,
        exclude_anchors_from_retrieved=bool(model.evaluation.exclude_anchors_from_retrieved),
        use_reachable_targets=bool(model.evaluation.use_reachable_targets),
    )
    first_rank, potential_rank = first_action_diagnostics(
        model=model,
        graph=contexts.graph,
        target=contexts.target,
        features=inputs.features,
        policy_input=inputs.policy_input,
        budget=int(budget),
    )
    stop_residuals, prefix_rows = stop_prefix_diagnostics(
        model=model,
        graph=contexts.graph,
        target=contexts.target,
        features=inputs.features,
        policy_input=inputs.policy_input,
        trajectories=trajectories,
        batch=batch,
    )

    summary: dict[str, float] = {f"rollout/{k}": float(v) for k, v in rollout_metrics.items()}
    summary.update(prefix_float("first_action", summarize_rank_records(first_rank)))
    summary.update(prefix_float("potential", summarize_rank_records(potential_rank)))
    summary.update(prefix_float("stop_residual", stop_residuals_to_dict(stop_residuals)))
    summary["oracle_stop/mean_edge_count"] = masked_mean(oracle_tensors.edge_count, oracle_tensors.valid_graph_mask)
    summary["oracle_stop/mean_recall"] = masked_mean(oracle_tensors.recall, oracle_tensors.valid_graph_mask)
    summary["oracle_stop/edge_count_delta"] = masked_mean(tensors.edge_count - oracle_tensors.edge_count, tensors.valid_graph_mask)
    summary["oracle_stop/recall_delta"] = masked_mean(oracle_tensors.recall - tensors.recall, tensors.valid_graph_mask)

    per_sample_rows = build_per_sample_rows(
        batch=batch,
        trajectories=trajectories,
        natural_recall=tensors.recall,
        oracle_recall=oracle_tensors.recall,
        natural_edge_count=tensors.edge_count,
        oracle_edge_count=oracle_tensors.edge_count,
        valid=tensors.valid_graph_mask,
    )
    return BatchDiagnostic(summary=summary, per_sample_rows=per_sample_rows, prefix_rows=prefix_rows)


def first_action_diagnostics(
    *,
    model,
    graph: GraphContext,
    target: TargetContext,
    features,
    policy_input,
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = StateBatch.initial(
        graph_ids=torch.arange(graph.num_graphs, device=graph.device, dtype=torch.long),
        budget=int(budget),
        graph_context=graph,
    )
    action_space = model.policy.prepare_action_space(state=state, graph_context=graph)
    output = model.policy(
        state=state,
        features=features,
        graph_context=graph,
        policy_input=policy_input,
        action_space=action_space,
        compute_log_flow=False,
    )
    reward = model.reward_model(
        state=state,
        target_context=target,
        graph_context=graph,
        active=action_space.active,
    )

    first_records: list[dict[str, Any]] = []
    potential_records: list[dict[str, Any]] = []
    edge_log_prob = {
        int(edge): float(logp)
        for edge, logp in zip(output.action_edge_ids.detach().cpu().tolist(), output.action_log_prob.detach().cpu().tolist(), strict=False)
    }

    for graph_id in range(int(graph.num_graphs)):
        gold_edges = weak_gold_frontier_edges(
            graph=graph,
            target=target,
            graph_id=graph_id,
            row_id=graph_id,
            frontier_edge_ids=action_space.frontier.edge_ids,
            frontier_row_ids=action_space.frontier.row_ids,
        )
        if not gold_edges:
            continue
        row_positions = output.action_row_ids.eq(graph_id)
        row_edge_ids = output.action_edge_ids[row_positions]
        row_logp = output.action_log_prob[row_positions]
        rank = rank_of_any_gold(
            edge_ids=row_edge_ids.detach().cpu().tolist(),
            scores=row_logp.detach().cpu().tolist(),
            gold_edges=gold_edges,
        )
        stop_rank = rank_of_edge(
            edge_ids=row_edge_ids.detach().cpu().tolist(),
            scores=row_logp.detach().cpu().tolist(),
            edge_id=STOP_EDGE_ID,
        )
        first_records.append(
            rank_record(
                graph_id=graph_id,
                rank=rank,
                stop_rank=stop_rank,
                gold_count=len(gold_edges),
                group=hop_group_for_graph(target=target, graph=graph, graph_id=graph_id),
            )
        )

        edge_rows = action_space.frontier.row_ids.eq(graph_id)
        branch_edges = action_space.frontier.edge_ids[edge_rows]
        if int(branch_edges.numel()) == 0:
            continue
        branch_state = state.branch(
            ExpansionBatch(
                state_ids=torch.full(
                    (int(branch_edges.numel()),),
                    int(graph_id),
                    dtype=torch.long,
                    device=graph.device,
                ),
                edge_ids=branch_edges,
            ),
            graph_context=graph,
        )
        branch_reward = model.reward_model(
            state=branch_state,
            target_context=target,
            graph_context=graph,
        )
        potential_rank = rank_of_any_gold(
            edge_ids=branch_edges.detach().cpu().tolist(),
            scores=branch_reward.state_potential.detach().cpu().tolist(),
            gold_edges=gold_edges,
        )
        potential_records.append(
            rank_record(
                graph_id=graph_id,
                rank=potential_rank,
                stop_rank=None,
                gold_count=len(gold_edges),
                group=hop_group_for_graph(target=target, graph=graph, graph_id=graph_id),
                extra={"policy_best_gold_logp": max(edge_log_prob.get(edge, -math.inf) for edge in gold_edges)},
            )
        )
    return first_records, potential_records


def weak_gold_frontier_edges(
    *,
    graph: GraphContext,
    target: TargetContext,
    graph_id: int,
    row_id: int,
    frontier_edge_ids: torch.Tensor,
    frontier_row_ids: torch.Tensor,
) -> set[int]:
    edge_ids = frontier_edge_ids[frontier_row_ids.eq(int(row_id))]
    if int(edge_ids.numel()) == 0:
        return set()
    graph_edge_ids = graph.edge_to_graph.index_select(0, edge_ids).eq(int(graph_id))
    gold = target.edge_on_shortest_path.index_select(0, edge_ids) & graph_edge_ids
    return set(edge_ids[gold].detach().cpu().tolist())


def stop_prefix_diagnostics(
    *,
    model,
    graph: GraphContext,
    target: TargetContext,
    features,
    policy_input,
    trajectories: TrajectoryBatch,
    batch,
) -> tuple[StopResidualStats, list[dict[str, Any]]]:
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph)
    action_space = model.policy.prepare_action_space(state=prepared.states, graph_context=graph)
    reward = model.reward_model(
        state=prepared.states,
        target_context=target,
        graph_context=graph,
        active=action_space.active,
    )
    scores = score_subtb_batch(
        batch=prepared,
        policy=model.policy,
        features=features,
        policy_input=policy_input,
        graph_context=graph,
        reward=reward,
        action_space=action_space,
    )
    valid = reward.terminal_valid_mask.detach()
    residual = scores.log_flow[valid] + scores.stop_log_prob_by_state[valid] - reward.log_reward.detach().float()[valid]
    stats = stop_residual_stats(residual)

    prefix_rows: list[dict[str, Any]] = []
    if int(prepared.states.num_states) == 0:
        return stats, prefix_rows
    prefix_state_to_traj: dict[int, tuple[int, int]] = {}
    for traj_id in range(int(prepared.prefix_state_ids.size(0))):
        for step in range(int(prepared.prefix_state_ids.size(1))):
            state_id = int(prepared.prefix_state_ids[traj_id, step].item())
            if state_id >= 0:
                prefix_state_to_traj[state_id] = (traj_id, step)
    sample_ids = sample_id_by_trajectory(trajectories, num_graphs=int(batch.num_graphs))
    stop_prob = scores.stop_log_prob_by_state.exp().detach().cpu()
    for state_id, (traj_id, step) in prefix_state_to_traj.items():
        graph_id = int(trajectories.graph_ids[traj_id].detach().cpu().item())
        prefix_rows.append(
            {
                "sample_id": sample_id_at(batch, graph_id),
                "sample_index": int(sample_ids[traj_id]),
                "graph_id": graph_id,
                "trajectory_id": traj_id,
                "step": step,
                "edge_count": int(prepared.states.edge_count[state_id].detach().cpu().item()),
                "stop_prob": float(stop_prob[state_id].item()),
                "log_reward": float(reward.log_reward[state_id].detach().cpu().item()),
                "log_flow": float(scores.log_flow[state_id].detach().cpu().item()),
                "stop_residual": (
                    float(
                        (
                            scores.log_flow[state_id]
                            + scores.stop_log_prob_by_state[state_id]
                            - reward.log_reward.detach().float()[state_id]
                        )
                        .detach()
                        .cpu()
                        .item()
                    )
                    if bool(valid[state_id])
                    else math.nan
                ),
                "answer_count": float(reward.answer_count[state_id].detach().cpu().item()),
                "target_count": float(reward.target_count[state_id].detach().cpu().item()),
                "terminal_valid": bool(valid[state_id].detach().cpu().item()),
            }
        )
    return stats, prefix_rows


def oracle_truncate_after_hit(
    *,
    trajectories: TrajectoryBatch,
    batch,
    context: GraphContext,
    budget: int,
) -> TrajectoryBatch:
    edge_index = batch.edge_index.detach().cpu()
    node_batch = batch.batch.detach().cpu()
    target_nodes = torch.zeros(int(batch.num_nodes_total), dtype=torch.bool)
    anchors = torch.zeros(int(batch.num_nodes_total), dtype=torch.bool)
    target_nodes[batch.reachable_target_node_ids.detach().cpu()] = True
    anchors[batch.anchor_node_ids.detach().cpu()] = True
    target_nodes &= ~anchors

    new_edge_ids = trajectories.edge_ids.detach().cpu().clone()
    new_edge_logp = trajectories.edge_logp.detach().cpu().clone()
    new_edge_count = trajectories.edge_count.detach().cpu().clone()
    new_stop_reason = trajectories.stop_reason.detach().cpu().clone()
    graph_ids = trajectories.graph_ids.detach().cpu()
    for row in range(int(trajectories.num_trajectories)):
        graph_id = int(graph_ids[row].item())
        active = anchors & node_batch.eq(graph_id)
        if bool((active & target_nodes).any()):
            new_edge_ids[row, :] = -1
            new_edge_logp[row, :] = 0.0
            new_edge_count[row] = 0
            new_stop_reason[row] = int(POLICY_STOP)
            continue
        for step in range(int(trajectories.edge_count[row].item())):
            edge_id = int(trajectories.edge_ids[row, step].detach().cpu().item())
            active[edge_index[0, edge_id]] = True
            active[edge_index[1, edge_id]] = True
            if bool((active & target_nodes).any()):
                keep = step + 1
                new_edge_ids[row, keep:] = -1
                new_edge_logp[row, keep:] = 0.0
                new_edge_count[row] = keep
                new_stop_reason[row] = int(POLICY_STOP)
                break
    return TrajectoryBatch(
        graph_ids=graph_ids.to(device=context.device),
        edge_ids=new_edge_ids.to(device=context.device),
        edge_logp=new_edge_logp.to(device=context.device),
        edge_count=new_edge_count.to(device=context.device),
        stop_reason=new_stop_reason.to(device=context.device),
        stop_logp=trajectories.stop_logp.detach().cpu().to(device=context.device),
        source=trajectories.source.detach().cpu().to(device=context.device),
    )


def rank_of_any_gold(*, edge_ids: list[int], scores: list[float], gold_edges: set[int]) -> int | None:
    ranked = sorted(zip(edge_ids, scores, strict=False), key=lambda item: item[1], reverse=True)
    for pos, (edge_id, _) in enumerate(ranked, start=1):
        if int(edge_id) in gold_edges:
            return pos
    return None


def rank_of_edge(*, edge_ids: list[int], scores: list[float], edge_id: int) -> int | None:
    ranked = sorted(zip(edge_ids, scores, strict=False), key=lambda item: item[1], reverse=True)
    for pos, (candidate, _) in enumerate(ranked, start=1):
        if int(candidate) == int(edge_id):
            return pos
    return None


def rank_record(
    *,
    graph_id: int,
    rank: int | None,
    stop_rank: int | None,
    gold_count: int,
    group: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = {
        "graph_id": int(graph_id),
        "rank": int(rank) if rank is not None else math.nan,
        "hit1": bool(rank == 1),
        "hit3": bool(rank is not None and rank <= 3),
        "hit_any": bool(rank is not None),
        "reciprocal_rank": 1.0 / float(rank) if rank is not None else 0.0,
        "stop_rank": int(stop_rank) if stop_rank is not None else math.nan,
        "stop_before_gold": bool(stop_rank is not None and rank is not None and stop_rank < rank),
        "gold_count": int(gold_count),
        "group": group,
    }
    if extra:
        out.update(extra)
    return out


def summarize_rank_records(records: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for group_name, group_records in grouped_records(records):
        count = len(group_records)
        prefix = "all" if group_name == "" else group_name
        summary[f"{prefix}/count"] = float(count)
        if count == 0:
            continue
        ranks = [float(row["rank"]) for row in group_records if not math.isnan(float(row["rank"]))]
        summary[f"{prefix}/hits@1"] = mean_bool(row["hit1"] for row in group_records)
        summary[f"{prefix}/hits@3"] = mean_bool(row["hit3"] for row in group_records)
        summary[f"{prefix}/hit_any"] = mean_bool(row["hit_any"] for row in group_records)
        summary[f"{prefix}/mrr"] = mean_float(float(row["reciprocal_rank"]) for row in group_records)
        summary[f"{prefix}/mean_rank"] = mean_float(ranks) if ranks else math.nan
        summary[f"{prefix}/stop_before_gold"] = mean_bool(row["stop_before_gold"] for row in group_records)
    return summary


def grouped_records(records: list[dict[str, Any]]) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    yield "", records
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_group[str(row.get("group", "unknown"))].append(row)
    for group, rows in sorted(by_group.items()):
        yield group, rows


def hop_group_for_graph(*, target: TargetContext, graph: GraphContext, graph_id: int) -> str:
    start = int(graph.anchor_ptr[graph_id].item())
    end = int(graph.anchor_ptr[graph_id + 1].item())
    anchors = graph.anchor_node_ids[start:end]
    if int(anchors.numel()) == 0:
        return "no_anchor"
    distances = target.node_target_distance.index_select(0, anchors)
    reachable = distances[distances.lt(UNREACHABLE_DISTANCE)]
    if int(reachable.numel()) == 0:
        return "unreachable"
    min_dist = int(reachable.min().item())
    if min_dist <= 1:
        return "one_hop"
    if min_dist == 2:
        return "two_hop"
    return "multi_hop"


def stop_residual_stats(residual: torch.Tensor) -> StopResidualStats:
    residual = residual.detach().cpu().float()
    if int(residual.numel()) == 0:
        return StopResidualStats(count=0, mean=0.0, abs_mean=0.0, min=math.nan, max=math.nan)
    return StopResidualStats(
        count=int(residual.numel()),
        mean=float(residual.mean().item()),
        abs_mean=float(residual.abs().mean().item()),
        min=float(residual.min().item()),
        max=float(residual.max().item()),
    )


def stop_residuals_to_dict(stats: StopResidualStats) -> dict[str, float]:
    return {
        "count": float(stats.count),
        "mean": stats.mean,
        "abs_mean": stats.abs_mean,
        "min": stats.min,
        "max": stats.max,
    }


def build_per_sample_rows(
    *,
    batch,
    trajectories: TrajectoryBatch,
    natural_recall: torch.Tensor,
    oracle_recall: torch.Tensor,
    natural_edge_count: torch.Tensor,
    oracle_edge_count: torch.Tensor,
    valid: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sample_ids = sample_id_by_trajectory(trajectories, num_graphs=int(batch.num_graphs))
    for traj_id in range(int(trajectories.num_trajectories)):
        sample_idx = int(sample_ids[traj_id])
        graph_id = int(trajectories.graph_ids[traj_id].detach().cpu().item())
        rows.append(
            {
                "sample_id": sample_id_at(batch, graph_id),
                "sample_index": sample_idx,
                "graph_id": graph_id,
                "trajectory_id": traj_id,
                "valid": bool(valid[graph_id].detach().cpu().item()),
                "natural_recall": float(natural_recall[sample_idx, graph_id].detach().cpu().item()),
                "oracle_recall": float(oracle_recall[sample_idx, graph_id].detach().cpu().item()),
                "natural_edge_count": float(natural_edge_count[sample_idx, graph_id].detach().cpu().item()),
                "oracle_edge_count": float(oracle_edge_count[sample_idx, graph_id].detach().cpu().item()),
                "stop_reason": int(trajectories.stop_reason[traj_id].detach().cpu().item()),
            }
        )
    return rows


def sample_id_by_trajectory(trajectories: TrajectoryBatch, *, num_graphs: int) -> list[int]:
    counts = torch.bincount(trajectories.graph_ids.detach().cpu(), minlength=int(num_graphs))
    seen = [0 for _ in range(int(num_graphs))]
    out: list[int] = []
    del counts
    for graph_id in trajectories.graph_ids.detach().cpu().tolist():
        out.append(seen[int(graph_id)])
        seen[int(graph_id)] += 1
    return out


def sample_id_at(batch, graph_id: int) -> str:
    sample_ids = getattr(batch, "sample_id", None)
    if isinstance(sample_ids, list) and 0 <= int(graph_id) < len(sample_ids):
        return str(sample_ids[int(graph_id)])
    return str(graph_id)


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> float:
    value = value.detach().cpu().float()
    valid = valid.detach().cpu().bool()
    if value.ndim == 2:
        if not bool(valid.any()):
            return 0.0
        return float(value[:, valid].mean().item())
    if not bool(valid.any()):
        return 0.0
    return float(value[valid].mean().item())


def prefix_float(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": float(value) for key, value in values.items()}


def mean_bool(values: Iterable[Any]) -> float:
    items = [1.0 if bool(value) else 0.0 for value in values]
    return mean_float(items)


def mean_float(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return math.nan
    return float(sum(items) / len(items))


class DiagnosticAccumulator:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._counts: dict[str, float] = defaultdict(float)

    def add(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                name = str(key)
                if name.endswith("/count"):
                    self._counts[name] += float(value)
                else:
                    self._values[name].append(float(value))

    def finalize(self) -> dict[str, float]:
        out = {key: mean_float(values) for key, values in sorted(self._values.items())}
        out.update({key: value for key, value in sorted(self._counts.items())})
        return out


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return torch.device(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
