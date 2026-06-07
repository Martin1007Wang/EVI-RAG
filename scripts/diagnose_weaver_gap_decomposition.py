from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_budget_recall_oracle import (  # noqa: E402
    PathCandidate,
    build_path_candidates,
    exact_budget_curve,
    greedy_budget_curve,
)
from src.eval.rollout import rollout_eval_tensors  # noqa: E402
from src.training.checkpoint import load_checkpoint_weights  # noqa: E402
from src.training.factory import build_model, prepare_training_components  # noqa: E402
from src.weaver.context import GraphContext, TargetContext  # noqa: E402
from src.weaver.policy import STOP_EDGE_ID  # noqa: E402
from src.weaver.policy.output import PolicyOutput  # noqa: E402
from src.weaver.reward import EvidenceStateScoreOutput  # noqa: E402
from src.weaver.rollout.trajectory import (  # noqa: E402
    BUDGET_TRUNCATED,
    NO_FRONTIER,
    POLICY_STOP,
    TrajectoryBatch,
)
from src.weaver.state import ExpansionBatch, StateBatch  # noqa: E402
from src.graph.oracle_replay import _frontier_legal_order  # noqa: E402


DEFAULT_RUN_DIR = Path("outputs/valfit/2026-06-06/18-55-06")
DEFAULT_CKPT = DEFAULT_RUN_DIR / "checkpoints" / "epoch_epoch=075-step_step=0000304.ckpt"
DEFAULT_OUTPUT = Path("outputs/diagnostics/valfit_2026-06-06_18-55-06_epoch044_gap_decomp")
DEFAULT_METADATA_DIR = Path("/mnt/data/retrieval/webqsp/metadata")


@dataclass(frozen=True, slots=True)
class OracleProxy:
    sample_id: str
    target_count: int
    candidates: list[PathCandidate]
    initial_target_bits: int
    target_pos_by_node: dict[int, int]
    recall_by_budget: dict[int, float]
    covered_by_budget: dict[int, int]
    used_edges_by_budget: dict[int, int]
    exact: bool
    best_edge_mask_by_budget: dict[int, int]


@dataclass(frozen=True, slots=True)
class PrefixValue:
    depth: int
    oracle_recall: float
    oracle_log_reward: float
    natural_mean_recall: float
    natural_best_recall: float
    no_stop_mean_recall: float
    no_stop_best_recall: float
    log_flow: float
    stop_prob: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose support / sequencing / STOP / flow gaps for a Weaver checkpoint.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_RUN_DIR / ".hydra" / "config.yaml")
    parser.add_argument("--ckpt-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--num-rollouts", type=int, default=64)
    parser.add_argument("--k-values", default="8,32,64")
    parser.add_argument("--prefix-rollouts", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = sorted({int(x) for x in str(args.k_values).split(",") if str(x).strip()})
    if not k_values:
        raise ValueError("--k-values must be non-empty.")

    cfg = OmegaConf.load(args.config_path)
    cfg.dataset.paths.metadata_dir = str(args.metadata_dir)
    cfg.datamodule.splits.validation = str(args.split)
    cfg.datamodule.eval_num_workers = 0
    cfg.datamodule.num_workers = 0
    cfg.datamodule.prefetch_factor = 2
    cfg.datamodule.eval_batch_size = 1
    cfg.model.budget = int(args.budget)
    cfg.model.runner.eval_rollouts = int(args.num_rollouts)
    cfg.trainer.accelerator = "cpu"
    cfg.trainer.devices = 1
    cfg.trainer.precision = "32-true"
    cfg.logger = None
    cfg.callbacks = None

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule, resources = prepare_training_components(cfg, stage="validate")
    model = build_model(cfg, resources)
    missing, unexpected = load_checkpoint_weights(model, str(args.ckpt_path), strict=False)
    model.to(device)
    model.eval()

    loader = datamodule.val_dataloader()
    summary_accum: dict[str, list[float]] = defaultdict(list)
    per_sample_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    firsthop_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if int(args.max_batches) > 0 and batch_idx >= int(args.max_batches):
                break
            batch = batch.to(device)
            result = diagnose_single_graph_batch(
                model=model,
                batch=batch,
                budget=int(args.budget),
                num_rollouts=int(args.num_rollouts),
                k_values=k_values,
                prefix_rollouts=int(args.prefix_rollouts),
            )
            for key, value in result["summary"].items():
                if math.isfinite(float(value)):
                    summary_accum[key].append(float(value))
            per_sample_rows.append(result["per_sample"])
            prefix_rows.extend(result["prefix_rows"])
            firsthop_rows.extend(result["firsthop_rows"])
            print(f"diagnosed sample {batch_idx + 1}")

    summary = {key: float(mean(values)) for key, values in sorted(summary_accum.items()) if values}
    summary["checkpoint"] = str(args.ckpt_path)
    summary["config_path"] = str(args.config_path)
    summary["split"] = str(args.split)
    summary["budget"] = int(args.budget)
    summary["num_rollouts"] = int(args.num_rollouts)
    summary["k_values"] = k_values
    summary["load_missing_keys"] = missing
    summary["load_unexpected_keys"] = unexpected

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "per_sample.csv", per_sample_rows)
    write_csv(output_dir / "prefix_continuation.csv", prefix_rows)
    write_csv(output_dir / "root_firsthop.csv", firsthop_rows)
    write_report(output_dir / "report.md", summary)
    print(f"wrote {output_dir / 'summary.json'}")


def diagnose_single_graph_batch(
    *,
    model,
    batch,
    budget: int,
    num_rollouts: int,
    k_values: list[int],
    prefix_rollouts: int,
) -> dict[str, Any]:
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
    recall, log_reward = terminal_scores(
        model=model,
        trajectories=trajectories,
        graph=contexts.graph,
        target=contexts.target,
        batch=batch,
    )
    oracle = build_oracle_proxy(batch=batch, budget=int(budget))
    firsthop = root_firsthop_diagnostics(
        model=model,
        graph=contexts.graph,
        features=inputs.features,
        policy_input=inputs.policy_input,
        oracle=oracle,
        budget=int(budget),
    )

    summary: dict[str, float] = {}
    per_sample: dict[str, Any] = {
        "sample_id": str(batch.sample_id[0]),
        "oracle_path_proxy_recall": oracle.recall_by_budget[int(budget)],
    }
    prefix_rows: list[dict[str, Any]] = []

    for k in k_values:
        use_k = min(int(k), int(trajectories.num_trajectories))
        recall_k = recall[:use_k]
        reward_k = log_reward[:use_k]
        expected_recall = float(recall_k.mean().item())
        expected_reward = float(reward_k.mean().item())
        best_recall = float(recall_k.max().item())
        best_reward = float(reward_k.max().item())
        union_oracle = constrained_union_oracle(
            trajectories=trajectories,
            oracle=oracle,
            k=use_k,
            budget=int(budget),
        )
        union_oracle_reward = proxy_reward_from_recall(
            recall=float(union_oracle["recall"]),
            edge_count=float(union_oracle["used_edges"]),
            budget=int(budget),
        )
        summary[f"policy_expected@{k}/recall"] = expected_recall
        summary[f"policy_best@{k}/recall"] = best_recall
        summary[f"union_oracle@{k}/recall"] = float(union_oracle["recall"])
        summary[f"support_gap@{k}/recall"] = oracle.recall_by_budget[int(budget)] - float(union_oracle["recall"])
        summary[f"construction_gap@{k}/recall"] = float(union_oracle["recall"]) - best_recall
        summary[f"sampling_gap@{k}/recall"] = best_recall - expected_recall
        summary[f"policy_expected@{k}/log_reward"] = expected_reward
        summary[f"policy_best@{k}/log_reward"] = best_reward
        summary[f"union_oracle@{k}/log_reward"] = union_oracle_reward
        summary[f"support_gap@{k}/log_reward"] = proxy_reward_from_recall(
            recall=oracle.recall_by_budget[int(budget)],
            edge_count=float(oracle.used_edges_by_budget[int(budget)]),
            budget=int(budget),
        ) - union_oracle_reward
        summary[f"construction_gap@{k}/log_reward"] = union_oracle_reward - best_reward
        summary[f"sampling_gap@{k}/log_reward"] = best_reward - expected_reward
        per_sample[f"policy_expected@{k}_recall"] = expected_recall
        per_sample[f"policy_best@{k}_recall"] = best_recall
        per_sample[f"union_oracle@{k}_recall"] = float(union_oracle["recall"])

    summary["root_firsthop/hit@1"] = 1.0 if firsthop["best_rank"] == 1 else 0.0
    summary["root_firsthop/hit@8"] = 1.0 if firsthop["best_rank"] is not None and firsthop["best_rank"] <= 8 else 0.0
    summary["root_firsthop/hit@32"] = 1.0 if firsthop["best_rank"] is not None and firsthop["best_rank"] <= 32 else 0.0
    summary["root_firsthop/hit@64"] = 1.0 if firsthop["best_rank"] is not None and firsthop["best_rank"] <= 64 else 0.0
    summary["root_firsthop/mrr"] = 0.0 if firsthop["best_rank"] is None else 1.0 / float(firsthop["best_rank"])

    reward_best_idx = int(log_reward.argmax().item()) if int(log_reward.numel()) > 0 else 0
    recall_best_idx = int(recall.argmax().item()) if int(recall.numel()) > 0 else 0
    summary["objective_mismatch/reward_best_differs_from_recall_best"] = 1.0 if reward_best_idx != recall_best_idx else 0.0
    summary["objective_mismatch/reward_recall_corr"] = pearson(log_reward, recall)

    prefix_values = oracle_prefix_continuation(
        model=model,
        graph=contexts.graph,
        target=contexts.target,
        batch=batch,
        features=inputs.features,
        policy_input=inputs.policy_input,
        oracle=oracle,
        budget=int(budget),
        num_rollouts=int(prefix_rollouts),
    )
    for item in prefix_values:
        prefix_rows.append(
            {
                "sample_id": str(batch.sample_id[0]),
                "depth": int(item.depth),
                "oracle_recall": float(item.oracle_recall),
                "oracle_log_reward": float(item.oracle_log_reward),
                "natural_mean_recall": float(item.natural_mean_recall),
                "natural_best_recall": float(item.natural_best_recall),
                "no_stop_mean_recall": float(item.no_stop_mean_recall),
                "no_stop_best_recall": float(item.no_stop_best_recall),
                "log_flow": float(item.log_flow),
                "stop_prob": float(item.stop_prob),
            }
        )
    if prefix_values:
        summary["oracle_prefix/natural_mean_recall"] = mean([x.natural_mean_recall for x in prefix_values])
        summary["oracle_prefix/natural_best_recall"] = mean([x.natural_best_recall for x in prefix_values])
        summary["oracle_prefix/no_stop_mean_recall"] = mean([x.no_stop_mean_recall for x in prefix_values])
        summary["oracle_prefix/no_stop_best_recall"] = mean([x.no_stop_best_recall for x in prefix_values])
        summary["flow/logflow_vs_vstar_pearson"] = pearson(
            torch.tensor([x.log_flow for x in prefix_values], dtype=torch.float32),
            torch.tensor([x.oracle_recall for x in prefix_values], dtype=torch.float32),
        )
    else:
        summary["oracle_prefix/natural_mean_recall"] = math.nan
        summary["oracle_prefix/natural_best_recall"] = math.nan
        summary["oracle_prefix/no_stop_mean_recall"] = math.nan
        summary["oracle_prefix/no_stop_best_recall"] = math.nan
        summary["flow/logflow_vs_vstar_pearson"] = math.nan

    firsthop_rows = [{**firsthop, "sample_id": str(batch.sample_id[0])}]
    per_sample["root_firsthop_rank"] = firsthop["best_rank"] if firsthop["best_rank"] is not None else math.nan
    return {
        "summary": summary,
        "per_sample": per_sample,
        "prefix_rows": prefix_rows,
        "firsthop_rows": firsthop_rows,
    }


def build_oracle_proxy(*, batch, budget: int) -> OracleProxy:
    targets = [int(x) for x in batch.reachable_target_node_ids.detach().cpu().tolist()]
    target_pos_by_node = {node: pos for pos, node in enumerate(targets)}
    candidates, initial_bits, _ = build_path_candidates(
        edge_index=batch.edge_index.detach().cpu(),
        anchor_node_ids=batch.anchor_node_ids.detach().cpu(),
        reachable_target_node_ids=batch.reachable_target_node_ids.detach().cpu(),
        node_target_distances_flat=batch.node_target_distance.detach().cpu(),
        num_nodes=int(batch.num_nodes_total),
        max_paths_per_target=64,
    )
    budgets = list(range(int(budget) + 1))
    exact = exact_budget_curve(
        candidates=candidates,
        initial_target_bits=initial_bits,
        target_count=len(targets),
        budgets=budgets,
        max_dp_states=200_000,
    )
    if exact is None:
        recall_by_budget, covered_by_budget, used_edges_by_budget = greedy_budget_curve(
            candidates=candidates,
            initial_target_bits=initial_bits,
            target_count=len(targets),
            budgets=budgets,
        )
        exact_flag = False
    else:
        recall_by_budget, covered_by_budget, used_edges_by_budget = exact
        exact_flag = True
    best_edge_mask_by_budget = best_edge_masks_by_budget(
        candidates=candidates,
        initial_target_bits=initial_bits,
        target_count=len(targets),
        budgets=budgets,
    )
    return OracleProxy(
        sample_id=str(batch.sample_id[0]),
        target_count=len(targets),
        candidates=candidates,
        initial_target_bits=initial_bits,
        target_pos_by_node=target_pos_by_node,
        recall_by_budget=recall_by_budget,
        covered_by_budget=covered_by_budget,
        used_edges_by_budget=used_edges_by_budget,
        exact=exact_flag,
        best_edge_mask_by_budget=best_edge_mask_by_budget,
    )


def best_edge_masks_by_budget(
    *,
    candidates: list[PathCandidate],
    initial_target_bits: int,
    target_count: int,
    budgets: list[int],
) -> dict[int, int]:
    max_budget = max(budgets) if budgets else 0
    states: set[tuple[int, int]] = {(0, int(initial_target_bits))}
    for candidate in candidates:
        next_states = set(states)
        for edge_mask, covered_bits in states:
            new_edge_mask = edge_mask | candidate.edge_bits
            if new_edge_mask.bit_count() > max_budget:
                continue
            next_states.add((new_edge_mask, covered_bits | candidate.covered_target_bits))
        if len(next_states) > 200_000:
            return greedy_edge_masks_by_budget(
                candidates=candidates,
                initial_target_bits=initial_target_bits,
                target_count=target_count,
                budgets=budgets,
            )
        states = next_states
    out: dict[int, int] = {}
    for budget in budgets:
        best_mask = 0
        best_cover = -1
        best_used = 1_000_000_000
        for edge_mask, covered_bits in states:
            used = edge_mask.bit_count()
            if used > int(budget):
                continue
            cover = covered_bits.bit_count()
            if cover > best_cover or (cover == best_cover and used < best_used):
                best_mask = edge_mask
                best_cover = cover
                best_used = used
        out[int(budget)] = int(best_mask)
    return out


def greedy_edge_masks_by_budget(
    *,
    candidates: list[PathCandidate],
    initial_target_bits: int,
    target_count: int,
    budgets: list[int],
) -> dict[int, int]:
    del target_count
    out: dict[int, int] = {}
    for budget in budgets:
        edge_mask = 0
        covered_bits = int(initial_target_bits)
        remaining = list(candidates)
        while True:
            best_idx = -1
            best_score = 0.0
            best_new_mask = 0
            best_new_cover = 0
            for idx, candidate in enumerate(remaining):
                candidate_mask = edge_mask | candidate.edge_bits
                if candidate_mask.bit_count() > int(budget):
                    continue
                new_cover = candidate.covered_target_bits & ~covered_bits
                if new_cover == 0:
                    continue
                new_edges = candidate.edge_bits & ~edge_mask
                score = new_cover.bit_count() / max(1, new_edges.bit_count())
                if score > best_score:
                    best_idx = idx
                    best_score = score
                    best_new_mask = new_edges
                    best_new_cover = new_cover
            if best_idx < 0:
                break
            edge_mask |= best_new_mask
            covered_bits |= best_new_cover
            remaining.pop(best_idx)
        out[int(budget)] = int(edge_mask)
    return out


def constrained_union_oracle(
    *,
    trajectories: TrajectoryBatch,
    oracle: OracleProxy,
    k: int,
    budget: int,
) -> dict[str, float]:
    allowed_edges: set[int] = set()
    for row in range(int(k)):
        count = int(trajectories.edge_count[row].item())
        allowed_edges.update(int(x) for x in trajectories.edge_ids[row, :count].detach().cpu().tolist())
    candidates = [candidate for candidate in oracle.candidates if candidate.edge_ids.issubset(allowed_edges)]
    if not candidates or oracle.target_count <= 0:
        return {"recall": 0.0, "used_edges": 0.0}
    exact = exact_budget_curve(
        candidates=candidates,
        initial_target_bits=oracle.initial_target_bits,
        target_count=oracle.target_count,
        budgets=[int(budget)],
        max_dp_states=200_000,
    )
    if exact is None:
        recall_by_budget, _, used_edges_by_budget = greedy_budget_curve(
            candidates=candidates,
            initial_target_bits=oracle.initial_target_bits,
            target_count=oracle.target_count,
            budgets=[int(budget)],
        )
    else:
        recall_by_budget, _, used_edges_by_budget = exact
    return {
        "recall": float(recall_by_budget[int(budget)]),
        "used_edges": float(used_edges_by_budget[int(budget)]),
    }


def root_firsthop_diagnostics(
    *,
    model,
    graph: GraphContext,
    features,
    policy_input,
    oracle: OracleProxy,
    budget: int,
) -> dict[str, Any]:
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long, device=graph.device),
        budget=int(budget),
        graph_context=graph,
    )
    output = model.policy(
        state=state,
        features=features,
        graph_context=graph,
        policy_input=policy_input,
    )
    productive = root_productive_edges(output=output, oracle=oracle)
    row_mask = output.action_row_ids.eq(0)
    edge_ids = output.action_edge_ids[row_mask].detach().cpu().tolist()
    scores = output.action_log_prob[row_mask].detach().cpu().tolist()
    ranked = sorted(zip(edge_ids, scores, strict=False), key=lambda item: item[1], reverse=True)
    best_rank = None
    for idx, (edge_id, _) in enumerate(ranked, start=1):
        if int(edge_id) in productive:
            best_rank = idx
            break
    return {
        "productive_count": len(productive),
        "best_rank": best_rank,
    }


def root_productive_edges(*, output: PolicyOutput, oracle: OracleProxy) -> set[int]:
    frontier_edges = set(int(x) for x in output.frontier.edge_ids.detach().cpu().tolist())
    productive: set[int] = set()
    for candidate in oracle.candidates:
        productive.update(int(edge_id) for edge_id in candidate.edge_ids if int(edge_id) in frontier_edges)
    return productive


def oracle_prefix_continuation(
    *,
    model,
    graph: GraphContext,
    target: TargetContext,
    batch,
    features,
    policy_input,
    oracle: OracleProxy,
    budget: int,
    num_rollouts: int,
) -> list[PrefixValue]:
    edge_mask = oracle.best_edge_mask_by_budget.get(int(budget), 0)
    edge_ids = edge_ids_from_mask(edge_mask)
    if not edge_ids:
        return []
    ordered = _frontier_legal_order(
        edge_ids=frozenset(edge_ids),
        anchors=set(int(x) for x in batch.anchor_node_ids.detach().cpu().tolist()),
        edge_index=batch.edge_index.detach().cpu(),
        token=0,
    )
    if not ordered:
        return []

    out: list[PrefixValue] = []
    for depth in range(1, min(len(ordered), int(budget))):
        prefix_state = state_from_edge_prefix(
            edge_ids=ordered[:depth],
            graph=graph,
            budget=int(budget),
        )
        natural = rollout_from_state(
            model=model,
            graph=graph,
            features=features,
            policy_input=policy_input,
            initial_state=prefix_state,
            budget=int(budget),
            num_rollouts=int(num_rollouts),
            allow_stop=True,
        )
        no_stop = rollout_from_state(
            model=model,
            graph=graph,
            features=features,
            policy_input=policy_input,
            initial_state=prefix_state,
            budget=int(budget),
            num_rollouts=int(num_rollouts),
            allow_stop=False,
        )
        natural_recall, _ = terminal_scores(
            model=model,
            trajectories=natural,
            graph=graph,
            target=target,
            batch=batch,
        )
        no_stop_recall, _ = terminal_scores(
            model=model,
            trajectories=no_stop,
            graph=graph,
            target=target,
            batch=batch,
        )
        action_space = model.policy.prepare_action_space(state=prefix_state, graph_context=graph)
        reward = model.reward_model(
            state=prefix_state,
            target_context=target,
            graph_context=graph,
            active=action_space.active,
        )
        output = model.policy(
            state=prefix_state,
            features=features,
            graph_context=graph,
            policy_input=policy_input,
            action_space=action_space,
            compute_log_flow=True,
        )
        log_flow = float((output.require_log_flow_base() + reward.state_potential).item())
        stop_prob = float(output.gather_log_prob(
            row_ids=torch.tensor([0], device=graph.device),
            edge_ids=torch.tensor([STOP_EDGE_ID], device=graph.device),
        ).exp().item())
        prefix_bits = covered_target_bits_from_state(batch=batch, state=prefix_state, oracle=oracle)
        prefix_mask = 0
        for edge_id in ordered[:depth]:
            prefix_mask |= 1 << int(edge_id)
        value = best_proxy_value_from_prefix(
            oracle=oracle,
            prefix_edge_mask=prefix_mask,
            prefix_target_bits=prefix_bits,
            budget=int(budget),
        )
        out.append(
            PrefixValue(
                depth=int(depth),
                oracle_recall=float(value["recall"]),
                oracle_log_reward=proxy_reward_from_recall(
                    recall=float(value["recall"]),
                    edge_count=float(value["used_edges"]),
                    budget=int(budget),
                ),
                natural_mean_recall=float(natural_recall.mean().item()),
                natural_best_recall=float(natural_recall.max().item()),
                no_stop_mean_recall=float(no_stop_recall.mean().item()),
                no_stop_best_recall=float(no_stop_recall.max().item()),
                log_flow=log_flow,
                stop_prob=stop_prob,
            )
        )
    return out


def state_from_edge_prefix(*, edge_ids: list[int], graph: GraphContext, budget: int) -> StateBatch:
    prefix = torch.full((1, int(budget)), -1, dtype=torch.long, device=graph.device)
    if edge_ids:
        prefix[0, : len(edge_ids)] = torch.tensor(edge_ids, dtype=torch.long, device=graph.device)
    return StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0], dtype=torch.long, device=graph.device),
        edge_ids=prefix,
        edge_count=torch.tensor([len(edge_ids)], dtype=torch.long, device=graph.device),
        budget=int(budget),
        graph_context=graph,
    )


def best_proxy_value_from_prefix(
    *,
    oracle: OracleProxy,
    prefix_edge_mask: int,
    prefix_target_bits: int,
    budget: int,
) -> dict[str, float]:
    states: set[tuple[int, int]] = {(int(prefix_edge_mask), int(prefix_target_bits | oracle.initial_target_bits))}
    for candidate in oracle.candidates:
        next_states = set(states)
        for edge_mask, covered_bits in states:
            new_edge_mask = edge_mask | candidate.edge_bits
            if new_edge_mask.bit_count() > int(budget):
                continue
            next_states.add((new_edge_mask, covered_bits | candidate.covered_target_bits))
        if len(next_states) > 200_000:
            break
        states = next_states
    best_cover = -1
    best_used = 0
    for edge_mask, covered_bits in states:
        used = edge_mask.bit_count()
        if used > int(budget):
            continue
        cover = covered_bits.bit_count()
        if cover > best_cover or (cover == best_cover and used < best_used):
            best_cover = cover
            best_used = used
    if oracle.target_count <= 0:
        return {"recall": 0.0, "used_edges": 0.0}
    return {
        "recall": float(max(best_cover, 0) / float(oracle.target_count)),
        "used_edges": float(best_used),
    }


def covered_target_bits_from_state(*, batch, state: StateBatch, oracle: OracleProxy) -> int:
    edge_index = batch.edge_index.detach().cpu()
    anchors = set(int(x) for x in batch.anchor_node_ids.detach().cpu().tolist())
    active = set(anchors)
    count = int(state.edge_count[0].item())
    for edge_id in state.edge_ids[0, :count].detach().cpu().tolist():
        active.add(int(edge_index[0, edge_id].item()))
        active.add(int(edge_index[1, edge_id].item()))
    bits = 0
    for node_id in active:
        pos = oracle.target_pos_by_node.get(int(node_id))
        if pos is not None:
            bits |= 1 << int(pos)
    return bits


def rollout_from_state(
    *,
    model,
    graph: GraphContext,
    features,
    policy_input,
    initial_state: StateBatch,
    budget: int,
    num_rollouts: int,
    allow_stop: bool,
) -> TrajectoryBatch:
    state = repeat_state(
        initial_state=initial_state,
        repeats=int(num_rollouts),
        graph=graph,
    )
    num_rows = state.num_states
    edge_ids = state.edge_ids.clone()
    edge_logp = torch.zeros((num_rows, int(budget)), dtype=torch.float32, device=graph.device)
    edge_count = state.edge_count.clone()
    stop_reason = torch.full((num_rows,), -1, dtype=torch.long, device=graph.device)
    stop_logp = torch.zeros((num_rows,), dtype=torch.float32, device=graph.device)
    done = torch.zeros((num_rows,), dtype=torch.bool, device=graph.device)
    all_rows = torch.arange(num_rows, dtype=torch.long, device=graph.device)

    while not bool(done.all()):
        active_rows = all_rows[~done]
        if int(active_rows.numel()) == 0:
            break
        budget_rows = active_rows[state.edge_count.index_select(0, active_rows).ge(int(budget))]
        if int(budget_rows.numel()) > 0:
            stop_reason[budget_rows] = int(BUDGET_TRUNCATED)
            stop_logp[budget_rows] = 0.0
            done[budget_rows] = True
        decision_rows = active_rows[~state.edge_count.index_select(0, active_rows).ge(int(budget))]
        if int(decision_rows.numel()) == 0:
            continue
        decision_state = state.take(decision_rows)
        output = model.policy(
            state=decision_state,
            features=features,
            graph_context=graph,
            policy_input=policy_input,
        )
        sampled = sample_actions(output=output, allow_stop=bool(allow_stop))
        row_to_global = decision_rows.index_select(0, sampled.row_ids)
        stop_mask = sampled.edge_ids.eq(int(STOP_EDGE_ID))
        if bool(stop_mask.any()):
            stopped_rows = row_to_global[stop_mask]
            frontier_count = torch.bincount(output.frontier.row_ids, minlength=decision_state.num_states)
            stopped_local_rows = sampled.row_ids[stop_mask]
            no_frontier = frontier_count.index_select(0, stopped_local_rows).eq(0)
            if bool(no_frontier.any()):
                stop_reason[stopped_rows[no_frontier]] = int(NO_FRONTIER)
            if bool((~no_frontier).any()):
                stop_reason[stopped_rows[~no_frontier]] = int(POLICY_STOP)
            stop_logp[stopped_rows] = sampled.log_prob[stop_mask]
            done[stopped_rows] = True
        expand_mask = sampled.edge_ids.ge(0)
        if bool(expand_mask.any()):
            expand_rows = row_to_global[expand_mask]
            expand_edges = sampled.edge_ids[expand_mask]
            expand_logp = sampled.log_prob[expand_mask]
            pos = state.edge_count.index_select(0, expand_rows)
            edge_ids[expand_rows, pos] = expand_edges
            edge_logp[expand_rows, pos] = expand_logp
            edge_count[expand_rows] = edge_count[expand_rows] + 1
            state = state.advance(
                ExpansionBatch(state_ids=expand_rows, edge_ids=expand_edges),
                graph_context=graph,
                trusted=True,
            )
    unfinished = stop_reason.lt(0)
    if bool(unfinished.any()):
        stop_reason[unfinished] = int(BUDGET_TRUNCATED)
    return TrajectoryBatch(
        graph_ids=state.graph_ids,
        edge_ids=edge_ids,
        edge_logp=edge_logp,
        edge_count=edge_count,
        stop_reason=stop_reason.to(dtype=torch.uint8),
        stop_logp=stop_logp,
        source=torch.zeros((num_rows,), dtype=torch.bool, device=graph.device),
    )


def sample_actions(*, output: PolicyOutput, allow_stop: bool) -> Any:
    row_ids = torch.arange(output.num_states, dtype=torch.long, device=output.device)
    chosen_edges: list[int] = []
    chosen_logp: list[float] = []
    for row in row_ids.tolist():
        row_mask = output.action_row_ids.eq(int(row))
        logits = output.action_logits[row_mask].float()
        edge_ids = output.action_edge_ids[row_mask]
        if not allow_stop and bool(edge_ids.ge(0).any()):
            keep = edge_ids.ge(0)
            logits = logits[keep]
            edge_ids = edge_ids[keep]
        probs = torch.softmax(logits, dim=0)
        idx = torch.multinomial(probs, num_samples=1).item()
        chosen_edges.append(int(edge_ids[idx].item()))
        chosen_logp.append(float(torch.log(probs[idx].clamp_min(1.0e-30)).item()))
    return type(
        "Sampled",
        (),
        {
            "row_ids": row_ids,
            "edge_ids": torch.tensor(chosen_edges, dtype=torch.long, device=output.device),
            "log_prob": torch.tensor(chosen_logp, dtype=torch.float32, device=output.device),
        },
    )()


def repeat_state(*, initial_state: StateBatch, repeats: int, graph: GraphContext) -> StateBatch:
    return StateBatch.from_selected_edges(
        graph_ids=initial_state.graph_ids.repeat(int(repeats)),
        edge_ids=initial_state.edge_ids.repeat(int(repeats), 1),
        edge_count=initial_state.edge_count.repeat(int(repeats)),
        budget=int(initial_state.edge_capacity),
        graph_context=graph,
    )


def terminal_scores(
    *,
    model,
    trajectories: TrajectoryBatch,
    graph: GraphContext,
    target: TargetContext,
    batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = rollout_eval_tensors(
        trajectories=trajectories,
        batch=batch,
        context=graph,
        exclude_anchors_from_retrieved=bool(model.evaluation.exclude_anchors_from_retrieved),
        use_reachable_targets=bool(model.evaluation.use_reachable_targets),
    )
    state = StateBatch.from_selected_edges(
        graph_ids=trajectories.graph_ids,
        edge_ids=trajectories.edge_ids,
        edge_count=trajectories.edge_count,
        budget=int(trajectories.budget),
        graph_context=graph,
    )
    reward = model.reward_model(state=state, target_context=target, graph_context=graph)
    return tensors.recall[:, 0].detach().cpu(), reward.log_reward.detach().cpu()


def edge_ids_from_mask(mask: int) -> list[int]:
    out: list[int] = []
    work = int(mask)
    while work:
        low = work & -work
        out.append(int(low.bit_length() - 1))
        work ^= low
    out.sort()
    return out


def proxy_reward_from_recall(*, recall: float, edge_count: float, budget: int) -> float:
    zero_hit = 1.0 if float(recall) <= 0.0 else 0.0
    return float(recall) - 0.1 * float(edge_count) / float(budget) - 1.0 * zero_hit


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().cpu().float().view(-1)
    y = y.detach().cpu().float().view(-1)
    if int(x.numel()) == 0 or int(y.numel()) == 0 or int(x.numel()) != int(y.numel()):
        return math.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom.item()) <= 0.0:
        return math.nan
    return float((x * y).sum().item() / denom.item())


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return torch.device(name)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Weaver Gap Decomposition Report",
        "",
        f"- Checkpoint: `{summary.get('checkpoint', '')}`",
        f"- Split: `{summary.get('split', '')}`",
        f"- Budget: `{summary.get('budget', '')}`",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "policy_expected@64/recall",
        "policy_best@64/recall",
        "union_oracle@64/recall",
        "support_gap@64/recall",
        "construction_gap@64/recall",
        "sampling_gap@64/recall",
        "root_firsthop/hit@32",
        "oracle_prefix/no_stop_mean_recall",
        "flow/logflow_vs_vstar_pearson",
        "objective_mismatch/reward_recall_corr",
    ):
        if key in summary:
            value = summary[key]
            if isinstance(value, float):
                lines.append(f"- {key}: `{value:.6f}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
