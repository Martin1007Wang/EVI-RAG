from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch

from src.training.factory import build_datamodule, build_model, setup_datamodule
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.action import StepAction, sample_step
from src.weaver.rollout.engine import forced_stop_rows
from src.weaver.state import State

NUM_FRONTIER_BUCKETS = 10
FRONTIER_BUCKET_WIDTH = 5


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if args.data_dir:
        os.environ.setdefault("DATA_DIR", args.data_dir)

    device = torch.device(args.device)
    cfg = compose_config(args)
    dm = build_datamodule(cfg)
    resources = setup_datamodule(dm, stage="fit")
    model = build_model(cfg, resources)
    load_checkpoint(model, args.ckpt)
    model.to(device).eval()

    dataset = dm.train_dataset if args.split == "train" else dm.val_dataset
    if dataset is None:
        raise RuntimeError(f"{args.split} dataset was not initialized")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx in choose_indices(dataset, args):
            batch = dm.collator([dataset[idx]]).to(device)
            record = trace_sample(
                model=model,
                batch=batch,
                sample_idx=int(idx),
                num_rollouts=int(args.rollouts),
                temperature=float(args.temperature),
            )
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            print_summary(record)

    print(f"wrote {out_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="outputs/stop_trace/trace.jsonl")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--experiment", default="debug/valfit")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-scan", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-edges", type=int, default=2500)
    parser.add_argument("--rollouts", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def compose_config(args: argparse.Namespace):
    overrides = [
        f"experiment={args.experiment}",
        "logger=none",
        "trainer=cpu",
        "datamodule.batch_size=1",
        "datamodule.eval_batch_size=1",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        "datamodule.train_shuffle=false",
    ]
    if args.data_dir:
        overrides.append(f"paths.data_dir={args.data_dir}")

    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_dir).resolve()),
        version_base=None,
    ):
        return hydra.compose(config_name="train", overrides=overrides)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    print(
        f"loaded ckpt={ckpt_path} epoch={ckpt.get('epoch')} step={ckpt.get('global_step')}",
        flush=True,
    )


def choose_indices(dataset: Any, args: argparse.Namespace) -> list[int]:
    out: list[int] = []
    end = min(len(dataset), int(args.start_idx) + int(args.max_scan))
    for idx in range(int(args.start_idx), end):
        data = dataset[idx]
        num_edges = int(getattr(data, "edge_index").size(1))
        reachable = int(getattr(data, "reachable_target_node_ids").numel())
        if reachable <= 0 or num_edges > int(args.max_edges):
            continue
        out.append(idx)
        if len(out) >= int(args.num_samples):
            break
    if not out:
        raise RuntimeError("no suitable samples found")
    return out


@torch.no_grad()
def trace_sample(
    *,
    model,
    batch,
    sample_idx: int,
    num_rollouts: int,
    temperature: float,
) -> dict[str, Any]:
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    features = model.policy_feature_encoder(batch)
    graph_ids = torch.arange(graph.num_graphs, dtype=torch.long, device=graph.device).repeat_interleave(
        int(num_rollouts)
    )
    state = State.initial(graph=graph, graph_ids=graph_ids)

    stopped = torch.zeros(state.num_rows, dtype=torch.bool, device=graph.device)
    stop_step = torch.full((state.num_rows,), -1, dtype=torch.long, device=graph.device)
    forced_terminal = torch.zeros(state.num_rows, dtype=torch.bool, device=graph.device)
    first_hit_step = torch.full((state.num_rows,), -1, dtype=torch.long, device=graph.device)

    depth_records: list[dict[str, Any]] = []
    bucket_accum = new_bucket_accumulator()
    budget = int(model.runner.engine.expand_budget)
    for depth in range(budget + 1):
        active_rows = (~stopped).nonzero(as_tuple=False).flatten()
        if active_rows.numel() == 0:
            break

        active_state = state.select_rows(active_rows)
        active_hit = row_hit_mask(active_state, target)
        newly_hit = active_hit & first_hit_step.index_select(0, active_rows).lt(0)
        if bool(newly_hit.any()):
            first_hit_step[active_rows[newly_hit]] = int(depth)

        physical_frontier = active_state.frontier(graph)
        active_frontier_counts = torch.bincount(
            physical_frontier.row_ids,
            minlength=active_state.num_rows,
        ).float()
        frontier = active_state.frontier(
            graph,
            expand_budget=budget,
        )
        legal_frontier_counts = torch.bincount(
            frontier.row_ids,
            minlength=active_state.num_rows,
        ).float()
        forced_local = forced_stop_rows(
            state=active_state,
            frontier=frontier,
            expand_budget=budget,
        )
        update_forced_buckets(
            accum=bucket_accum,
            frontier_counts=legal_frontier_counts,
            forced_local=forced_local,
        )
        sample_rows = active_rows_without_forced_rows(
            active_rows=active_rows,
            forced_local_rows=forced_local,
        )

        record = {
            "depth": int(depth),
            "active_rows": int(active_rows.numel()),
            "forced_rows": int(forced_local.numel()),
            "sample_rows": int(sample_rows.numel()),
            "active_hit_rate": scalar_mean(active_hit.float()),
            "active_frontier_mean": scalar_mean(active_frontier_counts),
            "active_frontier_max": scalar_max(active_frontier_counts),
            "legal_frontier_mean": scalar_mean(legal_frontier_counts),
            "legal_frontier_max": scalar_max(legal_frontier_counts),
            "forced_exhausted_rows": int(active_state.depth.ge(budget).sum().item()),
            "forced_no_frontier_rows": int(active_frontier_counts.eq(0).sum().item()),
            "forced_no_legal_frontier_rows": int(legal_frontier_counts.eq(0).sum().item()),
        }

        action = StepAction.forced_stop(
            rows=active_rows.index_select(0, forced_local),
            dtype=torch.float32,
            device=graph.device,
        )

        if sample_rows.numel() > 0:
            sample_state = state.select_rows(sample_rows)
            sample_frontier = sample_state.frontier(
                graph,
                expand_budget=budget,
            )
            policy_out = model.policy(
                features=features,
                state=sample_state,
                context=graph,
                frontier=sample_frontier,
            )
            sampled_local = sample_step(
                policy_out=policy_out,
                rows=torch.arange(sample_state.num_rows, dtype=torch.long, device=graph.device),
                temperature=float(temperature),
            )
            sampled = StepAction(
                row_ids=sample_rows.index_select(0, sampled_local.row_ids),
                edge_ids=sampled_local.edge_ids,
                policy_log_prob=sampled_local.policy_log_prob,
                behavior_log_prob=sampled_local.behavior_log_prob,
                forced=sampled_local.forced,
            )
            action = StepAction.concat([sampled, action])
            record.update(policy_stats(policy_out, sample_state, sample_frontier, target, sampled_local))
            update_policy_buckets(
                accum=bucket_accum,
                policy_out=policy_out,
                sample_state=sample_state,
                frontier=sample_frontier,
                target=target,
                sampled=sampled_local,
            )

        depth_records.append(record)

        stop_mask = action.stop_mask
        if bool(stop_mask.any()):
            stopped[action.stop_rows] = True
            stop_step[action.stop_rows] = int(depth)
            forced_terminal[action.stop_rows] = action.forced[stop_mask]

        if bool(action.expand_mask.any()):
            state = state.expand(
                graph=graph,
                rows=action.expand_rows,
                edge_ids=action.expand_edge_ids,
                expand_budget=budget,
            )

    unstopped = stop_step.lt(0)
    if bool(unstopped.any()):
        stop_step[unstopped] = budget
        forced_terminal[unstopped] = True

    final_hit = row_hit_mask(state, target)
    ever_hit = first_hit_step.ge(0)
    hit_then_continue = ever_hit & stop_step.gt(first_hit_step)

    reward = model.reward_model(
        state=state,
        graph_context=graph,
        target_context=target,
    )

    return {
        "sample_idx": int(sample_idx),
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "num_rollouts": int(num_rollouts),
        "budget": int(budget),
        "summary": {
            "policy_stop_rate": scalar_mean((~forced_terminal).float()),
            "forced_stop_rate": scalar_mean(forced_terminal.float()),
            "final_hit_rate": scalar_mean(final_hit.float()),
            "ever_hit_rate": scalar_mean(ever_hit.float()),
            "hit_then_continue_rate": masked_mean(hit_then_continue.float(), ever_hit),
            "stop_step_mean": scalar_mean(stop_step.float()),
            "first_hit_step_mean": masked_mean(first_hit_step.float(), ever_hit),
            "terminal_log_reward_mean": scalar_mean(reward.log_reward.float()),
            "terminal_raw_log_reward_mean": scalar_mean(reward.raw_log_reward.float()),
        },
        "frontier_size_buckets": summarize_bucket_accumulator(bucket_accum),
        "depth": depth_records,
    }


def policy_stats(
    policy_out,
    sample_state: State,
    frontier,
    target: TargetContext,
    sampled: StepAction,
) -> dict[str, Any]:
    hit = row_hit_mask(sample_state, target)
    support = supported_count(sample_state, target).float()
    stop_lp = policy_out.stop_log_prob().float()
    stop_p = policy_out.stop_prob().float()
    margin = stop_expand_margin(policy_out)
    frontier_counts = torch.bincount(
        frontier.row_ids,
        minlength=sample_state.num_rows,
    ).float()
    sampled_stop = sampled.edge_ids.lt(0)

    return {
        "frontier_mean": scalar_mean(frontier_counts),
        "frontier_max": scalar_max(frontier_counts),
        "hit_rate": scalar_mean(hit.float()),
        "supported_count_mean": scalar_mean(support),
        "stop_log_prob_mean": scalar_mean(stop_lp),
        "stop_prob_mean": scalar_mean(stop_p),
        "stop_expand_margin_mean": finite_mean(margin),
        "stop_logit_mean": scalar_mean(policy_out.stop_logit.float()),
        "state_log_flow_mean": scalar_mean(policy_out.state_log_flow.float()),
        "sampled_stop_rate": scalar_mean(sampled_stop.float()),
        "sampled_stop_rate_hit": masked_mean(sampled_stop.float(), hit),
        "sampled_stop_rate_nohit": masked_mean(sampled_stop.float(), ~hit),
        "stop_prob_hit": masked_mean(stop_p, hit),
        "stop_prob_nohit": masked_mean(stop_p, ~hit),
        "stop_margin_hit": masked_finite_mean(margin, hit),
        "stop_margin_nohit": masked_finite_mean(margin, ~hit),
    }


def stop_expand_margin(policy_out) -> torch.Tensor:
    stop_logit = policy_out.stop_logit.float()
    has_edge = torch.zeros(
        int(policy_out.num_rows),
        dtype=torch.bool,
        device=stop_logit.device,
    )
    if policy_out.edge_row_ids.numel() > 0:
        has_edge.index_fill_(
            0,
            policy_out.edge_row_ids.to(device=stop_logit.device, dtype=torch.long),
            True,
        )
    return torch.where(has_edge, stop_logit, torch.full_like(stop_logit, math.inf))


def new_bucket_accumulator() -> list[dict[str, float]]:
    return [
        {
            "active_count": 0.0,
            "forced_sum": 0.0,
            "policy_count": 0.0,
            "stop_prob_sum": 0.0,
            "margin_sum": 0.0,
            "margin_count": 0.0,
            "sampled_stop_sum": 0.0,
            "hit_sum": 0.0,
            "hit_count": 0.0,
            "nohit_count": 0.0,
            "stop_prob_hit_sum": 0.0,
            "stop_prob_nohit_sum": 0.0,
            "margin_hit_sum": 0.0,
            "margin_hit_count": 0.0,
            "margin_nohit_sum": 0.0,
            "margin_nohit_count": 0.0,
        }
        for _ in range(NUM_FRONTIER_BUCKETS)
    ]


def active_rows_without_forced_rows(
    *,
    active_rows: torch.Tensor,
    forced_local_rows: torch.Tensor,
) -> torch.Tensor:
    if forced_local_rows.numel() == 0:
        return active_rows
    keep = torch.ones(active_rows.numel(), dtype=torch.bool, device=active_rows.device)
    keep[forced_local_rows.to(device=active_rows.device, dtype=torch.long)] = False
    return active_rows[keep]


def frontier_bucket_ids(frontier_counts: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        torch.div(
            frontier_counts.to(dtype=torch.long),
            FRONTIER_BUCKET_WIDTH,
            rounding_mode="floor",
        ),
        max=NUM_FRONTIER_BUCKETS - 1,
    )


def update_forced_buckets(
    *,
    accum: list[dict[str, float]],
    frontier_counts: torch.Tensor,
    forced_local: torch.Tensor,
) -> None:
    if frontier_counts.numel() == 0:
        return
    forced = torch.zeros(frontier_counts.numel(), dtype=torch.bool, device=frontier_counts.device)
    if forced_local.numel() > 0:
        forced[forced_local.to(device=frontier_counts.device, dtype=torch.long)] = True
    buckets = frontier_bucket_ids(frontier_counts)
    for bucket in range(NUM_FRONTIER_BUCKETS):
        mask = buckets.eq(bucket)
        if not bool(mask.any()):
            continue
        accum[bucket]["active_count"] += float(mask.sum().detach().cpu().item())
        accum[bucket]["forced_sum"] += float(forced[mask].float().sum().detach().cpu().item())


def update_policy_buckets(
    *,
    accum: list[dict[str, float]],
    policy_out,
    sample_state: State,
    frontier,
    target: TargetContext,
    sampled: StepAction,
) -> None:
    frontier_counts = torch.bincount(
        frontier.row_ids,
        minlength=sample_state.num_rows,
    ).float()
    stop_p = policy_out.stop_prob().float()
    margin = stop_expand_margin(policy_out)
    hit = row_hit_mask(sample_state, target)

    sampled_stop = torch.zeros(sample_state.num_rows, dtype=torch.bool, device=frontier_counts.device)
    if sampled.row_ids.numel() > 0:
        sampled_stop[sampled.row_ids.to(dtype=torch.long, device=frontier_counts.device)] = sampled.edge_ids.lt(0)

    buckets = frontier_bucket_ids(frontier_counts)
    finite_margin = torch.isfinite(margin)
    for bucket in range(NUM_FRONTIER_BUCKETS):
        mask = buckets.eq(bucket)
        if not bool(mask.any()):
            continue
        accum[bucket]["policy_count"] += float(mask.sum().detach().cpu().item())
        accum[bucket]["stop_prob_sum"] += float(stop_p[mask].sum().detach().cpu().item())
        accum[bucket]["sampled_stop_sum"] += float(sampled_stop[mask].float().sum().detach().cpu().item())
        accum[bucket]["hit_sum"] += float(hit[mask].float().sum().detach().cpu().item())

        hit_mask = mask & hit
        if bool(hit_mask.any()):
            accum[bucket]["hit_count"] += float(hit_mask.sum().detach().cpu().item())
            accum[bucket]["stop_prob_hit_sum"] += float(stop_p[hit_mask].sum().detach().cpu().item())
        nohit_mask = mask & ~hit
        if bool(nohit_mask.any()):
            accum[bucket]["nohit_count"] += float(nohit_mask.sum().detach().cpu().item())
            accum[bucket]["stop_prob_nohit_sum"] += float(stop_p[nohit_mask].sum().detach().cpu().item())

        margin_mask = mask & finite_margin
        if bool(margin_mask.any()):
            accum[bucket]["margin_count"] += float(margin_mask.sum().detach().cpu().item())
            accum[bucket]["margin_sum"] += float(margin[margin_mask].sum().detach().cpu().item())
        margin_hit_mask = hit_mask & finite_margin
        if bool(margin_hit_mask.any()):
            accum[bucket]["margin_hit_count"] += float(margin_hit_mask.sum().detach().cpu().item())
            accum[bucket]["margin_hit_sum"] += float(margin[margin_hit_mask].sum().detach().cpu().item())
        margin_nohit_mask = nohit_mask & finite_margin
        if bool(margin_nohit_mask.any()):
            accum[bucket]["margin_nohit_count"] += float(margin_nohit_mask.sum().detach().cpu().item())
            accum[bucket]["margin_nohit_sum"] += float(margin[margin_nohit_mask].sum().detach().cpu().item())


def summarize_bucket_accumulator(accum: list[dict[str, float]]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for bucket, values in enumerate(accum):
        policy_count = values["policy_count"]
        active_count = values["active_count"]
        margin_count = values["margin_count"]
        out.append(
            {
                "bucket": bucket,
                "frontier_range": bucket_label(bucket),
                "active_count": int(active_count),
                "policy_count": int(policy_count),
                "stop_prob_mean": safe_ratio(values["stop_prob_sum"], policy_count),
                "stop_expand_margin_mean": safe_ratio(values["margin_sum"], margin_count),
                "sampled_stop_rate": safe_ratio(values["sampled_stop_sum"], policy_count),
                "forced_stop_rate": safe_ratio(values["forced_sum"], active_count),
                "hit_rate": safe_ratio(values["hit_sum"], policy_count),
                "stop_prob_hit_mean": safe_ratio(values["stop_prob_hit_sum"], values["hit_count"]),
                "stop_prob_nohit_mean": safe_ratio(values["stop_prob_nohit_sum"], values["nohit_count"]),
                "stop_expand_margin_hit_mean": safe_ratio(
                    values["margin_hit_sum"],
                    values["margin_hit_count"],
                ),
                "stop_expand_margin_nohit_mean": safe_ratio(
                    values["margin_nohit_sum"],
                    values["margin_nohit_count"],
                ),
            }
        )
    return out


def bucket_label(bucket: int) -> str:
    start = bucket * FRONTIER_BUCKET_WIDTH
    if bucket >= NUM_FRONTIER_BUCKETS - 1:
        return f"{start}+"
    return f"{start}-{start + FRONTIER_BUCKET_WIDTH - 1}"


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / denominator)


def row_hit_mask(state: State, target: TargetContext) -> torch.Tensor:
    return (state.active_node_mask & target.target_mask.view(1, -1)).any(dim=1)


def supported_count(state: State, target: TargetContext) -> torch.Tensor:
    return (state.active_node_mask & target.target_mask.view(1, -1)).sum(dim=1)


def scalar_mean(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().detach().cpu().item())


def scalar_max(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().max().detach().cpu().item())


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=torch.bool)
    if values.numel() == 0 or not bool(mask.any()):
        return float("nan")
    return float(values.float()[mask].mean().detach().cpu().item())


def finite_mean(values: torch.Tensor) -> float:
    finite = torch.isfinite(values)
    return masked_mean(values, finite)


def masked_finite_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(device=values.device, dtype=torch.bool) & torch.isfinite(values)
    return masked_mean(values, mask)


def print_summary(record: dict[str, Any]) -> None:
    summary = record["summary"]
    print(
        "sample "
        f"idx={record['sample_idx']} "
        f"N={record['num_nodes']} E={record['num_edges']} "
        f"policy_stop={summary['policy_stop_rate']:.4f} "
        f"forced_stop={summary['forced_stop_rate']:.4f} "
        f"ever_hit={summary['ever_hit_rate']:.4f} "
        f"hit_continue={summary['hit_then_continue_rate']:.4f} "
        f"logR={summary['terminal_log_reward_mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
