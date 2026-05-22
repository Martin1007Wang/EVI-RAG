from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
import torch
from omegaconf import DictConfig

from src.eval.rollout import evaluate_rollout_samples
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_datamodule, build_model, setup_datamodule
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.policy import ForwardPolicy, PolicyOutput
from src.weaver.rollout.action import StepAction
from src.weaver.rollout.engine import RolloutEngine, forced_stop_rows, rows_without_forced
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.tape import RolloutTape
from src.weaver.state import State


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--experiment", default="debug/valfit")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def compose_config(args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"experiment={args.experiment}",
        "logger=none",
        "callbacks=eval",
        "trainer=cpu",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        "datamodule.batch_size=1",
        "datamodule.eval_batch_size=1",
        "validate=true",
        "test=false",
    ]
    with hydra.initialize_config_dir(
        config_dir=str(Path(args.config_dir).resolve()),
        version_base=None,
    ):
        return hydra.compose(config_name="evaluate", overrides=overrides)


def scalar_weighted_mean(total: dict[str, float], weight: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in total.items():
        denom = weight.get(key, 0.0)
        out[key] = value / denom if denom > 0.0 else 0.0
    return out


def accumulate_metrics(
    accum: dict[str, float],
    weights: dict[str, float],
    metrics: dict[str, float],
    batch_weight: float,
) -> None:
    for key, value in metrics.items():
        accum[key] = accum.get(key, 0.0) + float(value) * batch_weight
        weights[key] = weights.get(key, 0.0) + batch_weight


def batch_num_graphs(batch: Any) -> int:
    return int(batch.num_graphs)


def always_continue_step(
    *,
    policy_out: PolicyOutput,
    rows: torch.Tensor,
) -> StepAction:
    rows = rows.to(device=policy_out.stop_logit.device, dtype=torch.long).view(-1)
    if rows.numel() == 0:
        empty_float = policy_out.stop_logit.new_empty((0,)).float()
        return StepAction(
            row_ids=rows,
            edge_ids=rows.new_empty((0,)),
            policy_log_prob=empty_float,
            behavior_log_prob=empty_float,
            forced=torch.zeros(0, dtype=torch.bool, device=rows.device),
        )

    picked_edge_ids = policy_out.sample(
        rows=rows,
    )
    stop_mask = picked_edge_ids.lt(0)
    if bool(stop_mask.any()):
        local_rows = rows[stop_mask]
        forced_continue = _sample_continue_edges(
            policy_out=policy_out,
            rows=local_rows,
        )
        picked_edge_ids[stop_mask] = forced_continue
    policy_log_prob = policy_out.gather_log_prob(
        row_ids=rows,
        edge_ids=picked_edge_ids,
    ).float()
    return StepAction(
        row_ids=rows,
        edge_ids=picked_edge_ids,
        policy_log_prob=policy_log_prob,
        behavior_log_prob=policy_log_prob,
        forced=torch.zeros(rows.numel(), dtype=torch.bool, device=rows.device),
    )


def _sample_continue_edges(
    *,
    policy_out: PolicyOutput,
    rows: torch.Tensor,
) -> torch.Tensor:
    rows = rows.to(device=policy_out.stop_logit.device, dtype=torch.long).view(-1)
    row_has_edge = policy_out.has_edge().index_select(0, rows)
    if not bool(row_has_edge.all()):
        raise RuntimeError("always_continue_step received a row without legal edges.")

    local_edge_row_ids, kept_edge_ids, values = _select_rows_for_sampling(
        edge_log_prob=policy_out.edge_log_cond_prob,
        edge_row_ids=policy_out.edge_row_ids,
        edge_ids=policy_out.edge_ids,
        rows=rows,
        num_rows=int(policy_out.num_rows),
    )
    picked_positions = _segment_gumbel_argmax(
        values=values,
        row_ids=local_edge_row_ids,
        num_rows=int(rows.numel()),
    )
    return kept_edge_ids.index_select(0, picked_positions)


def _select_rows_for_sampling(
    *,
    edge_log_prob: torch.Tensor,
    edge_row_ids: torch.Tensor,
    edge_ids: torch.Tensor,
    rows: torch.Tensor,
    num_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_row_ids = edge_row_ids.to(device=rows.device, dtype=torch.long)
    edge_ids = edge_ids.to(device=rows.device, dtype=torch.long)
    row_mask = torch.zeros(int(num_rows), dtype=torch.bool, device=rows.device)
    row_mask[rows] = True
    keep = row_mask.index_select(0, edge_row_ids)

    kept_row_ids = edge_row_ids[keep]
    kept_edge_ids = edge_ids[keep]
    kept_values = edge_log_prob[keep].float()

    row_to_local = torch.full((int(num_rows),), -1, dtype=torch.long, device=rows.device)
    row_to_local[rows] = torch.arange(rows.numel(), dtype=torch.long, device=rows.device)
    local_edge_row_ids = row_to_local.index_select(0, kept_row_ids)
    return local_edge_row_ids, kept_edge_ids, kept_values


def _segment_gumbel_argmax(
    *,
    values: torch.Tensor,
    row_ids: torch.Tensor,
    num_rows: int,
) -> torch.Tensor:
    if values.numel() == 0:
        raise RuntimeError("Cannot sample continue edges from an empty frontier.")
    noise = -torch.log(-torch.log(torch.rand_like(values).clamp_min(torch.finfo(values.dtype).tiny)))
    perturbed = values + noise
    best_scores = torch.full(
        (int(num_rows),),
        -torch.inf,
        dtype=perturbed.dtype,
        device=perturbed.device,
    )
    best_scores.scatter_reduce_(0, row_ids, perturbed, reduce="amax", include_self=True)
    is_best = perturbed.eq(best_scores.index_select(0, row_ids))
    positions = torch.arange(values.numel(), dtype=torch.long, device=values.device)
    large = torch.full_like(positions, values.numel())
    first = torch.full((int(num_rows),), values.numel(), dtype=torch.long, device=values.device)
    first.scatter_reduce_(0, row_ids, torch.where(is_best, positions, large), reduce="amin", include_self=True)
    if bool(first.ge(values.numel()).any()):
        raise RuntimeError("Failed to sample an edge for some rows.")
    return first


class NoStopRolloutEngine(RolloutEngine):
    def sample_fused_rollouts(
        self,
        *,
        policy: ForwardPolicy,
        context: GraphContext,
        features: EncodedFeatures,
        rollouts_per_graph: int,
    ) -> RolloutResult:
        graph_ids = torch.arange(
            int(context.num_graphs),
            dtype=torch.long,
            device=context.device,
        ).repeat_interleave(int(rollouts_per_graph))
        state = State.initial(
            graph=context,
            graph_ids=graph_ids,
        )
        tape = RolloutTape(
            R=state.num_rows,
            T=self.expand_budget + 1,
            device=context.device,
        )

        for t in range(self.expand_budget + 1):
            active_rows = (~tape.is_stopped).nonzero(as_tuple=False).flatten()
            if active_rows.numel() == 0:
                break

            active_state = state.select_rows(active_rows)
            frontier = active_state.frontier(
                context,
                expand_budget=self.expand_budget,
            )
            policy_out = policy(
                features=features,
                state=active_state,
                context=context,
                frontier=frontier,
            )
            forced_local = forced_stop_rows(
                state=active_state,
                frontier=frontier,
                expand_budget=self.expand_budget,
            )
            sample_rows = rows_without_forced(
                num_rows=active_state.num_rows,
                forced_rows=forced_local,
                device=context.device,
            )
            actions: list[StepAction] = []
            if sample_rows.numel() > 0:
                actions.append(
                    always_continue_step(
                        policy_out=policy_out,
                        rows=sample_rows,
                    )
                )
            if forced_local.numel() > 0:
                actions.append(
                    StepAction.forced_stop(
                        rows=forced_local,
                        dtype=policy_out.stop_logit.dtype,
                        device=context.device,
                    )
                )
            sampled = StepAction.concat(actions)
            action = StepAction(
                row_ids=active_rows.index_select(0, sampled.row_ids),
                edge_ids=sampled.edge_ids,
                policy_log_prob=sampled.policy_log_prob,
                behavior_log_prob=sampled.behavior_log_prob,
                forced=sampled.forced,
            )
            tape.write(t, action)

            if bool(action.expand_mask.any()):
                state = state.expand(
                    graph=context,
                    rows=action.expand_rows,
                    edge_ids=action.expand_edge_ids,
                    expand_budget=self.expand_budget,
                )

        stop_step = tape.stop_step.clone()
        unstopped = stop_step.lt(0)
        if bool(unstopped.any()):
            stop_step[unstopped] = self.expand_budget

        return RolloutResult(
            source_graph_id=graph_ids,
            selected_edge_ids=tape.selected_edge_ids,
            policy_action_log_prob=tape.policy_action_log_prob,
            behavior_action_log_prob=tape.behavior_action_log_prob,
            stop_step=stop_step,
            forced_stop=tape.forced_stop,
            expand_budget=self.expand_budget,
        )


def build_rollouts(
    *,
    engine: RolloutEngine,
    policy: ForwardPolicy,
    context: GraphContext,
    features: EncodedFeatures,
    num_rollouts: int,
) -> tuple[RolloutResult, ...]:
    return tuple(
        engine.sample_rollouts(
            policy=policy,
            context=context,
            features=features,
            num_rollouts=int(num_rollouts),
        )
    )


def run_eval(
    *,
    model: Any,
    dataloader: Any,
    split: str,
    device: torch.device,
    limit_batches: int,
) -> dict[str, dict[str, float]]:
    default_total: dict[str, float] = {}
    default_weight: dict[str, float] = {}
    nostop_total: dict[str, float] = {}
    nostop_weight: dict[str, float] = {}

    expand_budget = int(model.runner.engine.expand_budget)
    eval_rollouts = int(model.runner.eval_num_rollouts)
    nostop_engine = NoStopRolloutEngine(expand_budget=expand_budget)

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches > 0 and batch_idx >= limit_batches:
            break

        batch = batch.to(device)
        graph = GraphContext.from_batch(batch)
        target = TargetContext.from_batch(
            batch=batch,
            graph_context=graph,
        )
        features = model.policy_feature_encoder(batch)

        with torch.no_grad():
            default_rollouts = build_rollouts(
                engine=model.runner.engine,
                policy=model.policy,
                context=graph,
                features=features,
                num_rollouts=eval_rollouts,
            )
            nostop_rollouts = build_rollouts(
                engine=nostop_engine,
                policy=model.policy,
                context=graph,
                features=features,
                num_rollouts=eval_rollouts,
            )

            default_metrics = evaluate_rollout_samples(
                rollout_samples=default_rollouts,
                batch=batch,
                exclude_anchors_from_retrieved=model.evaluation.exclude_anchors_from_retrieved,
                use_reachable_targets=model.evaluation.use_reachable_targets,
                k_windows=model.evaluation.k_windows,
                context=graph,
                features=features,
                reward_model=model.reward_model,
                target_context=target,
                policy=model.policy,
            )
            nostop_metrics = evaluate_rollout_samples(
                rollout_samples=nostop_rollouts,
                batch=batch,
                exclude_anchors_from_retrieved=model.evaluation.exclude_anchors_from_retrieved,
                use_reachable_targets=model.evaluation.use_reachable_targets,
                k_windows=model.evaluation.k_windows,
                context=graph,
                features=features,
                reward_model=model.reward_model,
                target_context=target,
                policy=model.policy,
            )

        weight = float(batch_num_graphs(batch))
        accumulate_metrics(default_total, default_weight, default_metrics, weight)
        accumulate_metrics(nostop_total, nostop_weight, nostop_metrics, weight)

    default_summary = scalar_weighted_mean(default_total, default_weight)
    nostop_summary = scalar_weighted_mean(nostop_total, nostop_weight)
    delta_summary = {
        key: nostop_summary.get(key, 0.0) - default_summary.get(key, 0.0)
        for key in sorted(set(default_summary) | set(nostop_summary))
    }
    return {
        "default": default_summary,
        "no_stop": nostop_summary,
        "delta": delta_summary,
        "meta": {
            "split": split,
            "eval_num_rollouts": float(eval_rollouts),
            "expand_budget": float(expand_budget),
            "limit_batches": float(limit_batches),
        },
    }


def select_dataloader(datamodule: Any, split: str) -> Any:
    if split == "validation":
        return datamodule.val_dataloader()
    if split == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unsupported split: {split}")


def print_focus(summary: dict[str, dict[str, float]]) -> None:
    focus_keys = [
        "sample/mean_recall",
        "sample/mean_edges",
        "sample/mean_log_reward",
        "candidate_reward_best@8/recall",
        "candidate_union@8/recall",
        "candidate_union@8/edges",
        "selector_traj_prob@8/f1",
        "selector_flow@8/f1",
        "selector_stop_flow@8/f1",
        "calibration/traj_prob_reward_spearman",
        "calibration/flow_reward_spearman",
        "calibration/stop_flow_reward_spearman",
        "stop/policy_stop_rate",
        "stop/forced_stop_rate",
        "stop/hit_then_continue_rate",
    ]
    for key in focus_keys:
        default = summary["default"].get(key, 0.0)
        nostop = summary["no_stop"].get(key, 0.0)
        delta = summary["delta"].get(key, 0.0)
        print(f"{key}: default={default:.6f} no_stop={nostop:.6f} delta={delta:+.6f}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))

    cfg = compose_config(args)
    datamodule = build_datamodule(cfg)
    resources = setup_datamodule(
        datamodule,
        stage="validate" if args.split == "validation" else "test",
    )
    model = build_model(cfg, resources)

    missing, unexpected = load_checkpoint_weights(model, args.ckpt, strict=False)
    print(
        f"loaded ckpt={args.ckpt} missing={missing} unexpected={unexpected}",
        flush=True,
    )

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print(f"requested device {requested_device} unavailable; falling back to cpu", flush=True)
        device = torch.device("cpu")
    else:
        device = requested_device
    model.to(device).eval()

    dataloader = select_dataloader(datamodule, args.split)
    summary = run_eval(
        model=model,
        dataloader=dataloader,
        split=args.split,
        device=device,
        limit_batches=int(args.limit_batches),
    )

    print_focus(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
