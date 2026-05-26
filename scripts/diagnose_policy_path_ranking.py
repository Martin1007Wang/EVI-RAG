from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.replay import WeakReplaySource
from src.weaver.state import StateBatch


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_path = args.ckpt.resolve() if args.ckpt is not None else run_dir / "checkpoints" / "last.ckpt"

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
    missing, unexpected = load_checkpoint_weights(model, str(ckpt_path), strict=False)
    model.eval()

    stats: dict[str, list[float]] = defaultdict(list)
    weak_source = WeakReplaySource(
        budget=int(model.budget),
        states_per_graph=int(args.states_per_graph),
        branch_per_state=int(args.branch_per_state),
    )

    end = len(dataset) if args.max_samples <= 0 else min(len(dataset), int(args.max_samples))
    with torch.no_grad():
        for start in range(0, end, int(args.batch_size)):
            samples = [dataset[idx] for idx in range(start, min(end, start + int(args.batch_size)))]
            batch = datamodule.collator(samples)
            graph = GraphContext.from_batch(batch)
            target = TargetContext.from_batch(batch=batch, graph_context=graph)
            features = model.policy_feature_encoder(batch)

            initial = StateBatch.initial(
                graph_ids=torch.arange(int(graph.num_graphs), dtype=torch.long, device=graph.device),
                budget=int(model.budget),
            )
            collect_view(
                model=model,
                features=features,
                graph=graph,
                target=target,
                state=initial,
                prefix="initial",
                stats=stats,
            )

            weak = weak_source.sample(graph=graph, target=target)
            if weak.num_states > 0:
                add(stats, "oracle_prefix/state_count_per_batch", weak.num_states)
                collect_view(
                    model=model,
                    features=features,
                    graph=graph,
                    target=target,
                    state=weak.state,
                    prefix="oracle_prefix",
                    stats=stats,
                )

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"checkpoint={ckpt_path}")
    print(f"epoch={checkpoint.get('epoch')} step={checkpoint.get('global_step')}")
    print(f"missing={missing} unexpected={unexpected}")
    print(f"split={args.split} samples={end}")
    for key in summary_keys():
        print(f"{key:44s} {summarize(stats.get(key, []))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank weak shortest-path frontier edges under semantic, residual, and final policy scores.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--states-per-graph", type=int, default=8)
    parser.add_argument("--branch-per-state", type=int, default=2)
    return parser.parse_args()


def load_run_config(*, run_dir: Path, data_dir: str):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open_dict(cfg):
        cfg.paths.data_dir = str(data_dir)

        # Older debug runs used precomputed replay keys. The current runner uses
        # weak replay prefix states, so make archived configs loadable for diagnosis.
        runner = cfg.model.runner
        if "replay_source" in runner:
            runner.weak_replay_source = runner.pop("replay_source")
        if "train_replay_rollouts" in runner:
            runner.pop("train_replay_rollouts")
        if "weak_replay_loss" not in cfg.model:
            cfg.model.weak_replay_loss = None
    return cfg


def collect_view(
    *,
    model,
    features,
    graph: GraphContext,
    target: TargetContext,
    state: StateBatch,
    prefix: str,
    stats: dict[str, list[float]],
) -> None:
    action_space = state.action_space(graph)
    policy_out = model.policy(
        features=features,
        state=state,
        context=graph,
        action_space=action_space,
    )
    stop_flow, semantic, residual = score_components(
        model=model,
        features=features,
        graph=graph,
        state=state,
        action_space=action_space,
    )

    for value in action_space.expand_count.cpu().tolist():
        add(stats, f"{prefix}/frontier", value)
    for value in policy_out.stop_log_prob.exp().detach().cpu().tolist():
        add(stats, f"{prefix}/stop_prob", value)

    if action_space.num_expansions <= 0:
        return

    positive_all = target.shortest_path_edge_mask.index_select(
        0,
        action_space.expand_edge_ids,
    ).cpu()
    raw = policy_out.edge_raw_score.detach().cpu()
    action_prob = policy_out.edge_log_prob.detach().cpu().exp()
    semantic = semantic.detach().cpu()
    residual = residual.detach().cpu()
    stop_flow = stop_flow.detach().cpu()
    ptr = action_space.expand_ptr.detach().cpu()

    for row in range(int(state.num_states)):
        start = int(ptr[row].item())
        end = int(ptr[row + 1].item())
        if end <= start:
            continue

        positive = positive_all[start:end]
        add(stats, f"{prefix}/positive_count", int(positive.sum().item()))
        if not bool(positive.any()):
            continue

        raw_scores = raw[start:end]
        semantic_scores = semantic[start:end]
        residual_scores = residual[start:end]
        for name, scores in (
            ("raw", raw_scores),
            ("semantic", semantic_scores),
            ("residual", residual_scores),
        ):
            rank, top1 = best_positive_rank(scores=scores, positive=positive)
            add(stats, f"{prefix}/{name}_rank", rank)
            add(stats, f"{prefix}/{name}_top1", 1.0 if top1 else 0.0)

        add(stats, f"{prefix}/positive_action_mass", float(action_prob[start:end][positive].sum().item()))
        add(stats, f"{prefix}/best_positive_action_prob", float(action_prob[start:end][positive].max().item()))

        negative = ~positive
        if bool(negative.any()):
            add(stats, f"{prefix}/raw_gap", float(raw_scores[positive].mean().item() - raw_scores[negative].mean().item()))
            add(stats, f"{prefix}/semantic_gap", float(semantic_scores[positive].mean().item() - semantic_scores[negative].mean().item()))
            add(stats, f"{prefix}/residual_gap", float(residual_scores[positive].mean().item() - residual_scores[negative].mean().item()))

        frontier_size = float(end - start)
        normalized_edge_flow = raw_scores - math.log(frontier_size)
        normalized_continue_flow = torch.logsumexp(normalized_edge_flow, dim=0)
        normalized_state_flow = torch.logaddexp(stop_flow[row], normalized_continue_flow)
        add(stats, f"{prefix}/stop_prob_beta1_posthoc", float(torch.exp(stop_flow[row] - normalized_state_flow).item()))


def score_components(
    *,
    model,
    features,
    graph: GraphContext,
    state: StateBatch,
    action_space,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_h = model.policy.state_encoder.query_embeddings(features=features, state=state)
    selected_h = model.policy.state_encoder.selected_edge_summary(
        features=features,
        state=state,
        context=graph,
        query_h=query_h,
    )
    covered_h = model.policy.state_encoder.covered_node_summary(
        features=features,
        state=state,
        context=graph,
        query_h=query_h,
    )
    budget_h = model.policy.encode_budget(state)
    stop_flow = model.policy.score_stop_flow(
        query_h=query_h,
        selected_h=selected_h,
        covered_h=covered_h,
        budget_h=budget_h,
    )

    if action_space.num_expansions <= 0:
        empty = query_h.new_empty((0,), dtype=torch.float32)
        return stop_flow.float(), empty, empty

    edge_h = model.policy.state_encoder.encode_edge_tokens(
        features=features,
        context=graph,
        edge_ids=action_space.expand_edge_ids,
    )
    semantic = model.policy.score_edge_semantic_prior(
        features=features,
        context=graph,
        query_h=query_h,
        action_space=action_space,
    )
    residual = model.policy.score_edge_residual(
        query_h=query_h,
        selected_h=selected_h,
        covered_h=covered_h,
        budget_h=budget_h,
        edge_h=edge_h,
        row_ids=action_space.expand_state_ids,
    )
    return stop_flow.float(), semantic.float(), residual.float()


def best_positive_rank(*, scores: torch.Tensor, positive: torch.Tensor) -> tuple[int, bool]:
    best_positive = scores[positive].max()
    rank = int(scores.gt(best_positive).sum().item()) + 1
    top1 = bool(positive[int(torch.argmax(scores).item())].item())
    return rank, top1


def add(stats: dict[str, list[float]], key: str, value: float | int | torch.Tensor) -> None:
    if isinstance(value, torch.Tensor):
        value = float(value.detach().cpu().item())
    value = float(value)
    if math.isfinite(value):
        stats[key].append(value)


def summarize(values: list[float]) -> str:
    if not values:
        return "n=0"
    tensor = torch.tensor(values, dtype=torch.float32)
    return (
        f"n={len(values)} "
        f"mean={float(tensor.mean()):.6g} "
        f"p50={float(tensor.quantile(0.50)):.6g} "
        f"p90={float(tensor.quantile(0.90)):.6g} "
        f"max={float(tensor.max()):.6g}"
    )


def summary_keys() -> tuple[str, ...]:
    return (
        "initial/frontier",
        "initial/positive_count",
        "initial/stop_prob",
        "initial/stop_prob_beta1_posthoc",
        "initial/semantic_rank",
        "initial/semantic_top1",
        "initial/residual_rank",
        "initial/residual_top1",
        "initial/raw_rank",
        "initial/raw_top1",
        "initial/positive_action_mass",
        "initial/best_positive_action_prob",
        "initial/semantic_gap",
        "initial/residual_gap",
        "initial/raw_gap",
        "oracle_prefix/state_count_per_batch",
        "oracle_prefix/frontier",
        "oracle_prefix/positive_count",
        "oracle_prefix/stop_prob",
        "oracle_prefix/stop_prob_beta1_posthoc",
        "oracle_prefix/semantic_rank",
        "oracle_prefix/semantic_top1",
        "oracle_prefix/residual_rank",
        "oracle_prefix/residual_top1",
        "oracle_prefix/raw_rank",
        "oracle_prefix/raw_top1",
        "oracle_prefix/positive_action_mass",
        "oracle_prefix/best_positive_action_prob",
        "oracle_prefix/semantic_gap",
        "oracle_prefix/residual_gap",
        "oracle_prefix/raw_gap",
    )


if __name__ == "__main__":
    main()
