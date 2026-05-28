from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.retrieval import mean_over_valid_graphs
from src.training.checkpoint import load_checkpoint_weights
from src.training.factory import build_model, prepare_training_components
from src.training.optimization import build_optimizer
from src.weaver.context import GraphContext, TargetContext
from src.weaver.objectives.edge_flow_matching import (
    build_subtb_input,
    expansion_event_residual,
    terminal_event_residual,
)
from src.weaver.policy import STOP_EDGE_ID
from src.weaver.rollout.replay import (
    replay_trajectories_with_stats,
    training_from_trajectories,
)
from src.weaver.state import StateOps, dense_active_node_mask_debug
from src.weaver.transition import SRC_REPLAY, TrainingBatch


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Probe C: train only on a fixed oracle/replay transition set. " "No online rollout or policy-sampled transitions are used.")
    )
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--split", default="validation", choices=("validation", "val", "train"))
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-trajectories-per-graph", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    cfg = _compose_cfg(args.data_dir)
    dm, resources = prepare_training_components(cfg, stage="fit")
    model = build_model(cfg, resources).to(args.device)
    if args.ckpt:
        missing, unexpected = load_checkpoint_weights(model, args.ckpt, strict=False)
        print(f"loaded_ckpt path={args.ckpt} " f"missing={len(missing)} unexpected={len(unexpected)}")

    batch = _fixed_batch(
        dm=dm,
        split=args.split,
        start_idx=int(args.start_idx),
        num_samples=int(args.num_samples),
    ).to(args.device)
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)

    training, replay_stats = _fixed_training(
        model=model,
        batch=batch,
        graph=graph,
        target=target,
        max_trajectories_per_graph=int(args.max_trajectories_per_graph),
    )
    if training.num_items <= 0:
        raise RuntimeError("fixed oracle/replay transition set is empty.")

    print(
        "fixed_probe "
        f"split={args.split} start_idx={args.start_idx} "
        f"samples={args.num_samples} device={args.device} "
        f"graphs={graph.num_graphs} nodes={graph.num_nodes} edges={graph.num_edges}"
    )
    print(
        "fixed_transitions "
        f"expansions={training.num_expansions} terminals={training.num_terminals} "
        f"eligible_graphs={replay_stats.eligible_graphs} "
        f"covered_graphs={replay_stats.covered_graphs} "
        f"trajectories={replay_stats.generated_trajectories} "
        f"oracle_reward_mean={replay_stats.oracle_reward_mean:.6f}"
    )
    print("columns epoch loss delta_expand delta_stop " "terminal_flow_best_recall oracle_terminal_best_recall")

    model.train()
    model.policy_objective.alpha_replay = 1.0
    optimizer_cfg = replace(
        model.optimization.optimizer,
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    optimizer = build_optimizer(
        modules=(model.policy_feature_encoder, model.policy),
        cfg=optimizer_cfg,
    )

    _print_metrics(epoch=0, model=model, batch=batch, graph=graph, target=target, training=training)
    for epoch in range(1, int(args.epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        features = model.policy_feature_encoder(batch)
        output = model.policy_step_output(
            graph=graph,
            target=target,
            policy_features=features,
            training=training,
        )
        output.loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            _print_metrics(
                epoch=epoch,
                model=model,
                batch=batch,
                graph=graph,
                target=target,
                training=training,
            )


def _compose_cfg(data_dir: str):
    config_dir = str((Path.cwd() / "configs").resolve())
    overrides = [
        "experiment=train/webqsp",
        "logger=none",
        "trainer=cpu",
        "datamodule.num_workers=0",
        "datamodule.eval_num_workers=0",
        "datamodule.train_shuffle=false",
        f"paths.data_dir={data_dir}",
        "model.optimization.scheduler.type=none",
        "model.policy_objective.alpha_replay=1.0",
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="train", overrides=overrides)


def _fixed_batch(*, dm, split: str, start_idx: int, num_samples: int):
    dataset = {
        "train": dm.train_dataset,
        "validation": dm.val_dataset,
        "val": dm.val_dataset,
    }[split]
    end_idx = min(int(start_idx) + int(num_samples), len(dataset))
    samples = [dataset[idx] for idx in range(int(start_idx), end_idx)]
    if not samples:
        raise ValueError("selected fixed split slice is empty.")
    return dm.collator(samples)


@torch.no_grad()
def _fixed_training(
    *,
    model,
    batch,
    graph: GraphContext,
    target: TargetContext,
    max_trajectories_per_graph: int,
) -> tuple[TrainingBatch, object]:
    trajectories, stats = replay_trajectories_with_stats(
        batch=batch,
        context=graph,
        budget=int(model.budget),
        max_trajectories_per_graph=int(max_trajectories_per_graph),
        rollouts=(),
        reward_model=model.reward_model,
        target_context=target,
        allow_reward_skip=False,
    )
    training = training_from_trajectories(
        trajectories=trajectories,
        graph=graph,
        budget=int(model.budget),
    ).with_source_id(SRC_REPLAY)
    return training, stats


@torch.no_grad()
def _print_metrics(
    *,
    epoch: int,
    model,
    batch,
    graph: GraphContext,
    target: TargetContext,
    training: TrainingBatch,
) -> None:
    model.eval()
    features = model.policy_feature_encoder(batch)
    output = model.policy_step_output(
        graph=graph,
        target=target,
        policy_features=features,
        training=training,
    )
    delta_expand, delta_stop = _single_step_deltas(
        model=model,
        graph=graph,
        target=target,
        features=features,
        training=training,
    )
    terminal_best, oracle_best = _fixed_terminal_selector_recall(
        model=model,
        batch=batch,
        graph=graph,
        features=features,
        training=training,
    )
    print(
        f"epoch={epoch} "
        f"loss={float(output.loss.item()):.6f} "
        f"delta_expand={delta_expand:.6f} "
        f"delta_stop={delta_stop:.6f} "
        f"terminal_flow_best_recall={terminal_best:.6f} "
        f"oracle_terminal_best_recall={oracle_best:.6f}",
        flush=True,
    )
    model.train()


def _single_step_deltas(
    *,
    model,
    graph: GraphContext,
    target: TargetContext,
    features,
    training: TrainingBatch,
) -> tuple[float, float]:
    expansions = training.expansions
    terminals = training.terminals
    parent_frontier = StateOps.frontier(expansions.parent, graph)
    child_frontier = StateOps.frontier(expansions.child, graph)
    terminal_frontier = StateOps.frontier(terminals.state, graph)
    parent_out, child_out, terminal_out = model_module_batched_outputs(
        model=model,
        features=features,
        graph=graph,
        states=(expansions.parent, expansions.child, terminals.state),
        frontiers=(parent_frontier, child_frontier, terminal_frontier),
    )
    backward_log_prob = model_module_backward_log_prob(
        model=model,
        graph=graph,
        expansions=expansions,
    )
    reward_out = model.reward_model(
        state=terminals.state,
        graph_context=graph,
        target_context=target,
    )
    subtb_input = build_subtb_input(
        parent_out=parent_out,
        child_out=child_out,
        terminal_out=terminal_out,
        reward_out=reward_out,
        backward_log_prob=backward_log_prob,
        expansions=expansions,
        terminals=terminals,
    )
    events = subtb_input.events
    expand_mask = ~events.is_stop
    stop_mask = events.is_stop
    exp = expansion_event_residual(events).abs()
    stop = terminal_event_residual(events).abs()
    delta_expand = float(exp[expand_mask].mean().item()) if bool(expand_mask.any()) else 0.0
    delta_stop = float(stop[stop_mask].mean().item()) if bool(stop_mask.any()) else 0.0
    return delta_expand, delta_stop


def model_module_batched_outputs(*, model, features, graph, states, frontiers):
    from src.weaver.module import batched_policy_outputs

    return batched_policy_outputs(
        policy=model.policy,
        features=features,
        context=graph,
        states=states,
        frontiers=frontiers,
    )


def model_module_backward_log_prob(*, model, graph: GraphContext, expansions):
    from src.weaver.module import backward_action_log_prob

    return backward_action_log_prob(
        backward_policy=model.backward_policy,
        parent_state=expansions.parent,
        child_state=expansions.child,
        context=graph,
        action_edge_ids=expansions.edge_ids,
        budget=int(model.budget),
    )


def _fixed_terminal_selector_recall(
    *,
    model,
    batch,
    graph: GraphContext,
    features,
    training: TrainingBatch,
) -> tuple[float, float]:
    terminals = training.terminals
    if terminals.num_items <= 0:
        return 0.0, 0.0

    node_masks = dense_active_node_mask_debug(terminals.state, graph).to(
        device=torch.device("cpu"),
        dtype=torch.bool,
    )
    graph_ids = terminals.state.graph_ids.to(device=torch.device("cpu"), dtype=torch.long)
    recall = _candidate_recall(
        node_masks=node_masks,
        graph_ids=graph_ids,
        batch=batch,
    )
    valid_graph_mask = _valid_graph_mask(batch=batch).to(device=torch.device("cpu"))

    frontier = StateOps.frontier(terminals.state, graph)
    out = model.policy(
        features=features,
        state=terminals.state,
        context=graph,
        frontier=frontier,
    )
    rows = torch.arange(terminals.num_items, dtype=torch.long, device=out.row_ids.device)
    stop_edges = torch.full(
        (terminals.num_items,),
        int(STOP_EDGE_ID),
        dtype=torch.long,
        device=out.row_ids.device,
    )
    stop_flow = out.gather_action_log_flow(row_ids=rows, edge_ids=stop_edges).detach().cpu()
    selected = _best_by_graph(values=recall, scores=stop_flow, graph_ids=graph_ids, num_graphs=int(graph.num_graphs))
    oracle = _best_by_graph(values=recall, scores=recall, graph_ids=graph_ids, num_graphs=int(graph.num_graphs))
    return (
        mean_over_valid_graphs(selected.view(1, -1), valid_graph_mask),
        mean_over_valid_graphs(oracle.view(1, -1), valid_graph_mask),
    )


def _candidate_recall(*, node_masks: torch.Tensor, graph_ids: torch.Tensor, batch) -> torch.Tensor:
    targets = batch.reachable_target_node_ids.to(device=torch.device("cpu"), dtype=torch.long)
    node_batch = batch.batch.to(device=torch.device("cpu"), dtype=torch.long)
    anchors = batch.anchor_node_ids.to(device=torch.device("cpu"), dtype=torch.long)

    target_mask = torch.zeros(int(batch.num_nodes), dtype=torch.bool)
    if targets.numel() > 0:
        target_mask[targets] = True
    anchor_mask = torch.zeros(int(batch.num_nodes), dtype=torch.bool)
    if anchors.numel() > 0:
        anchor_mask[anchors] = True

    out = torch.zeros(int(node_masks.size(0)), dtype=torch.float32)
    for row in range(int(node_masks.size(0))):
        graph_id = int(graph_ids[row].item())
        graph_nodes = node_batch.eq(graph_id)
        gold = target_mask & graph_nodes & ~anchor_mask
        denom = int(gold.sum().item())
        if denom <= 0:
            continue
        hits = (node_masks[row] & gold).sum().float()
        out[row] = hits / float(denom)
    return out


def _valid_graph_mask(*, batch) -> torch.Tensor:
    targets = batch.reachable_target_node_ids.to(device=torch.device("cpu"), dtype=torch.long)
    node_batch = batch.batch.to(device=torch.device("cpu"), dtype=torch.long)
    valid = torch.zeros(int(batch.num_graphs), dtype=torch.bool)
    if targets.numel() > 0:
        valid.index_fill_(0, node_batch.index_select(0, targets), True)
    return valid


def _best_by_graph(
    *,
    values: torch.Tensor,
    scores: torch.Tensor,
    graph_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    out = torch.zeros(int(num_graphs), dtype=torch.float32)
    best = torch.full((int(num_graphs),), float("-inf"), dtype=torch.float32)
    for row in range(int(values.numel())):
        graph_id = int(graph_ids[row].item())
        score = float(scores[row].item())
        if score > float(best[graph_id].item()):
            best[graph_id] = score
            out[graph_id] = values[row]
    return out


if __name__ == "__main__":
    main()
