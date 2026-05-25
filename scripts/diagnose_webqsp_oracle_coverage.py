from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.factory import prepare_training_components
from src.weaver.context import GraphContext
from src.weaver.rollout.replay import (
    build_replay_graph_label_views,
    enumerate_replay_state_dag,
    initial_state_for_graph_ids,
    precomputed_replay_trajectories,
    replay_graph_ids,
    training_from_trajectories,
)
from src.weaver.state import State, StateOps, dense_active_node_mask_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare WebQSP graph, full-enumeration, and fixed replay oracle recall ceilings.")
    parser.add_argument("--data-dir", default="/mnt/data/retrieval")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "val", "test"))
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=0, help="0 means the full split.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument("--max-replay-trajectories", type=int, default=0, help="0 means use all precomputed trajectories.")
    parser.add_argument("--max-full-states-per-graph", type=int, default=250000)
    parser.add_argument("--skip-dag", action="store_true")
    parser.add_argument("--skip-frontier-full", action="store_true")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    cfg = _compose_cfg(args.data_dir)
    dm, _ = prepare_training_components(cfg, stage="fit")
    if args.split == "test" and dm.test_dataset is None:
        dm.setup("test")
    dataset = {
        "train": dm.train_dataset,
        "validation": dm.val_dataset,
        "val": dm.val_dataset,
        "test": dm.test_dataset,
    }[args.split]

    end = len(dataset) if int(args.num_samples) <= 0 else min(len(dataset), int(args.start_idx) + int(args.num_samples))
    indices = list(range(int(args.start_idx), end))
    if not indices:
        raise ValueError("selected split slice is empty.")

    totals = Counter()
    sums = Counter()
    examples: list[dict[str, object]] = []
    full_state_counts: list[int] = []

    for offset in range(0, len(indices), int(args.batch_size)):
        rows = indices[offset : offset + int(args.batch_size)]
        batch = dm.collator([dataset[idx] for idx in rows])
        graph = GraphContext.from_batch(batch)
        batch_stats, batch_examples, state_counts = diagnose_batch(
            batch=batch,
            graph=graph,
            budget=int(args.budget),
            max_replay_trajectories=None if int(args.max_replay_trajectories) <= 0 else int(args.max_replay_trajectories),
            max_full_states_per_graph=int(args.max_full_states_per_graph),
            skip_dag=bool(args.skip_dag),
            skip_frontier_full=bool(args.skip_frontier_full),
            global_indices=rows,
        )
        totals.update(batch_stats["totals"])
        sums.update(batch_stats["sums"])
        full_state_counts.extend(state_counts)
        remaining = max(0, int(args.examples) - len(examples))
        if remaining:
            examples.extend(batch_examples[:remaining])

    valid = max(1, totals["valid_graphs"])
    all_graphs = max(1, totals["graphs"])
    print(f"split={args.split} start_idx={args.start_idx} samples={len(indices)} budget={args.budget}")
    print(f"graphs={totals['graphs']} valid_graphs={totals['valid_graphs']}")
    print(f"answer_in_graph_rate={sums['answer_in_graph_graphs'] / all_graphs:.6f}")
    print(f"reachable_graph_rate={sums['reachable_graphs'] / all_graphs:.6f}")
    print(f"graph_target_node_best_recall={sums['graph_target_node_best_recall'] / valid:.6f}")
    print(f"reachable_target_best_recall={sums['reachable_target_best_recall'] / valid:.6f}")
    print(f"admissible_dag_oracle_best_recall={sums['dag_best_recall'] / valid:.6f}")
    print(f"full_enumeration_oracle_best_recall={sums['full_best_recall'] / valid:.6f}")
    print(f"fixed_transition_oracle_terminal_best_recall={sums['fixed_best_recall'] / valid:.6f}")
    print(f"fixed_minus_full={sums['fixed_best_recall'] / valid - sums['full_best_recall'] / valid:.6f}")
    print(f"precomputed_replay_covered_graph_rate={sums['fixed_has_terminal'] / valid:.6f}")
    print(f"full_enum_truncated_graphs={totals['full_truncated_graphs']}")
    if full_state_counts:
        counts = torch.tensor(full_state_counts, dtype=torch.float32)
        print(
            "full_enum_states_per_graph "
            f"mean={float(counts.mean()):.2f} "
            f"p50={float(counts.quantile(0.50)):.0f} "
            f"p90={float(counts.quantile(0.90)):.0f} "
            f"max={int(counts.max().item())}"
        )
    if examples:
        print("examples_with_gap:")
        for item in examples:
            print(item)


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
    ]
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="train", overrides=overrides)


def diagnose_batch(
    *,
    batch,
    graph: GraphContext,
    budget: int,
    max_replay_trajectories: int | None,
    max_full_states_per_graph: int,
    skip_dag: bool,
    skip_frontier_full: bool,
    global_indices: list[int],
) -> tuple[dict[str, Counter], list[dict[str, object]], list[int]]:
    targets = batch.reachable_target_node_ids.to(dtype=torch.long).view(-1)
    target_graph = graph.node_to_graph.index_select(0, targets) if targets.numel() else targets
    eligible_graphs = replay_graph_ids(targets=targets, target_graph=target_graph, context=graph)
    fixed, _ = precomputed_replay_trajectories(
        batch=batch,
        context=graph,
        eligible_graphs=eligible_graphs,
        budget=int(budget),
        max_trajectories_per_graph=max_replay_trajectories,
    ) or ([], None)
    fixed_training = training_from_trajectories(trajectories=fixed, graph=graph, budget=int(budget))

    fixed_best = best_recall_by_graph(batch=batch, graph=graph, states=fixed_training.terminals.state)
    full_best = torch.zeros(int(graph.num_graphs), dtype=torch.float32)
    dag_best = torch.zeros(int(graph.num_graphs), dtype=torch.float32)
    state_counts: list[int] = []
    truncated: set[int] = set()

    graph_views = {
        int(view.graph_id): view
        for view in build_replay_graph_label_views(
            batch=batch,
            context=graph,
            targets=targets,
            target_graph=target_graph,
        )
    }
    for graph_id in range(int(graph.num_graphs)):
        view = graph_views.get(graph_id)
        if view is not None and not skip_dag:
            dag_states = [
                node.state
                for node in enumerate_replay_state_dag(
                    context=graph,
                    graph_view=view,
                    budget=int(budget),
                ).values()
            ]
            if dag_states:
                dag_best[graph_id] = best_state_recall(
                    batch=batch,
                    graph=graph,
                    states=dag_states,
                    graph_id=graph_id,
                )
        if skip_frontier_full:
            state_counts.append(0)
            full_best[graph_id] = dag_best[graph_id]
        else:
            states, is_truncated = enumerate_all_frontier_states(
                graph=graph,
                graph_id=graph_id,
                budget=int(budget),
                max_states=int(max_full_states_per_graph),
            )
            state_counts.append(len(states))
            if is_truncated:
                truncated.add(graph_id)
            if states:
                full_best[graph_id] = best_state_recall(batch=batch, graph=graph, states=states, graph_id=graph_id)

    valid_mask = valid_graph_mask(batch=batch, graph=graph)
    graph_target_recall = graph_target_node_recall(batch=batch, graph=graph, use_reachable=False)
    reachable_recall = graph_target_node_recall(batch=batch, graph=graph, use_reachable=True)

    totals = Counter()
    sums = Counter()
    examples: list[dict[str, object]] = []
    for graph_id in range(int(graph.num_graphs)):
        totals["graphs"] += 1
        if bool(valid_mask[graph_id].item()):
            totals["valid_graphs"] += 1
            sums["graph_target_node_best_recall"] += float(graph_target_recall[graph_id].item())
            sums["reachable_target_best_recall"] += float(reachable_recall[graph_id].item())
            sums["dag_best_recall"] += float(dag_best[graph_id].item())
            sums["full_best_recall"] += float(full_best[graph_id].item())
            sums["fixed_best_recall"] += float(fixed_best[graph_id].item())
            if float(fixed_best[graph_id].item()) > 0.0:
                sums["fixed_has_terminal"] += 1.0
        if float(graph_target_recall[graph_id].item()) > 0.0:
            sums["answer_in_graph_graphs"] += 1.0
        if float(reachable_recall[graph_id].item()) > 0.0:
            sums["reachable_graphs"] += 1.0
        if graph_id in truncated:
            totals["full_truncated_graphs"] += 1
        gap = float(full_best[graph_id].item() - fixed_best[graph_id].item())
        gap = max(gap, float(dag_best[graph_id].item() - fixed_best[graph_id].item()))
        if gap > 1.0e-6:
            examples.append(
                {
                    "global_idx": int(global_indices[graph_id]),
                    "graph_id": int(graph_id),
                    "target_count": int(target_count(batch=batch, graph=graph, graph_id=graph_id)),
                    "dag": round(float(dag_best[graph_id].item()), 6),
                    "full": round(float(full_best[graph_id].item()), 6),
                    "fixed": round(float(fixed_best[graph_id].item()), 6),
                    "states": int(state_counts[graph_id]),
                    "admissible_edges": int(graph_views[graph_id].admissible_edge_ids.numel()) if graph_id in graph_views else 0,
                    "fixed_trajectories": int(sum(1 for t in fixed if t.graph_id == graph_id)),
                }
            )
    return {"totals": totals, "sums": sums}, examples, state_counts


def enumerate_all_frontier_states(
    *,
    graph: GraphContext,
    graph_id: int,
    budget: int,
    max_states: int,
) -> tuple[list[State], bool]:
    initial = initial_state_for_graph_ids(
        context=graph,
        graph_ids=torch.tensor([int(graph_id)], dtype=torch.long, device=graph.device),
        budget=int(budget),
    )
    states: dict[tuple[int, ...], State] = {(): initial}
    current = [()]
    truncated = False
    for _ in range(int(budget)):
        next_layer: list[tuple[int, ...]] = []
        for key in current:
            parent = states[key]
            frontier = StateOps.frontier(parent, graph)
            edge_ids = frontier.edge_ids[frontier.row_ids.eq(0)]
            for edge_id in edge_ids.tolist():
                child = StateOps.expand(
                    parent,
                    graph,
                    rows=torch.zeros(1, dtype=torch.long, device=graph.device),
                    edge_ids=torch.tensor([int(edge_id)], dtype=torch.long, device=graph.device),
                )
                child_key = tuple(int(x) for x in torch.sort(child.selected_edges()[1]).values.tolist())
                if child_key not in states:
                    states[child_key] = child
                    next_layer.append(child_key)
                    if len(states) >= int(max_states):
                        truncated = True
                        return list(states.values()), truncated
        current = next_layer
    return list(states.values()), truncated


def best_recall_by_graph(*, batch, graph: GraphContext, states: State) -> torch.Tensor:
    out = torch.zeros(int(graph.num_graphs), dtype=torch.float32)
    if states.num_rows == 0:
        return out
    masks = dense_active_node_mask_debug(states, graph).cpu()
    for row in range(states.num_rows):
        graph_id = int(states.graph_ids[row].item())
        recall = state_recall(batch=batch, graph=graph, node_mask=masks[row], graph_id=graph_id)
        out[graph_id] = max(float(out[graph_id].item()), recall)
    return out


def best_state_recall(*, batch, graph: GraphContext, states: list[State], graph_id: int) -> float:
    merged = StateOps.concat(states)
    masks = dense_active_node_mask_debug(merged, graph).cpu()
    best = 0.0
    for row in range(merged.num_rows):
        best = max(best, state_recall(batch=batch, graph=graph, node_mask=masks[row], graph_id=graph_id))
    return best


def state_recall(*, batch, graph: GraphContext, node_mask: torch.Tensor, graph_id: int) -> float:
    del graph
    targets = batch.reachable_target_node_ids.cpu().long().view(-1)
    anchors = batch.anchor_node_ids.cpu().long().view(-1)
    node_batch = batch.batch.cpu().long().view(-1)
    graph_nodes = node_batch.eq(int(graph_id))
    target_mask = torch.zeros(int(batch.num_nodes), dtype=torch.bool)
    anchor_mask = torch.zeros(int(batch.num_nodes), dtype=torch.bool)
    if targets.numel():
        target_mask[targets] = True
    if anchors.numel():
        anchor_mask[anchors] = True
    gold = target_mask & graph_nodes & ~anchor_mask
    denom = int(gold.sum().item())
    if denom <= 0:
        return 0.0
    return float((node_mask & gold).sum().item()) / float(denom)


def graph_target_node_recall(*, batch, graph: GraphContext, use_reachable: bool) -> torch.Tensor:
    del graph
    node_ids = (batch.reachable_target_node_ids if use_reachable else batch.target_node_ids).cpu().long().view(-1)
    node_batch = batch.batch.cpu().long().view(-1)
    anchors = batch.anchor_node_ids.cpu().long().view(-1)
    anchor_mask = torch.zeros(int(batch.num_nodes), dtype=torch.bool)
    if anchors.numel():
        anchor_mask[anchors] = True
    out = torch.zeros(int(batch.num_graphs), dtype=torch.float32)
    denom = torch.zeros(int(batch.num_graphs), dtype=torch.float32)
    for node_id in node_ids.tolist():
        if bool(anchor_mask[int(node_id)].item()):
            continue
        graph_id = int(node_batch[int(node_id)].item())
        denom[graph_id] += 1.0
        out[graph_id] += 1.0
    return torch.where(denom.gt(0), out / denom.clamp_min(1.0), out)


def valid_graph_mask(*, batch, graph: GraphContext) -> torch.Tensor:
    return graph_target_node_recall(batch=batch, graph=graph, use_reachable=True).gt(0)


def target_count(*, batch, graph: GraphContext, graph_id: int) -> int:
    del graph
    targets = batch.reachable_target_node_ids.cpu().long().view(-1)
    node_batch = batch.batch.cpu().long().view(-1)
    return int(node_batch.index_select(0, targets).eq(int(graph_id)).sum().item()) if targets.numel() else 0


if __name__ == "__main__":
    main()
