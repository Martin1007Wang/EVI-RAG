from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.artifacts import load_materialization_artifact
from src.data.collate import RetrievalCollator
from src.data.dataset import RetrievalDataset
from src.graph.paths import compute_replay_path_candidates
from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import (
    State,
    StateOps,
    active_node_count,
    answer_count_from_active_nodes,
    active_nodes,
)
from src.weaver.utility import Reward

DEFAULT_METADATA_DIR = Path("/mnt/data/retrieval/webqsp/metadata")
DEFAULT_REPORT_LIMIT = 10
DEFAULT_TRAIN_BUDGET = 3


@dataclass(frozen=True, slots=True)
class StateSummary:
    log_reward: float
    answer_count: int
    active_count: int
    selected_edge_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SampleDiagnosis:
    sample_id: str
    path_length: int
    oracle_path_edge_ids: tuple[int, ...]
    z0: StateSummary
    z1: StateSummary | None
    z2: StateSummary
    z3: StateSummary | None
    z3_edge_id: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Check WebQSP oracle-prefix reward ordering under the current " "src.weaver.utility.Reward implementation.")
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help="Materialized retrieval metadata directory.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples.",
    )
    parser.add_argument(
        "--train-budget",
        type=int,
        default=DEFAULT_TRAIN_BUDGET,
        help="Budget used to report budget-feasible subsets.",
    )
    parser.add_argument(
        "--report-limit",
        type=int,
        default=DEFAULT_REPORT_LIMIT,
        help="Maximum number of counterexamples to print per bucket.",
    )
    return parser.parse_args()


def unpack_replay_sequences(
    *,
    edge_ids: torch.Tensor,
    lengths: torch.Tensor,
) -> list[tuple[int, ...]]:
    sequences: list[tuple[int, ...]] = []
    offset = 0
    for length in lengths.tolist():
        length = int(length)
        sequence = tuple(int(edge_id) for edge_id in edge_ids[offset : offset + length].tolist())
        sequences.append(sequence)
        offset += length
    return sequences


def choose_shortest_oracle_path(sample) -> tuple[int, ...] | None:
    reachable_targets = sample.reachable_target_node_ids.view(-1)
    if reachable_targets.numel() == 0:
        return None

    num_targets = int(reachable_targets.numel())
    distances = sample.node_target_distances_flat.view(num_targets, int(sample.num_nodes))
    edge_counts = sample.node_target_shortest_path_edge_count_flat.view(
        num_targets,
        int(sample.num_edges),
    )

    best: tuple[int, int, int, int] | None = None
    for target_pos, target_node_id in enumerate(reachable_targets.tolist()):
        for anchor_node_id in sample.anchor_node_ids.tolist():
            distance = int(distances[target_pos, int(anchor_node_id)].item())
            if distance < 0:
                continue
            candidate = (
                distance,
                int(target_pos),
                int(anchor_node_id),
                int(target_node_id),
            )
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return None

    distance, target_pos, anchor_node_id, target_node_id = best
    replay = compute_replay_path_candidates(
        edge_index=sample.edge_index,
        anchor_node_ids=torch.tensor([anchor_node_id], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([target_node_id], dtype=torch.long),
        node_target_distances_flat=distances[target_pos].reshape(-1).contiguous(),
        node_target_shortest_path_edge_count_flat=edge_counts[target_pos].reshape(-1).contiguous(),
        num_nodes=int(sample.num_nodes),
        max_trajectories=1,
        max_length=int(distance),
    )
    sequences = unpack_replay_sequences(
        edge_ids=replay.edge_ids,
        lengths=replay.lengths,
    )
    if not sequences:
        return None
    return sequences[0]


def build_state(
    *,
    graph: GraphContext,
    sequence: tuple[int, ...],
    budget: int,
) -> State:
    state = StateOps.initial(
        graph,
        torch.tensor([0], dtype=torch.long, device=graph.device),
        budget=int(budget),
    )
    for edge_id in sequence:
        state = StateOps.expand(
            state,
            graph,
            rows=torch.tensor([0], dtype=torch.long, device=graph.device),
            edge_ids=torch.tensor([int(edge_id)], dtype=torch.long, device=graph.device),
            validate=True,
        )
    return state


def summarize_state(
    *,
    state: State,
    graph: GraphContext,
    target: TargetContext,
    reward_model: Reward,
) -> StateSummary:
    reward_out = reward_model(
        state=state,
        graph_context=graph,
        target_context=target,
    )
    answer_count = answer_count_from_active_nodes(
        state=state,
        graph=graph,
        target_mask=target.target_mask,
    )
    active_count = active_node_count(state, graph)
    selected = tuple(int(edge_id) for edge_id in state.selected_edge_ids[0, : int(state.step[0].item())].tolist())
    return StateSummary(
        log_reward=float(reward_out.log_reward[0].item()),
        answer_count=int(answer_count[0].item()),
        active_count=int(active_count[0].item()),
        selected_edge_ids=selected,
    )


def support_union_mask(sample) -> torch.Tensor:
    reachable_targets = int(sample.reachable_target_node_ids.numel())
    if reachable_targets <= 0:
        return torch.zeros(int(sample.num_edges), dtype=torch.bool)
    return sample.node_target_shortest_path_edge_mask_flat.view(
        reachable_targets,
        int(sample.num_edges),
    ).any(dim=0)


def choose_irrelevant_frontier_edge(
    *,
    state: State,
    graph: GraphContext,
    target: TargetContext,
    shortest_path_support_mask: torch.Tensor,
) -> int | None:
    frontier = StateOps.frontier(state, graph)
    if frontier.edge_ids.numel() == 0:
        return None

    _, active_node_ids = active_nodes(state=state, graph=graph)
    active_set = set(int(node_id) for node_id in active_node_ids.tolist())
    target_mask = target.target_mask.to(dtype=torch.bool, device=graph.device)

    preferred: list[int] = []
    fallback: list[int] = []
    for edge_id in frontier.edge_ids.tolist():
        edge_id = int(edge_id)
        if bool(shortest_path_support_mask[edge_id].item()):
            continue
        dst = int(graph.edge_dst[edge_id].item())
        if bool(target_mask[dst].item()):
            continue
        if dst not in active_set:
            preferred.append(edge_id)
        else:
            fallback.append(edge_id)

    if preferred:
        return min(preferred)
    if fallback:
        return min(fallback)
    return None


def diagnose_sample(
    *,
    sample,
    reward_model: Reward,
) -> SampleDiagnosis | None:
    oracle_path = choose_shortest_oracle_path(sample)
    if oracle_path is None:
        return None

    budget = max(len(oracle_path) + 1, 1)
    collator = RetrievalCollator()
    batch = collator([sample])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)

    z0_state = build_state(graph=graph, sequence=(), budget=budget)
    z1_state = None
    if oracle_path:
        z1_state = build_state(
            graph=graph,
            sequence=(oracle_path[0],),
            budget=budget,
        )
    z2_state = build_state(
        graph=graph,
        sequence=oracle_path,
        budget=budget,
    )

    z3_edge_id = choose_irrelevant_frontier_edge(
        state=z2_state,
        graph=graph,
        target=target,
        shortest_path_support_mask=support_union_mask(sample),
    )
    z3_state = None
    if z3_edge_id is not None:
        z3_state = StateOps.expand(
            z2_state,
            graph,
            rows=torch.tensor([0], dtype=torch.long, device=graph.device),
            edge_ids=torch.tensor([z3_edge_id], dtype=torch.long, device=graph.device),
            validate=True,
        )

    return SampleDiagnosis(
        sample_id=str(sample.sample_id),
        path_length=len(oracle_path),
        oracle_path_edge_ids=oracle_path,
        z0=summarize_state(
            state=z0_state,
            graph=graph,
            target=target,
            reward_model=reward_model,
        ),
        z1=(
            summarize_state(
                state=z1_state,
                graph=graph,
                target=target,
                reward_model=reward_model,
            )
            if z1_state is not None
            else None
        ),
        z2=summarize_state(
            state=z2_state,
            graph=graph,
            target=target,
            reward_model=reward_model,
        ),
        z3=(
            summarize_state(
                state=z3_state,
                graph=graph,
                target=target,
                reward_model=reward_model,
            )
            if z3_state is not None
            else None
        ),
        z3_edge_id=z3_edge_id,
    )


def count_if(items: list[SampleDiagnosis], predicate) -> int:
    return sum(1 for item in items if predicate(item))


def filter_items(
    items: list[SampleDiagnosis],
    predicate,
) -> list[SampleDiagnosis]:
    return [item for item in items if predicate(item)]


def print_bucket(
    *,
    title: str,
    items: list[SampleDiagnosis],
    limit: int,
) -> None:
    print(f"\n[{title}] count={len(items)}")
    for item in items[:limit]:
        print(
            json.dumps(
                {
                    "sample_id": item.sample_id,
                    "path_length": item.path_length,
                    "oracle_path_edge_ids": item.oracle_path_edge_ids,
                    "z0": asdict(item.z0),
                    "z1": asdict(item.z1) if item.z1 is not None else None,
                    "z2": asdict(item.z2),
                    "z3": asdict(item.z3) if item.z3 is not None else None,
                    "z3_edge_id": item.z3_edge_id,
                },
                ensure_ascii=True,
            )
        )


def print_stats(
    *,
    items: list[SampleDiagnosis],
    train_budget: int,
    report_limit: int,
) -> None:
    print(f"diagnosed_samples={len(items)}")
    if not items:
        return

    path_lengths = [item.path_length for item in items]
    budget_feasible = filter_items(
        items,
        lambda item: item.path_length <= int(train_budget),
    )
    z3_budget_feasible = filter_items(
        items,
        lambda item: item.path_length + 1 <= int(train_budget),
    )
    full_chain_eligible = filter_items(
        items,
        lambda item: (
            item.path_length >= 2 and item.z1 is not None and item.z0.answer_count == 0 and item.z1.answer_count == 0 and item.z2.answer_count > 0
        ),
    )
    z3_eligible = filter_items(items, lambda item: item.z3 is not None)

    print(
        json.dumps(
            {
                "path_length_min": min(path_lengths),
                "path_length_max": max(path_lengths),
                "path_length_mean": sum(path_lengths) / len(path_lengths),
                f"path_length_le_{train_budget}": len(budget_feasible),
                f"path_length_plus_one_le_{train_budget}": len(z3_budget_feasible),
                "full_chain_eligible": len(full_chain_eligible),
                "z3_eligible": len(z3_eligible),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )

    summary = {
        "z2_gt_z0": count_if(items, lambda item: item.z2.log_reward > item.z0.log_reward),
        "z2_eq_z0": count_if(items, lambda item: item.z2.log_reward == item.z0.log_reward),
        "z2_gt_z1_full_chain_eligible": count_if(
            full_chain_eligible,
            lambda item: item.z2.log_reward > item.z1.log_reward,
        ),
        "z2_le_z1_full_chain_eligible": count_if(
            full_chain_eligible,
            lambda item: item.z2.log_reward <= item.z1.log_reward,
        ),
        "z1_gt_z0_full_chain_eligible": count_if(
            full_chain_eligible,
            lambda item: item.z1.log_reward > item.z0.log_reward,
        ),
        "z1_eq_z0_full_chain_eligible": count_if(
            full_chain_eligible,
            lambda item: item.z1.log_reward == item.z0.log_reward,
        ),
        "z1_lt_z0_full_chain_eligible": count_if(
            full_chain_eligible,
            lambda item: item.z1.log_reward < item.z0.log_reward,
        ),
        "z3_lt_z2": count_if(
            z3_eligible,
            lambda item: item.z3.log_reward < item.z2.log_reward,
        ),
        "z3_eq_z2": count_if(
            z3_eligible,
            lambda item: item.z3.log_reward == item.z2.log_reward,
        ),
        "z3_gt_z2": count_if(
            z3_eligible,
            lambda item: item.z3.log_reward > item.z2.log_reward,
        ),
    }
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))

    print_bucket(
        title="counterexample_z2_le_z1",
        items=filter_items(
            full_chain_eligible,
            lambda item: item.z2.log_reward <= item.z1.log_reward,
        ),
        limit=report_limit,
    )
    print_bucket(
        title="counterexample_z1_not_gt_z0",
        items=filter_items(
            full_chain_eligible,
            lambda item: item.z1.log_reward <= item.z0.log_reward,
        ),
        limit=report_limit,
    )
    print_bucket(
        title="counterexample_z3_not_lt_z2",
        items=filter_items(
            z3_eligible,
            lambda item: item.z3.log_reward >= item.z2.log_reward,
        ),
        limit=report_limit,
    )


def main() -> None:
    args = parse_args()
    artifact = load_materialization_artifact(args.metadata_dir)
    if artifact is None:
        raise FileNotFoundError(f"Could not load materialization manifest from {args.metadata_dir}.")

    dataset = RetrievalDataset(
        materialization=artifact,
        split=str(args.split),
        lmdb_readahead=True,
    )
    reward_model = Reward()

    try:
        total = len(dataset)
        max_samples = total if args.max_samples is None else min(int(args.max_samples), total)
        diagnoses: list[SampleDiagnosis] = []
        for idx in range(max_samples):
            diagnosis = diagnose_sample(
                sample=dataset.get(idx),
                reward_model=reward_model,
            )
            if diagnosis is not None:
                diagnoses.append(diagnosis)
        print(
            json.dumps(
                {
                    "metadata_dir": str(args.metadata_dir),
                    "split": str(args.split),
                    "requested_samples": args.max_samples,
                    "dataset_size": total,
                    "inspected_samples": max_samples,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        print_stats(
            items=diagnoses,
            train_budget=int(args.train_budget),
            report_limit=int(args.report_limit),
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
