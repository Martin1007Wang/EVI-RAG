from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.eval.aggregation import grouped_sample_ids, trajectory_row_matrix
from src.eval.compactness import per_graph_counts
from src.eval.retrieval import mean_over_valid_graphs, safe_divide, safe_f1
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask
from src.utils.scatter import scatter_sum
from src.weaver.context import GraphContext
from src.weaver.rollout.trajectory import (
    BUDGET,
    NO_FRONTIER,
    POLICY_STOP,
    SRC_POLICY,
    TrajectoryBatch,
)
from src.weaver.state import StateBatch

Tensor = torch.Tensor

MAX_LOGGED_K = 8


@dataclass(frozen=True, slots=True)
class ReachableRecallScores:
    recall: Tensor
    valid_graph_mask: Tensor


@dataclass(frozen=True, slots=True)
class RolloutEvalTensors:
    node_masks: Tensor
    edge_masks: Tensor

    recall: Tensor
    f1: Tensor

    edge_count: Tensor
    trajectory_len: Tensor

    policy_stop: Tensor
    no_frontier_stop: Tensor
    budget_boundary: Tensor
    terminal_step: Tensor

    valid_graph_mask: Tensor
    budget: int


def evaluate_rollout_samples(
    *,
    trajectories: TrajectoryBatch,
    batch: RetrievalBatch,
    context: GraphContext,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
    k_windows: Sequence[int],
    enable_terminal_diagnostics: bool,
) -> dict[str, float]:
    ks = normalize_k_windows(
        k_windows,
        max_k=_num_samples(
            trajectories,
            num_graphs=int(batch.num_graphs),
        ),
    )

    tensors = rollout_eval_tensors(
        trajectories=trajectories,
        batch=batch,
        context=context,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    metrics: dict[str, float] = {}
    metrics.update(sample_metrics(tensors))
    metrics.update(
        union_metrics(
            tensors,
            batch=batch,
            ks=ks,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )
    )

    metrics["marginal/answer_support_prob"] = answer_support_probability(
        tensors=tensors,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    if enable_terminal_diagnostics:
        metrics.update(
            terminal_metrics(
                trajectories=trajectories,
                tensors=tensors,
                batch=batch,
            )
        )

    return metrics


def rollout_eval_tensors(
    *,
    trajectories: TrajectoryBatch,
    batch: RetrievalBatch,
    context: GraphContext,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> RolloutEvalTensors:
    device = torch.device("cpu")
    num_graphs = int(batch.num_graphs)

    terminal_state = terminal_state_from_trajectories(trajectories)

    node_masks, edge_masks = stacked_terminal_masks(
        state=terminal_state,
        trajectories=trajectories,
        context=context,
        num_graphs=num_graphs,
        num_nodes=int(context.num_nodes),
        num_edges=int(context.num_edges),
        device=device,
    )

    _, recall, f1, valid_graph_mask = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    edge_count = per_graph_counts(
        edge_masks,
        batch.edge_batch.to(device=device, dtype=torch.long),
        num_graphs=num_graphs,
    )

    terminal_step = _stack_terminal_step(
        trajectories=trajectories,
        num_graphs=num_graphs,
    )

    policy_stop, no_frontier_stop, budget_boundary = terminal_matrices(
        trajectories=trajectories,
        num_graphs=num_graphs,
    )

    trajectory_len = terminal_step.float() + policy_stop

    return RolloutEvalTensors(
        node_masks=node_masks,
        edge_masks=edge_masks,
        recall=recall,
        f1=f1,
        edge_count=edge_count,
        trajectory_len=trajectory_len,
        policy_stop=policy_stop,
        no_frontier_stop=no_frontier_stop,
        budget_boundary=budget_boundary,
        terminal_step=terminal_step,
        valid_graph_mask=valid_graph_mask,
        budget=int(trajectories.budget),
    )


def terminal_state_from_trajectories(
    trajectories: TrajectoryBatch,
) -> StateBatch:
    """
    Zero-replay conversion from trajectory records to terminal StateBatch.

    This does not expand edges step by step. It only wraps the canonical
    terminal selected-edge representation.
    """

    return StateBatch(
        graph_ids=trajectories.graph_ids,
        edge_ids=trajectories.edge_ids,
        edge_count=trajectories.edge_count,
        budget=int(trajectories.budget),
    )


def stacked_terminal_masks(
    *,
    state: StateBatch,
    trajectories: TrajectoryBatch,
    context: GraphContext,
    num_graphs: int,
    num_nodes: int,
    num_edges: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Build dense sample-level terminal node/edge masks.

    Shape:
    - node_masks: [K, num_nodes]
    - edge_masks: [K, num_edges]

    K is the maximum number of trajectories sampled per graph.
    """

    if state.num_states != int(trajectories.num_trajectories):
        raise ValueError("StateBatch rows must match TrajectoryBatch rows: " f"{state.num_states} vs {int(trajectories.num_trajectories)}.")

    num_samples = _num_samples(
        trajectories,
        num_graphs=int(num_graphs),
    )

    if num_samples == 0:
        return (
            torch.zeros((0, int(num_nodes)), dtype=torch.bool, device=device),
            torch.zeros((0, int(num_edges)), dtype=torch.bool, device=device),
        )

    source_device = state.device

    sample_ids = _sample_ids(
        trajectories,
        num_graphs=int(num_graphs),
    ).to(device=source_device, dtype=torch.long)

    node_masks = torch.zeros(
        (num_samples, int(num_nodes)),
        dtype=torch.bool,
        device=source_device,
    )
    edge_masks = torch.zeros(
        (num_samples, int(num_edges)),
        dtype=torch.bool,
        device=source_device,
    )

    node_state_ids, node_ids = state.covered_node_pairs(context)
    if int(node_ids.numel()) > 0:
        node_sample_ids = sample_ids.index_select(0, node_state_ids)
        node_masks[node_sample_ids, node_ids] = True

    edge_state_ids, edge_ids = selected_edge_pairs(state)
    if int(edge_ids.numel()) > 0:
        edge_sample_ids = sample_ids.index_select(0, edge_state_ids)
        edge_masks[edge_sample_ids, edge_ids] = True

    return (
        node_masks.to(device=device),
        edge_masks.to(device=device),
    )


def selected_edge_pairs(state: StateBatch) -> tuple[Tensor, Tensor]:
    """
    Return valid selected (state_id, edge_id) pairs from StateBatch.

    Uses edge_count rather than edge_ids >= 0, so padding is ignored even if a
    caller accidentally leaves non-negative garbage after edge_count.
    """

    num_states = int(state.num_states)
    budget = int(state.budget)

    if num_states == 0 or budget == 0:
        empty = torch.empty(0, dtype=torch.long, device=state.device)
        return empty, empty

    steps = torch.arange(
        budget,
        dtype=torch.long,
        device=state.device,
    ).view(1, budget)

    valid = steps.lt(state.edge_count.view(num_states, 1))
    edge_ids = state.edge_ids[valid]

    state_ids = torch.repeat_interleave(
        torch.arange(
            num_states,
            dtype=torch.long,
            device=state.device,
        ),
        state.edge_count.to(dtype=torch.long),
    )

    return state_ids, edge_ids


def retrieval_from_masks(
    *,
    node_masks: Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    device = node_masks.device
    num_samples = int(node_masks.size(0))
    num_graphs = int(batch.num_graphs)

    if num_samples == 0:
        empty = torch.zeros((0, num_graphs), dtype=torch.float32, device=device)
        return (
            empty,
            empty,
            empty,
            torch.zeros(num_graphs, dtype=torch.bool, device=device),
        )

    node_batch = batch.batch.to(device=device, dtype=torch.long)

    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )

    retrieved_nodes = node_masks
    if exclude_anchors_from_retrieved:
        retrieved_nodes = retrieved_nodes & ~anchor_node_mask(
            batch,
            device=device,
        ).unsqueeze(0)

    hit_nodes = retrieved_nodes & target_nodes.unsqueeze(0)

    expanded_index = _sample_item_graph_index(
        item_batch=node_batch,
        num_samples=num_samples,
        num_graphs=num_graphs,
    )

    hits = scatter_sum(
        hit_nodes.float().reshape(-1),
        expanded_index,
        dim=0,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)

    retrieved = scatter_sum(
        retrieved_nodes.float().reshape(-1),
        expanded_index,
        dim=0,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)

    gold = scatter_sum(
        target_nodes.float(),
        node_batch,
        dim=0,
        dim_size=num_graphs,
    )

    valid = gold.gt(0.0)

    precision = safe_divide(hits, retrieved)

    recall = safe_divide(
        hits,
        gold.unsqueeze(0).expand_as(hits),
    )
    recall = torch.where(
        valid.unsqueeze(0),
        recall,
        torch.zeros_like(recall),
    )

    return precision, recall, safe_f1(precision, recall), valid


def sample_metrics(tensors: RolloutEvalTensors) -> dict[str, float]:
    valid = tensors.valid_graph_mask

    metrics = {
        "single_rollout/mean_recall": mean_over_valid_graphs(tensors.recall, valid),
        "single_rollout/mean_f1": mean_over_valid_graphs(tensors.f1, valid),
        "rollout/edge_count_mean": mean_over_valid_graphs(tensors.edge_count, valid),
    }

    for edge_count in range(int(tensors.budget) + 1):
        metrics[f"rollout/edge_count_rate_{edge_count}"] = mean_over_valid_graphs(
            tensors.edge_count.eq(edge_count).float(),
            valid,
        )

    metrics["rollout/edge_budget_full_rate"] = mean_over_valid_graphs(
        tensors.edge_count.ge(int(tensors.budget)).float(),
        valid,
    )

    return metrics


def union_metrics(
    tensors: RolloutEvalTensors,
    *,
    batch: RetrievalBatch,
    ks: Sequence[int],
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}

    edge_batch = batch.edge_batch.to(
        device=tensors.edge_masks.device,
        dtype=torch.long,
    )

    for k in ks:
        node_masks = tensors.node_masks[:k].any(dim=0, keepdim=True)
        edge_masks = tensors.edge_masks[:k].any(dim=0, keepdim=True)

        _, recall, _, valid = retrieval_from_masks(
            node_masks=node_masks,
            batch=batch,
            exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
            use_reachable_targets=use_reachable_targets,
        )

        edge_count = per_graph_counts(
            edge_masks,
            edge_batch,
            num_graphs=int(batch.num_graphs),
        ).squeeze(0)

        denom = tensors.edge_count[:k].sum(dim=0)
        unique_ratio = safe_divide(edge_count, denom)

        prefix = f"rollout_union@{k}"
        metrics[f"{prefix}/recall"] = mean_over_valid_graphs(
            recall.squeeze(0),
            valid,
        )
        metrics[f"{prefix}/edges"] = mean_over_valid_graphs(
            edge_count,
            valid,
        )
        metrics[f"{prefix}/redundancy"] = mean_over_valid_graphs(
            1.0 - unique_ratio,
            valid,
        )

    return metrics


def answer_support_probability(
    *,
    tensors: RolloutEvalTensors,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> float:
    if tensors.node_masks.numel() == 0:
        return 0.0

    device = tensors.node_masks.device

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    targets = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )

    if exclude_anchors_from_retrieved:
        targets = targets & ~anchor_node_mask(batch, device=device)

    support = tensors.node_masks.float().mean(dim=0)

    values: list[Tensor] = []
    for graph_id in range(int(batch.num_graphs)):
        if not bool(tensors.valid_graph_mask[graph_id]):
            continue

        graph_targets = targets & node_batch.eq(graph_id)
        if bool(graph_targets.any()):
            values.append(support[graph_targets].mean())

    if not values:
        return 0.0

    return float(torch.stack(values).mean().item())


def terminal_metrics(
    *,
    trajectories: TrajectoryBatch,
    tensors: RolloutEvalTensors,
    batch: RetrievalBatch,
) -> dict[str, float]:
    stats = hit_terminal_stats(
        trajectories,
        batch=batch,
    )

    hit = stats["hit"]
    continued = stats["continued"]
    valid = tensors.valid_graph_mask

    return {
        "terminal/policy_stop_rate": mean_over_valid_graphs(
            tensors.policy_stop,
            valid,
        ),
        "terminal/structural_stop_rate": mean_over_valid_graphs(
            tensors.no_frontier_stop,
            valid,
        ),
        "terminal/budget_boundary_rate": mean_over_valid_graphs(
            tensors.budget_boundary,
            valid,
        ),
        "terminal/policy_terminal_rate": mean_over_valid_graphs(
            tensors.policy_stop,
            valid,
        ),
        "terminal/forced_terminal_rate": mean_over_valid_graphs(
            tensors.no_frontier_stop + tensors.budget_boundary,
            valid,
        ),
        "terminal/hit_then_continue_rate": _mean_hit_values(
            continued.float(),
            hit,
            valid,
        ),
    }


def reachable_recall_scores(
    *,
    node_masks: Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> ReachableRecallScores:
    _, recall, _, valid = retrieval_from_masks(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    return ReachableRecallScores(
        recall=recall,
        valid_graph_mask=valid,
    )


def one_sample_reachable_recall(scores: ReachableRecallScores) -> float:
    return mean_over_valid_graphs(
        scores.recall[:1],
        scores.valid_graph_mask,
    )


def budget_forced_terminal_rate(
    *,
    trajectories: TrajectoryBatch,
    valid_graph_mask: Tensor,
) -> float:
    _, _, boundary = terminal_matrices(
        trajectories=trajectories,
        num_graphs=int(valid_graph_mask.numel()),
    )

    return mean_over_valid_graphs(
        boundary,
        valid_graph_mask,
    )


def terminal_matrices(
    *,
    trajectories: TrajectoryBatch,
    num_graphs: int,
    policy_only: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Return [K, G] terminal-reason matrices.

    Outputs:
    - policy_stop: trajectory ended because policy sampled STOP;
    - no_frontier_stop: trajectory ended because no legal expansion existed;
    - budget_boundary: trajectory ended because expansion budget was exhausted.

    If policy_only=True, policy_stop only counts trajectories whose source is
    SRC_POLICY. Use this when trajectories may mix policy/replay/oracle rows.
    """

    if int(trajectories.num_trajectories) == 0:
        shape = (0, int(num_graphs))
        empty = torch.zeros(shape, dtype=torch.float32)
        return empty, empty.clone(), empty.clone()

    policy_stop = trajectories.stop_reason.eq(int(POLICY_STOP))
    no_frontier_stop = trajectories.stop_reason.eq(int(NO_FRONTIER))
    budget_boundary = trajectories.stop_reason.eq(int(BUDGET))

    if policy_only:
        policy_source = trajectories.source.eq(int(SRC_POLICY))
        policy_stop = policy_stop & policy_source

    return (
        trajectory_row_matrix(
            trajectories,
            policy_stop.float(),
            num_graphs=int(num_graphs),
        ).to(device=torch.device("cpu"), dtype=torch.float32),
        trajectory_row_matrix(
            trajectories,
            no_frontier_stop.float(),
            num_graphs=int(num_graphs),
        ).to(device=torch.device("cpu"), dtype=torch.float32),
        trajectory_row_matrix(
            trajectories,
            budget_boundary.float(),
            num_graphs=int(num_graphs),
        ).to(device=torch.device("cpu"), dtype=torch.float32),
    )


def hit_terminal_stats(
    trajectories: TrajectoryBatch,
    *,
    batch: RetrievalBatch,
) -> dict[str, Tensor]:
    num_graphs = int(batch.num_graphs)
    num_samples = _num_samples(
        trajectories,
        num_graphs=num_graphs,
    )

    hit = torch.zeros(
        (num_samples, num_graphs),
        dtype=torch.bool,
    )
    continued = torch.zeros(
        (num_samples, num_graphs),
        dtype=torch.bool,
    )

    if int(trajectories.num_trajectories) == 0:
        return {
            "hit": hit,
            "continued": continued,
        }

    device = torch.device("cpu")

    targets = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=True,
    )
    anchors = anchor_node_mask(
        batch,
        device=device,
    )
    node_batch = batch.batch.to(
        device=device,
        dtype=torch.long,
    )
    edge_index = batch.edge_index.to(
        device=device,
        dtype=torch.long,
    )

    selected = trajectories.edge_ids.to(
        device=device,
        dtype=torch.long,
    )
    edge_count = trajectories.edge_count.to(
        device=device,
        dtype=torch.long,
    )
    sample_ids = _sample_ids(
        trajectories,
        num_graphs=num_graphs,
    ).to(device=device, dtype=torch.long)
    graph_ids = trajectories.graph_ids.to(
        device=device,
        dtype=torch.long,
    )

    for row in range(int(trajectories.num_trajectories)):
        sample_id = int(sample_ids[row].item())
        graph_id = int(graph_ids[row].item())

        if graph_id >= num_graphs:
            continue

        graph_nodes = node_batch.eq(graph_id)
        active = anchors & graph_nodes

        seen_hit = bool((active & targets & graph_nodes).any())
        expanded_after_hit = False

        for step in range(int(edge_count[row].item())):
            edge_id = int(selected[row, step].item())

            if edge_id < 0:
                raise ValueError("Trajectory edge prefix contains negative edge id " f"at row={row}, step={step}.")

            if seen_hit:
                expanded_after_hit = True

            active[edge_index[0, edge_id]] = True
            active[edge_index[1, edge_id]] = True

            seen_hit = seen_hit or bool((active & targets & graph_nodes).any())

        hit[sample_id, graph_id] = seen_hit
        continued[sample_id, graph_id] = expanded_after_hit

    return {
        "hit": hit,
        "continued": continued,
    }


def normalize_k_windows(
    ks: Sequence[int],
    *,
    max_k: int,
) -> tuple[int, ...]:
    if max_k <= 0:
        return tuple()

    max_logged_k = min(
        int(max_k),
        MAX_LOGGED_K,
    )

    out = tuple(sorted({int(k) for k in ks if 1 <= int(k) <= max_logged_k}))

    return out or (1,)


def _stack_terminal_step(
    *,
    trajectories: TrajectoryBatch,
    num_graphs: int,
) -> Tensor:
    if int(trajectories.num_trajectories) == 0:
        return torch.zeros(
            (0, int(num_graphs)),
            dtype=torch.long,
        )

    return trajectory_row_matrix(
        trajectories,
        trajectories.edge_count,
        num_graphs=int(num_graphs),
        fill_value=0.0,
    ).to(
        device=torch.device("cpu"),
        dtype=torch.long,
    )


def _sample_ids(
    trajectories: TrajectoryBatch,
    *,
    num_graphs: int,
) -> Tensor:
    return grouped_sample_ids(
        trajectories,
        num_graphs=int(num_graphs),
    )


def _num_samples(
    trajectories: TrajectoryBatch,
    *,
    num_graphs: int,
) -> int:
    if int(trajectories.num_trajectories) == 0:
        return 0

    counts = torch.bincount(
        trajectories.graph_ids.to(dtype=torch.long),
        minlength=int(num_graphs),
    )

    return int(counts.max().item()) if counts.numel() > 0 else 0


def _sample_item_graph_index(
    *,
    item_batch: Tensor,
    num_samples: int,
    num_graphs: int,
) -> Tensor:
    offsets = torch.arange(
        int(num_samples),
        device=item_batch.device,
    ).unsqueeze(
        1
    ) * int(num_graphs)

    return (item_batch.unsqueeze(0) + offsets).reshape(-1)


def _mean_hit_values(
    values: Tensor,
    hit_mask: Tensor,
    valid_graph_mask: Tensor,
) -> float:
    valid = hit_mask & valid_graph_mask.unsqueeze(0)

    if not bool(valid.any()):
        return 0.0

    return float(values[valid].float().mean().item())


__all__ = [
    "ReachableRecallScores",
    "budget_forced_terminal_rate",
    "evaluate_rollout_samples",
    "one_sample_reachable_recall",
    "reachable_recall_scores",
]
