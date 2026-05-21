from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.schema import RetrievalBatch
from src.eval.targets import eval_target_node_mask
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor


def default_eval_device() -> torch.device:
    return torch.device("cpu")


def batch_num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs)


def compute_exploration_diversity(
    rollouts: Sequence[RolloutResult],
    batch: RetrievalBatch,
    *,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Diversity metrics across rollout samples.

    Metrics:
    - pairwise_edge_jaccard_distance: pairwise edge-set diversity.
    - node_jaccard_distance: pairwise node-set diversity.
    - target_hit_jaccard_distance: pairwise diversity over hit target nodes.
    - terminal_f1_std: variation in terminal answer F1.
    """
    device = device or default_eval_device()

    if len(rollouts) <= 1:
        return {
            "pairwise_edge_jaccard_distance": 0.0,
            "node_jaccard_distance": 0.0,
            "target_hit_jaccard_distance": 0.0,
            "unique_terminal_subgraph_rate": unique_terminal_subgraph_rate(rollouts),
            "unique_selected_edge_set_rate": unique_selected_edge_set_rate(rollouts),
            "terminal_f1_std": terminal_f1_std(rollouts),
        }

    node_masks, edge_masks = SubgraphReconstructor(batch, device=device).stack(rollouts)

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)
    targets = eval_target_node_mask(batch, device=device, use_reachable_targets=True)

    return {
        "pairwise_edge_jaccard_distance": mean_pairwise_jaccard_distance_by_graph(
            edge_masks,
            edge_batch,
            num_graphs=batch_num_graphs(batch),
        ),
        "node_jaccard_distance": mean_pairwise_jaccard_distance_by_graph(
            node_masks,
            node_batch,
            num_graphs=batch_num_graphs(batch),
        ),
        "target_hit_jaccard_distance": mean_pairwise_jaccard_distance_by_graph(
            node_masks & targets.unsqueeze(0),
            node_batch,
            num_graphs=batch_num_graphs(batch),
        ),
        "unique_terminal_subgraph_rate": unique_terminal_subgraph_rate(rollouts),
        "unique_selected_edge_set_rate": unique_selected_edge_set_rate(rollouts),
        "terminal_f1_std": terminal_f1_std(rollouts),
    }


def compute_exploration_diversity_at_ks(
    rollouts: Sequence[RolloutResult],
    batch: RetrievalBatch,
    *,
    ks: Sequence[int],
    device: torch.device | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    effective_ks = tuple(sorted({int(k) for k in ks if 1 <= int(k) <= len(rollouts)}))
    if not effective_ks:
        effective_ks = tuple(sorted({int(k) for k in ks if int(k) >= 1}))

    for k in effective_ks:
        current = compute_exploration_diversity(rollouts[:k], batch, device=device)
        metrics[f"unique_terminal_subgraph_rate_at_{k}"] = current[
            "unique_terminal_subgraph_rate"
        ]
        metrics[f"unique_selected_edge_set_rate_at_{k}"] = current[
            "unique_selected_edge_set_rate"
        ]
        metrics[f"pairwise_edge_jaccard_distance_at_{k}"] = current[
            "pairwise_edge_jaccard_distance"
        ]
    return metrics


def mean_pairwise_jaccard_distance_by_graph(
    masks: torch.Tensor,
    batch_index: torch.Tensor,
    *,
    num_graphs: int,
) -> float:
    """
    Mean pairwise Jaccard distance, computed separately per graph then averaged.

    masks:
        [R, M]

    batch_index:
        graph id per item, shape [M]
    """
    if masks.ndim != 2:
        raise ValueError(f"masks must have shape [R, M], got {tuple(masks.shape)}.")

    num_rollouts = int(masks.size(0))
    if num_rollouts <= 1:
        return 0.0

    batch_index = batch_index.to(device=masks.device, dtype=torch.long)
    pair_mask = torch.triu(
        torch.ones(
            num_rollouts,
            num_rollouts,
            dtype=torch.bool,
            device=masks.device,
        ),
        diagonal=1,
    )

    values: list[torch.Tensor] = []

    for graph_id in range(int(num_graphs)):
        item_mask = batch_index.eq(graph_id)
        if not bool(item_mask.any()):
            continue

        graph_masks = masks[:, item_mask].float()
        intersection = graph_masks @ graph_masks.T
        size = graph_masks.sum(dim=1)
        union = size.unsqueeze(0) + size.unsqueeze(1) - intersection

        jaccard = intersection / union.clamp_min(1e-8)
        jaccard = torch.where(union.gt(0.0), jaccard, torch.ones_like(jaccard))

        values.append((1.0 - jaccard)[pair_mask])

    if not values:
        return 0.0

    return float(torch.cat(values).mean().item())


def terminal_f1_std(rollouts: Sequence[RolloutResult]) -> float:
    if not rollouts:
        return 0.0

    values = torch.cat(
        [
            rollout.policy_action_log_prob[
                torch.arange(rollout.num_rollouts, device=rollout.device),
                rollout.stop_step,
            ].detach().to(dtype=torch.float32)
            for rollout in rollouts
        ],
        dim=0,
    )

    if values.numel() <= 1:
        return 0.0

    return float(values.std(unbiased=False).item())


def unique_terminal_subgraph_rate(rollouts: Sequence[RolloutResult]) -> float:
    return unique_selected_edge_set_rate(rollouts)


def unique_selected_edge_set_rate(rollouts: Sequence[RolloutResult]) -> float:
    if not rollouts:
        return 0.0

    num_rollouts = len(rollouts)
    num_graphs = int(rollouts[0].num_rollouts)
    rates: list[float] = []
    for graph_id in range(num_graphs):
        unique_sets = {
            _selected_edge_set_for_graph(rollout=rollout, graph_id=graph_id)
            for rollout in rollouts
        }
        rates.append(float(len(unique_sets)) / float(num_rollouts))

    return sum(rates) / float(len(rates)) if rates else 0.0


def _selected_edge_set_for_graph(
    *, rollout: RolloutResult, graph_id: int
) -> tuple[int, ...]:
    selected_edge_ids = rollout.selected_edge_ids[graph_id]
    continue_mask = rollout.expand_mask[graph_id]
    trajectory_length = int(rollout.stop_step[graph_id].item()) + 1
    if trajectory_length <= 0:
        return ()

    valid_steps = torch.arange(
        selected_edge_ids.numel(), device=selected_edge_ids.device
    ).lt(trajectory_length)
    edge_ids = selected_edge_ids[valid_steps & continue_mask & selected_edge_ids.ge(0)]
    return tuple(sorted(int(edge_id) for edge_id in edge_ids.tolist()))


__all__ = [
    "compute_exploration_diversity_at_ks",
    "compute_exploration_diversity",
    "mean_pairwise_jaccard_distance_by_graph",
    "terminal_f1_std",
    "unique_selected_edge_set_rate",
    "unique_terminal_subgraph_rate",
]
