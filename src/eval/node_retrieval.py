from __future__ import annotations

from collections.abc import Sequence

import torch
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.eval.compactness import compute_compactness_expectations
from src.weaver.rollout.schema import RolloutBatch
from src.weaver.rollout.terminal_subgraph import (
    anchor_node_mask,
    batch_num_graphs,
    default_eval_device,
    eval_target_node_mask,
    stack_terminal_subgraph_masks,
)


def compute_node_retrieval_matrix(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device | None = None,
    exclude_anchors_from_retrieved: bool = True,
    use_reachable_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute target-node precision / recall / F1 for each rollout and graph.

    Returns:
        precision: [R, B]
        recall: [R, B]
        f1: [R, B]
        valid_graph_mask: [B]

    Semantics:
        precision = hit target nodes / retrieved nodes
        recall    = hit target nodes / gold target nodes
        f1        = harmonic mean of precision and recall

    retrieved nodes are terminal active nodes reconstructed from rollout traces.
    target nodes are controlled by use_reachable_targets.
    """

    device = device or default_eval_device()

    num_rollouts = len(rollouts)
    num_graphs = batch_num_graphs(batch)

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )

    terminal_nodes, _ = stack_terminal_subgraph_masks(
        rollouts,
        batch,
        device=device,
    )

    if terminal_nodes.shape[0] != num_rollouts:
        raise ValueError(
            f"terminal node mask rollout dimension mismatch: "
            f"{terminal_nodes.shape[0]} != {num_rollouts}."
        )

    if exclude_anchors_from_retrieved:
        anchors = anchor_node_mask(batch, device=device)
        retrieved_nodes = terminal_nodes & ~anchors.unsqueeze(0)
    else:
        retrieved_nodes = terminal_nodes

    hit_nodes = terminal_nodes & target_nodes.unsqueeze(0)
    expanded_node_batch = _rollout_graph_index(
        node_batch=node_batch,
        num_rollouts=num_rollouts,
        num_graphs=num_graphs,
    )

    hits_per_graph = scatter_sum(
        hit_nodes.float().reshape(-1),
        expanded_node_batch,
        dim=0,
        dim_size=num_rollouts * num_graphs,
    ).view(num_rollouts, num_graphs)
    retrieved_per_graph = scatter_sum(
        retrieved_nodes.float().reshape(-1),
        expanded_node_batch,
        dim=0,
        dim_size=num_rollouts * num_graphs,
    ).view(num_rollouts, num_graphs)
    gold_per_graph = scatter_sum(
        target_nodes.float(),
        node_batch,
        dim=0,
        dim_size=num_graphs,
    )

    valid_graph_mask = gold_per_graph.gt(0.0)

    precision = safe_divide(hits_per_graph, retrieved_per_graph)
    recall = safe_divide(
        hits_per_graph,
        gold_per_graph.unsqueeze(0).expand_as(hits_per_graph),
    )
    recall = torch.where(
        valid_graph_mask.unsqueeze(0),
        recall,
        torch.zeros_like(recall),
    )

    f1 = safe_f1(precision, recall)

    return precision, recall, f1, valid_graph_mask


def compute_expected_node_retrieval_quality(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    device: torch.device | None = None,
    exclude_anchors_from_retrieved: bool = True,
    use_reachable_targets: bool = True,
) -> dict[str, float]:
    """
    Monte Carlo estimate of one sampled terminal subgraph's answer quality.
    """

    if not rollouts:
        return {
            "expected_target_precision": 0.0,
            "expected_target_recall": 0.0,
            "expected_target_f1": 0.0,
            "nonzero_f1_rate": 0.0,
        }

    precision, recall, f1, valid_graph_mask = compute_node_retrieval_matrix(
        rollouts,
        batch,
        device=device,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    return {
        "expected_target_precision": mean_over_valid_graphs(
            precision,
            valid_graph_mask,
        ),
        "expected_target_recall": mean_over_valid_graphs(
            recall,
            valid_graph_mask,
        ),
        "expected_target_f1": mean_over_valid_graphs(
            f1,
            valid_graph_mask,
        ),
        "nonzero_f1_rate": mean_over_valid_graphs(
            f1.gt(0.0).to(dtype=torch.float32),
            valid_graph_mask,
        ),
    }


def compute_best_of_k_node_retrieval_quality(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    ks: Sequence[int],
    device: torch.device | None = None,
    exclude_anchors_from_retrieved: bool = True,
    use_reachable_targets: bool = True,
) -> dict[str, float]:
    """
    Best-of-k answer discovery over rollout samples.

    For each k, use the best metric value among the first k rollout samples.
    """

    effective_ks = normalize_ks(ks, max_k=len(rollouts))
    metrics: dict[str, float] = {}

    if not rollouts:
        for k in effective_ks:
            metrics[f"max_target_precision_at_{k}"] = 0.0
            metrics[f"max_target_recall_at_{k}"] = 0.0
            metrics[f"max_target_f1_at_{k}"] = 0.0
            metrics[f"full_recall_rate_at_{k}"] = 0.0
        return metrics

    precision, recall, f1, valid_graph_mask = compute_node_retrieval_matrix(
        rollouts,
        batch,
        device=device,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    for k in effective_ks:
        best_precision = precision[:k].max(dim=0).values
        best_recall = recall[:k].max(dim=0).values
        best_f1 = f1[:k].max(dim=0).values

        metrics[f"max_target_precision_at_{k}"] = mean_over_valid_graphs(
            best_precision,
            valid_graph_mask,
        )
        metrics[f"max_target_recall_at_{k}"] = mean_over_valid_graphs(
            best_recall,
            valid_graph_mask,
        )
        metrics[f"max_target_f1_at_{k}"] = mean_over_valid_graphs(
            best_f1,
            valid_graph_mask,
        )
        metrics[f"nonzero_f1_rate_at_{k}"] = mean_over_valid_graphs(
            best_f1.gt(0.0).to(dtype=torch.float32),
            valid_graph_mask,
        )
        metrics[f"full_recall_rate_at_{k}"] = full_recall_rate(
            best_recall,
            valid_graph_mask,
        )

    return metrics


def compute_sample_retrieval_metrics(
    rollouts: Sequence[RolloutBatch],
    batch: RetrievalBatch,
    *,
    include_compactness: bool = True,
    include_dangling: bool = False,
    exclude_anchors_from_retrieved: bool = True,
    use_reachable_targets: bool = True,
) -> dict[str, float]:
    """
    Metrics for a single sampled rollout distribution.

    This estimates:
        If one terminal subgraph is sampled from the policy, what is its average
        target retrieval quality and structural cost?
    """

    metrics = compute_expected_node_retrieval_quality(
        rollouts,
        batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    if include_compactness:
        metrics.update(
            compute_compactness_expectations(
                rollouts,
                batch,
                include_dangling=include_dangling,
            )
        )
    elif include_dangling:
        compactness = compute_compactness_expectations(
            rollouts,
            batch,
            include_dangling=True,
        )
        metrics["dangling_edge_ratio"] = compactness["dangling_edge_ratio"]

    return metrics


def mean_over_valid_graphs(
    values: torch.Tensor,
    valid_graph_mask: torch.Tensor,
) -> float:
    """
    Average over valid graphs.

    values:
        [B] or [R, B]

    valid_graph_mask:
        [B]
    """

    if values.numel() == 0 or not bool(valid_graph_mask.any()):
        return 0.0

    if values.ndim == 1:
        return float(values[valid_graph_mask].mean().item())

    if values.ndim == 2:
        return float(values[:, valid_graph_mask].mean().item())

    raise ValueError(f"values must be 1D or 2D, got {tuple(values.shape)}.")


def full_recall_rate(
    recall: torch.Tensor,
    valid_graph_mask: torch.Tensor,
) -> float:
    if recall.numel() == 0 or not bool(valid_graph_mask.any()):
        return 0.0

    return float(recall[valid_graph_mask].eq(1.0).float().mean().item())


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        denominator.gt(0.0),
        numerator / denominator.clamp_min(1e-8),
        torch.zeros_like(numerator),
    )


def _rollout_graph_index(
    *,
    node_batch: torch.Tensor,
    num_rollouts: int,
    num_graphs: int,
) -> torch.Tensor:
    row_offsets = torch.arange(int(num_rollouts), device=node_batch.device).unsqueeze(
        1
    ) * int(num_graphs)
    return (node_batch.unsqueeze(0) + row_offsets).reshape(-1)


def safe_f1(
    precision: torch.Tensor,
    recall: torch.Tensor,
) -> torch.Tensor:
    denominator = precision + recall
    return torch.where(
        denominator.gt(0.0),
        2.0 * precision * recall / denominator.clamp_min(1e-8),
        torch.zeros_like(denominator),
    )


def normalize_ks(
    ks: Sequence[int],
    *,
    max_k: int,
) -> tuple[int, ...]:
    if max_k < 1:
        return tuple(sorted({int(k) for k in ks if int(k) >= 1}))

    normalized = tuple(sorted({int(k) for k in ks if 1 <= int(k) <= max_k}))
    if not normalized:
        raise ValueError(f"No valid k in {tuple(ks)} for max_k={max_k}.")

    return normalized


__all__ = [
    "compute_best_of_k_node_retrieval_quality",
    "compute_expected_node_retrieval_quality",
    "compute_node_retrieval_matrix",
    "compute_sample_retrieval_metrics",
    "full_recall_rate",
    "mean_over_valid_graphs",
    "normalize_ks",
    "safe_divide",
    "safe_f1",
]
