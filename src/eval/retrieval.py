from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask
from src.utils.scatter import scatter_sum
from src.weaver.context import GraphContext
from src.weaver.rollout.subgraph import stacked_subgraph_masks
from src.weaver.rollout.trajectory import TrajectoryBatch


def default_eval_device() -> torch.device:
    return torch.device("cpu")


def batch_num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs)


def target_nodes_for_retrieval(
    *,
    batch: RetrievalBatch,
    device: torch.device,
    use_reachable_targets: bool,
    exclude_anchors: bool,
) -> torch.Tensor:
    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )
    if exclude_anchors:
        target_nodes = target_nodes & ~anchor_node_mask(batch, device=device)
    return target_nodes


def compute_node_retrieval_matrix(
    rollouts: TrajectoryBatch,
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

    retrieved nodes are terminal active nodes reconstructed from rollout trajectories.
    target nodes are controlled by use_reachable_targets.
    """

    device = device or default_eval_device()

    num_graphs = batch_num_graphs(batch)
    num_rollouts = _num_samples(rollouts, num_graphs=num_graphs)

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    target_nodes = target_nodes_for_retrieval(
        batch=batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
        exclude_anchors=exclude_anchors_from_retrieved,
    )

    terminal_nodes, _ = stacked_subgraph_masks(
        rollouts,
        GraphContext.from_batch(batch),
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

    hit_nodes = retrieved_nodes & target_nodes.unsqueeze(0)
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


def _num_samples(rollouts: TrajectoryBatch, *, num_graphs: int) -> int:
    if rollouts.num_trajectories == 0:
        return 0
    counts = torch.bincount(rollouts.graph_ids, minlength=int(num_graphs))
    return int(counts.max().item()) if counts.numel() > 0 else 0


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


__all__ = [
    "compute_node_retrieval_matrix",
    "mean_over_valid_graphs",
    "safe_divide",
    "safe_f1",
    "target_nodes_for_retrieval",
]
