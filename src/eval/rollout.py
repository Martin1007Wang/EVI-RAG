from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.eval.compactness import compactness_from_masks
from src.eval.retrieval import mean_over_valid_graphs, safe_divide
from src.eval.targets import eval_target_node_mask
from src.graph.masks import node_mask_from_ids
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor


@dataclass(frozen=True, slots=True)
class ReachableRecallScores:
    """
    Reachable-answer recall for rollout samples.

    Shape convention:
        K: number of rollout samples
        B: number of source graphs in the batch
    """

    recall: torch.Tensor
    valid_graph_mask: torch.Tensor


def evaluate_rollout_samples(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    best_of_k: int,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> dict[str, float]:
    """
    Minimal online validation dashboard.

    The default validation surface should answer only five questions:
    1. Can the model reach answers if allowed K samples?
    2. How good is one sampled rollout by itself?
    3. How much does sampling help over one rollout?
    4. How large is the selected evidence subgraph?
    5. Does the model rely on forced budget stops?
    """

    node_masks, edge_masks = SubgraphReconstructor(
        batch,
        device=torch.device("cpu"),
    ).stack(rollout_samples)

    scores = reachable_recall_scores(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    best_recall = best_of_k_reachable_recall(
        scores=scores,
        best_of_k=best_of_k,
    )
    one_sample_recall = one_sample_reachable_recall(scores=scores)
    mean_selected_edges = compactness_from_masks(
        node_masks=node_masks,
        edge_masks=edge_masks,
        batch=batch,
        device=node_masks.device,
    )["selected_edge_count"]

    return {
        "best_of_k_reachable_recall": best_recall,
        "one_sample_reachable_recall": one_sample_recall,
        "sampling_recall_gain": best_recall - one_sample_recall,
        "mean_selected_edges": float(mean_selected_edges),
        "budget_forced_stop_rate": budget_forced_stop_rate(
            rollout_samples=rollout_samples,
            valid_graph_mask=scores.valid_graph_mask,
        ),
    }


def reachable_recall_scores(
    *,
    node_masks: torch.Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> ReachableRecallScores:
    device = node_masks.device
    num_samples = int(node_masks.size(0))
    num_graphs = int(batch.num_graphs)
    num_nodes = int(batch.num_nodes_total)

    if num_samples == 0:
        empty = torch.zeros((0, num_graphs), dtype=torch.float32, device=device)
        valid_graph_mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        return ReachableRecallScores(
            recall=empty,
            valid_graph_mask=valid_graph_mask,
        )

    node_batch = batch.batch.to(device=device, dtype=torch.long)
    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )

    if exclude_anchors_from_retrieved:
        anchor_nodes = node_mask_from_ids(
            batch.anchor_node_ids,
            num_nodes=num_nodes,
            device=device,
            name="anchor_node_ids",
        )
    else:
        anchor_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    retrieved_nodes = node_masks & ~anchor_nodes.unsqueeze(0)
    hit_nodes = retrieved_nodes & target_nodes.unsqueeze(0)

    expanded_node_batch = _sample_node_graph_index(
        node_batch=node_batch,
        num_samples=num_samples,
        num_graphs=num_graphs,
    )

    hits_per_graph = _scatter_sum_1d(
        hit_nodes.float().reshape(-1),
        expanded_node_batch,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)
    gold_per_graph = _scatter_sum_1d(
        target_nodes.float(),
        node_batch,
        dim_size=num_graphs,
    )

    valid_graph_mask = gold_per_graph.gt(0.0)
    recall = safe_divide(
        hits_per_graph,
        gold_per_graph.unsqueeze(0).expand_as(hits_per_graph),
    )
    recall = torch.where(
        valid_graph_mask.unsqueeze(0),
        recall,
        torch.zeros_like(recall),
    )

    return ReachableRecallScores(
        recall=recall,
        valid_graph_mask=valid_graph_mask,
    )


def best_of_k_reachable_recall(
    *,
    scores: ReachableRecallScores,
    best_of_k: int,
) -> float:
    if scores.recall.numel() == 0:
        return 0.0

    effective_k = min(int(best_of_k), int(scores.recall.size(0)))
    if effective_k <= 0:
        return 0.0

    best_recall = scores.recall[:effective_k].max(dim=0).values
    return mean_over_valid_graphs(best_recall, scores.valid_graph_mask)


def one_sample_reachable_recall(
    scores: ReachableRecallScores,
) -> float:
    return mean_over_valid_graphs(scores.recall, scores.valid_graph_mask)


def budget_forced_stop_rate(
    *,
    rollout_samples: Sequence[RolloutResult],
    valid_graph_mask: torch.Tensor,
) -> float:
    if not rollout_samples:
        return 0.0

    rates: list[torch.Tensor] = []
    for rollout in rollout_samples:
        valid_rows = _valid_rollout_rows(
            rollout=rollout,
            valid_graph_mask=valid_graph_mask,
        )
        if not bool(valid_rows.any()):
            continue
        valid_steps = rollout.valid_mask & valid_rows.unsqueeze(1)
        row_forced_stop = (rollout.forced_stop_mask & valid_steps).any(dim=1)
        rates.append(row_forced_stop[valid_rows].float().mean())

    if not rates:
        return 0.0

    return float(torch.stack(rates).mean().item())


def _valid_rollout_rows(
    *,
    rollout: RolloutResult,
    valid_graph_mask: torch.Tensor,
) -> torch.Tensor:
    graph_valid = valid_graph_mask.to(device=rollout.device, dtype=torch.bool)
    graph_ids = rollout.source_graph_id.to(device=rollout.device, dtype=torch.long)

    if graph_ids.numel() > 0:
        if int(graph_ids.min()) < 0:
            raise ValueError("rollout.source_graph_id contains negative graph ids.")
        if int(graph_ids.max()) >= int(graph_valid.numel()):
            raise ValueError(
                "rollout.source_graph_id contains ids outside valid_graph_mask: "
                f"max_id={int(graph_ids.max())}, "
                f"num_graphs={int(graph_valid.numel())}."
            )

    return graph_valid.index_select(0, graph_ids)


def _scatter_sum_1d(
    values: torch.Tensor,
    index: torch.Tensor,
    *,
    dim_size: int,
) -> torch.Tensor:
    out = torch.zeros(int(dim_size), dtype=values.dtype, device=values.device)
    if values.numel() > 0:
        out.scatter_add_(
            0,
            index.to(device=values.device, dtype=torch.long),
            values,
        )
    return out


def _sample_node_graph_index(
    *,
    node_batch: torch.Tensor,
    num_samples: int,
    num_graphs: int,
) -> torch.Tensor:
    sample_offsets = torch.arange(
        int(num_samples),
        device=node_batch.device,
        dtype=torch.long,
    ).unsqueeze(1) * int(num_graphs)
    return (node_batch.unsqueeze(0) + sample_offsets).reshape(-1)


__all__ = [
    "ReachableRecallScores",
    "best_of_k_reachable_recall",
    "budget_forced_stop_rate",
    "evaluate_rollout_samples",
    "one_sample_reachable_recall",
    "reachable_recall_scores",
]
