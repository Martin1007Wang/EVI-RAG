from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.eval.compactness import compactness_from_masks
from src.eval.groups import MetricGroups
from src.eval.retrieval import (
    mean_over_valid_graphs,
    safe_divide,
    safe_f1,
)
from src.eval.targets import eval_target_node_mask
from src.graph.masks import anchor_node_mask, node_mask_from_ids
from src.weaver.rollout.result import RolloutResult
from src.weaver.rollout.subgraph import SubgraphReconstructor


@dataclass(frozen=True, slots=True)
class TargetNodeRetrievalScores:
    """
    Target-node retrieval scores for rollout samples.

    Shape convention:
        K: number of rollout samples
        B: number of source graphs in the batch
    """

    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    valid_graph_mask: torch.Tensor


def evaluate_rollout_samples(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    best_of_k_values: Sequence[int],
    utility_k: int,
    utility_lambda: float,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> MetricGroups:
    node_masks, edge_masks = SubgraphReconstructor(
        batch,
        device=torch.device("cpu"),
    ).stack(rollout_samples)

    scores = target_node_retrieval_scores(
        node_masks=node_masks,
        batch=batch,
        exclude_anchors_from_retrieved=exclude_anchors_from_retrieved,
        use_reachable_targets=use_reachable_targets,
    )

    best_of_k = best_of_k_metrics(
        scores=scores,
        best_of_k_values=best_of_k_values,
    )
    evidence = evidence_metrics(
        node_masks=node_masks,
        edge_masks=edge_masks,
        batch=batch,
    )

    return {
        "main": main_metrics(
            best_of_k=best_of_k,
            evidence=evidence,
            utility_k=utility_k,
            utility_lambda=utility_lambda,
        ),
        "sample": sample_average_metrics(
            rollout_samples=rollout_samples,
            scores=scores,
        ),
        "best_of_k": best_of_k,
        "behavior": rollout_behavior_metrics(
            rollout_samples=rollout_samples,
            batch=batch,
            valid_graph_mask=scores.valid_graph_mask,
            use_reachable_targets=use_reachable_targets,
        ),
        "evidence": evidence,
    }


def target_node_retrieval_scores(
    *,
    node_masks: torch.Tensor,
    batch: RetrievalBatch,
    exclude_anchors_from_retrieved: bool,
    use_reachable_targets: bool,
) -> TargetNodeRetrievalScores:
    device = node_masks.device
    num_samples = int(node_masks.size(0))
    num_graphs = int(batch.num_graphs)
    num_nodes = int(batch.num_nodes_total)

    if num_samples == 0:
        empty = torch.zeros((0, num_graphs), dtype=torch.float32, device=device)
        valid_graph_mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        return TargetNodeRetrievalScores(
            precision=empty,
            recall=empty,
            f1=empty,
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
    retrieved_per_graph = _scatter_sum_1d(
        retrieved_nodes.float().reshape(-1),
        expanded_node_batch,
        dim_size=num_samples * num_graphs,
    ).view(num_samples, num_graphs)
    gold_per_graph = _scatter_sum_1d(
        target_nodes.float(),
        node_batch,
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

    return TargetNodeRetrievalScores(
        precision=precision,
        recall=recall,
        f1=f1,
        valid_graph_mask=valid_graph_mask,
    )


def sample_average_metrics(
    *,
    rollout_samples: Sequence[RolloutResult],
    scores: TargetNodeRetrievalScores,
) -> dict[str, float]:
    if scores.f1.numel() == 0:
        return {
            "target_precision_mean": 0.0,
            "target_recall_mean": 0.0,
            "target_f1_mean": 0.0,
            "nonzero_f1_rate": 0.0,
            "full_recall_rate": 0.0,
        }

    full_recall = scores.recall.eq(1.0).to(dtype=torch.float32)
    nonzero_f1 = scores.f1.gt(0.0).to(dtype=torch.float32)

    del rollout_samples
    return {
        "target_precision_mean": mean_over_valid_graphs(
            scores.precision,
            scores.valid_graph_mask,
        ),
        "target_recall_mean": mean_over_valid_graphs(
            scores.recall,
            scores.valid_graph_mask,
        ),
        "target_f1_mean": mean_over_valid_graphs(
            scores.f1,
            scores.valid_graph_mask,
        ),
        "nonzero_f1_rate": mean_over_valid_graphs(
            nonzero_f1,
            scores.valid_graph_mask,
        ),
        "full_recall_rate": mean_over_valid_graphs(
            full_recall,
            scores.valid_graph_mask,
        ),
    }


def best_of_k_metrics(
    *,
    scores: TargetNodeRetrievalScores,
    best_of_k_values: Sequence[int],
) -> dict[str, float]:
    effective_ks = _effective_best_of_k_values(
        best_of_k_values,
        max_k=int(scores.f1.size(0)),
    )

    metrics: dict[str, float] = {}

    for k in effective_ks:
        if int(scores.f1.size(0)) == 0:
            metrics[f"target_precision_at_{k}"] = 0.0
            metrics[f"target_recall_at_{k}"] = 0.0
            metrics[f"target_f1_at_{k}"] = 0.0
            metrics[f"full_recall_rate_at_{k}"] = 0.0
            continue

        best_precision = scores.precision[:k].max(dim=0).values
        best_recall = scores.recall[:k].max(dim=0).values
        best_f1 = scores.f1[:k].max(dim=0).values
        full_recall = best_recall.eq(1.0).to(dtype=torch.float32)

        metrics[f"target_precision_at_{k}"] = mean_over_valid_graphs(
            best_precision,
            scores.valid_graph_mask,
        )
        metrics[f"target_recall_at_{k}"] = mean_over_valid_graphs(
            best_recall,
            scores.valid_graph_mask,
        )
        metrics[f"target_f1_at_{k}"] = mean_over_valid_graphs(
            best_f1,
            scores.valid_graph_mask,
        )
        metrics[f"full_recall_rate_at_{k}"] = mean_over_valid_graphs(
            full_recall,
            scores.valid_graph_mask,
        )

    if 1 in effective_ks and 8 in effective_ks:
        metrics["gain_f1_1_to_8"] = (
            metrics["target_f1_at_8"] - metrics["target_f1_at_1"]
        )
        metrics["gain_recall_1_to_8"] = (
            metrics["target_recall_at_8"] - metrics["target_recall_at_1"]
        )

    return metrics


def main_metrics(
    *,
    best_of_k: dict[str, float],
    evidence: dict[str, float],
    utility_k: int,
    utility_lambda: float,
) -> dict[str, float]:
    recall = _resolve_best_of_k_metric_value(
        metrics=best_of_k,
        metric_prefix="target_recall_at_",
        requested_k=int(utility_k),
    )
    selected_edge_count_mean = float(evidence.get("selected_edge_count_mean", 0.0))

    return {
        f"utility_at_{int(utility_k)}": recall
        - float(utility_lambda) * selected_edge_count_mean,
    }


def evidence_metrics(
    *,
    node_masks: torch.Tensor,
    edge_masks: torch.Tensor,
    batch: RetrievalBatch,
) -> dict[str, float]:
    compactness = compactness_from_masks(
        node_masks=node_masks,
        edge_masks=edge_masks,
        batch=batch,
        device=node_masks.device,
    )
    return {
        "selected_edge_count_mean": compactness["selected_edge_count"],
    }


def rollout_behavior_metrics(
    *,
    rollout_samples: Sequence[RolloutResult],
    batch: RetrievalBatch,
    valid_graph_mask: torch.Tensor,
    use_reachable_targets: bool,
) -> dict[str, float]:
    if not rollout_samples:
        return _zero_behavior_metrics()

    device = valid_graph_mask.device
    target_nodes = eval_target_node_mask(
        batch,
        device=device,
        use_reachable_targets=use_reachable_targets,
    )
    base_node_mask = anchor_node_mask(batch, device=device)
    edge_index = batch.edge_index.to(device=device, dtype=torch.long)

    model_stop_rates: list[torch.Tensor] = []
    forced_stop_rates: list[torch.Tensor] = []
    stop_given_hit_rates: list[torch.Tensor] = []
    stop_without_answer_rates: list[torch.Tensor] = []
    answer_hit_continue_rates: list[torch.Tensor] = []
    extra_expansions_after_first_hit: list[torch.Tensor] = []

    for rollout in rollout_samples:
        valid_rows = _valid_rollout_rows(
            rollout=rollout,
            valid_graph_mask=valid_graph_mask,
        )
        if not bool(valid_rows.any()):
            continue

        valid_steps = rollout.valid_mask & valid_rows.unsqueeze(1)
        row_stop = (rollout.stop_mask & valid_steps).any(dim=1)
        row_forced_stop = (rollout.forced_stop_mask & valid_steps).any(dim=1)
        row_model_stop = row_stop & ~row_forced_stop

        model_stop_rates.append(row_model_stop[valid_rows].float().mean())
        forced_stop_rates.append(row_forced_stop[valid_rows].float().mean())

        behavior = _per_row_hit_behavior(
            rollout=rollout,
            batch=batch,
            edge_index=edge_index,
            base_node_mask=base_node_mask,
            target_nodes=target_nodes,
            valid_rows=valid_rows,
        )

        if behavior is not None:
            stop_given_hit_rates.append(behavior.stop_given_hit_rate)
            stop_without_answer_rates.append(behavior.stop_without_answer_rate)
            answer_hit_continue_rates.append(behavior.answer_hit_then_continue_rate)
            extra_expansions_after_first_hit.append(
                behavior.extra_expansions_after_first_hit_mean
            )

    if not model_stop_rates:
        return _zero_behavior_metrics()

    return {
        "model_stop_rate": _mean_scalar_tensors(model_stop_rates),
        "forced_stop_rate": _mean_scalar_tensors(forced_stop_rates),
        "stop_given_answer_hit_rate": _mean_scalar_tensors(stop_given_hit_rates),
        "stop_without_answer_rate": _mean_scalar_tensors(stop_without_answer_rates),
        "answer_hit_then_continue_rate": _mean_scalar_tensors(
            answer_hit_continue_rates
        ),
        "extra_expansions_after_first_hit_mean": _mean_scalar_tensors(
            extra_expansions_after_first_hit
        ),
    }


def rollout_action_metrics(
    *,
    rollout_samples: Sequence[RolloutResult],
    valid_graph_mask: torch.Tensor,
) -> dict[str, float]:
    """
    Backward-compatible action summary used by older training helpers.
    """
    if not rollout_samples:
        return _zero_legacy_action_metrics()

    model_stop_rates: list[torch.Tensor] = []
    forced_stop_rates: list[torch.Tensor] = []
    stop_rates: list[torch.Tensor] = []
    expansion_counts: list[torch.Tensor] = []
    valid_step_counts: list[torch.Tensor] = []
    stop_depths: list[torch.Tensor] = []
    terminal_stop_log_probs: list[torch.Tensor] = []

    for rollout in rollout_samples:
        valid_rows = _valid_rollout_rows(
            rollout=rollout,
            valid_graph_mask=valid_graph_mask,
        )
        if not bool(valid_rows.any()):
            continue

        valid_steps = rollout.valid_mask & valid_rows.unsqueeze(1)
        row_stop = (rollout.stop_mask & valid_steps).any(dim=1)
        row_forced_stop = (rollout.forced_stop_mask & valid_steps).any(dim=1)
        row_model_stop = row_stop & ~row_forced_stop
        num_expansions = (rollout.expand_mask & valid_steps).sum(dim=1).float()
        num_valid_steps = rollout.traj_len.float()

        model_stop_rates.append(row_model_stop[valid_rows].float().mean())
        forced_stop_rates.append(row_forced_stop[valid_rows].float().mean())
        stop_rates.append(row_stop[valid_rows].float().mean())
        expansion_counts.append(num_expansions[valid_rows].mean())
        valid_step_counts.append(num_valid_steps[valid_rows].mean())
        stop_depths.append(num_expansions[row_stop & valid_rows].mean())
        terminal_stop_log_probs.append(
            rollout.terminal_stop_log_prob[valid_rows].float().mean()
        )

    if not model_stop_rates:
        return _zero_legacy_action_metrics()

    return {
        "model_stop_rate": _mean_scalar_tensors(model_stop_rates),
        "forced_stop_rate": _mean_scalar_tensors(forced_stop_rates),
        "stop_rate": _mean_scalar_tensors(stop_rates),
        "mean_num_expansions": _mean_scalar_tensors(expansion_counts),
        "mean_num_valid_steps": _mean_scalar_tensors(valid_step_counts),
        "mean_stop_depth": _mean_scalar_tensors(stop_depths),
        "mean_terminal_stop_log_prob": _mean_scalar_tensors(terminal_stop_log_probs),
    }


@dataclass(frozen=True, slots=True)
class _HitBehavior:
    stop_given_hit_rate: torch.Tensor
    stop_without_answer_rate: torch.Tensor
    answer_hit_then_continue_rate: torch.Tensor
    extra_expansions_after_first_hit_mean: torch.Tensor


def _per_row_hit_behavior(
    *,
    rollout: RolloutResult,
    batch: RetrievalBatch,
    edge_index: torch.Tensor,
    base_node_mask: torch.Tensor,
    target_nodes: torch.Tensor,
    valid_rows: torch.Tensor,
) -> _HitBehavior | None:
    del batch
    device = rollout.device
    num_rows = rollout.num_rollouts
    num_steps = rollout.max_steps

    stop_given_hit: list[torch.Tensor] = []
    stop_without_hit: list[torch.Tensor] = []
    hit_then_continue: list[torch.Tensor] = []
    extra_expands: list[torch.Tensor] = []

    for row in range(num_rows):
        if not bool(valid_rows[row]):
            continue

        node_mask = base_node_mask.clone()
        hit_step = -1
        stop_step = -1
        stop_found = False
        hit_found = bool((node_mask & target_nodes).any())

        for step in range(num_steps):
            if not bool(rollout.valid_mask[row, step]):
                continue

            if bool(rollout.expand_mask[row, step]):
                edge_id = int(rollout.selected_edge_ids[row, step].item())
                if edge_id >= 0:
                    src = int(edge_index[0, edge_id].item())
                    dst = int(edge_index[1, edge_id].item())
                    node_mask[src] = True
                    node_mask[dst] = True
                    if hit_step < 0 and bool((node_mask & target_nodes).any()):
                        hit_found = True
                        hit_step = step

            if bool(rollout.stop_mask[row, step]):
                stop_step = step
                stop_found = True
                break

        hit_tensor = torch.tensor(float(hit_found), device=device)
        stop_tensor = torch.tensor(float(stop_found), device=device)
        stop_given_hit.append(stop_tensor if hit_found else torch.tensor(0.0, device=device))
        stop_without_hit.append(
            stop_tensor if (stop_found and not hit_found) else torch.tensor(0.0, device=device)
        )

        if hit_step >= 0:
            post_hit_expand = rollout.expand_mask[row, hit_step + 1 :].float().sum()
            hit_then_continue.append(post_hit_expand.gt(0).float())
            extra_expands.append(post_hit_expand)
        else:
            hit_then_continue.append(torch.tensor(0.0, device=device))
            extra_expands.append(torch.tensor(0.0, device=device))

    if not stop_given_hit:
        return None

    hit_rate_denom = torch.stack(
        [torch.tensor(float(x.item() > 0.0), device=device) for x in stop_given_hit]
    ).sum()
    stop_given_hit_rate = (
        torch.stack(stop_given_hit).sum() / hit_rate_denom.clamp_min(1.0)
        if float(hit_rate_denom.item()) > 0.0
        else torch.tensor(0.0, device=device)
    )

    total_rows = float(len(stop_without_hit))
    return _HitBehavior(
        stop_given_hit_rate=stop_given_hit_rate,
        stop_without_answer_rate=torch.stack(stop_without_hit).sum()
        / max(total_rows, 1.0),
        answer_hit_then_continue_rate=torch.stack(hit_then_continue).mean(),
        extra_expansions_after_first_hit_mean=torch.stack(extra_expands).mean(),
    )


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


def _zero_behavior_metrics() -> dict[str, float]:
    return {
        "model_stop_rate": 0.0,
        "forced_stop_rate": 0.0,
        "stop_given_answer_hit_rate": 0.0,
        "stop_without_answer_rate": 0.0,
        "answer_hit_then_continue_rate": 0.0,
        "extra_expansions_after_first_hit_mean": 0.0,
    }


def _zero_legacy_action_metrics() -> dict[str, float]:
    return {
        "model_stop_rate": 0.0,
        "forced_stop_rate": 0.0,
        "stop_rate": 0.0,
        "mean_num_expansions": 0.0,
        "mean_num_valid_steps": 0.0,
        "mean_stop_depth": 0.0,
        "mean_terminal_stop_log_prob": 0.0,
    }


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


def _effective_best_of_k_values(
    best_of_k_values: Sequence[int],
    *,
    max_k: int,
) -> tuple[int, ...]:
    requested = {int(k) for k in best_of_k_values if int(k) >= 1}
    if int(max_k) < 1:
        return tuple(sorted(requested))
    return tuple(sorted(k for k in requested if k <= int(max_k)))


def _resolve_best_of_k_metric_value(
    *,
    metrics: dict[str, float],
    metric_prefix: str,
    requested_k: int,
) -> float:
    exact_key = f"{metric_prefix}{int(requested_k)}"
    if exact_key in metrics:
        return float(metrics[exact_key])

    candidates: list[tuple[int, float]] = []
    for key, value in metrics.items():
        if not key.startswith(metric_prefix):
            continue
        suffix = key[len(metric_prefix) :]
        if suffix.isdigit():
            current_k = int(suffix)
            if current_k <= int(requested_k):
                candidates.append((current_k, float(value)))

    if not candidates:
        return 0.0

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _mean_scalar_tensors(values: Sequence[torch.Tensor]) -> float:
    finite_values = [
        value.detach().float()
        for value in values
        if value.numel() == 1 and bool(torch.isfinite(value.detach()).all())
    ]
    if not finite_values:
        return 0.0
    return float(torch.stack(finite_values).mean().item())


__all__ = [
    "TargetNodeRetrievalScores",
    "best_of_k_metrics",
    "evaluate_rollout_samples",
    "evidence_metrics",
    "main_metrics",
    "rollout_action_metrics",
    "rollout_behavior_metrics",
    "sample_average_metrics",
    "target_node_retrieval_scores",
]
