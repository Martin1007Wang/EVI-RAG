from __future__ import annotations

from typing import Iterable, Sequence


def compute_topk_set_metrics(
    *,
    ranked_ids: Sequence[int],
    relevant_ids: set[int],
    top_ks: Sequence[int],
    prefix: str,
    include_precision: bool = True,
    include_f1: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for top_k in top_ks:
        top_id_list = [int(value) for value in ranked_ids[: int(top_k)]]
        top_id_set = set(top_id_list)
        retrieved_count = len(top_id_list)
        hit_count = len(top_id_set & relevant_ids)
        precision = (
            float(hit_count) / float(retrieved_count) if retrieved_count > 0 else 0.0
        )
        recall = float(hit_count) / float(len(relevant_ids)) if relevant_ids else 0.0
        metrics[f"{prefix}/hit@{top_k}"] = 1.0 if hit_count > 0 else 0.0
        metrics[f"{prefix}/recall@{top_k}"] = recall
        if include_precision:
            metrics[f"{prefix}/precision@{top_k}"] = precision
        if include_precision and include_f1:
            metrics[f"{prefix}/f1@{top_k}"] = (
                (2.0 * precision * recall) / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
    return metrics


def reciprocal_rank(first_relevant_rank: int | None) -> float:
    if first_relevant_rank is None or int(first_relevant_rank) < 1:
        return 0.0
    return 1.0 / float(first_relevant_rank)


def mean_metric_dicts(metric_dicts: Iterable[dict[str, float]]) -> dict[str, float]:
    metric_list = list(metric_dicts)
    if not metric_list:
        return {}
    return {
        name: float(
            sum(metrics[name] for metrics in metric_list) / float(len(metric_list))
        )
        for name in metric_list[0]
    }


__all__ = [
    "compute_topk_set_metrics",
    "mean_metric_dicts",
    "reciprocal_rank",
]
