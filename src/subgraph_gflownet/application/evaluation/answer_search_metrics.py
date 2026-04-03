from __future__ import annotations

import math
from typing import Any

import torch


def _topk_metrics(
    *, predicted_entities: list[int], gold_entities: list[int], top_ks: tuple[int, ...]
) -> dict[str, float]:
    gold_set = {int(entity_id) for entity_id in gold_entities}
    if not gold_set:
        return {
            **{f"answer/hit@{int(k)}": 0.0 for k in top_ks},
            **{f"answer/recall@{int(k)}": 0.0 for k in top_ks},
        }
    metrics: dict[str, float] = {}
    for k in top_ks:
        top_predicted = predicted_entities[: int(k)]
        hits = gold_set.intersection(top_predicted)
        metrics[f"answer/hit@{int(k)}"] = 1.0 if hits else 0.0
        metrics[f"answer/recall@{int(k)}"] = float(len(hits)) / float(len(gold_set))
    return metrics


def _mean_metric_dict(metric_rows: list[dict[str, float]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    return {
        key: float(sum(float(row.get(key, 0.0)) for row in metric_rows))
        / float(len(metric_rows))
        for key in keys
    }


def _split_terminal_answer_log_mass(
    *, probability_mass: float, terminal_answer_set_entity_ids: tuple[int, ...]
) -> float | None:
    if probability_mass <= 0.0 or not terminal_answer_set_entity_ids:
        return None
    return float(
        math.log(float(probability_mass) / float(len(terminal_answer_set_entity_ids)))
    )


def _clamped_positive_weight(log_value: float) -> float:
    if not math.isfinite(log_value):
        return 0.0
    return float(math.exp(max(min(float(log_value), 80.0), -80.0)))


def _terminal_action_index(
    *, sample_batch: Any, graph_idx: int, rollout_idx: int
) -> int:
    action_index = (
        int(sample_batch.termination_action_steps[graph_idx, rollout_idx].item()) - 1
    )
    if action_index < 0:
        raise RuntimeError(
            "terminal action index must be >= 0 for Monte Carlo aggregation."
        )
    return int(action_index)


def _trajectory_log_prob(
    *, sample_batch: Any, graph_idx: int, rollout_idx: int
) -> float:
    action_mask = sample_batch.action_mask[graph_idx, rollout_idx].to(dtype=torch.bool)
    if not bool(action_mask.any().item()):
        return float("-inf")
    return float(
        sample_batch.log_pf_actions[graph_idx, rollout_idx]
        .to(dtype=torch.float32)[action_mask]
        .sum()
        .item()
    )


def _rollout_support_weight(
    *, sample_batch: Any, graph_idx: int, rollout_idx: int, backend: str
) -> float:
    if backend == "vote":
        return 1.0
    terminal_action_idx = _terminal_action_index(
        sample_batch=sample_batch,
        graph_idx=graph_idx,
        rollout_idx=rollout_idx,
    )
    if backend == "terminal_reward":
        return _clamped_positive_weight(
            float(
                sample_batch.log_reward_actions[
                    graph_idx, rollout_idx, terminal_action_idx
                ].item()
            )
        )
    if backend == "trajectory_prob":
        return _clamped_positive_weight(
            _trajectory_log_prob(
                sample_batch=sample_batch,
                graph_idx=graph_idx,
                rollout_idx=rollout_idx,
            )
        )
    if backend == "terminal_flow":
        return _clamped_positive_weight(
            float(
                sample_batch.state_log_flows[
                    graph_idx, rollout_idx, terminal_action_idx
                ].item()
            )
        )
    if backend == "hybrid":
        return _clamped_positive_weight(
            _trajectory_log_prob(
                sample_batch=sample_batch,
                graph_idx=graph_idx,
                rollout_idx=rollout_idx,
            )
            + float(
                sample_batch.log_reward_actions[
                    graph_idx, rollout_idx, terminal_action_idx
                ].item()
            )
        )
    raise ValueError(
        f"Unsupported Monte Carlo answer aggregation backend: {backend!r}."
    )


def _graph_candidate_answer_upper_bound(*, prepared_batch: Any, graph_idx: int) -> int:
    node_start = int(prepared_batch.node_ptr[graph_idx].item())
    node_end = int(prepared_batch.node_ptr[graph_idx + 1].item())
    return max(int(node_end - node_start), 1)


def _topk_stability_margin(
    *,
    answer_vote_counts: dict[int, float],
    executed_rollouts: int,
    candidate_answer_upper_bound: int,
    confidence: float,
    stability_top_k: int,
) -> float | None:
    if executed_rollouts < 1 or stability_top_k < 1:
        return None
    ranked_counts = sorted(
        answer_vote_counts.items(),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    if len(ranked_counts) < int(stability_top_k):
        return None
    delta = max(1.0 - float(confidence), 1.0e-12)
    support_size = max(int(candidate_answer_upper_bound), 1)
    radius = math.sqrt(
        math.log((4.0 * float(support_size)) / float(delta))
        / (2.0 * float(executed_rollouts))
    )
    kth_probability = float(ranked_counts[int(stability_top_k) - 1][1]) / float(
        executed_rollouts
    )
    next_probability = 0.0
    if len(ranked_counts) > int(stability_top_k):
        next_probability = float(ranked_counts[int(stability_top_k)][1]) / float(
            executed_rollouts
        )
    lower_bound = kth_probability - radius
    unseen_upper_bound = radius
    next_upper_bound = next_probability + radius
    return float(lower_bound - max(next_upper_bound, unseen_upper_bound))


def _topk_metrics_from_result(
    *, result: dict[str, Any], top_ks: tuple[int, ...]
) -> dict[str, float]:
    return _topk_metrics(
        predicted_entities=list(result["predicted_answer_entity_ids"]),
        gold_entities=list(result["gold_answer_entity_ids"]),
        top_ks=top_ks,
    )


def _support_mass_metrics_from_result(
    *, result: dict[str, Any], edge_top_ks: tuple[int, ...]
) -> dict[str, float]:
    witness_support_probabilities = [
        float(value) for value in result.get("witness_support_probabilities", [])
    ]
    return {
        f"support/mass@{int(k)}": float(sum(witness_support_probabilities[: int(k)]))
        for k in edge_top_ks
    }


def _secondary_metrics_from_result(
    *, result: dict[str, Any], edge_top_ks: tuple[int, ...]
) -> dict[str, float]:
    rollout_count = float(max(int(result["rollout_count"]), 1))
    secondary = {
        "answer_search/requested_rollout_count": float(
            result.get("requested_rollout_count", result["rollout_count"])
        ),
        "answer_search/rollout_count": float(result["rollout_count"]),
        "answer_search/early_stop_rate": 1.0
        if bool(result.get("stopped_early", False))
        else 0.0,
        "answer_search/nonempty_terminal_answer_set_rate": float(
            result["nonempty_terminal_answer_set_rollout_count"]
        )
        / rollout_count,
        "witness/gold_answer_in_state_rate": float(
            result["gold_answer_in_state_rollout_count"]
        )
        / rollout_count,
        "answer_search/predicted_answer_count": float(
            len(result["predicted_answer_entity_ids"])
        ),
        "witness/terminal_witness_count": float(result["terminal_witness_count"]),
        "witness/mean_stop_step": float(result["mean_stop_step"]),
        "witness/mean_anchor_component_count": float(
            result["mean_terminal_component_count"]
        ),
    }
    secondary.update(
        _support_mass_metrics_from_result(result=result, edge_top_ks=edge_top_ks)
    )
    return secondary


def _accumulate_metric_sums(
    totals: dict[str, float], row: dict[str, float]
) -> dict[str, float]:
    for key, value in row.items():
        totals[key] = float(totals.get(key, 0.0)) + float(value)
    return totals


def _average_metric_sums(totals: dict[str, float], *, count: int) -> dict[str, float]:
    if count <= 0:
        return {}
    return {key: float(value) / float(count) for key, value in sorted(totals.items())}


def _summarize_result_rows(
    *,
    results: list[dict[str, Any]],
    answer_top_ks: tuple[int, ...],
    edge_top_ks: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    primary_sums: dict[str, float] = {}
    secondary_sums: dict[str, float] = {}
    for result in results:
        _accumulate_metric_sums(
            primary_sums,
            _topk_metrics_from_result(result=result, top_ks=answer_top_ks),
        )
        _accumulate_metric_sums(
            secondary_sums,
            _secondary_metrics_from_result(result=result, edge_top_ks=edge_top_ks),
        )
    count = len(results)
    return (
        _average_metric_sums(primary_sums, count=count),
        _average_metric_sums(secondary_sums, count=count),
    )


__all__ = [
    "_accumulate_metric_sums",
    "_average_metric_sums",
    "_clamped_positive_weight",
    "_graph_candidate_answer_upper_bound",
    "_mean_metric_dict",
    "_rollout_support_weight",
    "_secondary_metrics_from_result",
    "_split_terminal_answer_log_mass",
    "_summarize_result_rows",
    "_support_mass_metrics_from_result",
    "_terminal_action_index",
    "_topk_metrics",
    "_topk_metrics_from_result",
    "_topk_stability_margin",
    "_trajectory_log_prob",
]
