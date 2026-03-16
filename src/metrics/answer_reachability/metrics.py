from __future__ import annotations

from .schema import SupportWindowEvalBatch, SupportWindowResult, TrajectoryRecord


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _context_stats(
    result: SupportWindowResult,
    *,
    trajectories: list[TrajectoryRecord] | None = None,
) -> tuple[float, float, float, float]:
    answer_set = set(result.gold_answer_entity_ids)
    if not answer_set:
        return 0.0, 0.0, 0.0, 0.0
    start_set = set(result.start_entity_ids)
    context_nodes = set(start_set)
    path_records = result.trajectories if trajectories is None else trajectories
    for trajectory in path_records:
        context_nodes.add(int(trajectory.terminal_entity_id))
        for edge in trajectory.edges:
            context_nodes.add(int(edge.src_entity_id))
            context_nodes.add(int(edge.dst_entity_id))
    recall = float(len(context_nodes & answer_set)) / float(len(answer_set))
    context_nonstart = context_nodes - start_set
    answer_nonstart = answer_set - start_set
    precision = (
        float(len(context_nonstart & answer_nonstart)) / float(len(context_nonstart))
        if context_nonstart
        else 0.0
    )
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0.0 else 0.0
    hit = 1.0 if recall > 0.0 else 0.0
    return recall, precision, f1, hit


def _prefix_window_stats(
    result: SupportWindowResult, *, top_k: int
) -> tuple[float, float, float, float]:
    kept = list(result.trajectories[: int(top_k)])
    recall, precision, f1, hit = _context_stats(result, trajectories=kept)
    return hit, recall, precision, f1


def _support_path_diversity(result: SupportWindowResult) -> float:
    per_answer_paths: dict[int, list[set[int]]] = {}
    for trajectory in result.trajectories:
        per_answer_paths.setdefault(int(trajectory.terminal_entity_id), []).append(
            {int(edge.edge_id) for edge in trajectory.edges}
        )
    if not per_answer_paths:
        return 0.0
    per_answer_diversity: list[float] = []
    for edge_sets in per_answer_paths.values():
        if len(edge_sets) <= 1:
            per_answer_diversity.append(1.0)
            continue
        pairwise_overlap: list[float] = []
        for left_idx in range(len(edge_sets)):
            for right_idx in range(left_idx + 1, len(edge_sets)):
                left = edge_sets[left_idx]
                right = edge_sets[right_idx]
                denom = max(len(left), len(right), 1)
                pairwise_overlap.append(float(len(left & right)) / float(denom))
        per_answer_diversity.append(1.0 - _safe_mean(pairwise_overlap))
    return _safe_mean(per_answer_diversity)


def compute_support_metrics(eval_batch: SupportWindowEvalBatch) -> dict[str, float]:
    results = eval_batch.results
    if not results:
        return {}

    metric_values: dict[str, list[float]] = {
        "window/adaptive/hit": [],
        "window/adaptive/recall": [],
        "window/adaptive/precision": [],
        "window/adaptive/f1": [],
        "window/adaptive/path_count": [],
        "window/adaptive/answer_count": [],
        "window/adaptive/path_mass": [],
        "window/adaptive/gold_mass": [],
        "window/adaptive/missed_gold_mass": [],
        "window/adaptive/support_coverage_mean": [],
        "window/adaptive/support_coverage_min": [],
        "window/adaptive/support_diversity": [],
        "inference/probe_count": [],
        "cert/remaining_mass_upper": [],
        "cert/coverage_rate": [],
    }
    window_top_ks = tuple(
        sorted({int(k) for k in eval_batch.window_top_ks if int(k) >= 1})
    )
    prefix_metrics: dict[int, dict[str, list[float]]] = {
        top_k: {
            "hit": [],
            "recall": [],
            "precision": [],
            "f1": [],
        }
        for top_k in window_top_ks
    }

    for result in results:
        recall, precision, f1, hit = _context_stats(result)
        selected_answers = [
            record for record in result.answer_posterior if bool(record.is_selected)
        ]
        support_coverage = [
            float(record.support_conditioned_mass) for record in selected_answers
        ]
        metric_values["window/adaptive/hit"].append(hit)
        metric_values["window/adaptive/recall"].append(recall)
        metric_values["window/adaptive/precision"].append(precision)
        metric_values["window/adaptive/f1"].append(f1)
        metric_values["window/adaptive/path_count"].append(float(result.window_size))
        metric_values["window/adaptive/answer_count"].append(
            float(len(selected_answers))
        )
        metric_values["window/adaptive/path_mass"].append(float(result.covered_mass))
        metric_values["window/adaptive/gold_mass"].append(
            float(result.covered_gold_mass)
        )
        metric_values["window/adaptive/missed_gold_mass"].append(
            float(result.missed_gold_mass)
        )
        metric_values["window/adaptive/support_coverage_mean"].append(
            _safe_mean(support_coverage)
        )
        metric_values["window/adaptive/support_coverage_min"].append(
            min(support_coverage) if support_coverage else 0.0
        )
        metric_values["window/adaptive/support_diversity"].append(
            _support_path_diversity(result)
        )
        metric_values["inference/probe_count"].append(float(result.probe_count))
        metric_values["cert/remaining_mass_upper"].append(
            float(result.remaining_mass_upper)
        )
        metric_values["cert/coverage_rate"].append(
            1.0 if result.coverage_certified else 0.0
        )
        for top_k in window_top_ks:
            topk_hit, topk_recall, topk_precision, topk_f1 = _prefix_window_stats(
                result,
                top_k=top_k,
            )
            prefix_metrics[top_k]["hit"].append(topk_hit)
            prefix_metrics[top_k]["recall"].append(topk_recall)
            prefix_metrics[top_k]["precision"].append(topk_precision)
            prefix_metrics[top_k]["f1"].append(topk_f1)

    metrics = {"meta/num_samples": float(len(results))}
    for name, values in metric_values.items():
        metrics[name] = _safe_mean(values)
    for top_k in window_top_ks:
        metric_prefix = f"window/top{top_k}"
        metrics[f"{metric_prefix}/hit"] = _safe_mean(prefix_metrics[top_k]["hit"])
        metrics[f"{metric_prefix}/recall"] = _safe_mean(prefix_metrics[top_k]["recall"])
        metrics[f"{metric_prefix}/precision"] = _safe_mean(
            prefix_metrics[top_k]["precision"]
        )
        metrics[f"{metric_prefix}/f1"] = _safe_mean(prefix_metrics[top_k]["f1"])
    return metrics
