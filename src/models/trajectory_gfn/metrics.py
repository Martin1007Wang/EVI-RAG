from __future__ import annotations

import math

from .schema import ElasticEvalBatch, ElasticWindowResult


def format_mass_threshold_tag(mass_threshold: float) -> str:
    return str(int(round(float(mass_threshold) * 100.0)))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _entropy_from_probs(probs: list[float]) -> float:
    total = sum(probs)
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for prob in probs:
        if prob <= 0.0:
            continue
        norm_prob = prob / total
        entropy -= norm_prob * math.log(norm_prob)
    return float(entropy)


def _context_stats(result: ElasticWindowResult) -> tuple[float, float, float, float]:
    answer_set = set(result.gold_answer_entity_ids)
    if not answer_set:
        return 0.0, 0.0, 0.0, 0.0
    start_set = set(result.start_entity_ids)
    context_nodes = set(start_set)
    for trajectory in result.trajectories:
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


def _support_path_diversity(result: ElasticWindowResult) -> float:
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


def compute_elastic_metrics(eval_batch: ElasticEvalBatch) -> dict[str, float]:
    results = eval_batch.results
    if not results:
        return {}
    rho_tag = format_mass_threshold_tag(eval_batch.mass_threshold)
    elastic_hit = []
    elastic_answer_recall = []
    elastic_context_recall = []
    elastic_context_precision = []
    elastic_context_f1 = []
    elastic_window_size = []
    elastic_mass = []
    elastic_unique_answers = []
    elastic_unique_paths = []
    path_entropy = []
    answer_entropy = []
    tail_rollout_mass = []
    covered_gold_mass = []
    missed_gold_mass = []
    gold_total_mass = []
    selected_answer_mass = []
    selected_answer_count = []
    support_conditioned_mass_min = []
    support_conditioned_mass_mean = []
    support_path_diversity = []
    probe_count = []
    emit_path_count = []
    remaining_mass_upper = []

    for result in results:
        gold_set = set(result.gold_answer_entity_ids)
        covered_answers = {traj.terminal_entity_id for traj in result.trajectories}
        hit = 1.0 if any(traj.is_gold for traj in result.trajectories) else 0.0
        answer_recall = (
            float(len(covered_answers & gold_set)) / float(len(gold_set))
            if gold_set
            else 0.0
        )
        ctx_recall, ctx_precision, ctx_f1, _ = _context_stats(result)
        probs = [float(traj.prob) for traj in result.trajectories]
        answer_prob_map: dict[int, float] = {}
        for traj in result.trajectories:
            answer_prob_map[traj.terminal_entity_id] = answer_prob_map.get(
                traj.terminal_entity_id, 0.0
            ) + float(traj.prob)

        path_h = _entropy_from_probs(probs)
        answer_h = _entropy_from_probs(list(answer_prob_map.values()))
        elastic_hit.append(hit)
        elastic_answer_recall.append(answer_recall)
        elastic_context_recall.append(ctx_recall)
        elastic_context_precision.append(ctx_precision)
        elastic_context_f1.append(ctx_f1)
        elastic_window_size.append(float(result.window_size))
        elastic_mass.append(float(result.covered_mass))
        elastic_unique_answers.append(float(result.unique_answer_count))
        elastic_unique_paths.append(float(result.unique_path_count))
        path_entropy.append(path_h)
        answer_entropy.append(answer_h)
        tail_rollout_mass.append(float(result.tail_rollout_mass))
        covered_gold_mass.append(float(result.covered_gold_mass))
        missed_gold_mass.append(float(result.missed_gold_mass))
        gold_total_mass.append(float(result.gold_total_mass))
        selected_answers = [
            record for record in result.answer_posterior if record.is_selected
        ]
        selected_answer_mass.append(
            float(sum(record.prob for record in selected_answers))
        )
        selected_answer_count.append(float(len(selected_answers)))
        conditioned = [
            float(record.support_conditioned_mass) for record in selected_answers
        ]
        support_conditioned_mass_min.append(min(conditioned) if conditioned else 0.0)
        support_conditioned_mass_mean.append(_safe_mean(conditioned))
        support_path_diversity.append(_support_path_diversity(result))
        probe_count.append(float(result.probe_count))
        emit_path_count.append(float(result.emit_path_count))
        remaining_mass_upper.append(float(result.remaining_mass_upper))

    metrics = {
        f"elastic_hit@{rho_tag}": _safe_mean(elastic_hit),
        f"elastic_answer_recall@{rho_tag}": _safe_mean(elastic_answer_recall),
        f"elastic_context_recall@{rho_tag}": _safe_mean(elastic_context_recall),
        f"elastic_context_precision@{rho_tag}": _safe_mean(elastic_context_precision),
        f"elastic_context_f1@{rho_tag}": _safe_mean(elastic_context_f1),
        f"elastic_window_size@{rho_tag}": _safe_mean(elastic_window_size),
        f"elastic_mass@{rho_tag}": _safe_mean(elastic_mass),
        f"elastic_unique_answers@{rho_tag}": _safe_mean(elastic_unique_answers),
        f"elastic_unique_paths@{rho_tag}": _safe_mean(elastic_unique_paths),
        f"path_entropy@{rho_tag}": _safe_mean(path_entropy),
        f"answer_entropy@{rho_tag}": _safe_mean(answer_entropy),
        f"effective_paths@{rho_tag}": _safe_mean(
            [math.exp(val) for val in path_entropy]
        ),
        f"effective_answers@{rho_tag}": _safe_mean(
            [math.exp(val) for val in answer_entropy]
        ),
        f"tail_rollout_mass@{rho_tag}": _safe_mean(tail_rollout_mass),
        f"covered_gold_mass@{rho_tag}": _safe_mean(covered_gold_mass),
        f"missed_gold_mass@{rho_tag}": _safe_mean(missed_gold_mass),
        f"gold_total_mass@{rho_tag}": _safe_mean(gold_total_mass),
        f"selected_answer_mass@{rho_tag}": _safe_mean(selected_answer_mass),
        f"selected_answer_count@{rho_tag}": _safe_mean(selected_answer_count),
        f"support_conditioned_mass_min@{rho_tag}": _safe_mean(
            support_conditioned_mass_min
        ),
        f"support_conditioned_mass_mean@{rho_tag}": _safe_mean(
            support_conditioned_mass_mean
        ),
        f"support_path_diversity@{rho_tag}": _safe_mean(support_path_diversity),
        f"probe_count@{rho_tag}": _safe_mean(probe_count),
        f"emit_path_count@{rho_tag}": _safe_mean(emit_path_count),
        f"remaining_mass_upper@{rho_tag}": _safe_mean(remaining_mass_upper),
        "num_samples": float(len(results)),
    }
    metrics["elastic_hit"] = metrics[f"elastic_hit@{rho_tag}"]
    metrics["elastic_answer_recall"] = metrics[f"elastic_answer_recall@{rho_tag}"]
    metrics["elastic_context_recall"] = metrics[f"elastic_context_recall@{rho_tag}"]
    metrics["elastic_context_precision"] = metrics[
        f"elastic_context_precision@{rho_tag}"
    ]
    metrics["elastic_context_f1"] = metrics[f"elastic_context_f1@{rho_tag}"]
    metrics["elastic_window_size"] = metrics[f"elastic_window_size@{rho_tag}"]
    metrics["elastic_mass"] = metrics[f"elastic_mass@{rho_tag}"]
    metrics["path_entropy"] = metrics[f"path_entropy@{rho_tag}"]
    metrics["answer_entropy"] = metrics[f"answer_entropy@{rho_tag}"]
    metrics["tail_rollout_mass"] = metrics[f"tail_rollout_mass@{rho_tag}"]
    metrics["covered_gold_mass"] = metrics[f"covered_gold_mass@{rho_tag}"]
    metrics["missed_gold_mass"] = metrics[f"missed_gold_mass@{rho_tag}"]
    metrics["gold_total_mass"] = metrics[f"gold_total_mass@{rho_tag}"]
    metrics["selected_answer_mass"] = metrics[f"selected_answer_mass@{rho_tag}"]
    metrics["selected_answer_count"] = metrics[f"selected_answer_count@{rho_tag}"]
    metrics["support_conditioned_mass_min"] = metrics[
        f"support_conditioned_mass_min@{rho_tag}"
    ]
    metrics["support_conditioned_mass_mean"] = metrics[
        f"support_conditioned_mass_mean@{rho_tag}"
    ]
    metrics["support_path_diversity"] = metrics[f"support_path_diversity@{rho_tag}"]
    metrics["probe_count"] = metrics[f"probe_count@{rho_tag}"]
    metrics["emit_path_count"] = metrics[f"emit_path_count@{rho_tag}"]
    metrics["remaining_mass_upper"] = metrics[f"remaining_mass_upper@{rho_tag}"]
    return metrics
