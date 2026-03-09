from __future__ import annotations

from dataclasses import dataclass
import math

from .analyzer import AnswerMassAnalysis
from .batch import TrajectoryBatch
from .schema import (
    AnswerPosteriorRecord,
    EdgeRecord,
    ElasticWindowResult,
    TrajectoryRecord,
)

_MASS_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class DiscoveredTrajectory:
    start_node: int
    terminal_node: int
    answer_entity_id: int
    edge_ids: tuple[int, ...]
    log_prob: float
    is_gold: bool

    @property
    def prob(self) -> float:
        return float(math.exp(self.log_prob))


def graph_gold_answers(*, batch: TrajectoryBatch) -> set[int]:
    return {int(value) for value in batch.answer_entity_ids.tolist()}


def graph_start_entity_ids(*, batch: TrajectoryBatch) -> list[int]:
    return [
        int(batch.node_global_ids[node_idx].item())
        for node_idx in batch.q_local_indices.tolist()
    ]


def build_answer_posterior(
    *,
    analysis: AnswerMassAnalysis,
    gold_answers: set[int],
    answer_mass_threshold: float,
) -> tuple[list[AnswerPosteriorRecord], list[int]]:
    answer_ids = [int(value) for value in analysis.answer_entity_ids.tolist()]
    answer_probs = [float(value) for value in analysis.answer_probs.tolist()]
    order = sorted(
        range(len(answer_ids)),
        key=lambda idx: (-answer_probs[idx], answer_ids[idx]),
    )
    cumulative = 0.0
    records: list[AnswerPosteriorRecord] = []
    selected_answer_ids: list[int] = []
    threshold = float(answer_mass_threshold)
    for idx in order:
        answer_id = answer_ids[idx]
        prob = answer_probs[idx]
        previous_cumulative = cumulative
        cumulative += prob
        is_selected = prob > 0.0 and (
            not selected_answer_ids or previous_cumulative < threshold - _MASS_TOLERANCE
        )
        if is_selected:
            selected_answer_ids.append(answer_id)
        records.append(
            AnswerPosteriorRecord(
                answer_entity_id=answer_id,
                prob=prob,
                cumulative_mass=min(cumulative, 1.0),
                is_gold=answer_id in gold_answers,
                is_selected=is_selected,
            )
        )
    return records, selected_answer_ids


def build_edge_records(
    *, batch: TrajectoryBatch, edge_ids: tuple[int, ...]
) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    for edge_id in edge_ids:
        src = int(batch.edge_index[0, edge_id].item())
        dst = int(batch.edge_index[1, edge_id].item())
        records.append(
            EdgeRecord(
                edge_id=int(edge_id),
                src_entity_id=int(batch.node_global_ids[src].item()),
                relation_id=int(batch.edge_rel_global[edge_id].item()),
                dst_entity_id=int(batch.node_global_ids[dst].item()),
            )
        )
    return records


def support_targets(
    *,
    answer_records: list[AnswerPosteriorRecord],
    selected_answer_ids: list[int],
    support_mass_threshold: float,
) -> dict[int, float]:
    answer_mass = {
        record.answer_entity_id: float(record.prob) for record in answer_records
    }
    threshold = float(support_mass_threshold)
    return {
        answer_id: threshold * answer_mass.get(answer_id, 0.0)
        for answer_id in selected_answer_ids
    }


def _path_edge_overlap(lhs: DiscoveredTrajectory, rhs: DiscoveredTrajectory) -> float:
    if lhs.edge_ids == rhs.edge_ids:
        return 1.0
    if not lhs.edge_ids or not rhs.edge_ids:
        return 0.0
    lhs_edges = set(lhs.edge_ids)
    rhs_edges = set(rhs.edge_ids)
    overlap = len(lhs_edges & rhs_edges)
    if overlap == 0:
        return 0.0
    denom = max(len(lhs_edges), len(rhs_edges))
    return float(overlap) / float(denom)


def _select_support_paths_for_answer(
    *,
    answer_paths: list[DiscoveredTrajectory],
    target_mass: float,
    overlap_penalty: float,
) -> tuple[list[DiscoveredTrajectory], float]:
    if target_mass <= 0.0 or not answer_paths:
        return [], 0.0
    selected: list[DiscoveredTrajectory] = []
    remaining = list(answer_paths)
    accumulated_mass = 0.0
    while remaining and accumulated_mass + _MASS_TOLERANCE < target_mass:
        if not selected or overlap_penalty <= 0.0:
            best_idx = max(
                range(len(remaining)),
                key=lambda idx: (remaining[idx].prob, -len(remaining[idx].edge_ids)),
            )
        else:
            best_idx = max(
                range(len(remaining)),
                key=lambda idx: (
                    math.log(max(remaining[idx].prob, _MASS_TOLERANCE))
                    - overlap_penalty
                    * max(
                        _path_edge_overlap(remaining[idx], chosen)
                        for chosen in selected
                    ),
                    remaining[idx].prob,
                    -len(remaining[idx].edge_ids),
                ),
            )
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        accumulated_mass += chosen.prob
    return selected, accumulated_mass


def build_window_result(
    *,
    batch: TrajectoryBatch,
    discovered_paths: list[DiscoveredTrajectory],
    analysis: AnswerMassAnalysis,
    inference_mode: str,
    answer_mass_threshold: float,
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
    probe_count: int,
    remaining_mass_upper: float,
    stop_reason: str,
) -> ElasticWindowResult:
    gold_answers = graph_gold_answers(batch=batch)
    answer_records, selected_answer_ids = build_answer_posterior(
        analysis=analysis,
        gold_answers=gold_answers,
        answer_mass_threshold=answer_mass_threshold,
    )
    selected_set = set(selected_answer_ids)
    emitted_paths, updated_answers = _build_support_outputs(
        batch=batch,
        discovered_paths=discovered_paths,
        answer_records=answer_records,
        selected_set=selected_set,
        support_mass_threshold=support_mass_threshold,
        support_path_overlap_penalty=support_path_overlap_penalty,
    )
    covered_mass = sum(path.prob for path in emitted_paths)
    covered_gold_mass = sum(path.prob for path in emitted_paths if path.is_gold)
    return ElasticWindowResult(
        sample_id=batch.sample_ids[0],
        dataset_scope=batch.dataset_scope,
        mass_threshold=float(support_mass_threshold),
        window_size=len(emitted_paths),
        covered_mass=covered_mass,
        tail_rollout_mass=max(1.0 - covered_mass, 0.0),
        gold_total_mass=float(analysis.gold_total_mass),
        covered_gold_mass=covered_gold_mass,
        missed_gold_mass=max(float(analysis.gold_total_mass) - covered_gold_mass, 0.0),
        unique_answer_count=len({path.terminal_entity_id for path in emitted_paths}),
        unique_path_count=len(emitted_paths),
        gold_answer_entity_ids=sorted(gold_answers),
        start_entity_ids=graph_start_entity_ids(batch=batch),
        trajectories=emitted_paths,
        inference_mode=str(inference_mode),
        answer_mass_threshold=float(answer_mass_threshold),
        support_mass_threshold=float(support_mass_threshold),
        probe_count=int(probe_count),
        emit_path_count=len(emitted_paths),
        remaining_mass_upper=max(float(remaining_mass_upper), 0.0),
        stop_reason=str(stop_reason),
        selected_answer_ids=selected_answer_ids,
        answer_posterior=updated_answers,
    )


def build_rank_only_result_from_discovered_paths(
    *,
    batch: TrajectoryBatch,
    discovered_paths: list[DiscoveredTrajectory],
    inference_mode: str,
    answer_mass_threshold: float,
    probe_count: int,
    remaining_mass_upper: float,
    stop_reason: str,
) -> ElasticWindowResult:
    gold_answers = graph_gold_answers(batch=batch)
    answer_mass: dict[int, float] = {}
    for path in discovered_paths:
        answer_mass[path.answer_entity_id] = (
            answer_mass.get(path.answer_entity_id, 0.0) + path.prob
        )
    ordered = sorted(answer_mass.items(), key=lambda item: (-item[1], item[0]))
    total_mass = sum(answer_mass.values())
    norm = total_mass if total_mass > _MASS_TOLERANCE else 1.0
    cumulative = 0.0
    selected_answer_ids: list[int] = []
    answer_posterior: list[AnswerPosteriorRecord] = []
    threshold = float(answer_mass_threshold)
    for answer_id, raw_prob in ordered:
        prob = raw_prob / norm
        previous = cumulative
        cumulative += prob
        is_selected = prob > 0.0 and (
            not selected_answer_ids or previous < threshold - _MASS_TOLERANCE
        )
        if is_selected:
            selected_answer_ids.append(int(answer_id))
        answer_posterior.append(
            AnswerPosteriorRecord(
                answer_entity_id=int(answer_id),
                prob=float(prob),
                cumulative_mass=min(float(cumulative), 1.0),
                is_gold=int(answer_id) in gold_answers,
                is_selected=is_selected,
            )
        )
    gold_total_mass = sum(
        record.prob for record in answer_posterior if bool(record.is_gold)
    )
    return ElasticWindowResult(
        sample_id=batch.sample_ids[0],
        dataset_scope=batch.dataset_scope,
        mass_threshold=1.0,
        window_size=0,
        covered_mass=float(total_mass),
        tail_rollout_mass=max(1.0 - float(total_mass), 0.0),
        gold_total_mass=float(gold_total_mass),
        covered_gold_mass=float(gold_total_mass),
        missed_gold_mass=max(1.0 - float(gold_total_mass), 0.0),
        unique_answer_count=len(answer_posterior),
        unique_path_count=len(discovered_paths),
        gold_answer_entity_ids=sorted(gold_answers),
        start_entity_ids=graph_start_entity_ids(batch=batch),
        trajectories=[],
        inference_mode=str(inference_mode),
        answer_mass_threshold=float(answer_mass_threshold),
        support_mass_threshold=1.0,
        probe_count=int(probe_count),
        emit_path_count=0,
        remaining_mass_upper=max(float(remaining_mass_upper), 0.0),
        stop_reason=str(stop_reason),
        selected_answer_ids=selected_answer_ids,
        answer_posterior=answer_posterior,
    )


def compute_rank_metrics(
    *, answer_records: list[AnswerPosteriorRecord], answer_top_ks: tuple[int, ...]
) -> dict[str, float]:
    gold_answers = {
        int(record.answer_entity_id)
        for record in answer_records
        if bool(record.is_gold)
    }
    ordered_answer_ids = [int(record.answer_entity_id) for record in answer_records]
    metrics: dict[str, float] = {}
    metrics["pass@1"] = (
        1.0 if ordered_answer_ids and ordered_answer_ids[0] in gold_answers else 0.0
    )
    for top_k in answer_top_ks:
        top_answers = set(ordered_answer_ids[: int(top_k)])
        metrics[f"hit@{top_k}"] = 1.0 if top_answers & gold_answers else 0.0
        metrics[f"answer_recall@{top_k}"] = (
            float(len(top_answers & gold_answers)) / float(len(gold_answers))
            if gold_answers
            else 0.0
        )
    return metrics


def aggregate_rank_metrics(
    *,
    results: list[ElasticWindowResult],
    answer_top_ks: tuple[int, ...],
) -> dict[str, float]:
    aggregated: dict[str, float] = {
        "num_samples": float(len(results)),
    }
    if not results:
        return aggregated
    per_graph = [
        compute_rank_metrics(
            answer_records=result.answer_posterior,
            answer_top_ks=answer_top_ks,
        )
        for result in results
    ]
    for name in per_graph[0]:
        aggregated[name] = float(
            sum(metrics[name] for metrics in per_graph) / float(len(per_graph))
        )
    return aggregated


def _build_support_outputs(
    *,
    batch: TrajectoryBatch,
    discovered_paths: list[DiscoveredTrajectory],
    answer_records: list[AnswerPosteriorRecord],
    selected_set: set[int],
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
) -> tuple[list[TrajectoryRecord], list[AnswerPosteriorRecord]]:
    answer_mass = {
        record.answer_entity_id: float(record.prob) for record in answer_records
    }
    support_paths: list[TrajectoryRecord] = []
    support_summary: dict[int, tuple[float, int]] = {}
    answer_rank = {
        record.answer_entity_id: idx
        for idx, record in enumerate(answer_records, start=1)
    }
    for answer_id in selected_set:
        answer_paths = [
            path for path in discovered_paths if path.answer_entity_id == answer_id
        ]
        answer_paths.sort(
            key=lambda item: (-item.prob, item.answer_entity_id, item.edge_ids)
        )
        target_mass = float(support_mass_threshold) * answer_mass.get(answer_id, 0.0)
        selected_paths, cumulative_mass = _select_support_paths_for_answer(
            answer_paths=answer_paths,
            target_mass=target_mass,
            overlap_penalty=float(support_path_overlap_penalty),
        )
        kept_count = 0
        cumulative_selected_mass = 0.0
        for support_rank, path in enumerate(selected_paths, start=1):
            cumulative_selected_mass += path.prob
            support_paths.append(
                TrajectoryRecord(
                    sample_id=batch.sample_ids[0],
                    rollout_rank=0,
                    log_prob=float(path.log_prob),
                    prob=float(path.prob),
                    cumulative_mass=0.0,
                    terminal_entity_id=int(path.answer_entity_id),
                    is_gold=bool(path.is_gold),
                    edges=build_edge_records(batch=batch, edge_ids=path.edge_ids),
                    start_entity_id=int(batch.node_global_ids[path.start_node].item()),
                    answer_rank=int(answer_rank.get(answer_id, 0)),
                    support_rank=int(support_rank),
                    conditional_prob=(
                        float(path.prob) / answer_mass[answer_id]
                        if answer_mass.get(answer_id, 0.0) > 0.0
                        else 0.0
                    ),
                    conditional_cumulative_mass=(
                        cumulative_selected_mass / answer_mass[answer_id]
                        if answer_mass.get(answer_id, 0.0) > 0.0
                        else 0.0
                    ),
                )
            )
            kept_count = support_rank
        support_summary[answer_id] = (cumulative_mass, kept_count)
    support_paths.sort(
        key=lambda record: (
            -float(record.prob),
            int(record.answer_rank),
            int(record.support_rank),
            int(record.terminal_entity_id),
        )
    )
    cumulative_global_mass = 0.0
    ranked_support_paths: list[TrajectoryRecord] = []
    for rollout_rank, record in enumerate(support_paths, start=1):
        cumulative_global_mass += float(record.prob)
        ranked_support_paths.append(
            TrajectoryRecord(
                sample_id=record.sample_id,
                rollout_rank=int(rollout_rank),
                log_prob=float(record.log_prob),
                prob=float(record.prob),
                cumulative_mass=float(cumulative_global_mass),
                terminal_entity_id=int(record.terminal_entity_id),
                is_gold=bool(record.is_gold),
                edges=list(record.edges),
                start_entity_id=record.start_entity_id,
                answer_rank=int(record.answer_rank),
                support_rank=int(record.support_rank),
                conditional_prob=float(record.conditional_prob),
                conditional_cumulative_mass=float(record.conditional_cumulative_mass),
            )
        )
    updated_answers: list[AnswerPosteriorRecord] = []
    for record in answer_records:
        support_mass, support_path_count = support_summary.get(
            record.answer_entity_id, (0.0, 0)
        )
        conditioned_mass = (
            support_mass / float(record.prob) if float(record.prob) > 0.0 else 0.0
        )
        updated_answers.append(
            AnswerPosteriorRecord(
                answer_entity_id=int(record.answer_entity_id),
                prob=float(record.prob),
                cumulative_mass=float(record.cumulative_mass),
                is_gold=bool(record.is_gold),
                is_selected=bool(record.answer_entity_id in selected_set),
                support_mass=float(support_mass),
                support_conditioned_mass=float(conditioned_mass),
                support_path_count=int(support_path_count),
            )
        )
    return ranked_support_paths, updated_answers
