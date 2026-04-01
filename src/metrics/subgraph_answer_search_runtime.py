from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from pathlib import Path
from typing import Any, Callable

from src.graph import TrajectoryBatch
from src.metrics.prediction_io import append_jsonl_records
from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.data.schema.constants import EntityVocabFields, RelationVocabFields
from src.models.gflownet.policy import SubgraphPolicy
from src.models.gflownet.sampler import SubgraphSampler
from src.utils.cuda_memory import profile_cuda_memory

from .base import BaseMetricRuntime
from .protocol import MetricEvaluationOutput


class _SubgraphPredictionCodec:
    kind = "subgraph_answer_search"

    @staticmethod
    def serialize_result(result: Any) -> dict[str, Any]:
        return dict(result)

    @staticmethod
    def serialize_label(label: Any) -> dict[str, Any]:
        return dict(label)

    @staticmethod
    def deserialize_result(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)

    @staticmethod
    def deserialize_label(record: dict[str, Any]) -> dict[str, Any]:
        return dict(record)


@dataclass
class _TerminalSampleAggregate:
    edge_ids: tuple[int, ...]
    selected_node_ids: tuple[int, ...]
    reachability_bits: dict[int, int]
    chosen_answer_entity_id: int | None
    answer_entities: tuple[int, ...]
    sample_count: int = 0


@dataclass
class _GraphPredictionAccumulator:
    original_graph_idx: int
    sample_id: str
    question: str
    gold_answer_entity_ids: list[int]
    a_entity_in_graph: bool
    candidate_answer_upper_bound: int
    answer_vote_counts: dict[int, int] = field(default_factory=dict)
    terminal_subgraphs: dict[
        tuple[tuple[int, ...], int | None], _TerminalSampleAggregate
    ] = field(default_factory=dict)
    answering_rollout_count: int = 0
    hit_rollout_count: int = 0
    total_stop_steps: float = 0.0
    total_terminal_component_count: float = 0.0
    rollout_count: int = 0
    early_stop_margin: float | None = None
    stopped_early: bool = False


@dataclass
class _PredictMetricsAccumulator:
    count: int = 0
    primary_sums: dict[str, float] = field(default_factory=dict)
    secondary_sums: dict[str, float] = field(default_factory=dict)


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
    *, probability_mass: float, answer_entities: tuple[int, ...]
) -> float | None:
    if probability_mass <= 0.0 or not answer_entities:
        return None
    return float(math.log(float(probability_mass)))


def _graph_candidate_answer_upper_bound(*, prepared_batch: Any, graph_idx: int) -> int:
    node_start = int(prepared_batch.node_ptr[graph_idx].item())
    node_end = int(prepared_batch.node_ptr[graph_idx + 1].item())
    return max(int(node_end - node_start), 1)


def _topk_stability_margin(
    *,
    answer_vote_counts: dict[int, int],
    executed_rollouts: int,
    candidate_answer_upper_bound: int,
    confidence: float,
    stability_top_k: int,
) -> float | None:
    if executed_rollouts < 1 or stability_top_k < 1:
        return None
    ranked_counts = sorted(
        answer_vote_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
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
    support_probabilities = [
        float(value) for value in result.get("support_probabilities", [])
    ]
    return {
        f"support/mass@{int(k)}": float(sum(support_probabilities[: int(k)]))
        for k in edge_top_ks
    }


def _secondary_metrics_from_result(
    *, result: dict[str, Any], edge_top_ks: tuple[int, ...]
) -> dict[str, float]:
    rollout_count = float(max(int(result["rollout_count"]), 1))
    secondary = {
        "answer_commit/requested_rollout_count": float(
            result.get("requested_rollout_count", result["rollout_count"])
        ),
        "answer_commit/rollout_count": float(result["rollout_count"]),
        "answer_commit/early_stop_rate": 1.0
        if bool(result.get("stopped_early", False))
        else 0.0,
        "answer_commit/commit_rate": float(result["answering_rollout_count"])
        / rollout_count,
        "answer_commit/gold_commit_rate": float(result["hit_rollout_count"])
        / rollout_count,
        "answer_commit/predicted_answer_count": float(
            len(result["predicted_answer_entity_ids"])
        ),
        "witness/terminal_count": float(result["terminal_subgraph_count"]),
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


def _edge_overlap_ratio(
    edge_ids: tuple[int, ...], other_edge_ids: tuple[int, ...]
) -> float:
    edge_set = set(int(edge_id) for edge_id in edge_ids)
    other_set = set(int(edge_id) for edge_id in other_edge_ids)
    if not edge_set or not other_set:
        return 0.0
    return float(len(edge_set.intersection(other_set))) / float(
        len(edge_set.union(other_set))
    )


def _select_terminal_support(
    *,
    ranked_terminals: list[_TerminalSampleAggregate],
    executed_rollouts: int,
    edge_emit_top_k: int,
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
) -> list[_TerminalSampleAggregate]:
    if edge_emit_top_k < 1 or not ranked_terminals:
        return []
    selected: list[_TerminalSampleAggregate] = []
    remaining = list(ranked_terminals)
    accumulated_mass = 0.0
    while remaining and len(selected) < int(edge_emit_top_k):
        if accumulated_mass >= float(support_mass_threshold) and selected:
            break
        if not selected or support_path_overlap_penalty <= 0.0:
            chosen_idx = 0
        else:
            best_score = float("-inf")
            chosen_idx = 0
            for idx, payload in enumerate(remaining):
                probability = float(payload.sample_count) / float(
                    max(executed_rollouts, 1)
                )
                max_overlap = max(
                    _edge_overlap_ratio(payload.edge_ids, chosen.edge_ids)
                    for chosen in selected
                )
                score = float(probability) - (
                    float(support_path_overlap_penalty) * float(max_overlap)
                )
                if score > best_score:
                    best_score = score
                    chosen_idx = idx
        chosen = remaining.pop(chosen_idx)
        selected.append(chosen)
        accumulated_mass += float(chosen.sample_count) / float(
            max(executed_rollouts, 1)
        )
    return selected


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


def _edge_records_from_terminal(
    *, batch: TrajectoryBatch, graph_idx: int, edge_ids: tuple[int, ...]
) -> list[dict[str, int]]:
    edge_start = int(batch.edge_ptr[graph_idx].item())
    records: list[dict[str, int]] = []
    for edge_id in edge_ids:
        edge_idx = int(edge_start + int(edge_id))
        src_node = int(batch.edge_index[0, edge_idx].item())
        dst_node = int(batch.edge_index[1, edge_idx].item())
        records.append(
            {
                "edge_id": int(edge_id),
                "src_entity_id": int(batch.node_entity_ids[src_node].item()),
                "relation_id": int(batch.edge_rel_global[edge_idx].item()),
                "dst_entity_id": int(batch.node_entity_ids[dst_node].item()),
            }
        )
    return records


def _trajectory_text_from_edge_records(edge_records: list[dict[str, int]]) -> str:
    if not edge_records:
        return "(start_only)"
    return " ; ".join(
        (f"{edge['src_entity_id']} --{edge['relation_id']}--> {edge['dst_entity_id']}")
        for edge in edge_records
    )


def _build_support_records(
    *,
    batch: TrajectoryBatch,
    graph_idx: int,
    support_payloads: list[_TerminalSampleAggregate],
    executed_rollouts: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_rank, payload in enumerate(support_payloads, start=1):
        probability = float(payload.sample_count) / float(max(executed_rollouts, 1))
        edge_records = _edge_records_from_terminal(
            batch=batch,
            graph_idx=graph_idx,
            edge_ids=payload.edge_ids,
        )
        terminal_entity_id = payload.chosen_answer_entity_id
        records.append(
            {
                "path_rank": int(path_rank),
                "edge_ids": [int(edge_id) for edge_id in payload.edge_ids],
                "selected_node_ids": [
                    int(node_id) for node_id in payload.selected_node_ids
                ],
                "answer_entities": [
                    int(entity_id) for entity_id in payload.answer_entities
                ],
                "chosen_answer_entity_id": (
                    None if terminal_entity_id is None else int(terminal_entity_id)
                ),
                "sample_count": int(payload.sample_count),
                "probability": float(probability),
                "prob": float(probability),
                "per_answer_log_mass": _split_terminal_answer_log_mass(
                    probability_mass=probability,
                    answer_entities=payload.answer_entities,
                ),
                "edges": edge_records,
                "trajectory_text": (
                    _trajectory_text_from_edge_records(edge_records)
                    if edge_records
                    else (
                        f"(start_only) {int(terminal_entity_id)}"
                        if terminal_entity_id is not None
                        else "(start_only)"
                    )
                ),
                "terminal_entity_id": (
                    None if terminal_entity_id is None else int(terminal_entity_id)
                ),
            }
        )
    return records


def _build_graph_prediction_accumulator(
    *,
    batch: TrajectoryBatch,
    prepared_batch: Any,
    graph_idx: int,
    original_graph_idx: int,
) -> _GraphPredictionAccumulator:
    gold_answer_entity_ids = [
        int(value)
        for value in batch.answer_entity_ids[
            int(batch.answer_ptr[graph_idx].item()) : int(
                batch.answer_ptr[graph_idx + 1].item()
            )
        ].tolist()
    ]
    graph_node_entities = prepared_batch.graph_node_entities[graph_idx]
    return _GraphPredictionAccumulator(
        original_graph_idx=int(original_graph_idx),
        sample_id=str(batch.sample_ids[graph_idx]),
        question=str(batch.questions[graph_idx]),
        gold_answer_entity_ids=gold_answer_entity_ids,
        a_entity_in_graph=bool(
            set(int(entity_id) for entity_id in gold_answer_entity_ids).intersection(
                graph_node_entities
            )
        ),
        candidate_answer_upper_bound=_graph_candidate_answer_upper_bound(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
        ),
    )


def _finalize_graph_result(
    *,
    accumulator: _GraphPredictionAccumulator,
    batch: TrajectoryBatch,
    include_answer_support: bool,
    edge_emit_top_k: int,
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
    requested_rollouts: int,
) -> dict[str, Any]:
    executed_rollouts = max(int(accumulator.rollout_count), 1)
    ranked_answers = sorted(
        accumulator.answer_vote_counts.items(),
        key=lambda item: (-int(item[1]), int(item[0])),
    )
    ranked_terminals = sorted(
        accumulator.terminal_subgraphs.values(),
        key=lambda item: (-int(item.sample_count), item.edge_ids),
    )
    top_subgraph = ranked_terminals[0] if ranked_terminals else None
    selected_support = _select_terminal_support(
        ranked_terminals=ranked_terminals,
        executed_rollouts=executed_rollouts,
        edge_emit_top_k=int(edge_emit_top_k),
        support_mass_threshold=float(support_mass_threshold),
        support_path_overlap_penalty=float(support_path_overlap_penalty),
    )
    support_records = _build_support_records(
        batch=batch,
        graph_idx=int(accumulator.original_graph_idx),
        support_payloads=selected_support,
        executed_rollouts=executed_rollouts,
    )
    result: dict[str, Any] = {
        "sample_id": str(accumulator.sample_id),
        "question": str(accumulator.question),
        "gold_answer_entity_ids": list(accumulator.gold_answer_entity_ids),
        "predicted_answer_entity_ids": [
            int(entity_id) for entity_id, _ in ranked_answers
        ],
        "answer_log_masses": [
            float(math.log(float(votes) / float(executed_rollouts)))
            for _, votes in ranked_answers
        ],
        "requested_rollout_count": int(requested_rollouts),
        "rollout_count": int(accumulator.rollout_count),
        "answering_rollout_count": int(accumulator.answering_rollout_count),
        "hit_rollout_count": int(accumulator.hit_rollout_count),
        "terminal_subgraph_count": int(len(ranked_terminals)),
        "mean_stop_step": float(accumulator.total_stop_steps)
        / float(executed_rollouts),
        "mean_terminal_component_count": float(
            accumulator.total_terminal_component_count
        )
        / float(executed_rollouts),
        "stopped_early": bool(accumulator.stopped_early),
        "early_stop_margin": accumulator.early_stop_margin,
        "a_entity_in_graph": bool(accumulator.a_entity_in_graph),
        "support_probabilities": [
            float(record["probability"]) for record in support_records
        ],
    }
    if top_subgraph is not None:
        top_probability = float(top_subgraph.sample_count) / float(executed_rollouts)
        result["top_subgraph_edge_ids"] = [
            int(edge_id) for edge_id in top_subgraph.edge_ids
        ]
        result["top_subgraph_node_ids"] = [
            int(node_id) for node_id in top_subgraph.selected_node_ids
        ]
        result["top_subgraph_probability"] = float(top_probability)
        result["top_subgraph_sample_count"] = int(top_subgraph.sample_count)
        result["top_subgraph_answer_entity_id"] = (
            None
            if top_subgraph.chosen_answer_entity_id is None
            else int(top_subgraph.chosen_answer_entity_id)
        )
    if include_answer_support:
        result["terminal_subgraphs"] = support_records
        result["support_probability_mass"] = float(
            sum(float(record["probability"]) for record in support_records)
        )
    return result


def _load_vocab_label_map(
    *, path: str | Path | None, id_field: str, label_field: str
) -> dict[int, str]:
    if path in (None, ""):
        return {}
    try:
        pq_module = importlib.import_module("pyarrow.parquet")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional at runtime
        raise ModuleNotFoundError(
            "pyarrow is required to load prediction artifact vocab files."
        ) from exc
    resolved = Path(str(path))
    if not resolved.exists():
        raise FileNotFoundError(f"Missing vocab parquet: {resolved}")
    table = pq_module.read_table(resolved)
    payload = table.to_pydict()
    ids = payload.get(id_field, [])
    labels = payload.get(label_field, [])
    return {
        int(entity_id): str(label)
        for entity_id, label in zip(ids, labels)
        if entity_id is not None
    }


def _decorate_trajectory_records(
    *,
    trajectories: list[dict[str, Any]],
    entity_labels: dict[int, str],
    relation_labels: dict[int, str],
) -> list[dict[str, Any]]:
    decorated: list[dict[str, Any]] = []
    for trajectory in trajectories:
        edges = [dict(edge) for edge in trajectory.get("edges", [])]
        for edge in edges:
            src_entity_id = edge.get("src_entity_id")
            relation_id = edge.get("relation_id")
            dst_entity_id = edge.get("dst_entity_id")
            if src_entity_id is not None:
                edge["src_text"] = str(
                    entity_labels.get(int(src_entity_id), str(src_entity_id))
                )
            if relation_id is not None:
                edge["relation_text"] = str(
                    relation_labels.get(int(relation_id), str(relation_id))
                )
            if dst_entity_id is not None:
                edge["dst_text"] = str(
                    entity_labels.get(int(dst_entity_id), str(dst_entity_id))
                )
        terminal_entity_id = trajectory.get("terminal_entity_id")
        decorated_trajectory = dict(trajectory)
        decorated_trajectory["edges"] = edges
        if terminal_entity_id is not None:
            decorated_trajectory["terminal_entity_text"] = str(
                entity_labels.get(int(terminal_entity_id), str(terminal_entity_id))
            )
        if edges:
            decorated_trajectory["trajectory_text"] = " ; ".join(
                f"{edge.get('src_text', edge.get('src_entity_id'))} --"
                f"{edge.get('relation_text', edge.get('relation_id'))}--> "
                f"{edge.get('dst_text', edge.get('dst_entity_id'))}"
                for edge in edges
            )
        decorated.append(decorated_trajectory)
    return decorated


class SubgraphAnswerSearchRuntime(BaseMetricRuntime):
    sampler: Any

    def __init__(
        self,
        *,
        eval_cfg: dict[str, Any],
        policy: SubgraphPolicy,
        sampler: SubgraphSampler,
    ) -> None:
        self.eval_cfg = normalize_search_eval_cfg(eval_cfg)
        self.policy = policy
        self.sampler = sampler
        self.search = self
        self._prediction_codec = _SubgraphPredictionCodec()

    def _predict_single_graph(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> dict[str, Any]:
        return self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )[0]

    def _predict_batch_results(
        self,
        *,
        batch: TrajectoryBatch,
        include_answer_support: bool,
    ) -> list[dict[str, Any]]:
        monte_carlo_cfg = self.eval_cfg["monte_carlo"]
        requested_rollouts = int(monte_carlo_cfg["rollouts"])
        batch_rollouts = min(
            requested_rollouts,
            int(monte_carlo_cfg.get("batch_rollouts", requested_rollouts)),
        )
        confidence = float(monte_carlo_cfg["confidence"])
        temperature = float(monte_carlo_cfg["temperature"])
        early_stop_cfg = monte_carlo_cfg["early_stop"]
        early_stop_enabled = bool(early_stop_cfg["enabled"])
        early_stop_min_rollouts = min(
            requested_rollouts,
            int(early_stop_cfg["min_rollouts"]),
        )
        stability_top_k = int(early_stop_cfg["stability_top_k"])
        action_pruning_cfg = monte_carlo_cfg["action_pruning"]

        active_graph_indices = list(range(int(batch.num_graphs)))
        with profile_cuda_memory(
            "eval.prepare_batch.initial",
            device=batch.edge_index.device,
            extra=(
                f"num_graphs={int(batch.num_graphs)} requested_rollouts={requested_rollouts} "
                f"batch_rollouts={batch_rollouts}"
            ),
        ):
            full_prepared_batch = self.policy.prepare_batch(batch)
        active_prepared_batch = full_prepared_batch
        accumulators = {
            int(graph_idx): _build_graph_prediction_accumulator(
                batch=batch,
                prepared_batch=full_prepared_batch,
                graph_idx=int(graph_idx),
                original_graph_idx=int(graph_idx),
            )
            for graph_idx in active_graph_indices
        }

        while active_graph_indices:
            processed_rollouts = accumulators[active_graph_indices[0]].rollout_count
            remaining_rollouts = int(requested_rollouts) - int(processed_rollouts)
            if remaining_rollouts <= 0:
                break
            current_rollouts = min(int(batch_rollouts), int(remaining_rollouts))
            chunk_extra = (
                f"active_graphs={len(active_graph_indices)} current_rollouts={current_rollouts} "
                f"processed_rollouts={processed_rollouts}"
            )
            with profile_cuda_memory(
                "eval.sampler.sample",
                device=active_prepared_batch.device,
                extra=chunk_extra,
            ):
                sample_batch = self.sampler.sample(
                    policy=self.policy,
                    prepared_batch=active_prepared_batch,
                    rollouts_per_graph=current_rollouts,
                    temperature=temperature,
                    proposal_bias_scale=0.0,
                    action_pruning=action_pruning_cfg,
                )
            next_active_graph_indices: list[int] = []
            for local_graph_idx, original_graph_idx in enumerate(active_graph_indices):
                accumulator = accumulators[original_graph_idx]
                original_node_start = int(batch.node_ptr[original_graph_idx].item())
                original_edge_start = int(batch.edge_ptr[original_graph_idx].item())
                stop_steps = (
                    sample_batch.termination_action_steps[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                terminal_component_counts = (
                    sample_batch.terminal_component_counts[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                hit_mask = (
                    sample_batch.terminal_hit_mask[local_graph_idx]
                    .detach()
                    .cpu()
                    .tolist()
                )
                accumulator.total_stop_steps += float(
                    sum(int(step) for step in stop_steps)
                )
                accumulator.total_terminal_component_count += float(
                    sum(int(count) for count in terminal_component_counts)
                )
                accumulator.hit_rollout_count += int(
                    sum(bool(value) for value in hit_mask)
                )

                for rollout_idx in range(current_rollouts):
                    flat_rollout_idx = (
                        local_graph_idx * current_rollouts
                    ) + rollout_idx
                    global_edge_ids = tuple(
                        int(edge_id)
                        for edge_id in sample_batch.terminal_edge_ids[flat_rollout_idx]
                    )
                    edge_ids = tuple(
                        int(edge_id) - int(original_edge_start)
                        for edge_id in global_edge_ids
                    )
                    selected_node_ids = tuple(
                        int(node_id) - int(original_node_start)
                        for node_id in sample_batch.terminal_node_ids[flat_rollout_idx]
                    )
                    reachability_bits = {
                        int(node_id) - int(original_node_start): int(bits)
                        for node_id, bits in sample_batch.terminal_reachability_bits[
                            flat_rollout_idx
                        ].items()
                    }
                    chosen_answer_entity_id = -1
                    if sample_batch.chosen_answer_entity_ids is not None:
                        chosen_answer_entity_id = int(
                            sample_batch.chosen_answer_entity_ids[
                                local_graph_idx, rollout_idx
                            ].item()
                        )
                    answer_entities = (
                        ()
                        if chosen_answer_entity_id < 0
                        else (int(chosen_answer_entity_id),)
                    )
                    if chosen_answer_entity_id >= 0:
                        accumulator.answering_rollout_count += 1
                        accumulator.answer_vote_counts[int(chosen_answer_entity_id)] = (
                            int(
                                accumulator.answer_vote_counts.get(
                                    int(chosen_answer_entity_id), 0
                                )
                            )
                            + 1
                        )
                    payload = accumulator.terminal_subgraphs.get(
                        (
                            edge_ids,
                            None
                            if chosen_answer_entity_id < 0
                            else chosen_answer_entity_id,
                        )
                    )
                    if payload is None:
                        payload = _TerminalSampleAggregate(
                            edge_ids=edge_ids,
                            selected_node_ids=selected_node_ids,
                            reachability_bits=reachability_bits,
                            chosen_answer_entity_id=(
                                None
                                if chosen_answer_entity_id < 0
                                else int(chosen_answer_entity_id)
                            ),
                            answer_entities=answer_entities,
                        )
                        accumulator.terminal_subgraphs[
                            (
                                edge_ids,
                                None
                                if chosen_answer_entity_id < 0
                                else int(chosen_answer_entity_id),
                            )
                        ] = payload
                    payload.sample_count += 1

                accumulator.rollout_count += int(current_rollouts)
                if accumulator.rollout_count >= int(requested_rollouts):
                    continue
                if not early_stop_enabled or accumulator.rollout_count < int(
                    early_stop_min_rollouts
                ):
                    next_active_graph_indices.append(int(original_graph_idx))
                    continue
                accumulator.early_stop_margin = _topk_stability_margin(
                    answer_vote_counts=accumulator.answer_vote_counts,
                    executed_rollouts=accumulator.rollout_count,
                    candidate_answer_upper_bound=accumulator.candidate_answer_upper_bound,
                    confidence=confidence,
                    stability_top_k=stability_top_k,
                )
                if (
                    accumulator.early_stop_margin is not None
                    and accumulator.early_stop_margin > 0.0
                ):
                    accumulator.stopped_early = True
                    continue
                next_active_graph_indices.append(int(original_graph_idx))

            if next_active_graph_indices == active_graph_indices:
                continue
            active_graph_indices = next_active_graph_indices
            if not active_graph_indices:
                break
            with profile_cuda_memory(
                "eval.prepared_batch.select_graphs",
                device=full_prepared_batch.device,
                extra=f"active_graphs={len(active_graph_indices)}",
            ):
                active_prepared_batch = full_prepared_batch.select_graphs(
                    active_graph_indices
                )

        return [
            _finalize_graph_result(
                accumulator=accumulators[graph_idx],
                batch=batch,
                include_answer_support=include_answer_support,
                edge_emit_top_k=int(self.eval_cfg["edge_emit_top_k"]),
                support_mass_threshold=float(self.eval_cfg["support_mass_threshold"]),
                support_path_overlap_penalty=float(
                    self.eval_cfg["support_path_overlap_penalty"]
                ),
                requested_rollouts=int(requested_rollouts),
            )
            for graph_idx in range(int(batch.num_graphs))
        ]

    def evaluate_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> MetricEvaluationOutput:
        del report_profile, on_invalid_start
        results = self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )
        primary_metrics, secondary_metrics = _summarize_result_rows(
            results=results,
            answer_top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
            edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
        )
        return MetricEvaluationOutput(
            model_metrics={},
            primary_metrics=primary_metrics,
            secondary_metrics=secondary_metrics,
            results=results,
        )

    def predict_batch(
        self,
        *,
        batch: TrajectoryBatch,
        report_profile: str,
        include_answer_support: bool,
        on_invalid_start: Callable[[TrajectoryBatch], None] | None = None,
    ) -> list[Any]:
        del report_profile, on_invalid_start
        return self._predict_batch_results(
            batch=batch,
            include_answer_support=include_answer_support,
        )

    def build_predict_labels(
        self, batch: TrajectoryBatch, outputs: list[Any]
    ) -> list[Any]:
        del batch
        return [
            {
                "sample_id": str(output["sample_id"]),
                "question": str(output["question"]),
                "gold_answer_entity_ids": list(output["gold_answer_entity_ids"]),
                "a_entity_in_graph": bool(output.get("a_entity_in_graph", False)),
            }
            for output in outputs
        ]

    def summarize_predict_epoch(
        self,
        *,
        predict_results: list[Any],
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        primary_metrics, secondary_metrics = _summarize_result_rows(
            results=[dict(result) for result in predict_results],
            answer_top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
            edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
        )
        return {**primary_metrics, **secondary_metrics}

    def initialize_predict_metrics_accumulator(
        self,
        *,
        report_profile: str,
    ) -> _PredictMetricsAccumulator:
        del report_profile
        return _PredictMetricsAccumulator()

    def update_predict_metrics_accumulator(
        self,
        *,
        accumulator: _PredictMetricsAccumulator,
        predict_results: list[Any],
        report_profile: str,
    ) -> None:
        del report_profile
        for result in [dict(item) for item in predict_results]:
            accumulator.count += 1
            _accumulate_metric_sums(
                accumulator.primary_sums,
                _topk_metrics_from_result(
                    result=result,
                    top_ks=tuple(int(k) for k in self.eval_cfg["answer_top_ks"]),
                ),
            )
            _accumulate_metric_sums(
                accumulator.secondary_sums,
                _secondary_metrics_from_result(
                    result=result,
                    edge_top_ks=tuple(int(k) for k in self.eval_cfg["edge_top_ks"]),
                ),
            )

    def finalize_predict_metrics_accumulator(
        self,
        *,
        accumulator: _PredictMetricsAccumulator,
        report_profile: str,
    ) -> dict[str, float]:
        del report_profile
        primary_metrics = _average_metric_sums(
            accumulator.primary_sums,
            count=accumulator.count,
        )
        secondary_metrics = _average_metric_sums(
            accumulator.secondary_sums,
            count=accumulator.count,
        )
        return {**primary_metrics, **secondary_metrics}

    def write_prediction_artifacts(
        self,
        *,
        results: list[Any],
        labels: list[Any],
        output_dir: str | Path,
        split: str,
        artifact_name: str,
        schema_version: int,
        entity_vocab_path: str | Path | None,
        relation_vocab_path: str | Path | None,
        questions_path: str | Path | None,
        overwrite: bool,
    ) -> dict[str, Path] | None:
        del artifact_name, schema_version, questions_path
        if not results:
            return None
        output_root = Path(str(output_dir))
        output_root.mkdir(parents=True, exist_ok=True)
        results_path = output_root / f"{split}.jsonl"
        labels_path = output_root / f"{split}.labels.jsonl"
        if overwrite:
            for path in (results_path, labels_path):
                if path.exists():
                    path.unlink()

        entity_labels = _load_vocab_label_map(
            path=entity_vocab_path,
            id_field=EntityVocabFields.ENTITY_ID,
            label_field=EntityVocabFields.LABEL,
        )
        relation_labels = _load_vocab_label_map(
            path=relation_vocab_path,
            id_field=RelationVocabFields.RELATION_ID,
            label_field=RelationVocabFields.LABEL,
        )

        serialized_results: list[dict[str, Any]] = []
        serialized_labels: list[dict[str, Any]] = []
        label_by_sample_id = {
            str(label["sample_id"]): dict(label)
            for label in [dict(item) for item in labels]
        }
        for result in [dict(item) for item in results]:
            sample_id = str(result["sample_id"])
            trajectories = _decorate_trajectory_records(
                trajectories=[
                    dict(item) for item in result.get("terminal_subgraphs", [])
                ],
                entity_labels=entity_labels,
                relation_labels=relation_labels,
            )
            serialized_results.append(
                {
                    "sample_id": sample_id,
                    "question": str(result["question"]),
                    "predicted_answer_entity_ids": list(
                        result.get("predicted_answer_entity_ids", [])
                    ),
                    "answer_log_masses": list(result.get("answer_log_masses", [])),
                    "requested_rollout_count": int(
                        result.get("requested_rollout_count", result["rollout_count"])
                    ),
                    "rollout_count": int(result["rollout_count"]),
                    "answering_rollout_count": int(result["answering_rollout_count"]),
                    "hit_rollout_count": int(result["hit_rollout_count"]),
                    "stopped_early": bool(result.get("stopped_early", False)),
                    "support_probability_mass": float(
                        result.get(
                            "support_probability_mass",
                            sum(
                                float(item.get("probability", 0.0))
                                for item in trajectories
                            ),
                        )
                    ),
                    "trajectories": trajectories,
                }
            )
            label_record = label_by_sample_id.get(
                sample_id,
                {
                    "sample_id": sample_id,
                    "question": str(result["question"]),
                    "gold_answer_entity_ids": list(result["gold_answer_entity_ids"]),
                    "a_entity_in_graph": bool(result.get("a_entity_in_graph", False)),
                },
            )
            gold_answer_entity_ids = [
                int(entity_id)
                for entity_id in label_record.get(
                    "gold_answer_entity_ids", result["gold_answer_entity_ids"]
                )
            ]
            serialized_labels.append(
                {
                    "sample_id": sample_id,
                    "question": str(label_record.get("question", result["question"])),
                    "answer_entity_ids": gold_answer_entity_ids,
                    "answer_texts": [
                        str(entity_labels.get(int(entity_id), str(entity_id)))
                        for entity_id in gold_answer_entity_ids
                    ],
                    "a_entity_in_graph": bool(
                        label_record.get(
                            "a_entity_in_graph", result.get("a_entity_in_graph", False)
                        )
                    ),
                }
            )

        append_jsonl_records(results_path, records=serialized_results)
        append_jsonl_records(labels_path, records=serialized_labels)
        return {
            "prompt_path": results_path,
            "results_path": results_path,
            "labels_path": labels_path,
        }


__all__ = ["SubgraphAnswerSearchRuntime"]
