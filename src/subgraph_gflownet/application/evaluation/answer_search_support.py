from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

from src.data.schema.constants import EntityVocabFields, RelationVocabFields
from src.graph import TrajectoryBatch

from .answer_search_metrics import (
    _graph_candidate_answer_upper_bound,
    _split_terminal_answer_log_mass,
)
from .answer_search_types import _GraphPredictionAccumulator, _TerminalSampleAggregate


def _edge_overlap_ratio(
    edge_ids: tuple[int, ...], other_edge_ids: tuple[int, ...]
) -> float:
    edge_set = {int(edge_id) for edge_id in edge_ids}
    other_set = {int(edge_id) for edge_id in other_edge_ids}
    if not edge_set or not other_set:
        return 0.0
    return float(len(edge_set.intersection(other_set))) / float(
        len(edge_set.union(other_set))
    )


def _select_terminal_support(
    *,
    ranked_terminals: list[_TerminalSampleAggregate],
    total_support_weight: float,
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
                probability = float(payload.score_sum) / float(
                    max(total_support_weight, 1.0e-12)
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
        accumulated_mass += float(chosen.score_sum) / float(
            max(total_support_weight, 1.0e-12)
        )
    return selected


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
    total_support_weight: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path_rank, payload in enumerate(support_payloads, start=1):
        probability = float(payload.score_sum) / float(
            max(total_support_weight, 1.0e-12)
        )
        sample_fraction = float(payload.sample_count) / float(max(executed_rollouts, 1))
        edge_records = _edge_records_from_terminal(
            batch=batch,
            graph_idx=graph_idx,
            edge_ids=payload.edge_ids,
        )
        records.append(
            {
                "path_rank": int(path_rank),
                "edge_ids": [int(edge_id) for edge_id in payload.edge_ids],
                "selected_node_ids": [
                    int(node_id) for node_id in payload.selected_node_ids
                ],
                "terminal_answer_set_entity_ids": [
                    int(entity_id)
                    for entity_id in payload.terminal_answer_set_entity_ids
                ],
                "singleton_terminal_answer_set_entity_id": (
                    int(payload.terminal_answer_set_entity_ids[0])
                    if len(payload.terminal_answer_set_entity_ids) == 1
                    else None
                ),
                "sample_count": int(payload.sample_count),
                "sample_fraction": float(sample_fraction),
                "score": float(payload.score_sum),
                "probability": float(probability),
                "prob": float(probability),
                "per_answer_log_posterior_surrogate_mass": _split_terminal_answer_log_mass(
                    probability_mass=probability,
                    terminal_answer_set_entity_ids=payload.terminal_answer_set_entity_ids,
                ),
                "edges": edge_records,
                "trajectory_text": _trajectory_text_from_edge_records(edge_records),
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
        gold_answer_in_graph=bool(
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
    aggregation_backend: str,
    edge_emit_top_k: int,
    support_mass_threshold: float,
    support_path_overlap_penalty: float,
    requested_rollouts: int,
) -> dict[str, Any]:
    executed_rollouts = max(int(accumulator.rollout_count), 1)
    total_support_weight = max(
        float(
            sum(
                payload.score_sum for payload in accumulator.terminal_witnesses.values()
            )
        ),
        1.0e-12,
    )
    ranked_answers = sorted(
        accumulator.answer_vote_counts.items(),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    ranked_terminals = sorted(
        accumulator.terminal_witnesses.values(),
        key=lambda item: (
            -float(item.score_sum),
            -int(item.sample_count),
            item.edge_ids,
        ),
    )
    top_witness = ranked_terminals[0] if ranked_terminals else None
    selected_support = _select_terminal_support(
        ranked_terminals=ranked_terminals,
        total_support_weight=total_support_weight,
        edge_emit_top_k=int(edge_emit_top_k),
        support_mass_threshold=float(support_mass_threshold),
        support_path_overlap_penalty=float(support_path_overlap_penalty),
    )
    support_records = _build_support_records(
        batch=batch,
        graph_idx=int(accumulator.original_graph_idx),
        support_payloads=selected_support,
        executed_rollouts=executed_rollouts,
        total_support_weight=total_support_weight,
    )
    result: dict[str, Any] = {
        "sample_id": str(accumulator.sample_id),
        "question": str(accumulator.question),
        "gold_answer_entity_ids": list(accumulator.gold_answer_entity_ids),
        "posterior_surrogate_aggregation_backend": str(aggregation_backend),
        "predicted_answer_entity_ids": [
            int(entity_id) for entity_id, _ in ranked_answers
        ],
        "answer_log_posterior_surrogate_masses": [
            float(math.log(float(votes) / float(total_support_weight)))
            for _, votes in ranked_answers
        ],
        "requested_rollout_count": int(requested_rollouts),
        "rollout_count": int(accumulator.rollout_count),
        "nonempty_terminal_answer_set_rollout_count": int(
            accumulator.nonempty_terminal_answer_set_rollout_count
        ),
        "gold_answer_in_state_rollout_count": int(
            accumulator.gold_answer_in_state_rollout_count
        ),
        "terminal_witness_count": int(len(ranked_terminals)),
        "mean_stop_step": float(accumulator.total_stop_steps)
        / float(executed_rollouts),
        "mean_terminal_component_count": float(
            accumulator.total_terminal_component_count
        )
        / float(executed_rollouts),
        "stopped_early": bool(accumulator.stopped_early),
        "early_stop_margin": accumulator.early_stop_margin,
        "gold_answer_in_graph": bool(accumulator.gold_answer_in_graph),
        "answer_score_total": float(sum(float(score) for _, score in ranked_answers)),
        "support_score_total": float(total_support_weight),
        "witness_support_probabilities": [
            float(record["probability"]) for record in support_records
        ],
    }
    if top_witness is not None:
        top_probability = float(top_witness.score_sum) / float(total_support_weight)
        result["top_witness_edge_ids"] = [
            int(edge_id) for edge_id in top_witness.edge_ids
        ]
        result["top_witness_node_ids"] = [
            int(node_id) for node_id in top_witness.selected_node_ids
        ]
        result["top_witness_probability"] = float(top_probability)
        result["top_witness_sample_count"] = int(top_witness.sample_count)
        result["top_witness_score"] = float(top_witness.score_sum)
        result["top_witness_terminal_answer_set_entity_ids"] = [
            int(entity_id) for entity_id in top_witness.terminal_answer_set_entity_ids
        ]
        result["top_witness_singleton_terminal_answer_set_entity_id"] = (
            int(top_witness.terminal_answer_set_entity_ids[0])
            if len(top_witness.terminal_answer_set_entity_ids) == 1
            else None
        )
    if include_answer_support:
        result["witness_supports"] = support_records
        result["witness_support_probability_mass"] = float(
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
        terminal_answer_set_entity_id = trajectory.get(
            "singleton_terminal_answer_set_entity_id"
        )
        decorated_trajectory = dict(trajectory)
        decorated_trajectory["edges"] = edges
        if terminal_answer_set_entity_id is not None:
            decorated_trajectory["singleton_terminal_answer_set_entity_text"] = str(
                entity_labels.get(
                    int(terminal_answer_set_entity_id),
                    str(terminal_answer_set_entity_id),
                )
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


__all__ = [
    "_build_graph_prediction_accumulator",
    "_build_support_records",
    "_decorate_trajectory_records",
    "_edge_overlap_ratio",
    "_edge_records_from_terminal",
    "_finalize_graph_result",
    "_load_vocab_label_map",
    "_select_terminal_support",
    "_trajectory_text_from_edge_records",
]
