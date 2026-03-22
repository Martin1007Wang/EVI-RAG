from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from src.metrics.answer_reachability.edge_eval import (
    EdgePredictionRecord,
    EdgeRetrievalLabelRecord,
    EdgeRetrievalResult,
)
from src.metrics.answer_reachability.schema import (
    AnswerPosteriorRecord,
    AnswerSupportRecord,
    EdgeRecord,
    SupportWindowLabelRecord,
    SupportWindowResult,
    TrajectoryRecord,
)
from src.utils.metrics_io import to_serializable

PredictionKind = Literal["support_window", "edge_retrieval"]


def append_jsonl_records(path: str | Path, *, records: Iterable[Any]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = record
            if is_dataclass(record):
                payload = asdict(record)
            handle.write(json.dumps(to_serializable(payload), ensure_ascii=True) + "\n")


def iter_jsonl_records(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path)
    if not resolved.exists():
        return
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield dict(json.loads(text))


def jsonl_has_records(path: str | Path | None) -> bool:
    if path is None:
        return False
    resolved = Path(path)
    if not resolved.exists():
        return False
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


def infer_prediction_kind(
    *,
    results: list[Any] | None = None,
    labels: list[Any] | None = None,
) -> PredictionKind | None:
    first_result = None if not results else results[0]
    first_label = None if not labels else labels[0]
    for item in (first_result, first_label):
        if isinstance(item, SupportWindowResult) or isinstance(
            item, SupportWindowLabelRecord
        ):
            return "support_window"
        if isinstance(item, EdgeRetrievalResult) or isinstance(
            item, EdgeRetrievalLabelRecord
        ):
            return "edge_retrieval"
    return None


def load_edge_record(record: dict[str, Any]) -> EdgeRecord:
    return EdgeRecord(
        edge_id=int(record.get("edge_id", 0)),
        src_entity_id=int(record.get("src_entity_id", 0)),
        relation_id=int(record.get("relation_id", 0)),
        dst_entity_id=int(record.get("dst_entity_id", 0)),
    )


def load_trajectory_record(record: dict[str, Any]) -> TrajectoryRecord:
    return TrajectoryRecord(
        sample_id=str(record.get("sample_id", "")),
        path_rank=int(record.get("path_rank", 0)),
        log_prob=float(record.get("log_prob", 0.0)),
        prob=float(record.get("prob", 0.0)),
        cumulative_mass=float(record.get("cumulative_mass", 0.0)),
        terminal_entity_id=int(record.get("terminal_entity_id", 0)),
        is_gold=bool(record.get("is_gold", False)),
        edges=[load_edge_record(edge) for edge in record.get("edges") or []],
        start_entity_id=(
            None
            if record.get("start_entity_id") is None
            else int(record.get("start_entity_id"))
        ),
        answer_rank=int(record.get("answer_rank", 0)),
        support_rank=int(record.get("support_rank", 0)),
        conditional_prob=float(record.get("conditional_prob", 0.0)),
        conditional_cumulative_mass=float(
            record.get("conditional_cumulative_mass", 0.0)
        ),
    )


def load_answer_posterior_record(record: dict[str, Any]) -> AnswerPosteriorRecord:
    return AnswerPosteriorRecord(
        answer_entity_id=int(record.get("answer_entity_id", 0)),
        prob=float(record.get("prob", 0.0)),
        cumulative_mass=float(record.get("cumulative_mass", 0.0)),
        is_gold=bool(record.get("is_gold", False)),
        is_selected=bool(record.get("is_selected", False)),
        support_mass=float(record.get("support_mass", 0.0)),
        support_conditioned_mass=float(record.get("support_conditioned_mass", 0.0)),
        support_path_count=int(record.get("support_path_count", 0)),
        prob_ci_low=float(record.get("prob_ci_low", 0.0)),
        prob_ci_high=float(record.get("prob_ci_high", 0.0)),
    )


def load_answer_support_record(record: dict[str, Any]) -> AnswerSupportRecord:
    return AnswerSupportRecord(
        answer_entity_id=int(record.get("answer_entity_id", 0)),
        answer_rank=int(record.get("answer_rank", 0)),
        prob=float(record.get("prob", 0.0)),
        cumulative_mass=float(record.get("cumulative_mass", 0.0)),
        is_gold=bool(record.get("is_gold", False)),
        is_selected=bool(record.get("is_selected", False)),
        support_mass=float(record.get("support_mass", 0.0)),
        support_conditioned_mass=float(record.get("support_conditioned_mass", 0.0)),
        support_path_count=int(record.get("support_path_count", 0)),
        trajectories=[
            load_trajectory_record(trajectory)
            for trajectory in record.get("trajectories") or []
        ],
        prob_ci_low=float(record.get("prob_ci_low", 0.0)),
        prob_ci_high=float(record.get("prob_ci_high", 0.0)),
    )


def load_support_window_result(record: dict[str, Any]) -> SupportWindowResult:
    return SupportWindowResult(
        sample_id=str(record.get("sample_id", "")),
        dataset_scope=str(record.get("dataset_scope", "")),
        mass_threshold=float(record.get("mass_threshold", 0.0)),
        window_size=int(record.get("window_size", 0)),
        covered_mass=float(record.get("covered_mass", 0.0)),
        residual_mass=float(record.get("residual_mass", 0.0)),
        gold_total_mass=float(record.get("gold_total_mass", 0.0)),
        covered_gold_mass=float(record.get("covered_gold_mass", 0.0)),
        missed_gold_mass=float(record.get("missed_gold_mass", 0.0)),
        unique_answer_count=int(record.get("unique_answer_count", 0)),
        unique_path_count=int(record.get("unique_path_count", 0)),
        gold_answer_entity_ids=[
            int(value) for value in record.get("gold_answer_entity_ids") or []
        ],
        start_entity_ids=[int(value) for value in record.get("start_entity_ids") or []],
        trajectories=[
            load_trajectory_record(trajectory)
            for trajectory in record.get("trajectories") or []
        ],
        inference_mode=str(record.get("inference_mode", "unspecified")),
        ci_confidence_level=(
            None
            if record.get("ci_confidence_level") is None
            else float(record.get("ci_confidence_level"))
        ),
        covered_mass_ci_low=(
            None
            if record.get("covered_mass_ci_low") is None
            else float(record.get("covered_mass_ci_low"))
        ),
        covered_mass_ci_high=(
            None
            if record.get("covered_mass_ci_high") is None
            else float(record.get("covered_mass_ci_high"))
        ),
        gold_total_mass_ci_low=(
            None
            if record.get("gold_total_mass_ci_low") is None
            else float(record.get("gold_total_mass_ci_low"))
        ),
        gold_total_mass_ci_high=(
            None
            if record.get("gold_total_mass_ci_high") is None
            else float(record.get("gold_total_mass_ci_high"))
        ),
        answer_mass_threshold=float(record.get("answer_mass_threshold", 1.0)),
        support_mass_threshold=float(record.get("support_mass_threshold", 1.0)),
        probe_count=int(record.get("probe_count", 0)),
        emit_path_count=int(record.get("emit_path_count", 0)),
        remaining_mass_upper=float(record.get("remaining_mass_upper", 0.0)),
        stop_reason=str(record.get("stop_reason", "")),
        coverage_certified=bool(record.get("coverage_certified", False)),
        answer_mass_reference=str(record.get("answer_mass_reference", "unspecified")),
        support_mass_reference=str(record.get("support_mass_reference", "unspecified")),
        selected_answer_ids=[
            int(value) for value in record.get("selected_answer_ids") or []
        ],
        answer_posterior=[
            load_answer_posterior_record(answer)
            for answer in record.get("answer_posterior") or []
        ],
        answer_support=[
            load_answer_support_record(answer_support)
            for answer_support in record.get("answer_support") or []
        ],
    )


def load_support_window_label(record: dict[str, Any]) -> SupportWindowLabelRecord:
    return SupportWindowLabelRecord(
        sample_id=str(record.get("sample_id", "")),
        question=str(record.get("question", "")),
        start_entity_ids=[int(value) for value in record.get("start_entity_ids") or []],
        answer_entity_ids=[
            int(value) for value in record.get("answer_entity_ids") or []
        ],
        a_entity_in_graph=bool(record.get("a_entity_in_graph", False)),
    )


def load_edge_prediction_record(record: dict[str, Any]) -> EdgePredictionRecord:
    return EdgePredictionRecord(
        edge_id=int(record.get("edge_id", 0)),
        src_entity_id=int(record.get("src_entity_id", 0)),
        relation_id=int(record.get("relation_id", 0)),
        dst_entity_id=int(record.get("dst_entity_id", 0)),
        score=float(record.get("score", 0.0)),
        conditional_score=float(record.get("conditional_score", 0.0)),
        is_positive=bool(record.get("is_positive", False)),
    )


def load_edge_retrieval_result(record: dict[str, Any]) -> EdgeRetrievalResult:
    return EdgeRetrievalResult(
        sample_id=str(record.get("sample_id", "")),
        dataset_scope=str(record.get("dataset_scope", "")),
        num_edges=int(record.get("num_edges", 0)),
        num_positive_edges=int(record.get("num_positive_edges", 0)),
        max_path_length=(
            None
            if record.get("max_path_length") is None
            else int(record.get("max_path_length"))
        ),
        gold_total_mass=float(record.get("gold_total_mass", 0.0)),
        first_positive_rank=(
            None
            if record.get("first_positive_rank") is None
            else int(record.get("first_positive_rank"))
        ),
        positive_edge_ids=[
            int(value) for value in record.get("positive_edge_ids") or []
        ],
        ranked_edge_ids=[int(value) for value in record.get("ranked_edge_ids") or []],
        ranked_edges=[
            load_edge_prediction_record(edge)
            for edge in record.get("ranked_edges") or []
        ],
    )


def load_edge_retrieval_label(record: dict[str, Any]) -> EdgeRetrievalLabelRecord:
    return EdgeRetrievalLabelRecord(
        sample_id=str(record.get("sample_id", "")),
        question=str(record.get("question", "")),
        num_edges=int(record.get("num_edges", 0)),
        positive_edge_ids=[
            int(value) for value in record.get("positive_edge_ids") or []
        ],
        max_path_length=(
            None
            if record.get("max_path_length") is None
            else int(record.get("max_path_length"))
        ),
    )


def load_prediction_result(record: dict[str, Any], *, kind: PredictionKind) -> Any:
    if kind == "edge_retrieval":
        return load_edge_retrieval_result(record)
    return load_support_window_result(record)


def load_prediction_label(record: dict[str, Any], *, kind: PredictionKind) -> Any:
    if kind == "edge_retrieval":
        return load_edge_retrieval_label(record)
    return load_support_window_label(record)


__all__ = [
    "PredictionKind",
    "append_jsonl_records",
    "infer_prediction_kind",
    "iter_jsonl_records",
    "jsonl_has_records",
    "load_edge_retrieval_label",
    "load_edge_retrieval_result",
    "load_prediction_label",
    "load_prediction_result",
    "load_support_window_label",
    "load_support_window_result",
]
