from __future__ import annotations

import ast
from datetime import datetime
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, Sequence

from .prediction_io import (
    iter_jsonl_records,
    load_support_window_label,
    load_support_window_result,
)
from .schema import (
    AnswerPosteriorRecord,
    AnswerSupportRecord,
    EdgeRecord,
    SupportWindowLabelRecord,
    SupportWindowResult,
    TrajectoryRecord,
)


def _load_vocab_map(
    path: str | Path | None,
    *,
    key_column: str,
    value_column: str,
) -> dict[int, str] | None:
    if path in (None, ""):
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    import pandas as pd

    frame = pd.read_parquet(resolved)
    actual_key = key_column
    if actual_key not in frame.columns and key_column == "entity_id":
        if "embedding_id" in frame.columns:
            actual_key = "embedding_id"
    if actual_key not in frame.columns or value_column not in frame.columns:
        return None
    return dict(zip(frame[actual_key].astype(int), frame[value_column].astype(str)))


def _coerce_answer_texts(raw_answers: Any) -> list[str]:
    if raw_answers is None:
        return []
    if isinstance(raw_answers, str):
        text = raw_answers.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed if str(item).strip()]
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed if str(item).strip()]
        return [text]
    if isinstance(raw_answers, (list, tuple, set)):
        return [str(item) for item in raw_answers if str(item).strip()]
    return [str(raw_answers)]


def _load_question_map(path: str | Path | None) -> dict[str, dict[str, Any]] | None:
    if path in (None, ""):
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    import pandas as pd

    frame = pd.read_parquet(resolved)
    id_column = "graph_id"
    if id_column not in frame.columns:
        raise ValueError("questions.parquet missing graph_id column.")
    out: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        sample_id = str(row.get(id_column) or "")
        if not sample_id:
            continue
        out[sample_id] = {
            "question": str(row.get("question") or ""),
            "answer_texts": _coerce_answer_texts(row.get("answer_texts")),
        }
    return out


def _edge_to_dict(
    edge: EdgeRecord,
    *,
    entity_map: dict[int, str] | None,
    relation_map: dict[int, str] | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    record["edge_id"] = int(edge.edge_id)
    record["src_entity_id"] = int(edge.src_entity_id)
    record["relation_id"] = int(edge.relation_id)
    record["dst_entity_id"] = int(edge.dst_entity_id)
    if entity_map is not None:
        record["src_text"] = entity_map.get(edge.src_entity_id, str(edge.src_entity_id))
        record["dst_text"] = entity_map.get(edge.dst_entity_id, str(edge.dst_entity_id))
    if relation_map is not None:
        record["relation_text"] = relation_map.get(
            edge.relation_id, str(edge.relation_id)
        )
    return record


def _trajectory_text(
    trajectory: TrajectoryRecord,
    *,
    entity_map: dict[int, str] | None,
    relation_map: dict[int, str] | None,
) -> str:
    if not trajectory.edges:
        terminal = (
            entity_map.get(
                trajectory.terminal_entity_id, str(trajectory.terminal_entity_id)
            )
            if entity_map is not None
            else str(trajectory.terminal_entity_id)
        )
        return f"(start_only) {terminal}"
    parts: list[str] = []
    for edge in trajectory.edges:
        src = (
            entity_map.get(edge.src_entity_id, str(edge.src_entity_id))
            if entity_map
            else str(edge.src_entity_id)
        )
        rel = (
            relation_map.get(edge.relation_id, str(edge.relation_id))
            if relation_map
            else str(edge.relation_id)
        )
        dst = (
            entity_map.get(edge.dst_entity_id, str(edge.dst_entity_id))
            if entity_map
            else str(edge.dst_entity_id)
        )
        parts.append(f"{src} --{rel}--> {dst}")
    return " ; ".join(parts)


def _trajectory_to_prompt_record(
    trajectory: TrajectoryRecord,
    *,
    entity_map: dict[int, str] | None,
    relation_map: dict[int, str] | None,
) -> dict[str, Any]:
    record = {
        "path_rank": int(trajectory.path_rank),
        "log_prob": float(trajectory.log_prob),
        "prob": float(trajectory.prob),
        "cumulative_mass": float(trajectory.cumulative_mass),
        "terminal_entity_id": int(trajectory.terminal_entity_id),
        "start_entity_id": trajectory.start_entity_id,
        "answer_rank": int(trajectory.answer_rank),
        "support_rank": int(trajectory.support_rank),
        "conditional_prob": float(trajectory.conditional_prob),
        "conditional_cumulative_mass": float(trajectory.conditional_cumulative_mass),
        "edges": [
            _edge_to_dict(edge, entity_map=entity_map, relation_map=relation_map)
            for edge in trajectory.edges
        ],
    }
    if entity_map is not None:
        record["terminal_entity_text"] = entity_map.get(
            trajectory.terminal_entity_id,
            str(trajectory.terminal_entity_id),
        )
    record["trajectory_text"] = _trajectory_text(
        trajectory,
        entity_map=entity_map,
        relation_map=relation_map,
    )
    return record


def _answer_to_dict(answer: AnswerPosteriorRecord) -> dict[str, Any]:
    return {
        "answer_entity_id": int(answer.answer_entity_id),
        "prob": float(answer.prob),
        "prob_ci_low": float(answer.prob_ci_low),
        "prob_ci_high": float(answer.prob_ci_high),
        "cumulative_mass": float(answer.cumulative_mass),
        "is_gold": bool(answer.is_gold),
        "is_selected": bool(answer.is_selected),
        "support_mass": float(answer.support_mass),
        "support_conditioned_mass": float(answer.support_conditioned_mass),
        "support_path_count": int(answer.support_path_count),
    }


def _answer_support_to_dict(
    answer_support: AnswerSupportRecord,
    *,
    entity_map: dict[int, str] | None,
    relation_map: dict[int, str] | None,
) -> dict[str, Any]:
    record = {
        "answer_entity_id": int(answer_support.answer_entity_id),
        "answer_rank": int(answer_support.answer_rank),
        "prob": float(answer_support.prob),
        "prob_ci_low": float(answer_support.prob_ci_low),
        "prob_ci_high": float(answer_support.prob_ci_high),
        "cumulative_mass": float(answer_support.cumulative_mass),
        "is_gold": bool(answer_support.is_gold),
        "is_selected": bool(answer_support.is_selected),
        "support_mass": float(answer_support.support_mass),
        "support_conditioned_mass": float(answer_support.support_conditioned_mass),
        "support_path_count": int(answer_support.support_path_count),
        "trajectories": [
            _trajectory_to_prompt_record(
                trajectory,
                entity_map=entity_map,
                relation_map=relation_map,
            )
            for trajectory in answer_support.trajectories
        ],
    }
    if entity_map is not None:
        record["answer_entity_text"] = entity_map.get(
            int(answer_support.answer_entity_id),
            str(answer_support.answer_entity_id),
        )
    return record


class SupportWindowArtifactWriter:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str,
        artifact_name: str = "rankflow",
        schema_version: int = 1,
        entity_vocab_path: str | Path | None = None,
        relation_vocab_path: str | Path | None = None,
        questions_path: str | Path | None = None,
        overwrite: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.artifact_name = str(artifact_name)
        self.schema_version = int(schema_version)
        self.overwrite = bool(overwrite)
        self.entity_map = _load_vocab_map(
            entity_vocab_path,
            key_column="entity_id",
            value_column="label",
        )
        self.relation_map = _load_vocab_map(
            relation_vocab_path,
            key_column="relation_id",
            value_column="label",
        )
        self.question_map = _load_question_map(questions_path)

    def write(
        self,
        *,
        results: Sequence[SupportWindowResult],
        labels: Sequence[SupportWindowLabelRecord],
    ) -> dict[str, Path]:
        prompt_path, labels_path, manifest_path = self._prepare_output_paths()
        label_map = {record.sample_id: record for record in labels}
        with (
            prompt_path.open("w", encoding="utf-8") as prompt_handle,
            labels_path.open("w", encoding="utf-8") as label_handle,
        ):
            for result in results:
                label = label_map.get(result.sample_id)
                self._write_jsonl_record(
                    prompt_handle,
                    self._build_prompt_record(result, label),
                )
                self._write_jsonl_record(
                    label_handle,
                    self._build_label_record(result, label),
                )
        self._write_manifest(
            manifest_path=manifest_path,
            prompt_path=prompt_path,
            labels_path=labels_path,
        )
        return {
            "prompt_path": prompt_path,
            "labels_path": labels_path,
            "manifest_path": manifest_path,
        }

    def write_from_jsonl(
        self,
        *,
        results_path: str | Path,
        labels_path: str | Path,
    ) -> dict[str, Path]:
        prompt_path, output_labels_path, manifest_path = self._prepare_output_paths()
        with (
            prompt_path.open("w", encoding="utf-8") as prompt_handle,
            output_labels_path.open("w", encoding="utf-8") as label_handle,
        ):
            for result_record, label_record in zip_longest(
                iter_jsonl_records(results_path),
                iter_jsonl_records(labels_path),
            ):
                if result_record is None or label_record is None:
                    raise ValueError(
                        "Prediction result/label jsonl files must contain the same number of records."
                    )
                result = load_support_window_result(result_record)
                label = load_support_window_label(label_record)
                if result.sample_id != label.sample_id:
                    raise ValueError(
                        "Prediction result/label jsonl files are not aligned by sample_id. "
                        f"result={result.sample_id!r} label={label.sample_id!r}."
                    )
                self._write_jsonl_record(
                    prompt_handle,
                    self._build_prompt_record(result, label),
                )
                self._write_jsonl_record(
                    label_handle,
                    self._build_label_record(result, label),
                )
        self._write_manifest(
            manifest_path=manifest_path,
            prompt_path=prompt_path,
            labels_path=output_labels_path,
        )
        return {
            "prompt_path": prompt_path,
            "labels_path": output_labels_path,
            "manifest_path": manifest_path,
        }

    def _prepare_output_paths(self) -> tuple[Path, Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.output_dir / f"{self.split}.jsonl"
        labels_path = self.output_dir / f"{self.split}.labels.jsonl"
        manifest_path = self.output_dir / f"{self.split}.manifest.json"
        if not self.overwrite:
            for path in (prompt_path, labels_path, manifest_path):
                if path.exists():
                    raise FileExistsError(f"Artifact already exists: {path}")
        return prompt_path, labels_path, manifest_path

    def _write_manifest(
        self,
        *,
        manifest_path: Path,
        prompt_path: Path,
        labels_path: Path,
    ) -> None:
        manifest = {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": prompt_path.name,
            "labels_file": labels_path.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "rankflow_artifact_writer",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _build_prompt_record(
        self,
        result: SupportWindowResult,
        label: SupportWindowLabelRecord | None,
    ) -> dict[str, Any]:
        fallback_question = "" if label is None else label.question
        question = self._question(result.sample_id, fallback=fallback_question)
        return {
            "sample_id": result.sample_id,
            "question": question,
            "dataset_scope": result.dataset_scope,
            "mass_threshold": float(result.mass_threshold),
            "inference_mode": result.inference_mode,
            "answer_mass_threshold": float(result.answer_mass_threshold),
            "support_mass_threshold": float(result.support_mass_threshold),
            "window_size": int(result.window_size),
            "covered_mass": float(result.covered_mass),
            "covered_mass_ci_low": (
                None
                if result.covered_mass_ci_low is None
                else float(result.covered_mass_ci_low)
            ),
            "covered_mass_ci_high": (
                None
                if result.covered_mass_ci_high is None
                else float(result.covered_mass_ci_high)
            ),
            "residual_mass": float(result.residual_mass),
            "remaining_mass_upper": float(result.remaining_mass_upper),
            "gold_total_mass": float(result.gold_total_mass),
            "gold_total_mass_ci_low": (
                None
                if result.gold_total_mass_ci_low is None
                else float(result.gold_total_mass_ci_low)
            ),
            "gold_total_mass_ci_high": (
                None
                if result.gold_total_mass_ci_high is None
                else float(result.gold_total_mass_ci_high)
            ),
            "ci_confidence_level": (
                None
                if result.ci_confidence_level is None
                else float(result.ci_confidence_level)
            ),
            "probe_count": int(result.probe_count),
            "emit_path_count": int(result.emit_path_count),
            "stop_reason": result.stop_reason,
            "coverage_certified": bool(result.coverage_certified),
            "answer_mass_reference": result.answer_mass_reference,
            "support_mass_reference": result.support_mass_reference,
            "unique_answer_count": int(result.unique_answer_count),
            "unique_path_count": int(result.unique_path_count),
            "start_entity_ids": [int(value) for value in result.start_entity_ids],
            "selected_answer_ids": [int(value) for value in result.selected_answer_ids],
            "answer_posterior": [
                _answer_to_dict(answer) for answer in result.answer_posterior
            ],
            "answer_support": [
                _answer_support_to_dict(
                    answer_support,
                    entity_map=self.entity_map,
                    relation_map=self.relation_map,
                )
                for answer_support in result.answer_support
            ],
            "trajectories": [
                _trajectory_to_prompt_record(
                    trajectory,
                    entity_map=self.entity_map,
                    relation_map=self.relation_map,
                )
                for trajectory in result.trajectories
            ],
        }

    def _build_label_record(
        self,
        result: SupportWindowResult,
        label: SupportWindowLabelRecord | None,
    ) -> dict[str, Any]:
        fallback_question = "" if label is None else label.question
        question = self._question(result.sample_id, fallback=fallback_question)
        answer_texts = self._answer_texts(result.sample_id)
        start_entity_ids = (
            result.start_entity_ids if label is None else label.start_entity_ids
        )
        answer_entity_ids = (
            result.gold_answer_entity_ids if label is None else label.answer_entity_ids
        )
        record = {
            "sample_id": result.sample_id,
            "question": question,
            "start_entity_ids": [int(value) for value in start_entity_ids],
            "answer_entity_ids": [int(value) for value in answer_entity_ids],
            "answer_texts": answer_texts,
        }
        if label is not None:
            record["a_entity_in_graph"] = bool(label.a_entity_in_graph)
        return record

    def _question(self, sample_id: str, *, fallback: str) -> str:
        if self.question_map is None:
            return str(fallback or "")
        meta = self.question_map.get(sample_id)
        if meta is None:
            return str(fallback or "")
        return str(meta.get("question") or fallback or "")

    def _answer_texts(self, sample_id: str) -> list[str]:
        if self.question_map is None:
            return []
        meta = self.question_map.get(sample_id)
        if meta is None:
            return []
        return [str(value) for value in meta.get("answer_texts") or []]

    def _write_jsonl_record(self, handle: Any, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["SupportWindowArtifactWriter"]
