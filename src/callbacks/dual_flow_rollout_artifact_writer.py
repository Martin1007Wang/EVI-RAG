from __future__ import annotations

import ast
import json
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter

_ZERO = 0
_ONE = 1
STOP_RELATION = -1
_DEFAULT_SPLIT = "predict"
_DEFAULT_ARTIFACT_NAME = "eval_dual_flow"
_DEFAULT_SCHEMA_VERSION = _ONE
_WRITE_INTERVAL = "batch"
_SUPER_SOURCE_ENTITY_ID = -1
_LABELS_SUFFIX = ".labels.jsonl"
_FIELD_SAMPLE_ID = "sample_id"


class DualFlowRolloutArtifactWriter(BasePredictionWriter):
    """Persist DualFlow rollout artifacts during predict()."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str = _DEFAULT_SPLIT,
        enabled: bool = True,
        artifact_name: str = _DEFAULT_ARTIFACT_NAME,
        schema_version: int = _DEFAULT_SCHEMA_VERSION,
        textualize: bool = False,
        entity_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        questions_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(write_interval=_WRITE_INTERVAL)
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= _ZERO:
            raise ValueError("schema_version must be a positive integer.")
        self.textualize = bool(textualize)
        self.entity_vocab_path = entity_vocab_path
        self.relation_vocab_path = relation_vocab_path
        self.questions_path = questions_path
        self.overwrite = bool(overwrite)
        self._processor: Optional[_RolloutArtifactProcessor] = None
        self._output_path: Optional[Path] = None
        self._labels_path: Optional[Path] = None
        self._manifest_path: Optional[Path] = None

    def on_predict_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / f"{self.split}.jsonl"
        self._labels_path = self.output_dir / f"{self.split}{_LABELS_SUFFIX}"
        self._manifest_path = self.output_dir / f"{self.split}.manifest.json"
        if self.overwrite and trainer.global_rank == _ZERO:
            self._output_path.write_text("", encoding="utf-8")
            self._labels_path.write_text("", encoding="utf-8")
        if self.textualize:
            self._processor = _RolloutArtifactProcessor(
                {
                    "textualize": True,
                    "entity_vocab_path": self.entity_vocab_path,
                    "relation_vocab_path": self.relation_vocab_path,
                    "questions_path": self.questions_path,
                }
            )
        elif self.questions_path:
            self._processor = _RolloutArtifactProcessor(
                {
                    "textualize": False,
                    "questions_path": self.questions_path,
                }
            )

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Any,
        batch_indices: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if not self.enabled or prediction is None:
            return
        records = prediction
        if not isinstance(records, list) or not records:
            return
        records = self._gather_records(records)
        if not records:
            return
        records = self._apply_processor(records)
        if trainer.global_rank != _ZERO:
            return
        self._append_records(records)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or trainer.global_rank != _ZERO:
            return
        if self._manifest_path is None or self._output_path is None or self._labels_path is None:
            raise RuntimeError("DualFlowRolloutArtifactWriter missing manifest/output path; on_predict_start was not called.")
        manifest = self._build_manifest(self._output_path.name, self._labels_path.name)
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _apply_processor(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self._processor is None:
            return records
        return self._processor.process(records)

    def _append_records(self, records: list[Dict[str, Any]]) -> None:
        if self._output_path is None or self._labels_path is None:
            raise RuntimeError("DualFlowRolloutArtifactWriter missing output path; on_predict_start was not called.")
        prompt_records, label_records = self._split_records(records)
        with self._output_path.open("a", encoding="utf-8") as f_main:
            for rec in prompt_records:
                f_main.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self._labels_path.open("a", encoding="utf-8") as f_labels:
            for rec in label_records:
                f_labels.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_manifest(self, file_name: str, labels_file_name: str) -> Dict[str, Any]:
        return {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": file_name,
            "labels_file": labels_file_name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "dual_flow_rollout_artifact_writer",
        }

    @staticmethod
    def _coerce_int_list(raw: Any) -> list[int]:
        if raw is None:
            return []
        if isinstance(raw, (list, tuple, set)):
            out: list[int] = []
            for item in raw:
                try:
                    out.append(int(item))
                except Exception:
                    continue
            return out
        try:
            return [int(raw)]
        except Exception:
            return []

    @staticmethod
    def _coerce_bool(raw: Any) -> Optional[bool]:
        if isinstance(raw, bool):
            return raw
        return None

    def _split_records(self, records: list[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
        prompt_records: list[Dict[str, Any]] = []
        label_records: list[Dict[str, Any]] = []
        for sample in records:
            sample_id = str(sample.get(_FIELD_SAMPLE_ID) or "")
            if not sample_id:
                continue
            rollouts = sample.get("rollouts")
            if not isinstance(rollouts, list):
                rollouts = []
            question = str(sample.get("question") or sample.get("question_text") or "")
            prompt_records.append(
                {
                    _FIELD_SAMPLE_ID: sample_id,
                    "question": question,
                    "rollouts": rollouts,
                }
            )

            answer_texts = _RolloutArtifactProcessor._coerce_answer_texts(sample.get("answer_texts"))
            answer_text = str(sample.get("answer_text") or "")
            if not answer_text:
                answer_text = _RolloutArtifactProcessor._coerce_answer_text(answer_texts)
            label: Dict[str, Any] = {
                _FIELD_SAMPLE_ID: sample_id,
                "question": question,
                "question_text": question,
                "start_entity_ids": self._coerce_int_list(sample.get("start_entity_ids")),
                "answer_entity_ids": self._coerce_int_list(sample.get("answer_entity_ids")),
                "answer_texts": answer_texts,
                "answer_text": answer_text,
            }
            a_entity_in_graph = self._coerce_bool(sample.get("a_entity_in_graph"))
            if a_entity_in_graph is not None:
                label["a_entity_in_graph"] = a_entity_in_graph
            label_records.append(label)
        return prompt_records, label_records

    @staticmethod
    def _gather_records(records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: list[Optional[list]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if dist.get_rank() != _ZERO:
                return []
            merged: list[Dict[str, Any]] = []
            for part in gathered:
                if part:
                    merged.extend(part)
            return merged
        return records


@dataclass
class _RolloutArtifactProcessor:
    """Textualize edge metadata for rollout artifacts."""

    cfg: Dict[str, Any]

    def __post_init__(self) -> None:
        self.ent_map, self.rel_map = self._resolve_vocab_maps(self.cfg)
        self.question_map = self._resolve_question_map(self.cfg)

    def process(self, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if self.ent_map is not None or self.rel_map is not None:
            self._inject_text(records)
        if self.question_map is not None:
            self._inject_question_text(records)
        self._inject_trajectory_text(records)
        return records

    def _inject_text(self, records: list[Dict[str, Any]]) -> None:
        for sample in records:
            for rollout in sample.get("rollouts", []):
                for edge in rollout.get("edges", []):
                    h = edge.get("head_entity_id")
                    r = edge.get("relation_id")
                    t = edge.get("tail_entity_id")
                    s_id = edge.get("src_entity_id")
                    d_id = edge.get("dst_entity_id")
                    if self.ent_map is not None:
                        edge["head_text"] = self.ent_map.get(h, str(h) if h is not None else None)
                        edge["tail_text"] = self.ent_map.get(t, str(t) if t is not None else None)
                        edge["src_text"] = self.ent_map.get(s_id, str(s_id) if s_id is not None else None)
                        edge["dst_text"] = self.ent_map.get(d_id, str(d_id) if d_id is not None else None)
                    if self.rel_map is not None:
                        edge["relation_text"] = self.rel_map.get(r, str(r) if r is not None else None)

    def _resolve_vocab_maps(self, cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            raise ValueError("textualize=true requires both entity_vocab_path and relation_vocab_path.")
        if not Path(entity_path).exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {entity_path}")
        if not Path(relation_path).exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {relation_path}")
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "textualize=true requires pandas to load parquet vocabularies."
            ) from exc
        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        ent_df = pd.read_parquet(entity_path)
        if "entity_id" in ent_df.columns:
            ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
        elif "embedding_id" in ent_df.columns:
            ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
        rel_df = pd.read_parquet(relation_path)
        rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        return ent_map, rel_map

    @staticmethod
    def _coerce_answer_texts(raw_answers: Any) -> list[str]:
        if raw_answers is None:
            return []

        # pandas may store missing values as NaN / pd.NA.
        try:
            import pandas as pd

            is_missing = pd.isna(raw_answers)
            if isinstance(is_missing, bool) and is_missing:
                return []
        except Exception:
            pass

        if isinstance(raw_answers, str):
            text = raw_answers.strip()
            if not text:
                return []
            # Handle cases like a JSON list string or a python literal list string.
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x) for x in parsed if str(x).strip()]

                # Numpy array repr can look like: ['A' 'B' "C"] (no commas).
                tokens: list[str] = []
                for m in re.finditer(r"(?:'([^']*)'|\"([^\"]*)\")", text, flags=re.DOTALL):
                    token = m.group(1) if m.group(1) is not None else m.group(2)
                    token = str(token or "").replace("\\'", "'").replace('\\"', '"').strip()
                    if token:
                        tokens.append(token)
                if tokens:
                    return tokens

                for parser in (ast.literal_eval,):
                    try:
                        parsed = parser(text)
                    except Exception:
                        continue
                    if isinstance(parsed, (list, tuple, set)):
                        return [str(x) for x in parsed if str(x).strip()]
            return [text]

        if isinstance(raw_answers, (list, tuple, set)):
            return [str(x) for x in raw_answers if str(x).strip()]

        # Handle numpy arrays / pandas arrays without hard dependency on numpy.
        tolist = getattr(raw_answers, "tolist", None)
        if callable(tolist):
            try:
                parsed = tolist()
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return [str(x) for x in parsed if str(x).strip()]

        return [str(raw_answers)]

    def _resolve_question_map(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Any]]]:
        questions_path = cfg.get("questions_path")
        if not questions_path:
            return None
        path = Path(questions_path)
        if not path.exists():
            raise FileNotFoundError(f"questions_path not found: {path}")
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "questions_path requires pandas to load questions.parquet."
            ) from exc
        df = pd.read_parquet(path, columns=["graph_id", "question_uid", "question", "answer_texts"])
        id_col = "graph_id" if "graph_id" in df.columns else "question_uid"
        if id_col not in df.columns:
            raise ValueError("questions.parquet missing graph_id/question_uid column.")
        question_map: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            sample_id = str(row.get(id_col))
            if not sample_id:
                continue
            question = str(row.get("question") or "")
            answers = self._coerce_answer_texts(row.get("answer_texts"))
            question_map[sample_id] = {
                "question_text": question,
                "answer_texts": answers,
            }
        return question_map

    @staticmethod
    def _coerce_answer_text(answer_texts: list[str]) -> str:
        if not answer_texts:
            return ""
        if len(answer_texts) == _ONE:
            return answer_texts[0]
        return " | ".join(answer_texts)

    def _inject_question_text(self, records: list[Dict[str, Any]]) -> None:
        if self.question_map is None:
            return
        for sample in records:
            sample_id = str(sample.get("sample_id", ""))
            meta = self.question_map.get(sample_id)
            if meta is None:
                question_text = str(sample.get("question") or "")
                answer_texts: list[str] = []
            else:
                question_text = str(meta.get("question_text") or "")
                answer_texts = list(meta.get("answer_texts") or [])
            sample["question_text"] = question_text
            sample["answer_texts"] = answer_texts
            sample["answer_text"] = self._coerce_answer_text(answer_texts)

    def _inject_trajectory_text(self, records: list[Dict[str, Any]]) -> None:
        for sample in records:
            rollouts = sample.get("rollouts")
            if not isinstance(rollouts, list):
                continue
            for rollout in rollouts:
                edges = rollout.get("edges")
                if not isinstance(edges, list):
                    continue
                parts: list[str] = []
                for edge in edges:
                    if self._is_super_source_edge(edge):
                        continue
                    src = edge.get("src_text") or edge.get("src_entity_id")
                    rel = edge.get("relation_text") or edge.get("relation_id")
                    dst = edge.get("dst_text") or edge.get("dst_entity_id")
                    if rel == STOP_RELATION or str(rel) == str(STOP_RELATION):
                        rel = "SELF"
                    parts.append(f"{src} --{rel}--> {dst}")
                rollout["trajectory_text"] = " ; ".join(parts)

    @staticmethod
    def _is_super_source_edge(edge: Dict[str, Any]) -> bool:
        for key in ("src_entity_id", "head_entity_id"):
            value = edge.get(key)
            try:
                if int(value) == _SUPER_SOURCE_ENTITY_ID:
                    return True
            except Exception:
                continue
        src_text = str(edge.get("src_text") or "").strip()
        return src_text == str(_SUPER_SOURCE_ENTITY_ID)


__all__ = ["DualFlowRolloutArtifactWriter"]
