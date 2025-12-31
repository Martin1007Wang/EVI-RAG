from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter

from src.models.components.retriever import RetrieverOutput
from src.utils import extract_sample_ids, infer_batch_size, normalize_k_values

logger = logging.getLogger(__name__)

_ZERO = 0
_ONE = 1
_DEFAULT_SPLIT = "test"
_DEFAULT_ARTIFACT_NAME = "eval_retriever"
_DEFAULT_SCHEMA_VERSION = _ONE
_WRITE_INTERVAL = "batch"
_DEFAULT_TOPK_VALUES = (1, 5, 10)
_RANK_OFFSET = _ONE


@dataclass(frozen=True)
class _BatchContext:
    num_graphs: int
    scores_all: torch.Tensor
    labels_all: torch.Tensor
    query_ids: torch.Tensor
    edge_ptr: Optional[List[int]]
    edge_index_all: torch.Tensor
    edge_attr_all: torch.Tensor
    node_global_ids: torch.Tensor
    answer_ids_all: torch.Tensor
    answer_ptr: List[int]
    sample_ids: List[str]
    questions: List[str]
    logits_fwd_all: Optional[torch.Tensor]
    logits_bwd_all: Optional[torch.Tensor]


@dataclass(frozen=True)
class _GraphSlice:
    scores: torch.Tensor
    labels: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    logits_fwd: Optional[torch.Tensor]
    logits_bwd: Optional[torch.Tensor]


@dataclass(frozen=True)
class _TopKEdges:
    scores: List[float]
    labels: List[float]
    head_ids: List[int]
    tail_ids: List[int]
    relation_ids: List[int]
    logits_fwd: Optional[List[float]]
    logits_bwd: Optional[List[float]]


class RetrieverTopKEdgeWriter(BasePredictionWriter):
    """Persist retriever top-k edges during predict/test."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str = _DEFAULT_SPLIT,
        enabled: bool = True,
        artifact_name: str = _DEFAULT_ARTIFACT_NAME,
        schema_version: int = _DEFAULT_SCHEMA_VERSION,
        topk_values: Optional[Sequence[int]] = None,
        textualize: bool = False,
        entity_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
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
        self.topk_values = normalize_k_values(topk_values, default=_DEFAULT_TOPK_VALUES)
        if not self.topk_values:
            raise ValueError("topk_values must be a non-empty list of positive integers.")
        self._max_topk = max(int(k) for k in self.topk_values)
        self.textualize = bool(textualize)
        self.entity_vocab_path = entity_vocab_path
        self.relation_vocab_path = relation_vocab_path
        self.overwrite = bool(overwrite)

        self._records: List[Dict[str, Any]] = []
        self._output_path: Optional[Path] = None
        self._manifest_path: Optional[Path] = None
        self._entity_map: Optional[Dict[int, str]] = None
        self._relation_map: Optional[Dict[int, str]] = None

    def on_predict_start(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        self._records = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._output_path = self.output_dir / f"{self.split}.pt"
        self._manifest_path = self.output_dir / f"{self.split}.manifest.json"
        if self._output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output path already exists: {self._output_path}")
        if self.textualize:
            self._entity_map, self._relation_map = self._resolve_vocab_maps()

    def _collect_prediction(self, *, prediction: Any, batch: Any) -> None:
        if not self.enabled or prediction is None:
            return
        output = self._coerce_prediction(prediction)
        if output is None:
            return
        records = self._build_records(batch=batch, output=output)
        if records:
            self._records.extend(records)

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
        self._collect_prediction(prediction=prediction, batch=batch)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or not self._records:
            return
        records = self._gather_records(self._records)
        if not records:
            return
        if trainer.global_rank != _ZERO:
            return
        if self._output_path is None or self._manifest_path is None:
            raise RuntimeError("RetrieverTopKEdgeWriter missing output path; on_predict_start was not called.")

        payload = {
            "settings": {
                "split": self.split,
                "topk_values": [int(k) for k in self.topk_values],
            },
            "samples": records,
        }
        torch.save(payload, self._output_path)
        manifest = {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": self._output_path.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "retriever_topk_edge_writer",
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def on_test_start(self, trainer, pl_module) -> None:
        self.on_predict_start(trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._collect_prediction(prediction=outputs, batch=batch)

    def on_test_end(self, trainer, pl_module) -> None:
        self.on_predict_end(trainer, pl_module)

    def _build_records(self, *, batch: Any, output: RetrieverOutput) -> List[Dict[str, Any]]:
        ctx = self._build_context(batch=batch, output=output)
        records: List[Dict[str, Any]] = []
        for gid in range(ctx.num_graphs):
            graph_slice = self._slice_graph(ctx, gid)
            if graph_slice is None:
                continue
            record = self._build_record(ctx, graph_slice, gid)
            if record is not None:
                records.append(record)
        return records

    def _build_context(self, *, batch: Any, output: RetrieverOutput) -> _BatchContext:
        num_graphs = infer_batch_size(batch)
        scores_all = torch.sigmoid(output.logits).detach().view(-1)
        labels_all = batch.labels.detach().view(-1).to(dtype=torch.float32)
        if scores_all.numel() != labels_all.numel():
            raise ValueError(f"scores/labels shape mismatch: {scores_all.shape} vs {labels_all.shape}")
        query_ids = output.query_ids.detach().view(-1)

        slice_dict = getattr(batch, "_slice_dict", None)
        edge_ptr = self._resolve_edge_ptr(slice_dict, num_graphs)
        if edge_ptr is None and query_ids.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids.shape} vs {scores_all.shape}")

        edge_index_all = self._require_tensor(batch, "edge_index", "Batch missing edge_index required for persistence.")
        edge_attr_all = self._require_tensor(batch, "edge_attr", "Batch missing edge_attr required for persistence.")
        node_global_ids = self._require_tensor(
            batch,
            "node_global_ids",
            "Batch missing node_global_ids required for persistence.",
        )
        answer_ids_all = self._require_tensor(
            batch,
            "answer_entity_ids",
            "Batch missing answer_entity_ids required for persistence.",
        )
        answer_ptr = self._resolve_answer_ptr(batch, slice_dict, num_graphs)

        sample_ids = extract_sample_ids(batch)
        questions = self._extract_questions(batch, num_graphs)
        logits_fwd_all = output.logits_fwd.detach().view(-1) if output.logits_fwd is not None else None
        logits_bwd_all = output.logits_bwd.detach().view(-1) if output.logits_bwd is not None else None

        return _BatchContext(
            num_graphs=num_graphs,
            scores_all=scores_all,
            labels_all=labels_all,
            query_ids=query_ids,
            edge_ptr=edge_ptr,
            edge_index_all=edge_index_all,
            edge_attr_all=edge_attr_all,
            node_global_ids=node_global_ids,
            answer_ids_all=answer_ids_all,
            answer_ptr=answer_ptr,
            sample_ids=sample_ids,
            questions=questions,
            logits_fwd_all=logits_fwd_all,
            logits_bwd_all=logits_bwd_all,
        )

    def _slice_graph(self, ctx: _BatchContext, gid: int) -> Optional[_GraphSlice]:
        if ctx.edge_ptr is not None:
            return self._slice_graph_by_ptr(ctx, gid)
        return self._slice_graph_by_mask(ctx, gid)

    def _slice_graph_by_ptr(self, ctx: _BatchContext, gid: int) -> Optional[_GraphSlice]:
        start = int(ctx.edge_ptr[gid])
        end = int(ctx.edge_ptr[gid + 1])
        if end <= start:
            return None
        return _GraphSlice(
            scores=ctx.scores_all[start:end],
            labels=ctx.labels_all[start:end],
            edge_index=ctx.edge_index_all[:, start:end],
            edge_attr=ctx.edge_attr_all[start:end],
            logits_fwd=ctx.logits_fwd_all[start:end] if ctx.logits_fwd_all is not None else None,
            logits_bwd=ctx.logits_bwd_all[start:end] if ctx.logits_bwd_all is not None else None,
        )

    def _slice_graph_by_mask(self, ctx: _BatchContext, gid: int) -> Optional[_GraphSlice]:
        mask = ctx.query_ids == gid
        if not bool(mask.any().item()):
            return None
        return _GraphSlice(
            scores=ctx.scores_all[mask],
            labels=ctx.labels_all[mask],
            edge_index=ctx.edge_index_all[:, mask],
            edge_attr=ctx.edge_attr_all[mask],
            logits_fwd=ctx.logits_fwd_all[mask] if ctx.logits_fwd_all is not None else None,
            logits_bwd=ctx.logits_bwd_all[mask] if ctx.logits_bwd_all is not None else None,
        )

    def _build_record(self, ctx: _BatchContext, graph: _GraphSlice, gid: int) -> Optional[Dict[str, Any]]:
        topk_edges = self._select_topk_edges(graph, ctx.node_global_ids)
        if topk_edges is None:
            return None
        triplets_by_k = self._build_triplets_by_k(topk_edges)
        record = {
            "sample_id": self._select_sample_id(ctx.sample_ids, gid),
            "question": self._select_question(ctx.questions, gid),
            "triplets_by_k": triplets_by_k,
            "answer_entity_ids": self._select_answer_ids(ctx, gid),
        }
        return record

    def _select_topk_edges(self, graph: _GraphSlice, node_global_ids: torch.Tensor) -> Optional[_TopKEdges]:
        num_edges = int(graph.scores.numel())
        if num_edges <= _ZERO:
            return None
        k_top = min(num_edges, self._max_topk)
        if k_top <= _ZERO:
            return None
        top_scores, top_idx = torch.topk(graph.scores, k=k_top, largest=True, sorted=True)
        edge_index_top = graph.edge_index[:, top_idx]
        head_ids = node_global_ids[edge_index_top[0]].detach().cpu().tolist()
        tail_ids = node_global_ids[edge_index_top[1]].detach().cpu().tolist()
        relation_ids = graph.edge_attr[top_idx].detach().cpu().tolist()
        labels_top = graph.labels[top_idx].detach().cpu().tolist()
        scores_top = top_scores.detach().cpu().tolist()
        logits_fwd_top = graph.logits_fwd[top_idx].detach().cpu().tolist() if graph.logits_fwd is not None else None
        logits_bwd_top = graph.logits_bwd[top_idx].detach().cpu().tolist() if graph.logits_bwd is not None else None
        return _TopKEdges(
            scores=scores_top,
            labels=labels_top,
            head_ids=head_ids,
            tail_ids=tail_ids,
            relation_ids=relation_ids,
            logits_fwd=logits_fwd_top,
            logits_bwd=logits_bwd_top,
        )

    def _build_triplets_by_k(self, topk: _TopKEdges) -> Dict[int, List[Dict[str, Any]]]:
        triplets_by_k: Dict[int, List[Dict[str, Any]]] = {}
        max_len = len(topk.scores)
        for k in self.topk_values:
            k_val = int(k)
            top_k = min(k_val, max_len)
            triplets_by_k[k_val] = [self._build_edge_record(topk, idx) for idx in range(top_k)]
        return triplets_by_k

    def _build_edge_record(self, topk: _TopKEdges, idx: int) -> Dict[str, Any]:
        head_val = int(topk.head_ids[idx]) if topk.head_ids else None
        rel_val = int(topk.relation_ids[idx]) if topk.relation_ids else None
        tail_val = int(topk.tail_ids[idx]) if topk.tail_ids else None
        record = {
            "head_entity_id": head_val,
            "relation_id": rel_val,
            "tail_entity_id": tail_val,
            "head_text": self._lookup_text(self._entity_map, head_val),
            "relation_text": self._lookup_text(self._relation_map, rel_val),
            "tail_text": self._lookup_text(self._entity_map, tail_val),
            "score": float(topk.scores[idx]),
            "label": float(topk.labels[idx]),
            "rank": int(idx + _RANK_OFFSET),
        }
        if topk.logits_fwd is not None:
            record["logit_fwd"] = float(topk.logits_fwd[idx])
        if topk.logits_bwd is not None:
            record["logit_bwd"] = float(topk.logits_bwd[idx])
        return record

    @staticmethod
    def _lookup_text(mapping: Optional[Dict[int, str]], key: Optional[int]) -> Optional[str]:
        if mapping is None or key is None:
            return None
        return mapping.get(key)

    @staticmethod
    def _select_sample_id(sample_ids: List[str], gid: int) -> str:
        if gid < len(sample_ids):
            return str(sample_ids[gid])
        return str(gid)

    @staticmethod
    def _select_question(questions: List[str], gid: int) -> str:
        if gid < len(questions):
            return str(questions[gid])
        return ""

    def _select_answer_ids(self, ctx: _BatchContext, gid: int) -> List[int]:
        start = int(ctx.answer_ptr[gid])
        end = int(ctx.answer_ptr[gid + 1])
        answer_ids = ctx.answer_ids_all[start:end].detach().cpu().tolist()
        return [int(x) for x in answer_ids]

    @staticmethod
    def _extract_questions(batch: Any, num_graphs: int) -> List[str]:
        raw = getattr(batch, "question", None)
        if raw is None:
            return ["" for _ in range(num_graphs)]
        if isinstance(raw, (list, tuple)):
            return [str(q) for q in raw]
        if torch.is_tensor(raw):
            if raw.numel() == num_graphs:
                return [str(v.item()) for v in raw]
            return [str(raw.cpu().tolist()) for _ in range(num_graphs)]
        return [str(raw) for _ in range(num_graphs)]

    @staticmethod
    def _resolve_edge_ptr(slice_dict: Any, num_graphs: int) -> Optional[List[int]]:
        if not isinstance(slice_dict, dict) or "edge_index" not in slice_dict:
            return None
        candidate = torch.as_tensor(slice_dict.get("edge_index"), dtype=torch.long).view(-1)
        if candidate.numel() != num_graphs + _ONE:
            return None
        return candidate.tolist()

    @staticmethod
    def _resolve_answer_ptr(batch: Any, slice_dict: Any, num_graphs: int) -> List[int]:
        answer_ptr_raw = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr_raw is None and isinstance(slice_dict, dict):
            answer_ptr_raw = slice_dict.get("answer_entity_ids")
        if answer_ptr_raw is None:
            raise ValueError("Batch missing answer_entity_ids_ptr required for persistence.")
        answer_ptr = torch.as_tensor(answer_ptr_raw, dtype=torch.long).view(-1).tolist()
        if len(answer_ptr) != num_graphs + _ONE:
            raise ValueError(
                f"answer_entity_ids_ptr length mismatch: {len(answer_ptr)} vs expected {num_graphs + _ONE}"
            )
        return answer_ptr

    @staticmethod
    def _require_tensor(batch: Any, name: str, error: str) -> torch.Tensor:
        value = getattr(batch, name, None)
        if value is None:
            raise ValueError(error)
        return value

    def _resolve_vocab_maps(self) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not self.textualize:
            return None, None
        if not self.entity_vocab_path or not self.relation_vocab_path:
            raise ValueError("textualize=true requires both entity_vocab_path and relation_vocab_path.")
        entity_path = Path(self.entity_vocab_path)
        relation_path = Path(self.relation_vocab_path)
        if not entity_path.exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {entity_path}")
        if not relation_path.exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {relation_path}")

        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        try:
            import pyarrow.parquet as pq

            ent_table = pq.read_table(entity_path, columns=["entity_id", "label"])
            ent_ids = ent_table.column("entity_id").to_pylist()
            ent_labels = ent_table.column("label").to_pylist()
            ent_map = {int(i): str(l) for i, l in zip(ent_ids, ent_labels) if i is not None and l is not None}

            rel_table = pq.read_table(relation_path, columns=["relation_id", "label"])
            rel_ids = rel_table.column("relation_id").to_pylist()
            rel_labels = rel_table.column("label").to_pylist()
            rel_map = {int(i): str(l) for i, l in zip(rel_ids, rel_labels) if i is not None and l is not None}
        except Exception as exc:  # pragma: no cover
            if _get_dist_rank() == _ZERO:
                logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map

    @staticmethod
    def _gather_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if dist.get_rank() != _ZERO:
                return []
            merged: List[Dict[str, Any]] = []
            for part in gathered:
                if part:
                    merged.extend(part)
            return merged
        return records

    @staticmethod
    def _coerce_prediction(prediction: Any) -> Optional[RetrieverOutput]:
        if isinstance(prediction, RetrieverOutput):
            return prediction
        if isinstance(prediction, dict) and "logits" in prediction and "query_ids" in prediction:
            return RetrieverOutput(
                logits=prediction["logits"],
                query_ids=prediction["query_ids"],
                relation_ids=prediction.get("relation_ids"),
                logits_fwd=prediction.get("logits_fwd"),
                logits_bwd=prediction.get("logits_bwd"),
            )
        return None


def _get_dist_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _ZERO


__all__ = ["RetrieverTopKEdgeWriter"]
