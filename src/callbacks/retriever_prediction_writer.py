from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import BasePredictionWriter

from src.models.components.retriever import RetrieverOutput
from src.utils import extract_sample_ids, infer_batch_size, normalize_k_values

logger = logging.getLogger(__name__)


class RetrieverPredictionWriter(BasePredictionWriter):
    """Persist retriever top-k edges during predict()."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        split: str = "test",
        enabled: bool = True,
        artifact_name: str = "eval_retriever",
        schema_version: int = 1,
        ranking_k: Optional[Sequence[int]] = None,
        textualize: bool = False,
        entity_vocab_path: Optional[str] = None,
        relation_vocab_path: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__(write_interval="batch")
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir)
        self.split = str(split)
        self.artifact_name = str(artifact_name).strip()
        if not self.artifact_name:
            raise ValueError("artifact_name must be a non-empty string.")
        self.schema_version = int(schema_version)
        if self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer.")
        self.ranking_k = normalize_k_values(ranking_k, default=[1, 5, 10])
        if not self.ranking_k:
            raise ValueError("ranking_k must be a non-empty list of positive integers.")
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
        output = self._coerce_prediction(prediction)
        if output is None:
            return
        records = self._build_records(batch=batch, output=output)
        if records:
            self._records.extend(records)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or not self._records:
            return
        records = self._gather_records(self._records)
        if not records:
            return
        if trainer.global_rank != 0:
            return
        if self._output_path is None or self._manifest_path is None:
            raise RuntimeError("RetrieverPredictionWriter missing output path; on_predict_start was not called.")

        payload = {
            "settings": {
                "split": self.split,
                "ranking_k": [int(k) for k in self.ranking_k],
            },
            "samples": records,
        }
        torch.save(payload, self._output_path)
        manifest = {
            "artifact": self.artifact_name,
            "schema_version": self.schema_version,
            "file": self._output_path.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "retriever_module",
        }
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _build_records(self, *, batch: Any, output: RetrieverOutput) -> List[Dict[str, Any]]:
        num_graphs = infer_batch_size(batch)
        scores_all = torch.sigmoid(output.logits).detach().view(-1)
        labels_all = batch.labels.detach().view(-1).to(dtype=torch.float32)
        if scores_all.numel() != labels_all.numel():
            raise ValueError(f"scores/labels shape mismatch: {scores_all.shape} vs {labels_all.shape}")
        query_ids = output.query_ids.detach().view(-1)

        slice_dict = getattr(batch, "_slice_dict", None)
        edge_ptr: Optional[List[int]] = None
        if isinstance(slice_dict, dict) and "edge_index" in slice_dict:
            candidate = torch.as_tensor(slice_dict.get("edge_index"), dtype=torch.long).view(-1)
            if candidate.numel() == num_graphs + 1:
                edge_ptr = candidate.tolist()
        if edge_ptr is None and query_ids.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids.shape} vs {scores_all.shape}")

        edge_index_all = getattr(batch, "edge_index", None)
        edge_attr_all = getattr(batch, "edge_attr", None)
        node_global_ids = getattr(batch, "node_global_ids", None)
        if edge_index_all is None or edge_attr_all is None or node_global_ids is None:
            raise ValueError("Batch missing edge_index/edge_attr/node_global_ids required for persistence.")

        answer_ids_all = getattr(batch, "answer_entity_ids", None)
        if answer_ids_all is None:
            raise ValueError("Batch missing answer_entity_ids required for persistence.")
        answer_ptr_raw = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr_raw is None and isinstance(slice_dict, dict):
            answer_ptr_raw = slice_dict.get("answer_entity_ids")
        if answer_ptr_raw is None:
            raise ValueError("Batch missing answer_entity_ids_ptr required for persistence.")
        answer_ptr = torch.as_tensor(answer_ptr_raw, dtype=torch.long).view(-1).tolist()
        if len(answer_ptr) != num_graphs + 1:
            raise ValueError(f"answer_entity_ids_ptr length mismatch: {len(answer_ptr)} vs expected {num_graphs + 1}")

        sample_ids = extract_sample_ids(batch)
        questions = self._extract_questions(batch, num_graphs)

        logits_fwd_all = output.logits_fwd.detach().view(-1) if output.logits_fwd is not None else None
        logits_bwd_all = output.logits_bwd.detach().view(-1) if output.logits_bwd is not None else None

        max_ranking_k = max(int(k) for k in self.ranking_k)
        records: List[Dict[str, Any]] = []
        for gid in range(int(num_graphs)):
            if edge_ptr is not None:
                start = int(edge_ptr[gid])
                end = int(edge_ptr[gid + 1])
                if end <= start:
                    continue
                scores = scores_all[start:end]
                labels = labels_all[start:end]
                edge_index_g = edge_index_all[:, start:end]
                edge_attr_g = edge_attr_all[start:end]
                logits_fwd = logits_fwd_all[start:end] if logits_fwd_all is not None else None
                logits_bwd = logits_bwd_all[start:end] if logits_bwd_all is not None else None
            else:
                mask = query_ids == gid
                if not bool(mask.any().item()):
                    continue
                scores = scores_all[mask]
                labels = labels_all[mask]
                edge_index_g = edge_index_all[:, mask]
                edge_attr_g = edge_attr_all[mask]
                logits_fwd = logits_fwd_all[mask] if logits_fwd_all is not None else None
                logits_bwd = logits_bwd_all[mask] if logits_bwd_all is not None else None

            num_edges = int(scores.numel())
            if num_edges <= 0:
                continue
            k_top = min(num_edges, max_ranking_k)
            if k_top <= 0:
                continue
            top_scores, top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True)

            edge_index_top = edge_index_g[:, top_idx]
            head_ids = node_global_ids[edge_index_top[0]].detach().cpu().tolist()
            tail_ids = node_global_ids[edge_index_top[1]].detach().cpu().tolist()
            relation_ids = edge_attr_g[top_idx].detach().cpu().tolist()
            labels_top = labels[top_idx].detach().cpu().tolist()
            scores_top = top_scores.detach().cpu().tolist()
            logits_fwd_top = logits_fwd[top_idx].detach().cpu().tolist() if logits_fwd is not None else None
            logits_bwd_top = logits_bwd[top_idx].detach().cpu().tolist() if logits_bwd is not None else None

            a_start = int(answer_ptr[gid])
            a_end = int(answer_ptr[gid + 1])
            answer_ids = answer_ids_all[a_start:a_end].detach().cpu().tolist()

            triplets_by_k: Dict[int, List[Dict[str, Any]]] = {}
            for k in self.ranking_k:
                top_k = min(int(k), int(len(scores_top)))
                edges: List[Dict[str, Any]] = []
                for rank_idx in range(top_k):
                    head_val = int(head_ids[rank_idx]) if head_ids else None
                    rel_val = int(relation_ids[rank_idx]) if relation_ids else None
                    tail_val = int(tail_ids[rank_idx]) if tail_ids else None
                    edges.append(
                        {
                            "head_entity_id": head_val,
                            "relation_id": rel_val,
                            "tail_entity_id": tail_val,
                            "head_text": self._entity_map.get(head_val) if self._entity_map is not None else None,
                            "relation_text": self._relation_map.get(rel_val) if self._relation_map is not None else None,
                            "tail_text": self._entity_map.get(tail_val) if self._entity_map is not None else None,
                            "score": float(scores_top[rank_idx]),
                            "label": float(labels_top[rank_idx]),
                            "rank": int(rank_idx + 1),
                        }
                    )
                    if logits_fwd_top is not None:
                        edges[-1]["logit_fwd"] = float(logits_fwd_top[rank_idx])
                    if logits_bwd_top is not None:
                        edges[-1]["logit_bwd"] = float(logits_bwd_top[rank_idx])
                triplets_by_k[int(k)] = edges

            sample_id = sample_ids[gid] if gid < len(sample_ids) else str(gid)
            question = questions[gid] if gid < len(questions) else ""
            record: Dict[str, Any] = {
                "sample_id": str(sample_id),
                "question": str(question),
                "triplets_by_k": triplets_by_k,
            }
            record["answer_entity_ids"] = [int(x) for x in answer_ids]
            records.append(record)
        return records

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
            if trainer_global_rank() == 0:
                logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map

    @staticmethod
    def _gather_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, records)
            if dist.get_rank() != 0:
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


def trainer_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0
