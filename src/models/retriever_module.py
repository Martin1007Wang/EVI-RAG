from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import pandas as pd
import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.models.components.retriever import RetrieverOutput
from src.losses import create_loss_function
from src.utils import (
    RankingStats,
    compute_answer_recall,
    compute_ranking_metrics,
    extract_sample_ids,
    infer_batch_size,
    log_metric,
    normalize_k_values,
)

logger = logging.getLogger(__name__)


class RetrieverModule(LightningModule):
    def __init__(
        self,
        retriever: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        evaluation_cfg: Optional[DictConfig | Dict] = None,
        compile_model: bool = False,
        compile_dynamic: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False,ignore=["retriever"]) 

        self.model = retriever
        
        # Optional: Torch 2.0 Compile
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model backbone with torch.compile (dynamic=%s)...", compile_dynamic)
            self.model = torch.compile(self.model, dynamic=compile_dynamic)

        self.loss =loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 3. Evaluation Settings
        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._ranking_k = normalize_k_values(eval_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        self._answer_k = normalize_k_values(eval_cfg.get("answer_recall_k", [5, 10]), default=[5, 10])
        
        # Buffer for accumulating eval predictions
        self._ranking_storage: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}
        # Optional eval-time persistence config (注入自 eval.py)
        self.eval_persist_cfg: Optional[Dict[str, Any]] = None

    @staticmethod
    def _to_dict(cfg: Union[DictConfig, Dict]) -> Dict:
        """Helper to sanitize Omegaconf configs."""
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    # ------------------------------------------------------------------ #
    # Core Forward & Prediction
    # ------------------------------------------------------------------ #
    def forward(self, batch) -> RetrieverOutput:
        """Training/Eval forward pass."""
        return self.model(batch)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Dict[str, Any]:
        """
        Interface for GAgent / Inference.
        Returns a lightweight dict on CPU (mostly) to avoid VRAM leaks in loops.
        """
        output = self(batch)
        # Detach + move to CPU to allow downstream CPU graph ops without GPU retention
        scores = output.scores.detach().cpu()
        logits = output.logits.detach().cpu()
        query_ids = output.query_ids.detach().cpu()
        relation_ids = output.relation_ids.detach().cpu() if output.relation_ids is not None else None

        def _maybe_cpu(attr_name: str):
            val = getattr(batch, attr_name, None)
            if val is None:
                return None
            return val.detach().cpu() if hasattr(val, "detach") else val

        q_ptr = getattr(batch, "q_local_indices_ptr", None)
        if q_ptr is None and hasattr(batch, "_slice_dict"):
            q_ptr = batch._slice_dict.get("q_local_indices")

        return {
            "query_ids": query_ids,
            "scores": scores,
            "logits": logits,
            "relation_ids": relation_ids,
            # Topology + metadata needed by GAgentBuilder
            "edge_index": _maybe_cpu("edge_index"),
            "edge_attr": _maybe_cpu("edge_attr"),
            "node_global_ids": _maybe_cpu("node_global_ids"),
            "ptr": _maybe_cpu("ptr"),
            "answer_entity_ids": _maybe_cpu("answer_entity_ids"),
            "answer_entity_ids_ptr": _maybe_cpu("answer_entity_ids_ptr"),
            "sample_id": getattr(batch, "sample_id", None),
            "q_local_indices": getattr(batch, "q_local_indices", None),
            "q_local_indices_ptr": q_ptr,
        }

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss_output = self.loss(output, batch.labels, training_step=self.global_step)
        batch_size = infer_batch_size(batch)
        
        log_metric(self, "train/loss", loss_output.loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        
        # Log auxiliary components if any (e.g. regularization terms)
        for k, v in getattr(loss_output, "components", {}).items():
            log_metric(self, f"train/loss/{k}", v, on_step=False, on_epoch=True, batch_size=batch_size)
            
        return loss_output.loss

    def configure_optimizers(self):
        # 1. 实例化优化器
        # params=self.parameters() 是必须在运行时注入的，所以用 partial instantiation
        optimizer = self.optimizer(params=self.parameters())
        scheduler = self.scheduler(optimizer)
        
        lr_scheduler_config = {
            "scheduler": scheduler,
        }
        
        return {"optimizer": optimizer, "lr_scheduler_config": lr_scheduler_config}

    # ------------------------------------------------------------------ #
    # Evaluation Loop (Validation & Test)
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self._shared_eval_step(batch, batch_idx, split="test")

    def _shared_eval_step(self, batch, batch_idx, split: str):
        """Unified evaluation logic: Calc Loss -> Log -> Buffer Predictions."""
        output = self(batch)
        loss_out = self.loss(output, batch.labels, training_step=self.global_step)
        batch_size = infer_batch_size(batch)
        
        log_metric(
            self,
            f"{split}/loss",
            loss_out.loss,
            batch_size=batch_size,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

        # Buffer predictions for epoch-end ranking metrics (all CPU to save VRAM)
        head_ids = batch.node_global_ids[batch.edge_index[0]].detach().cpu()
        tail_ids = batch.node_global_ids[batch.edge_index[1]].detach().cpu()
        relation_ids = batch.edge_attr.detach().cpu() if hasattr(batch, "edge_attr") else None
        if not hasattr(batch, "answer_entity_ids") or not hasattr(batch, "answer_entity_ids_ptr"):
            raise ValueError("Batch must provide answer_entity_ids and answer_entity_ids_ptr for metrics.")
        answer_ids = batch.answer_entity_ids.detach().cpu()
        answer_ptr = batch.answer_entity_ids_ptr.detach().cpu()
        sample_ids = extract_sample_ids(batch)
        questions = self._extract_questions(batch, len(sample_ids))
        self._ranking_storage[split].append({
            "scores": output.scores.detach().cpu(),
            "labels": batch.labels.detach().cpu(),
            "query_ids": output.query_ids.detach().cpu(),
            "head_ids": head_ids,
            "tail_ids": tail_ids,
            "answer_entity_ids": answer_ids,
            "answer_entity_ids_ptr": answer_ptr,
            "relation_ids": relation_ids,
            "sample_ids": sample_ids,
            "questions": questions,
        })

    def on_validation_epoch_end(self):
        self._compute_and_log_epoch_metrics(split="val")

    def on_test_epoch_end(self):
        self._compute_and_log_epoch_metrics(split="test")

    # ------------------------------------------------------------------ #
    # Metrics Aggregation
    # ------------------------------------------------------------------ #
    def _compute_and_log_epoch_metrics(self, split: str):
        local_samples = self._ranking_storage[split]
        if not local_samples:
            return

        # 1. DDP Gather: Collect samples from all GPUs
        all_samples_batches = self._ddp_gather(local_samples)
        
        # 2. Clear Buffer
        self._ranking_storage[split] = []
        
        # 3. Group by query_id on the fly to avoid extra tensor copies
        samples = list(self._iter_query_samples(all_samples_batches))

        # 4. Compute Metrics (Standard Utils)
        ranking_stats = compute_ranking_metrics(samples, self._ranking_k)
        answer_stats = compute_answer_recall(samples, self._answer_k)
        effective_batch = max(len(samples), 1)
        
        # 5. Log Results
        self._log_ranking_stats(split, ranking_stats, batch_size=effective_batch)
        for k, v in answer_stats.items():
            log_metric(
                self,
                f"{split}/answer/{k}",
                v,
                batch_size=effective_batch,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        # 6. Optional persistence for downstream LLM 评估
        self._maybe_persist_retriever_outputs(split, samples)

    def _ddp_gather(self, local_samples: List[Dict]) -> List[Dict]:
        """Gather data from all ranks."""
        if not dist.is_available() or not dist.is_initialized():
            return local_samples
            
        world_size = dist.get_world_size()
        gathered = [None for _ in range(world_size)]
        # distinct from all_gather, this handles pickles
        dist.all_gather_object(gathered, local_samples)
        
        # Flatten the list of lists: [Rank0_List, Rank1_List] -> Combined_List
        combined = []
        for rank_list in gathered:
            combined.extend(rank_list)
        return combined

    def _log_ranking_stats(self, split: str, stats: RankingStats, *, batch_size: int) -> None:
        log_metric(
            self,
            f"{split}/ranking/mrr",
            stats.mrr,
            batch_size=batch_size,
            sync_dist=False,
            on_step=False,
            on_epoch=True,
        )
        for k, v in stats.ndcg_at_k.items():
            log_metric(
                self,
                f"{split}/ranking/ndcg@{k}",
                v,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        for k, v in stats.recall_at_k.items():
            log_metric(
                self,
                f"{split}/ranking/recall@{k}",
                v,
                batch_size=batch_size,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )

    def _iter_query_samples(self, batch_list: List[Dict]) -> List[Dict]:
        """
        Generates per-query samples for metric computation directly from buffered batches.
        Avoids an additional flattening pass and keeps tensors on CPU.
        """
        for batch_data in batch_list:
            scores = batch_data["scores"]
            labels = batch_data["labels"]
            query_ids = batch_data["query_ids"]
            head_ids = batch_data.get("head_ids")
            tail_ids = batch_data.get("tail_ids")
            relation_ids = batch_data.get("relation_ids")
            answer_ids = batch_data.get("answer_entity_ids")
            answer_ptr = batch_data.get("answer_entity_ids_ptr")
            sample_ids = batch_data.get("sample_ids") or []
            questions = batch_data.get("questions") or []
            if head_ids is None or tail_ids is None or answer_ids is None or answer_ptr is None:
                continue

            unique_queries = query_ids.unique()
            for q_idx in unique_queries:
                mask = query_ids == q_idx
                start = int(answer_ptr[int(q_idx)].item())
                end = int(answer_ptr[int(q_idx) + 1].item())
                sample_answers = answer_ids[start:end]
                sample_index = int(q_idx)
                sample_id = sample_ids[sample_index] if sample_index < len(sample_ids) else str(sample_index)
                question = questions[sample_index] if sample_index < len(questions) else ""
                yield {
                    "scores": scores[mask],
                    "labels": labels[mask],
                    "answer_ids": sample_answers,
                    "head_ids": head_ids[mask] if head_ids is not None else None,
                    "tail_ids": tail_ids[mask] if tail_ids is not None else None,
                    "relation_ids": relation_ids[mask] if relation_ids is not None else None,
                    "sample_id": sample_id,
                    "question": question,
                }

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

    def _maybe_persist_retriever_outputs(self, split: str, samples: List[Dict[str, Any]]) -> None:
        cfg = getattr(self, "eval_persist_cfg", None) or {}
        if not cfg.get("enabled"):
            return
        splits = cfg.get("splits") or ["test"]
        if split not in splits or not samples:
            return

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        merged_samples = samples
        if world_size > 1:
            gathered: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, samples)
            if rank != 0:
                return
            merged_samples = []
            for part in gathered:
                if part:
                    merged_samples.extend(part)

        output_dir = Path(cfg.get("output_dir", "eval_retriever"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{split}_retriever_eval.pt"
        entity_map, relation_map = self._resolve_vocab_maps(cfg)

        ranking_k = [int(k) for k in self._ranking_k]
        records: List[Dict[str, Any]] = []
        for sample in merged_samples:
            scores = sample["scores"]
            labels = sample["labels"]
            head_ids = sample.get("head_ids")
            tail_ids = sample.get("tail_ids")
            relation_ids = sample.get("relation_ids")
            sample_id = sample.get("sample_id", "")
            question = sample.get("question", "")
            answer_ids = sample.get("answer_ids")
            order = torch.argsort(scores, descending=True)

            triplets_by_k: Dict[int, List[Dict[str, Any]]] = {}
            for k in ranking_k:
                top_indices = order[:k]
                edges: List[Dict[str, Any]] = []
                for rank_idx, edge_idx in enumerate(top_indices, start=1):
                    idx = int(edge_idx)
                    head_val = int(head_ids[idx].item()) if head_ids is not None else None
                    rel_val = int(relation_ids[idx].item()) if relation_ids is not None else None
                    tail_val = int(tail_ids[idx].item()) if tail_ids is not None else None
                    edges.append(
                        {
                            "head_entity_id": head_val,
                            "relation_id": rel_val,
                            "tail_entity_id": tail_val,
                            "head_text": entity_map.get(head_val) if entity_map is not None else None,
                            "relation_text": relation_map.get(rel_val) if relation_map is not None else None,
                            "tail_text": entity_map.get(tail_val) if entity_map is not None else None,
                            "score": float(scores[idx].item()),
                            "label": float(labels[idx].item()),
                            "rank": rank_idx,
                        }
                    )
                triplets_by_k[int(k)] = edges

            record: Dict[str, Any] = {
                "sample_id": str(sample_id),
                "question": str(question),
                "triplets_by_k": triplets_by_k,
            }
            if answer_ids is not None:
                record["answer_entity_ids"] = [int(x) for x in answer_ids.tolist()]
            records.append(record)

        payload = {
            "settings": {
                "split": split,
                "ranking_k": ranking_k,
                "answer_k": [int(k) for k in self._answer_k],
            },
            "samples": records,
        }
        torch.save(payload, output_path)
        logger.info("Persisted retriever eval outputs to %s (samples=%d)", output_path, len(records))

    @staticmethod
    def _resolve_vocab_maps(cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            ds = cfg.get("dataset_cfg") or {}
            out_dir = ds.get("out_dir")
            if out_dir:
                base = Path(out_dir)
                # 优先使用 entity_vocab.parquet（entity_id -> label），其次 embedding_vocab.parquet
                if not entity_path:
                    if (base / "entity_vocab.parquet").exists():
                        entity_path = base / "entity_vocab.parquet"
                    else:
                        entity_path = base / "embedding_vocab.parquet"
                if not relation_path:
                    relation_path = base / "relation_vocab.parquet"
        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        try:
            if entity_path and Path(entity_path).exists():
                ent_df = pd.read_parquet(entity_path)
                if "entity_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.entity_id.astype(int), ent_df.label.astype(str)))
                elif "embedding_id" in ent_df.columns:
                    ent_map = dict(zip(ent_df.embedding_id.astype(int), ent_df.label.astype(str)))
            if relation_path and Path(relation_path).exists():
                rel_df = pd.read_parquet(relation_path)
                rel_map = dict(zip(rel_df.relation_id.astype(int), rel_df.label.astype(str)))
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map
