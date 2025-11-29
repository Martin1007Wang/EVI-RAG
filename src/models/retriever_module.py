from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import hydra
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False,ignore=["retriever"]) 

        self.model = retriever
        
        # Optional: Torch 2.0 Compile
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model backbone with torch.compile...")
            self.model = torch.compile(self.model)

        self.loss =loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 3. Evaluation Settings
        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._ranking_k = normalize_k_values(eval_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        self._answer_k = normalize_k_values(eval_cfg.get("answer_recall_k", [5, 10]), default=[5, 10])
        
        # Buffer for accumulating eval predictions
        self._ranking_storage: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}

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
        
        log_metric(self, "train/loss", loss_output.loss, on_step=True, prog_bar=True, batch_size=batch_size)
        
        # Log auxiliary components if any (e.g. regularization terms)
        for k, v in getattr(loss_output, "components", {}).items():
            log_metric(self, f"train/loss/{k}", v, batch_size=batch_size)
            
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
        )

        # Buffer predictions for epoch-end ranking metrics (all CPU to save VRAM)
        head_ids = batch.node_global_ids[batch.edge_index[0]].detach().cpu()
        tail_ids = batch.node_global_ids[batch.edge_index[1]].detach().cpu()
        if not hasattr(batch, "answer_entity_ids") or not hasattr(batch, "answer_entity_ids_ptr"):
            raise ValueError("Batch must provide answer_entity_ids and answer_entity_ids_ptr for metrics.")
        answer_ids = batch.answer_entity_ids.detach().cpu()
        answer_ptr = batch.answer_entity_ids_ptr.detach().cpu()
        self._ranking_storage[split].append({
            "scores": output.scores.detach().cpu(),
            "labels": batch.labels.detach().cpu(),
            "query_ids": output.query_ids.detach().cpu(),
            "head_ids": head_ids,
            "tail_ids": tail_ids,
            "answer_entity_ids": answer_ids,
            "answer_entity_ids_ptr": answer_ptr,
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
            )

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
        )
        for k, v in stats.ndcg_at_k.items():
            log_metric(
                self,
                f"{split}/ranking/ndcg@{k}",
                v,
                batch_size=batch_size,
                sync_dist=False,
            )
        for k, v in stats.recall_at_k.items():
            log_metric(
                self,
                f"{split}/ranking/recall@{k}",
                v,
                batch_size=batch_size,
                sync_dist=False,
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
            answer_ids = batch_data.get("answer_entity_ids")
            answer_ptr = batch_data.get("answer_entity_ids_ptr")
            if head_ids is None or tail_ids is None or answer_ids is None or answer_ptr is None:
                continue

            unique_queries = query_ids.unique()
            for q_idx in unique_queries:
                mask = query_ids == q_idx
                start = int(answer_ptr[int(q_idx)].item())
                end = int(answer_ptr[int(q_idx) + 1].item())
                sample_answers = answer_ids[start:end]
                yield {
                    "scores": scores[mask],
                    "labels": labels[mask],
                    "answer_ids": sample_answers,
                    "head_ids": head_ids[mask] if head_ids is not None else None,
                    "tail_ids": tail_ids[mask] if tail_ids is not None else None,
                }
