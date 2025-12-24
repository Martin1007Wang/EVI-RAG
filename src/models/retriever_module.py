from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.models.components.retriever import RetrieverOutput
from src.losses import LossOutput
from src.utils import (
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
        optimizer: Any,
        scheduler: Any = None,
        evaluation_cfg: Optional[DictConfig | Dict] = None,
        eval_persist_cfg: Optional[DictConfig | Dict[str, Any]] = None,
        start_edge_loss_cfg: Optional[DictConfig | Dict[str, Any]] = None,
        compile_model: bool = False,
        compile_dynamic: bool = True,
    ) -> None:
        super().__init__()
        # 仅保存可序列化的原始标量；避免将 Hydra/nn.Module/partial 写入 checkpoint。
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "retriever",
                "loss",
                "optimizer",
                "scheduler",
                "evaluation_cfg",
                "eval_persist_cfg",
                "start_edge_loss_cfg",
            ],
        )

        self.model = retriever
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling retriever with torch.compile (dynamic=%s)...", compile_dynamic)
            self.model = torch.compile(self.model, dynamic=compile_dynamic)

        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._ranking_k = normalize_k_values(eval_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        self._answer_k = normalize_k_values(eval_cfg.get("answer_recall_k", [5, 10]), default=[5, 10])
        self._reachability_k = normalize_k_values(eval_cfg.get("reachability_k", self._ranking_k), default=self._ranking_k)
        self._path_k = normalize_k_values(eval_cfg.get("path_inclusion_k", self._ranking_k), default=self._ranking_k)
        self._emit_predict_outputs = bool(eval_cfg.get("emit_predict_outputs", False))
        self._predict_metrics: Dict[str, Any] = {}
        self._predict_split = str(eval_cfg.get("split", "test"))
        self._metrics_enabled = bool(eval_cfg.get("metrics_enabled", True))

        self._streaming_state: Dict[str, Dict[str, Any]] = {}
        self.eval_persist_cfg: Dict[str, Any] = self._to_dict(eval_persist_cfg or {})

        start_cfg = self._to_dict(start_edge_loss_cfg or {})
        self._start_edge_loss_enabled = bool(start_cfg.get("enabled", False))
        self._start_edge_loss_weight: Optional[float] = None
        self._start_edge_apply_to_eval = False
        if self._start_edge_loss_enabled:
            if "weight" not in start_cfg:
                raise ValueError("start_edge_loss_cfg.weight must be set when start_edge_loss is enabled.")
            if "apply_to_eval" not in start_cfg:
                raise ValueError("start_edge_loss_cfg.apply_to_eval must be set when start_edge_loss is enabled.")
            weight = float(start_cfg["weight"])
            if not (0.0 < weight <= 1.0):
                raise ValueError(f"start_edge_loss_cfg.weight must be in (0,1], got {weight}")
            self._start_edge_loss_weight = weight
            self._start_edge_apply_to_eval = bool(start_cfg["apply_to_eval"])

    @staticmethod
    def _to_dict(cfg: Union[DictConfig, Dict]) -> Dict[str, Any]:
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        return dict(cfg)

    # ------------------------------------------------------------------ #
    # Core Forward & Prediction
    # ------------------------------------------------------------------ #
    def forward(self, batch: Any) -> RetrieverOutput:
        return self.model(batch)

    def _build_predict_payload(self, batch: Any, output: RetrieverOutput) -> Dict[str, Any]:
        """统一的推理输出，供 predict/test 两个 loop 共享。"""
        logits = output.logits.detach().cpu()
        scores = torch.sigmoid(output.logits).detach().cpu()
        logits_fwd = output.logits_fwd.detach().cpu() if output.logits_fwd is not None else None
        logits_bwd = output.logits_bwd.detach().cpu() if output.logits_bwd is not None else None
        scores_fwd = torch.sigmoid(logits_fwd) if logits_fwd is not None else None
        scores_bwd = torch.sigmoid(logits_bwd) if logits_bwd is not None else None
        query_ids = output.query_ids.detach().cpu()
        relation_ids = output.relation_ids.detach().cpu() if output.relation_ids is not None else None

        def _maybe_cpu(attr_name: str) -> Any:
            val = getattr(batch, attr_name, None)
            if val is None:
                return None
            return val.detach().cpu() if hasattr(val, "detach") else val

        q_ptr = getattr(batch, "q_local_indices_ptr", None)
        if q_ptr is None and hasattr(batch, "_slice_dict"):
            q_ptr = batch._slice_dict.get("q_local_indices")
        if torch.is_tensor(q_ptr):
            q_ptr = q_ptr.detach().cpu()

        return {
            "query_ids": query_ids,
            "scores": scores,
            "logits": logits,
            "scores_fwd": scores_fwd,
            "scores_bwd": scores_bwd,
            "logits_fwd": logits_fwd,
            "logits_bwd": logits_bwd,
            "relation_ids": relation_ids,
            "edge_index": _maybe_cpu("edge_index"),
            "edge_attr": _maybe_cpu("edge_attr"),
            "node_global_ids": _maybe_cpu("node_global_ids"),
            "ptr": _maybe_cpu("ptr"),
            "answer_entity_ids": _maybe_cpu("answer_entity_ids"),
            "answer_entity_ids_ptr": _maybe_cpu("answer_entity_ids_ptr"),
            "sample_id": getattr(batch, "sample_id", None),
            "q_local_indices": _maybe_cpu("q_local_indices"),
            "q_local_indices_ptr": q_ptr,
        }

    def _compute_start_edge_mask(self, *, batch: Any, device: torch.device) -> torch.Tensor:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for start-edge loss.")
        q_local_indices = getattr(batch, "q_local_indices", None)
        if q_local_indices is None:
            raise ValueError("Batch missing q_local_indices required for start-edge loss.")
        q_local_indices = q_local_indices.to(device=device, dtype=torch.long).view(-1)
        if q_local_indices.numel() == 0:
            raise ValueError("q_local_indices is empty; start-edge loss requires non-empty start entities.")

        num_nodes = getattr(batch, "num_nodes", None)
        if num_nodes is None:
            ptr = getattr(batch, "ptr", None)
            if ptr is not None:
                num_nodes = int(ptr[-1].item())
        if num_nodes is None or int(num_nodes) <= 0:
            raise ValueError("Unable to infer num_nodes for start-edge loss.")
        num_nodes = int(num_nodes)

        node_is_start = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        valid = (q_local_indices >= 0) & (q_local_indices < num_nodes)
        if not bool(valid.any().item()):
            raise ValueError("q_local_indices are out of range for start-edge loss.")
        node_is_start[q_local_indices[valid]] = True

        edge_index = edge_index.to(device=device)
        return node_is_start[edge_index[0]] | node_is_start[edge_index[1]]

    def _compute_loss_output(
        self,
        *,
        batch: Any,
        output: RetrieverOutput,
        num_graphs: int,
        training_step: int,
    ) -> LossOutput:
        def _call_loss(edge_mask: Optional[torch.Tensor] = None) -> LossOutput:
            targets = getattr(batch, "labels", None)
            if targets is None:
                raise ValueError("Batch missing labels required for retriever loss.")
            return self.loss(
                output,
                targets,
                training_step=training_step,
                edge_batch=output.query_ids,
                num_graphs=num_graphs,
                edge_mask=edge_mask,
            )

        base = _call_loss(edge_mask=None)
        apply_mask = self._start_edge_loss_enabled and (self.training or self._start_edge_apply_to_eval)
        if not apply_mask:
            return base

        device = output.logits.device
        edge_mask = self._compute_start_edge_mask(batch=batch, device=device)
        masked = _call_loss(edge_mask=edge_mask)
        weight = float(self._start_edge_loss_weight) if self._start_edge_loss_weight is not None else 1.0
        total = base.loss + masked.loss * weight
        components = {f"start_edge_{k}": v for k, v in masked.components.items()}
        components.update({f"full_{k}": v for k, v in base.components.items()})
        components["start_edge"] = float(masked.loss.detach().cpu().item())
        components["full"] = float(base.loss.detach().cpu().item())
        metrics = {f"start_edge_{k}": v for k, v in masked.metrics.items()}
        metrics.update({f"full_{k}": v for k, v in base.metrics.items()})
        return LossOutput(loss=total, components=components, metrics=metrics)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Unified eval/export path (predict loop).
        - 更新 streaming metrics（predict split）。
        - 按需返回轻量 payload（仅当 emit_predict_outputs=true）。
        """
        split = self._predict_split
        num_graphs = infer_batch_size(batch)
        output = self(batch)
        loss_out = self._compute_loss_output(
            batch=batch,
            output=output,
            num_graphs=num_graphs,
            training_step=self.global_step,
        )
        # predict loop不依赖 log_metric，避免 Lightning 忽略；仅更新 streaming 状态。
        self._update_streaming_state(split=split, batch=batch, output=output, num_graphs=num_graphs)
        if self._emit_predict_outputs:
            return self._build_predict_payload(batch, output)
        return None

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        num_graphs = infer_batch_size(batch)
        output = self(batch)
        loss_output = self._compute_loss_output(
            batch=batch,
            output=output,
            num_graphs=num_graphs,
            training_step=self.global_step,
        )
        log_metric(
            self,
            "train/loss",
            loss_output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=num_graphs,
            sync_dist=True,
        )

        for k, v in getattr(loss_output, "components", {}).items():
            log_metric(
                self,
                f"train/loss/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=num_graphs,
                sync_dist=True,
            )
        for k, v in getattr(loss_output, "metrics", {}).items():
            log_metric(
                self,
                f"train/metric/{k}",
                v,
                on_step=False,
                on_epoch=True,
                batch_size=num_graphs,
                sync_dist=True,
            )

        return loss_output.loss

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

    # ------------------------------------------------------------------ #
    # Evaluation Loop (Validation & Test)
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self) -> None:
        self._reset_streaming_state(split="val")

    def on_test_epoch_start(self) -> None:
        self._reset_streaming_state(split="test")

    def on_predict_epoch_start(self) -> None:
        self._reset_streaming_state(split=self._predict_split)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, batch_idx, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any] | None:
        return self._shared_eval_step(
            batch,
            batch_idx,
            split="test",
            collect_predict_payload=self._emit_predict_outputs,
        )

    def on_validation_epoch_end(self) -> None:
        self._finalize_streaming_state(split="val")

    def on_test_epoch_end(self) -> None:
        self._finalize_streaming_state(split="test")

    def on_predict_epoch_end(self, results: Optional[List[Any]] = None) -> None:
        metrics = self._finalize_streaming_state(split=self._predict_split, log_metrics=False)
        self._predict_metrics = metrics
        if not metrics:
            return
        logger.info("Predict metrics: %s", {k: float(v) if hasattr(v, 'item') else v for k, v in metrics.items()})
        if self.trainer is not None and self.trainer.logger is not None:
            # 手动落日志，predict loop默认不会处理 self.log
            self.trainer.logger.log_metrics(metrics, step=0)
        if self.trainer is not None:
            try:
                self.trainer.callback_metrics.update(metrics)  # type: ignore[arg-type]
            except Exception:
                pass

    def _shared_eval_step(self, batch: Any, batch_idx: int, *, split: str, collect_predict_payload: bool = False) -> Optional[Dict[str, Any]]:
        num_graphs = infer_batch_size(batch)
        output = self(batch)
        loss_out = self._compute_loss_output(
            batch=batch,
            output=output,
            num_graphs=num_graphs,
            training_step=self.global_step,
        )

        log_metric(
            self,
            f"{split}/loss",
            loss_out.loss,
            batch_size=num_graphs,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )
        self._update_streaming_state(split=split, batch=batch, output=output, num_graphs=num_graphs)

        if collect_predict_payload:
            return self._build_predict_payload(batch, output)
        return None

    # ------------------------------------------------------------------ #
    # Streaming Metrics (no epoch-sized buffering)
    # ------------------------------------------------------------------ #
    def _reset_streaming_state(self, *, split: str) -> None:
        device = self.device
        ranking_k = [int(k) for k in self._ranking_k]
        max_ranking_k = max(ranking_k) if ranking_k else 1
        metrics_enabled = bool(self._metrics_enabled)
        answer_k = [int(k) for k in self._answer_k] if metrics_enabled else []
        max_answer_k = max(answer_k) if answer_k else 0
        reachability_k = [int(k) for k in self._reachability_k] if metrics_enabled else []
        max_reach_k = max(reachability_k) if reachability_k else 0
        path_k = [int(k) for k in self._path_k] if metrics_enabled else []
        max_path_k = max(path_k) if path_k else 0
        max_top_k = max(max_ranking_k, max_answer_k, max_reach_k, max_path_k)

        positions = torch.arange(1, max_ranking_k + 1, device=device, dtype=torch.float32)
        discounts = 1.0 / torch.log2(positions + 1.0)

        self._streaming_state[split] = {
            "device": device,
            "max_ranking_k": int(max_ranking_k),
            "max_answer_k": int(max_answer_k),
            "max_reach_k": int(max_reach_k),
            "max_path_k": int(max_path_k),
            "max_top_k": int(max_top_k),
            "metrics_enabled": metrics_enabled,
            "discounts": discounts,
            "discounts_cumsum": discounts.cumsum(0),
            "ranking_count": torch.zeros((), device=device, dtype=torch.float32),
            "mrr_sum": torch.zeros((), device=device, dtype=torch.float32),
            "precision_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "recall_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "f1_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "ndcg_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in ranking_k},
            "answer_count": torch.zeros((), device=device, dtype=torch.float32),
            "answer_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in answer_k},
            "reachability_count": torch.zeros((), device=device, dtype=torch.float32),
            "reachability_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in reachability_k},
            "path_inclusion_count": torch.zeros((), device=device, dtype=torch.float32),
            "path_inclusion_sum": {k: torch.zeros((), device=device, dtype=torch.float32) for k in path_k},
            "persist_samples": [],
        }

    @staticmethod
    def _all_reduce_inplace(value: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)

    def _finalize_streaming_state(self, *, split: str, log_metrics: bool = True) -> Dict[str, torch.Tensor]:
        state = self._streaming_state.get(split)
        if not state:
            return {}
        if not bool(state.get("metrics_enabled", True)):
            self._maybe_persist_retriever_outputs(split, state["persist_samples"])
            state["persist_samples"] = []
            return {}

        self._all_reduce_inplace(state["ranking_count"])
        self._all_reduce_inplace(state["mrr_sum"])
        for metric in ("precision_sum", "recall_sum", "f1_sum", "ndcg_sum"):
            for tensor in state[metric].values():
                self._all_reduce_inplace(tensor)
        self._all_reduce_inplace(state["answer_count"])
        for tensor in state["answer_sum"].values():
            self._all_reduce_inplace(tensor)
        self._all_reduce_inplace(state["reachability_count"])
        for tensor in state["reachability_sum"].values():
            self._all_reduce_inplace(tensor)
        self._all_reduce_inplace(state["path_inclusion_count"])
        for tensor in state["path_inclusion_sum"].values():
            self._all_reduce_inplace(tensor)

        ranking_count = state["ranking_count"]
        denom = ranking_count.clamp_min(1.0)
        effective_ranking_count = int(ranking_count.detach().cpu().item())
        batch_size = max(effective_ranking_count, 1)

        metrics: Dict[str, torch.Tensor] = {
            f"{split}/ranking/mrr": state["mrr_sum"] / denom,
        }
        for k in sorted(state["recall_sum"].keys()):
            metrics[f"{split}/ranking/recall@{k}"] = state["recall_sum"][k] / denom
        for k in sorted(state["precision_sum"].keys()):
            metrics[f"{split}/ranking/precision@{k}"] = state["precision_sum"][k] / denom
        for k in sorted(state["f1_sum"].keys()):
            metrics[f"{split}/ranking/f1@{k}"] = state["f1_sum"][k] / denom
        for k in sorted(state["ndcg_sum"].keys()):
            metrics[f"{split}/ranking/ndcg@{k}"] = state["ndcg_sum"][k] / denom

        answer_count = state["answer_count"]
        answer_denom = answer_count.clamp_min(1.0)
        effective_answer_count = int(answer_count.detach().cpu().item())
        answer_batch_size = max(effective_answer_count, 1)
        for k in sorted(state["answer_sum"].keys()):
            metrics[f"{split}/answer/answer_recall@{k}"] = state["answer_sum"][k] / answer_denom

        reach_count = state["reachability_count"]
        reach_denom = reach_count.clamp_min(1.0)
        effective_reach_count = int(reach_count.detach().cpu().item())
        reach_batch_size = max(effective_reach_count, 1)
        for k in sorted(state["reachability_sum"].keys()):
            metrics[f"{split}/reachability/answer_reachable@{k}"] = state["reachability_sum"][k] / reach_denom

        path_count = state["path_inclusion_count"]
        path_denom = path_count.clamp_min(1.0)
        effective_path_count = int(path_count.detach().cpu().item())
        path_batch_size = max(effective_path_count, 1)
        for k in sorted(state["path_inclusion_sum"].keys()):
            metrics[f"{split}/path_inclusion@{k}"] = state["path_inclusion_sum"][k] / path_denom

        if log_metrics:
            for name, value in metrics.items():
                log_metric(
                    self,
                    name,
                    value,
                    batch_size=(
                        batch_size
                        if "ranking" in name
                        else reach_batch_size
                        if "reachability" in name
                        else path_batch_size
                        if "path_inclusion" in name
                        else answer_batch_size
                    ),
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                )

        self._maybe_persist_retriever_outputs(split, state["persist_samples"])
        state["persist_samples"] = []
        return metrics

    def _update_streaming_state(
        self,
        *,
        split: str,
        batch: Any,
        output: RetrieverOutput,
        num_graphs: int,
    ) -> None:
        state = self._streaming_state.get(split)
        if state is None or state.get("device") != output.logits.device:
            self._reset_streaming_state(split=split)
            state = self._streaming_state[split]
        metrics_enabled = bool(state.get("metrics_enabled", True))

        scores_all = torch.sigmoid(output.logits).detach().view(-1)
        logits_fwd_all = output.logits_fwd.detach().view(-1) if output.logits_fwd is not None else None
        logits_bwd_all = output.logits_bwd.detach().view(-1) if output.logits_bwd is not None else None
        labels_all = batch.labels.detach().view(-1).to(dtype=torch.float32)
        if scores_all.numel() != labels_all.numel():
            raise ValueError(f"scores/labels shape mismatch: {scores_all.shape} vs {labels_all.shape}")

        answer_ids_all = getattr(batch, "answer_entity_ids", None)
        if answer_ids_all is None:
            raise ValueError("Batch missing answer_entity_ids required for metrics.")
        slice_dict = getattr(batch, "_slice_dict", None)
        answer_ptr_raw = getattr(batch, "answer_entity_ids_ptr", None)
        if answer_ptr_raw is None and isinstance(slice_dict, dict):
            answer_ptr_raw = slice_dict.get("answer_entity_ids")
        if answer_ptr_raw is None:
            raise ValueError("Batch missing answer_entity_ids_ptr required for metrics.")
        answer_ptr = torch.as_tensor(answer_ptr_raw, dtype=torch.long).view(-1).tolist()
        if len(answer_ptr) != num_graphs + 1:
            raise ValueError(f"answer_entity_ids_ptr length mismatch: {len(answer_ptr)} vs expected {num_graphs + 1}")

        edge_ptr: Optional[List[int]] = None
        if isinstance(slice_dict, dict) and "edge_index" in slice_dict:
            candidate = torch.as_tensor(slice_dict.get("edge_index"), dtype=torch.long).view(-1)
            if candidate.numel() == num_graphs + 1:
                edge_ptr = candidate.tolist()
        query_ids = output.query_ids.detach().view(-1)
        if edge_ptr is None and query_ids.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids.shape} vs {scores_all.shape}")

        persist_cfg = getattr(self, "eval_persist_cfg", None) or {}
        persist_splits = persist_cfg.get("splits") or ["test"]
        persist_enabled = bool(persist_cfg.get("enabled")) and split in persist_splits
        sample_ids = extract_sample_ids(batch) if persist_enabled else []
        questions = self._extract_questions(batch, num_graphs) if persist_enabled else []

        max_ranking_k = int(state["max_ranking_k"])
        max_answer_k = int(state["max_answer_k"])
        max_reach_k = int(state["max_reach_k"])
        max_path_k = int(state["max_path_k"])
        max_top_k = int(state["max_top_k"])
        discounts = state["discounts"]
        discounts_cumsum = state["discounts_cumsum"]
        need_edge_index = max_answer_k > 0 or persist_enabled or max_reach_k > 0 or max_path_k > 0

        node_ptr = getattr(batch, "ptr", None)
        if max_reach_k > 0 and node_ptr is None:
            raise ValueError("Batch missing ptr required for reachability metrics.")

        q_local_indices_all = getattr(batch, "q_local_indices", None)
        q_ptr_raw = getattr(batch, "q_local_indices_ptr", None)
        if q_ptr_raw is None and isinstance(slice_dict, dict):
            q_ptr_raw = slice_dict.get("q_local_indices")
        a_local_indices_all = getattr(batch, "a_local_indices", None)
        a_ptr_raw = getattr(batch, "a_local_indices_ptr", None)
        if a_ptr_raw is None and isinstance(slice_dict, dict):
            a_ptr_raw = slice_dict.get("a_local_indices")
        q_ptr = None
        a_ptr = None
        q_local_indices = None
        a_local_indices = None
        if max_reach_k > 0:
            if q_local_indices_all is None or a_local_indices_all is None or q_ptr_raw is None or a_ptr_raw is None:
                raise ValueError("Batch missing q_local_indices/a_local_indices required for reachability metrics.")
            q_ptr = torch.as_tensor(q_ptr_raw, dtype=torch.long, device=scores_all.device).view(-1)
            a_ptr = torch.as_tensor(a_ptr_raw, dtype=torch.long, device=scores_all.device).view(-1)
            q_local_indices = torch.as_tensor(q_local_indices_all, dtype=torch.long, device=scores_all.device).view(-1)
            a_local_indices = torch.as_tensor(a_local_indices_all, dtype=torch.long, device=scores_all.device).view(-1)
            if q_ptr.numel() != num_graphs + 1:
                raise ValueError(f"q_local_indices_ptr length mismatch: {q_ptr.numel()} vs expected {num_graphs + 1}")
            if a_ptr.numel() != num_graphs + 1:
                raise ValueError(f"a_local_indices_ptr length mismatch: {a_ptr.numel()} vs expected {num_graphs + 1}")

        gt_edge_indices_all = getattr(batch, "gt_path_edge_indices", None)
        gt_ptr_raw = getattr(batch, "gt_path_edge_indices_ptr", None)
        if gt_ptr_raw is None and isinstance(slice_dict, dict):
            gt_ptr_raw = slice_dict.get("gt_path_edge_indices")
        gt_ptr = None
        gt_edge_indices = None
        if max_path_k > 0:
            if gt_edge_indices_all is None or gt_ptr_raw is None:
                raise ValueError("Batch missing gt_path_edge_indices required for path_inclusion metrics.")
            gt_ptr = torch.as_tensor(gt_ptr_raw, dtype=torch.long, device=scores_all.device).view(-1)
            gt_edge_indices = torch.as_tensor(gt_edge_indices_all, dtype=torch.long, device=scores_all.device).view(-1)
            if gt_ptr.numel() != num_graphs + 1:
                raise ValueError(f"gt_path_edge_indices_ptr length mismatch: {gt_ptr.numel()} vs expected {num_graphs + 1}")

        for gid in range(int(num_graphs)):
            if edge_ptr is not None:
                start = int(edge_ptr[gid])
                end = int(edge_ptr[gid + 1])
                if end <= start:
                    continue
                scores = scores_all[start:end]
                labels = labels_all[start:end]
                edge_index_g = batch.edge_index[:, start:end] if need_edge_index else None
                edge_attr_g = batch.edge_attr[start:end] if persist_enabled else None
            else:
                mask = query_ids == gid
                if not bool(mask.any().item()):
                    continue
                scores = scores_all[mask]
                labels = labels_all[mask]
                edge_index_g = batch.edge_index[:, mask] if need_edge_index else None
                edge_attr_g = batch.edge_attr[mask] if persist_enabled else None

            num_edges = int(scores.numel())
            if num_edges <= 0:
                continue

            k_top = min(num_edges, max_top_k)
            top_scores, top_idx = torch.topk(scores, k=k_top, largest=True, sorted=True)

            # --- Ranking metrics (only defined when positives exist) ---
            if metrics_enabled:
                pos_mask = labels > 0.5
                pos_count = int(pos_mask.sum().item())
                if pos_count > 0:
                    best_pos = scores[pos_mask].max()
                    rank = (scores > best_pos).sum() + 1
                    state["mrr_sum"].add_(1.0 / rank.to(dtype=torch.float32))
                    state["ranking_count"].add_(1.0)

                    k_rank = min(k_top, max_ranking_k)
                    ranked_labels = labels[top_idx[:k_rank]].to(dtype=torch.float32)
                    hits_prefix = ranked_labels.cumsum(0)
                    dcg_prefix = (ranked_labels * discounts[:k_rank]).cumsum(0)
                    pos_denom = ranked_labels.new_tensor(float(pos_count))

                    for k in state["recall_sum"].keys():
                        k_int = int(k)
                        k_used = min(k_int, k_rank)
                        hits = hits_prefix[k_used - 1] if k_used > 0 else ranked_labels.new_zeros(())
                        recall = hits / pos_denom
                        precision = hits / ranked_labels.new_tensor(float(k_int))
                        denom_f1 = (precision + recall).clamp_min(1e-12)
                        f1 = (2 * precision * recall) / denom_f1

                        state["precision_sum"][k].add_(precision)
                        state["recall_sum"][k].add_(recall)
                        state["f1_sum"][k].add_(f1)

                        ideal_k = min(pos_count, k_used)
                        if ideal_k <= 0:
                            state["ndcg_sum"][k].add_(ranked_labels.new_zeros(()))
                        else:
                            ideal_dcg = discounts_cumsum[ideal_k - 1]
                            state["ndcg_sum"][k].add_(dcg_prefix[k_used - 1] / ideal_dcg)

            # --- Answer recall (depends on top-k endpoints, independent of positives) ---
            if metrics_enabled and state["answer_sum"]:
                a_start = int(answer_ptr[gid])
                a_end = int(answer_ptr[gid + 1])
                answer_ids = answer_ids_all[a_start:a_end]
                if answer_ids.numel() > 0 and max_answer_k > 0:
                    if edge_index_g is None:
                        raise ValueError("edge_index_g required for answer recall but missing.")
                    k_ans = min(k_top, max_answer_k)
                    top_edges = top_idx[:k_ans]
                    edge_index_top = edge_index_g[:, top_edges]
                    head_ids = batch.node_global_ids[edge_index_top[0]]
                    tail_ids = batch.node_global_ids[edge_index_top[1]]

                    answer_unique = torch.unique(answer_ids)
                    if answer_unique.numel() > 0:
                        answer_unique, _ = torch.sort(answer_unique)
                        num_answers = int(answer_unique.numel())
                        ranks = torch.arange(1, k_ans + 1, device=scores.device, dtype=torch.long)
                        first_rank = torch.full((num_answers,), k_ans + 1, dtype=torch.long, device=scores.device)

                        def _update_first_rank(entity_ids: torch.Tensor) -> None:
                            idx = torch.searchsorted(answer_unique, entity_ids)
                            valid = idx < num_answers
                            if not bool(valid.any().item()):
                                return
                            idx_valid = idx[valid]
                            ent_valid = entity_ids[valid]
                            match = answer_unique[idx_valid] == ent_valid
                            if not bool(match.any().item()):
                                return
                            first_rank.scatter_reduce_(
                                0,
                                idx_valid[match],
                                ranks[valid][match],
                                reduce="amin",
                                include_self=True,
                            )

                        _update_first_rank(head_ids)
                        _update_first_rank(tail_ids)

                        state["answer_count"].add_(1.0)
                        for k in state["answer_sum"].keys():
                            k_used = min(int(k), k_ans)
                            state["answer_sum"][k].add_((first_rank <= k_used).to(dtype=torch.float32).mean())

            # --- Answer reachability (undirected connectivity from start nodes) ---
            if metrics_enabled and state["reachability_sum"]:
                q_start = int(q_ptr[gid])
                q_end = int(q_ptr[gid + 1])
                a_start = int(a_ptr[gid])
                a_end = int(a_ptr[gid + 1])
                q_nodes = q_local_indices[q_start:q_end]
                a_nodes = a_local_indices[a_start:a_end]
                if q_nodes.numel() > 0 and a_nodes.numel() > 0 and edge_index_g is not None:
                    node_start = int(node_ptr[gid].item())
                    node_end = int(node_ptr[gid + 1].item())
                    num_nodes = max(0, node_end - node_start)
                    q_local = q_nodes[(q_nodes >= node_start) & (q_nodes < node_end)] - node_start
                    a_local = a_nodes[(a_nodes >= node_start) & (a_nodes < node_end)] - node_start
                    if num_nodes > 0 and q_local.numel() > 0 and a_local.numel() > 0:
                        edge_index_local = edge_index_g - node_start
                        reach_map = self._compute_reachability_at_k(
                            edge_index=edge_index_local,
                            top_idx=top_idx,
                            start_nodes=q_local,
                            answer_nodes=a_local,
                            num_nodes=num_nodes,
                            k_values=sorted(state["reachability_sum"].keys()),
                        )
                        if reach_map:
                            state["reachability_count"].add_(1.0)
                            for k, reachable in reach_map.items():
                                state["reachability_sum"][k].add_(1.0 if reachable else 0.0)

            # --- GT path inclusion (coverage of GT edges in Top-K) ---
            if metrics_enabled and state["path_inclusion_sum"]:
                gt_start = int(gt_ptr[gid])
                gt_end = int(gt_ptr[gid + 1])
                gt_edges = gt_edge_indices[gt_start:gt_end]
                if gt_edges.numel() > 0:
                    gt_edges = gt_edges[(gt_edges >= 0) & (gt_edges < num_edges)]
                if gt_edges.numel() > 0:
                    gt_unique = torch.unique(gt_edges)
                    gt_denom = float(gt_unique.numel())
                    if gt_denom > 0 and k_top > 0:
                        state["path_inclusion_count"].add_(1.0)
                        top_mask = torch.isin(top_idx[:k_top], gt_unique)
                        hits_prefix = top_mask.to(dtype=torch.float32).cumsum(0)
                        for k in state["path_inclusion_sum"].keys():
                            k_used = min(int(k), k_top)
                            hits = hits_prefix[k_used - 1] if k_used > 0 else hits_prefix.new_zeros(())
                            state["path_inclusion_sum"][k].add_(hits / gt_denom)

            # --- Optional persistence (store only top max_ranking_k edges) ---
            if persist_enabled:
                k_persist = min(k_top, max_ranking_k)
                if k_persist <= 0:
                    continue
                top_edges = top_idx[:k_persist]
                if edge_index_g is None or edge_attr_g is None:
                    raise ValueError("edge_index_g/edge_attr_g required for persistence but missing.")
                edge_index_top = edge_index_g[:, top_edges]
                head_ids = batch.node_global_ids[edge_index_top[0]].detach().cpu()
                tail_ids = batch.node_global_ids[edge_index_top[1]].detach().cpu()
                relation_ids = edge_attr_g[top_edges].detach().cpu()

                sample_id = sample_ids[gid] if gid < len(sample_ids) else str(gid)
                question = questions[gid] if gid < len(questions) else ""
                a_start = int(answer_ptr[gid])
                a_end = int(answer_ptr[gid + 1])
                answer_ids = answer_ids_all[a_start:a_end].detach().cpu()

                payload = {
                    "scores": top_scores[:k_persist].detach().cpu(),
                    "labels": labels[top_edges].detach().cpu(),
                    "head_ids": head_ids,
                    "tail_ids": tail_ids,
                    "relation_ids": relation_ids,
                    "sample_id": sample_id,
                    "question": question,
                    "answer_ids": answer_ids,
                }
                if logits_fwd_all is not None:
                    payload["logits_fwd"] = logits_fwd_all[top_edges].detach().cpu()
                if logits_bwd_all is not None:
                    payload["logits_bwd"] = logits_bwd_all[top_edges].detach().cpu()
                state["persist_samples"].append(payload)

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

    # ------------------------------------------------------------------ #
    # Optional persistence for downstream evaluation
    # ------------------------------------------------------------------ #
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
        output_path = output_dir / f"{split}.pt"
        manifest_path = output_dir / f"{split}.manifest.json"
        artifact_name = str(cfg.get("artifact_name", "eval_retriever")).strip()
        if not artifact_name:
            raise ValueError("eval_persist_cfg.artifact_name must be a non-empty string.")
        schema_version = int(cfg.get("schema_version", 1))
        if schema_version <= 0:
            raise ValueError("eval_persist_cfg.schema_version must be a positive integer.")
        entity_map, relation_map = self._resolve_vocab_maps(cfg)

        ranking_k = [int(k) for k in self._ranking_k]
        records: List[Dict[str, Any]] = []
        for sample in merged_samples:
            scores = sample["scores"]
            labels = sample["labels"]
            logits_fwd = sample.get("logits_fwd")
            logits_bwd = sample.get("logits_bwd")
            head_ids = sample.get("head_ids")
            tail_ids = sample.get("tail_ids")
            relation_ids = sample.get("relation_ids")
            sample_id = sample.get("sample_id", "")
            question = sample.get("question", "")
            answer_ids = sample.get("answer_ids")

            triplets_by_k: Dict[int, List[Dict[str, Any]]] = {}
            for k in ranking_k:
                top_k = min(int(k), int(scores.numel()))
                edges: List[Dict[str, Any]] = []
                for rank_idx in range(top_k):
                    head_val = int(head_ids[rank_idx].item()) if head_ids is not None else None
                    rel_val = int(relation_ids[rank_idx].item()) if relation_ids is not None else None
                    tail_val = int(tail_ids[rank_idx].item()) if tail_ids is not None else None
                    edges.append(
                        {
                            "head_entity_id": head_val,
                            "relation_id": rel_val,
                            "tail_entity_id": tail_val,
                            "head_text": entity_map.get(head_val) if entity_map is not None else None,
                            "relation_text": relation_map.get(rel_val) if relation_map is not None else None,
                            "tail_text": entity_map.get(tail_val) if entity_map is not None else None,
                            "score": float(scores[rank_idx].item()),
                            "label": float(labels[rank_idx].item()),
                            "rank": int(rank_idx + 1),
                        }
                    )
                    if logits_fwd is not None:
                        edges[-1]["logit_fwd"] = float(logits_fwd[rank_idx].item())
                    if logits_bwd is not None:
                        edges[-1]["logit_bwd"] = float(logits_bwd[rank_idx].item())
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
        manifest = {
            "artifact": artifact_name,
            "schema_version": schema_version,
            "file": output_path.name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "producer": "retriever_module",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(
            "Persisted retriever eval outputs to %s (samples=%d, manifest=%s)",
            output_path,
            len(records),
            manifest_path,
        )

    @staticmethod
    def _resolve_vocab_maps(cfg: Dict[str, Any]) -> tuple[Optional[Dict[int, str]], Optional[Dict[int, str]]]:
        if not cfg.get("textualize"):
            return None, None
        entity_path = cfg.get("entity_vocab_path")
        relation_path = cfg.get("relation_vocab_path")
        if not entity_path or not relation_path:
            raise ValueError("eval_persist_cfg.textualize=true requires both entity_vocab_path and relation_vocab_path.")
        if not Path(entity_path).exists():
            raise FileNotFoundError(f"entity_vocab_path not found: {entity_path}")
        if not Path(relation_path).exists():
            raise FileNotFoundError(f"relation_vocab_path not found: {relation_path}")

        ent_map: Optional[Dict[int, str]] = None
        rel_map: Optional[Dict[int, str]] = None
        try:
            import pyarrow.parquet as pq

            if entity_path and Path(entity_path).exists():
                ent_table = pq.read_table(entity_path, columns=["entity_id", "label"])
                ent_ids = ent_table.column("entity_id").to_pylist()
                ent_labels = ent_table.column("label").to_pylist()
                ent_map = {int(i): str(l) for i, l in zip(ent_ids, ent_labels) if i is not None and l is not None}
            if relation_path and Path(relation_path).exists():
                rel_table = pq.read_table(relation_path, columns=["relation_id", "label"])
                rel_ids = rel_table.column("relation_id").to_pylist()
                rel_labels = rel_table.column("label").to_pylist()
                rel_map = {int(i): str(l) for i, l in zip(rel_ids, rel_labels) if i is not None and l is not None}
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to load vocab for textualize: %s", exc)
        return ent_map, rel_map

    @staticmethod
    def _compute_reachability_at_k(
        *,
        edge_index: torch.Tensor,
        top_idx: torch.Tensor,
        start_nodes: torch.Tensor,
        answer_nodes: torch.Tensor,
        num_nodes: int,
        k_values: List[int],
    ) -> Dict[int, bool]:
        if num_nodes <= 0:
            return {}
        k_values = [int(k) for k in k_values if int(k) > 0]
        if not k_values:
            return {}
        start_nodes = start_nodes.view(-1)
        answer_nodes = answer_nodes.view(-1)
        if start_nodes.numel() == 0 or answer_nodes.numel() == 0:
            return {int(k): False for k in k_values}

        k_top = min(int(top_idx.numel()), max(k_values))
        if k_top <= 0:
            return {int(k): False for k in k_values}

        edge_index_top = edge_index[:, top_idx[:k_top]].detach().cpu()
        start_nodes_cpu = start_nodes.detach().cpu().view(-1).tolist()
        answer_nodes_cpu = answer_nodes.detach().cpu().view(-1).tolist()

        parent = list(range(num_nodes))
        rank = [0] * num_nodes

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            pa = _find(a)
            pb = _find(b)
            if pa == pb:
                return
            if rank[pa] < rank[pb]:
                parent[pa] = pb
            elif rank[pa] > rank[pb]:
                parent[pb] = pa
            else:
                parent[pb] = pa
                rank[pa] += 1

        def _reachable() -> bool:
            roots = {_find(s) for s in start_nodes_cpu}
            for a in answer_nodes_cpu:
                if _find(a) in roots:
                    return True
            return False

        k_check = sorted({min(int(k), k_top) for k in k_values})
        reach_map: Dict[int, bool] = {}
        next_idx = 0
        for idx in range(k_top):
            u = int(edge_index_top[0, idx].item())
            v = int(edge_index_top[1, idx].item())
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                _union(u, v)
            while next_idx < len(k_check) and idx + 1 >= k_check[next_idx]:
                reach_map[k_check[next_idx]] = _reachable()
                next_idx += 1
        while next_idx < len(k_check):
            reach_map[k_check[next_idx]] = _reachable()
            next_idx += 1

        return {int(k): reach_map[min(int(k), k_top)] for k in k_values}
