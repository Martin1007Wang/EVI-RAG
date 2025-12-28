from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall

from src.losses import LossOutput
from src.metrics import AnswerReachability, FeatureMonitor
from src.models.components.retriever import RetrieverOutput
from src.utils import (
    log_metric,
    normalize_k_values,
    setup_optimizer,
)

logger = logging.getLogger(__name__)

_LABEL_POSITIVE_THRESHOLD = 0.5


class RetrieverModule(LightningModule):
    def __init__(
        self,
        retriever: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer_cfg: Optional[DictConfig | Dict] = None,
        scheduler_cfg: Optional[DictConfig | Dict] = None,
        evaluation_cfg: Optional[DictConfig | Dict] = None,
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
                "optimizer_cfg",
                "scheduler_cfg",
                "evaluation_cfg",
            ],
        )

        self.model = retriever
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling retriever with torch.compile (dynamic=%s)...", compile_dynamic)
            self.model = torch.compile(self.model, dynamic=compile_dynamic)

        self.loss = loss
        self.optimizer_cfg = self._to_dict(optimizer_cfg or {})
        self.scheduler_cfg = self._to_dict(scheduler_cfg or {})

        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._ranking_k = normalize_k_values(eval_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        self._recall_k = normalize_k_values(eval_cfg.get("recall_k", self._ranking_k), default=self._ranking_k)
        self._reachability_k = normalize_k_values(eval_cfg.get("reachability_k", self._ranking_k), default=self._ranking_k)
        self._emit_predict_outputs = bool(eval_cfg.get("emit_predict_outputs", False))
        self._metrics_enabled = bool(eval_cfg.get("metrics_enabled", True))

        self.predict_metrics: Dict[str, Any] = {}
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        if self._metrics_enabled:
            metrics = MetricCollection(
                {
                    "ranking/mrr": RetrievalMRR(empty_target_action="skip"),
                    **{f"ranking/recall@{k}": RetrievalRecall(top_k=int(k), empty_target_action="skip") for k in self._recall_k},
                }
            )
            if self._reachability_k:
                metrics.add_metrics({"reachability": AnswerReachability(k_values=self._reachability_k)})
            metrics.add_metrics({"features": FeatureMonitor()})
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")

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

    def _profile(self, name: str):
        profiler = getattr(self, "profiler", None)
        if profiler is None and self.trainer is not None:
            profiler = getattr(self.trainer, "profiler", None)
        if profiler is None or not hasattr(profiler, "profile"):
            return nullcontext()
        return profiler.profile(name)

    def _update_metrics(
        self,
        metrics: Optional[MetricCollection],
        *,
        batch: Any,
        output: RetrieverOutput,
        num_graphs: int,
    ) -> None:
        if not self._metrics_enabled or metrics is None:
            return

        scores_all = output.logits.detach().view(-1)
        if scores_all.numel() == 0:
            return
        labels_all = batch.labels.detach().view(-1).to(dtype=torch.float32)
        if scores_all.numel() != labels_all.numel():
            raise ValueError(f"scores/labels shape mismatch: {scores_all.shape} vs {labels_all.shape}")
        query_ids = output.query_ids.detach().view(-1).to(dtype=torch.long)
        if query_ids.numel() != scores_all.numel():
            raise ValueError(f"query_ids/scores mismatch: {query_ids.shape} vs {scores_all.shape}")

        targets = labels_all > _LABEL_POSITIVE_THRESHOLD
        metrics.update(
            preds=scores_all,
            target=targets,
            indexes=query_ids,
            batch=batch,
            query_ids=query_ids,
            num_graphs=num_graphs,
            features=output.edge_embeddings,
        )

    def _log_metric_collection(self, metrics: Optional[MetricCollection]) -> Dict[str, torch.Tensor]:
        if not self._metrics_enabled or metrics is None:
            return {}
        values = metrics.compute()
        for name, value in values.items():
            log_metric(
                self,
                name,
                value,
                batch_size=1,
                sync_dist=False,
                on_step=False,
                on_epoch=True,
            )
        metrics.reset()
        return values

    @staticmethod
    def _require_num_graphs(batch: Any) -> int:
        node_ptr = getattr(batch, "ptr", None)
        if node_ptr is None:
            raise ValueError("Batch missing ptr required for explicit graph batching.")
        num_graphs = int(node_ptr.numel() - 1)
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")
        return num_graphs

    def _compute_loss_output(
        self,
        *,
        batch: Any,
        output: RetrieverOutput,
        num_graphs: int,
        training_step: int,
    ) -> LossOutput:
        targets = getattr(batch, "labels", None)
        if targets is None:
            raise ValueError("Batch missing labels required for retriever loss.")
        return self.loss(
            output,
            targets,
            training_step=training_step,
            edge_batch=output.query_ids,
            num_graphs=num_graphs,
        )

    @staticmethod
    def _assert_pos_neg_per_graph(
        *,
        targets: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> None:
        if targets.numel() == 0:
            raise ValueError("Retriever batch has empty labels; cannot compute InfoNCE.")
        if edge_batch.numel() != targets.numel():
            raise ValueError(f"edge_batch/labels mismatch: {edge_batch.shape} vs {targets.shape}")
        pos_mask = targets.view(-1) > _LABEL_POSITIVE_THRESHOLD
        edge_batch = edge_batch.view(-1).to(dtype=torch.long)
        counts_dtype = targets.dtype if torch.is_floating_point(targets) else torch.float32
        pos_counts = torch.zeros(num_graphs, device=targets.device, dtype=counts_dtype)
        neg_counts = torch.zeros_like(pos_counts)
        pos_counts.scatter_add_(0, edge_batch, pos_mask.to(dtype=counts_dtype))
        neg_counts.scatter_add_(0, edge_batch, (~pos_mask).to(dtype=counts_dtype))
        invalid = (pos_counts == 0) | (neg_counts == 0)
        if bool(invalid.any().item()):
            bad = torch.nonzero(invalid, as_tuple=False).view(-1).tolist()
            raise ValueError(f"Retriever batch has graphs without positives/negatives: {bad}")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Optional[RetrieverOutput]:
        """Predict loop returns detached outputs when emit_predict_outputs=true."""
        with self._profile("predict/forward"):
            output = self(batch)
        if self._emit_predict_outputs:
            pred_output = output.detach()
            pred_output.edge_embeddings = None
            return pred_output
        return None

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        num_graphs = self._require_num_graphs(batch)
        with self._profile("train/forward"):
            output = self(batch)
        with self._profile("train/loss"):
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
        optimizer = setup_optimizer(self, self.optimizer_cfg)
        scheduler_type = str(self.scheduler_cfg.get("type", "") or "").lower()
        if not scheduler_type:
            return optimizer
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", 10)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_cfg.get("interval", "epoch"),
                    "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
                },
            }
        if scheduler_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts"}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", 10)),
                T_mult=int(self.scheduler_cfg.get("t_mult", 1)),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.scheduler_cfg.get("interval", "epoch"),
                    "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
                },
            }
        return optimizer

    # ------------------------------------------------------------------ #
    # Evaluation Loop (Validation & Test)
    # ------------------------------------------------------------------ #
    def on_validation_epoch_start(self) -> None:
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_test_epoch_start(self) -> None:
        if self.test_metrics is not None:
            self.test_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_eval_step(batch, batch_idx, split="val", metrics=self.val_metrics)

    def test_step(self, batch: Any, batch_idx: int) -> RetrieverOutput | None:
        return self._shared_eval_step(
            batch,
            batch_idx,
            split="test",
            metrics=self.test_metrics,
            collect_predict_payload=self._emit_predict_outputs,
        )

    def on_validation_epoch_end(self) -> None:
        self._log_metric_collection(self.val_metrics)

    def on_test_epoch_end(self) -> None:
        self._log_metric_collection(self.test_metrics)

    def _shared_eval_step(
        self,
        batch: Any,
        batch_idx: int,
        *,
        split: str,
        metrics: Optional[MetricCollection],
        collect_predict_payload: bool = False,
    ) -> Optional[RetrieverOutput]:
        num_graphs = self._require_num_graphs(batch)
        with self._profile(f"{split}/forward"):
            output = self(batch)
        with self._profile(f"{split}/loss"):
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
        with self._profile(f"{split}/metrics_update"):
            self._update_metrics(metrics, batch=batch, output=output, num_graphs=num_graphs)

        if collect_predict_payload:
            pred_output = output.detach()
            pred_output.edge_embeddings = None
            return pred_output
        return None
