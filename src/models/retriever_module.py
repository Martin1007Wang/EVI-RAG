from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torchmetrics import MetricCollection

from src.losses import LossOutput
from src.metrics import (
    AnswerReachability,
    BridgeEdgeRecallAtK,
    BridgePositiveCoverage,
    BridgeProbQuality,
    EdgeRecallAtK,
    FeatureMonitor,
    ScoreMargin,
)
from src.models.components.retriever import RetrieverOutput
from src.utils.graph_utils import compute_qa_edge_mask
from src.utils import (
    log_metric,
    normalize_k_values,
    setup_optimizer,
)

logger = logging.getLogger(__name__)

_LABEL_POSITIVE_THRESHOLD = 0.5
_DEFAULT_K_VALUES = [1, 5, 10]


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
        if compile_model:
            logger.warning(
                "torch.compile is disabled for retriever; ignoring compile_model=true (dynamic=%s).",
                compile_dynamic,
            )

        self.loss = loss
        self.optimizer_cfg = self._to_dict(optimizer_cfg or {})
        self.scheduler_cfg = self._to_dict(scheduler_cfg or {})

        eval_cfg = self._to_dict(evaluation_cfg or {})
        self._edge_recall_k = self._resolve_k_values(
            eval_cfg,
            keys=("edge_recall_k",),
            default=_DEFAULT_K_VALUES,
        )
        self._connectivity_k = self._resolve_k_values(
            eval_cfg,
            keys=("connectivity_k",),
            default=self._edge_recall_k,
        )
        self._bridge_metrics = bool(eval_cfg.get("bridge_metrics", False))
        self._bridge_recall_k: list[int] = []
        if self._bridge_metrics:
            self._bridge_recall_k = self._resolve_k_values(
                eval_cfg,
                keys=("bridge_recall_k",),
                default=self._edge_recall_k,
            )
        self._emit_predict_outputs = bool(eval_cfg.get("emit_predict_outputs", False))
        self._metrics_enabled = bool(eval_cfg.get("metrics_enabled", True))
        self._feature_metrics = bool(eval_cfg.get("feature_metrics", False))
        self._ablate_topic = bool(eval_cfg.get("ablate_topic", False))

        self.predict_metrics: Dict[str, Any] = {}
        self.val_metrics: Optional[MetricCollection] = None
        self.test_metrics: Optional[MetricCollection] = None
        self.val_metrics_ablate: Optional[MetricCollection] = None
        self.test_metrics_ablate: Optional[MetricCollection] = None
        if self._metrics_enabled:
            metrics = MetricCollection({})
            if self._edge_recall_k:
                metrics.add_metrics({"edge_recall": EdgeRecallAtK(k_values=self._edge_recall_k)})
            if self._connectivity_k:
                metrics.add_metrics({"connectivity": AnswerReachability(k_values=self._connectivity_k)})
            metrics.add_metrics({"signal": ScoreMargin()})
            if self._bridge_metrics:
                if self._bridge_recall_k:
                    metrics.add_metrics({"bridge_edge_recall": BridgeEdgeRecallAtK(k_values=self._bridge_recall_k)})
                metrics.add_metrics({"bridge_signal": BridgeProbQuality()})
                metrics.add_metrics({"bridge_coverage": BridgePositiveCoverage()})
            if self._feature_metrics:
                metrics.add_metrics({"features": FeatureMonitor()})
            self.val_metrics = metrics.clone(prefix="val/")
            self.test_metrics = metrics.clone(prefix="test/")
            if self._ablate_topic:
                self.val_metrics_ablate = metrics.clone(prefix="val/ablate_topic/")
                self.test_metrics_ablate = metrics.clone(prefix="test/ablate_topic/")

    @staticmethod
    def _to_dict(cfg: Union[DictConfig, Dict]) -> Dict[str, Any]:
        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        return dict(cfg)

    @staticmethod
    def _resolve_k_values(eval_cfg: Dict[str, Any], *, keys: tuple[str, ...], default: Any) -> list[int]:
        for key in keys:
            if key in eval_cfg:
                return normalize_k_values(eval_cfg.get(key), default=default)
        return normalize_k_values(default, default=None)

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

    @staticmethod
    def _compute_edge_is_near(batch: Any, *, device: torch.device) -> torch.Tensor:
        cached = getattr(batch, "edge_is_near", None)
        if cached is not None:
            if not torch.is_tensor(cached):
                return torch.as_tensor(cached, dtype=torch.bool, device=device)
            return cached.to(device=device, dtype=torch.bool)
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for edge mask computation.")
        q_local_indices = getattr(batch, "q_local_indices", None)
        a_local_indices = getattr(batch, "a_local_indices", None)
        if q_local_indices is None or a_local_indices is None:
            raise ValueError("Batch missing q_local_indices/a_local_indices required for edge mask.")
        num_nodes = getattr(batch, "num_nodes", None)
        if num_nodes is None:
            raise ValueError("Batch missing num_nodes required for edge mask.")
        edge_index = edge_index.to(device=device)
        return compute_qa_edge_mask(
            edge_index,
            num_nodes=int(num_nodes),
            q_local_indices=q_local_indices,
            a_local_indices=a_local_indices,
        )

    def _edge_is_near_required(self) -> bool:
        if getattr(self.loss, "requires_edge_is_near", False):
            return True
        if not bool(getattr(self.model, "hide_seek_enabled", False)):
            return False
        if self.training:
            return True
        return bool(getattr(self.model, "hide_seek_apply_in_eval", False))

    def _ensure_edge_is_near(self, batch: Any) -> Optional[torch.Tensor]:
        if not self._edge_is_near_required():
            return None
        cached = getattr(batch, "edge_is_near", None)
        if cached is not None:
            return cached
        edge_index = getattr(batch, "edge_index", None)
        device = edge_index.device if torch.is_tensor(edge_index) else torch.device("cpu")
        edge_is_near = self._compute_edge_is_near(batch, device=device)
        batch.edge_is_near = edge_is_near
        return edge_is_near

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
        edge_is_near = None
        if getattr(self.loss, "requires_edge_is_near", False):
            edge_is_near = getattr(batch, "edge_is_near", None)
            if edge_is_near is None:
                edge_is_near = self._compute_edge_is_near(batch, device=output.logits.device)
                batch.edge_is_near = edge_is_near
        return self.loss(
            output,
            targets,
            training_step=training_step,
            edge_batch=output.query_ids,
            num_graphs=num_graphs,
            edge_is_near=edge_is_near,
        )

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
        self._ensure_edge_is_near(batch)
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
        if self.val_metrics_ablate is not None:
            self.val_metrics_ablate.reset()

    def on_test_epoch_start(self) -> None:
        if self.test_metrics is not None:
            self.test_metrics.reset()
        if self.test_metrics_ablate is not None:
            self.test_metrics_ablate.reset()

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
        self._log_metric_collection(self.val_metrics_ablate)

    def on_test_epoch_end(self) -> None:
        self._log_metric_collection(self.test_metrics)
        self._log_metric_collection(self.test_metrics_ablate)

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
        self._ensure_edge_is_near(batch)
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
        if self._ablate_topic:
            ablate_metrics = self.val_metrics_ablate if split == "val" else self.test_metrics_ablate
            if ablate_metrics is not None:
                with self._profile(f"{split}/metrics_ablate_topic"):
                    ablate_output = self._forward_with_topic_ablation(batch)
                self._update_metrics(ablate_metrics, batch=batch, output=ablate_output, num_graphs=num_graphs)

        if collect_predict_payload:
            pred_output = output.detach()
            pred_output.edge_embeddings = None
            return pred_output
        return None

    @staticmethod
    def _zero_like_topic(topic_one_hot: Any) -> torch.Tensor:
        if torch.is_tensor(topic_one_hot):
            return torch.zeros_like(topic_one_hot)
        return torch.zeros_like(torch.as_tensor(topic_one_hot))

    def _forward_with_topic_ablation(self, batch: Any) -> RetrieverOutput:
        topic_one_hot = getattr(batch, "topic_one_hot", None)
        if topic_one_hot is None:
            raise ValueError("topic_one_hot is required for ablation metrics.")
        batch.topic_one_hot = self._zero_like_topic(topic_one_hot)
        try:
            return self(batch)
        finally:
            batch.topic_one_hot = topic_one_hot
