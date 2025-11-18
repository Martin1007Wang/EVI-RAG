from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from lightning import LightningModule
from torchmetrics import Metric

from omegaconf import DictConfig, OmegaConf
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.losses import create_loss_function
from src.models.outputs import ProbabilisticOutput
from src.utils import (
    compute_answer_recall,
    RankingStats,
    compute_ranking_metrics,
    compute_selective_metrics,
    extract_answer_entity_ids,
    extract_sample_ids,
    normalize_k_values,
    setup_optimizer,
    summarize_uncertainty,
)
from .registry import create_retriever


logger = logging.getLogger(__name__)


class RetrieverModule(LightningModule):
    """LightningModule that wraps the legacy retriever + evidential losses."""

    def __init__(
        self,
        *,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]] = None,
        evaluation_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = create_retriever(**model_cfg)
        loss_cfg_dict = self._to_plain_dict(loss_cfg)
        self.loss_fn = create_loss_function(loss_cfg_dict)

        self.optimizer_cfg = self._to_plain_dict(optimizer_cfg or {})
        self.scheduler_cfg = self._to_plain_dict(scheduler_cfg or {})
        self.evaluation_cfg = self._to_plain_dict(evaluation_cfg or {})
        # Torchmetrics for validation/testing classification + calibration
        ranking_k = normalize_k_values(self.evaluation_cfg.get("ranking_k", [1, 5, 10]), default=[1, 5, 10])
        answer_k = normalize_k_values(self.evaluation_cfg.get("answer_recall_k", [5, 10]), default=[5, 10])
        self._ranking_k = ranking_k
        self._answer_k = answer_k
        self._ranking_storage: Dict[str, List[Dict[str, Any]]] = {"val": [], "test": []}
        self._run_identifier = self._derive_run_identifier()
        self.run_name = "default"
        self.metrics_db_path: Optional[Path] = None

        self._eval_metrics = {
            "val": self._create_eval_metric_bank(),
            "test": self._create_eval_metric_bank(),
        }

    def set_run_context(
        self,
        *,
        run_name: str,
        output_dir: str | Path,
        metrics_root: str | Path | None = None,
    ) -> None:
        self.run_name = str(run_name or "default")
        base = Path(metrics_root or output_dir)
        self.metrics_db_path = base / "metrics.db"

    def forward(self, batch):
        """Expose retriever forward for Lightning predict/eval APIs."""
        return self.model(batch)

    @staticmethod
    def _to_plain_dict(config: Dict[str, Any] | DictConfig) -> Dict[str, Any]:
        if isinstance(config, DictConfig):
            return dict(OmegaConf.to_container(config, resolve=True))  # type: ignore[arg-type]
        return dict(config)

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        model_output = self(batch)
        loss_output = self.loss_fn(model_output, batch.labels, training_step=self.global_step)
        loss = loss_output.loss
        metrics = self._collect_metrics(model_output, loss_output, loss, prefix="train")
        self._log_metrics(metrics, on_step=False, on_epoch=True, batch_size=batch.labels.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            model_output = self(batch)
            loss_output = self.loss_fn(model_output, batch.labels, training_step=self.global_step)
            metrics = self._collect_metrics(model_output, loss_output, loss_output.loss, prefix="val")
            self._log_metrics(metrics, on_step=False, on_epoch=True, batch_size=batch.labels.size(0))
            self._update_eval_metrics("val", model_output.scores, batch.labels)
            self._gather_predictions("val", model_output, batch)

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            model_output = self(batch)
            loss_output = self.loss_fn(model_output, batch.labels, training_step=self.global_step)
            metrics = self._collect_metrics(model_output, loss_output, loss_output.loss, prefix="test")
            self._log_metrics(metrics, on_step=False, on_epoch=True, batch_size=batch.labels.size(0))
            self._update_eval_metrics("test", model_output.scores, batch.labels)
            self._gather_predictions("test", model_output, batch)

    def configure_optimizers(self):
        optimizer = setup_optimizer(self.model, self.optimizer_cfg)
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        scheduler_interval = self.scheduler_cfg.get("interval", "epoch")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": scheduler_interval,
                "monitor": self.scheduler_cfg.get("monitor", "val/loss"),
            },
        }

    def _build_scheduler(self, optimizer):
        if not self.scheduler_cfg:
            return None
        sched_type = (self.scheduler_cfg.get("type") or self.scheduler_cfg.get("name") or "").lower()
        if not sched_type or sched_type in {"none", "null"}:
            return None
        if sched_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_cfg.get("mode", "min"),
                factor=float(self.scheduler_cfg.get("factor", 0.5)),
                patience=int(self.scheduler_cfg.get("patience", 5)),
                min_lr=float(self.scheduler_cfg.get("min_lr", 1e-6)),
            )
        if sched_type in {"cosine", "cosineannealing"}:
            max_epochs = getattr(self.trainer, "max_epochs", 1) or 1
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", max(1, max_epochs))),
                eta_min=float(self.scheduler_cfg.get("eta_min", 0.0)),
            )
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

    # ------------------------------------------------------------------ #
    # Metrics helpers
    # ------------------------------------------------------------------ #
    def _create_eval_metric_bank(self) -> Dict[str, Metric]:
        metrics: Dict[str, Metric] = {
            "calib/auroc": BinaryAUROC(),
            "calib/auprc": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
        }
        calib_metric = self._build_calibration_metric()
        if calib_metric is not None:
            metrics["calib/ece"] = calib_metric
        return metrics

    def _build_calibration_metric(self) -> Optional[Metric]:
        try:
            return BinaryCalibrationError()
        except Exception:
            return None

    @staticmethod
    def _move_metrics_to_device(metrics: Optional[Dict[str, Metric]], device: torch.device) -> None:
        if not metrics:
            return
        for metric in metrics.values():
            try:
                metric.to(device)
            except Exception:
                continue

    def _collect_metrics(self, model_output, loss_output, total_loss, prefix: str) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {
            f"{prefix}/loss": total_loss.detach(),
        }
        metrics |= self._extract_model_metrics(model_output, prefix, device=total_loss.device)
        metrics |= self._extract_loss_metrics(loss_output, prefix, device=total_loss.device)
        return metrics

    def _extract_model_metrics(self, model_output, prefix: str, device: torch.device) -> Dict[str, torch.Tensor]:
        collected: Dict[str, torch.Tensor] = {}

        def add_stats(tensor: Optional[torch.Tensor], name: str) -> None:
            stats = self._tensor_stats(tensor, name, device)
            collected.update(stats)

        add_stats(getattr(model_output, "scores", None), f"{prefix}/scores")
        add_stats(getattr(model_output, "logits", None), f"{prefix}/logits")
        if isinstance(model_output, ProbabilisticOutput):
            model_output.ensure_moments()
            add_stats(model_output.alpha, f"{prefix}/alpha")
            add_stats(model_output.beta, f"{prefix}/beta")
            add_stats(model_output.aleatoric, f"{prefix}/uncertainty/aleatoric")
            add_stats(model_output.epistemic, f"{prefix}/uncertainty/epistemic")
            add_stats(getattr(model_output, "evidence_logits", None), f"{prefix}/evidence_logits")
        return collected

    def _extract_loss_metrics(self, loss_output, prefix: str, device: torch.device) -> Dict[str, torch.Tensor]:
        collected: Dict[str, torch.Tensor] = {}

        def add_metric(name: str, value: Any) -> None:
            tensor = self._to_metric_tensor(value, device)
            if tensor is not None:
                collected[name] = tensor

        components = getattr(loss_output, "components", None) or {}
        for key, value in components.items():
            add_metric(f"{prefix}/loss/{key}", value)

        extras = getattr(loss_output, "metrics", None) or {}
        for key, value in extras.items():
            add_metric(f"{prefix}/{key}", value)

        return collected

    @staticmethod
    def _to_metric_tensor(value: Any, device: torch.device) -> Optional[torch.Tensor]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            tensor = value.detach().to(device=device, dtype=torch.float32)
            if tensor.numel() > 1:
                tensor = tensor.mean()
            return tensor
        try:
            return torch.tensor(float(value), device=device, dtype=torch.float32)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tensor_stats(tensor: Optional[torch.Tensor], prefix: str, device: torch.device) -> Dict[str, torch.Tensor]:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return {}
        data = tensor.detach().to(device=device, dtype=torch.float32).flatten()
        mean = data.mean()
        if data.numel() == 1:
            std = torch.zeros((), device=device, dtype=torch.float32)
        else:
            std = data.std(unbiased=False)
        return {
            f"{prefix}/mean": mean,
            f"{prefix}/std": std,
            f"{prefix}/min": data.min(),
            f"{prefix}/max": data.max(),
        }

    def _log_metrics(self, metrics: Dict[str, torch.Tensor], *, on_step: bool, on_epoch: bool, batch_size: int) -> None:
        for name, tensor in metrics.items():
            prog_bar = name.endswith("/loss")
            self.log(name, tensor, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar, batch_size=batch_size)

    # ------------------------------------------------------------------ #
    # Optimization diagnostics
    # ------------------------------------------------------------------ #
    def on_after_backward(self) -> None:
        """Log gradient norm for monitoring. Avoids clipping; purely diagnostic."""
        grad_sq_sum = None
        for param in self.model.parameters():
            if param.grad is None:
                continue
            v = param.grad.detach()
            g2 = torch.sum(v * v)
            grad_sq_sum = g2 if grad_sq_sum is None else grad_sq_sum + g2
        if grad_sq_sum is not None:
            grad_norm = torch.sqrt(grad_sq_sum)
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False, batch_size=1)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        """Log current learning rate after optimizer step."""
        opt = self.trainer.optimizers[0] if self.trainer is not None and self.trainer.optimizers else None
        if opt is None:
            return
        try:
            lr = opt.param_groups[0]["lr"]
        except Exception:
            return
        lr_tensor = torch.tensor(lr, device=self.device, dtype=torch.float32)
        self.log("train/lr", lr_tensor, on_step=True, on_epoch=False, prog_bar=False, batch_size=1)

    # ------------------------------------------------------------------ #
    # Validation/Test torchmetrics
    # ------------------------------------------------------------------ #
    def _update_eval_metrics(self, split: str, scores: torch.Tensor, targets: torch.Tensor) -> None:
        metrics = self._eval_metrics.get(split)
        if not metrics:
            return
        scores = scores.detach()
        targets = targets.detach().long()
        self._move_metrics_to_device(metrics, scores.device)
        for metric in metrics.values():
            metric.update(scores, targets)

    def _gather_predictions(self, split: str, model_output, batch) -> None:
        storage = self._ranking_storage.get(split)
        if storage is None:
            return
        sample_ids = extract_sample_ids(batch)
        if not sample_ids:
            return
        scores = model_output.scores.detach().float().cpu()
        labels = batch.labels.detach().float().cpu()
        query_ids = model_output.query_ids.detach().cpu()
        edge_index = batch.edge_index.detach().cpu()
        node_ids = batch.node_global_ids.detach().cpu()
        node_ptr = getattr(batch, "ptr", None)
        if node_ptr is not None:
            node_ptr = node_ptr.detach().cpu()
        aleatoric = getattr(model_output, "aleatoric", None)
        if aleatoric is not None:
            aleatoric = aleatoric.detach().float().cpu()
        epistemic = getattr(model_output, "epistemic", None)
        if epistemic is not None:
            epistemic = epistemic.detach().float().cpu()
        alpha = getattr(model_output, "alpha", None)
        if alpha is not None:
            alpha = alpha.detach().float().cpu()
        beta = getattr(model_output, "beta", None)
        if beta is not None:
            beta = beta.detach().float().cpu()
        posterior_mean = getattr(model_output, "posterior_mean", None)
        if posterior_mean is not None:
            posterior_mean = posterior_mean.detach().float().cpu()

        for sample_idx, sample_id in enumerate(sample_ids):
            mask = query_ids == sample_idx
            if mask.sum().item() == 0:
                continue
            sample_scores = scores[mask]
            sample_labels = labels[mask]
            head_ids = node_ids[edge_index[0][mask]]
            tail_ids = node_ids[edge_index[1][mask]]
            answer_ids = extract_answer_entity_ids(batch, sample_idx, node_ptr, node_ids)
            sample_aleatoric = aleatoric[mask] if aleatoric is not None else None
            sample_epistemic = epistemic[mask] if epistemic is not None else None
            sample_alpha = alpha[mask] if alpha is not None else None
            sample_beta = beta[mask] if beta is not None else None
            posterior = posterior_mean[mask] if posterior_mean is not None else None
            sample_entry = {
                "sample_id": str(sample_id),
                "scores": sample_scores,
                "labels": sample_labels,
                "head_ids": head_ids,
                "tail_ids": tail_ids,
                "answer_ids": answer_ids,
                "aleatoric": sample_aleatoric,
                "epistemic": sample_epistemic,
                "alpha": sample_alpha,
                "beta": sample_beta,
                "posterior_mean": posterior,
            }
            storage.append(sample_entry)

    def on_validation_epoch_end(self) -> None:
        self._log_eval_metrics("val")
        self._log_ranking_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_eval_metrics("test")
        self._log_ranking_metrics("test")

    def _log_eval_metrics(self, split: str) -> None:
        metrics = self._eval_metrics.get(split)
        if not metrics:
            return
        self._move_metrics_to_device(metrics, self.device)
        for name, metric in metrics.items():
            value = metric.compute()
            self.log(f"{split}/{name}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            metric.reset()

    def _log_ranking_metrics(self, split: str) -> None:
        samples = self._ranking_storage.get(split)
        if not samples:
            return
        samples = self._gather_all_samples(samples)
        ranking_stats = compute_ranking_metrics(samples, self._ranking_k)
        answer = compute_answer_recall(samples, self._answer_k)
        uncertainty = self._compute_uncertainty_summary(samples)
        selective = self._compute_selective_metrics(samples)
        self.log(f"{split}/ranking/mrr", ranking_stats.mrr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for k, value in ranking_stats.precision_at_k.items():
            self.log(f"{split}/ranking/precision@{k}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for k, value in ranking_stats.recall_at_k.items():
            self.log(f"{split}/ranking/recall@{k}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for k, value in ranking_stats.f1_at_k.items():
            self.log(f"{split}/ranking/f1@{k}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for k, value in ranking_stats.ndcg_at_k.items():
            self.log(f"{split}/ranking/ndcg@{k}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for name, value in answer.items():
            self.log(f"{split}/answer/{name}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for name, value in uncertainty.items():
            self.log(f"{split}/uncertainty/{name}", value, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        for domain, metrics in selective.items():
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    self._export_reliability(split, domain, metric_name, value)
                    continue
                self.log(
                    f"{split}/selective/{domain}/{metric_name}",
                    float(value),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )
        self._ranking_storage[split] = []
        if split == "test":
            self._write_metrics_to_db(split, ranking_stats, answer, uncertainty, selective)

    def _compute_uncertainty_summary(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        ale_mean, ale_p95 = summarize_uncertainty((sample.get("aleatoric") for sample in samples))
        epi_mean, epi_p95 = summarize_uncertainty((sample.get("epistemic") for sample in samples))
        return {"aleatoric_mean": ale_mean, "aleatoric_p95": ale_p95, "epistemic_mean": epi_mean, "epistemic_p95": epi_p95}

    def _compute_selective_metrics(self, samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        if not samples:
            return {}
        eps = 1e-8
        log2 = math.log(2.0)
        top_k = int(self.evaluation_cfg.get("selective_top_k", 1) or 1)

        def normalize_entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
            probs = probs.clamp(eps, 1.0 - eps)
            entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
            return torch.clamp(1.0 - entropy / log2, 0.0, 1.0)

        def normalize_uncertainty_as_confidence(uncertainty: torch.Tensor) -> torch.Tensor:
            return torch.clamp(1.0 - uncertainty / log2, 0.0, 1.0)

        concentration_offset = float(self.evaluation_cfg.get("selective_concentration_offset", math.log(32.0)))
        concentration_scale = float(self.evaluation_cfg.get("selective_concentration_scale", 8.0))

        def normalize_concentration(alpha_plus_beta: torch.Tensor) -> torch.Tensor:
            conc = alpha_plus_beta.clamp_min(eps)
            log_conc = torch.log(conc)
            return torch.sigmoid((log_conc - concentration_offset) / concentration_scale)

        def concat(tensors: List[torch.Tensor]) -> torch.Tensor:
            return torch.cat(tensors, dim=0) if tensors else torch.empty(0)

        # Build two views: query-level top-k (primary) and full candidate list (debugging).
        q_labels: List[torch.Tensor] = []
        q_scores: List[torch.Tensor] = []
        q_ale: List[torch.Tensor] = []
        q_epi: List[torch.Tensor] = []
        q_alpha_beta: List[torch.Tensor] = []
        q_combo: List[torch.Tensor] = []

        cand_labels: List[torch.Tensor] = []
        cand_scores: List[torch.Tensor] = []
        cand_ale: List[torch.Tensor] = []
        cand_epi: List[torch.Tensor] = []
        cand_alpha_beta: List[torch.Tensor] = []

        for sample in samples:
            scores = sample["scores"]
            labels = sample["labels"]
            if scores.numel() == 0 or labels.numel() == 0:
                continue
            # Candidate view (full list)
            posterior = sample.get("posterior_mean")
            if isinstance(posterior, torch.Tensor) and posterior.numel() == scores.numel():
                cand_scores.append(posterior)
            else:
                cand_scores.append(scores)
            cand_labels.append(labels)
            ale = sample.get("aleatoric")
            epi = sample.get("epistemic")
            alpha = sample.get("alpha")
            beta = sample.get("beta")
            if isinstance(ale, torch.Tensor) and ale.numel() > 0:
                cand_ale.append(ale)
            if isinstance(epi, torch.Tensor) and epi.numel() > 0:
                cand_epi.append(epi)
            if isinstance(alpha, torch.Tensor) and isinstance(beta, torch.Tensor) and alpha.numel() > 0 and beta.numel() > 0:
                cand_alpha_beta.append(alpha + beta)

            # Query view: top-k per query for selective metrics
            k = min(top_k, scores.numel())
            if k <= 0:
                continue
            values, idx = torch.topk(scores, k)
            q_scores.append(values)
            q_labels.append(labels[idx])
            if isinstance(ale, torch.Tensor) and ale.numel() >= k:
                q_ale.append(ale[idx])
            if isinstance(epi, torch.Tensor) and epi.numel() >= k:
                q_epi.append(epi[idx])
            if isinstance(alpha, torch.Tensor) and isinstance(beta, torch.Tensor) and alpha.numel() >= k and beta.numel() >= k:
                q_alpha_beta.append((alpha + beta)[idx])
            if isinstance(ale, torch.Tensor) and isinstance(epi, torch.Tensor) and ale.numel() >= k and epi.numel() >= k:
                q_combo.append(ale[idx] + epi[idx])

        labels_query = concat(q_labels)
        labels_cand = concat(cand_labels)
        if labels_query.numel() == 0 and labels_cand.numel() == 0:
            return {}

        def resolve_partial_coverages(base: List[float], labels_for_rate: torch.Tensor) -> List[float]:
            coverages: Set[float] = set(float(c) for c in base if 0.0 < float(c) <= 1.0)
            extra = self.evaluation_cfg.get("partial_coverages_extra", [])
            coverages.update(float(c) for c in extra if isinstance(c, (int, float)) and 0.0 < float(c) <= 1.0)
            if labels_for_rate.numel() > 0:
                pos_rate = float(labels_for_rate.float().mean().item())
                for factor in (1.0, 2.0, 4.0):
                    cov = min(1.0, pos_rate * factor)
                    if cov > 0:
                        coverages.add(cov)
            return sorted(coverages) or [0.5, 0.9, 0.95]

        base_partial = self.evaluation_cfg.get("partial_coverages", [0.8, 0.9, 0.95])
        partial_coverages = resolve_partial_coverages(base_partial, labels_query if labels_query.numel() > 0 else labels_cand)
        reliability_bins = int(self.evaluation_cfg.get("reliability_bins", 10))

        domains: Dict[str, Dict[str, float]] = {}

        # Query-level selective metrics (primary)
        if labels_query.numel() > 0:
            scores_query = concat(q_scores)
            domains["score"] = compute_selective_metrics(
                labels_query, scores_query, partial_coverages=partial_coverages, reliability_bins=reliability_bins
            )
            entropy_conf = normalize_entropy_from_probs(scores_query)
            domains["entropy"] = compute_selective_metrics(
                labels_query, entropy_conf, partial_coverages=partial_coverages, reliability_bins=reliability_bins
            )
            if q_ale:
                ale_tensor = concat(q_ale)
                ale_conf = normalize_uncertainty_as_confidence(ale_tensor)
                domains["aleatoric"] = compute_selective_metrics(
                    labels_query, ale_conf, partial_coverages=partial_coverages, reliability_bins=reliability_bins
                )
            if q_epi:
                epi_tensor = concat(q_epi)
                epi_conf = normalize_uncertainty_as_confidence(epi_tensor)
                domains["epistemic"] = compute_selective_metrics(
                    labels_query, epi_conf, partial_coverages=partial_coverages, reliability_bins=reliability_bins
                )
            if q_combo:
                combo_conf = torch.clamp(1.0 - concat(q_combo) / (2.0 * log2), 0.0, 1.0)
                domains["aleatoric_epistemic"] = compute_selective_metrics(
                    labels_query, combo_conf, partial_coverages=partial_coverages, reliability_bins=reliability_bins
                )
            if q_alpha_beta:
                conc_conf = normalize_concentration(concat(q_alpha_beta))
                domains["concentration"] = compute_selective_metrics(
                    labels_query, conc_conf, partial_coverages=partial_coverages, reliability_bins=reliability_bins
                )

        # Candidate-level metrics kept for debugging/diagnostics
        if labels_cand.numel() > 0:
            scores_cand = concat(cand_scores)
            domains["score_candidates"] = compute_selective_metrics(
                labels_cand, scores_cand, partial_coverages=partial_coverages, reliability_bins=reliability_bins
            )

        return domains

    def _export_reliability(self, split: str, domain: str, metric_name: str, data: Dict[str, List[float]]) -> None:
        if not data or not self._is_global_zero():
            return
        log_dir = None
        if self.logger is not None and hasattr(self.logger, "log_dir"):
            log_dir = getattr(self.logger, "log_dir", None)
        if log_dir is None and self.trainer is not None and getattr(self.trainer, "default_root_dir", None):
            log_dir = self.trainer.default_root_dir
        if log_dir is None:
            return
        base = Path(log_dir) / "reliability"
        base.mkdir(parents=True, exist_ok=True)
        step = getattr(self.trainer, "global_step", 0) if self.trainer is not None else 0
        filename = f"{split}_{domain}_{metric_name}_step{step}.json"
        payload = {
            "split": split,
            "domain": domain,
            "metric": metric_name,
            "step": step,
            "data": data,
        }
        (base / filename).write_text(json.dumps(payload, indent=2))

    def _is_global_zero(self) -> bool:
        if self.trainer is None:
            return True
        strategy = getattr(self.trainer, "strategy", None)
        if strategy is None:
            return getattr(self.trainer, "global_rank", 0) == 0
        try:
            return strategy.global_rank == 0
        except AttributeError:
            return getattr(self.trainer, "global_rank", 0) == 0

    def _gather_all_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not dist.is_available() or not dist.is_initialized():
            return samples
        world = dist.get_world_size()
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, samples)
        combined: List[Dict[str, Any]] = []
        for part in gathered:
            if part:
                combined.extend(part)
        return combined

    def _write_metrics_to_db(
        self,
        split: str,
        ranking_stats: RankingStats,
        answer: Dict[str, float],
        uncertainty: Dict[str, float],
        selective: Dict[str, Dict[str, float]],
    ) -> None:
        if self.metrics_db_path is None:
            return
        db_path = Path(self.metrics_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        run_name = getattr(self, "run_name", "default")
        timestamp = datetime.utcnow().isoformat()
        try:
            conn = sqlite3.connect(db_path)
        except Exception as exc:
            logger.warning("Failed to open metrics database %s: %s", db_path, exc)
            return

        scalar_rows: List[Tuple[str, str, str, float, str]] = []
        ranking_rows: List[Tuple[str, str, str, int, float, str]] = []
        selective_rows: List[Tuple[str, str, str, str, Optional[float], float, str]] = []
        reliability_rows: List[Tuple[str, str, str, float, float, float, float, str]] = []

        metric_map = {
            "precision": ranking_stats.precision_at_k,
            "recall": ranking_stats.recall_at_k,
            "f1": ranking_stats.f1_at_k,
            "ndcg": ranking_stats.ndcg_at_k,
        }
        for metric_name, values in metric_map.items():
            for k, value in values.items():
                ranking_rows.append((run_name, split, metric_name, int(k), float(value), timestamp))

        for k, value in self._parse_answer_metrics(answer).items():
            ranking_rows.append((run_name, split, "answer_recall", int(k), float(value), timestamp))

        scalar_rows.append((run_name, split, "ranking/mrr", float(ranking_stats.mrr), timestamp))
        for metric_name, value in uncertainty.items():
            scalar_rows.append((run_name, split, f"uncertainty/{metric_name}", float(value), timestamp))

        for domain, metrics in selective.items():
            reliability_data = metrics.get("reliability")
            if isinstance(reliability_data, dict):
                centers = reliability_data.get("centers", [])
                pred = reliability_data.get("predicted", [])
                obs = reliability_data.get("observed", [])
                count = reliability_data.get("count", [])
                for idx in range(len(centers)):
                    center = float(centers[idx])
                    predicted = float(pred[idx]) if idx < len(pred) else center
                    observed = float(obs[idx]) if idx < len(obs) else center
                    cnt = float(count[idx]) if idx < len(count) else 0.0
                    reliability_rows.append((run_name, split, domain, center, predicted, observed, cnt, timestamp))
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    continue
                if "@" in metric_name:
                    base, cov = metric_name.split("@", 1)
                    try:
                        coverage = float(cov)
                    except ValueError:
                        coverage = None
                    selective_rows.append((run_name, split, domain, base, coverage, float(value), timestamp))
                else:
                    scalar_rows.append((run_name, split, f"selective/{domain}/{metric_name}", float(value), timestamp))

        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scalar_metrics (
                        run_name TEXT,
                        split TEXT,
                        metric TEXT,
                        value REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ranking_metrics (
                        run_name TEXT,
                        split TEXT,
                        metric TEXT,
                        k INTEGER,
                        value REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS selective_metrics (
                        run_name TEXT,
                        split TEXT,
                        domain TEXT,
                        metric TEXT,
                        coverage REAL,
                        value REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reliability_bins (
                        run_name TEXT,
                        split TEXT,
                        domain TEXT,
                        bin_center REAL,
                        predicted REAL,
                        observed REAL,
                        count REAL,
                        timestamp TEXT
                    )
                    """
                )
                conn.execute("DELETE FROM scalar_metrics WHERE run_name=? AND split=?", (run_name, split))
                conn.execute("DELETE FROM ranking_metrics WHERE run_name=? AND split=?", (run_name, split))
                conn.execute("DELETE FROM selective_metrics WHERE run_name=? AND split=?", (run_name, split))
                conn.execute("DELETE FROM reliability_bins WHERE run_name=? AND split=?", (run_name, split))
                if scalar_rows:
                    conn.executemany(
                        "INSERT INTO scalar_metrics VALUES (?,?,?,?,?)",
                        scalar_rows,
                    )
                if ranking_rows:
                    conn.executemany(
                        "INSERT INTO ranking_metrics VALUES (?,?,?,?,?,?)",
                        ranking_rows,
                    )
                if selective_rows:
                    conn.executemany(
                        "INSERT INTO selective_metrics VALUES (?,?,?,?,?,?,?)",
                        selective_rows,
                    )
                if reliability_rows:
                    conn.executemany(
                        "INSERT INTO reliability_bins VALUES (?,?,?,?,?,?,?,?)",
                        reliability_rows,
                    )
        except Exception as exc:
            logger.warning("Failed to write metrics to database %s: %s", db_path, exc)
        finally:
            conn.close()


    def _derive_run_identifier(self) -> str:
        if hasattr(self, "hparams"):
            model_cfg = self.hparams.get("model_cfg", {})
            model_type = model_cfg.get("model_type", "model")
            loss_cfg = self.hparams.get("loss_cfg", {})
            loss_type = loss_cfg.get("type", "loss")
            return f"{model_type}_{loss_type}"
        return "default"

    def _parse_answer_metrics(self, answer: Dict[str, float]) -> Dict[int, float]:
        parsed: Dict[int, float] = {}
        for name, value in answer.items():
            if not name.startswith("answer_recall@"):
                continue
            try:
                k = int(name.split("@", 1)[1])
            except (IndexError, ValueError):
                continue
            parsed[k] = float(value)
        return parsed

    def _log_ranking_visualizations(self, split: str, stats, answer: Dict[str, float]) -> None:
        return

    def _log_uncertainty_visualizations(
        self, split: str, uncertainty: Dict[str, float], selective: Dict[str, Dict[str, float]]
    ) -> None:
        return
