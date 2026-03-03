from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional, Sequence

import torch
from hydra.utils import instantiate
from lightning.pytorch.callbacks import Callback
from torchmetrics import Metric, MetricCollection

from src.metrics.dual_flow.rollout_metrics import DualFlowRolloutMetrics


class DualFlowEvalMetrics(Callback):
    """Aggregate DualFlow rollout metrics during predict()."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        k_values: Optional[Sequence[int]] = None,
        composite_score_cfg: Optional[Any] = None,
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self._metrics = self._build_metric_collection(
            metrics_cfg=metrics,
            k_values=k_values,
            composite_score_cfg=composite_score_cfg,
        )

    @staticmethod
    def _build_metric_collection(
        *,
        metrics_cfg: Optional[Mapping[str, Any]],
        k_values: Optional[Sequence[int]],
        composite_score_cfg: Optional[Any],
    ) -> MetricCollection:
        if metrics_cfg is None:
            metrics: dict[str, Metric] = {
                "rollout": DualFlowRolloutMetrics(
                    k_values=k_values,
                    composite_score_cfg=composite_score_cfg,
                )
            }
            return MetricCollection(metrics)

        if not isinstance(metrics_cfg, Mapping):
            raise TypeError("metrics must be a mapping of name -> metric config")

        metrics = {}
        for name, cfg in metrics_cfg.items():
            if isinstance(cfg, Metric):
                metrics[str(name)] = cfg
                continue
            if isinstance(cfg, Mapping) and "_target_" in cfg:
                metrics[str(name)] = instantiate(cfg)
                continue
            raise TypeError(
                "metrics entries must be torchmetrics.Metric instances or Hydra configs"
            )
        return MetricCollection(metrics)

    @staticmethod
    def _flatten_metrics(raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict) or not raw:
            return {}
        if len(raw) == 1:
            only_value = next(iter(raw.values()))
            if isinstance(only_value, dict):
                return only_value
        flat: dict[str, Any] = {}
        for name, value in raw.items():
            if isinstance(value, dict):
                for subkey, subval in value.items():
                    flat[f"{name}/{subkey}"] = subval
            else:
                flat[str(name)] = value
        return flat

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.enabled or outputs is None:
            return
        records = outputs if isinstance(outputs, list) else None
        if not records:
            return
        self._metrics.update(records)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        raw_metrics = self._metrics.compute()
        metrics = self._flatten_metrics(raw_metrics)
        if metrics:
            tensor_metrics = {
                k: v.detach() if torch.is_tensor(v) else torch.tensor(v)
                for k, v in metrics.items()
            }
            setattr(pl_module, "predict_metrics", tensor_metrics)
            trainer.callback_metrics.update(tensor_metrics)
        self._metrics.reset()


__all__ = ["DualFlowEvalMetrics"]
