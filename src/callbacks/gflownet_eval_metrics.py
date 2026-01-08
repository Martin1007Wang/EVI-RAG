from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from lightning.pytorch.callbacks import Callback

from src.metrics.gflownet import GFlowNetEvalAccumulator


class GFlowNetEvalMetrics(Callback):
    """Aggregate GFlowNet rollout metrics during predict()."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        k_values: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self._accumulator = GFlowNetEvalAccumulator(k_values=k_values)

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
        self._accumulator.update_from_records(records)

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        metrics = self._accumulator.finalize()
        if metrics:
            setattr(pl_module, "predict_metrics", metrics)
            trainer.callback_metrics.update({k: torch.tensor(v) for k, v in metrics.items()})


__all__ = ["GFlowNetEvalMetrics"]
