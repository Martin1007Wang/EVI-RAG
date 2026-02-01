from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lightning.pytorch.callbacks import Callback

_DEFAULT_STAGE_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "predict": "predict.jsonl",
}


class LocalMetricsWriter(Callback):
    """Write Lightning callback metrics to local JSONL files."""

    def __init__(
        self,
        *,
        output_dir: Optional[str | Path] = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def on_fit_start(self, trainer, pl_module) -> None:
        _ = pl_module
        if not self.enabled:
            return
        self._ensure_output_dir(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        _ = pl_module
        self._write_stage(trainer, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        _ = pl_module
        self._write_stage(trainer, stage="val")

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        _ = pl_module
        self._write_stage(trainer, stage="test")

    def on_predict_end(self, trainer, pl_module) -> None:
        _ = pl_module
        self._write_stage(trainer, stage="predict", include_all_metrics=True)

    def _resolve_output_dir(self, trainer) -> Path:
        if self.output_dir is not None:
            return self.output_dir
        log_dir = getattr(trainer, "log_dir", None)
        if log_dir:
            return Path(log_dir)
        default_root = getattr(trainer, "default_root_dir", None)
        if default_root:
            return Path(default_root)
        return Path(".")

    def _ensure_output_dir(self, trainer) -> Path:
        output_dir = self._resolve_output_dir(trainer)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _write_stage(self, trainer, *, stage: str, include_all_metrics: bool = False) -> None:
        if not self.enabled or not getattr(trainer, "is_global_zero", True):
            return
        metrics = getattr(trainer, "callback_metrics", None)
        if not metrics:
            return
        payload = self._extract_metrics(metrics, stage=stage, include_all=include_all_metrics)
        if not payload:
            return
        output_dir = self._ensure_output_dir(trainer)
        file_name = _DEFAULT_STAGE_FILES.get(stage, f"{stage}.jsonl")
        path = output_dir / file_name
        record = {
            "stage": stage,
            "epoch": int(getattr(trainer, "current_epoch", 0)) if stage != "predict" else None,
            "step": int(getattr(trainer, "global_step", 0)),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "metrics": payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _extract_metrics(metrics: Dict[str, Any], *, stage: str, include_all: bool) -> Dict[str, Any]:
        extracted: Dict[str, Any] = {}
        prefix = f"{stage}/"
        for name, value in metrics.items():
            if not isinstance(name, str):
                continue
            if not include_all and not name.startswith(prefix):
                continue
            extracted[name] = LocalMetricsWriter._to_serializable(value)
        return extracted

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.reshape(()).item())
            return value.tolist()
        if isinstance(value, (bool, int, float, str)) or value is None:
            return value
        if isinstance(value, dict):
            return {k: LocalMetricsWriter._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [LocalMetricsWriter._to_serializable(v) for v in value]
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


__all__ = ["LocalMetricsWriter"]
