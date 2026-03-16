from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch.callbacks import Callback

from src.utils.metrics_io import to_serializable
from src.utils.output_sinks import StageMetricsSettings, append_stage_metrics

_WEIGHT_SUFFIX_PRIORITY = (
    "num_valid_graphs",
    "answer_eval_samples",
    "num_graphs",
    "num_samples",
)


class LocalMetricsWriter(Callback):
    """Write Lightning callback metrics to local JSONL files."""

    def __init__(
        self,
        *,
        output_dir: Optional[str | Path] = None,
        enabled: bool = True,
        train_window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._last_logged_step: Dict[str, int] = {}
        self._stage_sums: Dict[str, Dict[str, float]] = {}
        self._stage_weights: Dict[str, Dict[str, float]] = {}
        self.train_window_size = (
            int(train_window_size) if train_window_size is not None else 0
        )
        self._train_window: deque[Dict[str, float]] = deque()
        self._train_sums: Dict[str, float] = {}
        self._train_counts: Dict[str, int] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        _ = pl_module
        if not self.enabled:
            return
        self._ensure_output_dir(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        if not self.enabled:
            return
        if not self._should_log_step(trainer, stage="train"):
            self._accumulate_train(trainer)
            return
        payload = self._resolve_train_window_average()
        if payload:
            self._write_stage_payload(trainer, stage="train", payload=payload)
        self._accumulate_train(trainer)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx, dataloader_idx

    def on_validation_end(self, trainer, pl_module) -> None:
        del pl_module
        # Lightning already reduces on_epoch validation metrics; batch-end
        # callback_metrics can lag and replay stale values across val runs.
        self._clear_stage_accum(stage="val")
        self._write_stage(trainer, stage="val")

    def on_test_end(self, trainer, pl_module) -> None:
        del pl_module
        self._clear_stage_accum(stage="test")
        self._write_stage(trainer, stage="test")

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled:
            return
        payload = self._collect_model_predict_metrics(pl_module)
        if payload:
            self._write_stage_payload(trainer, stage="predict", payload=payload)
            return
        self._write_stage(trainer, stage="predict", include_all_metrics=True)

    def _should_log_step(self, trainer, *, stage: str) -> bool:
        step = int(getattr(trainer, "global_step", 0))
        last_step = self._last_logged_step.get(stage)
        if last_step is not None and last_step == step:
            return False
        log_every = int(getattr(trainer, "log_every_n_steps", 0) or 0)
        if log_every > 0 and stage == "train":
            if step % log_every != 0:
                return False
        self._last_logged_step[stage] = step
        return True

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

    def _write_stage(
        self, trainer, *, stage: str, include_all_metrics: bool = False
    ) -> bool:
        if not self.enabled or not getattr(trainer, "is_global_zero", True):
            return False
        metrics = self._collect_metrics(trainer)
        if not metrics:
            return False
        payload = self._extract_metrics(
            metrics, stage=stage, include_all=include_all_metrics
        )
        if not payload:
            return False
        self._write_stage_payload(trainer, stage=stage, payload=payload)
        return True

    def _write_stage_payload(
        self, trainer, *, stage: str, payload: Dict[str, Any]
    ) -> None:
        if not payload:
            return
        output_dir = self._ensure_output_dir(trainer)
        append_stage_metrics(
            metrics=payload,
            settings=StageMetricsSettings(
                output_dir=output_dir,
                stage=stage,
                step=int(getattr(trainer, "global_step", 0)),
            ),
        )

    def _accumulate_stage(self, trainer, *, stage: str) -> None:
        metrics = self._collect_metrics(trainer)
        if not metrics:
            return
        payload = self._extract_metrics(metrics, stage=stage, include_all=False)
        if not payload:
            return
        weight_map = self._build_weight_map(payload)
        sums = self._stage_sums.setdefault(stage, {})
        weights = self._stage_weights.setdefault(stage, {})
        for name, value in payload.items():
            scalar = self._to_float(value)
            if scalar is None:
                continue
            prefix = name.rsplit("/", 1)[0]
            if self._is_weight_metric_name(name):
                sums[name] = sums.get(name, 0.0) + scalar
                weights[name] = weights.get(name, 0.0) + 1.0
                continue
            weight = weight_map.get(prefix, 1.0)
            sums[name] = sums.get(name, 0.0) + scalar * weight
            weights[name] = weights.get(name, 0.0) + weight

    def _write_stage_from_accum(self, trainer, *, stage: str) -> bool:
        if not self.enabled or not getattr(trainer, "is_global_zero", True):
            return False
        sums = self._stage_sums.pop(stage, None)
        weights = self._stage_weights.pop(stage, None)
        if not sums or not weights:
            return False
        payload: Dict[str, Any] = {}
        for name, total in sums.items():
            weight = weights.get(name, 0.0)
            if weight > 0.0:
                payload[name] = total / weight
        self._write_stage_payload(trainer, stage=stage, payload=payload)
        return True

    def _clear_stage_accum(self, *, stage: str) -> None:
        self._stage_sums.pop(stage, None)
        self._stage_weights.pop(stage, None)

    def _accumulate_train(self, trainer) -> None:
        metrics = self._collect_metrics(trainer)
        if not metrics:
            return
        payload = self._extract_metrics(metrics, stage="train", include_all=False)
        if not payload:
            return
        window_entry: Dict[str, float] = {}
        for name, value in payload.items():
            scalar = self._to_float(value)
            if scalar is None:
                continue
            window_entry[name] = scalar
            self._train_sums[name] = self._train_sums.get(name, 0.0) + scalar
            self._train_counts[name] = self._train_counts.get(name, 0) + 1
        if not window_entry:
            return
        self._train_window.append(window_entry)
        if (
            self.train_window_size > 0
            and len(self._train_window) > self.train_window_size
        ):
            expired = self._train_window.popleft()
            for name, value in expired.items():
                self._train_sums[name] = self._train_sums.get(name, 0.0) - value
                self._train_counts[name] = self._train_counts.get(name, 0) - 1
                if self._train_counts[name] <= 0:
                    self._train_counts.pop(name, None)
                    self._train_sums.pop(name, None)

    def _resolve_train_window_average(self) -> Dict[str, Any]:
        if not self._train_counts:
            return {}
        averaged: Dict[str, Any] = {}
        for name, total in self._train_sums.items():
            count = self._train_counts.get(name, 0)
            if count > 0:
                averaged[name] = total / count
        return averaged

    @staticmethod
    def _extract_metrics(
        metrics: Dict[str, Any], *, stage: str, include_all: bool
    ) -> Dict[str, Any]:
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
    def _collect_metrics(trainer) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        callback_metrics = getattr(trainer, "callback_metrics", None)
        if callback_metrics:
            merged.update(callback_metrics)
        logged_metrics = getattr(trainer, "logged_metrics", None)
        if logged_metrics:
            for key, value in logged_metrics.items():
                if key not in merged:
                    merged[key] = value
        return merged

    @staticmethod
    def _collect_model_predict_metrics(pl_module) -> Dict[str, Any]:
        getter = getattr(pl_module, "get_predict_metrics", None)
        if not callable(getter):
            return {}
        metrics = getter()
        if not isinstance(metrics, dict):
            return {}
        return {
            str(name): LocalMetricsWriter._to_serializable(value)
            for name, value in metrics.items()
        }

    @staticmethod
    def _build_weight_map(payload: Dict[str, Any]) -> Dict[str, float]:
        resolved: Dict[str, tuple[int, float]] = {}
        for name, value in payload.items():
            for priority, suffix in enumerate(_WEIGHT_SUFFIX_PRIORITY):
                if not name.endswith(f"/{suffix}"):
                    continue
                scalar = LocalMetricsWriter._to_float(value)
                if scalar is None:
                    break
                prefix = name.rsplit("/", 1)[0]
                existing = resolved.get(prefix)
                if existing is None or priority < existing[0]:
                    resolved[prefix] = (priority, scalar)
                break
        return {prefix: scalar for prefix, (_, scalar) in resolved.items()}

    @staticmethod
    def _is_weight_metric_name(name: str) -> bool:
        return any(name.endswith(f"/{suffix}") for suffix in _WEIGHT_SUFFIX_PRIORITY)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        serializable = LocalMetricsWriter._to_serializable(value)
        if isinstance(serializable, (int, float)):
            return float(serializable)
        return None

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        return to_serializable(value)


__all__ = ["LocalMetricsWriter"]
