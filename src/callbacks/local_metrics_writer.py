from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

from src.utils.metrics_io import to_serializable
from src.utils.output_sinks import StageMetricsSettings, append_stage_metrics


class LocalMetricsWriter(Callback):
    """Write Lightning callback metrics to local JSONL files."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        enabled: bool = True,
        train_window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._last_logged_step: dict[str, int] = {}
        self.train_window_size = (
            int(train_window_size) if train_window_size is not None else 0
        )
        self._train_window: deque[dict[str, float]] = deque()
        self._train_sums: dict[str, float] = {}
        self._train_counts: dict[str, int] = {}

    def on_fit_start(self, trainer, pl_module) -> None:
        _ = pl_module
        if not self.enabled:
            return
        self._reset_state()
        if self._is_writer_process(trainer):
            self._ensure_output_dir(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        if not self.enabled:
            return
        self._accumulate_train(trainer)
        if not self._should_log_step(trainer, stage="train"):
            return
        payload = self._resolve_train_window_average()
        if payload:
            self._write_stage_payload(trainer, stage="train", payload=payload)

    def on_train_end(self, trainer, pl_module) -> None:
        del pl_module
        if not self.enabled:
            return
        payload = self._resolve_train_window_average()
        if not payload:
            return
        final_step = int(getattr(trainer, "global_step", 0))
        if self._last_logged_step.get("train") == final_step:
            return
        self._last_logged_step["train"] = final_step
        self._write_stage_payload(trainer, stage="train", payload=payload)

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
        self._write_stage(trainer, stage="val")

    def on_test_end(self, trainer, pl_module) -> None:
        del pl_module
        self._write_stage(trainer, stage="test")

    def on_predict_end(self, trainer, pl_module) -> None:
        if not self.enabled or not self._is_writer_process(trainer):
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

    @staticmethod
    def _is_writer_process(trainer) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    @staticmethod
    def _resolve_epoch(trainer) -> int | None:
        epoch = getattr(trainer, "current_epoch", None)
        return int(epoch) if epoch is not None else None

    def _reset_state(self) -> None:
        self._last_logged_step.clear()
        self._train_window.clear()
        self._train_sums.clear()
        self._train_counts.clear()

    def _write_stage(
        self, trainer, *, stage: str, include_all_metrics: bool = False
    ) -> bool:
        if not self.enabled or not self._is_writer_process(trainer):
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
        self, trainer, *, stage: str, payload: dict[str, Any]
    ) -> None:
        if not payload or not self.enabled or not self._is_writer_process(trainer):
            return
        output_dir = self._ensure_output_dir(trainer)
        append_stage_metrics(
            metrics=payload,
            settings=StageMetricsSettings(
                output_dir=output_dir,
                stage=stage,
                step=int(getattr(trainer, "global_step", 0)),
                epoch=self._resolve_epoch(trainer),
            ),
        )

    def _accumulate_train(self, trainer) -> None:
        metrics = self._collect_metrics(trainer)
        if not metrics:
            return
        payload = self._extract_metrics(metrics, stage="train", include_all=False)
        if not payload:
            return
        window_entry: dict[str, float] = {}
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

    def _resolve_train_window_average(self) -> dict[str, Any]:
        if not self._train_counts:
            return {}
        averaged: dict[str, Any] = {}
        for name, total in self._train_sums.items():
            count = self._train_counts.get(name, 0)
            if count > 0:
                averaged[name] = total / count
        return averaged

    @staticmethod
    def _extract_metrics(
        metrics: dict[str, Any], *, stage: str, include_all: bool
    ) -> dict[str, Any]:
        extracted: dict[str, Any] = {}
        prefix = f"{stage}/"
        for name, value in metrics.items():
            if not isinstance(name, str):
                continue
            if not include_all and not name.startswith(prefix):
                continue
            extracted[name] = value
        return extracted

    @staticmethod
    def _collect_metrics(trainer) -> dict[str, Any]:
        merged: dict[str, Any] = {}
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
    def _collect_model_predict_metrics(pl_module) -> dict[str, Any]:
        getter = getattr(pl_module, "get_predict_metrics", None)
        if not callable(getter):
            return {}
        metrics = getter()
        if not isinstance(metrics, dict):
            return {}
        return {str(name): value for name, value in metrics.items()}

    @staticmethod
    def _to_float(value: Any) -> float | None:
        serializable = LocalMetricsWriter._to_serializable(value)
        if isinstance(serializable, (int, float)):
            return float(serializable)
        return None

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        return to_serializable(value)


__all__ = ["LocalMetricsWriter"]
