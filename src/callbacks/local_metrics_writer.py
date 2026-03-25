from __future__ import annotations

from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import Callback

from src.utils.output_sinks import StageMetricsSettings, append_stage_metrics


class LocalMetricsWriter(Callback):
    """Write exact train snapshots at the configured logging cadence."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._pending_train_step: int | None = None
        self._pending_train_batch_idx: int | None = None
        self._pending_train_payload: dict[str, Any] | None = None
        self._last_logged_train_step: int | None = None

    def on_fit_start(self, trainer, pl_module) -> None:
        del pl_module
        if not self.enabled:
            return
        self._pending_train_step = None
        self._pending_train_batch_idx = None
        self._pending_train_payload = None
        self._last_logged_train_step = None
        if self._is_writer_process(trainer):
            self._ensure_output_dir(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del outputs, batch
        if not self.enabled or not self._is_writer_process(trainer):
            return
        payload = self._pop_train_metrics(pl_module)
        if not payload:
            return
        current_step = int(getattr(trainer, "global_step", 0))
        if (
            self._pending_train_step is not None
            and current_step != self._pending_train_step
        ):
            self._flush_pending_train(trainer, force=False)
        self._pending_train_step = current_step
        self._pending_train_batch_idx = int(batch_idx)
        self._pending_train_payload = payload

    def on_train_end(self, trainer, pl_module) -> None:
        del pl_module
        if not self.enabled or not self._is_writer_process(trainer):
            return
        self._flush_pending_train(trainer, force=True)

    def _flush_pending_train(self, trainer, *, force: bool) -> None:
        payload = self._pending_train_payload
        step = self._pending_train_step
        batch_idx = self._pending_train_batch_idx
        if payload is None or step is None:
            return
        if step < 1:
            self._clear_pending_train()
            return
        if not force and not self._should_log_train_step(trainer, step=step):
            self._clear_pending_train()
            return
        if self._last_logged_train_step == step:
            self._clear_pending_train()
            return
        metadata = None
        if batch_idx is not None:
            metadata = {"batch_idx": int(batch_idx)}
        self._write_stage_payload(
            trainer,
            stage="train",
            payload=payload,
            record_kind="train_step",
            metadata=metadata,
            step_override=step,
        )
        self._last_logged_train_step = step
        self._clear_pending_train()

    def _clear_pending_train(self) -> None:
        self._pending_train_step = None
        self._pending_train_batch_idx = None
        self._pending_train_payload = None

    @staticmethod
    def _should_log_train_step(trainer, *, step: int) -> bool:
        log_every_n_steps = int(getattr(trainer, "log_every_n_steps", 0) or 0)
        if log_every_n_steps <= 0:
            return True
        return step % log_every_n_steps == 0

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

    def _write_stage(
        self, trainer, *, stage: str, include_all_metrics: bool = False
    ) -> bool:
        if not self.enabled or not self._is_writer_process(trainer):
            return False
        metrics = self._collect_metrics(trainer)
        if not metrics:
            return False
        payload = self._extract_metrics(
            metrics,
            stage=stage,
            include_all=include_all_metrics,
        )
        if not payload:
            return False
        self._write_stage_payload(trainer, stage=stage, payload=payload)
        return True

    def _write_stage_payload(
        self,
        trainer,
        *,
        stage: str,
        payload: dict[str, Any],
        file_name: str | None = None,
        record_kind: str | None = None,
        metadata: dict[str, Any] | None = None,
        step_override: int | None = None,
    ) -> None:
        if not payload or not self.enabled or not self._is_writer_process(trainer):
            return
        output_dir = self._ensure_output_dir(trainer)
        append_stage_metrics(
            metrics=payload,
            settings=StageMetricsSettings(
                output_dir=output_dir,
                stage=stage,
                step=(
                    int(getattr(trainer, "global_step", 0))
                    if step_override is None
                    else int(step_override)
                ),
                epoch=self._resolve_epoch(trainer),
                record_kind=record_kind,
                metadata=metadata,
                file_name=file_name,
            ),
        )

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
    def _pop_train_metrics(pl_module) -> dict[str, Any] | None:
        getter = getattr(pl_module, "pop_latest_train_metrics", None)
        if not callable(getter):
            return None
        payload = getter()
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise TypeError(
                "Expected pop_latest_train_metrics() to return dict[str, Any] or None."
            )
        return {str(name): value for name, value in payload.items()}


__all__ = ["LocalMetricsWriter"]
