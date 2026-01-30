from __future__ import annotations

import logging
from typing import Any, Optional

import torch

from src.utils import setup_optimizer
from src.utils.logging_utils import get_logger, log_event

from .constants import (
    _DEFAULT_ONECYCLE_ANNEAL,
    _DEFAULT_ONECYCLE_BASE_MOMENTUM,
    _DEFAULT_ONECYCLE_CYCLE_MOMENTUM,
    _DEFAULT_ONECYCLE_DIV_FACTOR,
    _DEFAULT_ONECYCLE_FINAL_DIV_FACTOR,
    _DEFAULT_ONECYCLE_MAX_MOMENTUM,
    _DEFAULT_ONECYCLE_PCT_START,
    _DEFAULT_ONECYCLE_THREE_PHASE,
    _DEFAULT_SCHED_ETA_MIN,
    _DEFAULT_SCHED_T0,
    _DEFAULT_SCHED_T_MAX,
    _DEFAULT_SCHED_T_MULT,
    _SCHED_INTERVAL_EPOCH,
    _SCHED_INTERVAL_STEP,
    _SCHED_INTERVALS,
    _SCHED_TYPE_COSINE,
    _SCHED_TYPE_COSINE_WARM_RESTARTS,
    _SCHED_TYPE_ONECYCLE,
    _ZERO,
)

logger = get_logger(__name__)


class DualFlowSchedulerMixin:
    def configure_optimizers(self):
        optimizer = setup_optimizer(self, self.optimizer_cfg)
        scheduler = self._build_scheduler(optimizer)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[dict[str, Any]]:
        sched_type = str(self.scheduler_cfg.get("type", "") or "").strip().lower()
        if not sched_type:
            return None
        interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
        if interval not in _SCHED_INTERVALS:
            raise ValueError(f"scheduler_cfg.interval must be one of {sorted(_SCHED_INTERVALS)}, got {interval!r}.")
        if sched_type == _SCHED_TYPE_COSINE:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(self.scheduler_cfg.get("t_max", _DEFAULT_SCHED_T_MAX)),
                eta_min=float(self.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
        elif sched_type in {"cosine_restart", "cosine_warm_restarts", "cosine_restarts", _SCHED_TYPE_COSINE_WARM_RESTARTS}:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(self.scheduler_cfg.get("t_0", _DEFAULT_SCHED_T0)),
                T_mult=int(self.scheduler_cfg.get("t_mult", _DEFAULT_SCHED_T_MULT)),
                eta_min=float(self.scheduler_cfg.get("eta_min", _DEFAULT_SCHED_ETA_MIN)),
            )
        elif sched_type in {"onecycle", "one_cycle", "onecyclelr", _SCHED_TYPE_ONECYCLE}:
            if interval != _SCHED_INTERVAL_STEP:
                raise ValueError("OneCycleLR requires scheduler_cfg.interval='step'.")
            max_lr = self.scheduler_cfg.get("max_lr", None)
            if max_lr is None:
                raise ValueError("scheduler_cfg.max_lr must be set for OneCycleLR.")
            total_steps = self._resolve_onecycle_total_steps()
            pct_start = float(self.scheduler_cfg.get("pct_start", _DEFAULT_ONECYCLE_PCT_START))
            anneal_strategy = str(
                self.scheduler_cfg.get("anneal_strategy", _DEFAULT_ONECYCLE_ANNEAL) or _DEFAULT_ONECYCLE_ANNEAL
            ).strip().lower()
            cycle_momentum = bool(self.scheduler_cfg.get("cycle_momentum", _DEFAULT_ONECYCLE_CYCLE_MOMENTUM))
            base_momentum = float(self.scheduler_cfg.get("base_momentum", _DEFAULT_ONECYCLE_BASE_MOMENTUM))
            max_momentum = float(self.scheduler_cfg.get("max_momentum", _DEFAULT_ONECYCLE_MAX_MOMENTUM))
            div_factor = float(self.scheduler_cfg.get("div_factor", _DEFAULT_ONECYCLE_DIV_FACTOR))
            final_div_factor = float(self.scheduler_cfg.get("final_div_factor", _DEFAULT_ONECYCLE_FINAL_DIV_FACTOR))
            three_phase = bool(self.scheduler_cfg.get("three_phase", _DEFAULT_ONECYCLE_THREE_PHASE))
            last_epoch = int(self.scheduler_cfg.get("last_epoch", -1))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy=anneal_strategy,
                cycle_momentum=cycle_momentum,
                base_momentum=base_momentum,
                max_momentum=max_momentum,
                div_factor=div_factor,
                final_div_factor=final_div_factor,
                three_phase=three_phase,
                last_epoch=last_epoch,
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {sched_type}")
        return {"scheduler": scheduler, "interval": interval}

    def _resolve_onecycle_total_steps(self) -> int:
        total_steps = self.scheduler_cfg.get("total_steps", None)
        if total_steps is not None:
            total_steps = int(total_steps)
            if total_steps <= _ZERO:
                raise ValueError("scheduler_cfg.total_steps must be > 0.")
            return total_steps
        epochs = self.scheduler_cfg.get("epochs", None)
        steps_per_epoch = self.scheduler_cfg.get("steps_per_epoch", None)
        if epochs is not None or steps_per_epoch is not None:
            if epochs is None or steps_per_epoch is None:
                raise ValueError("scheduler_cfg.epochs and scheduler_cfg.steps_per_epoch must be set together.")
            epochs = int(epochs)
            steps_per_epoch = int(steps_per_epoch)
            if epochs <= _ZERO or steps_per_epoch <= _ZERO:
                raise ValueError("scheduler_cfg.epochs and steps_per_epoch must be > 0.")
            return epochs * steps_per_epoch
        trainer = getattr(self, "trainer", None)
        estimated = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        if estimated is None:
            raise ValueError("OneCycleLR requires total_steps or epochs+steps_per_epoch (trainer not initialized).")
        total_steps = int(estimated)
        if total_steps <= _ZERO:
            raise ValueError("trainer.estimated_stepping_batches must be > 0 for OneCycleLR.")
        return total_steps

    def _step_scheduler(self) -> None:
        sched = self.lr_schedulers()
        if sched is None:
            return
        schedulers = sched if isinstance(sched, list) else [sched]
        for scheduler in schedulers:
            self.lr_scheduler_step(scheduler, None)

    def on_train_epoch_end(self) -> None:
        interval = str(self.scheduler_cfg.get("interval", _SCHED_INTERVAL_EPOCH) or _SCHED_INTERVAL_EPOCH).strip().lower()
        if interval == _SCHED_INTERVAL_EPOCH:
            self._step_scheduler()

    def on_train_epoch_start(self) -> None:
        return

    def on_fit_start(self) -> None:
        self._check_onecycle_total_steps()

    def _check_onecycle_total_steps(self) -> None:
        if self._onecycle_checked:
            return
        self._onecycle_checked = True
        sched_type = str(self.scheduler_cfg.get("type", "") or "").strip().lower()
        if sched_type not in {"onecycle", "one_cycle", "onecyclelr", _SCHED_TYPE_ONECYCLE}:
            return
        trainer = getattr(self, "trainer", None)
        estimated = getattr(trainer, "estimated_stepping_batches", None) if trainer is not None else None
        configured = None
        source = None
        if "total_steps" in self.scheduler_cfg and self.scheduler_cfg.get("total_steps") is not None:
            configured = int(self.scheduler_cfg.get("total_steps"))
            source = "total_steps"
        elif self.scheduler_cfg.get("epochs") is not None or self.scheduler_cfg.get("steps_per_epoch") is not None:
            epochs = self.scheduler_cfg.get("epochs")
            steps_per_epoch = self.scheduler_cfg.get("steps_per_epoch")
            if epochs is not None and steps_per_epoch is not None:
                configured = int(epochs) * int(steps_per_epoch)
                source = "epochs*steps_per_epoch"
        if estimated is None:
            log_event(logger, "onecycle_total_steps_check", configured=configured, source=source, estimated=None)
            return
        estimated = int(estimated)
        if configured is None:
            log_event(logger, "onecycle_total_steps_check", configured=None, source=None, estimated=estimated)
            return
        diff = abs(int(configured) - estimated)
        ratio = float(diff) / float(estimated) if estimated > _ZERO else 0.0
        log_event(
            logger,
            "onecycle_total_steps_check",
            level=logging.WARNING if ratio >= 0.05 else logging.INFO,
            configured=int(configured),
            source=source,
            estimated=estimated,
            diff=diff,
            ratio=ratio,
        )


__all__ = ["DualFlowSchedulerMixin"]
