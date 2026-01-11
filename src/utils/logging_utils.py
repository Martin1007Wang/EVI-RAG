from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DEFAULT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_LOGGING_INITIALIZED = False

if TYPE_CHECKING:  # pragma: no cover
    from lightning import LightningModule


def init_logging(*, level: int = _DEFAULT_LOG_LEVEL, log_path: Optional[Path] = None) -> None:
    global _LOGGING_INITIALIZED
    root = logging.getLogger()
    if not _LOGGING_INITIALIZED:
        root.setLevel(level)
        formatter = logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_LOG_DATEFMT)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)
        _LOGGING_INITIALIZED = True
    if log_path is not None:
        resolved = Path(log_path).expanduser().resolve()
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler):
                if Path(handler.baseFilename).resolve() == resolved:
                    return
        file_handler = logging.FileHandler(resolved)
        file_handler.setFormatter(logging.Formatter(_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_LOG_DATEFMT))
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    if not _LOGGING_INITIALIZED:
        init_logging()
    return logging.getLogger(name)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return "[" + ",".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    if value is None:
        return "null"
    return str(value)


def log_event(logger: logging.Logger, event: str, *, level: int = _DEFAULT_LOG_LEVEL, **fields: Any) -> None:
    if not fields:
        logger.log(level, event)
        return
    parts = [f"{key}={_format_value(fields[key])}" for key in sorted(fields)]
    logger.log(level, "%s %s", event, " ".join(parts))


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, *args, rank: Optional[int] = None, **kwargs) -> None:
        if self.isEnabledFor(level):
            from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only

            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)


log = RankedLogger(__name__, rank_zero_only=True)


def _safe_numel(param: Any) -> int:
    try:
        from torch.nn.parameter import UninitializedParameter
    except Exception:  # pragma: no cover
        UninitializedParameter = ()
    if isinstance(param, UninitializedParameter):
        return 0
    try:
        return int(param.numel())
    except Exception:
        return 0


def _collect_model_param_counts(model: Any) -> Dict[str, int]:
    params = list(model.parameters())
    return {
        "model/params/total": sum(_safe_numel(p) for p in params),
        "model/params/trainable": sum(_safe_numel(p) for p in params if p.requires_grad),
        "model/params/non_trainable": sum(_safe_numel(p) for p in params if not p.requires_grad),
    }


def _resolve_cfg_container(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, dict):
        return cfg
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("OmegaConf is required for log_hyperparameters.") from exc
    return OmegaConf.to_container(cfg)


def _collect_cfg_sections(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": cfg.get("model"),
        "data": cfg.get("data"),
        "trainer": cfg.get("trainer"),
        "callbacks": cfg.get("callbacks"),
        "extras": cfg.get("extras"),
        "task_name": cfg.get("task_name"),
        "tags": cfg.get("tags"),
        "ckpt_path": cfg.get("ckpt_path"),
        "seed": cfg.get("seed"),
    }


def _log_hparams(trainer: Any, hparams: Dict[str, Any]) -> None:
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    from lightning_utilities.core.rank_zero import rank_zero_only

    @rank_zero_only
    def _log() -> None:
        cfg = _resolve_cfg_container(object_dict["cfg"])
        model = object_dict["model"]
        trainer = object_dict["trainer"]
        if not trainer.logger:
            log.warning("Logger not found! Skipping hyperparameter logging...")
            return
        hparams = _collect_cfg_sections(cfg)
        hparams.update(_collect_model_param_counts(model))
        _log_hparams(trainer, hparams)

    _log()


def log_metric(
    module: "LightningModule",
    name: str,
    value: Any,
    *,
    batch_size: int,
    on_step: bool = False,
    on_epoch: bool = True,
    prog_bar: bool = False,
    sync_dist: Optional[bool] = None,
    metric_attribute: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Centralized wrapper around LightningModule.log with strict batch sizing."""

    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() == 1:
            value = value.reshape(())

    log_kwargs: Dict[str, Any] = {
        "on_step": on_step,
        "on_epoch": on_epoch,
        "prog_bar": prog_bar,
        "batch_size": int(batch_size),
    }
    if sync_dist is not None:
        log_kwargs["sync_dist"] = sync_dist
    if metric_attribute is not None:
        log_kwargs["metric_attribute"] = metric_attribute
    log_kwargs.update(kwargs)
    module.log(name, value, **log_kwargs)


__all__ = [
    "init_logging",
    "get_logger",
    "log_event",
    "RankedLogger",
    "log_hyperparameters",
    "log_metric",
]
