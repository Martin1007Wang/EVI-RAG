from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
from torch.nn.parameter import UninitializedParameter

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    params = list(model.parameters())

    def _safe_numel(param):
        if isinstance(param, UninitializedParameter):
            return 0
        try:
            return param.numel()
        except Exception:
            return 0

    hparams["model/params/total"] = sum(_safe_numel(p) for p in params)
    hparams["model/params/trainable"] = sum(
        _safe_numel(p) for p in params if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        _safe_numel(p) for p in params if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return int(value.item())
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_batch_size(batch: Any) -> int:
    """Best-effort batch size inference covering PyG/Tensor/Dict batches."""

    if batch is None:
        raise ValueError("Cannot infer batch size from None batch")

    attr_candidates = ("num_graphs", "batch_size", "size", "length")
    for attr in attr_candidates:
        if hasattr(batch, attr):
            size = _to_int(getattr(batch, attr))
            if size and size > 0:
                return size

    if isinstance(batch, Mapping):
        for key in ("batch_size", "size", "length"):
            if key in batch:
                size = _to_int(batch[key])
                if size and size > 0:
                    return size
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.size(0))

    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            raise ValueError("Scalar tensors do not encode batch dimension")
        return int(batch.size(0))

    if isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)) and batch:
        first = batch[0]
        if isinstance(first, torch.Tensor) and first.ndim > 0:
            return int(first.size(0))
        candidate = _to_int(first)
        if candidate and candidate > 0:
            return candidate

    if hasattr(batch, "__len__") and not isinstance(batch, (str, bytes)):
        size = len(batch)  # type: ignore[arg-type]
        if size > 0:
            return int(size)

    raise ValueError(f"Unable to infer batch size from object of type {type(batch)!r}")


def log_metric(
    module: LightningModule,
    name: str,
    value: Any,
    *,
    batch_size: int,
    on_step: bool = False,
    on_epoch: bool = True,
    prog_bar: bool = False,
    sync_dist: Optional[bool] = None,
    **kwargs: Any,
) -> None:
    """Centralized wrapper around LightningModule.log with strict batch sizing."""

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
    log_kwargs.update(kwargs)
    module.log(name, value, **log_kwargs)
