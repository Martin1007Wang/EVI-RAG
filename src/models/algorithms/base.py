from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

from src.models.configs.training import OptimizerConfig, SchedulerConfig

BatchPayload = tuple[dict[str, Any], dict[str, Any]]


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return dict(cfg)
    raise TypeError(f"Expected dataclass or dict config, got {type(cfg)!r}.")


def build_optimizer_and_scheduler(
    *,
    model_parameters: list[tuple[str, torch.nn.Parameter]] | Any,
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    estimated_stepping_batches: int | None,
) -> dict[str, Any]:
    trainable_params = [p for _, p in model_parameters if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found in model.")

    opt_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if opt_type != "adamw":
        raise ValueError(f"Unsupported optimizer type: {opt_type}")
    optimizer = AdamW(
        trainable_params,
        lr=float(optimizer_cfg.get("lr", 1.0e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.999))),
    )

    scheduler = None
    if estimated_stepping_batches is not None and int(estimated_stepping_batches) > 0:
        scheduler_type = str(scheduler_cfg.get("type", "cosine")).lower()
        t_max = int(scheduler_cfg.get("t_max", int(estimated_stepping_batches)))
        eta_min = float(scheduler_cfg.get("eta_min", 0.0))

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        elif scheduler_type == "cosine_warm_restarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=t_max,
                T_mult=int(scheduler_cfg.get("t_mult", 1)),
                eta_min=eta_min,
            )
        elif scheduler_type == "onecycle":
            cycle_steps = int(
                scheduler_cfg.get("t_max", int(estimated_stepping_batches))
            )
            if cycle_steps <= 0:
                raise ValueError(
                    f"onecycle scheduler requires t_max > 0, got {cycle_steps}."
                )
            if cycle_steps < int(estimated_stepping_batches):
                raise ValueError(
                    "onecycle scheduler would exhaust before training ends: "
                    f"t_max={cycle_steps} estimated_steps={int(estimated_stepping_batches)}. "
                    "Set trainer.max_steps and scheduler t_max consistently."
                )
            scheduler = OneCycleLR(
                optimizer,
                max_lr=float(scheduler_cfg.get("lr", optimizer_cfg.get("lr", 1.0e-4))),
                total_steps=cycle_steps,
                pct_start=float(scheduler_cfg.get("pct_start", 0.3)),
                anneal_strategy=str(scheduler_cfg.get("anneal", "cos")),
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    result: dict[str, Any] = {"optimizer": optimizer}
    if scheduler is not None:
        result["lr_scheduler"] = {
            "scheduler": scheduler,
            "interval": scheduler_cfg.get("interval", "step"),
        }
    return result


class AlgorithmModule(LightningModule):
    """Algorithm-bound Lightning module with standardized optimizer handling."""

    def __init__(
        self,
        *,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: SchedulerConfig,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

    def configure_optimizers(self) -> dict[str, Any]:
        return build_optimizer_and_scheduler(
            model_parameters=self.named_parameters(),
            optimizer_cfg=_cfg_to_dict(self.optimizer_cfg),
            scheduler_cfg=_cfg_to_dict(self.scheduler_cfg),
            estimated_stepping_batches=(
                int(self.trainer.estimated_stepping_batches)
                if self.trainer is not None
                else None
            ),
        )

    @staticmethod
    def _unpack_batch(batch: Any) -> BatchPayload:
        if isinstance(batch, dict) and "inputs" in batch and "metadata" in batch:
            prepared = batch.get("inputs")
            metadata = batch.get("metadata")
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            prepared, metadata = batch
        else:
            raise RuntimeError(
                "AlgorithmModule expects batches prepared by GRetrievalDataModule hooks "
                "(src.datasets.g_retrieval_datamodule, on_after_batch_transfer)."
            )
        if not isinstance(prepared, dict) or not isinstance(metadata, dict):
            raise TypeError("Prepared batch must be a dict with metadata dict.")
        return prepared, metadata


__all__ = ["AlgorithmModule", "build_optimizer_and_scheduler"]
