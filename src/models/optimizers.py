# src/models/optimizers.py
"""
[系统实体] 优化器和调度器构建器
将恶心的优化器配置逻辑封装在独立文件中
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)

def build_optimizer_and_scheduler(
    model_parameters: Iterator[tuple[str, torch.nn.Parameter]],
    optimizer_cfg: Mapping[str, Any],
    scheduler_cfg: Mapping[str, Any],
    estimated_stepping_batches: int | None = None,
) -> dict[str, Any]:
    """
    构建优化器和调度器

    Args:
        model_parameters: 模型参数迭代器 (name, param)
        optimizer_cfg: 优化器配置
        scheduler_cfg: 调度器配置
        estimated_stepping_batches: 预计的训练步数

    Returns:
        dict: 包含 optimizer 和 scheduler 的配置字典
    """
    # 分离需要训练的参数
    trainable_params = [p for n, p in model_parameters if p.requires_grad]

    if not trainable_params:
        raise RuntimeError("No trainable parameters found in model.")

    # 构建优化器
    optimizer = _build_optimizer(trainable_params, optimizer_cfg)

    # 构建调度器
    scheduler = None
    if estimated_stepping_batches is not None and estimated_stepping_batches > 0:
        scheduler = _build_scheduler(optimizer, scheduler_cfg, estimated_stepping_batches)

    result = {"optimizer": optimizer}
    if scheduler is not None:
        result["lr_scheduler"] = {
            "scheduler": scheduler,
            "interval": scheduler_cfg.get("interval", "step"),
        }

    return result


def _build_optimizer(
    params: list[torch.nn.Parameter],
    cfg: Mapping[str, Any],
) -> Optimizer:
    """构建优化器"""
    opt_type = cfg.get("type", "adamw").lower()

    if opt_type == "adamw":
        optimizer = AdamW(
            params,
            lr=cfg.get("lr", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.01),
            betas=cfg.get("betas", (0.9, 0.999)),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    return optimizer


def _build_scheduler(
    optimizer: Optimizer,
    cfg: Mapping[str, Any],
    total_steps: int,
) -> Any:
    """构建学习率调度器"""
    scheduler_type = cfg.get("type", "cosine").lower()

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("t_max", total_steps),
            eta_min=cfg.get("eta_min", 0.0),
        )
    elif scheduler_type == "cosine_warm_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.get("t0", 10),
            T_mult=cfg.get("t_mult", 1),
            eta_min=cfg.get("eta_min", 0.0),
        )
    elif scheduler_type == "onecycle":
        cycle_steps = int(cfg.get("t_max", total_steps))
        if cycle_steps <= 0:
            raise ValueError(f"onecycle scheduler requires t_max > 0, got {cycle_steps}.")
        if cycle_steps < total_steps:
            raise ValueError(
                "onecycle scheduler would exhaust before training ends: "
                f"t_max={cycle_steps} estimated_steps={total_steps}. "
                "Set trainer.max_steps and scheduler t_max consistently."
            )
        scheduler = OneCycleLR(
            optimizer,
            max_lr=cfg.get("lr", 1e-4),
            total_steps=cycle_steps,
            pct_start=cfg.get("pct_start", 0.3),
            anneal_strategy=cfg.get("anneal", "cos"),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    return scheduler


__all__ = [
    "build_optimizer_and_scheduler",
]
