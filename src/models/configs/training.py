# src/models/configs/training.py
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """优化器配置"""

    type: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True)
class SchedulerConfig:
    """学习率调度器配置"""

    type: str = "cosine"  # cosine, cosine_warm_restarts, onecycle
    interval: str = "step"  # step, epoch
    t_max: int = 10
    t_mult: int = 1
    eta_min: float = 0.0
    pct_start: float = 0.3
    anneal: str = "cos"
    lr: float | None = None
