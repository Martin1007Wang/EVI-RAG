# src/models/configs/training.py
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConfig:
    """优化器配置"""

    type: str = "adamw"
    lr: float = 1e-4
    log_z_head_lr_multiplier: float = 5.0
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    no_decay_on_bias_and_norm: bool = True

    def __post_init__(self) -> None:
        if self.lr <= 0.0:
            raise ValueError("optimizer.lr must be > 0.")
        if self.log_z_head_lr_multiplier <= 0.0:
            raise ValueError("optimizer.log_z_head_lr_multiplier must be > 0.")
        if self.weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay must be >= 0.")
        if len(self.betas) != 2:
            raise ValueError("optimizer.betas must contain exactly two values.")
        beta1, beta2 = self.betas
        if not 0.0 <= float(beta1) < 1.0:
            raise ValueError("optimizer.betas[0] must be in [0, 1).")
        if not 0.0 <= float(beta2) < 1.0:
            raise ValueError("optimizer.betas[1] must be in [0, 1).")


@dataclass(frozen=True)
class SchedulerConfig:
    """学习率调度器配置"""

    type: str = "cosine"  # cosine, cosine_warm_restarts, onecycle
    interval: str = "step"  # step, epoch
    t_max: int | None = None
    t_mult: int = 1
    eta_min: float = 1e-6
    pct_start: float = 0.3
    anneal: str = "cos"
    lr: float | None = None
