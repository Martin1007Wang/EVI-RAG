# src/models/configs/environment.py
from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    pb_mode: str = "uniform_in_degree"
    logit_scale_init: float = 2.3


@dataclass(frozen=True)
class StopConfig:
    enabled: bool = True
    reward_base: float = 1.0
    reward_epsilon: float = 1e-6
    reward_beta_init: float = 0.6
    reward_beta_max: float = 1.0
    reward_beta_anneal_steps: int = 8000
    reward_beta_anneal_start_step: int = 0
    reward_beta_schedule: str = "linear"  # linear, exponential
    degree_penalty_alpha: float = 0.0
    degree_penalty_min_degree: float = 1.0


@dataclass(frozen=True)
class EnvironmentConfig:
    """纯净的环境配置：只包含先验物理法则与停止奖励，不含任何神经网络参数"""

    prior: PriorConfig = PriorConfig()
    stop: StopConfig = StopConfig()
    super_source_enabled: bool = False
