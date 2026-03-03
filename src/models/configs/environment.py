# src/models/configs/environment.py
from dataclasses import dataclass


@dataclass(frozen=True)
class StopConfig:
    reward_base: float = 1.0
    reward_epsilon: float = 1e-6
    reward_beta_init: float = 1.0
    reward_beta_max: float = 1.0
    reward_beta_anneal_steps: int = 0
    reward_beta_anneal_start_step: int = 0
    reward_beta_schedule: str = "linear"  # linear, exponential


@dataclass(frozen=True)
class EnvironmentConfig:
    """纯净的环境配置：只包含先验物理法则与停止奖励，不含任何神经网络参数"""

    stop: StopConfig = StopConfig()
    super_source_enabled: bool = False
