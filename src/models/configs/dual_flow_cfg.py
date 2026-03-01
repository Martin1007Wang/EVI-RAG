# src/models/configs/dual_flow_cfg.py
from dataclasses import dataclass
from .environment import EnvironmentConfig
from .policy import PolicyConfig
from .search import RolloutConfig, BeamSearchConfig
from .objective import SubTBConfig
from .training import TrainingConfig, OptimizerConfig, SchedulerConfig


@dataclass(frozen=True)
class DualFlowConfig:
    """
    GFlowNet 系统的顶层配置契约。
    禁止在这里写任何计算逻辑，这里只有数据结构。
    """

    env_cfg: EnvironmentConfig
    policy_cfg: PolicyConfig
    sampling_cfg: RolloutConfig
    eval_cfg: BeamSearchConfig
    subtb_cfg: SubTBConfig
    training_cfg: TrainingConfig
    optimizer_cfg: OptimizerConfig
    scheduler_cfg: SchedulerConfig
