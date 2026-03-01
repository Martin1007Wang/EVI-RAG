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
    eta_min: float = 0.0
    warmup_steps: int = 0


@dataclass(frozen=True)
class ReplayBufferConfig:
    """高能轨迹回放配置"""

    enabled: bool = True
    alpha_init: float = 0.35
    alpha_final: float = 0.1
    alpha_anneal_epochs: int = 20
    max_paths_per_pair: int = 24
    max_paths_per_graph: int = 256
    max_shortest_paths_per_pair: int = 4
    max_dfs_paths_per_pair: int = 12
    max_depth: int = 10
    allow_cycles: bool = True
    max_node_visits: int = 2
    track_visited_mask: bool = True
    path_sampling_temperature: float = 1.0
    shortest_gap_weight: float = 1.0
    revisit_penalty_weight: float = 0.5


@dataclass(frozen=True)
class TrainingConfig:
    """训练流程配置"""

    replay_cfg: ReplayBufferConfig = ReplayBufferConfig()
