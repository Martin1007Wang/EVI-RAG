# src/models/configs/objective.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SubTBConfig:
    """Sub-Trajectory Balance Loss 的严格数学配置"""

    lambda_weight: float = 0.9  # 折扣因子 \lambda
    normalize: bool = False  # 是否对轨迹长度进行归一化
    detach_end_flow: bool = True  # 是否切断末端流的梯度
    miss_length_penalty: float = 0.0
    ranking_weight: float = 0.0
    ranking_temperature: float = 1.0
    backward_weight: float = 0.0

    def __post_init__(self):
        # 审计官指令：在初始化时立即执行严格的数学边界检查
        if not (0.0 <= self.lambda_weight <= 1.0):
            raise ValueError(f"SubTB lambda must be in [0, 1], got {self.lambda_weight}")
        if self.miss_length_penalty < 0.0:
            raise ValueError("miss_length_penalty cannot be negative.")
        if self.ranking_weight < 0.0:
            raise ValueError("ranking_weight cannot be negative.")
        if self.ranking_temperature <= 0.0:
            raise ValueError("ranking_temperature must be > 0.")
        if self.backward_weight < 0.0:
            raise ValueError("backward_weight cannot be negative.")
