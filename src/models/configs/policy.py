# src/models/configs/policy.py
from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneConfig:
    """策略 Backbone 黄金基准配置"""

    embedding_dim: int = 1024
    hidden_dim: int = 256  # 强化学习的决策空间，256 足矣，避免维度灾难
    gnn_layers: int = 2  # 2跳足够，避免图节点的过度平滑 (Oversmoothing)
    gnn_dropout: float = 0.1  # 必须是 float，轻微正则化防止 Hub Nodes 过拟合

    use_adapter: bool = True
    adapter_dim: int = 128  # [系统级修正] 强制低秩瓶颈 (通常为 embedding_dim 的 1/8 或 1/4)
    adapter_dropout: float = 0.1

    use_positional_encoding: bool = True
    use_film: bool = False


@dataclass(frozen=True)
class FlowHeadConfig:
    """状态流量预测头配置"""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    qcbia_alpha_init: float = 1.0


@dataclass(frozen=True)
class ActionHeadConfig:
    """动作打分头配置"""

    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    """完整策略网络配置"""

    backbone: BackboneConfig = BackboneConfig()
    flow_head: FlowHeadConfig = FlowHeadConfig()
    action_head: ActionHeadConfig = ActionHeadConfig()
    flow_lookahead_mode: str = "mlp_action"
    stop_bias_init: float = -10.0
