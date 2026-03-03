# src/models/configs/policy.py
from dataclasses import dataclass


@dataclass(frozen=True)
class BackboneConfig:
    """策略 Backbone 黄金基准配置"""

    embedding_dim: int = 1024
    hidden_dim: int = 512  # 图编码瓶颈维度：1024 -> 512 压缩后做 GNN 消息传递
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

    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.0
    relation_low_rank: int = 16
    relation_low_rank_edge_chunk_size: int = 8192


@dataclass(frozen=True)
class PriorityHeadConfig:
    """节点优先级打分头配置（用于答案排序辅助监督）"""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    """完整策略网络配置"""

    backbone: BackboneConfig = BackboneConfig()
    flow_head: FlowHeadConfig = FlowHeadConfig()
    priority_head: PriorityHeadConfig = PriorityHeadConfig()
    stop_bias_init: float = -1.5
    stop_delta_scale: float = 2.0
    stop_delta_temperature: float = 1.0
    doob_h_alpha: float = 1.0
    doob_h_node_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.stop_delta_scale <= 0.0:
            raise ValueError("stop_delta_scale must be > 0.")
        if self.stop_delta_temperature <= 0.0:
            raise ValueError("stop_delta_temperature must be > 0.")
        if self.doob_h_alpha < 0.0:
            raise ValueError("doob_h_alpha must be >= 0.")
        if self.doob_h_node_temperature <= 0.0:
            raise ValueError("doob_h_node_temperature must be > 0.")
