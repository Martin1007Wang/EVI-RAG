from dataclasses import dataclass

from .backbone import BackboneConfig


@dataclass(frozen=True)
class StateScoreHeadConfig:
    """Question-conditioned state scoring head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class StartHeadConfig:
    """Question-conditioned start scoring head."""

    hidden_dim: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class GraphLogZHeadConfig:
    """Graph-level log-Z head used by the mainline GFlowNet."""

    hidden_dim: int = 512
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    """Full answer-reachability policy configuration."""

    backbone: BackboneConfig = BackboneConfig()
    state_score_head: StateScoreHeadConfig = StateScoreHeadConfig()
    start_head: StartHeadConfig = StartHeadConfig()
    graph_log_z_head: GraphLogZHeadConfig = GraphLogZHeadConfig()
