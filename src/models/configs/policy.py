from dataclasses import dataclass

from .backbone import BackboneConfig


@dataclass(frozen=True)
class StateScoreHeadConfig:
    """Question-conditioned state scoring head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    """Full answer-reachability policy configuration."""

    backbone: BackboneConfig = BackboneConfig()
    state_score_head: StateScoreHeadConfig = StateScoreHeadConfig()
