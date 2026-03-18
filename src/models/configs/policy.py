from dataclasses import dataclass, field

from .backbone import BackboneConfig


@dataclass(frozen=True)
class StateScoreHeadConfig:
    """Question-conditioned state scoring head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class TransitionPolicyHeadConfig:
    """Question-conditioned transition-policy head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0


@dataclass(frozen=True)
class PolicyConfig:
    """Full answer-reachability policy configuration."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    state_score_head: StateScoreHeadConfig = field(default_factory=StateScoreHeadConfig)
    forward_policy_head: TransitionPolicyHeadConfig = field(
        default_factory=TransitionPolicyHeadConfig
    )
    backward_policy_head: TransitionPolicyHeadConfig = field(
        default_factory=TransitionPolicyHeadConfig
    )
