from dataclasses import dataclass, field

from .backbone import BackboneConfig


@dataclass(frozen=True)
class StateScoreHeadConfig:
    """Question-conditioned state scoring head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    conditioning: str = "concat"

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("state_score_head.hidden_dim must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("state_score_head.num_layers must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("state_score_head.dropout must be in [0, 1).")
        if self.conditioning not in {"concat", "none"}:
            raise ValueError(
                "state_score_head.conditioning must be one of {'concat', 'none'}."
            )


@dataclass(frozen=True)
class TransitionPolicyHeadConfig:
    """Question-conditioned transition-policy head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0
    detach_input_features: bool = False

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("transition_head.hidden_dim must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("transition_head.num_layers must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("transition_head.dropout must be in [0, 1).")


@dataclass(frozen=True)
class PrefixControllerConfig:
    """Recurrent prefix tracker that compresses path history into a control state."""

    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("prefix_controller.dropout must be in [0, 1).")


@dataclass(frozen=True)
class PolicyConfig:
    """Full answer-reachability policy configuration."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    prefix_controller: PrefixControllerConfig = field(
        default_factory=PrefixControllerConfig
    )
    state_score_head: StateScoreHeadConfig = field(default_factory=StateScoreHeadConfig)
    forward_policy_head: TransitionPolicyHeadConfig = field(
        default_factory=TransitionPolicyHeadConfig
    )
