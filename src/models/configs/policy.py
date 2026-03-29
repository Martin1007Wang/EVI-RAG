from dataclasses import dataclass, field

from .backbone import BackboneConfig


SUBGRAPH_STATE_MODE = "subgraph"


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
class TransitionHeadConfig:
    """Proposal-only edge bias head layered on top of strict successor-flow logits."""

    enabled: bool = False
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
class SubgraphStateEncoderConfig:
    """Graph-level encoder used by the subgraph-growth search state."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("subgraph_state_encoder.hidden_dim must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("subgraph_state_encoder.num_layers must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("subgraph_state_encoder.dropout must be in [0, 1).")


@dataclass(frozen=True)
class SubgraphActionHeadConfig:
    """Action scorer used by the subgraph-growth target policy head."""

    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("subgraph_action_head.hidden_dim must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("subgraph_action_head.num_layers must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("subgraph_action_head.dropout must be in [0, 1).")


@dataclass(frozen=True)
class PolicyConfig:
    """Full answer-reachability policy configuration."""

    state_mode: str = SUBGRAPH_STATE_MODE
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    state_score_head: StateScoreHeadConfig = field(default_factory=StateScoreHeadConfig)
    transition_head: TransitionHeadConfig = field(default_factory=TransitionHeadConfig)
    subgraph_state_encoder: SubgraphStateEncoderConfig = field(
        default_factory=SubgraphStateEncoderConfig
    )
    subgraph_action_head: SubgraphActionHeadConfig = field(
        default_factory=SubgraphActionHeadConfig
    )

    def __post_init__(self) -> None:
        if self.state_mode != SUBGRAPH_STATE_MODE:
            raise ValueError("policy.state_mode must be 'subgraph'.")
