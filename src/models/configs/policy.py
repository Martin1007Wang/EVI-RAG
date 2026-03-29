from dataclasses import dataclass, field

from .backbone import BackboneConfig


PATH_PREFIX_STATE_MODE = "path_prefix"
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
class PrefixControllerConfig:
    """Recurrent prefix tracker that compresses path history into a control state."""

    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("prefix_controller.dropout must be in [0, 1).")


@dataclass(frozen=True)
class VisitedSetEncoderConfig:
    """Exact visited-set sketch encoder derived from symbolic prefix state."""

    enabled: bool = True
    sketch_dim: int = 64
    num_hashes: int = 2
    hidden_dim: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.sketch_dim < 4:
            raise ValueError("visited_set_encoder.sketch_dim must be >= 4.")
        if self.num_hashes < 1:
            raise ValueError("visited_set_encoder.num_hashes must be >= 1.")
        if self.hidden_dim < 1:
            raise ValueError("visited_set_encoder.hidden_dim must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("visited_set_encoder.dropout must be in [0, 1).")


@dataclass(frozen=True)
class PrefixMemoryConfig:
    """External prefix-addressable memory used as conditional context."""

    enabled: bool = False
    capacity: int = 8192
    min_entries: int = 128
    top_k: int = 8
    temperature: float = 0.25
    min_prefix_steps: int = 0
    store_successes: bool = True
    store_failures: bool = True

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("prefix_memory.capacity must be >= 1.")
        if self.min_entries < 1:
            raise ValueError("prefix_memory.min_entries must be >= 1.")
        if self.min_entries > self.capacity:
            raise ValueError("prefix_memory.min_entries must be <= capacity.")
        if self.top_k < 1:
            raise ValueError("prefix_memory.top_k must be >= 1.")
        if self.temperature <= 0.0:
            raise ValueError("prefix_memory.temperature must be > 0.")
        if self.min_prefix_steps < 0:
            raise ValueError("prefix_memory.min_prefix_steps must be >= 0.")
        if not self.store_successes and not self.store_failures:
            raise ValueError(
                "prefix_memory requires at least one of store_successes or store_failures to be True."
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

    state_mode: str = PATH_PREFIX_STATE_MODE
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    prefix_controller: PrefixControllerConfig = field(
        default_factory=PrefixControllerConfig
    )
    visited_set_encoder: VisitedSetEncoderConfig = field(
        default_factory=VisitedSetEncoderConfig
    )
    prefix_memory: PrefixMemoryConfig = field(default_factory=PrefixMemoryConfig)
    state_score_head: StateScoreHeadConfig = field(default_factory=StateScoreHeadConfig)
    transition_head: TransitionHeadConfig = field(default_factory=TransitionHeadConfig)
    subgraph_state_encoder: SubgraphStateEncoderConfig = field(
        default_factory=SubgraphStateEncoderConfig
    )
    subgraph_action_head: SubgraphActionHeadConfig = field(
        default_factory=SubgraphActionHeadConfig
    )

    def __post_init__(self) -> None:
        if self.state_mode not in {
            PATH_PREFIX_STATE_MODE,
            SUBGRAPH_STATE_MODE,
        }:
            raise ValueError(
                "policy.state_mode must be one of {'path_prefix', 'subgraph'}."
            )
