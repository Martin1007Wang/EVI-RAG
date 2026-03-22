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
    microbatch_size: int = 4096

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("transition_head.hidden_dim must be >= 1.")
        if self.num_layers < 1:
            raise ValueError("transition_head.num_layers must be >= 1.")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("transition_head.dropout must be in [0, 1).")
        if self.microbatch_size < 1:
            raise ValueError("transition_head.microbatch_size must be >= 1.")


@dataclass(frozen=True)
class PrefixControllerConfig:
    """Recurrent prefix tracker that compresses path history into a control state."""

    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("prefix_controller.dropout must be in [0, 1).")


@dataclass(frozen=True)
class CandidateShortlistConfig:
    """Approximate shortlist used to cap heavy transition scoring work."""

    enabled: bool = False
    topk: int = 16
    degree_threshold: int = 32
    heuristic_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError("candidate_shortlist.topk must be >= 1.")
        if self.degree_threshold < 1:
            raise ValueError("candidate_shortlist.degree_threshold must be >= 1.")
        if self.heuristic_weight < 0.0:
            raise ValueError("candidate_shortlist.heuristic_weight must be >= 0.")


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
    candidate_shortlist: CandidateShortlistConfig = field(
        default_factory=CandidateShortlistConfig
    )
