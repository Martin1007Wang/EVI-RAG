from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HorizonConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("horizon.max_steps must be >= 1.")


@dataclass(frozen=True)
class AnswerReachabilityInferenceConfig:
    """Inference settings for answer-reachability style evaluation.

    The public Hydra field names remain `eval_profile` / `eval_view` for backward
    compatibility. Internally, prefer the clearer aliases `metrics_profile` and
    `task_view` when consuming this config.
    """

    eval_profile: str = "full"
    eval_view: str = "answer_reachability"
    answer_mass_threshold: float = 0.9
    support_mass_threshold: float = 0.9
    support_path_overlap_penalty: float = 0.25
    window_top_ks: tuple[int, ...] = (1, 10, 25, 50, 100)
    answer_top_ks: tuple[int, ...] = (1, 5, 10)
    edge_top_ks: tuple[int, ...] = (1, 5, 10, 25, 50)
    edge_emit_top_k: int = 25
    max_expansions: int = 20000
    max_frontier_size: int = 4096
    strict_search: bool = True

    @property
    def metrics_profile(self) -> str:
        return str(self.eval_profile)

    @property
    def task_view(self) -> str:
        return str(self.eval_view)

    def __post_init__(self) -> None:
        if self.metrics_profile not in {"full", "rank_only"}:
            raise ValueError(
                "inference.eval_profile must be one of {'full', 'rank_only'}."
            )
        if self.task_view not in {"answer_reachability", "edge_retrieval"}:
            raise ValueError(
                "inference.eval_view must be one of {'answer_reachability', 'edge_retrieval'}."
            )
        if self.task_view == "edge_retrieval" and self.metrics_profile != "rank_only":
            raise ValueError(
                "edge_retrieval view only supports inference.eval_profile='rank_only'."
            )
        if not 0.0 < self.answer_mass_threshold <= 1.0:
            raise ValueError("inference.answer_mass_threshold must be in (0, 1].")
        if not 0.0 < self.support_mass_threshold <= 1.0:
            raise ValueError("inference.support_mass_threshold must be in (0, 1].")
        if self.support_path_overlap_penalty < 0.0:
            raise ValueError("inference.support_path_overlap_penalty must be >= 0.")
        if len(self.window_top_ks) == 0:
            raise ValueError("inference.window_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.window_top_ks):
            raise ValueError("inference.window_top_ks values must be >= 1.")
        if len(self.answer_top_ks) == 0:
            raise ValueError("inference.answer_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.answer_top_ks):
            raise ValueError("inference.answer_top_ks values must be >= 1.")
        if len(self.edge_top_ks) == 0:
            raise ValueError("inference.edge_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.edge_top_ks):
            raise ValueError("inference.edge_top_ks values must be >= 1.")
        if self.edge_emit_top_k < 1:
            raise ValueError("inference.edge_emit_top_k must be >= 1.")
        if self.max_expansions < 1:
            raise ValueError("inference.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("inference.max_frontier_size must be >= 1.")


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for the supported trajectory-heuristic variants.

    `critic` is kept as a compatibility alias for the learned heuristic.
    The auxiliary critic-specific loss is intentionally retired; only the
    heuristic itself remains as a variant selector.
    """

    kind: str = "topology"
    beta: float = 1.0
    topology_restart_prob: float = 0.25
    topology_num_iters: int = 8
    topology_eps: float = 1.0e-8
    embedding_temperature: float = 1.0
    critic_hidden_dim: int = 128
    critic_dropout: float = 0.0
    critic_loss_weight: float = 0.0
    critic_target_floor: float = 1.0e-3

    @property
    def canonical_kind(self) -> str:
        return "learned" if self.kind == "critic" else self.kind

    def __post_init__(self) -> None:
        if self.kind not in {"topology", "embedding", "learned", "critic"}:
            raise ValueError(
                "heuristic.kind must be one of {'topology', 'embedding', 'learned'} "
                "(legacy alias: 'critic')."
            )
        if self.beta < 0.0:
            raise ValueError("heuristic.beta must be >= 0.")
        if not 0.0 < self.topology_restart_prob <= 1.0:
            raise ValueError("heuristic.topology_restart_prob must be in (0, 1].")
        if self.topology_num_iters < 1:
            raise ValueError("heuristic.topology_num_iters must be >= 1.")
        if self.topology_eps <= 0.0:
            raise ValueError("heuristic.topology_eps must be > 0.")
        if self.embedding_temperature <= 0.0:
            raise ValueError("heuristic.embedding_temperature must be > 0.")
        if self.critic_hidden_dim < 1:
            raise ValueError("heuristic.learned_hidden_dim must be >= 1.")
        if self.critic_dropout < 0.0 or self.critic_dropout >= 1.0:
            raise ValueError("heuristic.learned_dropout must be in [0, 1).")
        if self.critic_loss_weight < 0.0:
            raise ValueError("heuristic.critic_loss_weight must be >= 0.")
        if not 0.0 < self.critic_target_floor < 1.0:
            raise ValueError("heuristic.critic_target_floor must be in (0, 1).")


@dataclass(frozen=True)
class SubTrajectoryBalanceConfig:
    lambda_weight: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("training.subtb.lambda_weight must be in [0, 1].")


@dataclass(frozen=True)
class GFlowNetTrainingConfig:
    rollout_batch_size: int = 8
    reward_epsilon: float = 1.0e-3
    failure_reward_mode: str = "graph_normalized"
    sampling_temperature: float = 1.0
    subtb: SubTrajectoryBalanceConfig = field(
        default_factory=SubTrajectoryBalanceConfig
    )
    # Deprecated compatibility fields. The active objective is always SubTB.
    root_loss_weight: float = 1.0
    move_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.rollout_batch_size < 1:
            raise ValueError("training.rollout_batch_size must be >= 1.")
        if self.reward_epsilon <= 0.0:
            raise ValueError("training.reward_epsilon must be > 0.")
        if self.failure_reward_mode not in {"constant", "graph_normalized"}:
            raise ValueError(
                "training.failure_reward_mode must be one of {'constant', 'graph_normalized'}."
            )
        if self.sampling_temperature <= 0.0:
            raise ValueError("training.sampling_temperature must be > 0.")
        if (
            self.root_loss_weight < 0.0
            or self.move_loss_weight < 0.0
            or self.terminal_loss_weight < 0.0
        ):
            raise ValueError("Deprecated compatibility loss weights must be >= 0.")


__all__ = [
    "AnswerReachabilityInferenceConfig",
    "GFlowNetTrainingConfig",
    "HeuristicConfig",
    "HorizonConfig",
    "SubTrajectoryBalanceConfig",
]
