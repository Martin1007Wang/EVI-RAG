from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HorizonConfig:
    max_steps: int = 4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("horizon.max_steps must be >= 1.")


@dataclass(frozen=True)
class SearchEvalConfig:
    """Evaluation settings for graph-search tasks.

    The same search policy supports both answer-reachability and edge-retrieval
    reporting, so this config owns the task selector together with the metrics
    budget.
    """

    metrics_profile: str = "full"
    task: str = "answer_ranking"
    answer_mass_threshold: float = 0.9
    support_mass_threshold: float = 0.9
    support_path_overlap_penalty: float = 0.25
    window_top_ks: tuple[int, ...] = (1, 10, 25, 50, 100)
    answer_top_ks: tuple[int, ...] = (1, 5, 10)
    edge_top_ks: tuple[int, ...] = (1, 5, 10, 25, 50)
    edge_emit_top_k: int = 25
    support_search_method: str = "flow_frontier"
    flow_prune_epsilon: float = 1.0e-3
    monte_carlo_rollouts: int = 4096
    monte_carlo_confidence: float = 0.95
    max_expansions: int = 20000
    max_frontier_size: int = 4096
    strict_search: bool = True

    def __post_init__(self) -> None:
        if self.metrics_profile not in {"full", "rank_only"}:
            raise ValueError(
                "eval_cfg.metrics_profile must be one of {'full', 'rank_only'}."
            )
        if self.task not in {"answer_ranking", "answer_reachability", "edge_retrieval"}:
            raise ValueError(
                "eval_cfg.task must be one of {'answer_ranking', 'answer_reachability', 'edge_retrieval'}."
            )
        if self.task == "edge_retrieval" and self.metrics_profile != "rank_only":
            raise ValueError(
                "edge_retrieval only supports eval_cfg.metrics_profile='rank_only'."
            )
        if not 0.0 < self.answer_mass_threshold <= 1.0:
            raise ValueError("eval_cfg.answer_mass_threshold must be in (0, 1].")
        if not 0.0 < self.support_mass_threshold <= 1.0:
            raise ValueError("eval_cfg.support_mass_threshold must be in (0, 1].")
        if self.support_path_overlap_penalty < 0.0:
            raise ValueError("eval_cfg.support_path_overlap_penalty must be >= 0.")
        if len(self.window_top_ks) == 0:
            raise ValueError("eval_cfg.window_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.window_top_ks):
            raise ValueError("eval_cfg.window_top_ks values must be >= 1.")
        if len(self.answer_top_ks) == 0:
            raise ValueError("eval_cfg.answer_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.answer_top_ks):
            raise ValueError("eval_cfg.answer_top_ks values must be >= 1.")
        if len(self.edge_top_ks) == 0:
            raise ValueError("eval_cfg.edge_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.edge_top_ks):
            raise ValueError("eval_cfg.edge_top_ks values must be >= 1.")
        if self.edge_emit_top_k < 1:
            raise ValueError("eval_cfg.edge_emit_top_k must be >= 1.")
        if self.support_search_method not in {"monte_carlo", "flow_frontier"}:
            raise ValueError(
                "eval_cfg.support_search_method must be one of {'monte_carlo', 'flow_frontier'}."
            )
        if (
            self.task == "edge_retrieval"
            and self.support_search_method != "monte_carlo"
        ):
            raise ValueError(
                "edge_retrieval only supports eval_cfg.support_search_method='monte_carlo'."
            )
        if not 0.0 <= self.flow_prune_epsilon <= 1.0:
            raise ValueError("eval_cfg.flow_prune_epsilon must be in [0, 1].")
        if self.monte_carlo_rollouts < 1:
            raise ValueError("eval_cfg.monte_carlo_rollouts must be >= 1.")
        if not 0.0 < self.monte_carlo_confidence < 1.0:
            raise ValueError("eval_cfg.monte_carlo_confidence must be in (0, 1).")
        if self.max_expansions < 1:
            raise ValueError("eval_cfg.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("eval_cfg.max_frontier_size must be >= 1.")


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for the supported search-heuristic variants."""

    kind: str = "none"
    beta: float = 1.0
    topology_restart_prob: float = 0.25
    topology_num_iters: int = 8
    topology_eps: float = 1.0e-8
    embedding_temperature: float = 1.0
    learned_hidden_dim: int = 128
    learned_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"none", "topology", "embedding", "learned"}:
            raise ValueError(
                "heuristic.kind must be one of {'none', 'topology', 'embedding', 'learned'}."
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
        if self.learned_hidden_dim < 1:
            raise ValueError("heuristic.learned_hidden_dim must be >= 1.")
        if self.learned_dropout < 0.0 or self.learned_dropout >= 1.0:
            raise ValueError("heuristic.learned_dropout must be in [0, 1).")


@dataclass(frozen=True)
class GuidanceLossConfig:
    loss_weight: float = 0.0
    detach_features: bool = True

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("training.guidance.loss_weight must be >= 0.")


@dataclass(frozen=True)
class SubTrajectoryBalanceConfig:
    lambda_weight: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("training.subtb.lambda_weight must be in [0, 1].")


@dataclass(frozen=True)
class SamplingTemperatureScheduleConfig:
    type: str = "constant"
    initial_temperature: float | None = None
    final_temperature: float | None = None
    total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "training.sampling_temperature_schedule.type must be one of "
                "{'constant', 'linear', 'cosine'}."
            )
        if self.initial_temperature is not None and self.initial_temperature <= 0.0:
            raise ValueError(
                "training.sampling_temperature_schedule.initial_temperature must be > 0."
            )
        if self.final_temperature is not None and self.final_temperature <= 0.0:
            raise ValueError(
                "training.sampling_temperature_schedule.final_temperature must be > 0."
            )
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError(
                "training.sampling_temperature_schedule.total_steps must be >= 1."
            )
        if self.type != "constant" and self.final_temperature is None:
            raise ValueError(
                "training.sampling_temperature_schedule.final_temperature must be set "
                "for annealed schedules."
            )


@dataclass(frozen=True)
class ShortestPathRewardConfig:
    weight: float = 0.0
    schedule_type: str = "constant"
    completion_power: float = 1.0
    warmup_steps: int = 0
    total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("training.shortest_path_reward.weight must be >= 0.")
        if self.schedule_type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "training.shortest_path_reward.schedule_type must be one of "
                "{'constant', 'linear', 'cosine'}."
            )
        if self.completion_power < 1.0:
            raise ValueError(
                "training.shortest_path_reward.completion_power must be >= 1.0."
            )
        if self.warmup_steps < 0:
            raise ValueError("training.shortest_path_reward.warmup_steps must be >= 0.")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError("training.shortest_path_reward.total_steps must be >= 1.")


@dataclass(frozen=True)
class GFlowNetTrainingConfig:
    rollout_batch_size: int = 8
    guidance: GuidanceLossConfig = field(default_factory=GuidanceLossConfig)
    sampling_temperature: float = 1.0
    force_stop_on_answer_hit: bool = False
    trajectory_length_discount: float = 0.97
    sampling_temperature_schedule: SamplingTemperatureScheduleConfig = field(
        default_factory=SamplingTemperatureScheduleConfig
    )
    shortest_path_reward: ShortestPathRewardConfig = field(
        default_factory=ShortestPathRewardConfig
    )
    subtb: SubTrajectoryBalanceConfig = field(
        default_factory=SubTrajectoryBalanceConfig
    )

    def __post_init__(self) -> None:
        if self.rollout_batch_size < 1:
            raise ValueError("training.rollout_batch_size must be >= 1.")
        if self.sampling_temperature <= 0.0:
            raise ValueError("training.sampling_temperature must be > 0.")
        if not 0.0 < self.trajectory_length_discount <= 1.0:
            raise ValueError("training.trajectory_length_discount must be in (0, 1].")


__all__ = [
    "GFlowNetTrainingConfig",
    "GuidanceLossConfig",
    "HeuristicConfig",
    "HorizonConfig",
    "SamplingTemperatureScheduleConfig",
    "SearchEvalConfig",
    "ShortestPathRewardConfig",
    "SubTrajectoryBalanceConfig",
]
