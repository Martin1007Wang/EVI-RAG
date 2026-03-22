from __future__ import annotations

from dataclasses import dataclass, field
import math


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
class AnswerRewardConfig:
    mode: str = "legacy"
    positive_utility: float = 1.0
    negative_utility: float = -1.0
    beta: float = 1.0
    cycle_penalty: float = 1.0
    failure_length_penalty_alpha: float = 0.0
    length_penalty_alpha: float | None = None
    terminal_reward_scale: str | None = None
    terminal_backward_mode: str | None = None
    normalize_by_entity_count: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "binary_ranking", "entity_sink"}:
            raise ValueError(
                "training.answer_reward.mode must be one of "
                "{'legacy', 'binary_ranking', 'entity_sink'}."
            )
        if self.beta <= 0.0:
            raise ValueError("training.answer_reward.beta must be > 0.")
        if not 0.0 < self.cycle_penalty <= 1.0:
            raise ValueError("training.answer_reward.cycle_penalty must be in (0, 1].")
        resolved_failure_alpha = float(self.failure_length_penalty_alpha)
        if self.length_penalty_alpha is not None:
            if resolved_failure_alpha > 0.0 and not math.isclose(
                resolved_failure_alpha,
                float(self.length_penalty_alpha),
            ):
                raise ValueError(
                    "training.answer_reward.length_penalty_alpha conflicts with "
                    "training.answer_reward.failure_length_penalty_alpha."
                )
            resolved_failure_alpha = float(self.length_penalty_alpha)
        if resolved_failure_alpha < 0.0:
            raise ValueError(
                "training.answer_reward.failure_length_penalty_alpha must be >= 0."
            )
        object.__setattr__(self, "failure_length_penalty_alpha", resolved_failure_alpha)
        if self.length_penalty_alpha is None:
            object.__setattr__(
                self,
                "length_penalty_alpha",
                resolved_failure_alpha,
            )
        # Deprecated knobs are accepted for compatibility but ignored: the tree
        # policy uses P_B == 1 and reward scaling stays in R(tau) directly.
        if self.terminal_reward_scale is None:
            object.__setattr__(self, "terminal_reward_scale", "none")
        if self.terminal_backward_mode is None:
            object.__setattr__(self, "terminal_backward_mode", "none")


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
class SuccessfulTrajectoryReplayConfig:
    """Replay settings.

    `ratio` targets the realized replay fraction `replay / (on_policy + replay)`
    per graph after replay trajectories are added to the on-policy batch.
    """

    enabled: bool = False
    ratio: float = 0.25
    warmup_passes: float = 1.0
    min_buffer_size: int = 256
    max_buffer_size: int = 50000
    max_trajectories_per_sample: int = 8
    max_rollouts_per_graph: int | None = None

    def __post_init__(self) -> None:
        if self.ratio < 0.0 or self.ratio >= 1.0:
            raise ValueError("training.success_replay.ratio must be in [0, 1).")
        if self.warmup_passes < 0.0:
            raise ValueError("training.success_replay.warmup_passes must be >= 0.")
        if self.min_buffer_size < 1:
            raise ValueError("training.success_replay.min_buffer_size must be >= 1.")
        if self.max_buffer_size < 1:
            raise ValueError("training.success_replay.max_buffer_size must be >= 1.")
        if self.min_buffer_size > self.max_buffer_size:
            raise ValueError(
                "training.success_replay.min_buffer_size must be <= max_buffer_size."
            )
        if self.max_trajectories_per_sample < 1:
            raise ValueError(
                "training.success_replay.max_trajectories_per_sample must be >= 1."
            )
        if self.max_rollouts_per_graph is not None and self.max_rollouts_per_graph < 1:
            raise ValueError(
                "training.success_replay.max_rollouts_per_graph must be >= 1 when set."
            )


@dataclass(frozen=True)
class AdaptiveSamplingConfig:
    enabled: bool = False
    min_rollout_batch_size: int | None = None
    max_rollout_batch_size: int | None = None
    warmup_steps: int = 0
    ema_alpha: float = 0.2
    low_success_rate_threshold: float = 0.05
    high_success_rate_threshold: float = 0.35
    low_unique_success_paths_per_100_rollouts: float = 0.5
    high_unique_success_paths_per_100_rollouts: float = 5.0
    low_start_entropy_normalized: float = 0.35
    high_start_entropy_normalized: float = 0.85
    low_subtb_residual_variance: float = 0.05
    high_subtb_residual_variance: float = 0.5
    rollout_growth_factor: float = 1.5
    rollout_shrink_factor: float = 0.75
    temperature_multiplier_up: float = 1.15
    temperature_multiplier_down: float = 0.92
    min_temperature_multiplier: float = 0.5
    max_temperature_multiplier: float = 2.5

    def __post_init__(self) -> None:
        if self.min_rollout_batch_size is not None and self.min_rollout_batch_size < 1:
            raise ValueError(
                "training.adaptive_sampling.min_rollout_batch_size must be >= 1."
            )
        if self.max_rollout_batch_size is not None and self.max_rollout_batch_size < 1:
            raise ValueError(
                "training.adaptive_sampling.max_rollout_batch_size must be >= 1."
            )
        if (
            self.min_rollout_batch_size is not None
            and self.max_rollout_batch_size is not None
            and self.min_rollout_batch_size > self.max_rollout_batch_size
        ):
            raise ValueError(
                "training.adaptive_sampling.min_rollout_batch_size must be <= max_rollout_batch_size."
            )
        if self.warmup_steps < 0:
            raise ValueError("training.adaptive_sampling.warmup_steps must be >= 0.")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("training.adaptive_sampling.ema_alpha must be in (0, 1].")
        if not 0.0 <= self.low_success_rate_threshold <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.low_success_rate_threshold must be in [0, 1]."
            )
        if not 0.0 <= self.high_success_rate_threshold <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.high_success_rate_threshold must be in [0, 1]."
            )
        if self.low_success_rate_threshold > self.high_success_rate_threshold:
            raise ValueError(
                "training.adaptive_sampling.low_success_rate_threshold must be <= high_success_rate_threshold."
            )
        if self.low_unique_success_paths_per_100_rollouts < 0.0:
            raise ValueError(
                "training.adaptive_sampling.low_unique_success_paths_per_100_rollouts must be >= 0."
            )
        if (
            self.high_unique_success_paths_per_100_rollouts
            < self.low_unique_success_paths_per_100_rollouts
        ):
            raise ValueError(
                "training.adaptive_sampling.high_unique_success_paths_per_100_rollouts must be >= low_unique_success_paths_per_100_rollouts."
            )
        if not 0.0 <= self.low_start_entropy_normalized <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.low_start_entropy_normalized must be in [0, 1]."
            )
        if not 0.0 <= self.high_start_entropy_normalized <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.high_start_entropy_normalized must be in [0, 1]."
            )
        if self.low_start_entropy_normalized > self.high_start_entropy_normalized:
            raise ValueError(
                "training.adaptive_sampling.low_start_entropy_normalized must be <= high_start_entropy_normalized."
            )
        if self.low_subtb_residual_variance < 0.0:
            raise ValueError(
                "training.adaptive_sampling.low_subtb_residual_variance must be >= 0."
            )
        if self.high_subtb_residual_variance < self.low_subtb_residual_variance:
            raise ValueError(
                "training.adaptive_sampling.high_subtb_residual_variance must be >= low_subtb_residual_variance."
            )
        if self.rollout_growth_factor < 1.0:
            raise ValueError(
                "training.adaptive_sampling.rollout_growth_factor must be >= 1.0."
            )
        if not 0.0 < self.rollout_shrink_factor <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.rollout_shrink_factor must be in (0, 1]."
            )
        if self.temperature_multiplier_up < 1.0:
            raise ValueError(
                "training.adaptive_sampling.temperature_multiplier_up must be >= 1.0."
            )
        if not 0.0 < self.temperature_multiplier_down <= 1.0:
            raise ValueError(
                "training.adaptive_sampling.temperature_multiplier_down must be in (0, 1]."
            )
        if self.min_temperature_multiplier <= 0.0:
            raise ValueError(
                "training.adaptive_sampling.min_temperature_multiplier must be > 0."
            )
        if self.max_temperature_multiplier <= 0.0:
            raise ValueError(
                "training.adaptive_sampling.max_temperature_multiplier must be > 0."
            )
        if self.min_temperature_multiplier > self.max_temperature_multiplier:
            raise ValueError(
                "training.adaptive_sampling.min_temperature_multiplier must be <= max_temperature_multiplier."
            )
        if not (
            self.min_temperature_multiplier <= 1.0 <= self.max_temperature_multiplier
        ):
            raise ValueError(
                "training.adaptive_sampling temperature multiplier bounds must contain 1.0."
            )


@dataclass(frozen=True)
class GFlowNetTrainingConfig:
    rollout_batch_size: int = 8
    reward_epsilon: float = 1.0e-3
    failure_reward_mode: str = "graph_normalized"
    answer_reward: AnswerRewardConfig = field(default_factory=AnswerRewardConfig)
    guidance: GuidanceLossConfig = field(default_factory=GuidanceLossConfig)
    sampling_temperature: float = 1.0
    sampling_temperature_schedule: SamplingTemperatureScheduleConfig = field(
        default_factory=SamplingTemperatureScheduleConfig
    )
    success_replay: SuccessfulTrajectoryReplayConfig = field(
        default_factory=SuccessfulTrajectoryReplayConfig
    )
    adaptive_sampling: AdaptiveSamplingConfig = field(
        default_factory=AdaptiveSamplingConfig
    )
    subtb: SubTrajectoryBalanceConfig = field(
        default_factory=SubTrajectoryBalanceConfig
    )

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


__all__ = [
    "AdaptiveSamplingConfig",
    "AnswerRewardConfig",
    "GFlowNetTrainingConfig",
    "GuidanceLossConfig",
    "HeuristicConfig",
    "HorizonConfig",
    "SamplingTemperatureScheduleConfig",
    "SearchEvalConfig",
    "SuccessfulTrajectoryReplayConfig",
    "SubTrajectoryBalanceConfig",
]
