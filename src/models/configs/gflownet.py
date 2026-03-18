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
        if self.max_expansions < 1:
            raise ValueError("eval_cfg.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("eval_cfg.max_frontier_size must be >= 1.")


@dataclass(frozen=True)
class HeuristicConfig:
    """Configuration for the supported search-heuristic variants."""

    kind: str = "topology"
    beta: float = 1.0
    topology_restart_prob: float = 0.25
    topology_num_iters: int = 8
    topology_eps: float = 1.0e-8
    embedding_temperature: float = 1.0
    learned_hidden_dim: int = 128
    learned_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"topology", "embedding", "learned"}:
            raise ValueError(
                "heuristic.kind must be one of {'topology', 'embedding', 'learned'}."
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
    enabled: bool = False
    ratio: float = 0.25
    warmup_passes: float = 1.0
    min_buffer_size: int = 256
    max_buffer_size: int = 50000
    max_trajectories_per_sample: int = 8

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


@dataclass(frozen=True)
class ExactAnswerObjectiveConfig:
    enabled: bool = False
    success_weight: float = 0.0
    coverage_weight: float = 0.0
    warmup_passes: float = 1.0
    interval_steps: int = 1
    max_graphs_per_batch: int = 1
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        if self.success_weight < 0.0:
            raise ValueError("training.exact_aux.success_weight must be >= 0.")
        if self.coverage_weight < 0.0:
            raise ValueError("training.exact_aux.coverage_weight must be >= 0.")
        if self.warmup_passes < 0.0:
            raise ValueError("training.exact_aux.warmup_passes must be >= 0.")
        if self.interval_steps < 1:
            raise ValueError("training.exact_aux.interval_steps must be >= 1.")
        if self.max_graphs_per_batch < 1:
            raise ValueError("training.exact_aux.max_graphs_per_batch must be >= 1.")
        if self.eps <= 0.0:
            raise ValueError("training.exact_aux.eps must be > 0.")


@dataclass(frozen=True)
class GFlowNetTrainingConfig:
    rollout_batch_size: int = 8
    reward_epsilon: float = 1.0e-3
    failure_reward_mode: str = "graph_normalized"
    sampling_temperature: float = 1.0
    sampling_temperature_schedule: SamplingTemperatureScheduleConfig = field(
        default_factory=SamplingTemperatureScheduleConfig
    )
    success_replay: SuccessfulTrajectoryReplayConfig = field(
        default_factory=SuccessfulTrajectoryReplayConfig
    )
    exact_aux: ExactAnswerObjectiveConfig = field(
        default_factory=ExactAnswerObjectiveConfig
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
    "ExactAnswerObjectiveConfig",
    "GFlowNetTrainingConfig",
    "HeuristicConfig",
    "HorizonConfig",
    "SamplingTemperatureScheduleConfig",
    "SearchEvalConfig",
    "SuccessfulTrajectoryReplayConfig",
    "SubTrajectoryBalanceConfig",
]
