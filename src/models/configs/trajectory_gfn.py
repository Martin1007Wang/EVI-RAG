from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class HorizonConfig:
    max_steps: int = 4
    min_stop_steps: int = 1

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("horizon.max_steps must be >= 1.")
        if self.min_stop_steps < 0:
            raise ValueError("horizon.min_stop_steps must be >= 0.")
        if self.min_stop_steps > self.max_steps:
            raise ValueError("horizon.min_stop_steps cannot exceed horizon.max_steps.")


@dataclass(frozen=True)
class TrajectoryTrainingConfig:
    loss_type: str = "db"
    rollout_batch_size: int = 8
    lambda_start: float = 1.0
    lambda_move: float = 1.0
    lambda_stop: float = 1.0
    reward_epsilon: float = 1.0e-3
    wrong_stop_reward_mode: str = "graph_normalized"
    sampling_temperature: float = 1.0
    invalid_logits_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.loss_type != "db":
            raise ValueError("training.loss_type must be 'db' in trajectory_gfn v1.")
        if self.rollout_batch_size < 1:
            raise ValueError("training.rollout_batch_size must be >= 1.")
        if self.lambda_start < 0.0 or self.lambda_move < 0.0 or self.lambda_stop < 0.0:
            raise ValueError("DB loss weights must be >= 0.")
        if self.reward_epsilon <= 0.0:
            raise ValueError("training.reward_epsilon must be > 0.")
        if self.wrong_stop_reward_mode not in {"constant", "graph_normalized"}:
            raise ValueError(
                "training.wrong_stop_reward_mode must be one of {'constant', 'graph_normalized'}."
            )
        if self.sampling_temperature <= 0.0:
            raise ValueError("training.sampling_temperature must be > 0.")
        if self.invalid_logits_policy not in {"raise", "stop"}:
            raise ValueError(
                "training.invalid_logits_policy must be one of {'raise', 'stop'}."
            )


@dataclass(frozen=True)
class TrajectoryInferenceConfig:
    mode: str = "sampled"
    answer_mass_threshold: float = 0.9
    support_mass_threshold: float = 0.9
    support_path_overlap_penalty: float = 0.25
    compute_support_windows: bool = True
    rollout_chunk_size: int = 16
    max_rollouts: int = 256
    answer_top_ks: Tuple[int, ...] = (1, 5, 10)
    max_expansions: int = 20000
    max_frontier_size: int = 4096

    def __post_init__(self) -> None:
        if self.mode not in {"sampled", "exact", "sampled_rank_only"}:
            raise ValueError(
                "inference.mode must be one of {'sampled', 'exact', 'sampled_rank_only'}."
            )
        if not 0.0 < self.answer_mass_threshold <= 1.0:
            raise ValueError("inference.answer_mass_threshold must be in (0, 1].")
        if not 0.0 < self.support_mass_threshold <= 1.0:
            raise ValueError("inference.support_mass_threshold must be in (0, 1].")
        if self.support_path_overlap_penalty < 0.0:
            raise ValueError("inference.support_path_overlap_penalty must be >= 0.")
        if self.rollout_chunk_size < 1:
            raise ValueError("inference.rollout_chunk_size must be >= 1.")
        if self.max_rollouts < self.rollout_chunk_size:
            raise ValueError(
                "inference.max_rollouts must be >= inference.rollout_chunk_size."
            )
        if len(self.answer_top_ks) == 0:
            raise ValueError("inference.answer_top_ks must be non-empty.")
        if any(int(k) < 1 for k in self.answer_top_ks):
            raise ValueError("inference.answer_top_ks values must be >= 1.")
        if self.max_expansions < 1:
            raise ValueError("inference.max_expansions must be >= 1.")
        if self.max_frontier_size < 1:
            raise ValueError("inference.max_frontier_size must be >= 1.")
