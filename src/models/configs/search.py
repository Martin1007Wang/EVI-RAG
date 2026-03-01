# src/models/configs/search.py
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RolloutConfig:
    num_rollouts: int = 4
    max_steps: int = 10
    stop_min_steps: int = 0
    backward_prior_mode: Literal["uniform_in_degree", "uniform"] = "uniform_in_degree"
    train_oracle_force_stop: bool = False
    sampling_temperature: float = 1.0
    sampling_mode: Literal["gumbel", "greedy"] = "gumbel"
    eval_sampling_temperature: float = 0.5
    eval_sample_without_replacement: bool = True


@dataclass(frozen=True)
class BeamSearchConfig:
    beam_size: int = 10
    max_steps: int = 10
    diverse_penalty: float = 0.0
    require_done: bool = True
    rollout_metrics_every_n_batches: int = 0
