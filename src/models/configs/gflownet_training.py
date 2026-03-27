from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ActionPriorConfig:
    """Proposal-policy action priors used only for training-time sampling."""

    root_beta: float = 0.0
    edge_beta: float = 0.0
    stop_beta: float = 0.0
    node_topology_weight: float = 1.0
    node_embedding_weight: float = 1.0
    relation_embedding_weight: float = 1.0
    target_node_weight: float = 0.5
    progress_weight: float = 1.0
    intent_alignment_weight: float = 0.0
    intent_relation_weight: float = 1.0
    intent_target_weight: float = 1.0
    shortest_path_edge_weight: float = 0.0
    answer_distance_weight: float = 0.0
    stop_node_weight: float = 1.0
    topology_restart_prob: float = 0.25
    topology_num_iters: int = 8
    topology_eps: float = 1.0e-8
    embedding_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.root_beta < 0.0:
            raise ValueError("action_prior.root_beta must be >= 0.")
        if self.edge_beta < 0.0:
            raise ValueError("action_prior.edge_beta must be >= 0.")
        if self.stop_beta < 0.0:
            raise ValueError("action_prior.stop_beta must be >= 0.")
        if self.node_topology_weight < 0.0:
            raise ValueError("action_prior.node_topology_weight must be >= 0.")
        if self.node_embedding_weight < 0.0:
            raise ValueError("action_prior.node_embedding_weight must be >= 0.")
        if self.relation_embedding_weight < 0.0:
            raise ValueError("action_prior.relation_embedding_weight must be >= 0.")
        if self.target_node_weight < 0.0:
            raise ValueError("action_prior.target_node_weight must be >= 0.")
        if self.progress_weight < 0.0:
            raise ValueError("action_prior.progress_weight must be >= 0.")
        if self.intent_alignment_weight < 0.0:
            raise ValueError("action_prior.intent_alignment_weight must be >= 0.")
        if self.intent_relation_weight < 0.0:
            raise ValueError("action_prior.intent_relation_weight must be >= 0.")
        if self.intent_target_weight < 0.0:
            raise ValueError("action_prior.intent_target_weight must be >= 0.")
        if self.shortest_path_edge_weight < 0.0:
            raise ValueError("action_prior.shortest_path_edge_weight must be >= 0.")
        if self.answer_distance_weight < 0.0:
            raise ValueError("action_prior.answer_distance_weight must be >= 0.")
        if self.stop_node_weight < 0.0:
            raise ValueError("action_prior.stop_node_weight must be >= 0.")
        if not 0.0 < self.topology_restart_prob <= 1.0:
            raise ValueError("action_prior.topology_restart_prob must be in (0, 1].")
        if self.topology_num_iters < 1:
            raise ValueError("action_prior.topology_num_iters must be >= 1.")
        if self.topology_eps <= 0.0:
            raise ValueError("action_prior.topology_eps must be > 0.")
        if self.embedding_temperature <= 0.0:
            raise ValueError("action_prior.embedding_temperature must be > 0.")
        if self.intent_alignment_weight > 0.0 and (
            self.intent_relation_weight == 0.0 and self.intent_target_weight == 0.0
        ):
            raise ValueError(
                "action_prior.intent_alignment_weight requires at least one of "
                "intent_relation_weight or intent_target_weight to be > 0."
            )

    @property
    def enabled(self) -> bool:
        return any(
            beta > 0.0
            for beta in (
                float(self.root_beta),
                float(self.edge_beta),
                float(self.stop_beta),
            )
        )


@dataclass(frozen=True)
class SubTrajectoryBalanceConfig:
    lambda_weight: float = 1.0
    normalize: bool = True
    root_loss_weight: float = 1.0
    pairwise_loss_weight: float = 1.0
    terminal_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.lambda_weight <= 1.0:
            raise ValueError("training.subtb.lambda_weight must be in [0, 1].")
        if self.root_loss_weight < 0.0:
            raise ValueError("training.subtb.root_loss_weight must be >= 0.")
        if self.pairwise_loss_weight < 0.0:
            raise ValueError("training.subtb.pairwise_loss_weight must be >= 0.")
        if self.terminal_loss_weight < 0.0:
            raise ValueError("training.subtb.terminal_loss_weight must be >= 0.")


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
class ActionPriorScheduleConfig:
    type: str = "constant"
    initial_scale: float | None = None
    final_scale: float | None = None
    total_steps: int | None = None

    def __post_init__(self) -> None:
        if self.type not in {"constant", "linear", "cosine"}:
            raise ValueError(
                "training.action_prior_schedule.type must be one of "
                "{'constant', 'linear', 'cosine'}."
            )
        if self.initial_scale is not None and self.initial_scale < 0.0:
            raise ValueError(
                "training.action_prior_schedule.initial_scale must be >= 0."
            )
        if self.final_scale is not None and self.final_scale < 0.0:
            raise ValueError("training.action_prior_schedule.final_scale must be >= 0.")
        if self.total_steps is not None and self.total_steps < 1:
            raise ValueError("training.action_prior_schedule.total_steps must be >= 1.")
        if self.type != "constant" and self.final_scale is None:
            raise ValueError(
                "training.action_prior_schedule.final_scale must be set for "
                "annealed schedules."
            )


@dataclass(frozen=True)
class SuccessReplayConfig:
    mix_alpha: float = 0.0
    capacity: int = 1024
    min_buffer_size: int = 64
    replay_trajectories_per_step: int | None = None
    deduplicate: bool = True
    add_shortest_path_guidance: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.mix_alpha) < 1.0:
            raise ValueError("training.success_replay.mix_alpha must be in [0, 1).")
        if int(self.capacity) < 1:
            raise ValueError("training.success_replay.capacity must be >= 1.")
        if int(self.min_buffer_size) < 1:
            raise ValueError("training.success_replay.min_buffer_size must be >= 1.")
        if int(self.min_buffer_size) > int(self.capacity):
            raise ValueError(
                "training.success_replay.min_buffer_size must be <= capacity."
            )
        if (
            self.replay_trajectories_per_step is not None
            and int(self.replay_trajectories_per_step) < 1
        ):
            raise ValueError(
                "training.success_replay.replay_trajectories_per_step must be >= 1."
            )

    @property
    def enabled(self) -> bool:
        return float(self.mix_alpha) > 0.0


@dataclass(frozen=True)
class AnswerQuotientConfig:
    enabled: bool = False
    weight: float = 0.0
    direct_entity_ranking_weight: float = 0.0
    replace_terminal_loss: bool = False
    gold_reward_mode: str = "shared"
    allocate_stop_mass: bool = False

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ValueError("training.answer_quotient.weight must be >= 0.")
        if self.direct_entity_ranking_weight < 0.0:
            raise ValueError(
                "training.answer_quotient.direct_entity_ranking_weight must be >= 0."
            )
        if self.gold_reward_mode not in {"shared", "unit"}:
            raise ValueError(
                "training.answer_quotient.gold_reward_mode must be one of {'shared', 'unit'}."
            )
        if float(self.direct_entity_ranking_weight) > 0.0 and not bool(self.enabled):
            raise ValueError(
                "training.answer_quotient.direct_entity_ranking_weight requires enabled=True."
            )
        if bool(self.replace_terminal_loss) and not bool(self.enabled):
            raise ValueError(
                "training.answer_quotient.replace_terminal_loss requires enabled=True."
            )
        if bool(self.replace_terminal_loss) and float(self.weight) <= 0.0:
            raise ValueError(
                "training.answer_quotient.replace_terminal_loss requires weight > 0."
            )

    @property
    def active(self) -> bool:
        return bool(self.enabled) and (
            float(self.weight) > 0.0 or bool(self.replace_terminal_loss)
        )

    @property
    def stop_allocation_active(self) -> bool:
        return bool(self.enabled) and bool(self.allocate_stop_mass)

    @property
    def direct_entity_ranking_active(self) -> bool:
        return bool(self.enabled) and float(self.direct_entity_ranking_weight) > 0.0


@dataclass(frozen=True)
class PotentialRewardConfig:
    answer_distance_weight: float = 0.0
    unreachable_distance: int | None = None

    def __post_init__(self) -> None:
        if float(self.answer_distance_weight) < 0.0:
            raise ValueError(
                "training.potential_reward.answer_distance_weight must be >= 0."
            )
        if self.unreachable_distance is not None and int(self.unreachable_distance) < 0:
            raise ValueError(
                "training.potential_reward.unreachable_distance must be >= 0 when set."
            )

    @property
    def active(self) -> bool:
        return float(self.answer_distance_weight) > 0.0


@dataclass(frozen=True)
class GFlowNetTrainingConfig:
    rollouts_per_graph: int = 8
    sampling_temperature: float = 1.0
    force_stop_on_answer_hit: bool = False
    terminal_failure_log_reward: float = -3.0
    step_log_penalty: float = 0.0
    answer_stop_log_reward_bonus: float = 0.0
    sampling_temperature_schedule: SamplingTemperatureScheduleConfig = field(
        default_factory=SamplingTemperatureScheduleConfig
    )
    action_prior_schedule: ActionPriorScheduleConfig = field(
        default_factory=ActionPriorScheduleConfig
    )
    success_replay: SuccessReplayConfig = field(default_factory=SuccessReplayConfig)
    answer_quotient: AnswerQuotientConfig = field(default_factory=AnswerQuotientConfig)
    potential_reward: PotentialRewardConfig = field(
        default_factory=PotentialRewardConfig
    )
    subtb: SubTrajectoryBalanceConfig = field(
        default_factory=SubTrajectoryBalanceConfig
    )

    def __post_init__(self) -> None:
        if int(self.rollouts_per_graph) < 1:
            raise ValueError("training.rollouts_per_graph must be >= 1.")
        if self.sampling_temperature <= 0.0:
            raise ValueError("training.sampling_temperature must be > 0.")
        if self.terminal_failure_log_reward > 0.0:
            raise ValueError("training.terminal_failure_log_reward must be <= 0.")
        if self.step_log_penalty > 0.0:
            raise ValueError("training.step_log_penalty must be <= 0.")
        if self.answer_stop_log_reward_bonus < 0.0:
            raise ValueError("training.answer_stop_log_reward_bonus must be >= 0.")


__all__ = [
    "ActionPriorConfig",
    "ActionPriorScheduleConfig",
    "AnswerQuotientConfig",
    "GFlowNetTrainingConfig",
    "PotentialRewardConfig",
    "SamplingTemperatureScheduleConfig",
    "SubTrajectoryBalanceConfig",
    "SuccessReplayConfig",
]
