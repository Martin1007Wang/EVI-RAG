from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.losses import LossOutput, SubTrajectoryBalanceLoss
from src.weaver.policy import Policy
from src.weaver.reward import RewardModel

from .engine import RolloutEngine
from .schema import RolloutBatch, RolloutStats, RolloutTraces


BackwardFn = Callable[[torch.Tensor], None]


@dataclass(frozen=True)
class TrainingRolloutResult:
    loss_output: LossOutput
    rollouts: tuple[RolloutBatch, ...]


@dataclass(frozen=True)
class RolloutRunConfig:
    temperature: float
    collect_stop_counterfactual: bool
    collect_policy_diagnostics: bool
    validate_synchronous_depth: bool


class RolloutRunner:
    """
    Chunked rollout coordinator.

    Responsibilities:
        1. split rollout requests into memory-safe chunks;
        2. call RolloutEngine to generate policy rollouts;
        3. compute loss per chunk and backpropagate with stable normalization;
        4. concatenate rollout chunks for logging/evaluation.

    Non-responsibilities:
        - no coverage proposal;
        - no behavior-policy teacher;
        - no shortest-path guide;
        - no VIGOR teacher configuration;
        - no action sampling logic;
        - no reward definition;
        - no SubTB internals.

    VIGOR, StopTB, and SubTB are owned by SubTrajectoryBalanceLoss and the
    rollout/loss data it consumes. This runner only executes chunks and applies
    backward.
    """

    def __init__(
        self,
        *,
        expand_budget: int,
        train_num_rollout: int,
        eval_num_rollout: int,
        train_chunk_size: int,
        eval_chunk_size: int,
    ) -> None:
        self.train_num_rollout = positive_int(train_num_rollout, "train_num_rollout")
        self.eval_num_rollout = positive_int(eval_num_rollout, "eval_num_rollout")
        self.train_chunk_size = positive_int(train_chunk_size, "train_chunk_size")
        self.eval_chunk_size = positive_int(eval_chunk_size, "eval_chunk_size")
        self.engine = RolloutEngine(expand_budget=int(expand_budget))

    def run_training_rollouts_and_backward(
        self,
        *,
        policy: Policy,
        reward_model: RewardModel,
        loss_fn: SubTrajectoryBalanceLoss,
        backward_fn: BackwardFn,
        batch: RetrievalBatch,
        rollout_temperature: float,
        accumulation_batches: int,
        collect_stop_counterfactual: bool = True,
        collect_policy_diagnostics: bool = False,
        validate_synchronous_depth: bool = False,
    ) -> TrainingRolloutResult:
        """
        Run one training step worth of rollouts and backpropagate chunk losses.

        Loss normalization uses:

            train_num_rollout * accumulation_batches

        This must not depend on chunk size. Otherwise changing chunk_size would
        silently change the effective learning rate.
        """
        accumulation_batches = positive_int(
            accumulation_batches,
            "accumulation_batches",
        )
        normalize_by = self.train_num_rollout * accumulation_batches

        config = RolloutRunConfig(
            temperature=float(rollout_temperature),
            collect_stop_counterfactual=bool(collect_stop_counterfactual),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
        )

        rollouts: list[RolloutBatch] = []
        loss_outputs: list[LossOutput] = []

        for current_size in rollout_chunk_sizes(
            total=self.train_num_rollout,
            chunk_size=self.train_chunk_size,
        ):
            current_rollouts = self._generate_chunk(
                policy=policy,
                reward_model=reward_model,
                batch=batch,
                num_rollouts=current_size,
                config=config,
            )

            loss_output = backward_rollouts(
                rollouts=current_rollouts,
                loss_fn=loss_fn,
                backward_fn=backward_fn,
                normalize_by=normalize_by,
            )

            rollouts.extend(current_rollouts)
            loss_outputs.append(loss_output)

        return TrainingRolloutResult(
            loss_output=LossOutput.aggregate(loss_outputs),
            rollouts=tuple(rollouts),
        )

    @torch.no_grad()
    def generate_eval_rollouts(
        self,
        *,
        policy: Policy,
        reward_model: RewardModel,
        batch: RetrievalBatch,
        temperature: float,
        num_rollouts: int | None = None,
        collect_stop_counterfactual: bool = True,
        collect_policy_diagnostics: bool = True,
        validate_synchronous_depth: bool = False,
    ) -> list[RolloutBatch]:
        total = self.eval_num_rollout if num_rollouts is None else int(num_rollouts)

        return self.generate_rollouts(
            policy=policy,
            reward_model=reward_model,
            batch=batch,
            num_rollouts=total,
            temperature=float(temperature),
            chunk_size=self.eval_chunk_size,
            collect_stop_counterfactual=collect_stop_counterfactual,
            collect_policy_diagnostics=collect_policy_diagnostics,
            validate_synchronous_depth=validate_synchronous_depth,
        )

    @torch.no_grad()
    def generate_rollouts(
        self,
        *,
        policy: Policy,
        reward_model: RewardModel,
        batch: RetrievalBatch,
        num_rollouts: int,
        temperature: float,
        chunk_size: int | None = None,
        collect_stop_counterfactual: bool = True,
        collect_policy_diagnostics: bool = True,
        validate_synchronous_depth: bool = False,
    ) -> list[RolloutBatch]:
        total = non_negative_int(num_rollouts, "num_rollouts")
        if total == 0:
            return []

        size = (
            self.eval_chunk_size
            if chunk_size is None
            else positive_int(chunk_size, "chunk_size")
        )

        config = RolloutRunConfig(
            temperature=float(temperature),
            collect_stop_counterfactual=bool(collect_stop_counterfactual),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
        )

        rollouts: list[RolloutBatch] = []
        for current_size in rollout_chunk_sizes(total=total, chunk_size=size):
            rollouts.extend(
                self._generate_chunk(
                    policy=policy,
                    reward_model=reward_model,
                    batch=batch,
                    num_rollouts=current_size,
                    config=config,
                )
            )

        return rollouts

    def _generate_chunk(
        self,
        *,
        policy: Policy,
        reward_model: RewardModel,
        batch: RetrievalBatch,
        num_rollouts: int,
        config: RolloutRunConfig,
    ) -> list[RolloutBatch]:
        num_rollouts = positive_int(num_rollouts, "num_rollouts")

        return self.engine.run_vectorized(
            policy=policy,
            retrieval_batch=batch,
            reward_model=reward_model,
            num_rollouts=num_rollouts,
            temperature=float(config.temperature),
            collect_stop_counterfactual=config.collect_stop_counterfactual,
            collect_policy_diagnostics=config.collect_policy_diagnostics,
            validate_synchronous_depth=config.validate_synchronous_depth,
        )


def backward_rollouts(
    *,
    rollouts: Sequence[RolloutBatch],
    loss_fn: SubTrajectoryBalanceLoss,
    backward_fn: BackwardFn,
    normalize_by: int,
) -> LossOutput:
    """
    Backpropagate one rollout execution chunk.

    normalize_by is the full training-step rollout count times gradient
    accumulation batches. It is not the current chunk size.
    """
    if not rollouts:
        raise ValueError("Cannot backpropagate over an empty rollout sequence.")

    normalize_by = positive_int(normalize_by, "normalize_by")

    rollout = concat_rollout_batches(rollouts)
    output = loss_fn(rollout)

    scale = float(len(rollouts)) / float(normalize_by)
    backward_fn(output.loss * scale)

    return output


def concat_rollout_batches(rollouts: Sequence[RolloutBatch]) -> RolloutBatch:
    if not rollouts:
        raise ValueError("Cannot concatenate an empty rollout sequence.")
    if len(rollouts) == 1:
        return rollouts[0]

    return RolloutBatch(
        stats=concat_rollout_stats(rollouts),
        traces=concat_rollout_traces(rollouts),
    )


def concat_rollout_stats(rollouts: Sequence[RolloutBatch]) -> RolloutStats:
    return RolloutStats(
        root_log_z=_cat_stat(rollouts, lambda rollout: rollout.stats.root_log_z),
        trajectory_length=_cat_stat(
            rollouts,
            lambda rollout: rollout.stats.trajectory_length,
        ),
        terminal_log_reward=_cat_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_log_reward,
        ),
        terminal_answer_f1=_cat_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_answer_f1,
        ),
        proposal_intervention_count=_cat_optional_or_zero_stat(
            rollouts,
            lambda rollout: rollout.stats.proposal_intervention_count,
        ),
        edge_action_entropy=_cat_stat(
            rollouts,
            lambda rollout: rollout.stats.edge_action_entropy,
        ),
        edge_action_entropy_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.stats.edge_action_entropy_valid_mask,
        ),
        terminal_edge_penalty=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_edge_penalty,
        ),
        terminal_base_log_reward=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_base_log_reward,
        ),
        terminal_utility=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_utility,
        ),
        terminal_expanded_edge_count=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_expanded_edge_count,
        ),
        terminal_minimal_edge_count=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_minimal_edge_count,
        ),
        terminal_minimality_gap=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_minimality_gap,
        ),
        terminal_minimality_penalty=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.stats.terminal_minimality_penalty,
        ),
    )


def concat_rollout_traces(rollouts: Sequence[RolloutBatch]) -> RolloutTraces:
    return RolloutTraces(
        state_log_flows=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.state_log_flows,
        ),
        log_pf=_cat_stat(rollouts, lambda rollout: rollout.traces.log_pf),
        log_pb=_cat_stat(rollouts, lambda rollout: rollout.traces.log_pb),
        action_type=_cat_stat(rollouts, lambda rollout: rollout.traces.action_type),
        continue_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.continue_mask,
        ),
        stop_mask=_cat_stat(rollouts, lambda rollout: rollout.traces.stop_mask),
        selected_edge_ids=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.selected_edge_ids,
        ),
        stop_now_log_reward=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.stop_now_log_reward,
        ),
        stop_now_answer_f1=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.stop_now_answer_f1,
        ),
        stop_now_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.stop_now_valid_mask,
        ),
        stop_log_pf=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.stop_log_pf,
        ),
        stop_tb_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.stop_tb_valid_mask,
        ),
        target_stop_prob=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.target_stop_prob,
        ),
        target_continue_prob=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.target_continue_prob,
        ),
        policy_action_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.policy_action_valid_mask,
        ),
        edge_action_entropy=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.edge_action_entropy,
        ),
        edge_action_entropy_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.edge_action_entropy_valid_mask,
        ),
        advantage_aux_loss=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.advantage_aux_loss,
        ),
        advantage_aux_valid_mask=_cat_stat(
            rollouts,
            lambda rollout: rollout.traces.advantage_aux_valid_mask,
        ),
        proposal_intervention_mask=_cat_optional_or_zero_stat(
            rollouts,
            lambda rollout: rollout.traces.proposal_intervention_mask,
        ),
        budget_exhausted_mask=_cat_optional_stat(
            rollouts,
            lambda rollout: rollout.traces.budget_exhausted_mask,
        ),
    )


def _cat_stat(
    rollouts: Sequence[RolloutBatch],
    getter: Callable[[RolloutBatch], torch.Tensor],
) -> torch.Tensor:
    return torch.cat([getter(rollout) for rollout in rollouts], dim=0)


def _cat_optional_stat(
    rollouts: Sequence[RolloutBatch],
    getter: Callable[[RolloutBatch], torch.Tensor | None],
) -> torch.Tensor | None:
    values = [getter(rollout) for rollout in rollouts]

    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "Cannot concatenate a mix of present and missing rollout tensors."
        )

    return torch.cat([value for value in values if value is not None], dim=0)


def _cat_optional_or_zero_stat(
    rollouts: Sequence[RolloutBatch],
    getter: Callable[[RolloutBatch], torch.Tensor | None],
) -> torch.Tensor:
    values = [getter(rollout) for rollout in rollouts]
    first_present = next((value for value in values if value is not None), None)

    if first_present is None:
        first = values[0]
        if first is None:
            # Fallback shape follows trajectory length when no proposal field exists.
            reference = rollouts[0].stats.trajectory_length
            return torch.zeros_like(reference)

    materialized = []
    for rollout, value in zip(rollouts, values):
        if value is None:
            reference = rollout.stats.trajectory_length
            materialized.append(torch.zeros_like(reference))
        else:
            materialized.append(value)

    return torch.cat(materialized, dim=0)


def rollout_chunk_sizes(
    *,
    total: int,
    chunk_size: int,
) -> Iterator[int]:
    remaining = non_negative_int(total, "total")
    chunk_size = positive_int(chunk_size, "chunk_size")

    while remaining > 0:
        current = min(chunk_size, remaining)
        remaining -= current
        yield current


def positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def non_negative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


__all__ = [
    "BackwardFn",
    "RolloutRunConfig",
    "RolloutRunner",
    "TrainingRolloutResult",
    "backward_rollouts",
    "concat_rollout_batches",
    "rollout_chunk_sizes",
]
