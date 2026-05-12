from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, fields

import torch

from src.data.schema import RetrievalBatch
from src.weaver.loss import LossOutput
from src.weaver.policy import Policy
from src.weaver.reward import RewardModel

from .engine import RolloutEngine
from .schema import RolloutBatch, RolloutStats, RolloutTraces


@dataclass(frozen=True, slots=True)
class TrainingRolloutResult:
    loss_output: LossOutput
    rollouts: tuple[RolloutBatch, ...]


@dataclass(frozen=True, slots=True)
class TrainingRolloutChunk:
    rollouts: tuple[RolloutBatch, ...]
    num_rollouts: int


@dataclass(frozen=True, slots=True)
class RolloutRunConfig:
    temperature: float
    collect_policy_diagnostics: bool
    validate_synchronous_depth: bool
    store_stop_now_reward: bool
    store_te_bfm: bool
    store_bdb: bool
    store_budgeted_flow: bool


class RolloutRunner:
    """
    Chunked rollout coordinator.

    Responsibilities:
        1. split rollout requests into memory-safe chunks;
        2. call RolloutEngine to generate policy rollouts;
        3. yield rollout chunks for module-owned loss and backward;
        4. concatenate rollout chunks for logging / evaluation.
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

    def iter_training_rollout_chunks(
        self,
        *,
        policy: Policy,
        reward_model: RewardModel,
        batch: RetrievalBatch,
        rollout_temperature: float,
        collect_policy_diagnostics: bool = False,
        validate_synchronous_depth: bool = False,
        loss_fn: torch.nn.Module | None = None,
    ) -> Iterator[TrainingRolloutChunk]:
        """
        Yield one training-step worth of memory-safe rollout chunks.

        The Lightning module owns loss computation and backward so optimizer
        semantics stay visible at the training entry point.
        """
        store_stop_now_reward = requires_stop_now_trace(
            loss_fn=loss_fn,
        )
        store_te_bfm = requires_te_bfm_trace(loss_fn=loss_fn)
        store_bdb = requires_bdb_trace(loss_fn=loss_fn)
        store_budgeted_flow = requires_budgeted_flow_trace(loss_fn=loss_fn)

        config = RolloutRunConfig(
            temperature=float(rollout_temperature),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
            store_stop_now_reward=store_stop_now_reward,
            store_te_bfm=store_te_bfm,
            store_bdb=store_bdb,
            store_budgeted_flow=store_budgeted_flow,
        )

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

            yield TrainingRolloutChunk(
                rollouts=tuple(current_rollouts),
                num_rollouts=current_size,
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
        collect_policy_diagnostics: bool = True,
        validate_synchronous_depth: bool = False,
        loss_fn: torch.nn.Module | None = None,
    ) -> list[RolloutBatch]:
        total = (
            self.eval_num_rollout
            if num_rollouts is None
            else non_negative_int(num_rollouts, "num_rollouts")
        )

        return self.generate_rollouts(
            policy=policy,
            reward_model=reward_model,
            batch=batch,
            num_rollouts=total,
            temperature=float(temperature),
            chunk_size=self.eval_chunk_size,
            collect_policy_diagnostics=collect_policy_diagnostics,
            validate_synchronous_depth=validate_synchronous_depth,
            loss_fn=loss_fn,
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
        collect_policy_diagnostics: bool = True,
        validate_synchronous_depth: bool = False,
        loss_fn: torch.nn.Module | None = None,
    ) -> list[RolloutBatch]:
        total = non_negative_int(num_rollouts, "num_rollouts")
        if total == 0:
            return []

        size = (
            self.eval_chunk_size
            if chunk_size is None
            else positive_int(chunk_size, "chunk_size")
        )

        store_stop_now_reward = requires_stop_now_trace(
            loss_fn=loss_fn,
        )
        store_te_bfm = requires_te_bfm_trace(loss_fn=loss_fn)
        store_bdb = requires_bdb_trace(loss_fn=loss_fn)
        store_budgeted_flow = requires_budgeted_flow_trace(loss_fn=loss_fn)

        config = RolloutRunConfig(
            temperature=float(temperature),
            collect_policy_diagnostics=bool(collect_policy_diagnostics),
            validate_synchronous_depth=bool(validate_synchronous_depth),
            store_stop_now_reward=store_stop_now_reward,
            store_te_bfm=store_te_bfm,
            store_bdb=store_bdb,
            store_budgeted_flow=store_budgeted_flow,
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
            collect_policy_diagnostics=config.collect_policy_diagnostics,
            validate_synchronous_depth=config.validate_synchronous_depth,
            store_stop_now_reward=config.store_stop_now_reward,
            store_te_bfm=config.store_te_bfm,
            store_bdb=config.store_bdb,
            store_budgeted_flow=config.store_budgeted_flow,
        )


def detach_rollout_for_metrics(
    rollout: RolloutBatch,
    *,
    device: torch.device | str | None = None,
) -> RolloutBatch:
    """
    Return a rollout snapshot that is safe to retain after chunk backward.
    """
    return RolloutBatch(
        stats=_detach_rollout_dataclass(rollout.stats, device=device),
        traces=_detach_rollout_dataclass(rollout.traces, device=device),
    )


def _detach_rollout_dataclass(
    value: RolloutStats | RolloutTraces,
    *,
    device: torch.device | str | None,
) -> RolloutStats | RolloutTraces:
    detached = {
        field.name: _detach_rollout_tensor(
            getattr(value, field.name),
            device=device,
        )
        for field in fields(value)
    }
    return type(value)(**detached)


def _detach_rollout_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if tensor is None:
        return None

    detached = tensor.detach()
    if device is None:
        return detached
    return detached.to(device=device)


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
        trajectory_length=_cat(
            rollouts,
            lambda rollout: rollout.stats.trajectory_length,
        ),
        terminal_log_reward=_cat(
            rollouts,
            lambda rollout: rollout.stats.terminal_log_reward,
        ),
        terminal_answer_f1=_cat(
            rollouts,
            lambda rollout: rollout.stats.terminal_answer_f1,
        ),
        edge_action_entropy=_cat(
            rollouts,
            lambda rollout: rollout.stats.edge_action_entropy,
        ),
        edge_action_count=_cat(
            rollouts,
            lambda rollout: rollout.stats.edge_action_count,
        ),
        source_graph_id=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.source_graph_id,
        ),
        terminal_complexity_penalty=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_complexity_penalty,
        ),
        terminal_base_log_reward=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_base_log_reward,
        ),
        terminal_utility=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_utility,
        ),
        terminal_shortest_path_potential=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_shortest_path_potential,
        ),
        terminal_expanded_edge_count=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_expanded_edge_count,
        ),
        terminal_answer_degree_excess=_cat_optional(
            rollouts,
            lambda rollout: rollout.stats.terminal_answer_degree_excess,
        ),
    )


def concat_rollout_traces(rollouts: Sequence[RolloutBatch]) -> RolloutTraces:
    return RolloutTraces(
        log_pf=_cat(
            rollouts,
            lambda rollout: rollout.traces.log_pf,
        ),
        log_pb=_cat(
            rollouts,
            lambda rollout: rollout.traces.log_pb,
        ),
        state_log_flow=_cat(
            rollouts,
            lambda rollout: rollout.traces.state_log_flow,
        ),
        db_parent_log_reward=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_parent_log_reward,
        ),
        db_child_log_reward=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_child_log_reward,
        ),
        db_parent_shortest_path_potential=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_parent_shortest_path_potential,
        ),
        db_child_shortest_path_potential=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_child_shortest_path_potential,
        ),
        db_parent_process_log_bonus=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_parent_process_log_bonus,
        ),
        db_child_process_log_bonus=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_child_process_log_bonus,
        ),
        db_log_p_stop_parent=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_log_p_stop_parent,
        ),
        db_log_p_stop_child=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_log_p_stop_child,
        ),
        db_log_pf_expand=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_log_pf_expand,
        ),
        db_log_pb=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_log_pb,
        ),
        db_valid_mask=_cat(
            rollouts,
            lambda rollout: rollout.traces.db_valid_mask,
        ),
        action_type=_cat(
            rollouts,
            lambda rollout: rollout.traces.action_type,
        ),
        continue_mask=_cat(
            rollouts,
            lambda rollout: rollout.traces.continue_mask,
        ),
        stop_mask=_cat(
            rollouts,
            lambda rollout: rollout.traces.stop_mask,
        ),
        selected_edge_ids=_cat(
            rollouts,
            lambda rollout: rollout.traces.selected_edge_ids,
        ),
        stop_now_log_reward=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.stop_now_log_reward,
        ),
        stop_now_answer_f1=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.stop_now_answer_f1,
        ),
        stop_now_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.stop_now_valid_mask,
        ),
        target_stop_prob=_cat(
            rollouts,
            lambda rollout: rollout.traces.target_stop_prob,
        ),
        target_continue_prob=_cat(
            rollouts,
            lambda rollout: rollout.traces.target_continue_prob,
        ),
        policy_action_valid_mask=_cat(
            rollouts,
            lambda rollout: rollout.traces.policy_action_valid_mask,
        ),
        edge_action_entropy=_cat(
            rollouts,
            lambda rollout: rollout.traces.edge_action_entropy,
        ),
        edge_action_entropy_valid_mask=_cat(
            rollouts,
            lambda rollout: rollout.traces.edge_action_entropy_valid_mask,
        ),
        log_p_stop=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.log_p_stop,
        ),
        budget_exhausted_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budget_exhausted_mask,
        ),
        te_bfm_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_loss,
        ),
        te_bfm_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_valid_mask,
        ),
        te_bfm_residual_abs=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_residual_abs,
        ),
        te_bfm_target_log_value=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_target_log_value,
        ),
        te_bfm_log_reward=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_log_reward,
        ),
        te_bfm_stop_prob=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_stop_prob,
        ),
        te_bfm_frontier_edge_count=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_frontier_edge_count,
        ),
        te_bfm_counterfactual_child_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_counterfactual_child_loss,
        ),
        te_bfm_frontier_cap_used=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_frontier_cap_used,
        ),
        te_bfm_frontier_cap_dropped_edge_count=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.te_bfm_frontier_cap_dropped_edge_count,
        ),
        bdb_stop_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_stop_loss,
        ),
        bdb_edge_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_edge_loss,
        ),
        bdb_base_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_base_loss,
        ),
        bdb_stop_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_stop_valid_mask,
        ),
        bdb_edge_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_edge_valid_mask,
        ),
        bdb_base_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_base_valid_mask,
        ),
        bdb_delta_stop=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_delta_stop,
        ),
        bdb_delta_edge=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_delta_edge,
        ),
        bdb_delta_base=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_delta_base,
        ),
        bdb_frontier_size=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_frontier_size,
        ),
        bdb_parent_count=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_parent_count,
        ),
        bdb_log_reward=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_log_reward,
        ),
        bdb_log_flow=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.bdb_log_flow,
        ),
        budgeted_policy_kl=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budgeted_policy_kl,
        ),
        budgeted_terminal_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budgeted_terminal_loss,
        ),
        budgeted_value_loss=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budgeted_value_loss,
        ),
        budgeted_valid_mask=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budgeted_valid_mask,
        ),
        oracle_v_star=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.oracle_v_star,
        ),
        oracle_terminal_j=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.oracle_terminal_j,
        ),
        oracle_stop_prob=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.oracle_stop_prob,
        ),
        oracle_edge_entropy=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.oracle_edge_entropy,
        ),
        model_stop_prob=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.model_stop_prob,
        ),
        budgeted_oracle_good_edge_policy_mass=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.budgeted_oracle_good_edge_policy_mass,
        ),
        sampled_oracle_good_edge_rate=_cat_optional(
            rollouts,
            lambda rollout: rollout.traces.sampled_oracle_good_edge_rate,
        ),
    )


def _cat(
    rollouts: Sequence[RolloutBatch],
    getter: Callable[[RolloutBatch], torch.Tensor],
) -> torch.Tensor:
    return torch.cat([getter(rollout) for rollout in rollouts], dim=0)


def _cat_optional(
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


def requires_stop_now_trace(
    *,
    loss_fn: object | None = None,
) -> bool:
    return bool(getattr(loss_fn, "requires_stop_now_reward", False))


def requires_te_bfm_trace(
    *,
    loss_fn: object | None = None,
) -> bool:
    return bool(getattr(loss_fn, "requires_te_bfm_trace", False))


def requires_bdb_trace(
    *,
    loss_fn: object | None = None,
) -> bool:
    return bool(getattr(loss_fn, "requires_bdb_trace", False))


def requires_budgeted_flow_trace(
    *,
    loss_fn: object | None = None,
) -> bool:
    return bool(getattr(loss_fn, "requires_budgeted_flow_trace", False))


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
    "RolloutRunConfig",
    "RolloutRunner",
    "TrainingRolloutChunk",
    "TrainingRolloutResult",
    "concat_rollout_batches",
    "concat_rollout_stats",
    "concat_rollout_traces",
    "detach_rollout_for_metrics",
    "requires_stop_now_trace",
    "requires_te_bfm_trace",
    "requires_bdb_trace",
    "requires_budgeted_flow_trace",
    "rollout_chunk_sizes",
]
