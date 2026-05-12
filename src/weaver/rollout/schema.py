from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .buffer import RolloutBuffer


@dataclass(frozen=True, slots=True)
class StepResult:
    """
    One executed rollout transition.

    Shape convention:
        all tensors: [B]

    terminal_* fields are only meaningful for graphs whose stop_mask is true
    at this transition. RolloutBuffer copies them into trajectory-level fields.
    """

    log_pf: torch.Tensor
    log_pb: torch.Tensor

    action_type: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    selected_edge_ids: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    terminal_complexity_penalty: torch.Tensor
    terminal_base_log_reward: torch.Tensor
    terminal_utility: torch.Tensor
    terminal_shortest_path_potential: torch.Tensor
    terminal_expanded_edge_count: torch.Tensor
    terminal_answer_degree_excess: torch.Tensor


@dataclass(frozen=True, slots=True)
class RolloutTraceSpec:
    """
    Explicit storage contract for optional rollout traces.

    Current training consumes SubTB traces. Legacy DB/DAGDB-shaped fields may
    still be present on RolloutTraces for diagnostics or checkpoint-era
    compatibility, but they are not the active objective.
    """

    store_stop_now_reward: bool = False
    store_te_bfm: bool = False
    store_bdb: bool = False
    store_budgeted_flow: bool = False


@dataclass(frozen=True, slots=True)
class RolloutTraces:
    """
    Step-level rollout traces.

    Shape convention:
        all tensors: [B, T]
    """

    log_pf: torch.Tensor
    log_pb: torch.Tensor
    state_log_flow: torch.Tensor

    # Legacy diagnostic traces. The current loss only supports SubTB and does
    # not treat these DB/DAGDB-shaped fields as the main training path.
    db_parent_log_reward: torch.Tensor
    db_child_log_reward: torch.Tensor
    db_parent_shortest_path_potential: torch.Tensor
    db_child_shortest_path_potential: torch.Tensor
    db_parent_process_log_bonus: torch.Tensor
    db_child_process_log_bonus: torch.Tensor
    db_log_p_stop_parent: torch.Tensor
    db_log_p_stop_child: torch.Tensor
    db_log_pf_expand: torch.Tensor
    db_log_pb: torch.Tensor
    db_valid_mask: torch.Tensor

    action_type: torch.Tensor
    continue_mask: torch.Tensor
    stop_mask: torch.Tensor
    selected_edge_ids: torch.Tensor

    target_stop_prob: torch.Tensor
    target_continue_prob: torch.Tensor
    policy_action_valid_mask: torch.Tensor

    edge_action_entropy: torch.Tensor
    edge_action_entropy_valid_mask: torch.Tensor

    log_p_stop: torch.Tensor | None = None

    stop_now_log_reward: torch.Tensor | None = None
    stop_now_answer_f1: torch.Tensor | None = None
    stop_now_valid_mask: torch.Tensor | None = None

    budget_exhausted_mask: torch.Tensor | None = None

    te_bfm_loss: torch.Tensor | None = None
    te_bfm_valid_mask: torch.Tensor | None = None
    te_bfm_residual_abs: torch.Tensor | None = None
    te_bfm_target_log_value: torch.Tensor | None = None
    te_bfm_log_reward: torch.Tensor | None = None
    te_bfm_stop_prob: torch.Tensor | None = None
    te_bfm_frontier_edge_count: torch.Tensor | None = None
    te_bfm_counterfactual_child_loss: torch.Tensor | None = None
    te_bfm_frontier_cap_used: torch.Tensor | None = None
    te_bfm_frontier_cap_dropped_edge_count: torch.Tensor | None = None

    bdb_stop_loss: torch.Tensor | None = None
    bdb_edge_loss: torch.Tensor | None = None
    bdb_base_loss: torch.Tensor | None = None
    bdb_stop_valid_mask: torch.Tensor | None = None
    bdb_edge_valid_mask: torch.Tensor | None = None
    bdb_base_valid_mask: torch.Tensor | None = None
    bdb_delta_stop: torch.Tensor | None = None
    bdb_delta_edge: torch.Tensor | None = None
    bdb_delta_base: torch.Tensor | None = None
    bdb_frontier_size: torch.Tensor | None = None
    bdb_parent_count: torch.Tensor | None = None
    bdb_log_reward: torch.Tensor | None = None
    bdb_log_flow: torch.Tensor | None = None
    budgeted_policy_kl: torch.Tensor | None = None
    budgeted_terminal_loss: torch.Tensor | None = None
    budgeted_value_loss: torch.Tensor | None = None
    budgeted_valid_mask: torch.Tensor | None = None
    oracle_v_star: torch.Tensor | None = None
    oracle_terminal_j: torch.Tensor | None = None
    oracle_stop_prob: torch.Tensor | None = None
    oracle_edge_entropy: torch.Tensor | None = None
    model_stop_prob: torch.Tensor | None = None
    budgeted_oracle_good_edge_policy_mass: torch.Tensor | None = None
    sampled_oracle_good_edge_rate: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_all_or_none(
            (
                self.stop_now_log_reward,
                self.stop_now_answer_f1,
                self.stop_now_valid_mask,
            ),
            name="stop-now rollout traces",
        )
        _require_all_or_none(
            (
                self.te_bfm_loss,
                self.te_bfm_valid_mask,
                self.te_bfm_residual_abs,
                self.te_bfm_target_log_value,
                self.te_bfm_log_reward,
                self.te_bfm_stop_prob,
                self.te_bfm_frontier_edge_count,
                self.te_bfm_counterfactual_child_loss,
                self.te_bfm_frontier_cap_used,
                self.te_bfm_frontier_cap_dropped_edge_count,
            ),
            name="TE-BFM rollout traces",
        )
        _require_all_or_none(
            (
                self.bdb_stop_loss,
                self.bdb_edge_loss,
                self.bdb_base_loss,
                self.bdb_stop_valid_mask,
                self.bdb_edge_valid_mask,
                self.bdb_base_valid_mask,
                self.bdb_delta_stop,
                self.bdb_delta_edge,
                self.bdb_delta_base,
                self.bdb_frontier_size,
                self.bdb_parent_count,
                self.bdb_log_reward,
                self.bdb_log_flow,
            ),
            name="BDB rollout traces",
        )
        _require_all_or_none(
            (
                self.budgeted_policy_kl,
                self.budgeted_terminal_loss,
                self.budgeted_value_loss,
                self.budgeted_valid_mask,
                self.oracle_v_star,
                self.oracle_terminal_j,
                self.oracle_stop_prob,
                self.oracle_edge_entropy,
                self.model_stop_prob,
                self.budgeted_oracle_good_edge_policy_mass,
                self.sampled_oracle_good_edge_rate,
            ),
            name="budgeted-flow rollout traces",
        )


@dataclass(frozen=True, slots=True)
class RolloutStats:
    """
    Trajectory-level rollout quantities.

    Shape convention:
        all tensors: [B]
    """

    trajectory_length: torch.Tensor

    terminal_log_reward: torch.Tensor
    terminal_answer_f1: torch.Tensor

    edge_action_entropy: torch.Tensor
    edge_action_count: torch.Tensor
    source_graph_id: torch.Tensor | None = None

    terminal_complexity_penalty: torch.Tensor | None = None
    terminal_base_log_reward: torch.Tensor | None = None
    terminal_utility: torch.Tensor | None = None
    terminal_shortest_path_potential: torch.Tensor | None = None
    terminal_expanded_edge_count: torch.Tensor | None = None
    terminal_answer_degree_excess: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    """
    Immutable rollout result consumed by losses and metrics.
    """

    stats: RolloutStats
    traces: RolloutTraces

    @classmethod
    def from_buffer(
        cls,
        *,
        buffer: RolloutBuffer,
        source_graph_id: torch.Tensor | None = None,
    ) -> "RolloutBatch":
        edge_entropy = buffer.edge_action_entropy
        edge_entropy_valid = buffer.edge_action_entropy_valid_mask.to(
            dtype=torch.float32
        )

        return cls(
            stats=RolloutStats(
                trajectory_length=buffer.traj_len,
                terminal_log_reward=buffer.terminal_log_reward,
                terminal_answer_f1=buffer.terminal_answer_f1,
                edge_action_entropy=edge_entropy.sum(dim=1),
                edge_action_count=edge_entropy_valid.sum(dim=1),
                source_graph_id=source_graph_id,
                terminal_complexity_penalty=buffer.terminal_complexity_penalty,
                terminal_base_log_reward=buffer.terminal_base_log_reward,
                terminal_utility=buffer.terminal_utility,
                terminal_shortest_path_potential=buffer.terminal_shortest_path_potential,
                terminal_expanded_edge_count=buffer.terminal_expanded_edge_count,
                terminal_answer_degree_excess=buffer.terminal_answer_degree_excess,
            ),
            traces=RolloutTraces(
                log_pf=buffer.step_log_pf,
                log_pb=buffer.step_log_pb,
                state_log_flow=buffer.state_log_flow,
                log_p_stop=buffer.log_p_stop,
                db_parent_log_reward=buffer.db_parent_log_reward,
                db_child_log_reward=buffer.db_child_log_reward,
                db_parent_shortest_path_potential=buffer.db_parent_shortest_path_potential,
                db_child_shortest_path_potential=buffer.db_child_shortest_path_potential,
                db_parent_process_log_bonus=buffer.db_parent_process_log_bonus,
                db_child_process_log_bonus=buffer.db_child_process_log_bonus,
                db_log_p_stop_parent=buffer.db_log_p_stop_parent,
                db_log_p_stop_child=buffer.db_log_p_stop_child,
                db_log_pf_expand=buffer.db_log_pf_expand,
                db_log_pb=buffer.db_log_pb,
                db_valid_mask=buffer.db_valid_mask,
                action_type=buffer.action_type,
                continue_mask=buffer.continue_mask,
                stop_mask=buffer.stop_mask,
                selected_edge_ids=buffer.selected_edge_ids,
                stop_now_log_reward=buffer.stop_now_log_reward,
                stop_now_answer_f1=buffer.stop_now_answer_f1,
                stop_now_valid_mask=buffer.stop_now_valid_mask,
                target_stop_prob=buffer.target_stop_prob,
                target_continue_prob=buffer.target_continue_prob,
                policy_action_valid_mask=buffer.policy_action_valid_mask,
                edge_action_entropy=edge_entropy,
                edge_action_entropy_valid_mask=buffer.edge_action_entropy_valid_mask,
                budget_exhausted_mask=buffer.budget_exhausted_mask,
                te_bfm_loss=buffer.te_bfm_loss,
                te_bfm_valid_mask=buffer.te_bfm_valid_mask,
                te_bfm_residual_abs=buffer.te_bfm_residual_abs,
                te_bfm_target_log_value=buffer.te_bfm_target_log_value,
                te_bfm_log_reward=buffer.te_bfm_log_reward,
                te_bfm_stop_prob=buffer.te_bfm_stop_prob,
                te_bfm_frontier_edge_count=buffer.te_bfm_frontier_edge_count,
                te_bfm_counterfactual_child_loss=(
                    buffer.te_bfm_counterfactual_child_loss
                ),
                te_bfm_frontier_cap_used=buffer.te_bfm_frontier_cap_used,
                te_bfm_frontier_cap_dropped_edge_count=(
                    buffer.te_bfm_frontier_cap_dropped_edge_count
                ),
                bdb_stop_loss=buffer.bdb_stop_loss,
                bdb_edge_loss=buffer.bdb_edge_loss,
                bdb_base_loss=buffer.bdb_base_loss,
                bdb_stop_valid_mask=buffer.bdb_stop_valid_mask,
                bdb_edge_valid_mask=buffer.bdb_edge_valid_mask,
                bdb_base_valid_mask=buffer.bdb_base_valid_mask,
                bdb_delta_stop=buffer.bdb_delta_stop,
                bdb_delta_edge=buffer.bdb_delta_edge,
                bdb_delta_base=buffer.bdb_delta_base,
                bdb_frontier_size=buffer.bdb_frontier_size,
                bdb_parent_count=buffer.bdb_parent_count,
                bdb_log_reward=buffer.bdb_log_reward,
                bdb_log_flow=buffer.bdb_log_flow,
                budgeted_policy_kl=buffer.budgeted_policy_kl,
                budgeted_terminal_loss=buffer.budgeted_terminal_loss,
                budgeted_value_loss=buffer.budgeted_value_loss,
                budgeted_valid_mask=buffer.budgeted_valid_mask,
                oracle_v_star=buffer.oracle_v_star,
                oracle_terminal_j=buffer.oracle_terminal_j,
                oracle_stop_prob=buffer.oracle_stop_prob,
                oracle_edge_entropy=buffer.oracle_edge_entropy,
                model_stop_prob=buffer.model_stop_prob,
                budgeted_oracle_good_edge_policy_mass=(
                    buffer.budgeted_oracle_good_edge_policy_mass
                ),
                sampled_oracle_good_edge_rate=buffer.sampled_oracle_good_edge_rate,
            ),
        )


def _require_all_or_none(values: tuple[torch.Tensor | None, ...], *, name: str) -> None:
    if all(value is None for value in values):
        return
    if any(value is None for value in values):
        raise ValueError(f"{name} must be either fully present or fully absent.")


__all__ = [
    "RolloutBatch",
    "RolloutStats",
    "RolloutTraceSpec",
    "RolloutTraces",
    "StepResult",
]
