from __future__ import annotations

from dataclasses import fields

import torch

from src.data.schema import RetrievalBatch

from .schema import RolloutBatch, RolloutStats, RolloutTraces


def split_static_rollout_batch(
    *,
    rollout: RolloutBatch,
    original_batch: RetrievalBatch,
    repeats: int,
) -> list[RolloutBatch]:
    """
    Split fused static-batch rollouts back into logical rollout batches.

    Required layout:

        rollout_row = repeat_id * B + graph_id
        selected_edge_id = original-batch edge id

    Unlike physical repeat, edge ids are already in original coordinates.
    """
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")

    num_graphs = _num_graphs(original_batch)
    expected_graphs = num_graphs * repeats
    _validate_rollout_first_dim(
        rollout=rollout,
        expected_graphs=expected_graphs,
    )

    return [
        _slice_rollout_batch(
            rollout=rollout,
            graph_slice=slice(i * num_graphs, (i + 1) * num_graphs),
        )
        for i in range(repeats)
    ]


def _slice_rollout_batch(
    *,
    rollout: RolloutBatch,
    graph_slice: slice,
) -> RolloutBatch:
    return RolloutBatch(
        stats=_slice_stats(rollout.stats, graph_slice),
        traces=_slice_traces(
            traces=rollout.traces,
            graph_slice=graph_slice,
        ),
    )


def _slice_stats(
    stats: RolloutStats,
    graph_slice: slice,
) -> RolloutStats:
    return RolloutStats(
        trajectory_length=stats.trajectory_length[graph_slice],
        terminal_log_reward=stats.terminal_log_reward[graph_slice],
        terminal_answer_f1=stats.terminal_answer_f1[graph_slice],
        edge_action_entropy=stats.edge_action_entropy[graph_slice],
        edge_action_count=stats.edge_action_count[graph_slice],
        source_graph_id=_slice_optional(stats.source_graph_id, graph_slice),
        terminal_complexity_penalty=_slice_optional(
            stats.terminal_complexity_penalty,
            graph_slice,
        ),
        terminal_base_log_reward=_slice_optional(
            stats.terminal_base_log_reward,
            graph_slice,
        ),
        terminal_utility=_slice_optional(stats.terminal_utility, graph_slice),
        terminal_shortest_path_potential=_slice_optional(
            stats.terminal_shortest_path_potential,
            graph_slice,
        ),
        terminal_expanded_edge_count=_slice_optional(
            stats.terminal_expanded_edge_count,
            graph_slice,
        ),
        terminal_answer_degree_excess=_slice_optional(
            stats.terminal_answer_degree_excess,
            graph_slice,
        ),
    )


def _slice_traces(
    *,
    traces: RolloutTraces,
    graph_slice: slice,
) -> RolloutTraces:
    return RolloutTraces(
        log_pf=traces.log_pf[graph_slice],
        log_pb=traces.log_pb[graph_slice],
        state_log_flow=traces.state_log_flow[graph_slice],
        db_parent_log_reward=traces.db_parent_log_reward[graph_slice],
        db_child_log_reward=traces.db_child_log_reward[graph_slice],
        db_parent_shortest_path_potential=traces.db_parent_shortest_path_potential[
            graph_slice
        ],
        db_child_shortest_path_potential=traces.db_child_shortest_path_potential[
            graph_slice
        ],
        db_parent_process_log_bonus=traces.db_parent_process_log_bonus[graph_slice],
        db_child_process_log_bonus=traces.db_child_process_log_bonus[graph_slice],
        db_log_p_stop_parent=traces.db_log_p_stop_parent[graph_slice],
        db_log_p_stop_child=traces.db_log_p_stop_child[graph_slice],
        db_log_pf_expand=traces.db_log_pf_expand[graph_slice],
        db_log_pb=traces.db_log_pb[graph_slice],
        db_valid_mask=traces.db_valid_mask[graph_slice],
        action_type=traces.action_type[graph_slice],
        continue_mask=traces.continue_mask[graph_slice],
        stop_mask=traces.stop_mask[graph_slice],
        selected_edge_ids=traces.selected_edge_ids[graph_slice],
        target_stop_prob=traces.target_stop_prob[graph_slice],
        target_continue_prob=traces.target_continue_prob[graph_slice],
        policy_action_valid_mask=traces.policy_action_valid_mask[graph_slice],
        edge_action_entropy=traces.edge_action_entropy[graph_slice],
        edge_action_entropy_valid_mask=traces.edge_action_entropy_valid_mask[
            graph_slice
        ],
        log_p_stop=_slice_optional(
            traces.log_p_stop,
            graph_slice,
        ),
        stop_now_log_reward=_slice_optional(
            traces.stop_now_log_reward,
            graph_slice,
        ),
        stop_now_answer_f1=_slice_optional(
            traces.stop_now_answer_f1,
            graph_slice,
        ),
        stop_now_valid_mask=_slice_optional(
            traces.stop_now_valid_mask,
            graph_slice,
        ),
        budget_exhausted_mask=_slice_optional(
            traces.budget_exhausted_mask,
            graph_slice,
        ),
        te_bfm_loss=_slice_optional(
            traces.te_bfm_loss,
            graph_slice,
        ),
        te_bfm_valid_mask=_slice_optional(
            traces.te_bfm_valid_mask,
            graph_slice,
        ),
        te_bfm_residual_abs=_slice_optional(
            traces.te_bfm_residual_abs,
            graph_slice,
        ),
        te_bfm_target_log_value=_slice_optional(
            traces.te_bfm_target_log_value,
            graph_slice,
        ),
        te_bfm_log_reward=_slice_optional(
            traces.te_bfm_log_reward,
            graph_slice,
        ),
        te_bfm_stop_prob=_slice_optional(
            traces.te_bfm_stop_prob,
            graph_slice,
        ),
        te_bfm_frontier_edge_count=_slice_optional(
            traces.te_bfm_frontier_edge_count,
            graph_slice,
        ),
        te_bfm_counterfactual_child_loss=_slice_optional(
            traces.te_bfm_counterfactual_child_loss,
            graph_slice,
        ),
        te_bfm_frontier_cap_used=_slice_optional(
            traces.te_bfm_frontier_cap_used,
            graph_slice,
        ),
        te_bfm_frontier_cap_dropped_edge_count=_slice_optional(
            traces.te_bfm_frontier_cap_dropped_edge_count,
            graph_slice,
        ),
        bdb_stop_loss=_slice_optional(traces.bdb_stop_loss, graph_slice),
        bdb_edge_loss=_slice_optional(traces.bdb_edge_loss, graph_slice),
        bdb_base_loss=_slice_optional(traces.bdb_base_loss, graph_slice),
        bdb_stop_valid_mask=_slice_optional(
            traces.bdb_stop_valid_mask,
            graph_slice,
        ),
        bdb_edge_valid_mask=_slice_optional(
            traces.bdb_edge_valid_mask,
            graph_slice,
        ),
        bdb_base_valid_mask=_slice_optional(
            traces.bdb_base_valid_mask,
            graph_slice,
        ),
        bdb_delta_stop=_slice_optional(traces.bdb_delta_stop, graph_slice),
        bdb_delta_edge=_slice_optional(traces.bdb_delta_edge, graph_slice),
        bdb_delta_base=_slice_optional(traces.bdb_delta_base, graph_slice),
        bdb_frontier_size=_slice_optional(traces.bdb_frontier_size, graph_slice),
        bdb_parent_count=_slice_optional(traces.bdb_parent_count, graph_slice),
        bdb_log_reward=_slice_optional(traces.bdb_log_reward, graph_slice),
        bdb_log_flow=_slice_optional(traces.bdb_log_flow, graph_slice),
        budgeted_policy_kl=_slice_optional(traces.budgeted_policy_kl, graph_slice),
        budgeted_terminal_loss=_slice_optional(
            traces.budgeted_terminal_loss,
            graph_slice,
        ),
        budgeted_value_loss=_slice_optional(traces.budgeted_value_loss, graph_slice),
        budgeted_valid_mask=_slice_optional(traces.budgeted_valid_mask, graph_slice),
        oracle_v_star=_slice_optional(traces.oracle_v_star, graph_slice),
        oracle_terminal_j=_slice_optional(traces.oracle_terminal_j, graph_slice),
        oracle_stop_prob=_slice_optional(traces.oracle_stop_prob, graph_slice),
        oracle_edge_entropy=_slice_optional(traces.oracle_edge_entropy, graph_slice),
        model_stop_prob=_slice_optional(traces.model_stop_prob, graph_slice),
        budgeted_oracle_good_edge_policy_mass=_slice_optional(
            traces.budgeted_oracle_good_edge_policy_mass,
            graph_slice,
        ),
        sampled_oracle_good_edge_rate=_slice_optional(
            traces.sampled_oracle_good_edge_rate,
            graph_slice,
        ),
    )


def _slice_optional(
    tensor: torch.Tensor | None,
    graph_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[graph_slice]


def _validate_rollout_first_dim(
    *,
    rollout: RolloutBatch,
    expected_graphs: int,
) -> None:
    expected_graphs = int(expected_graphs)

    for field in fields(RolloutStats):
        value = getattr(rollout.stats, field.name)
        if value is not None:
            _validate_first_dim(value, expected_graphs, f"stats.{field.name}")

    for field in fields(RolloutTraces):
        value = getattr(rollout.traces, field.name)
        if value is not None:
            _validate_first_dim(value, expected_graphs, f"traces.{field.name}")


def _validate_first_dim(
    tensor: torch.Tensor,
    expected: int,
    name: str,
) -> None:
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")

    actual = int(tensor.size(0))
    if actual != int(expected):
        raise ValueError(
            f"{name} first dimension mismatch: expected {expected}, got {actual}."
        )


def _num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.num_graphs)


__all__ = ["split_static_rollout_batch"]
