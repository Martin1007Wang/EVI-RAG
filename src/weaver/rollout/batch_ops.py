from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch

from .schema import RolloutBatch, RolloutStats, RolloutTraces


def split_repeated_rollout_batch(
    *,
    rollout: RolloutBatch,
    original_batch: RetrievalBatch,
    repeats: int,
) -> list[RolloutBatch]:
    """
    Split rollout over a repeated RetrievalBatch back into logical rollout batches.

    Requires repeat_retrieval_batch layout:

        repeated_graph_id = repeat_id * B + graph_id
        repeated_edge_id  = repeat_id * E + original_edge_id

    After splitting, selected_edge_ids are mapped back to original-batch edge ids.
    STOP actions remain -1.
    """

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")

    num_graphs = _num_graphs(original_batch)
    num_edges = _num_edges(original_batch)
    expected_graphs = num_graphs * repeats

    _validate_rollout_first_dim(
        rollout=rollout,
        expected_graphs=expected_graphs,
    )

    return [
        _slice_rollout_batch(
            rollout=rollout,
            graph_slice=slice(i * num_graphs, (i + 1) * num_graphs),
            edge_offset=i * num_edges,
        )
        for i in range(repeats)
    ]


def _slice_rollout_batch(
    *,
    rollout: RolloutBatch,
    graph_slice: slice,
    edge_offset: int,
) -> RolloutBatch:
    return RolloutBatch(
        stats=_slice_stats(rollout.stats, graph_slice),
        traces=_slice_traces(
            traces=rollout.traces,
            graph_slice=graph_slice,
            edge_offset=edge_offset,
        ),
    )


def _slice_stats(
    stats: RolloutStats,
    graph_slice: slice,
) -> RolloutStats:
    return RolloutStats(
        root_log_z=stats.root_log_z[graph_slice],
        trajectory_length=stats.trajectory_length[graph_slice],
        terminal_log_reward=stats.terminal_log_reward[graph_slice],
        terminal_answer_f1=stats.terminal_answer_f1[graph_slice],
        proposal_intervention_count=stats.proposal_intervention_count[graph_slice],
        edge_action_entropy=stats.edge_action_entropy[graph_slice],
        edge_action_entropy_valid_mask=stats.edge_action_entropy_valid_mask[
            graph_slice
        ],
        terminal_edge_penalty=_slice_optional_stat(
            stats.terminal_edge_penalty,
            graph_slice,
        ),
        terminal_base_log_reward=_slice_optional_stat(
            stats.terminal_base_log_reward,
            graph_slice,
        ),
        terminal_utility=_slice_optional_stat(stats.terminal_utility, graph_slice),
        terminal_expanded_edge_count=_slice_optional_stat(
            stats.terminal_expanded_edge_count,
            graph_slice,
        ),
        terminal_minimal_edge_count=_slice_optional_stat(
            stats.terminal_minimal_edge_count,
            graph_slice,
        ),
        terminal_minimality_gap=_slice_optional_stat(
            stats.terminal_minimality_gap,
            graph_slice,
        ),
        terminal_minimality_penalty=_slice_optional_stat(
            stats.terminal_minimality_penalty,
            graph_slice,
        ),
    )


def _slice_traces(
    *,
    traces: RolloutTraces,
    graph_slice: slice,
    edge_offset: int,
) -> RolloutTraces:
    return RolloutTraces(
        state_log_flows=traces.state_log_flows[graph_slice],
        log_pf=traces.log_pf[graph_slice],
        log_pb=traces.log_pb[graph_slice],
        action_type=traces.action_type[graph_slice],
        continue_mask=traces.continue_mask[graph_slice],
        stop_mask=traces.stop_mask[graph_slice],
        selected_edge_ids=_shift_selected_edge_ids(
            traces.selected_edge_ids[graph_slice],
            edge_offset=edge_offset,
        ),
        stop_now_log_reward=traces.stop_now_log_reward[graph_slice],
        stop_now_answer_f1=traces.stop_now_answer_f1[graph_slice],
        stop_now_valid_mask=traces.stop_now_valid_mask[graph_slice],
        stop_log_pf=traces.stop_log_pf[graph_slice],
        stop_tb_valid_mask=traces.stop_tb_valid_mask[graph_slice],
        target_stop_prob=traces.target_stop_prob[graph_slice],
        target_continue_prob=traces.target_continue_prob[graph_slice],
        policy_action_valid_mask=traces.policy_action_valid_mask[graph_slice],
        edge_action_entropy=traces.edge_action_entropy[graph_slice],
        edge_action_entropy_valid_mask=traces.edge_action_entropy_valid_mask[
            graph_slice
        ],
        advantage_aux_loss=traces.advantage_aux_loss[graph_slice],
        advantage_aux_valid_mask=traces.advantage_aux_valid_mask[graph_slice],
        proposal_intervention_mask=traces.proposal_intervention_mask[graph_slice],
        budget_exhausted_mask=(
            None
            if traces.budget_exhausted_mask is None
            else traces.budget_exhausted_mask[graph_slice]
        ),
    )


def _shift_selected_edge_ids(
    selected_edge_ids: torch.Tensor,
    *,
    edge_offset: int,
) -> torch.Tensor:
    if edge_offset == 0:
        return selected_edge_ids

    shifted = selected_edge_ids.clone()
    valid = shifted.ge(0)
    shifted[valid] -= int(edge_offset)
    return shifted


def _validate_rollout_first_dim(
    *,
    rollout: RolloutBatch,
    expected_graphs: int,
) -> None:
    checks = {
        "stats.root_log_z": rollout.stats.root_log_z,
        "stats.trajectory_length": rollout.stats.trajectory_length,
        "stats.terminal_log_reward": rollout.stats.terminal_log_reward,
        "stats.terminal_answer_f1": rollout.stats.terminal_answer_f1,
        "stats.proposal_intervention_count": rollout.stats.proposal_intervention_count,
        "stats.edge_action_entropy": rollout.stats.edge_action_entropy,
        "stats.edge_action_entropy_valid_mask": rollout.stats.edge_action_entropy_valid_mask,
        "traces.state_log_flows": rollout.traces.state_log_flows,
        "traces.log_pf": rollout.traces.log_pf,
        "traces.log_pb": rollout.traces.log_pb,
        "traces.action_type": rollout.traces.action_type,
        "traces.continue_mask": rollout.traces.continue_mask,
        "traces.stop_mask": rollout.traces.stop_mask,
        "traces.selected_edge_ids": rollout.traces.selected_edge_ids,
        "traces.stop_now_log_reward": rollout.traces.stop_now_log_reward,
        "traces.stop_now_answer_f1": rollout.traces.stop_now_answer_f1,
        "traces.stop_now_valid_mask": rollout.traces.stop_now_valid_mask,
        "traces.stop_log_pf": rollout.traces.stop_log_pf,
        "traces.stop_tb_valid_mask": rollout.traces.stop_tb_valid_mask,
        "traces.target_stop_prob": rollout.traces.target_stop_prob,
        "traces.target_continue_prob": rollout.traces.target_continue_prob,
        "traces.policy_action_valid_mask": rollout.traces.policy_action_valid_mask,
        "traces.edge_action_entropy": rollout.traces.edge_action_entropy,
        "traces.edge_action_entropy_valid_mask": (
            rollout.traces.edge_action_entropy_valid_mask
        ),
        "traces.proposal_intervention_mask": rollout.traces.proposal_intervention_mask,
    }
    if rollout.traces.budget_exhausted_mask is not None:
        checks["traces.budget_exhausted_mask"] = rollout.traces.budget_exhausted_mask
    if rollout.stats.terminal_edge_penalty is not None:
        checks["stats.terminal_edge_penalty"] = rollout.stats.terminal_edge_penalty
    if rollout.stats.terminal_base_log_reward is not None:
        checks[
            "stats.terminal_base_log_reward"
        ] = rollout.stats.terminal_base_log_reward
    if rollout.stats.terminal_utility is not None:
        checks["stats.terminal_utility"] = rollout.stats.terminal_utility
    if rollout.stats.terminal_expanded_edge_count is not None:
        checks[
            "stats.terminal_expanded_edge_count"
        ] = rollout.stats.terminal_expanded_edge_count
    if rollout.stats.terminal_minimal_edge_count is not None:
        checks[
            "stats.terminal_minimal_edge_count"
        ] = rollout.stats.terminal_minimal_edge_count
    if rollout.stats.terminal_minimality_gap is not None:
        checks["stats.terminal_minimality_gap"] = rollout.stats.terminal_minimality_gap
    if rollout.stats.terminal_minimality_penalty is not None:
        checks[
            "stats.terminal_minimality_penalty"
        ] = rollout.stats.terminal_minimality_penalty

    for name, tensor in checks.items():
        _validate_first_dim(tensor, expected_graphs, name)


def _slice_optional_stat(
    tensor: torch.Tensor | None,
    graph_slice: slice,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[graph_slice]


def _validate_first_dim(
    tensor: torch.Tensor,
    expected: int,
    name: str,
) -> None:
    if tensor.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")

    actual = int(tensor.size(0))
    expected = int(expected)

    if actual != expected:
        raise ValueError(
            f"{name} first dimension mismatch: expected {expected}, got {actual}."
        )


def _num_graphs(batch: RetrievalBatch) -> int:
    return int(batch.ptr.numel()) - 1


def _num_edges(batch: RetrievalBatch) -> int:
    return int(batch.edge_index.size(1))


__all__ = ["split_repeated_rollout_batch"]
