from __future__ import annotations

from dataclasses import fields

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
    Split rollouts from a repeated RetrievalBatch back into logical rollout batches.

    Required repeat layout:

        repeated_graph_id = repeat_id * B + graph_id
        repeated_edge_id  = repeat_id * E + original_edge_id

    After splitting, selected_edge_ids are mapped back to original-batch edge ids.
    STOP / non-expand actions remain -1.
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
            edge_offset=0,
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
        edge_action_entropy=stats.edge_action_entropy[graph_slice],
        edge_action_count=stats.edge_action_count[graph_slice],
        terminal_complexity_penalty=_slice_optional(
            stats.terminal_complexity_penalty,
            graph_slice,
        ),
        terminal_base_log_reward=_slice_optional(
            stats.terminal_base_log_reward,
            graph_slice,
        ),
        terminal_utility=_slice_optional(stats.terminal_utility, graph_slice),
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
    edge_offset: int,
) -> RolloutTraces:
    return RolloutTraces(
        state_log_flows=traces.state_log_flows[graph_slice],
        log_pf=traces.log_pf[graph_slice],
        log_pb=traces.log_pb[graph_slice],
        action_type=traces.action_type[graph_slice],
        continue_mask=traces.continue_mask[graph_slice],
        stop_mask=traces.stop_mask[graph_slice],
        selected_edge_ids=_unrepeat_edge_ids(
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
        budget_exhausted_mask=_slice_optional(
            traces.budget_exhausted_mask,
            graph_slice,
        ),
        stop_adv_target=_slice_optional(
            traces.stop_adv_target,
            graph_slice,
        ),
        stop_adv_valid_mask=_slice_optional(
            traces.stop_adv_valid_mask,
            graph_slice,
        ),
        stop_adv_continue_log_reward=_slice_optional(
            traces.stop_adv_continue_log_reward,
            graph_slice,
        ),
    )


def _unrepeat_edge_ids(
    selected_edge_ids: torch.Tensor,
    *,
    edge_offset: int,
) -> torch.Tensor:
    """
    Map repeated-batch edge ids back to original-batch edge ids.

    Non-expand actions use negative ids and are left unchanged.
    """
    edge_offset = int(edge_offset)
    if edge_offset == 0:
        return selected_edge_ids

    out = selected_edge_ids.clone()
    valid = out.ge(0)
    out[valid] -= edge_offset
    return out


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


def _num_edges(batch: RetrievalBatch) -> int:
    return int(batch.edge_index.size(1))


__all__ = ["split_repeated_rollout_batch", "split_static_rollout_batch"]
