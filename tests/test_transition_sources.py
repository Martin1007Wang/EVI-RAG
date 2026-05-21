from __future__ import annotations

import torch

from src.weaver.objectives import source_rollout_metrics
from src.weaver.state import Frontier, State
from src.weaver.transition import (
    ExpansionBatch,
    SampleMeta,
    TerminalBatch,
    TrainingBatch,
    SRC_UNKNOWN,
    SRC_POLICY,
    SRC_REPLAY,
)


def test_concat_reindex_keeps_policy_and_replay_trajectories_distinct() -> None:
    policy = _transition_batch(
        trajectory_ids=torch.tensor([0, 0, 1]),
        action_edge_ids=torch.tensor([0, -1, -1]),
    ).with_source_id(SRC_POLICY)
    replay = _transition_batch(
        trajectory_ids=torch.tensor([0, 0]),
        action_edge_ids=torch.tensor([0, -1]),
    ).with_source_id(SRC_REPLAY)

    out = TrainingBatch.concat_reindex_trajectories([policy, replay])

    assert torch.equal(
        out.expansions.meta.trajectory_ids,
        torch.tensor([0, 2]),
    )
    assert torch.equal(
        out.expansions.meta.source_ids,
        torch.tensor(
            [
                SRC_POLICY,
                SRC_REPLAY,
            ]
        ),
    )
    assert torch.equal(
        out.terminals.meta.trajectory_ids,
        torch.tensor([0, 1, 2]),
    )
    assert torch.equal(
        out.terminals.meta.source_ids,
        torch.tensor(
            [
                SRC_POLICY,
                SRC_POLICY,
                SRC_REPLAY,
            ]
        ),
    )


def test_source_rollout_metrics_split_policy_stop_and_structural_stop() -> None:
    metrics = source_rollout_metrics(
        source_ids=torch.tensor(
            [
                SRC_POLICY,
                SRC_POLICY,
                SRC_REPLAY,
                SRC_POLICY,
            ]
        ),
        action_edge_ids=torch.tensor([0, -1, -1, -1]),
        step_ids=torch.tensor([0, 1, 1, 2]),
        parent_frontier=Frontier(
            row_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        num_rows=4,
    )

    assert torch.isclose(metrics["rollout/stop_rate"], torch.tensor(2.0 / 3.0))
    assert torch.isclose(metrics["rollout/structural_stop_rate"], torch.tensor(1.0))
    assert torch.isclose(metrics["rollout/forced_stop_rate"], torch.tensor(0.0))
    assert torch.isclose(metrics["rollout/mean_depth"], torch.tensor(1.5))


def _transition_batch(
    *,
    trajectory_ids: torch.Tensor,
    action_edge_ids: torch.Tensor,
) -> TrainingBatch:
    num_rows = int(action_edge_ids.numel())
    state = _state(num_rows=num_rows)
    expand_mask = action_edge_ids.ge(0)
    stop_mask = ~expand_mask
    return TrainingBatch(
        expansions=ExpansionBatch(
            parent=state,
            child=state.clone(),
            edge_ids=action_edge_ids[expand_mask],
            meta=SampleMeta(
                trajectory_ids=trajectory_ids[expand_mask],
                step_ids=torch.arange(num_rows, dtype=torch.long)[expand_mask],
                source_ids=torch.full(
                    (int(expand_mask.sum().item()),),
                    SRC_UNKNOWN,
                    dtype=torch.long,
                ),
            ),
        ),
        terminals=TerminalBatch(
            state=state.select_rows(stop_mask.nonzero(as_tuple=False).flatten()),
            meta=SampleMeta(
                trajectory_ids=trajectory_ids[stop_mask],
                step_ids=torch.arange(num_rows, dtype=torch.long)[stop_mask],
                source_ids=torch.full(
                    (int(stop_mask.sum().item()),),
                    SRC_UNKNOWN,
                    dtype=torch.long,
                ),
            ),
        ),
    )


def _state(
    *,
    num_rows: int,
) -> State:
    return State(
        graph_ids=torch.zeros(num_rows, dtype=torch.long),
        selected_edge_mask=torch.zeros((num_rows, 1), dtype=torch.bool),
        active_node_mask=torch.zeros((num_rows, 1), dtype=torch.bool),
        step=torch.zeros(num_rows, dtype=torch.long),
    )
