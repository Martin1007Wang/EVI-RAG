from __future__ import annotations

import torch
import pytest

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.weaver.context import GraphContext
from src.weaver.policy import PolicyOutput
from src.weaver.rollout.engine import RolloutEngine, forced_stop_rows
from src.weaver.state import Frontier, State


def logit(values: torch.Tensor) -> torch.Tensor:
    return torch.log(values / (1.0 - values))


def test_hierarchical_policy_probabilities_are_jointly_normalized() -> None:
    policy_out = PolicyOutput(
        stop_logit=torch.log(torch.tensor([2.0, 7.0], dtype=torch.float32)),
        edge_logit=torch.log(torch.tensor([3.0, 5.0, 1.0], dtype=torch.float32)),
        state_log_flow=torch.zeros(2, dtype=torch.float32),
        edge_row_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_ids=torch.tensor([4, 5, 6], dtype=torch.long),
        num_rows=2,
        num_edges=7,
    )

    assert torch.allclose(
        policy_out.stop_prob() + policy_out.edge_prob_mass(),
        torch.ones(2, dtype=torch.float32),
        atol=1.0e-5,
    )
    assert torch.allclose(
        policy_out.log_flow(),
        torch.log(torch.tensor([10.0, 8.0], dtype=torch.float32)),
    )


def test_forced_stop_rows_marks_rows_without_frontier() -> None:
    state = State(
        graph_ids=torch.zeros(2, dtype=torch.long),
        selected_edge_mask=torch.zeros((2, 8), dtype=torch.bool),
        active_node_mask=torch.zeros((2, 2), dtype=torch.bool),
        step=torch.zeros(2, dtype=torch.long),
    )
    frontier = Frontier(
        row_ids=torch.tensor([1], dtype=torch.long),
        edge_ids=torch.tensor([7], dtype=torch.long),
    )

    forced = forced_stop_rows(
        state=state,
        frontier=frontier,
        expand_budget=2,
    )

    assert forced.tolist() == [0]


def test_state_frontier_clears_edges_at_horizon() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    assert state.frontier(context, expand_budget=1).edge_ids.numel() > 0
    assert state.frontier(context, expand_budget=0).edge_ids.numel() == 0

    child = state.expand(
        graph=context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=1,
    )

    assert child.frontier(context).edge_ids.numel() > 0
    assert child.frontier(context, expand_budget=1).edge_ids.numel() == 0


def test_state_expand_rejects_actions_outside_current_frontier() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)
    state = State.initial(
        graph=context,
        graph_ids=torch.tensor([0], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="current frontier"):
        state.expand(
            graph=context,
            rows=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([1], dtype=torch.long),
            expand_budget=2,
        )

    with pytest.raises(ValueError, match="valid non-STOP"):
        state.expand(
            graph=context,
            rows=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([-1], dtype=torch.long),
            expand_budget=2,
        )

    with pytest.raises(ValueError, match="current frontier"):
        state.expand(
            graph=context,
            rows=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
            expand_budget=0,
        )


def test_policy_with_only_stop_actions_returns_unit_stop_probability() -> None:
    out = PolicyOutput(
        stop_logit=torch.zeros(2, dtype=torch.float32),
        edge_logit=torch.empty(0, dtype=torch.float32),
        state_log_flow=torch.zeros(2, dtype=torch.float32),
        edge_row_ids=torch.empty(0, dtype=torch.long),
        edge_ids=torch.empty(0, dtype=torch.long),
        num_rows=2,
        num_edges=0,
    )

    assert out.edge_ids.numel() == 0
    assert torch.allclose(out.log_flow(), torch.zeros(2, dtype=torch.float32))
    assert torch.allclose(out.stop_prob(), torch.ones(2, dtype=torch.float32))


def test_rollout_engine_produces_terminal_rows_with_flat_actions() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)
    engine = RolloutEngine(expand_budget=1)
    rollouts = engine.sample_rollouts(
        policy=_AllStopPolicy(),
        context=context,
        features=_features(),
        num_rollouts=1,
        temperature=1.0,
    )

    assert len(rollouts) == 1
    rollout = rollouts[0]
    assert rollout.terminal_mask.any().item() is True
    assert rollout.expand_mask.any().item() is False
    assert rollout.forced_stop.any().item() is False


def test_rollout_engine_marks_horizon_stop_as_forced() -> None:
    batch = _batch()
    context = GraphContext.from_batch(batch)
    policy = _CountingStopPolicy()
    engine = RolloutEngine(expand_budget=0)

    rollouts = engine.sample_rollouts(
        policy=policy,
        context=context,
        features=_features(),
        num_rollouts=1,
        temperature=1.0,
    )

    assert policy.calls == 1
    assert policy.frontier_sizes == [0]
    assert rollouts[0].stop_step.tolist() == [0]
    assert rollouts[0].forced_stop.tolist() == [True]


class _AllStopPolicy:
    def __call__(self, *, features, state: State, context: GraphContext, frontier: Frontier):
        del features, context
        return PolicyOutput(
            stop_logit=torch.full((state.num_rows,), 100.0, dtype=torch.float32),
            edge_logit=torch.zeros(frontier.edge_ids.numel(), dtype=torch.float32),
            state_log_flow=torch.zeros(state.num_rows, dtype=torch.float32),
            edge_row_ids=frontier.row_ids,
            edge_ids=frontier.edge_ids,
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )


class _CountingStopPolicy:
    def __init__(self) -> None:
        self.calls = 0
        self.frontier_sizes: list[int] = []

    def __call__(self, *, features, state: State, context: GraphContext, frontier: Frontier):
        del features, context
        self.calls += 1
        self.frontier_sizes.append(int(frontier.edge_ids.numel()))
        return PolicyOutput(
            stop_logit=torch.zeros(state.num_rows, dtype=torch.float32),
            edge_logit=torch.empty(0, dtype=torch.float32),
            state_log_flow=torch.zeros(state.num_rows, dtype=torch.float32),
            edge_row_ids=torch.empty(0, dtype=torch.long),
            edge_ids=torch.empty(0, dtype=torch.long),
            num_rows=state.num_rows,
            num_edges=state.num_edges,
        )


def _features():
    return type("Features", (), {})()


def _batch():
    data = _build_retrieval_data(
        raw={
            SampleFields.EDGE_INDEX: torch.tensor(
                [[0, 1], [1, 2]],
                dtype=torch.long,
            ),
            SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 1], dtype=torch.long),
            SampleFields.NUM_NODES: torch.tensor(3, dtype=torch.long),
            SampleFields.NUM_EDGES: torch.tensor(2, dtype=torch.long),
            SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
            SampleFields.TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 2], dtype=torch.long),
            SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 0], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.ones(3, dtype=torch.float32),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1], dtype=torch.long),
            SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(2, dtype=torch.float32),
        },
        sample_id="canonical-flat-flow",
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])
