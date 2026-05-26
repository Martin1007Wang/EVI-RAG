from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.objectives.prefix import ExpansionPrefixBatch, PrefixBatch, TerminalPrefixBatch
from src.weaver.objectives.subtb import (
    _deduplicate_ordered_states,
    build_subtb_input_from_prefix,
)
from src.weaver.policy.output import PolicyOutput
from src.weaver.reward import RewardOutput
from src.weaver.rollout.trajectory import BUDGET, POLICY_STOP, SRC_POLICY
from src.weaver.state import ActionSpace, ExpansionBatch, StateBatch


def test_ordered_state_dedup_preserves_edge_order() -> None:
    state = StateBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor(
            [
                [0, 1],
                [0, 1],
                [1, 0],
            ],
            dtype=torch.long,
        ),
        edge_count=torch.tensor([2, 2, 2]),
        budget=2,
    )

    unique, inverse = _deduplicate_ordered_states(state)

    assert unique.num_states == 2
    assert inverse.tolist() == [0, 0, 1]
    assert unique.edge_ids.tolist() == [[0, 1], [1, 0]]


def test_build_subtb_input_deduplicates_policy_and_terminal_reward() -> None:
    graph = _graph()
    target = _target()

    parent = StateBatch.initial(graph_ids=torch.tensor([0, 0]), budget=2)
    child = parent.branch(
        ExpansionBatch(
            state_ids=torch.tensor([0, 1]),
            edge_ids=torch.tensor([0, 0]),
        )
    )
    prefix = PrefixBatch(
        expansions=ExpansionPrefixBatch(
            parent=parent,
            child=child,
            edge_ids=torch.tensor([0, 0]),
            traj_ids=torch.tensor([0, 1]),
            step_ids=torch.tensor([0, 0]),
            source=torch.full((2,), SRC_POLICY, dtype=torch.long),
        ),
        terminals=TerminalPrefixBatch(
            state=child,
            traj_ids=torch.tensor([0, 1]),
            step_ids=torch.tensor([1, 1]),
            reason=torch.tensor([POLICY_STOP, BUDGET]),
            source=torch.full((2,), SRC_POLICY, dtype=torch.long),
        ),
    )
    policy = _CountingPolicy()
    reward = _CountingReward()

    out = build_subtb_input_from_prefix(
        policy=policy,
        reward_model=reward,
        features=None,
        graph_context=graph,
        target_context=target,
        prefix=prefix,
    )

    assert policy.call_count == 1
    assert policy.seen_state_counts == [2]
    assert reward.call_count == 1
    assert reward.seen_state_counts == [1]

    events = out.events
    assert events.is_terminal.tolist() == [False, False, True, True]
    assert torch.allclose(events.action_log_flow, torch.tensor([100.0, 100.0, 2.0, 11.0]))
    assert torch.allclose(events.child_state_log_flow[:2], torch.tensor([11.0, 11.0]))
    assert torch.allclose(events.terminal_log_reward[2:], torch.tensor([21.0, 21.0]))


class _CountingPolicy:
    def __init__(self) -> None:
        self.call_count = 0
        self.seen_state_counts: list[int] = []

    def __call__(self, *, features, state: StateBatch, context: GraphContext, action_space: ActionSpace) -> PolicyOutput:
        del features, context
        self.call_count += 1
        self.seen_state_counts.append(state.num_states)

        state_log_flow = 10.0 + state.edge_count.float()
        stop_log_flow = 1.0 + state.edge_count.float()
        continue_log_flow = torch.full((state.num_states,), float("-inf"))
        if action_space.num_expansions > 0:
            edge_log_flow = 100.0 + action_space.expand_edge_ids.float()
            continue_log_flow = continue_log_flow.scatter_reduce(
                0,
                action_space.expand_state_ids,
                edge_log_flow,
                reduce="amax",
                include_self=False,
            )
        else:
            edge_log_flow = torch.empty(0)

        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=stop_log_flow - state_log_flow,
            continue_log_prob=continue_log_flow - state_log_flow,
            edge_log_flow=edge_log_flow,
            edge_log_prob=edge_log_flow - state_log_flow.index_select(0, action_space.expand_state_ids),
            conditional_edge_log_prob=torch.zeros_like(edge_log_flow),
            edge_raw_score=edge_log_flow,
        )


class _CountingReward:
    def __init__(self) -> None:
        self.call_count = 0
        self.seen_state_counts: list[int] = []

    def __call__(self, *, state: StateBatch, graph_context: GraphContext, target_context: TargetContext) -> RewardOutput:
        del graph_context, target_context
        self.call_count += 1
        self.seen_state_counts.append(state.num_states)

        log_reward = 20.0 + state.edge_count.float()
        zeros = torch.zeros(state.num_states)
        return RewardOutput(
            log_reward=log_reward,
            raw_log_reward=log_reward,
            answer_count=torch.ones(state.num_states),
            target_count=torch.ones(state.num_states),
            target_recall=torch.ones(state.num_states),
            target_proximity=zeros,
            path_edge_count=zeros,
            path_edge_precision=zeros,
            edge_count=state.edge_count.float(),
            valid_mask=torch.ones(state.num_states, dtype=torch.bool),
            success_mask=torch.ones(state.num_states, dtype=torch.bool),
            metrics={},
        )


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0]),
        edge_to_graph=torch.tensor([0, 0]),
        edge_ptr=torch.tensor([0, 2]),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 2, 2]),
            edge_ids_by_src=torch.tensor([0, 1]),
            in_ptr=torch.tensor([0, 0, 1, 2]),
            edge_ids_by_dst=torch.tensor([0, 1]),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, False, True]),
        reachable_target_node_ids=torch.tensor([2]),
        reachable_target_node_ids_ptr=torch.tensor([0, 1]),
        target_count_by_graph=torch.tensor([1]),
        node_target_distance=torch.tensor([2, 1, 0]),
        shortest_path_edge_mask=torch.tensor([True, True]),
        shortest_path_edge_weight=torch.tensor([1.0, 1.0]),
    )
