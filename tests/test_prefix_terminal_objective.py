from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.objectives.prefix import ExpansionPrefixBatch, PrefixBatch, TerminalPrefixBatch
from src.weaver.objectives.subtb import prefix_terminal_objective
from src.weaver.policy.output import PolicyOutput
from src.weaver.reward import RewardOutput
from src.weaver.rollout.trajectory import POLICY_STOP, SRC_POLICY
from src.weaver.state import ActionSpace, StateBatch


def test_prefix_terminal_objective_calibrates_stop_and_margin() -> None:
    graph = _graph()
    target = _target()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=1)
    prefix = PrefixBatch(
        expansions=_empty_expansions(budget=1),
        terminals=TerminalPrefixBatch(
            state=state,
            traj_ids=torch.tensor([0]),
            step_ids=torch.tensor([0]),
            reason=torch.tensor([POLICY_STOP]),
            source=torch.tensor([SRC_POLICY]),
        ),
    )

    out = prefix_terminal_objective(
        policy=_FakePolicy(stop_log_flow=torch.tensor([1.0]), edge_log_flow=torch.tensor([3.0])),
        reward_model=_FakeReward(log_reward=torch.tensor([2.0]), target_recall=torch.tensor([1.0])),
        features=None,
        graph_context=graph,
        target_context=target,
        prefix=prefix,
        prefix_calibration_weight=2.0,
        prefix_calibration_huber_delta=2.0,
        sufficient_stop_margin_weight=3.0,
        sufficient_stop_margin=0.0,
        sufficient_recall_threshold=1.0,
    )

    assert out is not None
    assert torch.allclose(out.metrics["prefix_calib/loss"], torch.tensor(0.5))
    assert torch.allclose(out.metrics["sufficient/stop_margin_loss"], torch.tensor(2.0))
    assert torch.allclose(out.loss, torch.tensor(7.0))
    assert torch.allclose(out.metrics["sufficient/stop_margin_mean"], torch.tensor(-2.0))


def test_prefix_terminal_objective_skips_margin_for_insufficient_prefix() -> None:
    graph = _graph()
    target = _target()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=1)
    prefix = PrefixBatch(
        expansions=_empty_expansions(budget=1),
        terminals=TerminalPrefixBatch(
            state=state,
            traj_ids=torch.tensor([0]),
            step_ids=torch.tensor([0]),
            reason=torch.tensor([POLICY_STOP]),
            source=torch.tensor([SRC_POLICY]),
        ),
    )

    out = prefix_terminal_objective(
        policy=_FakePolicy(stop_log_flow=torch.tensor([1.0]), edge_log_flow=torch.tensor([3.0])),
        reward_model=_FakeReward(log_reward=torch.tensor([2.0]), target_recall=torch.tensor([0.5])),
        features=None,
        graph_context=graph,
        target_context=target,
        prefix=prefix,
        prefix_calibration_weight=0.0,
        prefix_calibration_huber_delta=2.0,
        sufficient_stop_margin_weight=1.0,
        sufficient_stop_margin=0.0,
        sufficient_recall_threshold=1.0,
    )

    assert out is not None
    assert torch.allclose(out.loss, torch.tensor(0.0))
    assert torch.allclose(out.metrics["sufficient/active_fraction"], torch.tensor(0.0))


class _FakePolicy:
    def __init__(self, *, stop_log_flow: torch.Tensor, edge_log_flow: torch.Tensor) -> None:
        self.stop_log_flow = stop_log_flow.float()
        self.edge_log_flow = edge_log_flow.float()

    def __call__(self, *, features, state: StateBatch, context: GraphContext, action_space: ActionSpace) -> PolicyOutput:
        del features, context
        continue_log_flow = torch.full((state.num_states,), float("-inf"))
        if action_space.num_expansions > 0:
            continue_log_flow = torch.zeros(state.num_states).scatter_reduce(
                0,
                action_space.expand_state_ids,
                self.edge_log_flow,
                reduce="amax",
                include_self=False,
            )
        state_log_flow = torch.logaddexp(self.stop_log_flow, continue_log_flow)
        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=self.stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=self.stop_log_flow - state_log_flow,
            continue_log_prob=continue_log_flow - state_log_flow,
            edge_log_flow=self.edge_log_flow,
            edge_log_prob=self.edge_log_flow - state_log_flow.index_select(0, action_space.expand_state_ids),
            conditional_edge_log_prob=torch.zeros_like(self.edge_log_flow),
            edge_raw_score=self.edge_log_flow,
        )


class _FakeReward:
    def __init__(self, *, log_reward: torch.Tensor, target_recall: torch.Tensor) -> None:
        self.log_reward = log_reward.float()
        self.target_recall = target_recall.float()

    def __call__(self, *, state: StateBatch, graph_context: GraphContext, target_context: TargetContext) -> RewardOutput:
        del graph_context, target_context
        zeros = torch.zeros(state.num_states)
        return RewardOutput(
            log_reward=self.log_reward,
            raw_log_reward=self.log_reward,
            answer_count=self.target_recall,
            target_count=torch.ones(state.num_states),
            target_recall=self.target_recall,
            target_proximity=zeros,
            path_edge_count=zeros,
            path_edge_precision=zeros,
            edge_count=state.edge_count.float(),
            valid_mask=torch.ones(state.num_states, dtype=torch.bool),
            success_mask=self.target_recall.gt(0),
            metrics={},
        )


def _empty_expansions(*, budget: int) -> ExpansionPrefixBatch:
    state = StateBatch.initial(graph_ids=torch.empty(0, dtype=torch.long), budget=budget)
    empty = torch.empty(0, dtype=torch.long)
    return ExpansionPrefixBatch(
        parent=state,
        child=state,
        edge_ids=empty,
        traj_ids=empty,
        step_ids=empty,
        source=empty,
    )


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0]),
        edge_to_graph=torch.tensor([0]),
        edge_ptr=torch.tensor([0, 1]),
        anchor_mask=torch.tensor([True, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 1]),
            edge_ids_by_src=torch.tensor([0]),
            in_ptr=torch.tensor([0, 0, 1]),
            edge_ids_by_dst=torch.tensor([0]),
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, True]),
        reachable_target_node_ids=torch.tensor([1]),
        reachable_target_node_ids_ptr=torch.tensor([0, 1]),
        target_count_by_graph=torch.tensor([1]),
        node_target_distance=torch.tensor([1, 0]),
        shortest_path_edge_mask=torch.tensor([True]),
        shortest_path_edge_weight=torch.tensor([1.0]),
    )
