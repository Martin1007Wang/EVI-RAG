from __future__ import annotations

import torch

from src.data.collate import RetrievalCollator
from src.data.schema import RetrievalData
from src.weaver.context import GraphContext
from src.weaver.objectives.prefix import ExpansionPrefixBatch, TerminalPrefixBatch
from src.weaver.objectives.subtb import (
    SubTBEventBatch,
    build_subtb_input,
    subtrajectory_terms,
    terminal_db_residual,
)
from src.weaver.policy.output import PolicyOutput
from src.weaver.reward import RewardOutput
from src.weaver.rollout.trajectory import BUDGET, NO_FRONTIER, POLICY_STOP, SRC_POLICY
from src.weaver.state import ActionSpace, StateBatch


def test_budget_exhausted_state_has_empty_frontier() -> None:
    sample = RetrievalData(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        node_entity_catalog_ids=torch.arange(2),
        edge_relation_catalog_ids=torch.arange(1),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([1]),
        reachable_target_node_ids=torch.tensor([1]),
        node_target_distance=torch.tensor([1, 0]),
    )
    sample.sample_id = "toy"

    graph = GraphContext.from_batch(RetrievalCollator()([sample]))
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=0)

    action_space = state.action_space(graph)

    assert action_space.num_expansions == 0


def test_terminal_events_separate_policy_stop_from_boundaries() -> None:
    device = torch.device("cpu")
    terminals = TerminalPrefixBatch(
        state=StateBatch.initial(graph_ids=torch.tensor([0, 0, 0]), budget=2),
        traj_ids=torch.tensor([0, 1, 2]),
        step_ids=torch.tensor([1, 2, 2]),
        reason=torch.tensor([POLICY_STOP, NO_FRONTIER, BUDGET]),
        source=torch.full((3,), SRC_POLICY, dtype=torch.long),
    )
    terminal_out = _policy_output(
        state_log_flow=torch.tensor([10.0, 20.0, 30.0]),
        stop_log_flow=torch.tensor([1.0, 2.0, 3.0]),
    )
    reward = _reward_output(
        log_reward=torch.tensor([0.5, 1.5, 2.5]),
        valid_mask=torch.ones(3, dtype=torch.bool),
    )

    events = build_subtb_input(
        parent_out=_empty_policy_output(device),
        child_out=_empty_policy_output(device),
        terminal_out=terminal_out,
        terminal_reward=reward,
        backward_log_prob=torch.empty(0),
        expansions=_empty_expansions(device=device, budget=2),
        terminals=terminals,
    ).events

    assert events.terminal_reason.tolist() == [POLICY_STOP, NO_FRONTIER, BUDGET]
    assert events.is_terminal.tolist() == [True, True, True]
    assert torch.allclose(events.action_log_flow, torch.tensor([1.0, 20.0, 30.0]))
    assert torch.allclose(events.action_log_prob, torch.tensor([-9.0, 0.0, 0.0]))
    assert torch.allclose(terminal_db_residual(events), torch.tensor([0.5, 18.5, 27.5]))


def test_subtrajectory_terms_preserve_terminal_tail_reason() -> None:
    events = SubTBEventBatch(
        trajectory_ids=torch.tensor([0, 0]),
        step_ids=torch.tensor([0, 1]),
        source_ids=torch.tensor([SRC_POLICY, SRC_POLICY]),
        parent_state_log_flow=torch.tensor([5.0, 11.0]),
        child_state_log_flow=torch.tensor([3.0, 0.0]),
        action_log_flow=torch.tensor([7.0, 11.0]),
        backward_log_prob=torch.tensor([0.0, 0.0]),
        terminal_log_reward=torch.tensor([0.0, 4.0]),
        terminal_reason=torch.tensor([-1, BUDGET]),
        is_terminal=torch.tensor([False, True]),
    )

    terms = subtrajectory_terms(events, subtb_lambda=1.0, max_len=None)

    assert terms.terminal_reason.tolist() == [-1, BUDGET, BUDGET]
    assert terms.is_terminal.tolist() == [False, True, True]
    assert torch.allclose(terms.residual, torch.tensor([4.0, 7.0, 3.0]))


def _empty_expansions(*, device: torch.device, budget: int) -> ExpansionPrefixBatch:
    state = StateBatch.initial(graph_ids=torch.empty(0, dtype=torch.long, device=device), budget=budget)
    empty_long = torch.empty(0, dtype=torch.long, device=device)
    return ExpansionPrefixBatch(
        parent=state,
        child=state,
        edge_ids=empty_long,
        traj_ids=empty_long,
        step_ids=empty_long,
        source=empty_long,
    )


def _policy_output(*, state_log_flow: torch.Tensor, stop_log_flow: torch.Tensor) -> PolicyOutput:
    n = int(state_log_flow.numel())
    empty = torch.empty(0, dtype=torch.float32)
    return PolicyOutput(
        action_space=ActionSpace.empty(num_states=n, device=state_log_flow.device),
        state_log_flow=state_log_flow,
        stop_log_flow=stop_log_flow,
        continue_log_flow=torch.full((n,), float("-inf")),
        stop_log_prob=stop_log_flow - state_log_flow,
        continue_log_prob=torch.full((n,), float("-inf")),
        edge_log_flow=empty,
        edge_log_prob=empty,
        conditional_edge_log_prob=empty,
        edge_raw_score=empty,
    )


def _empty_policy_output(device: torch.device) -> PolicyOutput:
    empty = torch.empty(0, dtype=torch.float32, device=device)
    return _policy_output(state_log_flow=empty, stop_log_flow=empty)


def _reward_output(*, log_reward: torch.Tensor, valid_mask: torch.Tensor) -> RewardOutput:
    zeros = torch.zeros_like(log_reward)
    return RewardOutput(
        log_reward=log_reward,
        raw_log_reward=log_reward,
        answer_count=zeros,
        target_count=torch.ones_like(log_reward),
        target_recall=zeros,
        target_proximity=zeros,
        path_edge_count=zeros,
        path_edge_precision=zeros,
        edge_count=zeros,
        valid_mask=valid_mask,
        success_mask=torch.zeros_like(valid_mask),
        metrics={},
    )
