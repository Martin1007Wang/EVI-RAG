from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.objectives import (
    EdgeFlowMatchingObjective,
    NonterminalTransitionBatch,
    TerminalTransitionBatch,
    build_edge_flow_matching_batches_from_trajectories,
    terminal_edge_reward_matching,
)
from src.weaver.objectives.transition_builder import deduplicate_state_batch
from src.weaver.policy.output import STOP_EDGE_ID, PolicyOutput
from src.weaver.reward import RewardOutput
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch
from src.weaver.state import ActionSpace, StateBatch


def test_build_edge_flow_matching_batches_from_trajectories_reconstructs_prefix_states() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor(
            [
                [1, 0],
                [-1, -1],
            ],
            dtype=torch.long,
        ),
        edge_logp=torch.zeros((2, 2), dtype=torch.float32),
        edge_count=torch.tensor([2, 0], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(2, dtype=torch.float32),
        source=torch.zeros(2, dtype=torch.bool),
    )

    nonterminal, terminal = build_edge_flow_matching_batches_from_trajectories(
        trajectories=trajectories,
    )

    assert nonterminal is not None
    assert terminal is not None
    assert nonterminal.edge_ids.tolist() == [1, 0]
    assert nonterminal.parent_state.edge_count.tolist() == [0, 1]
    assert nonterminal.parent_state.edge_ids.tolist() == [[-1, -1], [1, -1]]
    assert terminal.state.edge_count.tolist() == [0, 1, 2, 0]
    assert terminal.state.edge_ids.tolist() == [
        [-1, -1],
        [1, -1],
        [0, 1],
        [-1, -1],
    ]


def test_build_edge_flow_matching_batches_from_zero_step_trajectory_is_terminal_only() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[-1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 2), dtype=torch.float32),
        edge_count=torch.tensor([0], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )

    nonterminal, terminal = build_edge_flow_matching_batches_from_trajectories(
        trajectories=trajectories,
    )

    assert nonterminal is None
    assert terminal is not None
    assert terminal.state.edge_count.tolist() == [0]


def test_build_edge_flow_matching_batches_from_one_step_trajectory_includes_root_and_final() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 2), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )

    nonterminal, terminal = build_edge_flow_matching_batches_from_trajectories(
        trajectories=trajectories,
    )

    assert nonterminal is not None
    assert terminal is not None
    assert nonterminal.parent_state.edge_count.tolist() == [0]
    assert terminal.state.edge_count.tolist() == [0, 1]
    assert terminal.state.edge_ids.tolist() == [[-1, -1], [1, -1]]


def test_build_edge_flow_matching_batches_from_trajectories_marks_policy_source() -> None:
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 2), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )

    nonterminal, terminal = build_edge_flow_matching_batches_from_trajectories(
        trajectories=trajectories,
    )

    assert nonterminal is not None
    assert terminal is not None
    assert nonterminal.source.tolist() == [0]
    assert terminal.source.tolist() == [0, 0]


def test_deduplicate_state_batch_keeps_first_occurrence_per_canonical_state() -> None:
    states = StateBatch(
        graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ids=torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [0, -1],
            ],
            dtype=torch.long,
        ),
        edge_count=torch.tensor([0, 1, 1], dtype=torch.long),
        budget=2,
    )

    unique, inverse = deduplicate_state_batch(states=states)

    assert unique.edge_count.tolist() == [0, 1]
    assert inverse.tolist() == [0, 1, 1]


def test_terminal_edge_reward_matching_uses_stop_log_flow_minus_reward() -> None:
    graph = _graph()
    target = _target()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2)

    terms = terminal_edge_reward_matching(
        policy=_ConstantPolicy(stop_log_flow=torch.tensor([3.0])),
        reward_model=_ConstantReward(log_reward=torch.tensor([1.0])),
        features=None,
        graph_context=graph,
        target_context=target,
        transitions=TerminalTransitionBatch(state=state),
    )

    assert torch.allclose(terms.residual, torch.tensor([2.0]))
    assert torch.allclose(terms.loss_units, torch.tensor([2.0]))
    assert terms.valid_mask.tolist() == [True]


def test_policy_output_sample_returns_legal_actions_for_unique_rows() -> None:
    policy = _policy_output_for_rows()

    torch.manual_seed(0)
    sampled = policy.sample(
        rows=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    assert sampled.shape == (3,)
    assert int(sampled[0].item()) == 5
    assert int(sampled[1].item()) == STOP_EDGE_ID
    assert int(sampled[2].item()) in {7, 8}


def test_policy_output_sample_falls_back_for_repeated_rows() -> None:
    policy = _policy_output_for_rows()

    torch.manual_seed(1)
    sampled = policy.sample(
        rows=torch.tensor([2, 2, 0], dtype=torch.long),
    )

    assert sampled.shape == (3,)
    assert int(sampled[0].item()) in {7, 8}
    assert int(sampled[1].item()) in {STOP_EDGE_ID, 7, 8}
    assert int(sampled[2].item()) == 5


def test_policy_output_gather_log_prob_mixes_stop_and_expand() -> None:
    policy = _policy_output_for_rows()

    gathered = policy.gather_log_prob(
        row_ids=torch.tensor([1, 0, 2], dtype=torch.long),
        edge_ids=torch.tensor([STOP_EDGE_ID, 5, 8], dtype=torch.long),
    )

    expected = torch.tensor([-0.2, -0.1, -0.4], dtype=torch.float32)
    assert torch.allclose(gathered, expected)


def test_policy_output_gather_action_log_flow_mixes_stop_and_expand() -> None:
    policy = _policy_output_for_rows()

    gathered = policy.gather_action_log_flow(
        row_ids=torch.tensor([2, 1, 0], dtype=torch.long),
        edge_ids=torch.tensor([7, STOP_EDGE_ID, 5], dtype=torch.long),
    )

    expected = torch.tensor([2.0, 1.3, 0.5], dtype=torch.float32)
    assert torch.allclose(gathered, expected)


def test_policy_output_gather_rejects_illegal_expand_action() -> None:
    policy = _policy_output_for_rows()

    try:
        policy.gather_log_prob(
            row_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([8], dtype=torch.long),
        )
    except ValueError as exc:
        assert "legal action" in str(exc)
    else:
        raise AssertionError("Expected illegal expand action to raise ValueError.")


def test_edge_flow_matching_objective_combines_nonterminal_and_terminal_terms() -> None:
    graph = _graph()
    target = _target()
    root = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2)
    nonterminal = NonterminalTransitionBatch(
        parent_state=root,
        parent_state_ids=torch.tensor([0]),
        edge_ids=torch.tensor([0]),
        log_backward=torch.tensor([0.0]),
    )
    terminal = TerminalTransitionBatch(
        state=root.advance(
            expansion=_expansion(edge_id=0),
        )
    )

    objective = EdgeFlowMatchingObjective(
        nonterminal_weight=1.0,
        terminal_weight=1.0,
        residual_loss="l2",
    )

    output = objective(
        policy=_StructuredPolicy(),
        reward_model=_ConstantReward(log_reward=torch.tensor([1.0])),
        features=None,
        graph_context=graph,
        target_context=target,
        nonterminal=nonterminal,
        terminal=terminal,
    )

    assert torch.allclose(output.loss, torch.tensor(2.8623), atol=1e-4)
    assert output.num_states == 2
    assert output.per_unit_loss is not None
    assert torch.allclose(output.per_unit_loss, torch.tensor([0.8623, 2.0]), atol=1e-4)
    assert output.metrics["objective/nonterminal_edge_flow_matching/num_units"] == 1.0
    assert output.metrics["objective/terminal_edge_reward_matching/num_units"] == 1.0


def test_edge_flow_matching_objective_chunking_matches_unchunked() -> None:
    graph = _graph()
    target = _target()
    parent_state = StateBatch(
        graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ids=torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
            ],
            dtype=torch.long,
        ),
        edge_count=torch.tensor([0, 1, 1], dtype=torch.long),
        budget=2,
    )
    nonterminal = NonterminalTransitionBatch(
        parent_state=parent_state,
        parent_state_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_ids=torch.tensor([0, 1, 0], dtype=torch.long),
        log_backward=torch.tensor([0.0, 0.0, 0.0]),
    )
    terminal = TerminalTransitionBatch(state=parent_state)

    unchunked = EdgeFlowMatchingObjective(
        nonterminal_weight=1.0,
        terminal_weight=1.0,
        residual_loss="l2",
    )
    chunked = EdgeFlowMatchingObjective(
        nonterminal_weight=1.0,
        terminal_weight=1.0,
        residual_loss="l2",
        policy_state_chunk_size=2,
    )

    policy_a = _StructuredPolicy()
    reward_a = _ConstantReward(
        log_reward=torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
    )
    output_a = unchunked(
        policy=policy_a,
        reward_model=reward_a,
        features=None,
        graph_context=graph,
        target_context=target,
        nonterminal=nonterminal,
        terminal=terminal,
    )

    policy_b = _StructuredPolicy()
    reward_b = _ConstantReward(
        log_reward=torch.tensor([1.0, 1.5, 2.0], dtype=torch.float32)
    )
    output_b = chunked(
        policy=policy_b,
        reward_model=reward_b,
        features=None,
        graph_context=graph,
        target_context=target,
        nonterminal=nonterminal,
        terminal=terminal,
    )

    assert torch.allclose(output_a.loss, output_b.loss)
    assert output_a.per_unit_loss is not None
    assert output_b.per_unit_loss is not None
    assert torch.allclose(output_a.per_unit_loss, output_b.per_unit_loss)
    assert output_a.metrics == output_b.metrics


def test_edge_flow_matching_objective_chunking_splits_policy_calls() -> None:
    graph = _graph()
    target = _target()
    parent_state = StateBatch(
        graph_ids=torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
        edge_ids=torch.tensor(
            [
                [-1, -1],
                [0, -1],
                [1, -1],
                [0, -1],
                [1, -1],
            ],
            dtype=torch.long,
        ),
        edge_count=torch.tensor([0, 1, 1, 1, 1], dtype=torch.long),
        budget=2,
    )
    nonterminal = NonterminalTransitionBatch(
        parent_state=parent_state,
        parent_state_ids=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        edge_ids=torch.tensor([0, 1, 0, 1, 0], dtype=torch.long),
        log_backward=torch.zeros(5, dtype=torch.float32),
    )
    terminal = TerminalTransitionBatch(state=parent_state)

    policy = _StructuredPolicy()
    reward = _ConstantReward(log_reward=torch.ones(5, dtype=torch.float32))
    objective = EdgeFlowMatchingObjective(
        nonterminal_weight=1.0,
        terminal_weight=1.0,
        residual_loss="l2",
        policy_state_chunk_size=2,
    )

    objective(
        policy=policy,
        reward_model=reward,
        features=None,
        graph_context=graph,
        target_context=target,
        nonterminal=nonterminal,
        terminal=terminal,
    )

    assert policy.seen_state_sizes == [2, 2, 1, 2, 2, 1, 2, 2, 1]


class _ConstantPolicy:
    def __init__(self, *, stop_log_flow: torch.Tensor) -> None:
        self.stop_log_flow = stop_log_flow.float()

    def __call__(
        self,
        *,
        features,
        state: StateBatch,
        context: GraphContext,
        action_space: ActionSpace,
    ) -> PolicyOutput:
        del features, context
        edge_log_flow = torch.zeros(action_space.num_expansions, dtype=torch.float32)
        continue_log_flow = torch.full((state.num_states,), float("-inf"))
        state_log_flow = torch.logaddexp(self.stop_log_flow, continue_log_flow)
        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=self.stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=self.stop_log_flow - state_log_flow,
            continue_log_prob=torch.full_like(state_log_flow, float("-inf")),
            edge_log_flow=edge_log_flow,
            edge_log_prob=edge_log_flow,
            conditional_edge_log_prob=edge_log_flow,
            edge_raw_score=edge_log_flow,
        )


class _StructuredPolicy:
    def __init__(self) -> None:
        self.seen_state_sizes: list[int] = []

    def __call__(
        self,
        *,
        features,
        state: StateBatch,
        context: GraphContext,
        action_space: ActionSpace,
    ) -> PolicyOutput:
        del features, context
        self.seen_state_sizes.append(int(state.num_states))
        stop_log_flow = 2.0 + state.edge_count.float()
        if action_space.num_expansions > 0:
            edge_log_flow = 3.0 + action_space.expand_edge_ids.float()
            continue_log_flow = torch.full((state.num_states,), float("-inf"))
            continue_log_flow = continue_log_flow.scatter_reduce(
                0,
                action_space.expand_state_ids,
                edge_log_flow,
                reduce="amax",
                include_self=False,
            )
        else:
            edge_log_flow = torch.empty(0, dtype=torch.float32)
            continue_log_flow = torch.full((state.num_states,), float("-inf"))

        state_log_flow = torch.logaddexp(stop_log_flow, continue_log_flow)
        edge_log_prob = (
            edge_log_flow - state_log_flow.index_select(0, action_space.expand_state_ids)
            if action_space.num_expansions > 0
            else edge_log_flow
        )
        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=stop_log_flow - state_log_flow,
            continue_log_prob=torch.where(
                torch.isfinite(continue_log_flow),
                continue_log_flow - state_log_flow,
                torch.full_like(state_log_flow, float("-inf")),
            ),
            edge_log_flow=edge_log_flow,
            edge_log_prob=edge_log_prob,
            conditional_edge_log_prob=edge_log_flow,
            edge_raw_score=edge_log_flow,
        )


class _ConstantReward:
    def __init__(self, *, log_reward: torch.Tensor) -> None:
        self.log_reward = log_reward.float()

    def __call__(
        self,
        *,
        state: StateBatch,
        graph_context: GraphContext,
        target_context: TargetContext,
    ) -> RewardOutput:
        del graph_context, target_context
        zeros = torch.zeros(state.num_states, dtype=torch.float32)
        return RewardOutput(
            log_reward=self.log_reward,
            raw_log_reward=self.log_reward,
            answer_count=zeros,
            candidate_count=torch.ones(state.num_states, dtype=torch.float32),
            target_count=torch.ones(state.num_states, dtype=torch.float32),
            answer_precision=zeros,
            target_recall=zeros,
            answer_f_score=zeros,
            edge_count=state.edge_count.float(),
            valid_mask=torch.ones(state.num_states, dtype=torch.bool),
            success_mask=torch.zeros(state.num_states, dtype=torch.bool),
            metrics={},
        )


def _expansion(*, edge_id: int) -> object:
    from src.weaver.state import ExpansionBatch

    return ExpansionBatch(
        state_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([edge_id], dtype=torch.long),
    )


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 2], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 2, 2], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 1], dtype=torch.long),
            in_ptr=torch.tensor([0, 0, 1, 2], dtype=torch.long),
            edge_ids_by_dst=torch.tensor([0, 1], dtype=torch.long),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, False, True]),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        target_count_by_graph=torch.tensor([1], dtype=torch.long),
        node_target_distance=torch.tensor([2, 1, 0], dtype=torch.long),
        shortest_path_edge_mask=torch.tensor([True, False]),
        shortest_path_edge_weight=torch.tensor([1.0, 0.0]),
    )


def _policy_output_for_rows() -> PolicyOutput:
    action_space = ActionSpace(
        num_states=3,
        expand_state_ids=torch.tensor([0, 2, 2], dtype=torch.long),
        expand_edge_ids=torch.tensor([5, 7, 8], dtype=torch.long),
        expand_ptr=torch.tensor([0, 1, 1, 3], dtype=torch.long),
    )
    return PolicyOutput(
        action_space=action_space,
        state_log_flow=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        stop_log_flow=torch.tensor([0.4, 1.3, 1.1], dtype=torch.float32),
        continue_log_flow=torch.tensor([0.5, float("-inf"), 2.5], dtype=torch.float32),
        stop_log_prob=torch.tensor([-2.0, -0.2, -3.0], dtype=torch.float32),
        continue_log_prob=torch.tensor([-0.1, float("-inf"), -0.05], dtype=torch.float32),
        edge_log_flow=torch.tensor([0.5, 2.0, 2.4], dtype=torch.float32),
        edge_log_prob=torch.tensor([-0.1, -0.9, -0.4], dtype=torch.float32),
        conditional_edge_log_prob=torch.tensor([0.0, -1.0, -0.5], dtype=torch.float32),
        edge_raw_score=torch.tensor([0.5, 2.0, 2.4], dtype=torch.float32),
    )
