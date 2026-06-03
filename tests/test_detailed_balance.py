from __future__ import annotations

import torch
from torch import nn

from src.weaver.context import DirectedAdjacencyIndex, GraphContext, TargetContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.objectives.subtb import (
    SubTBBatch,
    SubTBPolicyScores,
    SubTBTermTable,
    SubTrajectoryBalanceObjective,
    prepare_subtb_batch,
    score_subtb_batch,
)
from src.weaver.policy import FlowEstimator, ForwardPolicy, PolicyActionSpace, PolicyOutput, StateFlowHead
from src.weaver.reward import EvidenceSubgraphReward, TerminalRewardOutput
from src.weaver.rollout.trajectory import BUDGET, EXTERNAL_TERMINAL, NO_FRONTIER, POLICY_STOP, TrajectoryBatch
from src.weaver.state import FrontierEncoding, NodeSelection, StateBatch


def test_boundary_state_has_only_stop_and_unit_probability() -> None:
    graph, features = _fixture()
    state = StateBatch.from_selected_edges(
        graph_ids=torch.tensor([0]), edge_ids=torch.tensor([[0]]), edge_count=torch.tensor([1]), budget=1, graph_context=graph
    )
    output = _policy().forward(state=state, features=features, graph_context=graph)
    assert output.forced_terminal_mask.tolist() == [True]
    assert output.action_edge_ids.tolist() == [-1]
    assert output.gather_log_prob(row_ids=torch.tensor([0]), edge_ids=torch.tensor([-1])).tolist() == [0.0]


def test_subtb_batch_deduplicates_prefix_states() -> None:
    graph, _ = _fixture()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0]),
        edge_ids=torch.tensor([[0], [0]]),
        edge_logp=torch.zeros(2, 1),
        edge_count=torch.tensor([1, 1]),
        stop_reason=torch.tensor([POLICY_STOP, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(2),
        source=torch.tensor([False, True]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph)
    assert prepared.states.edge_count.tolist() == [0, 1]
    assert prepared.prefix_state_ids.tolist() == [[0, 1], [0, 1]]


def test_subtb_term_tables_match_expected_spans() -> None:
    graph, _ = _chain_fixture()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor([[-1, -1, -1], [0, -1, -1], [0, 1, 2]]),
        edge_logp=torch.zeros(3, 3),
        edge_count=torch.tensor([0, 1, 3]),
        stop_reason=torch.tensor([POLICY_STOP, EXTERNAL_TERMINAL, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(3),
        source=torch.tensor([False, False, True]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph, max_subtrajectory_length=2)

    assert prepared.policy_transition_terms.traj_ids.tolist() == []
    assert list(zip(prepared.replay_transition_terms.start_steps.tolist(), prepared.replay_transition_terms.end_steps.tolist(), strict=True)) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert list(zip(prepared.terminal_terms.traj_ids.tolist(), prepared.terminal_terms.start_steps.tolist(), strict=True)) == [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
        (2, 2),
        (2, 3),
    ]


def test_subtb_batch_handles_mixed_zero_and_full_budget_rows() -> None:
    graph, _ = _chain_fixture()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor([[-1, -1, -1], [-1, -1, -1], [0, 1, 2]]),
        edge_logp=torch.zeros(3, 3),
        edge_count=torch.tensor([0, 0, 3]),
        stop_reason=torch.tensor([POLICY_STOP, EXTERNAL_TERMINAL, BUDGET], dtype=torch.uint8),
        stop_logp=torch.zeros(3),
        source=torch.tensor([False, True, False]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph)

    assert list(zip(prepared.policy_transition_terms.traj_ids.tolist(), prepared.policy_transition_terms.end_steps.tolist(), strict=True)) == [
        (2, 1),
        (2, 2),
        (2, 2),
    ]
    assert prepared.policy_transition_terms.end_steps.max().item() == 2
    assert prepared.replay_transition_terms.num_terms == 0
    assert list(zip(prepared.terminal_terms.traj_ids.tolist(), prepared.terminal_terms.start_steps.tolist(), strict=True)) == [
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ]
    for terms in (prepared.policy_transition_terms, prepared.replay_transition_terms, prepared.terminal_terms):
        assert bool(terms.start_state_ids.ge(0).all())
        assert bool(terms.end_state_ids.ge(0).all())


def test_budget_terminal_gets_terminal_reward_terms() -> None:
    graph, _ = _fixture()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0]]),
        edge_logp=torch.zeros(1, 1),
        edge_count=torch.tensor([1]),
        stop_reason=torch.tensor([BUDGET], dtype=torch.uint8),
        stop_logp=torch.zeros(1),
        source=torch.tensor([False]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph)

    assert prepared.trainable_terminal_mask.tolist() == [False]
    assert prepared.terminal_stop_action_mask.tolist() == [False]
    assert list(zip(prepared.terminal_terms.traj_ids.tolist(), prepared.terminal_terms.start_steps.tolist(), strict=True)) == [
        (0, 0),
        (0, 1),
    ]


def test_subtrajectory_balance_objective_runs_and_backprops() -> None:
    graph, features = _fixture()
    trajectories = _policy_stop_trajectories()
    policy = _policy()
    output = _run_objective(
        objective=SubTrajectoryBalanceObjective(),
        policy=policy,
        trajectories=trajectories,
        graph=graph,
        features=features,
    )
    assert output.loss.dtype == torch.float32
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_subtrajectory_balance_uses_exact_backward_kernel() -> None:
    graph, features = _fixture()
    output = _run_objective(
        objective=SubTrajectoryBalanceObjective(),
        policy=_policy(),
        trajectories=_policy_stop_trajectories(),
        graph=graph,
        features=features,
    )
    assert "objective/backward_log_prob_abs_mean" in output.metrics
    assert output.metrics["objective/backward_log_prob_abs_mean"] >= 0.0


def test_subtb_scores_policy_once_for_prepared_states() -> None:
    graph, features = _fixture()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0]),
        edge_ids=torch.tensor([[0], [0]]),
        edge_logp=torch.zeros(2, 1),
        edge_count=torch.tensor([1, 1]),
        stop_reason=torch.tensor([POLICY_STOP, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(2),
        source=torch.tensor([False, False]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph)
    policy = _CountingPolicy(base=_policy())
    scored = score_subtb_batch(
        batch=prepared,
        policy=policy,
        features=features,
        cache=policy.build_cache(features),
        graph_context=graph,
    )
    assert policy.forward_calls == 1
    assert scored.step_log_prob.shape == trajectories.edge_logp.shape


def test_subtrajectory_balance_stop_head_receives_gradient() -> None:
    graph, features = _fixture()
    policy = _policy()
    output = _run_objective(
        objective=SubTrajectoryBalanceObjective(),
        policy=policy,
        trajectories=_policy_stop_trajectories(),
        graph=graph,
        features=features,
    )
    output.loss.backward()
    grads = [parameter.grad for parameter in policy.flow_estimator.stop_head.parameters()]
    assert any(grad is not None and torch.count_nonzero(grad).item() > 0 for grad in grads)


def test_subtb_vectorized_loss_matches_reference() -> None:
    graph, features = _fixture()
    policy = _policy()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor([[0, -1], [0, -1], [0, -1]]),
        edge_logp=torch.zeros((3, 2)),
        edge_count=torch.tensor([1, 1, 1]),
        stop_reason=torch.tensor([POLICY_STOP, NO_FRONTIER, EXTERNAL_TERMINAL], dtype=torch.uint8),
        stop_logp=torch.zeros(3),
        source=torch.tensor([False, False, True]),
    )
    prepared = prepare_subtb_batch(trajectories=trajectories, graph_context=graph, max_subtrajectory_length=2)
    scores = score_subtb_batch(
        batch=prepared,
        policy=policy,
        features=features,
        cache=policy.build_cache(features),
        graph_context=graph,
    )
    reward = EvidenceSubgraphReward()(state=prepared.states, target_context=_target(), graph_context=graph, active=scores.action_space.active)
    objective = SubTrajectoryBalanceObjective(subtb_lambda=0.5, max_subtrajectory_length=2)
    output = objective(batch=prepared, scores=scores, reward=reward)
    ref_loss = _reference_objective_loss(prepared=prepared, scores=scores, reward=reward, subtb_lambda=0.5)
    assert torch.allclose(output.loss, ref_loss)


def test_subtb_loss_matches_fixed_hand_computed_regression() -> None:
    batch = _fixed_loss_batch()
    scores = _fixed_loss_scores()
    reward = TerminalRewardOutput(
        log_reward=torch.tensor([0.0, 0.0, 0.3]),
        raw_log_reward=torch.tensor([0.0, 0.0, 0.3]),
        answer_count=torch.zeros(3),
        candidate_count=torch.zeros(3),
        target_count=torch.ones(3),
        target_recall=torch.zeros(3),
        edge_count=torch.tensor([0.0, 1.0, 2.0]),
        valid_mask=torch.tensor([True, True, True]),
        success_mask=torch.tensor([False, False, True]),
    )

    output = SubTrajectoryBalanceObjective(subtb_lambda=0.5)(batch=batch, scores=scores, reward=reward)
    expected = torch.tensor((1.1**2 + 0.5 * 0.9**2 + (-0.2) ** 2 + (-1.1) ** 2) / 3.5)
    assert torch.allclose(output.loss, expected)


def test_external_terminal_uses_reward_with_terminal_stop_logp() -> None:
    graph, features = _fixture()
    policy = _policy()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0]]),
        edge_logp=torch.zeros(1, 1),
        edge_count=torch.tensor([1]),
        stop_reason=torch.tensor([EXTERNAL_TERMINAL], dtype=torch.uint8),
        stop_logp=torch.tensor([5.0]),
        source=torch.tensor([True]),
    )
    output = _run_objective(
        objective=SubTrajectoryBalanceObjective(),
        policy=policy,
        trajectories=trajectories,
        graph=graph,
        features=features,
    )
    output.loss.backward()
    assert output.metrics["objective/terminal_skipped_external_count"] == 1.0
    assert output.metrics["objective/terminal_stop_action_count"] == 1.0
    assert output.metrics["objective/terminal_stop_action_external_count"] == 1.0
    assert output.metrics["objective/terminal_reward_trajectory_count"] == 1.0
    assert output.metrics["objective/terminal_reward_term_count"] == 2.0
    assert output.metrics["objective/subtb_terminal_abs_residual_mean"] > 0.0


def test_external_terminal_uses_terminal_stop_logp_even_when_not_policy_stop() -> None:
    batch = _fixed_loss_batch(stop_reason=EXTERNAL_TERMINAL, source=True, trainable_terminal=False)
    scores = _fixed_loss_scores()
    reward = TerminalRewardOutput(
        log_reward=torch.tensor([0.0, 0.0, 0.3]),
        raw_log_reward=torch.tensor([0.0, 0.0, 0.3]),
        answer_count=torch.zeros(3),
        candidate_count=torch.zeros(3),
        target_count=torch.ones(3),
        target_recall=torch.zeros(3),
        edge_count=torch.tensor([0.0, 1.0, 2.0]),
        valid_mask=torch.tensor([True, True, True]),
        success_mask=torch.tensor([False, False, True]),
    )

    output = SubTrajectoryBalanceObjective(subtb_lambda=0.5)(batch=batch, scores=scores, reward=reward)
    changed_scores = SubTBPolicyScores(
        action_space=scores.action_space,
        action_count=scores.action_count,
        frontier_count=scores.frontier_count,
        log_flow=scores.log_flow,
        stop_log_prob_by_state=scores.stop_log_prob_by_state,
        log_backward_by_state=scores.log_backward_by_state,
        step_log_prob=scores.step_log_prob,
        forward_prefix_by_traj=scores.forward_prefix_by_traj,
        backward_prefix_by_traj=scores.backward_prefix_by_traj,
        terminal_stop_logp_by_traj=torch.tensor([-100.0]),
    )
    changed = SubTrajectoryBalanceObjective(subtb_lambda=0.5)(batch=batch, scores=changed_scores, reward=reward)

    assert not torch.allclose(output.loss, changed.loss)


def test_budget_terminal_ignores_terminal_stop_logp() -> None:
    batch = _fixed_loss_batch(stop_reason=BUDGET, source=False, trainable_terminal=False, terminal_stop_action=False)
    scores = _fixed_loss_scores()
    reward = TerminalRewardOutput(
        log_reward=torch.tensor([0.0, 0.0, 0.3]),
        raw_log_reward=torch.tensor([0.0, 0.0, 0.3]),
        answer_count=torch.zeros(3),
        candidate_count=torch.zeros(3),
        target_count=torch.ones(3),
        target_recall=torch.zeros(3),
        edge_count=torch.tensor([0.0, 1.0, 2.0]),
        valid_mask=torch.tensor([True, True, True]),
        success_mask=torch.tensor([False, False, True]),
    )

    output = SubTrajectoryBalanceObjective(subtb_lambda=0.5)(batch=batch, scores=scores, reward=reward)
    changed_scores = SubTBPolicyScores(
        action_space=scores.action_space,
        action_count=scores.action_count,
        frontier_count=scores.frontier_count,
        log_flow=scores.log_flow,
        stop_log_prob_by_state=scores.stop_log_prob_by_state,
        log_backward_by_state=scores.log_backward_by_state,
        step_log_prob=scores.step_log_prob,
        forward_prefix_by_traj=scores.forward_prefix_by_traj,
        backward_prefix_by_traj=scores.backward_prefix_by_traj,
        terminal_stop_logp_by_traj=torch.tensor([-100.0]),
    )
    changed = SubTrajectoryBalanceObjective(subtb_lambda=0.5)(batch=batch, scores=changed_scores, reward=reward)

    assert torch.allclose(output.loss, changed.loss)


def _run_objective(
    *,
    objective: SubTrajectoryBalanceObjective,
    policy: ForwardPolicy,
    trajectories: TrajectoryBatch,
    graph: GraphContext,
    features: FeaturePack,
):
    prepared = prepare_subtb_batch(
        trajectories=trajectories,
        graph_context=graph,
        max_subtrajectory_length=objective.max_subtrajectory_length,
    )
    scores = score_subtb_batch(
        batch=prepared,
        policy=policy,
        features=features,
        cache=policy.build_cache(features),
        graph_context=graph,
    )
    reward = EvidenceSubgraphReward()(state=prepared.states, target_context=_target(), graph_context=graph, active=scores.action_space.active)
    return objective(batch=prepared, scores=scores, reward=reward)


def _policy_stop_trajectories() -> TrajectoryBatch:
    return TrajectoryBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0]]),
        edge_logp=torch.zeros(1, 1),
        edge_count=torch.tensor([1]),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1),
        source=torch.tensor([False]),
    )


def _policy() -> ForwardPolicy:
    return ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=4),
        flow_estimator=FlowEstimator(hidden_dim=4),
        state_flow_head=StateFlowHead(state_dim=4),
    )


def _target() -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, True]),
        reachable_target_node_ids=torch.tensor([1]),
        reachable_target_node_ids_ptr=torch.tensor([0, 1]),
        target_count_by_graph=torch.tensor([1]),
        node_target_distance=torch.tensor([1, 0]),
    )


def _fixture() -> tuple[GraphContext, FeaturePack]:
    edge_index = torch.tensor([[0], [1]])
    graph = GraphContext(
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
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )
    features = FeaturePack(
        question_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        entity_h=torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        edge_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        relation_h=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        device=torch.device("cpu"),
    )
    return graph, features


def _chain_fixture() -> tuple[GraphContext, FeaturePack]:
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    graph = GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0, 0]),
        edge_to_graph=torch.tensor([0, 0, 0]),
        edge_ptr=torch.tensor([0, 3]),
        anchor_mask=torch.tensor([True, False, False, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 2, 3, 3]),
            edge_ids_by_src=torch.tensor([0, 1, 2]),
        ),
        num_nodes=4,
        num_edges=3,
        num_graphs=1,
    )
    features = FeaturePack(
        question_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        entity_h=torch.eye(4),
        edge_h=torch.eye(4)[:3],
        relation_h=torch.eye(4)[1:4],
        device=torch.device("cpu"),
    )
    return graph, features


class _CountingPolicy(nn.Module):
    def __init__(self, *, base: ForwardPolicy) -> None:
        super().__init__()
        self.base = base
        self.forward_calls = 0

    def build_cache(self, features: FeaturePack):
        return self.base.build_cache(features)

    def prepare_action_space(self, *, state: StateBatch, graph_context: GraphContext):
        return self.base.prepare_action_space(state=state, graph_context=graph_context)

    def forward(self, **kwargs) -> PolicyOutput:
        self.forward_calls += 1
        return self.base(**kwargs)


def _fixed_loss_batch(
    *,
    stop_reason: int = POLICY_STOP,
    source: bool = False,
    trainable_terminal: bool = True,
    terminal_stop_action: bool | None = None,
) -> SubTBBatch:
    if terminal_stop_action is None:
        terminal_stop_action = trainable_terminal or stop_reason == EXTERNAL_TERMINAL
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0]),
        edge_ids=torch.tensor([[0, 1]]),
        edge_logp=torch.zeros(1, 2),
        edge_count=torch.tensor([2]),
        stop_reason=torch.tensor([stop_reason], dtype=torch.uint8),
        stop_logp=torch.zeros(1),
        source=torch.tensor([source]),
    )
    states = StateBatch(
        graph_ids=torch.tensor([0, 0, 0]),
        edge_ids=torch.tensor([[-1, -1], [0, -1], [0, 1]]),
        edge_count=torch.tensor([0, 1, 2]),
        budget=2,
    )
    transition_terms = SubTBTermTable(
        traj_ids=torch.tensor([0]),
        start_steps=torch.tensor([0]),
        end_steps=torch.tensor([1]),
        start_state_ids=torch.tensor([0]),
        end_state_ids=torch.tensor([1]),
        lambda_exponent=torch.tensor([0]),
    )
    terminal_terms = SubTBTermTable(
        traj_ids=torch.tensor([0, 0, 0]),
        start_steps=torch.tensor([0, 1, 2]),
        end_steps=torch.tensor([2, 2, 2]),
        start_state_ids=torch.tensor([0, 1, 2]),
        end_state_ids=torch.tensor([2, 2, 2]),
        lambda_exponent=torch.tensor([1, 0, 0]),
    )
    empty_terms = SubTBTermTable(
        traj_ids=torch.empty(0, dtype=torch.long),
        start_steps=torch.empty(0, dtype=torch.long),
        end_steps=torch.empty(0, dtype=torch.long),
        start_state_ids=torch.empty(0, dtype=torch.long),
        end_state_ids=torch.empty(0, dtype=torch.long),
        lambda_exponent=torch.empty(0, dtype=torch.long),
    )
    return SubTBBatch(
        trajectories=trajectories,
        states=states,
        prefix_state_ids=torch.tensor([[0, 1, 2]]),
        valid_steps=torch.tensor([[True, True]]),
        step_traj_ids=torch.tensor([0, 0]),
        step_ids=torch.tensor([0, 1]),
        step_parent_state_ids=torch.tensor([0, 1]),
        step_edge_ids=torch.tensor([0, 1]),
        terminal_state_ids=torch.tensor([2]),
        terminal_step_by_traj=torch.tensor([2]),
        terminal_kind_by_traj=torch.tensor([stop_reason]),
        trainable_terminal_mask=torch.tensor([trainable_terminal]),
        terminal_stop_action_mask=torch.tensor([terminal_stop_action]),
        policy_transition_terms=transition_terms,
        replay_transition_terms=empty_terms,
        terminal_terms=terminal_terms,
    )


def _fixed_loss_scores() -> SubTBPolicyScores:
    action_space = PolicyActionSpace(
        active=NodeSelection(row_ids=torch.empty(0, dtype=torch.long), node_ids=torch.empty(0, dtype=torch.long)),
        frontier=FrontierEncoding(
            row_ids=torch.empty(0, dtype=torch.long),
            edge_ids=torch.empty(0, dtype=torch.long),
        ),
    )
    step_log_prob = torch.tensor([[0.4, 0.3]])
    backward_step_logp = torch.tensor([[0.1, 0.2]])
    return SubTBPolicyScores(
        action_space=action_space,
        action_count=torch.zeros(3),
        frontier_count=torch.zeros(3),
        log_flow=torch.tensor([1.0, 0.2, -0.6]),
        stop_log_prob_by_state=torch.tensor([0.0, 0.0, -0.2]),
        log_backward_by_state=torch.tensor([0.0, 0.1, 0.2]),
        step_log_prob=step_log_prob,
        forward_prefix_by_traj=torch.cat([torch.zeros(1, 1), torch.cumsum(step_log_prob, dim=1)], dim=1),
        backward_prefix_by_traj=torch.cat([torch.zeros(1, 1), torch.cumsum(backward_step_logp, dim=1)], dim=1),
        terminal_stop_logp_by_traj=torch.tensor([-0.2]),
    )


def _reference_objective_loss(
    *,
    prepared,
    scores,
    reward,
    subtb_lambda: float,
) -> torch.Tensor:
    total_loss = scores.log_flow.new_zeros(())
    total_weight = scores.log_flow.new_zeros(())
    for traj_idx in range(prepared.trajectories.num_trajectories):
        length = int(prepared.terminal_step_by_traj[traj_idx].item())
        prefix = prepared.prefix_state_ids[traj_idx, : length + 1]
        for start in range(length + 1):
            max_end = length - 1
            if max_end >= start + 1:
                for end in range(start + 1, max_end + 1):
                    residual = (
                        scores.log_flow[prefix[start]]
                        + (scores.forward_prefix_by_traj[traj_idx, end] - scores.forward_prefix_by_traj[traj_idx, start])
                        - (scores.backward_prefix_by_traj[traj_idx, end] - scores.backward_prefix_by_traj[traj_idx, start])
                        - scores.log_flow[prefix[end]]
                    )
                    weight = subtb_lambda ** max(end - start - 1, 0)
                    total_loss = total_loss + weight * residual.square()
                    total_weight = total_weight + weight
            terminal_state_id = int(prepared.terminal_state_ids[traj_idx].item())
            if not bool(reward.valid_mask[terminal_state_id].item()):
                continue
            terminal_action_logp = scores.terminal_stop_logp_by_traj[traj_idx] if bool(prepared.terminal_stop_action_mask[traj_idx].item()) else 0.0
            residual = (
                scores.log_flow[prefix[start]]
                + (scores.forward_prefix_by_traj[traj_idx, length] - scores.forward_prefix_by_traj[traj_idx, start])
                + terminal_action_logp
                - (scores.backward_prefix_by_traj[traj_idx, length] - scores.backward_prefix_by_traj[traj_idx, start])
                - reward.log_reward.float()[terminal_state_id]
            )
            weight = subtb_lambda ** max(max(length - start, 1) - 1, 0)
            total_loss = total_loss + weight * residual.square()
            total_weight = total_weight + weight
    return total_loss / total_weight.clamp_min(1.0)
