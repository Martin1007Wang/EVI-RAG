from __future__ import annotations

import copy

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.objectives.subtb.batch import prepare_subtb_batch
from src.weaver.objectives.subtb.scoring import (
    combine_subtb_scores,
    score_backward_step_log_probs,
    score_forward_subtb_batch,
)
from src.weaver.policy import (
    BackwardPolicy,
    BackwardScoringModel,
    FlowEstimator,
    ForwardPolicy,
    StateFlowHead,
)
from src.weaver.reward import EvidenceStateScoreOutput
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch
from src.weaver.state import StateBatch


def _graph_context() -> GraphContext:
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
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _forward_policy(hidden_dim: int) -> ForwardPolicy:
    return ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=hidden_dim, num_heads=1),
        flow_estimator=FlowEstimator(hidden_dim=hidden_dim),
        state_flow_head=StateFlowHead(state_dim=hidden_dim),
    )


def _backward_model(hidden_dim: int) -> BackwardScoringModel:
    return BackwardScoringModel(
        state_encoder=StateEncoder(hidden_dim=hidden_dim, num_heads=1),
        backward_policy=BackwardPolicy(hidden_dim=hidden_dim),
    )


def _reward_output(*, state_potential: torch.Tensor, log_reward: torch.Tensor, terminal_valid_mask: torch.Tensor) -> EvidenceStateScoreOutput:
    zeros = torch.zeros_like(log_reward, dtype=torch.float32)
    return EvidenceStateScoreOutput(
        state_potential=state_potential,
        remaining_log_reward=log_reward - state_potential,
        log_reward=log_reward,
        raw_log_reward=log_reward,
        answer_count=zeros,
        candidate_count=zeros,
        target_count=zeros,
        target_recall=state_potential,
        target_precision=zeros,
        terminal_quality=log_reward,
        edge_count=zeros,
        valid_target_mask=terminal_valid_mask,
        nonempty_mask=terminal_valid_mask,
        success_mask=terminal_valid_mask,
        terminal_valid_mask=terminal_valid_mask,
        metrics={},
    )


def _trajectory() -> TrajectoryBatch:
    return TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[0]], dtype=torch.long),
        edge_logp=torch.zeros((1, 1), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.tensor([False]),
    )


def test_score_forward_subtb_batch_rebuilds_policy_input_for_training_gradients() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    policy = _forward_policy(hidden_dim)
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim, requires_grad=True),
        entity_h=torch.randn(3, hidden_dim, requires_grad=True),
        edge_h=torch.randn(2, hidden_dim, requires_grad=True),
        relation_h=torch.randn(2, hidden_dim, requires_grad=True),
        frontier_prune_score=torch.randn(2, dtype=torch.float32),
    )
    batch = prepare_subtb_batch(
        trajectories=_trajectory(),
        graph_context=graph,
    )
    reward = _reward_output(
        state_potential=torch.zeros(batch.states.num_states, dtype=torch.float32),
        log_reward=torch.zeros(batch.states.num_states, dtype=torch.float32),
        terminal_valid_mask=torch.ones(batch.states.num_states, dtype=torch.bool),
    )

    with torch.no_grad():
        stale_policy_input = policy.build_policy_input(features, graph_context=graph)

    scores = score_forward_subtb_batch(
        batch=batch,
        policy=policy,
        features=features,
        policy_input=stale_policy_input,
        graph_context=graph,
        reward=reward,
    )
    loss = (
        scores.log_flow.sum()
        + scores.stop_log_prob_by_state.sum()
        + scores.step_log_prob.sum()
        + scores.terminal_stop_logp_by_traj.sum()
        + scores.frontier_log_prob.sum()
    )
    loss.backward()

    assert features.question_h.grad is not None
    assert features.edge_h.grad is not None
    assert torch.count_nonzero(features.question_h.grad).item() > 0
    assert torch.count_nonzero(features.edge_h.grad).item() > 0
    assert features.relation_h.grad is None


def test_forward_policy_masks_stop_on_empty_states_with_frontier() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    policy = _forward_policy(hidden_dim)
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim),
        entity_h=torch.randn(3, hidden_dim),
        edge_h=torch.randn(2, hidden_dim),
        relation_h=torch.randn(2, hidden_dim),
        frontier_prune_score=torch.randn(2, dtype=torch.float32),
    )
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=1,
        graph_context=graph,
    )

    output = policy(
        state=state,
        features=features,
        graph_context=graph,
        policy_input=policy.build_policy_input(features, graph_context=graph),
    )

    assert output.action_edge_ids[0].item() == -1
    assert output.action_logits[0].item() == -1.0e9


def test_score_forward_subtb_batch_detaches_reward_state_potential_from_training_graph() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    policy = _forward_policy(hidden_dim)
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim, requires_grad=True),
        entity_h=torch.randn(3, hidden_dim, requires_grad=True),
        edge_h=torch.randn(2, hidden_dim, requires_grad=True),
        relation_h=torch.randn(2, hidden_dim, requires_grad=True),
        frontier_prune_score=torch.randn(2, dtype=torch.float32),
    )
    batch = prepare_subtb_batch(
        trajectories=_trajectory(),
        graph_context=graph,
    )
    state_potential = torch.randn(batch.states.num_states, dtype=torch.float32, requires_grad=True)
    reward = _reward_output(
        state_potential=state_potential,
        log_reward=torch.zeros(batch.states.num_states, dtype=torch.float32),
        terminal_valid_mask=torch.ones(batch.states.num_states, dtype=torch.bool),
    )

    with torch.no_grad():
        stale_policy_input = policy.build_policy_input(features, graph_context=graph)

    scores = score_forward_subtb_batch(
        batch=batch,
        policy=policy,
        features=features,
        policy_input=stale_policy_input,
        graph_context=graph,
        reward=reward,
    )
    scores.log_flow.sum().backward()

    assert state_potential.grad is None


def test_score_backward_step_log_probs_produces_gradients_for_online_model_only() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    batch = prepare_subtb_batch(
        trajectories=_trajectory(),
        graph_context=graph,
    )
    online_model = _backward_model(hidden_dim)
    target_model = copy.deepcopy(online_model)
    target_model.requires_grad_(False)
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim, requires_grad=True),
        entity_h=torch.randn(3, hidden_dim, requires_grad=True),
        edge_h=torch.randn(2, hidden_dim, requires_grad=True),
        relation_h=torch.randn(2, hidden_dim, requires_grad=True),
        frontier_prune_score=torch.randn(2, dtype=torch.float32),
    )

    online_scores = score_backward_step_log_probs(
        batch=batch,
        model=online_model,
        features=features,
        graph_context=graph,
    )
    online_scores[batch.valid_steps].sum().backward()
    assert any(param.grad is not None for param in online_model.parameters())

    target_features = FeaturePack(
        question_h=features.question_h.detach().clone().requires_grad_(True),
        entity_h=features.entity_h.detach().clone().requires_grad_(True),
        edge_h=features.edge_h.detach().clone().requires_grad_(True),
        relation_h=features.relation_h.detach().clone().requires_grad_(True),
        frontier_prune_score=features.frontier_prune_score.detach().clone(),
    )
    with torch.no_grad():
        target_scores = score_backward_step_log_probs(
            batch=batch,
            model=target_model,
            features=target_features,
            graph_context=graph,
        )
    assert not target_scores.requires_grad


def test_combine_subtb_scores_preserves_forward_values_and_injects_backward_steps() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    batch = prepare_subtb_batch(
        trajectories=_trajectory(),
        graph_context=graph,
    )
    policy = _forward_policy(hidden_dim)
    backward_model = _backward_model(hidden_dim)
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim, requires_grad=True),
        entity_h=torch.randn(3, hidden_dim, requires_grad=True),
        edge_h=torch.randn(2, hidden_dim, requires_grad=True),
        relation_h=torch.randn(2, hidden_dim, requires_grad=True),
        frontier_prune_score=torch.randn(2, dtype=torch.float32),
    )
    reward = _reward_output(
        state_potential=torch.zeros(batch.states.num_states, dtype=torch.float32),
        log_reward=torch.zeros(batch.states.num_states, dtype=torch.float32),
        terminal_valid_mask=torch.ones(batch.states.num_states, dtype=torch.bool),
    )
    forward_scores = score_forward_subtb_batch(
        batch=batch,
        policy=policy,
        features=features,
        policy_input=policy.build_policy_input(features, graph_context=graph),
        graph_context=graph,
        reward=reward,
    )
    backward_step_logp = score_backward_step_log_probs(
        batch=batch,
        model=backward_model,
        features=features,
        graph_context=graph,
    ).detach()

    scores = combine_subtb_scores(
        forward_scores=forward_scores,
        backward_step_log_prob=backward_step_logp,
    )

    assert torch.equal(scores.log_flow, forward_scores.log_flow)
    assert torch.equal(scores.step_log_prob, forward_scores.step_log_prob)
    assert torch.equal(scores.backward_step_log_prob, backward_step_logp)
