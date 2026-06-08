from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.objectives.subtb.batch import prepare_subtb_batch
from src.weaver.objectives.subtb.scoring import score_forward_subtb_batch
from src.weaver.policy import FlowEstimator, ForwardPolicy, FrontierPruningConfig, StateFlowHead
from src.weaver.reward import EvidenceStateScoreOutput
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch
from src.weaver.state import StateBatch


def _graph_context() -> GraphContext:
    edge_index = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 3], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 3, 3, 3, 3], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 1, 2], dtype=torch.long),
        ),
        num_nodes=4,
        num_edges=3,
        num_graphs=1,
    )


def _policy(hidden_dim: int) -> ForwardPolicy:
    return ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=hidden_dim, num_heads=1),
        flow_estimator=FlowEstimator(hidden_dim=hidden_dim),
        state_flow_head=StateFlowHead(state_dim=hidden_dim),
        frontier_pruning=FrontierPruningConfig(
            enabled=True,
            threshold=0.95,
            min_keep_per_state=1,
            apply_train=True,
            apply_eval=True,
            keep_recorded_edges_in_train=True,
        ),
    )


def _features() -> FeaturePack:
    return FeaturePack(
        question_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        entity_h=torch.randn(4, 4),
        edge_h=torch.randn(3, 4),
        relation_h=torch.tensor(
            [
                [0.2, 0.98, 0.0, 0.0],
                [0.8, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        frontier_prune_score=torch.tensor([0.0, 0.2, 1.0], dtype=torch.float32),
    )


def _reward_output(num_states: int) -> EvidenceStateScoreOutput:
    zeros = torch.zeros(num_states, dtype=torch.float32)
    mask = torch.ones(num_states, dtype=torch.bool)
    return EvidenceStateScoreOutput(
        state_potential=zeros,
        remaining_log_reward=zeros,
        log_reward=zeros,
        raw_log_reward=zeros,
        answer_count=zeros,
        candidate_count=zeros,
        target_count=zeros,
        target_recall=zeros,
        target_precision=zeros,
        terminal_quality=zeros,
        edge_count=zeros,
        valid_target_mask=mask,
        nonempty_mask=mask,
        success_mask=mask,
        terminal_valid_mask=mask,
        metrics={},
    )


def test_prepare_action_space_keeps_top_edge_when_threshold_prunes_everything() -> None:
    graph = _graph_context()
    policy = _policy(hidden_dim=4)
    features = _features()
    policy_input = policy.build_policy_input(features, graph_context=graph, compute_align_score=False)

    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    action_space = policy.prepare_action_space(
        state=root,
        graph_context=graph,
        policy_input=policy_input,
        training=False,
    )

    assert int(action_space.frontier.edge_ids.numel()) == 1
    assert int(action_space.frontier.edge_ids[0].item()) == 2


def test_prepare_action_space_uses_static_frontier_prune_score_instead_of_projected_relation_h() -> None:
    graph = _graph_context()
    policy = _policy(hidden_dim=4)
    features = FeaturePack(
        question_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        entity_h=torch.randn(4, 4),
        edge_h=torch.randn(3, 4),
        relation_h=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        frontier_prune_score=torch.tensor([-0.8, 0.95, -0.7], dtype=torch.float32),
    )
    policy_input = policy.build_policy_input(features, graph_context=graph, compute_align_score=False)
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    action_space = policy.prepare_action_space(
        state=root,
        graph_context=graph,
        policy_input=policy_input,
        training=False,
    )

    assert int(action_space.frontier.edge_ids.numel()) == 1
    assert int(action_space.frontier.edge_ids[0].item()) == 1


def test_score_subtb_batch_keeps_recorded_training_edge_even_if_below_threshold() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    policy = _policy(hidden_dim=4)
    features = _features()
    policy_input = policy.build_policy_input(features, graph_context=graph, compute_align_score=False)
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 1), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.tensor([False]),
    )
    batch = prepare_subtb_batch(
        trajectories=trajectories,
        graph_context=graph,
    )
    reward = _reward_output(batch.states.num_states)

    scores = score_forward_subtb_batch(
        batch=batch,
        policy=policy,
        features=features,
        policy_input=policy_input,
        graph_context=graph,
        reward=reward,
    )

    root_row = int(batch.step_parent_state_ids[0].item())
    root_frontier_edges = scores.frontier_edge_ids[scores.frontier_row_ids.eq(root_row)]
    assert bool(root_frontier_edges.eq(1).any())


def test_score_subtb_batch_uses_full_legal_frontier_when_scoring_pruning_disabled() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    policy = _policy(hidden_dim=4)
    features = _features()
    policy_input = policy.build_policy_input(features, graph_context=graph, compute_align_score=False)
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 1), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.tensor([False]),
    )
    batch = prepare_subtb_batch(
        trajectories=trajectories,
        graph_context=graph,
    )
    reward = _reward_output(batch.states.num_states)

    pruned_action_space = policy.prepare_action_space(
        state=batch.states,
        graph_context=graph,
        policy_input=policy_input,
        training=True,
    )
    pruned_root_edges = pruned_action_space.frontier.edge_ids[
        pruned_action_space.frontier.row_ids.eq(int(batch.step_parent_state_ids[0].item()))
    ]
    assert not bool(pruned_root_edges.eq(1).any())

    scores = score_forward_subtb_batch(
        batch=batch,
        policy=policy,
        features=features,
        policy_input=policy_input,
        graph_context=graph,
        reward=reward,
        action_space=pruned_action_space,
    )

    root_row = int(batch.step_parent_state_ids[0].item())
    root_frontier_edges = scores.frontier_edge_ids[scores.frontier_row_ids.eq(root_row)]
    assert bool(root_frontier_edges.eq(1).any())


def test_prepare_action_space_applies_pruning_during_scoring_only_when_enabled() -> None:
    graph = _graph_context()
    base_policy = _policy(hidden_dim=4)
    policy = ForwardPolicy(
        state_encoder=base_policy.state_encoder,
        flow_estimator=base_policy.flow_estimator,
        state_flow_head=base_policy.state_flow_head,
        frontier_pruning=FrontierPruningConfig(
            enabled=True,
            threshold=0.95,
            min_keep_per_state=1,
            apply_train=True,
            apply_eval=True,
            apply_scoring=True,
            keep_recorded_edges_in_train=True,
        ),
    )
    features = _features()
    policy_input = policy.build_policy_input(features, graph_context=graph, compute_align_score=False)
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    action_space = policy.prepare_action_space(
        state=root,
        graph_context=graph,
        policy_input=policy_input,
        training=True,
        scoring=True,
    )

    assert int(action_space.frontier.edge_ids.numel()) == 1
    assert int(action_space.frontier.edge_ids[0].item()) == 2
