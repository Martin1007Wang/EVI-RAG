from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.policy import FlowEstimator, ForwardPolicy, StateFlowHead
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.trajectory import BUDGET_TRUNCATED


def _graph_context() -> GraphContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 1], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0], dtype=torch.long),
        ),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )


def test_rollout_engine_records_policy_stop_log_prob_for_budget_truncation() -> None:
    torch.manual_seed(0)
    graph = _graph_context()
    hidden_dim = 4
    policy = ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=hidden_dim, num_heads=1),
        flow_estimator=FlowEstimator(hidden_dim=hidden_dim),
        state_flow_head=StateFlowHead(state_dim=hidden_dim),
    )
    features = FeaturePack(
        question_h=torch.randn(1, hidden_dim),
        entity_h=torch.randn(graph.num_nodes, hidden_dim),
        edge_h=torch.randn(graph.num_edges, hidden_dim),
        relation_h=torch.randn(graph.num_edges, hidden_dim),
        frontier_prune_score=torch.randn(graph.num_edges, dtype=torch.float32),
    )
    policy_input = policy.build_policy_input(features, graph_context=graph)

    trajectories = RolloutEngine().sample(
        policy=policy,
        context=graph,
        features=features,
        policy_input=policy_input,
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=0,
    )

    assert int(trajectories.stop_reason[0].item()) == int(BUDGET_TRUNCATED)
    assert torch.isfinite(trajectories.stop_logp).all()
