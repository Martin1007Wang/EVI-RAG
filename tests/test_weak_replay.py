from __future__ import annotations

import torch

from src.data.schema import RetrievalData
from src.data.collate import RetrievalCollator
from src.data.schema.fields import SampleFields
from src.graph.paths import compute_path_labels
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureEncoder
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.objectives.weak_replay import WeakReplayLoss
from src.weaver.policy import ForwardPolicy
from src.weaver.policy.output import PolicyOutput
from src.weaver.rollout.replay import WeakReplayBatch
from src.weaver.rollout.replay import WeakReplaySource
from src.weaver.state import ActionSpace
from src.weaver.state import StateBatch


def _edge_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_weak_replay_edges_are_batched_as_local_edge_ids() -> None:
    sample = RetrievalData(
        edge_index=_edge_tensor([(0, 1), (1, 2)]),
        num_nodes=3,
        node_entity_catalog_ids=torch.arange(3),
        edge_relation_catalog_ids=torch.arange(2),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=torch.tensor([2]),
        node_target_distance=torch.tensor([2, 1, 0]),
        weak_replay_edge_ids=torch.tensor([0, 1]),
        weak_replay_edge_weight=torch.tensor([1.0, 1.0]),
    )
    sample.sample_id = "toy"

    batch = RetrievalCollator()([sample])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)

    assert target.shortest_path_edge_mask.tolist() == [True, True]
    assert target.shortest_path_edge_weight.tolist() == [1.0, 1.0]


def test_weak_replay_source_collects_positive_frontier_states() -> None:
    edge_index = _edge_tensor([(0, 1), (1, 2), (0, 3)])
    labels = compute_path_labels(
        edge_index=edge_index,
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        num_nodes=4,
    )
    data = RetrievalData(
        edge_index=edge_index,
        num_nodes=4,
        node_entity_catalog_ids=torch.arange(4),
        edge_relation_catalog_ids=torch.arange(3),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([2]),
        reachable_target_node_ids=labels.reachable_target_node_ids,
        node_target_distance=labels.node_target_distance,
        weak_replay_edge_ids=torch.tensor([0, 1]),
        weak_replay_edge_weight=torch.tensor([1.0, 1.0]),
    )
    data.sample_id = "toy"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    weak = WeakReplaySource(budget=2, states_per_graph=4, branch_per_state=1).sample(
        graph=graph,
        target=target,
    )

    assert weak.num_states == 2
    assert weak.state.edge_count.tolist() == [0, 1]


def test_edge_residual_weight_zero_preserves_semantic_ranking() -> None:
    data = RetrievalData(
        edge_index=_edge_tensor([(0, 1), (0, 2)]),
        num_nodes=3,
        node_entity_catalog_ids=torch.arange(3),
        edge_relation_catalog_ids=torch.arange(2),
        question_emb=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([1]),
        reachable_target_node_ids=torch.tensor([1]),
        node_target_distance=torch.tensor([1, 0, -1]),
        weak_replay_edge_ids=torch.tensor([0]),
        weak_replay_edge_weight=torch.tensor([1.0]),
    )
    data.sample_id = "toy"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    features = FeatureEncoder(
        entity_text_semantic_table=torch.eye(4)[:3],
        text_row_by_entity_id=torch.arange(3),
        relation_semantic_table=torch.eye(4)[:2],
    )(batch)
    policy = ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=4),
        budget=2,
        edge_residual_weight=0.0,
    )
    with torch.no_grad():
        policy.edge_residual_head[-1].bias.fill_(10.0)

    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2)
    action_space = state.action_space(graph)
    query_h = policy.state_encoder.query_embeddings(features=features, state=state)
    semantic = policy.score_edge_semantic_prior(
        features=features,
        context=graph,
        query_h=query_h,
        action_space=action_space,
    )
    output = policy(
        features=features,
        state=state,
        context=graph,
        action_space=action_space,
    )

    assert torch.allclose(output.edge_raw_score, semantic)


def test_weak_replay_gate_skips_sufficient_prefix() -> None:
    data = RetrievalData(
        edge_index=_edge_tensor([(0, 1)]),
        num_nodes=2,
        node_entity_catalog_ids=torch.arange(2),
        edge_relation_catalog_ids=torch.arange(1),
        question_emb=torch.ones(4),
        anchor_node_ids=torch.tensor([0]),
        target_node_ids=torch.tensor([0]),
        reachable_target_node_ids=torch.tensor([0]),
        node_target_distance=torch.tensor([0, -1]),
        weak_replay_edge_ids=torch.tensor([0]),
        weak_replay_edge_weight=torch.tensor([1.0]),
    )
    data.sample_id = "toy"

    batch = RetrievalCollator()([data])
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=1)

    out = WeakReplayLoss(weight=1.0, gate_sufficient=True)(
        policy=_ConstantPolicy(),
        features=None,
        graph_context=graph,
        target_context=target,
        weak_replay=WeakReplayBatch(state=state),
    )

    assert torch.allclose(out.loss, torch.tensor(0.0))
    assert torch.allclose(out.metrics["weak_replay/active_state_count_before_gate"], torch.tensor(1.0))
    assert torch.allclose(out.metrics["weak_replay/gated_state_count"], torch.tensor(1.0))
    assert torch.allclose(out.metrics["weak_replay/sufficient_state_fraction"], torch.tensor(1.0))


class _ConstantPolicy:
    def __call__(self, *, features, state: StateBatch, context: GraphContext, action_space: ActionSpace) -> PolicyOutput:
        del features, context
        stop_log_flow = torch.zeros(state.num_states)
        edge_log_flow = torch.zeros(action_space.num_expansions)
        continue_log_flow = torch.zeros(state.num_states)
        state_log_flow = torch.logaddexp(stop_log_flow, continue_log_flow)
        return PolicyOutput(
            action_space=action_space,
            state_log_flow=state_log_flow,
            stop_log_flow=stop_log_flow,
            continue_log_flow=continue_log_flow,
            stop_log_prob=stop_log_flow - state_log_flow,
            continue_log_prob=continue_log_flow - state_log_flow,
            edge_log_flow=edge_log_flow,
            edge_log_prob=edge_log_flow - state_log_flow.index_select(0, action_space.expand_state_ids),
            conditional_edge_log_prob=edge_log_flow,
            edge_raw_score=edge_log_flow,
        )
