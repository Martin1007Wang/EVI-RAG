from __future__ import annotations

import torch
import pytest

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeaturePack, StateEncoder
from src.weaver.policy import FlowEstimator, ForwardPolicy, StateFlowHead
from src.weaver.policy.output import PolicyOutput, STOP_EDGE_ID
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.state import ExpansionBatch, FrontierEncoding, StateBatch


def test_flow_estimator_matches_direct_formula() -> None:
    estimator = FlowEstimator(hidden_dim=4, relation_lambda=0.5)
    state_h = torch.randn(3, 4)
    question_h = torch.randn(3, 4)
    edge_h = torch.randn(3, 4)
    relation_h = torch.randn(3, 4)
    logits = estimator.score_edges(
        state_h=state_h,
        question_h=question_h,
        frontier_edge_h=edge_h,
        frontier_relation_h=relation_h,
    )
    relation_score = (question_h * relation_h).sum(dim=-1) * estimator.scale
    edge_score = (question_h * edge_h).sum(dim=-1) * estimator.scale
    phi_relation = relation_score + estimator.relation_lambda * edge_score
    phi_mgn = estimator.marginal_mlp(
        torch.cat([state_h, edge_h, state_h * edge_h], dim=-1)
    ).squeeze(-1)
    assert torch.allclose(logits, phi_relation + phi_mgn)


def test_flow_estimator_scores_stop_per_state() -> None:
    estimator = FlowEstimator(hidden_dim=4)
    question_h = torch.randn(2, 4)
    state_h = torch.randn(2, 4)
    stop_logits = estimator.score_stop(question_h=question_h, state_h=state_h)
    assert stop_logits.shape == (2,)


def test_policy_builds_active_nodes_once_for_frontier(monkeypatch) -> None:
    graph = _graph()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2, graph_context=graph)
    calls = 0
    original = StateBatch.active_node_index

    def capture(self, graph_context):
        nonlocal calls
        calls += 1
        return original(self, graph_context)

    monkeypatch.setattr(StateBatch, "active_node_index", capture)
    _policy()(state=state, features=_features(), graph_context=graph)
    assert calls == 1


def test_policy_reuses_prepared_action_space(monkeypatch) -> None:
    graph = _graph()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2, graph_context=graph)
    policy = _policy()
    action_space = policy.prepare_action_space(state=state, graph_context=graph)

    def fail_active(*args, **kwargs):
        raise AssertionError("prepared action space must avoid rebuilding active nodes")

    monkeypatch.setattr(StateBatch, "active_node_index", fail_active)
    output = policy(state=state, features=_features(), graph_context=graph, action_space=action_space)
    assert output.frontier.edge_ids.tolist() == [0]


def test_policy_output_vectorized_lookup_and_sampling() -> None:
    frontier = FrontierEncoding(
        row_ids=torch.tensor([0, 0, 1]),
        edge_ids=torch.tensor([2, 5, 3]),
    )
    output = PolicyOutput(
        action_logits=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
        action_row_ids=torch.tensor([0, 1, 0, 0, 1]),
        action_edge_ids=torch.tensor([STOP_EDGE_ID, STOP_EDGE_ID, 2, 5, 3]),
        frontier=frontier,
        log_flow=None,
    )
    gathered = output.gather_log_prob(
        row_ids=torch.tensor([0, 0, 1]),
        edge_ids=torch.tensor([STOP_EDGE_ID, 5, 3]),
    )
    assert torch.allclose(gathered, output.action_log_prob.index_select(0, torch.tensor([0, 3, 4])))
    sampled = output.sample(rows=torch.tensor([0, 1]))
    assert sampled.row_ids.tolist() == [0, 1]
    assert sampled.edge_ids[0].item() in {STOP_EDGE_ID, 2, 5}
    assert sampled.edge_ids[1].item() in {STOP_EDGE_ID, 3}
    assert output.forced_terminal_mask.tolist() == [False, False]


def test_trusted_advance_skips_redundant_frontier_validation(monkeypatch) -> None:
    graph = _graph()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2, graph_context=graph)

    def fail_frontier(*args, **kwargs):
        raise AssertionError("trusted advance must not reconstruct frontier")

    monkeypatch.setattr("src.weaver.state.frontier_from_graph", fail_frontier)
    advanced = state.advance(
        ExpansionBatch(state_ids=torch.tensor([0]), edge_ids=torch.tensor([0])),
        graph_context=graph,
        trusted=True,
    )
    assert advanced.edge_ids.tolist() == [[0, -1]]


def test_untrusted_advance_still_rejects_illegal_edge() -> None:
    graph = _graph()
    state = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2, graph_context=graph)
    with pytest.raises(ValueError, match="outside the legal frontier"):
        state.advance(
            ExpansionBatch(state_ids=torch.tensor([0]), edge_ids=torch.tensor([1])),
            graph_context=graph,
        )


def test_rollout_engine_uses_shared_policy_cache() -> None:
    graph = _graph()
    features = _features()
    policy = _policy()
    trajectories = RolloutEngine().sample(
        policy=policy,
        context=graph,
        features=features,
        cache=policy.build_cache(features),
        graph_ids=torch.tensor([0]),
        budget=2,
    )
    assert trajectories.num_trajectories == 1
    assert trajectories.edge_count.le(2).all()


class _FeatureBatch:
    edge_index = torch.tensor([[0, 1], [1, 2]])
    question_emb = torch.tensor([[1.0, 0.0, 0.0]])
    node_entity_catalog_ids = torch.tensor([0, 1, 2])
    edge_relation_catalog_ids = torch.tensor([0, 0])


def _policy() -> ForwardPolicy:
    return ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=4),
        flow_estimator=FlowEstimator(hidden_dim=4),
        state_flow_head=StateFlowHead(state_dim=4),
    )


def _features() -> FeaturePack:
    return FeaturePack(
        question_h=torch.randn(1, 4),
        entity_h=torch.randn(3, 4),
        edge_h=torch.randn(2, 4),
        relation_h=torch.randn(2, 4),
        device=torch.device("cpu"),
    )


def _graph() -> GraphContext:
    return GraphContext(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        node_to_graph=torch.zeros(3, dtype=torch.long),
        edge_to_graph=torch.zeros(2, dtype=torch.long),
        edge_ptr=torch.tensor([0, 2]),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 2, 2]),
            edge_ids_by_src=torch.tensor([0, 1]),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )
