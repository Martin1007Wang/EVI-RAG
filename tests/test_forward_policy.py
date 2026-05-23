from __future__ import annotations

import torch

from src.weaver.context import GraphContext, build_directed_adjacency_index
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.nn.state_encoder import StateEncoder
from src.weaver.policy.forward import ForwardPolicy
from src.weaver.state import Frontier, State


def test_action_log_flows_accepts_keyword_only_segment_logsumexp() -> None:
    hidden_dim = 2
    state_encoder = StateEncoder(hidden_dim=hidden_dim)
    policy = ForwardPolicy(
        state_encoder=state_encoder,
        max_expand_budget=3,
    )

    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    graph = GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        adjacency=build_directed_adjacency_index(
            edge_index=edge_index,
            num_nodes=3,
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )
    state = State.initial(
        graph=graph,
        graph_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=3,
    )
    features = EncodedFeatures(
        node_text_semantic=torch.zeros((3, hidden_dim)),
        node_has_text=torch.tensor([True, True, True]),
        edge_relation_semantic=torch.zeros((2, hidden_dim)),
        query_semantic=torch.zeros((1, hidden_dim)),
        node_model=torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        ),
        edge_relation_model=torch.tensor(
            [
                [0.5, 1.5],
                [2.5, 3.5],
            ]
        ),
        query_model=torch.tensor([[7.0, 8.0]]),
        edge_token_model=torch.empty((2, hidden_dim * 3)),
    )
    frontier = Frontier(
        row_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([0, 1], dtype=torch.long),
    )

    encoding = state_encoder(
        features=features,
        state=state,
        context=graph,
    )
    budget_h = policy.encode_budget(state)

    (
        terminal_log_flow,
        continue_log_flow,
        state_log_flow,
        edge_logit,
        edge_log_prob,
        edge_log_flow,
        frontier_row_ids,
        frontier_edge_ids,
    ) = policy.action_log_flows(
        features=features,
        context=graph,
        query_h=encoding.query_h,
        state_h=encoding.row_state_h,
        budget_h=budget_h,
        frontier=frontier,
    )

    assert terminal_log_flow.shape == (1,)
    assert continue_log_flow.shape == (1,)
    assert state_log_flow.shape == (1,)
    assert edge_logit.shape == (2,)
    assert edge_log_prob.shape == (2,)
    assert edge_log_flow.shape == (2,)
    assert torch.equal(frontier_row_ids, frontier.row_ids)
    assert torch.equal(frontier_edge_ids, frontier.edge_ids)
    assert torch.allclose(
        state_log_flow,
        torch.logaddexp(terminal_log_flow, continue_log_flow),
    )
    assert torch.allclose(
        edge_log_flow,
        continue_log_flow.index_select(0, frontier.row_ids) + edge_log_prob,
    )


def test_forward_policy_stop_continue_depend_on_row_state_and_budget_only() -> None:
    hidden_dim = 4
    state_encoder = StateEncoder(hidden_dim=hidden_dim)
    policy = ForwardPolicy(
        state_encoder=state_encoder,
        max_expand_budget=3,
    )
    row_state_h = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    budget_h = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float32)

    score_a = policy.score_terminal_flow(
        query_h=torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32),
        state_h=row_state_h,
        budget_h=budget_h,
    )
    score_b = policy.score_terminal_flow(
        query_h=torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        state_h=row_state_h,
        budget_h=budget_h,
    )
    continue_a = policy.score_continue_flow(
        query_h=torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32),
        state_h=row_state_h,
        budget_h=budget_h,
    )
    continue_b = policy.score_continue_flow(
        query_h=torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32),
        state_h=row_state_h,
        budget_h=budget_h,
    )

    assert torch.allclose(score_a, score_b)
    assert torch.allclose(continue_a, continue_b)
