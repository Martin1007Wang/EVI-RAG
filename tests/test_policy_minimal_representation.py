from __future__ import annotations

import torch

from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeatureEncoder, FeaturePack, StateEncoder
from src.weaver.policy import FlowEstimator
from src.weaver.state import StateBatch


def test_state_encoder_returns_hidden_dim_states() -> None:
    encoder = StateEncoder(hidden_dim=4)
    state_h = encoder(
        question_h=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        selected_edge_h=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )
    assert state_h.shape == (1, 4)


def test_state_encoder_uses_empty_state_for_rows_without_selected_edges() -> None:
    encoder = StateEncoder(hidden_dim=4)
    empty_state_h = encoder(
        question_h=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        selected_edge_h=None,
    )
    non_empty_state_h = encoder(
        question_h=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        selected_edge_h=torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    )
    assert empty_state_h.shape == (1, 4)
    assert non_empty_state_h.shape == (1, 4)


def test_flow_estimator_produces_expected_shapes() -> None:
    estimator = FlowEstimator(hidden_dim=2)
    logits, stop_logits = estimator(
        question_h=torch.tensor([[3.0, 4.0], [3.0, 4.0]]),
        state_h=torch.tensor([[1.0, 2.0], [1.0, 2.0]]),
        frontier_edge_h=torch.zeros((2, 2)),
        frontier_relation_h=torch.zeros((2, 2)),
    )
    assert logits.shape == (2,)
    assert stop_logits.shape == (2,)


def test_feature_encoder_question_no_longer_changes_edge_h() -> None:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        text_row_by_entity_id=torch.tensor([0, 1]),
        entity_relation_neighborhood_semantic_table=torch.empty((0, 3)),
        relation_neighborhood_row_by_entity_id=torch.tensor([-1, -1]),
        relation_semantic_table=torch.tensor([[0.0, 1.0, 0.0]]),
        sem_dim=3,
        hidden_dim=3,
    )
    features_a = encoder(_feature_batch(torch.tensor([[10.0, 0.0, 0.0]])))
    features_b = encoder(_feature_batch(torch.tensor([[0.0, 10.0, 0.0]])))
    assert torch.allclose(features_a.edge_h, features_b.edge_h)


def test_feature_encoder_role_sensitive_edge_h() -> None:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        text_row_by_entity_id=torch.tensor([0, 1]),
        entity_relation_neighborhood_semantic_table=torch.empty((0, 3)),
        relation_neighborhood_row_by_entity_id=torch.tensor([-1, -1]),
        relation_semantic_table=torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
        sem_dim=3,
        hidden_dim=3,
    )
    batch_a = _feature_batch(torch.tensor([[1.0, 0.0, 0.0]]), edge_relation_catalog_ids=torch.tensor([0]))
    batch_b = _feature_batch(torch.tensor([[1.0, 0.0, 0.0]]), edge_relation_catalog_ids=torch.tensor([1]))
    features_a = encoder(batch_a)
    features_b = encoder(batch_b)
    assert not torch.allclose(features_a.edge_h, features_b.edge_h)
    assert not torch.allclose(features_a.edge_h.norm(dim=-1), torch.ones(1))


def test_pointer_policy_state_changes_frontier_ranking() -> None:
    torch.manual_seed(0)
    graph = _graph()
    features = FeaturePack(
        question_h=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        entity_h=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        ),
        edge_h=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        relation_h=torch.zeros(4, 4),
        device=torch.device("cpu"),
    )
    encoder = StateEncoder(hidden_dim=4)
    estimator = FlowEstimator(hidden_dim=4)
    optimizer = torch.optim.Adam([*encoder.parameters(), *estimator.parameters()], lr=0.03)
    root = StateBatch.initial(graph_ids=torch.tensor([0]), budget=2, graph_context=graph)
    expanded = root.advance(_expansion(0), graph_context=graph)
    for _ in range(200):
        optimizer.zero_grad()
        loss = _ranking_loss(encoder, estimator, root, features, graph, target_edge=0)
        loss = loss + _ranking_loss(encoder, estimator, expanded, features, graph, target_edge=2)
        loss.backward()
        optimizer.step()
    assert _top_edge(encoder, estimator, root, features, graph) == 0
    root_logits, _ = _edge_logits(encoder, estimator, root, features, graph)
    expanded_logits, _ = _edge_logits(encoder, estimator, expanded, features, graph)
    assert not torch.allclose(root_logits[:2], expanded_logits[:2])


def _ranking_loss(encoder, estimator, state, features, graph, *, target_edge: int) -> torch.Tensor:
    logits, frontier = _edge_logits(encoder, estimator, state, features, graph)
    target = frontier.edge_ids.eq(target_edge).nonzero(as_tuple=True)[0]
    return torch.nn.functional.cross_entropy(logits.view(1, -1), target)


def _top_edge(encoder, estimator, state, features, graph) -> int:
    logits, frontier = _edge_logits(encoder, estimator, state, features, graph)
    return int(frontier.edge_ids[int(logits.argmax().item())].item())


def _edge_logits(encoder, estimator, state, features, graph):
    frontier = state.frontier(graph_context=graph)
    selected = state.selected_edge_index()
    state_h = encoder(
        question_h=features.question_h.index_select(0, state.graph_ids)[0],
        selected_edge_h=(
            features.edge_h.index_select(0, selected.edge_ids)
            if int(selected.edge_ids.numel()) > 0
            else None
        ),
    )
    logits = estimator.score_edges(
        question_h=features.question_h.index_select(0, state.graph_ids).index_select(0, frontier.row_ids),
        state_h=state_h.expand(frontier.edge_ids.numel(), -1),
        frontier_edge_h=features.edge_h.index_select(0, frontier.edge_ids),
        frontier_relation_h=features.relation_h.index_select(0, frontier.edge_ids),
    )
    return logits, frontier


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 0, 1, 1], [1, 2, 3, 4]])
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.zeros(5, dtype=torch.long),
        edge_to_graph=torch.zeros(4, dtype=torch.long),
        edge_ptr=torch.tensor([0, 4]),
        anchor_mask=torch.tensor([True, False, False, False, False]),
        anchor_ptr=torch.tensor([0, 1]),
        anchor_node_ids=torch.tensor([0]),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 4, 4, 4, 4]),
            edge_ids_by_src=torch.tensor([0, 1, 2, 3]),
        ),
        num_nodes=5,
        num_edges=4,
        num_graphs=1,
    )


def _expansion(edge_id: int):
    from src.weaver.state import ExpansionBatch

    return ExpansionBatch(state_ids=torch.tensor([0]), edge_ids=torch.tensor([edge_id]))


class _FeatureBatch:
    def __init__(
        self,
        question_emb: torch.Tensor,
        *,
        node_entity_catalog_ids: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_relation_catalog_ids: torch.Tensor | None = None,
    ) -> None:
        self.edge_index = torch.tensor([[0], [1]]) if edge_index is None else edge_index
        self.question_emb = question_emb
        self.node_entity_catalog_ids = torch.tensor([0, 1]) if node_entity_catalog_ids is None else node_entity_catalog_ids
        self.edge_relation_catalog_ids = torch.tensor([0]) if edge_relation_catalog_ids is None else edge_relation_catalog_ids
        self.batch = torch.zeros(int(self.node_entity_catalog_ids.numel()), dtype=torch.long)


def _feature_batch(
    question_emb: torch.Tensor,
    *,
    node_entity_catalog_ids: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_relation_catalog_ids: torch.Tensor | None = None,
) -> _FeatureBatch:
    return _FeatureBatch(
        question_emb,
        node_entity_catalog_ids=node_entity_catalog_ids,
        edge_index=edge_index,
        edge_relation_catalog_ids=edge_relation_catalog_ids,
    )
