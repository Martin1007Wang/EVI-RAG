from __future__ import annotations

import torch

from src.graph.segments import segment_logsumexp, segment_softmax
from src.weaver.context import DirectedAdjacencyIndex, GraphContext
from src.weaver.feature import FeatureEncoder, FeaturePack
from src.weaver.nn import PolicyCacheBuilder, StateEncoder
from src.weaver.policy.forward import EdgeFlowHead, ForwardPolicy, LowRankInteraction, StopFlowHead
from src.weaver.state import ExpansionBatch, StateBatch, cat_state_batches


def test_feature_encoder_builds_role_sensitive_edge_tokens() -> None:
    encoder = _feature_encoder()
    batch = _batch_for_encoder()
    features = encoder(batch)

    assert features.edge_h.shape == (2, 4)
    assert not torch.allclose(features.edge_h[0], features.edge_h[1])


def test_state_encoder_is_permutation_invariant_over_selected_edge_set() -> None:
    graph = _graph()
    state_a = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    ).advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    ).advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([1], dtype=torch.long),
        ),
        graph_context=graph,
    )
    state_b = StateBatch(
        graph_ids=state_a.graph_ids.clone(),
        selected_edge_count=state_a.selected_edge_count.clone(),
        selected_edge_ids=torch.tensor([[1, 0]], dtype=torch.long),
        activated_node_count=state_a.activated_node_count.clone(),
        activated_node_ids=state_a.activated_node_ids.clone(),
        anchor_count=state_a.anchor_count.clone(),
        budget=2,
    )
    features = _features()
    cache = PolicyCacheBuilder(hidden_dim=4)(features)
    encoder = StateEncoder(hidden_dim=4, max_budget=2)
    frontier_a = state_a.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=state_a.budget_left,
    )
    frontier_b = state_b.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=state_b.budget_left,
    )

    enc_a = encoder(
        state=state_a,
        cache=cache,
        frontier=frontier_a,
        remaining_budget=state_a.budget_left,
    )
    enc_b = encoder(
        state=state_b,
        cache=cache,
        frontier=frontier_b,
        remaining_budget=state_b.budget_left,
    )

    assert torch.allclose(enc_a.state_selected_h, enc_b.state_selected_h)


def test_state_encoder_initial_state_uses_query_fallback() -> None:
    graph = _graph()
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    features = _features()
    cache_builder = PolicyCacheBuilder(hidden_dim=4)
    cache = cache_builder(features)
    frontier = state.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=state.budget_left,
    )

    encoder = StateEncoder(hidden_dim=4, max_budget=2)
    encoding = encoder(
        state=state,
        cache=cache,
        frontier=frontier,
        remaining_budget=state.budget_left,
    )

    expected = cache.query_base_h_by_graph.index_select(0, state.graph_ids).float()

    assert torch.allclose(encoding.state_selected_h, expected)


def test_state_encoder_pools_selected_edges_with_query_attention() -> None:
    graph = _graph()
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    state = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    )
    features = _features()
    cache = PolicyCacheBuilder(hidden_dim=4)(features)
    frontier = state.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=state.budget_left,
    )
    encoder = StateEncoder(hidden_dim=4, max_budget=2)

    encoding = encoder(
        state=state,
        cache=cache,
        frontier=frontier,
        remaining_budget=state.budget_left,
    )

    selected = state.selected_edge_index()
    expected = encoder.query_pool(
        query_h=cache.query_base_h_by_graph.index_select(0, state.graph_ids).float(),
        tokens=cache.edge_h.index_select(0, selected.edge_ids),
        row_ids=selected.row_ids,
        num_rows=state.num_states,
    )

    assert torch.allclose(encoding.state_selected_h, expected)


def test_state_encoder_selected_representation_ignores_frontier_changes() -> None:
    graph = _graph()
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    state = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    )
    features = _features()
    cache = PolicyCacheBuilder(hidden_dim=4)(features)
    encoder = StateEncoder(hidden_dim=4, max_budget=2)

    frontier = state.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=state.budget_left,
    )
    changed_frontier = state.frontier(
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        remaining_budget=torch.zeros_like(state.budget_left),
    )
    encoding_a = encoder(
        state=state,
        cache=cache,
        frontier=frontier,
        remaining_budget=state.budget_left,
    )
    encoding_b = encoder(
        state=state,
        cache=cache,
        frontier=changed_frontier,
        remaining_budget=torch.zeros_like(state.budget_left),
    )

    assert torch.allclose(encoding_a.state_selected_h, encoding_b.state_selected_h)


def test_state_batch_active_node_index_includes_anchors() -> None:
    graph = _graph()
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )

    active = state.active_node_index(graph)

    assert active.row_ids.tolist() == [0]
    assert active.node_ids.tolist() == [0]


def test_policy_edge_flow_depends_on_selected_state_not_only_edge() -> None:
    graph = _graph()
    features = _features()
    policy = _policy(hidden_dim=4)
    with torch.no_grad():
        policy.edge_head.edge_unary.weight.zero_()
        policy.edge_head.edge_unary.bias.zero_()
        policy.edge_head.interaction.out.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float32))
        policy.edge_head.interaction.left_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        policy.edge_head.interaction.right_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    cache = policy.compute_cache(features)
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    expanded = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    )
    state = cat_state_batches([root, expanded])

    output = policy(
        state=state,
        features=features,
        cache=cache,
    )

    pos_root = ((output.frontier.row_ids == 0) & (output.frontier.edge_ids == 0)).nonzero(as_tuple=True)[0]
    pos_expanded = ((output.frontier.row_ids == 1) & (output.frontier.edge_ids == 1)).nonzero(as_tuple=True)[0]

    assert int(pos_root.numel()) == 1
    assert int(pos_expanded.numel()) == 1
    assert not torch.allclose(
        output.edge_log_flow.index_select(0, pos_root),
        output.edge_log_flow.index_select(0, pos_expanded),
    )


def test_policy_frontier_summary_uses_edge_log_flow_weighted_pooling() -> None:
    graph = _graph_with_frontier_competition()
    features = _features_with_frontier_competition()
    policy = _policy(hidden_dim=4)
    cache = policy.compute_cache(features)
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=3,
        graph_context=graph,
    )

    output = policy(
        state=state,
        features=features,
        cache=cache,
    )

    assert output.state_frontier_h is not None
    weights = segment_softmax(
        output.edge_log_flow,
        segment_ids=output.frontier.row_ids,
        num_segments=state.num_states,
    ).view(-1, 1)
    expected = torch.zeros_like(output.state_frontier_h)
    expected.scatter_add_(
        0,
        output.frontier.row_ids.view(-1, 1).expand_as(features.edge_h.index_select(0, output.frontier.edge_ids)),
        features.edge_h.index_select(0, output.frontier.edge_ids) * weights,
    )

    assert torch.allclose(output.state_frontier_h, expected)


def test_empty_frontier_produces_zero_frontier_summary() -> None:
    graph = _graph()
    features = _features()
    policy = _policy(hidden_dim=4)
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    cache = policy.compute_cache(features)

    output = policy(
        state=state,
        features=features,
        cache=cache,
        remaining_budget=torch.zeros(1, dtype=torch.long),
    )

    assert int(output.frontier.num_actions) == 0
    assert output.state_frontier_h is not None
    assert torch.allclose(output.state_frontier_h, torch.zeros_like(output.state_frontier_h))
    assert torch.isneginf(output.continue_log_flow).all()


def test_stop_head_responds_to_frontier_summary() -> None:
    interaction = LowRankInteraction(hidden_dim=4, rank=2)
    head = StopFlowHead(hidden_dim=4, interaction=interaction)
    selected = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32)
    frontier_a = torch.zeros((1, 4), dtype=torch.float32)
    frontier_b = torch.tensor([[0.0, 1.0, 0.0, 1.0]], dtype=torch.float32)
    with torch.no_grad():
        head.frontier_unary.weight.fill_(1.0)

    out_a = head(state_selected_h=selected, state_frontier_h=frontier_a)
    out_b = head(state_selected_h=selected, state_frontier_h=frontier_b)

    assert not torch.allclose(out_a, out_b)


def test_selected_edge_context_can_change_action_flow_for_same_candidate_edge() -> None:
    graph = _graph_with_frontier_competition()
    features = _features_with_frontier_competition()
    policy = _policy(hidden_dim=4)
    with torch.no_grad():
        policy.edge_head.edge_unary.weight.zero_()
        policy.edge_head.edge_unary.bias.zero_()
        policy.edge_head.interaction.out.weight.copy_(torch.tensor([[1.0, -1.0]], dtype=torch.float32))
        policy.edge_head.interaction.left_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
        policy.edge_head.interaction.right_proj.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    cache = policy.compute_cache(features)

    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=3,
        graph_context=graph,
    )
    expanded = root.advance(
        ExpansionBatch(
            state_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([0], dtype=torch.long),
        ),
        graph_context=graph,
    )
    batched_state = cat_state_batches([expanded, root])

    output = policy(
        state=batched_state,
        features=features,
        cache=cache,
    )

    expanded_state = 0
    root_state = 1
    edge_id = 2
    pos_expanded = ((output.frontier.row_ids == expanded_state) & (output.frontier.edge_ids == edge_id)).nonzero(as_tuple=True)[0]
    pos_root = ((output.frontier.row_ids == root_state) & (output.frontier.edge_ids == edge_id)).nonzero(as_tuple=True)[0]

    assert int(pos_expanded.numel()) == 1
    assert int(pos_root.numel()) == 1
    assert not torch.allclose(
        output.edge_log_flow.index_select(0, pos_expanded),
        output.edge_log_flow.index_select(0, pos_root),
    )


def test_segment_logsumexp_handles_low_precision_inputs() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    segment_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    out = segment_logsumexp(
        values=values,
        segment_ids=segment_ids,
        num_segments=3,
    )

    expected = torch.tensor(
        [
            torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0).item(),
            3.0,
            float("-inf"),
        ],
        dtype=torch.float32,
    )

    assert out.dtype == torch.float32
    assert torch.allclose(out[:2], expected[:2], atol=2e-3, rtol=1e-3)
    assert torch.isneginf(out[2])


def test_segment_softmax_handles_low_precision_inputs() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)
    segment_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    out = segment_softmax(
        logits,
        segment_ids=segment_ids,
        num_segments=2,
    )

    expected = torch.tensor(
        [
            0.26894143,
            0.7310586,
            1.0,
        ],
        dtype=torch.float32,
    )

    assert out.dtype == torch.float32
    assert torch.allclose(out, expected, atol=2e-3, rtol=1e-3)


def test_forward_policy_outputs_float32_inside_bfloat16_autocast() -> None:
    graph = _graph()
    features = _features()
    policy = _policy(hidden_dim=4)
    state = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
        graph_context=graph,
    )
    cache = policy.compute_cache(features)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output = policy(
            state=state,
            features=features,
            cache=cache,
        )

    assert output.state_log_flow.dtype == torch.float32
    assert output.stop_log_flow.dtype == torch.float32
    assert output.continue_log_flow.dtype == torch.float32
    assert output.edge_log_flow.dtype == torch.float32


def test_policy_cache_builder_ignores_anchor_features() -> None:
    features = _features()
    altered = FeaturePack(
        query_sem_h=features.query_sem_h,
        node_sem_h=features.node_sem_h,
        rel_sem_h=features.rel_sem_h,
        query_h=features.query_h,
        node_h=features.node_h.index_copy(
            0,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([[9.0, 9.0, 9.0, 9.0]], dtype=torch.float32),
        ),
        node_has_text=features.node_has_text,
        node_graph_ids=features.node_graph_ids,
        anchor_node_ids=features.anchor_node_ids,
        anchor_graph_ids=features.anchor_graph_ids,
        edge_h=features.edge_h,
        edge_src=features.edge_src,
        edge_dst=features.edge_dst,
        edge_graph_ids=features.edge_graph_ids,
        device=features.device,
    )

    builder = PolicyCacheBuilder(hidden_dim=4)
    cache_a = builder(features)
    cache_b = builder(altered)

    assert torch.allclose(cache_a.query_base_h_by_graph, cache_b.query_base_h_by_graph)


def _policy(*, hidden_dim: int) -> ForwardPolicy:
    interaction = LowRankInteraction(hidden_dim=hidden_dim, rank=2)
    return ForwardPolicy(
        cache_builder=PolicyCacheBuilder(hidden_dim=hidden_dim),
        state_encoder=StateEncoder(hidden_dim=hidden_dim, max_budget=3),
        stop_head=StopFlowHead(hidden_dim=hidden_dim, interaction=interaction),
        edge_head=EdgeFlowHead(hidden_dim=hidden_dim, interaction=interaction),
    )


def _features() -> FeaturePack:
    return FeaturePack(
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_sem_h=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        rel_sem_h=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        query_h=torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        node_h=torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        node_has_text=torch.tensor([True, True, True]),
        node_graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        anchor_graph_ids=torch.tensor([0], dtype=torch.long),
        edge_h=torch.tensor(
            [
                [0.4, 0.1, 0.2, 0.3],
                [0.2, 0.7, 0.1, 0.5],
            ],
            dtype=torch.float32,
        ),
        edge_src=torch.tensor([0, 1], dtype=torch.long),
        edge_dst=torch.tensor([1, 2], dtype=torch.long),
        edge_graph_ids=torch.tensor([0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )


def _features_with_frontier_competition() -> FeaturePack:
    return FeaturePack(
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_sem_h=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        rel_sem_h=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        query_h=torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        node_h=torch.tensor(
            [
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        node_has_text=torch.tensor([True, True, True, True]),
        node_graph_ids=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        anchor_graph_ids=torch.tensor([0], dtype=torch.long),
        edge_h=torch.tensor(
            [
                [0.3, 0.2, 0.4, 0.1],
                [0.2, 0.7, 0.1, 0.5],
                [0.9, 0.1, 0.3, 0.6],
            ],
            dtype=torch.float32,
        ),
        edge_src=torch.tensor([0, 1, 0], dtype=torch.long),
        edge_dst=torch.tensor([1, 2, 3], dtype=torch.long),
        edge_graph_ids=torch.tensor([0, 0, 0], dtype=torch.long),
        device=torch.device("cpu"),
    )


def _feature_encoder() -> FeatureEncoder:
    encoder = FeatureEncoder(
        entity_text_semantic_table=torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=torch.float32,
        ),
        text_row_by_entity_id=torch.tensor([0, 1, 2], dtype=torch.long),
        relation_semantic_table=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        hidden_dim=4,
    )
    with torch.no_grad():
        encoder.query_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
        encoder.src_proj.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]]))
        encoder.rel_proj.weight.copy_(torch.tensor([[0.2, 0.0], [0.0, 0.2], [1.0, 0.0], [0.0, 1.0]]))
        encoder.dst_proj.weight.copy_(torch.tensor([[0.3, 0.0], [0.0, 0.3], [0.7, 0.0], [0.0, 0.7]]))
    return encoder


class _Batch:
    def __init__(self) -> None:
        self.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        self.batch = torch.tensor([0, 0, 0], dtype=torch.long)
        self.anchor_node_ids = torch.tensor([0], dtype=torch.long)
        self.question_emb = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        self.node_entity_catalog_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        self.edge_relation_catalog_ids = torch.tensor([0, 1], dtype=torch.long)


def _batch_for_encoder() -> _Batch:
    return _Batch()


def _graph() -> GraphContext:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 2], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 1, 2, 2], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 1], dtype=torch.long),
            in_ptr=torch.tensor([0, 0, 1, 2], dtype=torch.long),
            edge_ids_by_dst=torch.tensor([0, 1], dtype=torch.long),
        ),
        num_nodes=3,
        num_edges=2,
        num_graphs=1,
    )


def _graph_with_frontier_competition() -> GraphContext:
    edge_index = torch.tensor([[0, 1, 0], [1, 2, 3]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_ptr=torch.tensor([0, 3], dtype=torch.long),
        anchor_mask=torch.tensor([True, False, False, False]),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        adjacency=DirectedAdjacencyIndex(
            out_ptr=torch.tensor([0, 2, 3, 3, 3], dtype=torch.long),
            edge_ids_by_src=torch.tensor([0, 2, 1], dtype=torch.long),
            in_ptr=torch.tensor([0, 0, 1, 2, 3], dtype=torch.long),
            edge_ids_by_dst=torch.tensor([0, 1, 2], dtype=torch.long),
        ),
        num_nodes=4,
        num_edges=3,
        num_graphs=1,
    )
