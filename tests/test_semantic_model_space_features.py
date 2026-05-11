from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] = torch.logaddexp(out[dest], src[row])
        return out

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out_shape = (size,) + tuple(src.shape[1:])
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        if dim_size is not None:
            size = dim_size
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        argmax = torch.full((size,), -1, dtype=torch.long, device=index.device)
        for row, dest in enumerate(index.tolist()):
            if argmax[dest] == -1 or src[row] > out[dest]:
                out[dest] = src[row]
                argmax[dest] = row
        return out, argmax

    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_sum = _scatter_sum
    sys.modules["torch_scatter"] = torch_scatter_stub


from src.weaver.config import build_policy_runtime_config
from src.weaver.nn.frontier_context import FrontierContext, frontier_semantic_scores
from src.weaver.nn.dde import DirectionalDDE
from src.weaver.nn.edge_encoder import EdgeEncoder
from src.weaver.nn.evidence_state_encoder import (
    EvidenceStateEncoder,
    StateContext,
)
from src.weaver.nn.evidence_tokens import build_evidence_tokens
from src.weaver.nn.edge_residual_scorer import EdgeResidualScorer
from src.weaver.nn.feature_encoder import FeatureBank, FeatureEncoder, node_incidence
from src.weaver.nn.frontier_builder import build_frontier
from src.weaver.nn.frontier_pointer import (
    FrontierPointerDiagnostics,
    FrontierPointerPolicy,
)
from src.weaver.policy import Policy, hazard_policy_log_probs
from src.weaver.state import RolloutState, State


def test_directional_dde_handles_chain_directions_and_batch_offsets() -> None:
    dde = DirectionalDDE(
        num_forward_rounds=2,
        num_backward_rounds=2,
        include_anchor_indicator=True,
    )

    chain = dde(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        num_nodes=3,
    )
    assert torch.allclose(
        chain,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        ),
    )

    reverse = dde(
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        anchor_node_ids=torch.tensor([2], dtype=torch.long),
        num_nodes=3,
    )
    assert torch.allclose(
        reverse[:, 3:],
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
    )

    batched = dde(
        edge_index=torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 2], dtype=torch.long),
        num_nodes=4,
    )
    assert torch.allclose(batched[:, 1], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_directional_dde_handles_multi_anchor_and_empty_anchor() -> None:
    dde = DirectionalDDE(
        num_forward_rounds=1,
        num_backward_rounds=0,
        include_anchor_indicator=True,
    )

    multi_anchor = dde(
        edge_index=torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 1], dtype=torch.long),
        num_nodes=3,
    )
    assert torch.allclose(multi_anchor[:, 1], torch.tensor([0.0, 0.0, 1.0]))

    no_anchor = dde(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        anchor_node_ids=torch.empty(0, dtype=torch.long),
        num_nodes=2,
    )
    assert torch.equal(no_anchor, torch.zeros((2, 2)))


def test_feature_encoder_splits_semantic_and_model_space() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        question_emb=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, -1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        non_text_node_mask=torch.tensor([False, True], dtype=torch.bool),
    )
    encoder = FeatureEncoder(
        entity_text_embeddings=torch.tensor([[0.6, 0.8], [5.0 / 13.0, 12.0 / 13.0]]),
        entity_embedding_map=torch.tensor([0, -1], dtype=torch.long),
        relation_embeddings=torch.tensor([[8.0 / 17.0, 15.0 / 17.0]]),
        embedding_dim=2,
        hidden_dim=2,
    )

    fb = encoder(batch)

    assert torch.allclose(fb.node_sem_h[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(fb.rel_sem_h.norm(dim=-1), torch.ones(1), atol=1e-6)

    assert fb.node_h.shape == fb.node_sem_h.shape
    assert torch.allclose(fb.rel_h, encoder.rel_projection(fb.rel_sem_h))
    assert torch.allclose(fb.query_h, encoder.query_projection(fb.query_sem_h))
    assert fb.edge_h.shape == (1, 2)
    assert torch.allclose(
        fb.edge_h,
        encoder.edge_encoder(
            src_h=fb.node_h.index_select(0, batch.edge_index[0]),
            rel_h=fb.rel_h,
            dst_h=fb.node_h.index_select(0, batch.edge_index[1]),
        ),
    )
    assert not torch.allclose(fb.query_h, fb.query_sem_h)


def test_feature_encoder_derives_non_text_mask_from_entity_embedding_map() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        question_emb=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0, 1, 1], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, -1, -1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
    )
    encoder = FeatureEncoder(
        entity_text_embeddings=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        entity_embedding_map=torch.tensor([0, 1, -1], dtype=torch.long),
        relation_embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        embedding_dim=2,
        hidden_dim=2,
    )

    fb = encoder(batch)

    assert fb.node_is_non_text is not None
    assert torch.equal(
        fb.node_is_non_text,
        torch.tensor([False, False, True], dtype=torch.bool),
    )

    semantic = frontier_semantic_scores(
        fb=fb,
        frontier=FrontierContext(
            edge_ids=torch.tensor([0, 1], dtype=torch.long),
            src=torch.tensor([0, 0], dtype=torch.long),
            dst=torch.tensor([1, 2], dtype=torch.long),
            graph_id=torch.tensor([0, 0], dtype=torch.long),
            src_active=torch.tensor([True, True], dtype=torch.bool),
            dst_active=torch.tensor([False, False], dtype=torch.bool),
        ),
    )

    assert torch.equal(
        semantic.new_text_mask,
        torch.tensor([True, False], dtype=torch.bool),
    )
    assert torch.allclose(semantic.query_new_node_score, torch.tensor([1.0, 0.0]))


def test_feature_encoder_supports_distinct_hidden_dim() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
        question_emb=torch.tensor([[3.0, 4.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        non_text_node_mask=torch.tensor([False], dtype=torch.bool),
    )
    encoder = FeatureEncoder(
        entity_text_embeddings=torch.tensor([[3.0, 4.0]]),
        entity_embedding_map=torch.tensor([0], dtype=torch.long),
        relation_embeddings=torch.tensor([[8.0, 15.0]]),
        embedding_dim=2,
        hidden_dim=3,
        role_projection_init="xavier",
    )

    fb = encoder(batch)

    assert fb.node_sem_h.shape == (1, 2)
    assert fb.rel_sem_h.shape == (1, 2)
    assert fb.query_sem_h.shape == (1, 2)
    assert fb.node_h.shape == (1, 3)
    assert fb.rel_h.shape == (1, 3)
    assert fb.query_h.shape == (1, 3)


def test_feature_encoder_caches_static_branching_features() -> None:
    batch = types.SimpleNamespace(
        node_entity_catalog_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        anchor_node_forward_distances_flat=torch.tensor([0, 1, 2], dtype=torch.long),
        anchor_node_backward_distances_flat=torch.tensor([0, 2, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        non_text_node_mask=torch.tensor([False, False, False], dtype=torch.bool),
    )
    encoder = FeatureEncoder(
        entity_text_embeddings=torch.eye(3, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1, 2], dtype=torch.long),
        relation_embeddings=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        embedding_dim=3,
        hidden_dim=3,
    )

    fb = encoder(batch)

    assert fb.edge_h.shape == (3, 3)
    assert fb.node_log_degree is not None
    assert torch.allclose(
        fb.node_log_degree, torch.log1p(torch.tensor([2.0, 2.0, 2.0]))
    )
    assert fb.edge_relation_log_frequency is not None
    assert torch.allclose(
        fb.edge_relation_log_frequency,
        torch.log1p(torch.tensor([2.0, 2.0, 1.0])),
    )


def test_edge_encoder_avoids_materializing_concat() -> None:
    encoder = EdgeEncoder(hidden_dim=2)
    with torch.no_grad():
        encoder.src_proj.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        encoder.src_proj.bias.copy_(torch.tensor([0.5, -0.25]))
        encoder.rel_proj.weight.copy_(torch.tensor([[0.5, 0.0], [0.0, 0.5]]))
        encoder.dst_proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, -1.0]]))

    src_h = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    rel_h = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    dst_h = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
    expected = (
        torch.nn.functional.linear(src_h, encoder.src_proj.weight, encoder.src_proj.bias)
        + torch.nn.functional.linear(rel_h, encoder.rel_proj.weight)
        + torch.nn.functional.linear(dst_h, encoder.dst_proj.weight)
    )

    assert torch.allclose(
        encoder(src_h=src_h, rel_h=rel_h, dst_h=dst_h),
        expected,
    )


def test_policy_caches_edge_features_for_pointer() -> None:
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "entity_text_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "entity_embedding_map": torch.tensor([0, 1], dtype=torch.long),
            "relation_embeddings": torch.tensor([[1.0, 0.0]]),
            "embedding_dim": 2,
            "hidden_dim": 2,
        },
    )
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=2,
        batch=torch.tensor([0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch=torch.tensor([0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(2, dtype=torch.bool),
    )

    assert policy.frontier_pointer.hidden_dim == policy.hidden_dim
    assert not hasattr(policy.state_encoder, "edge_encoder")
    context = policy.prepare_rollout_context(batch)
    assert context.fb.edge_h.shape == (1, 2)
    assert context.fb.edge_h.shape == (1, 2)


def test_policy_runtime_config_rejects_unknown_top_level_kwargs() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'unknown_key'"):
        build_policy_runtime_config(
            hidden_dim=2,
            unknown_key=True,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
        )


def test_policy_runtime_config_rejects_removed_legacy_config_kwarg() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument 'legacy_config'"):
        build_policy_runtime_config(
            hidden_dim=2,
            legacy_config={"top_k": 2},
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
        )


def test_policy_runtime_config_frontier_pointer_defaults_and_removed_keys() -> None:
    runtime = build_policy_runtime_config(
        hidden_dim=8,
        entity_text_embeddings=torch.eye(2, dtype=torch.float32),
        entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
        relation_embeddings=torch.eye(2, dtype=torch.float32),
    )

    assert runtime.mode == "bdb"
    assert runtime.flow_budget_conditioning == "additive"
    assert runtime.edge_scorer == "pointer"
    assert runtime.continuation_mass_reduction == "logsumexp"
    assert runtime.frontier_pointer_cfg == {}
    assert runtime.stop_head_cfg == {}

    with pytest.raises(ValueError, match="Removed policy keys"):
        build_policy_runtime_config(
            hidden_dim=8,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
            policy={"edge_policy_head": {}},
        )

    with pytest.raises(ValueError, match="policy.mode"):
        build_policy_runtime_config(
            hidden_dim=8,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
            policy={"mode": "te_bfm", "edge_scorer": "successor_value"},
        )

    with pytest.raises(ValueError, match="policy.edge_scorer"):
        build_policy_runtime_config(
            hidden_dim=8,
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
            policy={"mode": "bdb", "edge_scorer": "unknown"},
        )


def test_edge_residual_scorer_shapes_empty_and_backprop() -> None:
    scorer = EdgeResidualScorer(hidden_dim=2, feature_dim=3)

    empty = scorer(
        state_h=torch.ones((2, 2), dtype=torch.float32),
        edge_h=torch.empty((0, 2), dtype=torch.float32),
        query_h=torch.ones((2, 2), dtype=torch.float32),
        row_ids=torch.empty((0,), dtype=torch.long),
        edge_feat=torch.empty((0, 3), dtype=torch.float32),
    )
    assert empty.shape == (0,)

    logits = scorer(
        state_h=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        edge_h=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        query_h=torch.ones((2, 2), dtype=torch.float32),
        row_ids=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_feat=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )
    assert logits.shape == (3,)
    logits.sum().backward()
    first_linear = scorer.mlp[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.weight.grad is not None


def _tiny_frontier_fb() -> FeatureBank:
    return FeatureBank(
        node_h=torch.ones((4, 2), dtype=torch.float32),
        rel_h=torch.ones((3, 2), dtype=torch.float32),
        query_h=torch.ones((2, 2), dtype=torch.float32),
        edge_h=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32),
        node_sem_h=torch.zeros((4, 2), dtype=torch.float32),
        rel_sem_h=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        node_is_non_text=torch.zeros(4, dtype=torch.bool),
    )


def test_frontier_pointer_returns_logits_and_diagnostics() -> None:
    fb = _tiny_frontier_fb()
    batch = types.SimpleNamespace(
        num_graphs=2,
        num_nodes_total=4,
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 2], [1, 3, 3]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    state = State(
        active_nodes=torch.tensor([True, False, True, False]),
        active_edges=torch.zeros(3, dtype=torch.bool),
        root_edges=torch.zeros(3, dtype=torch.bool),
        expand_budget=2,
    )
    context = StateContext(
        state_h=torch.ones((2, 2), dtype=torch.float32),
        query_h=fb.query_h,
        node_h=fb.node_h,
        rel_h=fb.rel_h,
    )
    frontier = FrontierContext(
        edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        src=torch.tensor([0, 0, 2], dtype=torch.long),
        dst=torch.tensor([1, 3, 3], dtype=torch.long),
        graph_id=torch.tensor([0, 0, 1], dtype=torch.long),
        src_active=torch.tensor([True, True, True], dtype=torch.bool),
        dst_active=torch.tensor([False, False, False], dtype=torch.bool),
    )
    evidence_tokens, evidence_mask = build_evidence_tokens(
        fb=fb,
        batch=batch,
        state=state,
        query_h=context.query_h,
    )
    head = FrontierPointerPolicy(hidden_dim=2, num_heads=1)

    output = head(
        fb=fb,
        context=context,
        frontier_edge_ids=frontier.edge_ids,
        frontier_batch_ids=frontier.graph_id,
        frontier_context=frontier,
        evidence_tokens=evidence_tokens,
        evidence_mask=evidence_mask,
        return_diagnostics=True,
    )

    assert isinstance(output, FrontierPointerDiagnostics)
    assert output.final_logits.shape == (3,)
    assert output.frontier_h.shape == (3, 2)
    assert output.pointer_context.shape == (2, 2)
    assert torch.allclose(output.semantic_score, output.query_relation_score + output.query_new_node_score)


def test_frontier_pointer_backpropagates_through_pointer_score() -> None:
    torch.manual_seed(0)
    fb = _tiny_frontier_fb()
    batch = types.SimpleNamespace(
        num_graphs=2,
        num_nodes_total=4,
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 2], [1, 3, 3]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 1], dtype=torch.long),
    )
    state = State(
        active_nodes=torch.tensor([True, False, True, False]),
        active_edges=torch.zeros(3, dtype=torch.bool),
        root_edges=torch.zeros(3, dtype=torch.bool),
        expand_budget=2,
    )
    context = StateContext(
        state_h=torch.randn((2, 2), dtype=torch.float32),
        query_h=fb.query_h,
        node_h=fb.node_h,
        rel_h=fb.rel_h,
    )
    frontier = FrontierContext(
        edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        src=torch.tensor([0, 0, 2], dtype=torch.long),
        dst=torch.tensor([1, 3, 3], dtype=torch.long),
        graph_id=torch.tensor([0, 0, 1], dtype=torch.long),
        src_active=torch.tensor([True, True, True], dtype=torch.bool),
        dst_active=torch.tensor([False, False, False], dtype=torch.bool),
    )
    evidence_tokens, evidence_mask = build_evidence_tokens(
        fb=fb,
        batch=batch,
        state=state,
        query_h=context.query_h,
    )
    head = FrontierPointerPolicy(hidden_dim=2, num_heads=1)

    output = head(
        fb=fb,
        context=context,
        frontier_edge_ids=frontier.edge_ids,
        frontier_batch_ids=frontier.graph_id,
        frontier_context=frontier,
        evidence_tokens=evidence_tokens,
        evidence_mask=evidence_mask,
        return_diagnostics=True,
    )

    assert isinstance(output, FrontierPointerDiagnostics)
    output.final_logits.sum().backward()
    assert head.context_proj.weight.grad is not None
    assert head.frontier_proj.weight.grad is not None


def test_forward_policy_exposes_frontier_pointer_diagnostics() -> None:
    batch = types.SimpleNamespace(
        num_graphs=2,
        num_nodes_total=6,
        batch=torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 0, 4], [1, 2, 3, 5]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 4], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(6, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(4, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(6, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((6, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(6, dtype=torch.long),
            "relation_embeddings": torch.ones((4, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )
    out = policy(
        batch,
        State.create_initial(batch, expand_budget=2),
        return_edge_diagnostics=True,
    )

    assert torch.equal(out.frontier_edge_ids, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(out.frontier_batch_ids, torch.tensor([0, 0, 0, 1]))
    assert out.edge_logits.shape == (4,)
    assert out.stop_logits.shape == (2,)
    assert out.edge_policy_diagnostics is not None
    assert isinstance(out.edge_policy_diagnostics, FrontierPointerDiagnostics)
    assert out.edge_policy_diagnostics.final_logits.shape == out.edge_logits.shape
    assert torch.allclose(
        out.edge_expand_logprob,
        out.log_p_continue.index_select(0, out.frontier_batch_ids) + out.edge_cond_logprob,
        atol=1e-6,
    )


def test_default_policy_edge_logits_are_learned_pointer_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=3,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(3, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(2, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(3, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((3, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(3, dtype=torch.long),
            "relation_embeddings": torch.zeros((2, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )

    out = policy(batch, State.create_initial(batch, expand_budget=2))

    assert policy.edge_scorer == "pointer"
    assert out.edge_logits.shape == (2,)
    assert torch.isfinite(out.edge_logits).all()


def test_successor_policy_matches_full_action_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("REMOVED: successor-value action logits — see methodology.md §3.6")
    batch = types.SimpleNamespace(
        num_graphs=2,
        num_nodes_total=5,
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 3], [1, 2, 4]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 1], dtype=torch.long),
        anchor_node_ids=torch.tensor([0, 3], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(5, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(3, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(5, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        edge_scorer="successor_value",
        continuation_logit_bias_init=0.0,
        continuation_mass_reduction="logsumexp",
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((5, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(5, dtype=torch.long),
            "relation_embeddings": torch.zeros((3, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )
    successor_values = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float32)

    def _successor_log_mass(**_: object) -> torch.Tensor:
        return successor_values

    monkeypatch.setattr(policy, "_successor_edge_log_mass", _successor_log_mass)
    state = State.create_initial(batch, expand_budget=2)
    context = types.SimpleNamespace(
        state_h=torch.zeros((2, 2), dtype=torch.float32),
        query_h=torch.zeros((2, 2), dtype=torch.float32),
    )
    frontier_batch_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    edge_logits, log_c, log_z, stop_logits = policy._learned_action_terms(
        batch=batch,
        state=state,
        rollout_context=policy.prepare_rollout_context(batch),
        frontier_batch_ids=frontier_batch_ids,
        frontier_edge_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        context=context,
        frontier_context=types.SimpleNamespace(),
        stop_action_logits=torch.tensor([0.25, -0.75], dtype=torch.float32),
        num_graphs=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    log_p_stop, _, _, edge_expand_logprob = hazard_policy_log_probs(
        stop_logits=stop_logits,
        edge_logits=edge_logits,
        frontier_batch_ids=frontier_batch_ids,
        num_graphs=2,
    )

    graph0_z = torch.logsumexp(torch.tensor([0.25, 2.0, -1.0]), dim=0)
    graph1_z = torch.logsumexp(torch.tensor([-0.75, 0.5]), dim=0)
    assert torch.allclose(
        log_c,
        torch.stack(
            [torch.logsumexp(successor_values[:2], dim=0), torch.tensor(0.5)]
        ),
    )
    assert torch.allclose(log_z, torch.stack([graph0_z, graph1_z]))
    assert torch.allclose(
        log_p_stop,
        torch.tensor([0.25 - graph0_z, -0.75 - graph1_z]),
    )
    assert torch.allclose(
        edge_expand_logprob,
        torch.tensor([2.0 - graph0_z, -1.0 - graph0_z, 0.5 - graph1_z]),
    )


def test_successor_policy_ranks_bridge_by_successor_value_not_semantic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("REMOVED: successor-value action logits — see methodology.md §3.6")
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=3,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(3, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(2, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(3, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        edge_scorer="successor_value",
        continuation_logit_bias_init=0.0,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((3, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(3, dtype=torch.long),
            "relation_embeddings": torch.zeros((2, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )

    def _semantic_logits(**_: object) -> torch.Tensor:
        return torch.tensor([10.0, -10.0], dtype=torch.float32)

    def _successor_log_mass(**_: object) -> torch.Tensor:
        return torch.tensor([-2.0, 3.0], dtype=torch.float32)

    monkeypatch.setattr(policy, "_semantic_residual_edge_logits", _semantic_logits)
    monkeypatch.setattr(policy, "_successor_edge_log_mass", _successor_log_mass)

    out = policy(batch, State.create_initial(batch, expand_budget=2))

    assert torch.equal(out.frontier_edge_ids, torch.tensor([0, 1]))
    assert int(out.edge_logits.argmax().item()) == 1


def test_successor_value_policy_does_not_backprop_edge_loss_to_flow_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.skip("REMOVED: successor-value action logits — see methodology.md §3.6")
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=3,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(3, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(2, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(3, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        edge_scorer="successor_value",
        continuation_logit_bias_init=0.0,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((3, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(3, dtype=torch.long),
            "relation_embeddings": torch.zeros((2, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )

    def _state_context(
        *,
        fb: FeatureBank,
        state: State | RolloutState,
        **_: object,
    ) -> StateContext:
        rows = (
            state.num_rollouts
            if isinstance(state, RolloutState)
            else batch.num_graphs
        )
        state_h = torch.ones((int(rows), 2), dtype=torch.float32)
        query_h = torch.ones((int(rows), 2), dtype=torch.float32)
        return StateContext(
            state_h=state_h,
            query_h=query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
        )

    monkeypatch.setattr(policy.state_encoder, "forward", _state_context)

    out = policy(batch, State.create_initial(batch, expand_budget=2))
    out.edge_logits.sum().backward()

    flow_grads = [p.grad for p in policy.flow_head.parameters()]
    value_grads = [p.grad for p in policy.successor_value_head.parameters()]
    advantage_grads = [
        p.grad for p in policy.successor_edge_advantage_scorer.parameters()
    ]

    assert all(
        grad is None or torch.count_nonzero(grad).item() == 0
        for grad in flow_grads
    )
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in value_grads
    )
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in advantage_grads
    )


def test_no_scalar_state_head_changes_conditional_edge_distribution() -> None:
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=3,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        node_entity_catalog_ids=torch.arange(3, dtype=torch.long),
        edge_relation_catalog_ids=torch.arange(2, dtype=torch.long),
        question_emb=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        non_text_node_mask=torch.zeros(3, dtype=torch.bool),
    )
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "hidden_dim": 2,
            "entity_text_embeddings": torch.zeros((3, 2), dtype=torch.float32),
            "entity_embedding_map": torch.arange(3, dtype=torch.long),
            "relation_embeddings": torch.zeros((2, 2), dtype=torch.float32),
            "embedding_dim": 2,
            "dde": {"enabled": False},
        },
        frontier_pointer_cfg={"num_heads": 1},
    )
    out = policy(
        batch,
        State.create_initial(batch, expand_budget=2),
        return_edge_diagnostics=True,
    )

    assert out.edge_policy_diagnostics is not None
    assert hasattr(policy, "flow_head")
    assert out.state_log_flow.shape == (1,)
    assert out.edge_logits.shape == (2,)
    assert torch.allclose(
        out.edge_cond_logprob,
        out.edge_logits - torch.logsumexp(out.edge_logits, dim=0),
        atol=1e-6,
    )

def test_frontier_builder_matches_canonical_edge_definition() -> None:
    batch = types.SimpleNamespace(
        num_graphs=2,
        batch=torch.tensor([0, 0, 0, 0, 1], dtype=torch.long),
        edge_index=torch.tensor(
            [
                [0, 1, 2, 4],
                [1, 2, 3, 0],
            ],
            dtype=torch.long,
        ),
        edge_batch=torch.tensor([0, 0, 0, 1], dtype=torch.long),
    )
    fb = FeatureBank(
        node_h=torch.ones((5, 2), dtype=torch.float32),
        rel_h=torch.ones((4, 2), dtype=torch.float32),
        query_h=torch.ones((2, 2), dtype=torch.float32),
        edge_h=torch.arange(8, dtype=torch.float32).view(4, 2),
        node_sem_h=torch.ones((5, 2), dtype=torch.float32),
        rel_sem_h=torch.ones((4, 2), dtype=torch.float32),
        query_sem_h=torch.ones((2, 2), dtype=torch.float32),
    )
    state = State(
        active_nodes=torch.tensor([True, False, True, False, True], dtype=torch.bool),
        active_edges=torch.tensor([False, True, False, False], dtype=torch.bool),
        root_edges=torch.zeros(4, dtype=torch.bool),
        expand_budget=3,
        boundary_nodes=torch.tensor([True, False, True, False, True], dtype=torch.bool),
    )

    context = build_frontier(
        fb=fb,
        batch=batch,
        state=state,
        frontier_mode="boundary",
    )
    src, dst = batch.edge_index
    frontier = (
        state.boundary_nodes.index_select(0, src)
        & ~state.active_nodes.index_select(0, dst)
        & ~state.active_edges
    )
    expected_edge_ids = frontier.nonzero(as_tuple=False).flatten()
    expected_edge_batch = batch.edge_batch.index_select(0, expected_edge_ids)

    assert torch.equal(context.edge_ids, expected_edge_ids)
    assert torch.equal(context.graph_id, expected_edge_batch)
    assert not hasattr(context, "frontier_edge_h")


def test_rollout_evidence_state_encoder_sparse_trace_matches_dense_fallback() -> None:
    batch = types.SimpleNamespace(
        num_graphs=1,
        num_nodes_total=4,
        batch=torch.tensor([0, 0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 3]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
    )
    incident_edges, incident_ptr = node_incidence(
        edge_index=batch.edge_index,
        num_nodes=4,
    )
    fb = FeatureBank(
        node_h=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]],
            dtype=torch.float32,
        ),
        rel_h=torch.tensor(
            [[0.5, 0.5], [1.0, -1.0], [0.25, 0.75]],
            dtype=torch.float32,
        ),
        query_h=torch.tensor([[1.0, 0.25]], dtype=torch.float32),
        edge_h=torch.tensor(
            [[0.25, 0.5], [1.0, 0.0], [0.5, 0.25]],
            dtype=torch.float32,
        ),
        node_sem_h=torch.ones((4, 2), dtype=torch.float32),
        rel_sem_h=torch.ones((3, 2), dtype=torch.float32),
        query_sem_h=torch.ones((1, 2), dtype=torch.float32),
        node_incident_edge_ids=incident_edges,
        node_incident_ptr=incident_ptr,
    )
    state = RolloutState.create_initial(
        batch,
        expand_budget=2,
        rollout_to_graph=torch.tensor([0, 0], dtype=torch.long),
    )
    state.apply_expansion(
        rollout_ids=torch.tensor([0], dtype=torch.long),
        chosen_edges=torch.tensor([0], dtype=torch.long),
        edge_index=batch.edge_index,
    )

    readout = EvidenceStateEncoder(hidden_dim=2)
    sparse_context = readout(fb=fb, batch=batch, state=state)
    dense_state = RolloutState(
        rollout_to_graph=state.rollout_to_graph,
        expand_budget=state.expand_budget,
        edge_index=state.edge_index,
        num_nodes=state.num_nodes,
        num_edges=state.num_edges,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        root_edges=state.root_edges,
        anchor_nodes=state.anchor_nodes,
    )
    dense_state.anchor_node_trace = None
    dense_state.anchor_node_lengths = None
    dense_state.root_edge_trace = None
    dense_state.root_edge_lengths = None
    dense_state.expanded_edge_trace = None
    dense_state.expanded_edge_lengths = None
    dense_fb = FeatureBank(
        node_h=fb.node_h,
        rel_h=fb.rel_h,
        query_h=fb.query_h,
        edge_h=fb.edge_h,
        node_sem_h=fb.node_sem_h,
        rel_sem_h=fb.rel_sem_h,
        query_sem_h=fb.query_sem_h,
        node_incident_edge_ids=incident_edges,
        node_incident_ptr=incident_ptr,
    )
    dense_context = readout(fb=dense_fb, batch=batch, state=dense_state)

    assert torch.allclose(sparse_context.state_h, dense_context.state_h)
    assert sparse_context.node_pool is not None
    assert sparse_context.edge_pool is not None
    assert sparse_context.node_log_norm is not None
    assert sparse_context.edge_log_norm is not None


def test_evidence_state_encoder_does_not_materialize_frontier() -> None:
    batch = types.SimpleNamespace(
        num_graphs=1,
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
    )
    fb = FeatureBank(
        node_h=torch.ones((3, 2), dtype=torch.float32),
        rel_h=torch.ones((2, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        edge_h=torch.ones((2, 2), dtype=torch.float32),
        node_sem_h=torch.ones((3, 2), dtype=torch.float32),
        rel_sem_h=torch.ones((2, 2), dtype=torch.float32),
        query_sem_h=torch.ones((1, 2), dtype=torch.float32),
    )
    state = State(
        active_nodes=torch.tensor([True, False, False], dtype=torch.bool),
        active_edges=torch.tensor([False, False], dtype=torch.bool),
        root_edges=torch.zeros(2, dtype=torch.bool),
        expand_budget=2,
    )
    readout = EvidenceStateEncoder(hidden_dim=2)

    context = readout(fb=fb, batch=batch, state=state)

    assert isinstance(context, StateContext)
    assert context.state_h.shape == (1, 2)
    assert not hasattr(context, "frontier_edge_ids")
