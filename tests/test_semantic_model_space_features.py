from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
from src.weaver.nn.candidate_context import CandidateContext, candidate_semantic_scores
from src.weaver.nn.dde import DirectionalDDE
from src.weaver.nn.edge_scorer import EdgeScoreBreakdown, EdgeScorer
from src.weaver.nn.feature_encoder import FeatureBank, FeatureEncoder
from src.weaver.nn.flow_head import FlowHead
from src.weaver.nn.state_readout import StateContext, StateReadout
from src.weaver.nn.stop_gate import StopExpandGate
from src.weaver.nn.transition_features import TransitionFeatureBuilder
from src.weaver.policy import Policy
from src.weaver.state import State
from src.weaver.state_ops import frontier_edges


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
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
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

    semantic = candidate_semantic_scores(
        fb=fb,
        candidates=CandidateContext(
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

    assert fb.node_log_degree is not None
    assert torch.allclose(
        fb.node_log_degree, torch.log1p(torch.tensor([2.0, 2.0, 2.0]))
    )
    assert fb.edge_relation_log_frequency is not None
    assert torch.allclose(
        fb.edge_relation_log_frequency,
        torch.log1p(torch.tensor([2.0, 2.0, 1.0])),
    )


def test_policy_shares_edge_encoder_between_readout_and_residual() -> None:
    policy = Policy(
        hidden_dim=2,
        feature_encoder_cfg={
            "entity_text_embeddings": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "entity_embedding_map": torch.tensor([0, 1], dtype=torch.long),
            "relation_embeddings": torch.tensor([[1.0, 0.0]]),
            "embedding_dim": 2,
            "hidden_dim": 2,
        },
        edge_scorer_cfg={
            "share_edge_encoder_with_readout": True,
        },
    )

    assert policy.state_readout.edge_encoder is policy.edge_encoder
    assert policy.edge_scorer.edge_encoder is policy.edge_encoder
    assert policy.edge_scorer.residual_features == (
        "state",
        "edge",
        "transition_type",
        "semantic_prior",
    )
    assert policy.edge_scorer.residual_head[0].in_features == (
        4 + policy.transition_feature_builder.feature_dim + 1
    )
    assert not hasattr(policy.edge_scorer, "action_head")
    assert not hasattr(policy.edge_scorer, "candidate_encoder")


def test_policy_config_rejects_legacy_action_feature_switches() -> None:
    with pytest.raises(ValueError, match="transition_features"):
        build_policy_runtime_config(
            policy_cfg={
                "hidden_dim": 2,
                "feature_encoder": {
                    "embedding_dim": 2,
                    "hidden_dim": 2,
                    "dde": {"enabled": False},
                },
                "transition_features": {"use_semantic_weak_features": True},
            },
            entity_text_embeddings=torch.eye(2, dtype=torch.float32),
            entity_embedding_map=torch.tensor([0, 1], dtype=torch.long),
            relation_embeddings=torch.eye(2, dtype=torch.float32),
        )


def test_expand_edge_prior_uses_semantic_space_not_model_space() -> None:
    fb = FeatureBank(
        node_h=torch.zeros((2, 2), dtype=torch.float32),
        rel_h=torch.zeros((1, 2), dtype=torch.float32),
        query_h=torch.zeros((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    scorer = EdgeScorer(
        hidden_dim=2,
        type="semantic_prior_residual",
        entity_weight_init=2.0,
        logit_scale_init=3.0,
        residual_scale_init=1.0,
    )

    output = scorer(
        fb=fb,
        context=StateContext(
            state_h=torch.zeros((1, 2), dtype=torch.float32),
            query_h=fb.query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=torch.zeros(1, dtype=torch.float32),
        ),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert torch.allclose(output.query_relation_score, torch.tensor([0.5]))
    assert torch.allclose(output.query_new_node_score, torch.tensor([0.25]))
    assert torch.allclose(output.semantic_logits, torch.tensor([3.0]))
    assert torch.allclose(output.final_logits, output.semantic_logits)


def test_expand_edge_can_disable_residual_online_logits() -> None:
    fb = FeatureBank(
        node_h=torch.ones((2, 2), dtype=torch.float32),
        rel_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    scorer = EdgeScorer(
        hidden_dim=2,
        type="semantic_prior_residual",
        entity_weight_init=2.0,
        logit_scale_init=3.0,
        residual_scale_init=1.0,
        use_residual=False,
    )
    with torch.no_grad():
        scorer.residual_head[-1].bias.fill_(7.0)

    output = scorer(
        fb=fb,
        context=StateContext(
            state_h=torch.ones((1, 2), dtype=torch.float32),
            query_h=fb.query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=torch.zeros(1, dtype=torch.float32),
        ),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert torch.allclose(output.semantic_logits, torch.tensor([3.0]))
    assert torch.allclose(output.residual_logits, torch.zeros(1))
    assert torch.allclose(output.final_logits, output.semantic_logits)


def test_edge_scorer_can_freeze_prior_and_residual_parameters() -> None:
    scorer = EdgeScorer(
        hidden_dim=2,
        type="semantic_prior_residual",
        entity_weight_init=2.0,
        logit_scale_init=3.0,
        residual_scale_init=0.0,
        trainable_entity_weight=False,
        trainable_logit_scale=False,
        trainable_residual=False,
    )

    assert not scorer.entity_weight.requires_grad
    assert not scorer.logit_scale.requires_grad
    assert not scorer.residual_scale.requires_grad
    assert not any(
        parameter.requires_grad for parameter in scorer.residual_head.parameters()
    )


def test_edge_scorer_rejects_legacy_action_role() -> None:
    with pytest.raises(ValueError, match="semantic_prior_residual"):
        EdgeScorer(hidden_dim=2, type="action_role")


def test_edge_scorer_requires_zero_initialized_residual_head() -> None:
    with pytest.raises(ValueError, match="zero_init_residual_output=false"):
        EdgeScorer(hidden_dim=2, zero_init_residual_output=False)


def test_edge_scorer_residual_warmup_starts_from_zero() -> None:
    scorer = EdgeScorer(
        hidden_dim=2,
        residual_scale_init=1.0,
        residual_warmup_start_step=500,
        residual_warmup_steps=1500,
        residual_max_multiplier=1.0,
    )

    assert torch.allclose(scorer.effective_residual_scale(), torch.tensor(0.0))
    assert scorer.update_residual_schedule(step=500)[
        "residual_effective_scale"
    ] == pytest.approx(0.0)
    assert scorer.update_residual_schedule(step=1250)[
        "residual_effective_scale"
    ] == pytest.approx(0.5)
    assert scorer.update_residual_schedule(step=2000)[
        "residual_effective_scale"
    ] == pytest.approx(1.0)


def test_edge_scorer_detaches_semantic_prior_inside_residual() -> None:
    fb = FeatureBank(
        node_h=torch.ones((2, 2), dtype=torch.float32),
        rel_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    scorer = EdgeScorer(
        hidden_dim=2,
        type="semantic_prior_residual",
        entity_weight_init=0.0,
        logit_scale_init=3.0,
        residual_scale_init=1.0,
        residual_warmup_steps=0,
    )
    scorer.residual_head = torch.nn.Sequential(
        torch.nn.Linear(5, 1, bias=False),
    )
    with torch.no_grad():
        scorer.residual_head[0].weight.zero_()
        scorer.residual_head[0].weight[0, -1] = 1.0

    output = scorer(
        fb=fb,
        context=StateContext(
            state_h=torch.ones((1, 2), dtype=torch.float32),
            query_h=fb.query_h,
            node_h=fb.node_h,
            rel_h=fb.rel_h,
            progress=torch.zeros(1, dtype=torch.float32),
        ),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert torch.allclose(output.semantic_logits, torch.tensor([1.5]))
    assert torch.allclose(output.residual_logits, torch.tensor([1.5]))
    assert torch.allclose(output.final_logits, torch.tensor([3.0]))

    output.final_logits.sum().backward()
    assert scorer.logit_scale.grad is not None
    assert float(scorer.logit_scale.grad.item()) == pytest.approx(0.5)


def test_flow_head_uses_state_only() -> None:
    head = FlowHead(hidden_dim=2, num_layers=1, zero_init=False)
    with torch.no_grad():
        head.net.weight.copy_(torch.tensor([[1.0, 0.0]]))
        head.net.bias.zero_()

    state_h = torch.tensor([[2.0, 0.0]], dtype=torch.float32)
    flow_a = head(state_h=state_h)
    flow_b = head(state_h=state_h)

    assert torch.allclose(flow_a, flow_b)


def test_stop_gate_default_uses_state_only() -> None:
    gate = StopExpandGate(hidden_dim=2, use_frontier_summary=False, use_progress=False)
    with torch.no_grad():
        first = gate.net[0]
        last = gate.net[-1]
        first.weight.zero_()
        first.bias.zero_()
        last.weight.zero_()
        last.bias.zero_()
        first.weight[0, 0] = 1.0
        last.weight[0, 0] = 1.0

    kwargs = {
        "state_h": torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        "progress_ratio": torch.tensor([0.5], dtype=torch.float32),
        "has_candidate_edge": torch.tensor([True]),
    }

    stop_a, expand_a = gate(
        **kwargs,
        edge_logmeanexp=torch.tensor([100.0], dtype=torch.float32),
        edge_max=torch.tensor([120.0], dtype=torch.float32),
    )
    stop_b, expand_b = gate(
        **kwargs,
        edge_logmeanexp=torch.tensor([-100.0], dtype=torch.float32),
        edge_max=torch.tensor([-80.0], dtype=torch.float32),
    )

    assert torch.allclose(stop_a, stop_b)
    assert torch.allclose(expand_a, expand_b)
    assert torch.allclose(stop_a, torch.nn.functional.gelu(torch.tensor([0.5])))
    assert torch.allclose(expand_a, torch.zeros(1))


def test_transition_feature_builder_frontier_features_do_not_use_labels() -> None:
    batch = _NoLeakBatch(
        num_graphs=1,
        edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
        edge_batch=torch.tensor([0, 0], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0, 0], dtype=torch.long),
    )
    fb = FeatureBank(
        node_h=torch.zeros((3, 2), dtype=torch.float32),
        rel_h=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        query_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_sem_h=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        rel_sem_h=torch.tensor([[0.25, 0.0], [0.25, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_dde=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=torch.float32,
        ),
        node_is_non_text=torch.tensor([False, False, False], dtype=torch.bool),
    )
    context = StateContext(
        state_h=torch.zeros((1, 2), dtype=torch.float32),
        query_h=fb.query_h,
        node_h=fb.node_h,
        rel_h=fb.rel_h,
        progress=torch.tensor([0.25], dtype=torch.float32),
        relation_path_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )
    state = State(
        active_nodes=torch.tensor([True, False, False], dtype=torch.bool),
        active_edges=torch.tensor([False, False], dtype=torch.bool),
        root_edges=torch.tensor([False, True], dtype=torch.bool),
        expand_budget=4,
    )

    builder = TransitionFeatureBuilder(hidden_dim=2, dde_dim=2)
    features = builder(
        fb=fb,
        context=context,
        batch=batch,
        state=state,
        candidate_edge_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    name_to_idx = {name: idx for idx, name in enumerate(features.names)}

    assert features.names == (
        "src_active_dst_new",
        "dst_active_src_new",
        "both_active",
    )
    assert torch.equal(
        features.values[:, name_to_idx["src_active_dst_new"]],
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(
        features.values[:, name_to_idx["dst_active_src_new"]],
        torch.tensor([0.0, 1.0]),
    )
    assert torch.equal(features.values[:, name_to_idx["both_active"]], torch.zeros(2))
    assert "neither_active" not in name_to_idx
    assert "frontier_log_size" not in name_to_idx
    assert "relation_history_score" not in name_to_idx
    assert "query_relation" not in name_to_idx
    assert torch.allclose(features.query_relation_score, torch.tensor([0.25, 0.25]))
    assert torch.allclose(features.query_new_node_score, torch.tensor([0.0, 0.5]))


def test_semantic_prior_residual_scorer_preserves_prior_at_initialization() -> None:
    fb = FeatureBank(
        node_h=torch.ones((2, 2), dtype=torch.float32),
        rel_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_dde=torch.zeros((2, 0), dtype=torch.float32),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    context = StateContext(
        state_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=fb.query_h,
        node_h=fb.node_h,
        rel_h=fb.rel_h,
        progress=torch.zeros(1, dtype=torch.float32),
    )
    transition_features = TransitionFeatureBuilder(
        hidden_dim=2,
        dde_dim=0,
    )(
        fb=fb,
        context=context,
        batch=types.SimpleNamespace(
            num_graphs=1,
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_batch=torch.tensor([0], dtype=torch.long),
            edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        ),
        state=State(
            active_nodes=torch.tensor([True, False], dtype=torch.bool),
            active_edges=torch.tensor([False], dtype=torch.bool),
            root_edges=torch.tensor([False], dtype=torch.bool),
            expand_budget=1,
        ),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
    )
    scorer = EdgeScorer(
        hidden_dim=2,
        transition_feature_dim=transition_features.values.size(-1),
        entity_weight_init=2.0,
        logit_scale_init=3.0,
    )

    output = scorer(
        fb=fb,
        context=context,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        transition_features=transition_features,
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert output.final_logits.shape == (1,)
    assert output.residual_logits.shape == (1,)
    assert torch.allclose(output.residual_logits, torch.zeros(1))
    assert torch.allclose(output.final_logits, output.semantic_logits)
    assert torch.allclose(output.query_relation_score, torch.tensor([0.5]))
    assert torch.allclose(output.query_new_node_score, torch.tensor([0.25]))
    assert torch.allclose(output.semantic_logits, torch.tensor([3.0]))
    assert torch.allclose(output.residual_scale, torch.tensor(1.0))


def test_edge_scorer_reuses_cached_frontier_edge_h() -> None:
    class _FailingEdgeEncoder(torch.nn.Module):
        def forward(self, **kwargs):  # pragma: no cover - should not be called.
            del kwargs
            raise AssertionError("edge encoder should be reused from StateContext")

    fb = FeatureBank(
        node_h=torch.ones((2, 2), dtype=torch.float32),
        rel_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=torch.ones((1, 2), dtype=torch.float32),
        node_sem_h=torch.tensor([[0.0, 1.0], [0.25, 0.0]], dtype=torch.float32),
        rel_sem_h=torch.tensor([[0.5, 0.0]], dtype=torch.float32),
        query_sem_h=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        node_is_non_text=torch.tensor([False, False], dtype=torch.bool),
    )
    context = StateContext(
        state_h=torch.ones((1, 2), dtype=torch.float32),
        query_h=fb.query_h,
        node_h=fb.node_h,
        rel_h=fb.rel_h,
        progress=torch.zeros(1, dtype=torch.float32),
        frontier_edge_ids=torch.tensor([0], dtype=torch.long),
        frontier_edge_h=torch.tensor([[0.5, -0.5]], dtype=torch.float32),
    )
    scorer = EdgeScorer(
        hidden_dim=2,
        edge_encoder=_FailingEdgeEncoder(),  # type: ignore[arg-type]
        transition_feature_dim=0,
    )

    output = scorer(
        fb=fb,
        context=context,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        active_nodes=torch.tensor([True, False], dtype=torch.bool),
        candidate_edge_ids=torch.tensor([0], dtype=torch.long),
        candidate_context=CandidateContext(
            edge_ids=torch.tensor([0], dtype=torch.long),
            src=torch.tensor([0], dtype=torch.long),
            dst=torch.tensor([1], dtype=torch.long),
            graph_id=torch.tensor([0], dtype=torch.long),
            src_active=torch.tensor([True], dtype=torch.bool),
            dst_active=torch.tensor([False], dtype=torch.bool),
        ),
        return_breakdown=True,
    )

    assert isinstance(output, EdgeScoreBreakdown)
    assert output.final_logits.shape == (1,)


def test_state_readout_frontier_matches_legacy_frontier_edges() -> None:
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
        node_sem_h=torch.ones((5, 2), dtype=torch.float32),
        rel_sem_h=torch.ones((4, 2), dtype=torch.float32),
        query_sem_h=torch.ones((2, 2), dtype=torch.float32),
    )
    state = State(
        active_nodes=torch.tensor([True, False, True, False, True], dtype=torch.bool),
        active_edges=torch.tensor([False, True, False, False], dtype=torch.bool),
        root_edges=torch.zeros(4, dtype=torch.bool),
        expand_budget=3,
    )

    context = StateReadout(hidden_dim=2)(fb=fb, batch=batch, state=state)
    old_edge_ids, old_edge_batch = frontier_edges(batch=batch, state=state)

    assert torch.equal(context.frontier_edge_ids, old_edge_ids)
    assert torch.equal(context.frontier_edge_batch, old_edge_batch)
    assert context.frontier_edge_h is not None
    assert context.frontier_edge_h.shape == (old_edge_ids.numel(), 2)


def test_stop_gate_expand_logit_is_frontier_logsumexp() -> None:
    gate = StopExpandGate(
        hidden_dim=2,
        use_progress=True,
        use_frontier_summary=True,
        progress_penalty_init=0.5,
        trainable_progress_penalty=False,
    )

    stop, expand = gate(
        state_h=torch.zeros((2, 2), dtype=torch.float32),
        edge_logits=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        edge_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        progress_ratio=torch.tensor([0.5, 1.0], dtype=torch.float32),
        frontier_summary=torch.zeros((2, 3), dtype=torch.float32),
        has_candidate_edge=torch.tensor([True, True]),
    )

    assert torch.allclose(stop, torch.zeros(2))
    assert torch.allclose(
        expand,
        torch.tensor(
            [
                torch.logsumexp(torch.tensor([1.0, 2.0]), dim=0) - 0.25,
                3.0 - 0.5,
            ]
        ),
    )


class _NoLeakBatch(types.SimpleNamespace):
    @property
    def target_node_ids(self):  # pragma: no cover - only fails on policy leakage.
        raise AssertionError("target_node_ids must not be read by policy features")

    @property
    def node_target_distance(self):  # pragma: no cover
        raise AssertionError("node_target_distance must not be read by policy features")

    @property
    def target_shortest_path_edge_mask_flat(self):  # pragma: no cover
        raise AssertionError(
            "target_shortest_path_edge_mask_flat must not be read by policy features"
        )
