from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("torch_geometric")
pytest.importorskip("torch_scatter")

from src.data.schema import RetrievalBatch
from src.models.modules.backbone import GNNBackbone
from src.models.modules.heads import ActionHead
from src.models.policy import Policy
from src.models.subgraph_state import SubgraphState


def _build_batch() -> RetrievalBatch:
    batch = RetrievalBatch()
    batch.node_tokens = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.5, 0.0, 1.0],
        ]
    )
    batch.edge_relation_tokens = torch.tensor(
        [
            [0.25, 0.0, 0.75, 0.0],
            [0.0, 0.25, 0.0, 0.75],
        ]
    )
    batch.question_emb = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    batch.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    batch.batch = torch.tensor([0, 0, 0], dtype=torch.long)
    batch.edge_batch = torch.tensor([0, 0], dtype=torch.long)
    batch.ptr = torch.tensor([0, 3], dtype=torch.long)
    batch.num_nodes = 3
    return batch


def _build_policy() -> Policy:
    return Policy(
        backbone_cfg={
            "embedding_dim": 4,
            "hidden_dim": 8,
            "use_adapter": False,
            "gnn_num_layers": 1,
            "gnn_dropout": 0.0,
            "gnn_use_question_conditioning": False,
        },
        hidden_dim=8,
    )


def _build_backbone() -> GNNBackbone:
    return GNNBackbone(
        embedding_dim=4,
        hidden_dim=8,
        use_adapter=False,
        gnn_num_layers=1,
        gnn_dropout=0.0,
        gnn_use_question_conditioning=False,
    )


def test_subgraph_state_copies_rollout_masks() -> None:
    active_nodes = torch.tensor([True, False, False])
    active_edges = torch.tensor([True, False])

    state = SubgraphState.from_tensors(active_nodes, active_edges)
    active_nodes[0] = False
    active_edges[0] = False

    assert state.active_nodes.tolist() == [True, False, False]
    assert state.active_edges.tolist() == [True, False]


def test_policy_backward_survives_rollout_mask_mutation() -> None:
    torch.manual_seed(0)

    policy = _build_policy()
    batch = _build_batch()
    active_nodes = torch.tensor([True, True, False])
    active_edges = torch.tensor([True, False])

    step_output = policy(
        batch,
        SubgraphState.from_tensors(active_nodes, active_edges),
    )

    # Emulate rollout continuing to mutate its live state after policy forward.
    active_nodes[2] = True
    active_edges[1] = True

    loss = (
        step_output.subgraph_h.sum()
        + step_output.action_logits["type_logits"].sum()
        + step_output.action_logits["expand_edge_logits"].sum()
    )
    loss.backward()

    assert policy.backbone.node_proj.weight.grad is not None
    assert policy.backbone.rel_proj.weight.grad is not None
    assert policy.state_encoder[0].weight.grad is not None
    assert policy.action_head.type_scorer[0].weight.grad is not None


def test_backbone_backward_survives_direct_mask_mutation() -> None:
    torch.manual_seed(0)

    backbone = _build_backbone()
    batch = _build_batch()
    active_edges = torch.tensor([True, False])

    node_h, edge_h, question_h = backbone(batch, active_edges=active_edges)

    # Emulate a direct caller mutating a live mask after the forward returns.
    active_edges[1] = True

    loss = node_h.sum() + edge_h.sum() + question_h.sum()
    loss.backward()

    assert backbone.node_proj.weight.grad is not None
    assert backbone.rel_proj.weight.grad is not None
    assert backbone.q_proj.weight.grad is not None


def test_backbone_respects_autocast_dtype_without_forcing_fp32() -> None:
    backbone = GNNBackbone(
        embedding_dim=4,
        hidden_dim=8,
        use_adapter=False,
        gnn_num_layers=0,
        gnn_dropout=0.0,
        gnn_use_question_conditioning=False,
    )
    batch = _build_batch()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        node_h, edge_h, question_h = backbone(batch)

    assert node_h.dtype == torch.bfloat16
    assert edge_h.dtype == torch.bfloat16
    assert question_h.dtype == torch.bfloat16


def test_policy_adds_relation_only_prior_to_expand_edge_logits() -> None:
    policy = _build_policy()
    batch = _build_batch()
    state = SubgraphState.from_tensors(
        torch.tensor([True, False, False]),
        torch.tensor([False, False]),
    )

    step_output = policy(batch, state)
    relation_prior = policy._build_relation_prior_logits(batch)
    node_query_scores = policy._build_node_query_scores(batch)
    node_h, edge_relation_h, _question_h = policy.backbone(
        batch, active_edges=state.active_edges
    )
    edge_state_h = policy._build_edge_state(
        batch=batch,
        node_h=node_h,
        edge_relation_h=edge_relation_h,
        active_nodes=state.active_nodes,
        relation_prior_logits=relation_prior,
        node_query_scores=node_query_scores,
    )
    subgraph_h, component_labels = policy._summarize_subgraph(
        batch=batch,
        node_h=node_h,
        edge_state_h=edge_state_h,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
    )
    edge_struct_features = policy._build_edge_struct_features(
        edge_index=batch.edge_index,
        active_nodes=state.active_nodes,
        component_labels=component_labels,
    )
    base_logits = policy.action_head(
        edge_state_h=edge_state_h,
        subgraph_h=subgraph_h,
        edge_batch_index=batch.edge_batch,
        edge_struct_features=edge_struct_features,
        expand_edge_prior_logits=torch.zeros_like(relation_prior),
    )["expand_edge_logits"]

    assert torch.allclose(
        step_output.action_logits["expand_edge_logits"],
        base_logits + policy.action_head.prior_scale * relation_prior,
        atol=1e-6,
    )


def test_expand_edge_residual_head_is_zero_initialized() -> None:
    policy = _build_policy()
    batch = _build_batch()
    state = SubgraphState.from_tensors(
        torch.tensor([True, False, False]),
        torch.tensor([False, False]),
    )

    relation_prior = policy._build_relation_prior_logits(batch)
    node_query_scores = policy._build_node_query_scores(batch)
    node_h, edge_relation_h, _question_h = policy.backbone(
        batch, active_edges=state.active_edges
    )
    edge_state_h = policy._build_edge_state(
        batch=batch,
        node_h=node_h,
        edge_relation_h=edge_relation_h,
        active_nodes=state.active_nodes,
        relation_prior_logits=relation_prior,
        node_query_scores=node_query_scores,
    )
    subgraph_h, component_labels = policy._summarize_subgraph(
        batch=batch,
        node_h=node_h,
        edge_state_h=edge_state_h,
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
    )
    edge_struct_features = policy._build_edge_struct_features(
        edge_index=batch.edge_index,
        active_nodes=state.active_nodes,
        component_labels=component_labels,
    )
    residual_logits = policy.action_head(
        edge_state_h=edge_state_h,
        subgraph_h=subgraph_h,
        edge_batch_index=batch.edge_batch,
        edge_struct_features=edge_struct_features,
        expand_edge_prior_logits=torch.zeros(batch.edge_index.size(1)),
    )["expand_edge_logits"]

    assert torch.allclose(residual_logits, torch.zeros_like(residual_logits), atol=1e-6)


def test_expand_edge_retrieval_head_is_zero_initialized() -> None:
    action_head = ActionHead(
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
    )

    action_logits = action_head(
        edge_state_h=torch.zeros((2, 4)),
        subgraph_h=torch.zeros((1, 4)),
        edge_batch_index=torch.tensor([0, 0], dtype=torch.long),
        edge_struct_features=torch.zeros((2, 5)),
        edge_discrimination_features=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 1.0], [0.6, 0.5, 0.4, 0.3, 0.2, 0.0]]
        ),
        expand_edge_prior_logits=torch.zeros(2),
    )

    assert torch.allclose(
        action_logits["expand_edge_logits"],
        torch.zeros_like(action_logits["expand_edge_logits"]),
        atol=1e-6,
    )


def test_policy_initial_expand_logits_equal_scaled_prior() -> None:
    policy = _build_policy()
    batch = _build_batch()
    state = SubgraphState.from_tensors(
        torch.tensor([True, False, False]),
        torch.tensor([False, False]),
    )

    step_output = policy(batch, state)
    relation_prior = policy._build_relation_prior_logits(batch)

    assert torch.allclose(
        step_output.action_logits["expand_edge_logits"],
        policy.action_head.prior_scale * relation_prior,
        atol=1e-6,
    )


def test_policy_initial_type_logits_remain_zero_despite_frontier_prior() -> None:
    policy = _build_policy()
    batch = _build_batch()
    state = SubgraphState.from_tensors(
        torch.tensor([True, False, False]),
        torch.tensor([False, False]),
    )

    step_output = policy(batch, state)

    assert torch.allclose(
        step_output.action_logits["type_logits"],
        torch.zeros_like(step_output.action_logits["type_logits"]),
        atol=1e-6,
    )


def test_policy_builds_hazard_type_features_from_legal_frontier() -> None:
    type_features = Policy._build_type_features(
        node_query_scores=torch.tensor([0.3, 0.7, 0.4]),
        relation_prior_logits=torch.tensor([0.9, 0.1, 0.8]),
        valid_edges_mask=torch.tensor([False, True, False]),
        active_nodes=torch.tensor([True, False, False]),
        active_edges=torch.tensor([False, False, False]),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        node_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        num_graphs=2,
    )

    assert type_features[0].tolist() == pytest.approx(
        [
            0.1,
            0.4,
            0.3,
            -0.1,
            torch.log1p(torch.tensor(1.0)).item(),
            torch.log1p(torch.tensor(1.0)).item(),
        ]
    )
    assert type_features[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_policy_type_features_ignore_non_finite_semantic_inputs() -> None:
    type_features = Policy._build_type_features(
        node_query_scores=torch.tensor([0.3, float("nan"), 0.4]),
        relation_prior_logits=torch.tensor([float("nan"), 0.2, float("inf")]),
        valid_edges_mask=torch.tensor([True, True, True]),
        active_nodes=torch.tensor([True, False, False]),
        active_edges=torch.tensor([False, False, False]),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        node_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        edge_batch_index=torch.tensor([0, 0, 1], dtype=torch.long),
        num_graphs=2,
    )

    assert torch.isfinite(type_features).all()


def test_policy_builds_edge_discrimination_features_from_tail_semantics() -> None:
    batch = _build_batch()
    node_h = torch.tensor(
        [
            [1.0, 0.0, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4],
            [0.2, 1.0, 0.0, 0.4, 0.4, 0.3, 0.2, 0.1],
            [0.7, 0.1, 0.3, 1.0, 0.6, 0.7, 0.8, 0.9],
        ]
    )
    edge_relation_h = torch.tensor(
        [
            [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.1, 0.4, 0.6, 0.5, 0.8, 0.7],
        ]
    )
    q_h = torch.tensor([[0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    active_nodes = torch.tensor([True, False, False])
    relation_prior_logits = torch.tensor([0.1, 0.2])
    node_query_scores_a = torch.tensor([0.3, 0.4, 0.5])
    node_query_scores_b = torch.tensor([0.3, 0.9, 0.5])

    features_a = Policy._build_edge_discrimination_features(
        batch=batch,
        node_h=node_h,
        edge_relation_h=edge_relation_h,
        q_h=q_h,
        active_nodes=active_nodes,
        relation_prior_logits=relation_prior_logits,
        node_query_scores=node_query_scores_a,
    )
    features_b = Policy._build_edge_discrimination_features(
        batch=batch,
        node_h=node_h,
        edge_relation_h=edge_relation_h,
        q_h=q_h,
        active_nodes=active_nodes,
        relation_prior_logits=relation_prior_logits,
        node_query_scores=node_query_scores_b,
    )

    assert not torch.allclose(features_a[0], features_b[0], atol=1e-6)


def test_type_scorer_is_zero_initialized_even_with_prior_feature() -> None:
    action_head = ActionHead(
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
    )

    action_logits = action_head(
        edge_state_h=torch.zeros((1, 4)),
        subgraph_h=torch.zeros((1, 4)),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        edge_struct_features=torch.zeros((1, 5)),
        type_features=torch.full((1, 6), 0.7),
    )

    assert action_logits["type_logits"].tolist() == pytest.approx([[0.0, 0.0]])


def test_action_head_sanitizes_non_finite_type_features() -> None:
    action_head = ActionHead(
        hidden_dim=4,
        num_layers=1,
        dropout=0.0,
    )

    action_logits = action_head(
        edge_state_h=torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
        subgraph_h=torch.tensor([[float("nan"), 1.0, float("inf"), -float("inf")]]),
        edge_batch_index=torch.tensor([0], dtype=torch.long),
        edge_struct_features=torch.tensor([[float("nan"), 0.0, 1.0, 0.0, 0.0]]),
        type_features=torch.full((1, 6), float("nan")),
        expand_edge_prior_logits=torch.tensor([float("nan")]),
    )

    assert torch.isfinite(action_logits["type_logits"]).all()
    assert torch.isfinite(action_logits["expand_edge_logits"]).all()


def test_relation_prior_logits_sanitize_non_finite_embeddings() -> None:
    policy = _build_policy()
    batch = _build_batch()
    batch.question_emb = torch.tensor([[float("nan"), 1.0, 0.0, float("inf")]])
    batch.edge_relation_tokens = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, float("nan"), 0.0, 0.0],
        ]
    )

    relation_prior = policy._build_relation_prior_logits(batch)

    assert torch.isfinite(relation_prior).all()


def test_edge_state_uses_inactive_endpoint_semantics() -> None:
    policy = _build_policy()
    batch = _build_batch()
    active_nodes = torch.tensor([True, False, False])
    edge_relation_h = torch.tensor(
        [
            [0.4, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8],
            [0.2, 0.3, 0.1, 0.4, 0.6, 0.5, 0.8, 0.7],
        ]
    )
    node_h_a = torch.tensor(
        [
            [1.0, 0.0, 0.5, 0.0, 0.1, 0.2, 0.3, 0.4],
            [0.0, 1.0, 0.0, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.5, 0.5, 0.0, 1.0, 0.6, 0.7, 0.8, 0.9],
        ]
    )
    node_h_b = node_h_a.clone()
    node_h_b[1] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0])
    relation_prior = torch.tensor([0.1, 0.2])
    node_query_scores = torch.tensor([0.3, 0.4, 0.5])

    edge_state_a = policy._build_edge_state(
        batch=batch,
        node_h=node_h_a,
        edge_relation_h=edge_relation_h,
        active_nodes=active_nodes,
        relation_prior_logits=relation_prior,
        node_query_scores=node_query_scores,
    )
    edge_state_b = policy._build_edge_state(
        batch=batch,
        node_h=node_h_b,
        edge_relation_h=edge_relation_h,
        active_nodes=active_nodes,
        relation_prior_logits=relation_prior,
        node_query_scores=node_query_scores,
    )

    assert not torch.allclose(edge_state_a[0], edge_state_b[0], atol=1e-6)
