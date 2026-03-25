from __future__ import annotations

from dataclasses import replace

import torch

from src.models.components.scoring import NodeFlowHead, TransitionPolicyHead
from src.models.gflownet import SearchState

from .conftest import make_policy, make_toy_batch


def test_transition_logits_change_with_question_context_tokens() -> None:
    torch.manual_seed(0)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch()
    ctx_a = torch.tensor(
        [
            [
                [2.0, -1.0, 0.5, 1.5, -0.5, 0.25, 1.0, -1.5],
                [-1.5, 0.75, 1.25, -0.25, 0.5, -0.5, 1.5, 0.0],
            ]
        ],
        dtype=batch.question_ctx.dtype,
    )
    ctx_b = -3.0 * ctx_a
    batch_a = replace(batch, question_ctx=ctx_a)
    batch_b = replace(batch, question_ctx=ctx_b)

    prepared_a = policy.prepare_batch(batch_a)
    prepared_b = policy.prepare_batch(batch_b)
    forward_state = SearchState(
        topology=prepared_a.topology,
        observation=prepared_a.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )
    backward_state = SearchState(
        topology=prepared_a.topology,
        observation=prepared_a.observation,
        current_nodes=torch.tensor([[1]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.ones((1, 1), dtype=torch.long),
        path_token_ids=SearchState.from_edge_path(
            topology=prepared_a.topology,
            observation=prepared_a.observation,
            start_node=0,
            edge_ids=(0,),
            max_steps=2,
            device=prepared_a.topology.edge_index.device,
        ).path_token_ids,
    )

    forward_a = policy.compute_forward_distribution(prepared_a, forward_state)
    forward_b = policy.compute_forward_distribution(prepared_b, forward_state)
    backward_a = policy.compute_backward_distribution(prepared_a, backward_state)
    backward_b = policy.compute_backward_distribution(prepared_b, backward_state)

    assert torch.equal(forward_a.edge_ids, forward_b.edge_ids)
    assert torch.equal(backward_a.edge_ids, backward_b.edge_ids)
    assert not torch.allclose(forward_a.edge_logits, forward_b.edge_logits)
    assert torch.allclose(backward_a.edge_logits, backward_b.edge_logits, atol=1.0e-6)


def test_masked_question_context_tokens_do_not_change_transition_logits() -> None:
    torch.manual_seed(1)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch()
    ctx_a = torch.tensor(
        [
            [
                [0.5, -1.0, 1.5, 0.0, -0.25, 0.75, 1.0, -0.5],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ],
        dtype=batch.question_ctx.dtype,
    )
    ctx_b = ctx_a.clone()
    ctx_b[:, 1, :] = 500.0
    mask = torch.tensor([[True, False]], dtype=torch.bool)
    batch_a = replace(batch, question_ctx=ctx_a, question_ctx_mask=mask)
    batch_b = replace(batch, question_ctx=ctx_b, question_ctx_mask=mask)

    prepared_a = policy.prepare_batch(batch_a)
    prepared_b = policy.prepare_batch(batch_b)
    state = SearchState(
        topology=prepared_a.topology,
        observation=prepared_a.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    forward_a = policy.compute_forward_distribution(prepared_a, state)
    forward_b = policy.compute_forward_distribution(prepared_b, state)

    assert torch.allclose(forward_a.edge_logits, forward_b.edge_logits, atol=1.0e-6)


def test_transition_logits_handle_bfloat16_autocast() -> None:
    torch.manual_seed(2)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        prepared = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared.topology,
        observation=prepared.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        distribution = policy.compute_forward_distribution(prepared, state)

    assert distribution.edge_logits.dtype == torch.float32
    assert torch.isfinite(distribution.edge_logits).all()


def test_transition_logits_handle_bfloat16_inputs_without_autocast() -> None:
    torch.manual_seed(2)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch().to("cpu", feature_dtype=torch.bfloat16)
    prepared = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared.topology,
        observation=prepared.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    distribution = policy.compute_forward_distribution(prepared, state)

    assert distribution.edge_logits.dtype == torch.float32
    assert torch.isfinite(distribution.edge_logits).all()


def test_transition_policy_head_matches_manual_external_chunking() -> None:
    torch.manual_seed(3)
    head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )

    current_state_features = torch.randn((9, 8), dtype=torch.float32)
    candidate_state_features = torch.randn((9, 8), dtype=torch.float32)
    relation_features = torch.randn((9, 8), dtype=torch.float32)
    full_logits = head(
        current_state_features,
        candidate_state_features,
        relation_features,
    )
    chunked_logits = torch.cat(
        [
            head(
                current_state_features[start:end],
                candidate_state_features[start:end],
                relation_features[start:end],
            )
            for start, end in ((0, 3), (3, 6), (6, 9))
        ],
        dim=0,
    )

    assert torch.allclose(full_logits, chunked_logits, atol=1.0e-6)


def test_node_flow_head_uses_question_features_directly() -> None:
    torch.manual_seed(4)
    head = NodeFlowHead(
        node_dim=8,
        question_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        conditioning="concat",
    )
    node_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
    question_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)

    scores = head(node_features, question_features)
    scores.sum().backward()

    assert node_features.grad is not None
    assert question_features.grad is not None
    assert float(node_features.grad.abs().sum().item()) > 0.0
    assert float(question_features.grad.abs().sum().item()) > 0.0


def test_node_flow_head_can_disable_direct_question_conditioning() -> None:
    torch.manual_seed(5)
    head = NodeFlowHead(
        node_dim=8,
        question_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        conditioning="none",
    )
    node_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
    question_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)

    scores = head(node_features, question_features)
    scores.sum().backward()

    assert node_features.grad is not None
    assert float(node_features.grad.abs().sum().item()) > 0.0
    assert question_features.grad is None


def test_node_flow_head_autocast_preserves_bfloat16_outputs() -> None:
    torch.manual_seed(5)
    head = NodeFlowHead(
        node_dim=8,
        question_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        conditioning="concat",
    )
    node_features = torch.randn((5, 8), dtype=torch.float32)
    question_features = torch.randn((5, 8), dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = head(node_features, question_features)

    assert scores.dtype == torch.bfloat16
    assert torch.isfinite(scores).all()


def test_transition_policy_head_preserves_end_to_end_gradients() -> None:
    torch.manual_seed(6)
    head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    current_state_features = torch.randn(
        (5, 8), dtype=torch.float32, requires_grad=True
    )
    candidate_state_features = torch.randn(
        (5, 8), dtype=torch.float32, requires_grad=True
    )
    relation_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
    logits = head(
        current_state_features,
        candidate_state_features,
        relation_features,
    )
    logits.sum().backward()

    assert current_state_features.grad is not None
    assert candidate_state_features.grad is not None
    assert relation_features.grad is not None
    assert float(current_state_features.grad.abs().sum().item()) > 0.0
    assert float(candidate_state_features.grad.abs().sum().item()) > 0.0
    assert float(relation_features.grad.abs().sum().item()) > 0.0


def test_transition_policy_head_autocast_preserves_bfloat16_outputs() -> None:
    torch.manual_seed(6)
    head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )
    current_state_features = torch.randn((5, 8), dtype=torch.float32)
    candidate_state_features = torch.randn((5, 8), dtype=torch.float32)
    relation_features = torch.randn((5, 8), dtype=torch.float32)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = head(
            current_state_features,
            candidate_state_features,
            relation_features,
        )

    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()


def test_start_control_state_depends_on_question_global_vector() -> None:
    torch.manual_seed(7)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch()
    prepared = policy.prepare_batch(batch)
    prepared_shifted = replace(
        prepared,
        question_tokens=prepared.question_tokens + 3.0,
    )

    start_nodes = torch.tensor([[0]], dtype=torch.long)
    control_a = policy.build_start_control_states(prepared, start_nodes)
    control_b = policy.build_start_control_states(prepared_shifted, start_nodes)

    assert not torch.allclose(control_a, control_b)


def test_transition_policy_head_can_explicitly_detach_encoder_features() -> None:
    torch.manual_seed(8)
    head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        detach_input_features=True,
    )
    current_state_features = torch.randn(
        (5, 8), dtype=torch.float32, requires_grad=True
    )
    candidate_state_features = torch.randn(
        (5, 8), dtype=torch.float32, requires_grad=True
    )
    relation_features = torch.randn((5, 8), dtype=torch.float32, requires_grad=True)
    logits = head(
        current_state_features,
        candidate_state_features,
        relation_features,
    )
    logits.sum().backward()

    assert current_state_features.grad is None
    assert candidate_state_features.grad is None
    assert relation_features.grad is None


def test_forward_distribution_backpropagates_into_shared_encoder() -> None:
    torch.manual_seed(9)
    policy = make_policy(max_steps=2)
    batch = make_toy_batch()
    batch.node_embeddings.requires_grad_()
    assert batch.edge_embeddings is not None
    batch.edge_embeddings.requires_grad_()
    assert batch.question_emb is not None
    batch.question_emb.requires_grad_()
    assert batch.question_ctx is not None
    batch.question_ctx.requires_grad_()

    prepared = policy.prepare_batch(batch)
    state = SearchState(
        topology=prepared.topology,
        observation=prepared.observation,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
    )

    distribution = policy.compute_forward_distribution(prepared, state)
    distribution.edge_logits.sum().backward()

    assert batch.node_embeddings.grad is not None
    assert batch.edge_embeddings.grad is not None
    assert batch.question_emb.grad is not None
    assert batch.question_ctx.grad is not None
    assert float(batch.node_embeddings.grad.abs().sum().item()) > 0.0
    assert float(batch.edge_embeddings.grad.abs().sum().item()) > 0.0
    assert float(batch.question_emb.grad.abs().sum().item()) > 0.0
    assert float(batch.question_ctx.grad.abs().sum().item()) > 0.0
