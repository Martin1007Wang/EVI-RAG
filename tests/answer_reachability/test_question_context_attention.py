from __future__ import annotations

from dataclasses import replace

import torch

from src.models.components.scoring import TransitionPolicyHead
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


def test_transition_policy_head_microbatch_matches_full_batch() -> None:
    torch.manual_seed(3)
    full_head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        microbatch_size=128,
    )
    chunked_head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        microbatch_size=3,
    )
    chunked_head.load_state_dict(full_head.state_dict())

    current_state_features = torch.randn((9, 8), dtype=torch.float32)
    candidate_state_features = torch.randn((9, 8), dtype=torch.float32)
    relation_features = torch.randn((9, 8), dtype=torch.float32)
    full_logits = full_head(
        current_state_features,
        candidate_state_features,
        relation_features,
    )
    chunked_logits = chunked_head(
        current_state_features,
        candidate_state_features,
        relation_features,
    )

    assert torch.allclose(full_logits, chunked_logits, atol=1.0e-6)


def test_start_control_state_depends_on_question_global_vector() -> None:
    torch.manual_seed(4)
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


def test_transition_policy_head_detaches_encoder_features() -> None:
    torch.manual_seed(4)
    head = TransitionPolicyHead(
        state_dim=8,
        relation_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
        microbatch_size=32,
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
