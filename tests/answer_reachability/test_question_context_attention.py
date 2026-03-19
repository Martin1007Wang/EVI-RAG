from __future__ import annotations

from dataclasses import replace

import torch

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
