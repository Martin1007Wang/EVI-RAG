from __future__ import annotations

import torch
from torch.testing import assert_close

from src.models.components.gflownet_policies import EdgeLocalGATPolicy


def test_edge_local_gat_policy_masks_non_candidates() -> None:
    policy = EdgeLocalGATPolicy(hidden_dim=8, max_steps=2, dropout=0.0, direction_vocab_size=3).eval()
    edge_tokens = torch.randn(5, 8)
    question_tokens = torch.randn(2, 8)
    state_tokens = torch.randn(2, 8)
    edge_batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    selected_mask = torch.zeros(5, dtype=torch.bool)
    valid_edges_mask = torch.tensor([False, True, False, True, False], dtype=torch.bool)
    edge_direction = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    edge_logits, stop_logits, state_out = policy(
        edge_tokens,
        question_tokens,
        state_tokens,
        edge_batch,
        selected_mask,
        valid_edges_mask=valid_edges_mask,
        edge_direction=edge_direction,
    )

    assert edge_logits.shape == (5,)
    assert stop_logits.shape == (2,)
    assert state_out.shape == (2, 8)
    assert torch.isfinite(stop_logits).all()
    assert torch.isfinite(state_out).all()

    neg_inf = torch.finfo(edge_logits.dtype).min
    expected = torch.full_like(edge_logits, neg_inf)
    expected[valid_edges_mask] = 0.0  # zero-init last linear => finite uniform logits for candidates
    assert_close(edge_logits, expected)


def test_edge_local_gat_policy_all_stop_when_no_candidates() -> None:
    policy = EdgeLocalGATPolicy(hidden_dim=4, max_steps=2, dropout=0.0, direction_vocab_size=3).eval()
    edge_tokens = torch.randn(3, 4)
    question_tokens = torch.randn(1, 4)
    state_tokens = torch.randn(1, 4)
    edge_batch = torch.tensor([0, 0, 0], dtype=torch.long)
    selected_mask = torch.zeros(3, dtype=torch.bool)
    valid_edges_mask = torch.zeros(3, dtype=torch.bool)

    edge_logits, stop_logits, state_out = policy(
        edge_tokens,
        question_tokens,
        state_tokens,
        edge_batch,
        selected_mask,
        valid_edges_mask=valid_edges_mask,
    )

    neg_inf = torch.finfo(edge_logits.dtype).min
    assert_close(edge_logits, torch.full_like(edge_logits, neg_inf))
    assert torch.isfinite(stop_logits).all()
    assert torch.isfinite(state_out).all()

