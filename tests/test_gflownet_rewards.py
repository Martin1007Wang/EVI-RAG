from __future__ import annotations

import torch
from torch.testing import assert_close

from src.models.components.gflownet_rewards import GFlowNetReward


def test_gflownet_reward_energy_based() -> None:
    edge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    selected_mask = torch.tensor([1, 0, 1, 1], dtype=torch.bool)
    edge_scores = torch.tensor([2.0, -1.0, 0.0, 2.0], dtype=torch.float32)
    answer_hit = torch.tensor([1.0, 1.0], dtype=torch.float32)

    node_ptr = torch.tensor([0, 3, 6], dtype=torch.long)
    pair_start_node_locals = torch.tensor([0, 4], dtype=torch.long)
    pair_answer_node_locals = torch.tensor([2, 5], dtype=torch.long)
    pair_shortest_lengths = torch.tensor([1, 1], dtype=torch.long)
    start_node_hit = torch.tensor([0, 1], dtype=torch.long)
    answer_node_hit = torch.tensor([2, 2], dtype=torch.long)

    reward = GFlowNetReward(
        success_reward=1.0,
        failure_reward=0.01,
        semantic_coef=1.0,
        length_coef=1.0,
    )
    out = reward(
        selected_mask=selected_mask,
        edge_scores=edge_scores,
        edge_batch=edge_batch,
        answer_hit=answer_hit,
        pair_start_node_locals=pair_start_node_locals,
        pair_answer_node_locals=pair_answer_node_locals,
        pair_shortest_lengths=pair_shortest_lengths,
        start_node_hit=start_node_hit,
        answer_node_hit=answer_node_hit,
        node_ptr=node_ptr,
    )

    sig = torch.sigmoid(edge_scores)
    sem0 = sig[0]
    sem1 = (sig[2] + sig[3]) / 2.0
    expected_sem = torch.tensor([sem0, sem1], dtype=torch.float32)
    expected_len = torch.tensor([1.0, 2.0], dtype=torch.float32)
    expected_shortest = torch.tensor([1.0, 1.0], dtype=torch.float32)
    expected_cost = torch.tensor([0.0, 1.0], dtype=torch.float32)
    expected_log_reward = expected_sem - expected_cost

    assert_close(out.log_reward, expected_log_reward, atol=1e-6, rtol=0.0)
    assert_close(out.reward, torch.exp(expected_log_reward), atol=1e-6, rtol=0.0)
    assert_close(out.success, answer_hit, atol=0.0, rtol=0.0)
    assert_close(out.semantic_score, expected_sem, atol=1e-6, rtol=0.0)
    assert_close(out.length_cost, expected_cost, atol=1e-6, rtol=0.0)
    assert_close(out.path_len, expected_len, atol=1e-6, rtol=0.0)
    assert_close(out.shortest_len, expected_shortest, atol=1e-6, rtol=0.0)
