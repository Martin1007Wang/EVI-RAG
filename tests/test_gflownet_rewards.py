from __future__ import annotations

import math

import torch
from torch.testing import assert_close

from src.models.components.gflownet_rewards import GFlowNetReward


def test_gflownet_reward_uses_env_answer_hit_and_f1_shaping() -> None:
    edge_batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    selected_mask = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    edge_labels = torch.tensor([1, 1, 1, 0], dtype=torch.float32)
    answer_hit = torch.tensor([1.0, 0.0], dtype=torch.float32)

    reward = GFlowNetReward(
        success_reward=1.0,
        failure_reward=0.01,
        shaping_coef=1.0,
    )
    out = reward(
        selected_mask=selected_mask,
        edge_labels=edge_labels,
        edge_batch=edge_batch,
        answer_hit=answer_hit,
    )

    f1_graph0 = 2.0 / 3.0
    f1_graph1 = 1.0
    expected_log_reward = torch.tensor(
        [f1_graph0, math.log(0.01) + f1_graph1],
        dtype=torch.float32,
    )
    expected_precision = torch.tensor([1.0, 1.0], dtype=torch.float32)
    expected_recall = torch.tensor([0.5, 1.0], dtype=torch.float32)
    expected_f1 = torch.tensor([f1_graph0, f1_graph1], dtype=torch.float32)

    assert_close(out.log_reward, expected_log_reward, atol=1e-6, rtol=0.0)
    assert_close(out.reward, torch.exp(expected_log_reward), atol=1e-6, rtol=0.0)
    assert_close(out.success, answer_hit, atol=0.0, rtol=0.0)
    assert_close(out.pos_precision, expected_precision, atol=1e-6, rtol=0.0)
    assert_close(out.pos_recall, expected_recall, atol=1e-6, rtol=0.0)
    assert_close(out.pos_f1, expected_f1, atol=1e-6, rtol=0.0)
