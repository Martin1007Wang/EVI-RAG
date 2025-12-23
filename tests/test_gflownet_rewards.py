from __future__ import annotations

import math

import torch
from torch.testing import assert_close

from src.models.components.gflownet_rewards import (
    AnswerAndPositiveEdgeF1Reward,
    AnswerOnlyReward,
)


def _toy_reward_inputs() -> dict[str, torch.Tensor]:
    # Two graphs in a flat (PyG-like) batch:
    #   Graph0: nodes {0,1,2}, edges (0-1),(1-2),(0-2), start=0, answer=2
    #   Graph1: nodes {3,4},   edges (3-4),(4-3),(3-3),(4-4),(3-4), start=3, answer=4
    edge_index = torch.tensor(
        [[0, 1, 0, 3, 4, 3, 4, 3], [1, 2, 2, 4, 3, 3, 4, 4]],
        dtype=torch.long,
    )
    edge_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long)
    node_ptr = torch.tensor([0, 3, 5], dtype=torch.long)
    start_node_locals = torch.tensor([0, 3], dtype=torch.long)
    answer_node_locals = torch.tensor([2, 4], dtype=torch.long)
    answer_node_ptr = torch.tensor([0, 1, 2], dtype=torch.long)

    # Edge labels: graph0 edges are positive; graph1 edges are negative.
    edge_labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    # Selected edges: take both edges in graph0 (hits answer), none in graph1.
    selected_mask = torch.tensor([True, True, False, False, False, False, False, False], dtype=torch.bool)

    return {
        "selected_mask": selected_mask,
        "edge_labels": edge_labels,
        "edge_batch": edge_batch,
        "edge_index": edge_index,
        "node_ptr": node_ptr,
        "start_node_locals": start_node_locals,
        "answer_node_locals": answer_node_locals,
        "answer_node_ptr": answer_node_ptr,
    }


def test_answer_pos_f1_reward_reduces_to_answer_only_when_coef_zero() -> None:
    inputs = _toy_reward_inputs()
    success_reward = 1.0
    failure_action_topk = 2
    failure_path_length = 2.0
    log_failure_bias = 0.0

    base = AnswerOnlyReward(
        success_reward=success_reward,
        failure_action_topk=failure_action_topk,
        failure_path_length=failure_path_length,
        log_failure_bias=log_failure_bias,
    )(**inputs)
    shaped = AnswerAndPositiveEdgeF1Reward(
        success_reward=success_reward,
        failure_action_topk=failure_action_topk,
        failure_path_length=failure_path_length,
        log_failure_bias=log_failure_bias,
        pos_f1_coef=0.0,
    )(**inputs)

    assert_close(shaped.log_reward, base.log_reward, atol=0.0, rtol=0.0)
    assert_close(shaped.reward, base.reward, atol=0.0, rtol=0.0)


def test_answer_pos_f1_reward_adds_log_shaping_on_positive_edge_f1() -> None:
    inputs = _toy_reward_inputs()
    success_reward = 1.0
    failure_action_topk = 2
    failure_path_length = 2.0
    log_failure_bias = 0.0
    coef = 2.0

    shaped = AnswerAndPositiveEdgeF1Reward(
        success_reward=success_reward,
        failure_action_topk=failure_action_topk,
        failure_path_length=failure_path_length,
        log_failure_bias=log_failure_bias,
        pos_f1_coef=coef,
    )(**inputs)

    # Graph0: answer hit, pos_f1=1 -> log_reward = log(success) + coef
    # Graph1: answer miss, pos_f1=0 -> log_reward = log_failure
    nodes_graph1 = 2.0
    avg_degree_graph1 = 5.0 / 2.0
    k_eff_graph1 = min(avg_degree_graph1, float(failure_action_topk))
    log_failure_graph1 = -math.log(nodes_graph1) - failure_path_length * math.log(k_eff_graph1) + log_failure_bias
    expected = torch.tensor([math.log(success_reward) + coef, log_failure_graph1], dtype=torch.float32)
    assert_close(shaped.log_reward, expected, atol=1e-6, rtol=0.0)
    assert_close(shaped.reward, torch.exp(expected), atol=1e-6, rtol=0.0)
