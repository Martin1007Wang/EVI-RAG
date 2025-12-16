from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torch_geometric.data import Batch, Data

pytest.importorskip("torch_scatter")

from src.models.components.gflownet_actor import GFlowNetActor


@dataclass
class _DummyState:
    done: torch.Tensor
    selected_mask: torch.Tensor
    selection_order: torch.Tensor
    current_tail: torch.Tensor
    answer_hits: torch.Tensor


class _DummyEnv:
    def reset(self, batch: dict[str, torch.Tensor], *, device: torch.device) -> _DummyState:
        num_graphs = int(batch["node_ptr"].numel() - 1)
        num_edges = int(batch["edge_index"].size(1))
        return _DummyState(
            done=torch.zeros(num_graphs, dtype=torch.bool, device=device),
            selected_mask=torch.zeros(num_edges, dtype=torch.bool, device=device),
            selection_order=torch.zeros(num_edges, dtype=torch.long, device=device),
            current_tail=torch.zeros(num_graphs, dtype=torch.long, device=device),
            answer_hits=torch.zeros(num_graphs, dtype=torch.bool, device=device),
        )

    def action_mask_edges(self, state: _DummyState) -> torch.Tensor:
        return torch.ones_like(state.selected_mask, dtype=torch.bool)

    def frontier_mask_edges(self, state: _DummyState) -> torch.Tensor:
        return torch.ones_like(state.selected_mask, dtype=torch.bool)

    def step(self, state: _DummyState, actions: torch.Tensor, *, step_index: int) -> _DummyState:
        num_edges = int(state.selected_mask.numel())
        is_edge = actions < num_edges
        if bool(is_edge.any().item()):
            selected = state.selected_mask.clone()
            selected[actions[is_edge]] = True
            state.selected_mask = selected
        state.done = torch.ones_like(state.done, dtype=torch.bool)
        return state


class _DummyPolicy(nn.Module):
    def forward(
        self,
        edge_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        selected_mask: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_residual = torch.zeros(edge_tokens.size(0), device=edge_tokens.device, dtype=edge_tokens.dtype)
        stop_residual = torch.zeros(question_tokens.size(0), device=edge_tokens.device, dtype=edge_tokens.dtype)
        state_emb = question_tokens.to(dtype=edge_tokens.dtype)
        return edge_residual, stop_residual, state_emb


def test_gflownet_actor_rollout_smoke_no_name_error() -> None:
    graph = Data(
        num_nodes=3,
        edge_index=torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
        edge_attr=torch.zeros(2, dtype=torch.long),
        edge_labels=torch.tensor([1.0, 0.0], dtype=torch.float32),
        edge_scores=torch.tensor([0.2, 0.8], dtype=torch.float32),
        node_global_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        start_node_locals=torch.tensor([0], dtype=torch.long),
        start_entity_ids=torch.tensor([10], dtype=torch.long),
        answer_node_locals=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([11], dtype=torch.long),
        is_answer_reachable=torch.tensor([True], dtype=torch.bool),
    )
    batch = Batch.from_data_list([graph])

    device = torch.device("cpu")
    edge_index = batch.edge_index.to(device)
    edge_batch = batch.batch[edge_index[0]].to(device)
    edge_ptr = torch.tensor([0, int(edge_index.size(1))], dtype=torch.long, device=device)
    node_ptr = batch.ptr.to(device)

    edge_tokens = torch.zeros((edge_index.size(1), 4), dtype=torch.float32, device=device)
    question_tokens = torch.zeros((1, 4), dtype=torch.float32, device=device)
    edge_scores = batch.edge_scores.to(device)

    actor = GFlowNetActor(
        policy=_DummyPolicy(),
        env=_DummyEnv(),
        max_steps=1,
        policy_temperature=1.0,
        eval_policy_temperature=None,
        stop_logit_bias=0.0,
        random_action_prob=0.1,
        score_eps=1e-6,
        debug_actions=False,
        debug_actions_steps=0,
    ).eval()

    out = actor.rollout(
        batch=batch,
        edge_tokens=edge_tokens,
        question_tokens=question_tokens,
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        edge_scores=edge_scores,
        training=True,
        batch_idx=0,
    )
    assert out["log_pf"].shape == (1,)


def test_gflownet_actor_log_pf_matches_behavior_distribution_under_epsilon_greedy() -> None:
    graph = Data(
        num_nodes=3,
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros(2, dtype=torch.long),
        edge_labels=torch.tensor([0.0, 0.0], dtype=torch.float32),
        edge_scores=torch.tensor([0.2, 0.8], dtype=torch.float32),
        node_global_ids=torch.tensor([10, 11, 12], dtype=torch.long),
        start_node_locals=torch.tensor([0], dtype=torch.long),
        start_entity_ids=torch.tensor([10], dtype=torch.long),
        answer_node_locals=torch.empty(0, dtype=torch.long),
        answer_entity_ids=torch.empty(0, dtype=torch.long),
        is_answer_reachable=torch.tensor([False], dtype=torch.bool),
    )
    batch = Batch.from_data_list([graph])

    device = torch.device("cpu")
    edge_index = batch.edge_index.to(device)
    edge_batch = batch.batch[edge_index[0]].to(device)
    edge_ptr = torch.tensor([0, int(edge_index.size(1))], dtype=torch.long, device=device)
    node_ptr = batch.ptr.to(device)

    edge_tokens = torch.zeros((edge_index.size(1), 4), dtype=torch.float32, device=device)
    question_tokens = torch.zeros((1, 4), dtype=torch.float32, device=device)
    edge_scores = batch.edge_scores.to(device)

    epsilon = 0.25
    stop_bias = -1.0
    actor = GFlowNetActor(
        policy=_DummyPolicy(),
        env=_DummyEnv(),
        max_steps=1,
        policy_temperature=1.0,
        eval_policy_temperature=None,
        stop_logit_bias=stop_bias,
        random_action_prob=epsilon,
        score_eps=1e-6,
        debug_actions=False,
        debug_actions_steps=0,
    ).train()

    torch.manual_seed(0)
    out = actor.rollout(
        batch=batch,
        edge_tokens=edge_tokens,
        question_tokens=question_tokens,
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        edge_scores=edge_scores,
        training=True,
        batch_idx=0,
    )
    chosen = int(out["actions_seq"][0, 0].item())
    assert 0 <= chosen < int(edge_scores.numel())

    stop_weight = math.exp(stop_bias)
    denom = float(edge_scores.sum().item() + stop_weight)
    clean_edge_prob = edge_scores / denom
    uniform = 1.0 / float(edge_scores.numel() + 1)
    behavior_edge_prob = (1.0 - epsilon) * clean_edge_prob + epsilon * uniform
    expected_log_pf = torch.log(behavior_edge_prob[chosen])
    assert_close(out["log_pf"][0], expected_log_pf, rtol=1e-5, atol=1e-6)
