from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from torch.testing import assert_close
from torch_geometric.data import Batch, Data

pytest.importorskip("torch_scatter")

from src.models.components.gflownet_actor import GFlowNetActor
from src.models.components.gflownet_env import STOP_RELATION


class _DummyStateEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def precompute(
        self,
        *,
        node_ptr: torch.Tensor,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if question_tokens.dim() != 2 or question_tokens.size(-1) != self.hidden_dim:
            raise ValueError("question_tokens shape mismatch.")
        return question_tokens.to(dtype=node_tokens.dtype)

    def encode_state(self, *, cache: torch.Tensor, state: Any) -> torch.Tensor:
        return cache


@dataclass
class _DummyState:
    done: torch.Tensor
    selection_order: torch.Tensor
    step_counts: torch.Tensor
    answer_hits: torch.Tensor
    used_edge_mask: torch.Tensor
    actions: torch.Tensor
    action_embeddings: torch.Tensor
    graph: Any


@dataclass
class _DummyGraph:
    edge_index: torch.Tensor
    edge_batch: torch.Tensor
    node_is_start: torch.Tensor
    heads_global: torch.Tensor
    tails_global: torch.Tensor
    edge_scores_norm: torch.Tensor


class _DummyEnv:
    def __init__(self, max_steps: int = 1) -> None:
        self.max_steps = int(max_steps)

    def reset(self, batch: dict[str, torch.Tensor], *, device: torch.device) -> _DummyState:
        num_graphs = int(batch["node_ptr"].numel() - 1)
        num_edges = int(batch["edge_index"].size(1))
        edge_index = batch["edge_index"].to(device)
        edge_batch = batch["edge_batch"].to(device)
        node_global_ids = batch["node_global_ids"].to(device)
        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]
        node_is_start = torch.zeros(int(batch["node_ptr"][-1].item()), dtype=torch.bool, device=device)
        if batch["start_node_locals"].numel() > 0:
            node_is_start[batch["start_node_locals"].to(device).long()] = True
        graph = _DummyGraph(
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_is_start=node_is_start,
            heads_global=heads_global,
            tails_global=tails_global,
            edge_scores_norm=torch.zeros(num_edges, dtype=torch.float32, device=device),
        )
        return _DummyState(
            done=torch.zeros(num_graphs, dtype=torch.bool, device=device),
            selection_order=torch.full((num_edges,), -1, dtype=torch.long, device=device),
            step_counts=torch.zeros(num_graphs, dtype=torch.long, device=device),
            answer_hits=torch.zeros(num_graphs, dtype=torch.bool, device=device),
            used_edge_mask=torch.zeros(num_edges, dtype=torch.bool, device=device),
            actions=torch.full((num_graphs, self.max_steps + 1), STOP_RELATION, dtype=torch.long, device=device),
            action_embeddings=torch.zeros(num_graphs, self.max_steps, 0, dtype=torch.float32, device=device),
            graph=graph,
        )

    def candidate_edge_masks(self, state: _DummyState) -> tuple[torch.Tensor, torch.Tensor]:
        edge_batch = state.graph.edge_batch
        valid = ~state.done[edge_batch]
        forward = valid.clone()
        backward = torch.zeros_like(forward, dtype=torch.bool)
        return forward, backward

    def potential(self, state: _DummyState, *, valid_edges_override: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(int(state.done.numel()), device=state.done.device, dtype=torch.float32)

    def step(
        self,
        state: _DummyState,
        actions: torch.Tensor,
        action_embeddings: torch.Tensor,
        *,
        step_index: int,
    ) -> _DummyState:
        num_edges = int(state.used_edge_mask.numel())
        is_edge = actions >= 0
        if bool(is_edge.any().item()):
            selected = state.used_edge_mask.clone()
            order = state.selection_order.clone()
            selected[actions[is_edge]] = True
            order[actions[is_edge]] = int(step_index)
            state.used_edge_mask = selected
            state.selection_order = order
            state.step_counts = state.step_counts + is_edge.to(dtype=state.step_counts.dtype)
            state.actions[is_edge, step_index] = actions[is_edge]
            if state.action_embeddings.numel() != 0:
                state.action_embeddings[is_edge, step_index] = action_embeddings[is_edge]
        horizon = state.step_counts >= self.max_steps
        state.done = state.done | (actions == STOP_RELATION) | horizon
        return state


class _MultiStepEnv(_DummyEnv):
    pass


class _DummyPolicy(nn.Module):
    def forward(
        self,
        edge_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        valid_edges_mask: torch.Tensor,
        edge_direction: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_logits = torch.zeros(edge_tokens.size(0), device=edge_tokens.device, dtype=edge_tokens.dtype)
        batch_size = int(state_tokens.size(0))
        stop_logits = torch.zeros(batch_size, device=edge_tokens.device, dtype=edge_tokens.dtype)
        state_emb = state_tokens.to(dtype=edge_tokens.dtype)
        return edge_logits, stop_logits, state_emb


class _FixedLogitPolicy(nn.Module):
    def __init__(self, edge_logits: torch.Tensor, stop_logits: torch.Tensor) -> None:
        super().__init__()
        self.edge_logits = edge_logits
        self.stop_logits = stop_logits

    def forward(
        self,
        edge_tokens: torch.Tensor,
        state_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        valid_edges_mask: torch.Tensor,
        edge_direction: torch.Tensor | None = None,
        question_tokens: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_logits = self.edge_logits.to(device=edge_tokens.device, dtype=edge_tokens.dtype)
        stop_logits = self.stop_logits.to(device=edge_tokens.device, dtype=edge_tokens.dtype)
        state_emb = state_tokens.to(dtype=edge_tokens.dtype)
        return edge_logits, stop_logits, state_emb


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
    node_tokens = torch.zeros((int(batch.num_nodes), 4), dtype=torch.float32, device=device)
    question_tokens = torch.zeros((1, 4), dtype=torch.float32, device=device)
    state_encoder = _DummyStateEncoder(hidden_dim=4)
    actor = GFlowNetActor(
        policy=_DummyPolicy(),
        env=_DummyEnv(),
        state_encoder=state_encoder,
        max_steps=1,
        policy_temperature=1.0,
    ).eval()

    out = actor.rollout(
        batch=batch,
        edge_tokens=edge_tokens,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        batch_idx=0,
    )
    assert out["log_pf"].shape == (1,)


def test_gflownet_actor_log_pf_matches_policy_distribution() -> None:
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
    node_tokens = torch.zeros((int(batch.num_nodes), 4), dtype=torch.float32, device=device)
    question_tokens = torch.zeros((1, 4), dtype=torch.float32, device=device)
    edge_logits = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    stop_logits = torch.tensor([-0.4], dtype=torch.float32, device=device)
    state_encoder = _DummyStateEncoder(hidden_dim=4)
    actor = GFlowNetActor(
        policy=_FixedLogitPolicy(edge_logits=edge_logits, stop_logits=stop_logits),
        env=_DummyEnv(),
        state_encoder=state_encoder,
        max_steps=1,
        policy_temperature=1.0,
    ).train()

    out = actor.rollout(
        batch=batch,
        edge_tokens=edge_tokens,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        batch_idx=0,
    )
    chosen = int(out["actions_seq"][0, 0].item())
    valid_edges = torch.ones_like(edge_logits, dtype=torch.bool)
    log_edge, log_stop, _, _ = actor._log_probs_edges(
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        edge_batch=edge_batch,
        valid_edges=valid_edges,
        num_graphs=1,
        temp=1.0,
    )
    if chosen == STOP_RELATION:
        expected_log_pf = log_stop[0]
    else:
        expected_log_pf = log_edge[chosen]
    assert_close(out["log_pf"][0], expected_log_pf, rtol=1e-5, atol=1e-6)


def test_gflownet_actor_forced_actions_sequence_log_pf() -> None:
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
    node_tokens = torch.zeros((int(batch.num_nodes), 4), dtype=torch.float32, device=device)
    question_tokens = torch.zeros((1, 4), dtype=torch.float32, device=device)
    edge_logits = torch.tensor([0.2, 0.8], dtype=torch.float32, device=device)
    stop_logits = torch.tensor([-0.4], dtype=torch.float32, device=device)
    state_encoder = _DummyStateEncoder(hidden_dim=4)
    actor = GFlowNetActor(
        policy=_FixedLogitPolicy(edge_logits=edge_logits, stop_logits=stop_logits),
        env=_MultiStepEnv(max_steps=2),
        state_encoder=state_encoder,
        max_steps=2,
        policy_temperature=1.0,
    ).train()

    forced_actions_seq = torch.tensor([[0, 1, STOP_RELATION]], dtype=torch.long, device=device)
    out = actor.rollout(
        batch=batch,
        edge_tokens=edge_tokens,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        batch_idx=0,
        forced_actions_seq=forced_actions_seq,
    )

    # Actor excludes already-selected edges from the legal action set at each step.
    # Step 0: both edges are available.
    valid_edges0 = torch.ones_like(edge_logits, dtype=torch.bool)
    log_edge0, _, _, _ = actor._log_probs_edges(
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        edge_batch=edge_batch,
        valid_edges=valid_edges0,
        num_graphs=1,
        temp=1.0,
    )
    # Step 1: edge 0 is selected, only edge 1 remains.
    valid_edges1 = torch.tensor([False, True], dtype=torch.bool, device=device)
    log_edge1, _, _, _ = actor._log_probs_edges(
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        edge_batch=edge_batch,
        valid_edges=valid_edges1,
        num_graphs=1,
        temp=1.0,
    )
    # Step 2: both edges are selected, only stop remains (log prob = 0).
    valid_edges2 = torch.zeros_like(edge_logits, dtype=torch.bool)
    _, log_stop2, _, _ = actor._log_probs_edges(
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        edge_batch=edge_batch,
        valid_edges=valid_edges2,
        num_graphs=1,
        temp=1.0,
    )
    expected_log_pf = log_edge0[0] + log_edge1[1] + log_stop2[0]
    assert_close(out["log_pf"][0], expected_log_pf, rtol=1e-5, atol=1e-6)
