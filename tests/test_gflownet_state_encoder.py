from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch_scatter")

from src.models.components.gflownet_env import GraphEnv
from src.models.components.gflownet_state_encoder import GNNStateEncoder


def test_gnn_state_encoder_changes_with_state() -> None:
    env = GraphEnv(max_steps=2, mode="path", forbid_backtrack=False, forbid_revisit=False, debug=False)
    device = torch.device("cpu")

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
    edge_batch = torch.zeros(2, dtype=torch.long, device=device)
    node_global_ids = torch.tensor([10, 11, 12], dtype=torch.long, device=device)
    node_ptr = torch.tensor([0, 3], dtype=torch.long, device=device)
    edge_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)
    start_node_locals = torch.tensor([0], dtype=torch.long, device=device)
    start_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
    answer_node_locals = torch.empty(0, dtype=torch.long, device=device)
    answer_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)

    graph_dict = {
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "node_global_ids": node_global_ids,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "edge_scores": torch.zeros(2, dtype=torch.float32, device=device),
        "start_node_locals": start_node_locals,
        "start_ptr": start_ptr,
        "answer_node_locals": answer_node_locals,
        "answer_ptr": answer_ptr,
    }

    state = env.reset(graph_dict, device=device)

    hidden_dim = 8
    torch.manual_seed(0)
    node_tokens = torch.randn(3, hidden_dim, device=device)
    edge_tokens = torch.randn(2, hidden_dim, device=device)
    question_tokens = torch.randn(1, hidden_dim, device=device)

    encoder = GNNStateEncoder(
        hidden_dim=hidden_dim,
        max_steps=2,
        num_layers=1,
        dropout=0.0,
        use_start_state=True,
        use_question_film=True,
        use_direction_emb=True,
        direction_vocab_size=3,
    )
    cache = encoder.precompute(
        edge_index=edge_index,
        edge_batch=edge_batch,
        node_ptr=node_ptr,
        start_node_locals=start_node_locals,
        start_ptr=start_ptr,
        node_tokens=node_tokens,
        edge_tokens=edge_tokens,
        question_tokens=question_tokens,
    )

    s0 = encoder.encode_state(cache=cache, state=state)
    state = env.step(state, torch.tensor([0], dtype=torch.long, device=device), step_index=0)
    s1 = encoder.encode_state(cache=cache, state=state)

    assert s0.shape == (1, hidden_dim)
    assert s1.shape == (1, hidden_dim)
    assert not torch.allclose(s0, s1)
