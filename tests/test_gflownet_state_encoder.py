from __future__ import annotations

import torch

from src.models.components.gflownet_env import GraphEnv
from src.models.components.state_encoder import StateEncoder


def test_current_node_encoder_changes_with_state() -> None:
    env = GraphEnv(max_steps=2)
    device = torch.device("cpu")

    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
    edge_batch = torch.zeros(2, dtype=torch.long, device=device)
    edge_relations = torch.zeros(2, dtype=torch.long, device=device)
    node_ptr = torch.tensor([0, 3], dtype=torch.long, device=device)
    edge_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)
    start_node_locals = torch.tensor([0], dtype=torch.long, device=device)
    start_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
    answer_node_locals = torch.empty(0, dtype=torch.long, device=device)
    answer_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)

    graph_dict = {
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "edge_relations": edge_relations,
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
    question_tokens = torch.randn(1, hidden_dim, device=device)

    encoder = StateEncoder(hidden_dim=hidden_dim, max_steps=2)
    cache = encoder.precompute(
        node_ptr=node_ptr,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
    )

    s0 = encoder.encode_state(cache=cache, state=state)
    action_embeddings = torch.zeros(1, hidden_dim, dtype=torch.float32, device=device)
    state = env.step(state, torch.tensor([0], dtype=torch.long, device=device), action_embeddings, step_index=0)
    s1 = encoder.encode_state(cache=cache, state=state)

    assert s0.shape == (1, hidden_dim)
    assert s1.shape == (1, hidden_dim)
    assert not torch.allclose(s0, s1)
