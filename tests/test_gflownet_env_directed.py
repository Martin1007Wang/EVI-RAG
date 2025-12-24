from __future__ import annotations

import torch

from src.models.components.gflownet_env import GraphEnv


def test_gflownet_env_directed_action_mask_and_backtrack() -> None:
    env = GraphEnv(max_steps=2)
    device = torch.device("cpu")

    # One graph with 2 directed edges: 0->1 and 1->0.
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_batch = torch.zeros(2, dtype=torch.long, device=device)
    node_global_ids = torch.tensor([10, 11], dtype=torch.long, device=device)
    node_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)
    edge_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)

    start_node_locals = torch.tensor([0], dtype=torch.long, device=device)
    start_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
    start_entity_ids = torch.tensor([10], dtype=torch.long, device=device)
    start_entity_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)

    answer_node_locals = torch.empty(0, dtype=torch.long, device=device)
    answer_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)
    answer_entity_ids = torch.empty(0, dtype=torch.long, device=device)

    graph_dict = {
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "node_global_ids": node_global_ids,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "edge_scores": torch.zeros(2, dtype=torch.float32, device=device),
        "start_node_locals": start_node_locals,
        "start_ptr": start_ptr,
        "start_entity_ids": start_entity_ids,
        "start_entity_ptr": start_entity_ptr,
        "answer_node_locals": answer_node_locals,
        "answer_ptr": answer_ptr,
        "answer_entity_ids": answer_entity_ids,
        "edge_relations": torch.zeros(2, dtype=torch.long, device=device),
        "edge_labels": torch.zeros(2, dtype=torch.float32, device=device),
        "is_answer_reachable": torch.tensor([False], dtype=torch.bool, device=device),
    }

    state = env.reset(graph_dict, device=device)

    # Step0: directed graph allows only outgoing edges from active nodes.
    forward0, backward0 = env.candidate_edge_masks(state)
    assert forward0.detach().cpu().tolist() == [True, False]
    assert backward0.detach().cpu().tolist() == [False, True]
    mask0 = (forward0 | backward0).detach().cpu().tolist()
    assert mask0 == [True, True]

    # Take edge 0->1.
    action_emb = torch.zeros((1, 1), dtype=torch.float32, device=device)
    state = env.step(state, torch.tensor([0], dtype=torch.long, device=device), action_emb, step_index=0)
    assert int(state.step_counts.item()) == 1
    assert state.active_nodes.detach().cpu().tolist() == [False, True]

    # Step1: outgoing from 1 is edge 1->0.
    forward1, backward1 = env.candidate_edge_masks(state)
    assert forward1.detach().cpu().tolist() == [False, True]
    assert backward1.detach().cpu().tolist() == [True, False]
    mask1 = (forward1 | backward1).detach().cpu().tolist()
    assert mask1 == [True, True]


def test_gflownet_env_start_answer_sets_hit_on_reset() -> None:
    env = GraphEnv(max_steps=1)
    device = torch.device("cpu")

    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    edge_batch = torch.zeros(1, dtype=torch.long, device=device)
    node_global_ids = torch.tensor([10, 11], dtype=torch.long, device=device)
    node_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)
    edge_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)

    start_node_locals = torch.tensor([0], dtype=torch.long, device=device)
    start_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
    start_entity_ids = torch.tensor([10], dtype=torch.long, device=device)
    start_entity_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)

    answer_node_locals = torch.tensor([0], dtype=torch.long, device=device)
    answer_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)
    answer_entity_ids = torch.tensor([10], dtype=torch.long, device=device)

    graph_dict = {
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "node_global_ids": node_global_ids,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "edge_scores": torch.zeros(1, dtype=torch.float32, device=device),
        "start_node_locals": start_node_locals,
        "start_ptr": start_ptr,
        "start_entity_ids": start_entity_ids,
        "start_entity_ptr": start_entity_ptr,
        "answer_node_locals": answer_node_locals,
        "answer_ptr": answer_ptr,
        "answer_entity_ids": answer_entity_ids,
        "edge_relations": torch.zeros(1, dtype=torch.long, device=device),
        "edge_labels": torch.zeros(1, dtype=torch.float32, device=device),
        "is_answer_reachable": torch.tensor([True], dtype=torch.bool, device=device),
    }

    state = env.reset(graph_dict, device=device)
    assert bool(state.answer_hits.item()) is True


def test_gflownet_env_empty_start_marks_done() -> None:
    env = GraphEnv(max_steps=1)
    device = torch.device("cpu")

    edge_index = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    edge_batch = torch.zeros(1, dtype=torch.long, device=device)
    node_global_ids = torch.tensor([10, 11], dtype=torch.long, device=device)
    node_ptr = torch.tensor([0, 2], dtype=torch.long, device=device)
    edge_ptr = torch.tensor([0, 1], dtype=torch.long, device=device)

    start_node_locals = torch.empty(0, dtype=torch.long, device=device)
    start_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)
    start_entity_ids = torch.empty(0, dtype=torch.long, device=device)
    start_entity_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)

    answer_node_locals = torch.empty(0, dtype=torch.long, device=device)
    answer_ptr = torch.tensor([0, 0], dtype=torch.long, device=device)
    answer_entity_ids = torch.empty(0, dtype=torch.long, device=device)

    graph_dict = {
        "edge_index": edge_index,
        "edge_batch": edge_batch,
        "node_global_ids": node_global_ids,
        "node_ptr": node_ptr,
        "edge_ptr": edge_ptr,
        "edge_scores": torch.zeros(1, dtype=torch.float32, device=device),
        "start_node_locals": start_node_locals,
        "start_ptr": start_ptr,
        "start_entity_ids": start_entity_ids,
        "start_entity_ptr": start_entity_ptr,
        "answer_node_locals": answer_node_locals,
        "answer_ptr": answer_ptr,
        "answer_entity_ids": answer_entity_ids,
        "edge_relations": torch.zeros(1, dtype=torch.long, device=device),
        "edge_labels": torch.zeros(1, dtype=torch.float32, device=device),
        "is_answer_reachable": torch.tensor([False], dtype=torch.bool, device=device),
    }

    state = env.reset(graph_dict, device=device)
    assert bool(state.done.item()) is True
