from __future__ import annotations

import torch

from src.graph import build_graph_batch
from src.models.gflownet import PreparedSearchBatch

from .conftest import make_batch_from_graph, make_policy, make_toy_batch


def test_base_search_policy_prepare_batch_encodes_search_context() -> None:
    batch = make_toy_batch()
    topology, observation = build_graph_batch(batch)
    policy = make_policy()

    prepared = policy.prepare_batch(batch)

    assert isinstance(prepared, PreparedSearchBatch)
    assert torch.equal(prepared.topology.edge_index, topology.edge_index)
    assert torch.equal(prepared.topology.edge_type, topology.edge_type)
    assert torch.equal(
        prepared.observation.node_entity_ids,
        observation.node_entity_ids,
    )
    assert torch.equal(
        prepared.observation.q_local_indices.local_indices,
        observation.q_local_indices.local_indices,
    )
    assert prepared.observation.sample_ids == observation.sample_ids
    assert tuple(prepared.node_tokens.shape) == (topology.num_nodes, 8)
    assert tuple(prepared.relation_tokens.shape) == (
        int(observation.relation_features.size(0)),
        8,
    )
    assert tuple(prepared.question_tokens.shape) == (topology.num_graphs, 8)
    assert tuple(prepared.question_context_tokens.shape) == (
        topology.num_graphs,
        int(observation.question_context.size(1)),
        8,
    )
    assert torch.equal(prepared.question_context_mask, observation.question_valid_mask)


def test_graph_topology_infers_super_source_masks_from_virtual_nodes() -> None:
    batch = make_batch_from_graph(
        num_nodes=4,
        edge_index=torch.tensor([[2, 2, 0, 1], [0, 1, 3, 3]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 1, 0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, -1, -2], dtype=torch.long),
        sample_id="super-source",
    )
    topology, observation = build_graph_batch(batch)

    question_super_abs, answer_super_abs = topology.infer_super_source_indices(
        node_entity_ids=observation.node_entity_ids,
    )
    super_node_mask = torch.zeros((topology.num_nodes,), dtype=torch.bool)
    super_node_mask[question_super_abs] = True
    super_node_mask[answer_super_abs] = True
    edge_sources = topology.edge_index[0]
    edge_targets = topology.edge_index[1]
    edge_disallowed_forward = super_node_mask.index_select(0, edge_targets)
    edge_disallowed_backward = super_node_mask.index_select(0, edge_sources)

    assert topology.has_super_source_layout(node_entity_ids=observation.node_entity_ids)
    assert torch.equal(question_super_abs, torch.tensor([2], dtype=torch.long))
    assert torch.equal(answer_super_abs, torch.tensor([3], dtype=torch.long))
    assert torch.equal(
        super_node_mask,
        torch.tensor([False, False, True, True], dtype=torch.bool),
    )
    assert torch.equal(
        edge_disallowed_forward,
        torch.tensor([False, False, True, True], dtype=torch.bool),
    )
    assert torch.equal(
        edge_disallowed_backward,
        torch.tensor([True, True, False, False], dtype=torch.bool),
    )
