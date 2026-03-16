from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.graph_runtime import GroupedLocalNodeIndex, build_graph_batch

from .conftest import make_toy_batch


def _make_two_graph_protocol_batch() -> SimpleNamespace:
    emb_dim = 8
    torch.manual_seed(11)
    return SimpleNamespace(
        num_graphs=2,
        node_ptr=torch.tensor([0, 3, 5], dtype=torch.long),
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long),
        edge_rel_global=torch.tensor([5, 7, 5], dtype=torch.long),
        node_embeddings=torch.randn(5, emb_dim),
        edge_embeddings=torch.randn(3, emb_dim),
        question_emb=torch.randn(2, emb_dim),
        question_ctx=torch.randn(2, 2, emb_dim),
        question_ctx_mask=torch.tensor([[True, True], [True, False]], dtype=torch.bool),
        q_local_indices=torch.tensor([0, 2, 1], dtype=torch.long),
        q_ptr=torch.tensor([0, 2, 3], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102, 200, 201], dtype=torch.long),
        sample_ids=["graph-0", "graph-1"],
    )


def test_build_graph_batch_compacts_relations_and_builds_csr() -> None:
    batch = make_toy_batch()

    topology, observation = build_graph_batch(batch)

    expected_relation_tokens = batch.edge_embeddings.index_select(
        0, torch.tensor([0, 1], dtype=torch.long)
    )
    assert torch.equal(topology.edge_type, torch.tensor([0, 1, 0], dtype=torch.long))
    assert torch.equal(observation.relation_features, expected_relation_tokens)
    assert torch.equal(
        topology.adjacency.crow,
        torch.tensor([0, 2, 3, 3], dtype=torch.long),
    )
    assert torch.equal(
        topology.adjacency.col,
        torch.tensor([1, 2, 2], dtype=torch.long),
    )
    assert torch.equal(
        topology.adjacency.edge_ids,
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    start_node_index, start_graph_index = topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )
    assert torch.equal(start_node_index, torch.tensor([0], dtype=torch.long))
    assert torch.equal(start_graph_index, torch.tensor([0], dtype=torch.long))


def test_topology_derives_graph_ids_without_exposing_batch_vectors() -> None:
    topology, observation = build_graph_batch(_make_two_graph_protocol_batch())

    start_node_index, start_graph_index = topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )

    assert torch.equal(start_node_index, torch.tensor([0, 2, 4], dtype=torch.long))
    assert torch.equal(start_graph_index, torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(
        topology.graph_index_from_nodes(torch.tensor([0, 2, 3, 4], dtype=torch.long)),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    assert torch.equal(
        topology.graph_index_from_edges(torch.tensor([0, 1, 2], dtype=torch.long)),
        torch.tensor([0, 0, 1], dtype=torch.long),
    )


def test_build_graph_batch_rejects_out_of_range_q_local_indices() -> None:
    batch = make_toy_batch()
    batch.q_local_indices[0] = 3

    with pytest.raises(ValueError, match="out-of-range local node indices"):
        build_graph_batch(batch)


def test_build_graph_batch_rejects_invalid_question_context_mask_shape() -> None:
    batch = _make_two_graph_protocol_batch()
    batch.question_ctx_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

    with pytest.raises(ValueError, match="question_ctx_mask shape mismatch"):
        build_graph_batch(batch)


def test_build_graph_batch_rejects_cross_graph_edges() -> None:
    batch = _make_two_graph_protocol_batch()
    batch.edge_index = torch.tensor([[2], [3]], dtype=torch.long)
    batch.edge_rel_global = torch.tensor([5], dtype=torch.long)
    batch.edge_embeddings = torch.randn(1, 8)

    with pytest.raises(ValueError, match="crosses graph boundaries"):
        build_graph_batch(batch)


def test_topology_build_node_membership_mask_uses_batch_offsets() -> None:
    topology, observation = build_graph_batch(_make_two_graph_protocol_batch())

    mask = topology.build_node_membership_mask(
        observation.q_local_indices,
        field_name="q_local_indices",
    )

    assert torch.equal(mask, torch.tensor([True, False, True, False, True]))


def test_topology_build_node_membership_mask_debug_checks_are_opt_in() -> None:
    topology, _ = build_graph_batch(_make_two_graph_protocol_batch())
    local_node_index = GroupedLocalNodeIndex(
        local_indices=torch.tensor([0, 1], dtype=torch.long),
        _group_ptr=torch.tensor([0, 2, 1], dtype=torch.long),
    )

    mask = topology.build_node_membership_mask(
        local_node_index,
        field_name="q_local_indices",
    )

    assert int(mask.sum().item()) == 2
    with pytest.raises(ValueError, match="non-decreasing"):
        topology.build_node_membership_mask(
            local_node_index,
            field_name="q_local_indices",
            debug_checks=True,
        )
