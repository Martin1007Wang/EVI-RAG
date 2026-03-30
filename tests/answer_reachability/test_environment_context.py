from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import src.graph.topology as graph_topology_module
import torch

from src.graph import GroupedLocalNodeIndex, TrajectoryBatch, build_graph_batch

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
        anchor_local_indices=torch.tensor([0, 2, 1], dtype=torch.long),
        anchor_ptr=torch.tensor([0, 2, 3], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102, 200, 201], dtype=torch.long),
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
        observation.anchor_local_indices,
        field_name="anchor_local_indices",
    )
    assert torch.equal(start_node_index, torch.tensor([0], dtype=torch.long))
    assert torch.equal(start_graph_index, torch.tensor([0], dtype=torch.long))


def test_topology_derives_graph_ids_without_exposing_batch_vectors() -> None:
    topology, observation = build_graph_batch(_make_two_graph_protocol_batch())

    start_node_index, start_graph_index = topology.resolve_local_node_indices(
        observation.anchor_local_indices,
        field_name="anchor_local_indices",
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


def test_build_graph_batch_uses_relation_table_without_per_edge_embeddings() -> None:
    emb_dim = 8
    torch.manual_seed(17)
    batch = SimpleNamespace(
        num_graphs=1,
        node_ptr=torch.tensor([0, 3], dtype=torch.long),
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([5, 7, 5], dtype=torch.long),
        node_embeddings=torch.randn(3, emb_dim),
        edge_embeddings=None,
        relation_embeddings=torch.randn(3, emb_dim),
        edge_rel_local=torch.tensor([2, 1, 2], dtype=torch.long),
        question_emb=torch.randn(1, emb_dim),
        question_ctx=torch.randn(1, 2, emb_dim),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        anchor_local_indices=torch.tensor([0], dtype=torch.long),
        anchor_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_ids=["graph-0"],
    )

    topology, observation = build_graph_batch(batch)

    assert torch.equal(topology.edge_type, torch.tensor([1, 0, 1], dtype=torch.long))
    assert torch.equal(
        observation.relation_features,
        batch.relation_embeddings.index_select(
            0, torch.tensor([1, 2], dtype=torch.long)
        ),
    )


def test_build_graph_batch_rejects_out_of_range_anchor_local_indices() -> None:
    batch = make_toy_batch()
    batch.anchor_local_indices[0] = 3

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
        observation.anchor_local_indices,
        field_name="anchor_local_indices",
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
        field_name="anchor_local_indices",
    )

    assert int(mask.sum().item()) == 2
    with pytest.raises(ValueError, match="non-decreasing"):
        topology.build_node_membership_mask(
            local_node_index,
            field_name="anchor_local_indices",
            debug_checks=True,
        )


def test_topology_build_node_membership_mask_rejects_negative_local_indices() -> None:
    topology, _ = build_graph_batch(_make_two_graph_protocol_batch())
    local_node_index = GroupedLocalNodeIndex(
        local_indices=torch.tensor([0, -1], dtype=torch.long),
        _group_ptr=torch.tensor([0, 1, 2], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="out of range"):
        topology.build_node_membership_mask(
            local_node_index,
            field_name="anchor_local_indices",
        )


def test_topology_gather_outgoing_edges_only_queries_active_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology, _ = build_graph_batch(_make_two_graph_protocol_batch())
    captured: dict[str, torch.Tensor] = {}
    original = graph_topology_module._gather_actions_from_csr

    def _record_nodes(*, adjacency, nodes):
        captured["nodes"] = nodes.clone()
        return original(adjacency=adjacency, nodes=nodes)

    monkeypatch.setattr(
        graph_topology_module, "_gather_actions_from_csr", _record_nodes
    )

    edge_ids, target_nodes, edge_agent_index, out_degrees = (
        topology.gather_outgoing_edges(
            current_nodes=torch.tensor([0, 3], dtype=torch.long),
            active_mask=torch.tensor([False, True], dtype=torch.bool),
        )
    )

    assert torch.equal(captured["nodes"], torch.tensor([3], dtype=torch.long))
    assert torch.equal(edge_ids, torch.tensor([2], dtype=torch.long))
    assert torch.equal(target_nodes, torch.tensor([4], dtype=torch.long))
    assert torch.equal(edge_agent_index, torch.tensor([1], dtype=torch.long))
    assert torch.equal(out_degrees, torch.tensor([0, 1], dtype=torch.long))


def test_trajectory_batch_validate_rejects_edge_batch_mismatch() -> None:
    batch_one = make_toy_batch()
    batch_two = make_toy_batch()
    batch = TrajectoryBatch.concatenate([batch_one, batch_two], validate=False)
    invalid_batch = replace(
        batch,
        edge_batch=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="edge_batch mismatch"):
        invalid_batch.validate()
