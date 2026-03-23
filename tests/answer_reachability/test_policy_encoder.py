from __future__ import annotations

from typing import cast

import torch

from src.models.components import EmbeddingBackbone, NodeFlowHead
from src.models.configs import BackboneConfig
from src.graph import build_graph_batch
from src.archive.policy.encoder import PolicyEncoder, PreparedPolicyContext
from src.archive.policy.modules import QuestionContextModule

from .conftest import make_batch_from_graph, make_toy_batch


def _make_policy_encoder() -> PolicyEncoder:
    backbone = EmbeddingBackbone(
        BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=0,
            gnn_dropout=0.0,
            use_adapter=False,
            adapter_dim=4,
            adapter_dropout=0.0,
        )
    )
    question_modules = QuestionContextModule(
        policy_dim=8,
        graph_hidden_dim=8,
        embedding_dim=8,
        dropout=0.0,
        lexical_rank=4,
    )
    return PolicyEncoder(
        backbone=backbone,
        question_modules=question_modules,
        node_flow_head=NodeFlowHead(
            node_dim=8,
            question_dim=8,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        ),
        committor_head=NodeFlowHead(
            node_dim=8,
            question_dim=8,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        ),
        doob_h_node_temperature=1.0,
    )


def test_policy_encoder_encodes_split_topology_and_observation() -> None:
    topology, observation = build_graph_batch(make_toy_batch())
    encoder = _make_policy_encoder()

    prepared = encoder.encode_context(topology=topology, observation=observation)

    assert isinstance(prepared, PreparedPolicyContext)
    assert prepared.topology is topology
    assert prepared.observation is observation
    assert tuple(prepared.node_tokens.shape) == (topology.num_nodes, 8)
    assert tuple(prepared.relation_tokens.shape) == (2, 8)
    assert tuple(prepared.question_tokens.shape) == (topology.num_graphs, 8)


def test_policy_encoder_build_action_cache_uses_topology_super_source_metadata() -> (
    None
):
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
    encoder = _make_policy_encoder()
    prepared = encoder.encode_context(topology=topology, observation=observation)

    cache = encoder.build_action_cache(prepared_context=prepared)

    assert torch.equal(
        cache["super_node_mask"],
        torch.tensor([False, False, True, True], dtype=torch.bool),
    )
    assert torch.equal(
        cache["edge_disallowed_forward"],
        torch.tensor([False, False, True, True], dtype=torch.bool),
    )
    assert torch.equal(
        cache["edge_disallowed_backward"],
        torch.tensor([True, True, False, False], dtype=torch.bool),
    )
    node_log_f = cast(torch.Tensor, cache["node_log_f"])
    assert tuple(node_log_f.shape) == (topology.num_nodes,)
