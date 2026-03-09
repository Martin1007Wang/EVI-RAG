from __future__ import annotations

import torch

from src.models.configs.policy import (
    BackboneConfig,
    FlowHeadConfig,
    PolicyConfig,
    PriorityHeadConfig,
)
from src.models.trajectory_gfn.batch import TrajectoryBatch
from src.models.trajectory_gfn.policy import TrajectoryPolicy


def make_batch_from_graph(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_rel_global: torch.Tensor,
    q_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    node_global_ids: torch.Tensor | None = None,
    sample_id: str = "toy-sample",
) -> TrajectoryBatch:
    if node_global_ids is None:
        node_global_ids = torch.arange(100, 100 + num_nodes, dtype=torch.long)
    emb_dim = 8
    return TrajectoryBatch(
        num_graphs=1,
        node_ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        edge_index=edge_index,
        edge_rel_global=edge_rel_global,
        edge_batch=torch.zeros((edge_index.size(1),), dtype=torch.long),
        node_batch=torch.zeros((num_nodes,), dtype=torch.long),
        node_embeddings=torch.randn(num_nodes, emb_dim),
        edge_embeddings=torch.randn(edge_index.size(1), emb_dim),
        question_emb=torch.randn(1, emb_dim),
        question_ctx=torch.randn(1, 2, emb_dim),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        q_local_indices=q_local_indices,
        q_ptr=torch.tensor([0, int(q_local_indices.numel())], dtype=torch.long),
        a_local_indices=a_local_indices,
        a_ptr=torch.tensor([0, int(a_local_indices.numel())], dtype=torch.long),
        answer_entity_ids=answer_entity_ids,
        answer_ptr=torch.tensor([0, int(answer_entity_ids.numel())], dtype=torch.long),
        node_global_ids=node_global_ids,
        sample_ids=[sample_id],
        questions=["toy question"],
        dataset_scope="sub",
    )


def make_toy_batch() -> TrajectoryBatch:
    torch.manual_seed(7)
    return make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1, 0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        answer_entity_ids=torch.tensor([102], dtype=torch.long),
        node_global_ids=torch.tensor([100, 101, 102], dtype=torch.long),
    )


def make_dead_end_batch() -> TrajectoryBatch:
    torch.manual_seed(13)
    return make_batch_from_graph(
        num_nodes=1,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_rel_global=torch.empty((0,), dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_global_ids=torch.tensor([100], dtype=torch.long),
        sample_id="dead-end",
    )


def make_policy() -> TrajectoryPolicy:
    cfg = PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=1,
            gnn_dropout=0.0,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        flow_head=FlowHeadConfig(hidden_dim=16, dropout=0.0, relation_low_rank=2),
        priority_head=PriorityHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        stop_bias_init=-0.5,
        stop_delta_scale=2.0,
        stop_delta_temperature=1.0,
        doob_h_alpha=0.0,
        doob_h_node_temperature=1.0,
    )
    return TrajectoryPolicy(cfg, max_steps=2, min_stop_steps=1)
