from __future__ import annotations

import torch

from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
    StartLogitHead,
)
from src.models.configs import (
    BackboneConfig,
    GraphLogZHeadConfig,
    PolicyConfig,
    StartHeadConfig,
    StateScoreHeadConfig,
)
from src.graph_runtime import TrajectoryBatch
from src.models.policy.trajectory_policy import TrajectoryPolicy


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


def make_policy_config() -> PolicyConfig:
    return PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            gnn_layers=1,
            gnn_dropout=0.0,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        state_score_head=StateScoreHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        start_head=StartHeadConfig(hidden_dim=16, dropout=0.0),
        graph_log_z_head=GraphLogZHeadConfig(hidden_dim=16, dropout=0.0),
    )


def make_policy(*, max_steps: int = 2) -> TrajectoryPolicy:
    cfg = make_policy_config()
    graph_hidden_dim = int(cfg.backbone.hidden_dim)
    backbone = EmbeddingBackbone(cfg.backbone)
    state_score_head = NodeFlowHead(
        node_dim=graph_hidden_dim,
        question_dim=graph_hidden_dim,
        hidden_dim=int(cfg.state_score_head.hidden_dim),
        num_layers=int(cfg.state_score_head.num_layers),
        dropout=float(cfg.state_score_head.dropout),
    )
    start_head = StartLogitHead(
        policy_dim=graph_hidden_dim,
        hidden_dim=int(cfg.start_head.hidden_dim),
        dropout=float(cfg.start_head.dropout),
    )
    return TrajectoryPolicy(
        cfg,
        max_steps=max_steps,
        backbone=backbone,
        state_score_head=state_score_head,
        start_head=start_head,
    )
