from __future__ import annotations

import torch

from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
)
from src.models.configs import (
    BackboneConfig,
    PolicyConfig,
    StateScoreHeadConfig,
)
from src.graph import TrajectoryBatch
from src.models.gflownet import BaseSearchPolicy


def make_batch_from_graph(
    *,
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_rel_global: torch.Tensor,
    q_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    answer_entity_ids: torch.Tensor,
    node_entity_ids: torch.Tensor | None = None,
    sample_id: str = "toy-sample",
    question_emb: torch.Tensor | None = None,
    question_ctx: torch.Tensor | None = None,
    question_ctx_mask: torch.Tensor | None = None,
) -> TrajectoryBatch:
    if node_entity_ids is None:
        node_entity_ids = torch.arange(100, 100 + num_nodes, dtype=torch.long)
    emb_dim = 8
    if question_emb is None:
        question_emb = torch.randn(1, emb_dim)
    if question_ctx is None:
        question_ctx = torch.randn(1, 2, emb_dim)
    if question_ctx_mask is None:
        question_ctx_mask = torch.tensor([[True, True]], dtype=torch.bool)
    return TrajectoryBatch(
        num_graphs=1,
        node_ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        edge_index=edge_index,
        edge_rel_global=edge_rel_global,
        edge_batch=torch.zeros((edge_index.size(1),), dtype=torch.long),
        node_batch=torch.zeros((num_nodes,), dtype=torch.long),
        node_embeddings=torch.randn(num_nodes, emb_dim),
        edge_embeddings=torch.randn(edge_index.size(1), emb_dim),
        question_emb=question_emb,
        question_ctx=question_ctx,
        question_ctx_mask=question_ctx_mask,
        q_local_indices=q_local_indices,
        q_ptr=torch.tensor([0, int(q_local_indices.numel())], dtype=torch.long),
        a_local_indices=a_local_indices,
        a_ptr=torch.tensor([0, int(a_local_indices.numel())], dtype=torch.long),
        answer_entity_ids=answer_entity_ids,
        answer_ptr=torch.tensor([0, int(answer_entity_ids.numel())], dtype=torch.long),
        node_entity_ids=node_entity_ids,
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
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
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
        node_entity_ids=torch.tensor([100], dtype=torch.long),
        sample_id="dead-end",
    )


def make_policy_config() -> PolicyConfig:
    return PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=8,
            hidden_dim=8,
            use_adapter=True,
            adapter_dim=4,
            adapter_dropout=0.0,
        ),
        state_score_head=StateScoreHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
    )


def make_policy(*, max_steps: int = 2) -> BaseSearchPolicy:
    cfg = make_policy_config()
    graph_hidden_dim = int(cfg.backbone.hidden_dim)
    backbone = EmbeddingBackbone(cfg.backbone)
    state_score_head = NodeFlowHead(
        node_dim=graph_hidden_dim,
        question_dim=graph_hidden_dim,
        hidden_dim=int(cfg.state_score_head.hidden_dim),
        num_layers=int(cfg.state_score_head.num_layers),
        dropout=float(cfg.state_score_head.dropout),
        conditioning=str(cfg.state_score_head.conditioning),
    )
    return BaseSearchPolicy(
        cfg,
        max_steps=max_steps,
        backbone=backbone,
        state_score_head=state_score_head,
        transition_policy_head=None,
        step_log_penalty=0.0,
        non_gold_terminal_log_reward=-3.0,
        answer_stop_log_reward_bonus=0.0,
        answer_quotient_allocate_stop_mass=False,
        answer_quotient_gold_reward_mode="shared",
    )
