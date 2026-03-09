from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.backbone import EmbeddingBackbone
from src.models.environment import CsrAdjacency, GraphEnvContext
from src.models.policy.encoder import PolicyEncoder
from src.models.policy.modules import NodeFlowHead, QuestionContextModule

from .batch import TrajectoryBatch
from .heads import GraphLogZHead


def _build_csr_with_edge_ids(
    *,
    edge_index: torch.Tensor,
    num_nodes_total: int,
) -> CsrAdjacency:
    device = edge_index.device
    edge_ids = torch.arange(int(edge_index.size(1)), device=device, dtype=torch.long)
    if int(edge_ids.numel()) == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        crow = torch.zeros((num_nodes_total + 1,), device=device, dtype=torch.long)
        return CsrAdjacency(
            crow=crow,
            col=empty,
            edge_ids=empty,
            size=(num_nodes_total, num_nodes_total),
        )
    heads = edge_index[0]
    order = torch.argsort(heads)
    heads_sorted = heads.index_select(0, order)
    tails_sorted = edge_index[1].index_select(0, order)
    edge_ids_sorted = edge_ids.index_select(0, order)
    row_ids = torch.arange(num_nodes_total + 1, device=device, dtype=torch.long)
    crow = torch.searchsorted(heads_sorted, row_ids, right=False)
    return CsrAdjacency(
        crow=crow,
        col=tails_sorted,
        edge_ids=edge_ids_sorted,
        size=(num_nodes_total, num_nodes_total),
    )


def _build_relation_table(
    *,
    edge_rel_global: torch.Tensor,
    edge_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(edge_rel_global.numel()) == 0:
        return edge_embeddings.new_empty(
            (0, int(edge_embeddings.size(-1)))
        ), edge_rel_global.new_empty((0,))
    _, edge_relations = torch.unique(edge_rel_global, sorted=True, return_inverse=True)
    num_rel = int(edge_relations.max().item()) + 1
    first_occ = torch.full(
        (num_rel,),
        fill_value=int(edge_relations.numel()),
        device=edge_relations.device,
        dtype=torch.long,
    )
    edge_ids = torch.arange(
        int(edge_relations.numel()), device=edge_relations.device, dtype=torch.long
    )
    first_occ.scatter_reduce_(
        0, edge_relations, edge_ids, reduce="amin", include_self=True
    )
    relation_embeddings = edge_embeddings.index_select(0, first_occ)
    return relation_embeddings, edge_relations


def _segment_mean_2d(
    *,
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError(f"values must be 2D, got {tuple(values.shape)}.")
    out = values.new_zeros((num_segments, int(values.size(1))))
    counts = values.new_zeros((num_segments, 1))
    if int(values.numel()) == 0:
        return out
    expanded_ids = segment_ids.unsqueeze(1).expand(-1, int(values.size(1)))
    out.scatter_add_(0, expanded_ids, values)
    counts.scatter_add_(
        0,
        segment_ids.unsqueeze(1),
        values.new_ones((int(values.size(0)), 1)),
    )
    return out / counts.clamp(min=1.0)


def build_env_context(batch: TrajectoryBatch) -> tuple[GraphEnvContext, torch.Tensor]:
    relation_embeddings, edge_relations = _build_relation_table(
        edge_rel_global=batch.edge_rel_global,
        edge_embeddings=batch.edge_embeddings,
    )
    num_nodes_total = batch.num_nodes_total
    adj_t_fwd = _build_csr_with_edge_ids(
        edge_index=batch.edge_index,
        num_nodes_total=num_nodes_total,
    )
    adj_t_bwd = _build_csr_with_edge_ids(
        edge_index=batch.edge_index[[1, 0]],
        num_nodes_total=num_nodes_total,
    )
    context = GraphEnvContext(
        num_graphs=batch.num_graphs,
        num_nodes_total=num_nodes_total,
        node_ptr=batch.node_ptr,
        edge_index=batch.edge_index,
        edge_relations=edge_relations,
        edge_rel_global=batch.edge_rel_global,
        edge_batch=batch.edge_batch,
        node_batch=batch.node_batch,
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=batch.node_embeddings,
        node_tokens=batch.node_embeddings,
        relation_tokens=relation_embeddings,
        question_emb=batch.question_emb,
        q_local_indices=batch.q_local_indices,
        a_local_indices=batch.a_local_indices,
        q_ptr=batch.q_ptr,
        a_ptr=batch.a_ptr,
        answer_entity_ids=batch.answer_entity_ids,
        answer_ptr=batch.answer_ptr,
        node_global_ids=batch.node_global_ids,
        dummy_mask=batch.dummy_mask,
        sample_ids=batch.sample_ids,
        question_ctx=batch.question_ctx,
        question_ctx_mask=batch.question_ctx_mask,
        heuristic_log_v=batch.heuristic_log_v,
    )
    return context, edge_relations


@dataclass(frozen=True)
class TrajectoryPolicyContext:
    batch: TrajectoryBatch
    env_context: GraphEnvContext
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    graph_log_z: torch.Tensor
    action_cache: dict[str, torch.Tensor]


class TrajectoryEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        backbone: EmbeddingBackbone,
        question_modules: QuestionContextModule,
        node_flow_head: NodeFlowHead,
        graph_log_z_head: GraphLogZHead,
        doob_h_node_temperature: float,
    ) -> None:
        super().__init__()
        self.policy_encoder = PolicyEncoder(
            backbone=backbone,
            question_modules=question_modules,
            node_flow_head=node_flow_head,
            doob_h_node_temperature=doob_h_node_temperature,
        )
        self.graph_log_z_head = graph_log_z_head

    def encode(self, batch: TrajectoryBatch) -> TrajectoryPolicyContext:
        env_context, _ = build_env_context(batch)
        node_tokens, relation_tokens, question_tokens = (
            self.policy_encoder.encode_context(env_context)
        )
        action_cache = self.policy_encoder.build_action_cache(
            env_context=env_context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        q_counts = (batch.q_ptr[1:] - batch.q_ptr[:-1]).clamp(min=0)
        if bool((q_counts <= 0).any().item()):
            raise ValueError(
                "q_local_indices contains empty graphs; start summary is undefined."
            )
        q_offsets = batch.node_ptr[:-1].repeat_interleave(q_counts)
        q_abs = batch.q_local_indices + q_offsets
        q_graph_ids = torch.arange(
            batch.num_graphs, device=batch.node_ptr.device, dtype=torch.long
        ).repeat_interleave(q_counts)
        q_node_tokens = node_tokens.index_select(0, q_abs)
        start_summary = _segment_mean_2d(
            values=q_node_tokens,
            segment_ids=q_graph_ids,
            num_segments=batch.num_graphs,
        )
        graph_log_z = self.graph_log_z_head(
            question_features=question_tokens,
            start_summary=start_summary,
        )
        return TrajectoryPolicyContext(
            batch=batch,
            env_context=env_context,
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            question_tokens=question_tokens,
            graph_log_z=graph_log_z.to(dtype=torch.float32),
            action_cache=action_cache,
        )
