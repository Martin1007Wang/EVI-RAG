from __future__ import annotations

import torch
from torch import nn

from src.models.backbone import EmbeddingBackbone
from src.models.environment import (
    GraphEnvContext,
    has_super_source_layout,
    infer_super_source_absolute_indices,
)
from .modules import NodeFlowHead, QuestionContextModule


class PolicyEncoder(nn.Module):
    def __init__(
        self,
        *,
        backbone: EmbeddingBackbone,
        question_modules: QuestionContextModule,
        node_flow_head: NodeFlowHead,
        doob_h_node_temperature: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.question_modules = question_modules
        self.node_flow_head = node_flow_head
        self.doob_h_node_temperature = float(doob_h_node_temperature)

    def encode_context(
        self, context: GraphEnvContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_tokens = self.backbone.project_node_embeddings(context.node_embeddings)
        relation_tokens = self.backbone.project_relation_embeddings(
            context.relation_tokens
        )
        question_tokens = self.backbone.project_question_embeddings(
            context.question_emb
        )

        node_tokens = self.backbone.encode_graph(
            node_tokens=node_tokens,
            relation_tokens=relation_tokens,
            edge_index=context.edge_index,
            edge_relations=context.edge_relations,
            num_nodes=context.num_nodes_total,
            question_tokens=question_tokens,
            node_batch=context.node_batch,
        )
        return node_tokens, relation_tokens, question_tokens

    def compute_node_log_f(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        node_graph_ids = env_context.node_batch.to(
            device=node_tokens.device, dtype=torch.long
        ).clamp(min=0)
        node_questions = question_tokens.index_select(0, node_graph_ids)
        node_scores = self.node_flow_head(node_tokens, node_questions)
        node_log_f = node_scores.to(dtype=torch.float32) / self.doob_h_node_temperature
        node_log_f = torch.where(
            torch.isfinite(node_log_f), node_log_f, torch.zeros_like(node_log_f)
        )
        return node_log_f.to(dtype=node_tokens.dtype)

    @staticmethod
    def gather_state_log_flows(
        *,
        node_log_f: torch.Tensor,
        current_nodes: torch.Tensor,
        num_nodes_total: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        safe_nodes = current_nodes.clamp(min=0, max=max(num_nodes_total - 1, 0))
        gathered = node_log_f.index_select(0, safe_nodes).to(dtype=dtype)
        valid = current_nodes >= 0
        return torch.where(valid, gathered, torch.zeros_like(gathered))

    def build_action_cache(
        self,
        *,
        env_context: GraphEnvContext,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        question_context_tokens = self.question_modules.build_question_context_tokens(
            env_context=env_context,
            question_tokens=question_tokens,
        )
        question_padding_mask = self.question_modules.build_question_padding_mask(
            env_context=env_context,
            question_context_tokens=question_context_tokens,
        )
        lexical_question_tokens = self.question_modules.build_question_lexical_tokens(
            question_context_tokens=question_context_tokens
        )
        node_log_f = self.compute_node_log_f(
            env_context=env_context,
            node_tokens=node_tokens,
            question_tokens=question_tokens,
        )
        cache = {
            "question_context_tokens": question_context_tokens,
            "question_padding_mask": question_padding_mask,
            "lexical_question_tokens": lexical_question_tokens,
            "node_log_f": node_log_f,
        }
        if has_super_source_layout(
            node_ptr=env_context.node_ptr,
            node_global_ids=env_context.node_global_ids,
            num_nodes_total=env_context.num_nodes_total,
            device=question_tokens.device,
        ):
            cache["super_node_mask"] = (
                env_context.node_global_ids.to(
                    device=question_tokens.device, dtype=torch.long
                )
                < 0
            )
            question_super_abs, answer_super_abs = infer_super_source_absolute_indices(
                node_ptr=env_context.node_ptr,
                node_global_ids=env_context.node_global_ids,
                num_nodes_total=env_context.num_nodes_total,
                device=question_tokens.device,
            )
            edge_index = env_context.edge_index
            if edge_index.numel() == 0:
                edge_disallowed_forward = edge_index.new_empty((0,), dtype=torch.bool)
                edge_disallowed_backward = edge_index.new_empty((0,), dtype=torch.bool)
            else:
                edge_batch = env_context.edge_batch
                forward_disallowed = answer_super_abs.index_select(0, edge_batch)
                backward_disallowed = question_super_abs.index_select(0, edge_batch)
                edge_disallowed_forward = (edge_index[0] == forward_disallowed) | (
                    edge_index[1] == forward_disallowed
                )
                edge_disallowed_backward = (edge_index[0] == backward_disallowed) | (
                    edge_index[1] == backward_disallowed
                )
            cache["edge_disallowed_forward"] = edge_disallowed_forward
            cache["edge_disallowed_backward"] = edge_disallowed_backward
        return cache

    def compute_agent_potentials(
        self,
        *,
        env_context: GraphEnvContext,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        num_agents: int,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
        lexical_question_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.question_modules.compute_agent_potentials(
            env_context=env_context,
            question_tokens=question_tokens,
            agent_history=agent_history,
            num_agents=num_agents,
            question_context_tokens=question_context_tokens,
            question_padding_mask=question_padding_mask,
            lexical_question_tokens=lexical_question_tokens,
        )


__all__ = ["PolicyEncoder"]
