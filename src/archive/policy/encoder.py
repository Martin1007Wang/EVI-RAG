from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.models.components import EmbeddingBackbone, NodeFlowHead
from src.models.components.embedding import BackboneInput
from src.graph_runtime import GraphObservation, GraphTopology

from .modules import QuestionContextModule


@dataclass(frozen=True)
class PreparedPolicyContext:
    topology: GraphTopology
    observation: GraphObservation
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor


class PolicyEncoder(nn.Module):
    def __init__(
        self,
        *,
        backbone: EmbeddingBackbone,
        question_modules: QuestionContextModule,
        node_flow_head: NodeFlowHead,
        committor_head: NodeFlowHead,
        doob_h_node_temperature: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.question_modules = question_modules
        self.node_flow_head = node_flow_head
        self.committor_head = committor_head
        self.doob_h_node_temperature = float(doob_h_node_temperature)

    def encode_context(
        self,
        *,
        topology: GraphTopology,
        observation: GraphObservation,
    ) -> PreparedPolicyContext:
        encoded = self.backbone.encode(
            BackboneInput(
                node_features=observation.node_features,
                relation_features=observation.relation_features,
                question_embedding=observation.question_embedding,
                edge_index=topology.edge_index,
                edge_relations=topology.edge_type,
                num_nodes=topology.num_nodes,
            )
        )
        return PreparedPolicyContext(
            topology=topology,
            observation=observation,
            node_tokens=encoded.node_tokens,
            relation_tokens=encoded.relation_tokens,
            question_tokens=encoded.question_tokens,
        )

    def compute_node_log_f(
        self,
        *,
        prepared_context: PreparedPolicyContext,
    ) -> torch.Tensor:
        node_tokens = prepared_context.node_tokens
        question_tokens = prepared_context.question_tokens
        node_graph_ids = prepared_context.topology.all_node_graph_index(
            device=node_tokens.device
        )
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
        prepared_context: PreparedPolicyContext,
    ) -> dict[str, object]:
        observation = prepared_context.observation
        topology = prepared_context.topology
        question_tokens = prepared_context.question_tokens
        question_context_tokens = self.question_modules.build_question_context_tokens(
            observation=observation,
            question_tokens=question_tokens,
        )
        question_padding_mask = self.question_modules.build_question_padding_mask(
            observation=observation,
            question_context_tokens=question_context_tokens,
        )
        lexical_question_tokens = self.question_modules.build_question_lexical_tokens(
            question_context_tokens=question_context_tokens
        )
        node_log_f = self.compute_node_log_f(prepared_context=prepared_context)
        cache = {
            "question_context_tokens": question_context_tokens,
            "question_padding_mask": question_padding_mask,
            "lexical_question_tokens": lexical_question_tokens,
            "node_log_f": node_log_f,
        }
        if topology.has_super_source_layout(
            node_ids=observation.node_ids,
            device=question_tokens.device,
        ):
            cache["super_node_mask"] = (
                observation.node_ids.to(device=question_tokens.device, dtype=torch.long)
                < 0
            )
            question_super_abs, answer_super_abs = topology.infer_super_source_indices(
                node_ids=observation.node_ids,
                device=question_tokens.device,
            )
            edge_index = topology.edge_index
            if edge_index.numel() == 0:
                edge_disallowed_forward = edge_index.new_empty((0,), dtype=torch.bool)
                edge_disallowed_backward = edge_index.new_empty((0,), dtype=torch.bool)
            else:
                edge_ids = torch.arange(
                    int(edge_index.size(1)),
                    device=question_tokens.device,
                    dtype=torch.long,
                )
                edge_graph_index = topology.graph_index_from_edges(edge_ids)
                forward_disallowed = answer_super_abs.index_select(0, edge_graph_index)
                backward_disallowed = question_super_abs.index_select(
                    0, edge_graph_index
                )
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
        observation: GraphObservation,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        num_agents: int,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
        lexical_question_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.question_modules.compute_agent_potentials(
            observation=observation,
            question_tokens=question_tokens,
            agent_history=agent_history,
            num_agents=num_agents,
            question_context_tokens=question_context_tokens,
            question_padding_mask=question_padding_mask,
            lexical_question_tokens=lexical_question_tokens,
        )

    def compute_agent_potentials_from_graph_ids(
        self,
        *,
        observation: GraphObservation,
        question_tokens: torch.Tensor,
        agent_history: torch.Tensor,
        agent_graph_ids: torch.Tensor,
        question_context_tokens: torch.Tensor | None = None,
        question_padding_mask: torch.Tensor | None = None,
        lexical_question_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.question_modules.compute_agent_potentials_from_graph_ids(
            observation=observation,
            question_tokens=question_tokens,
            agent_history=agent_history,
            agent_graph_ids=agent_graph_ids,
            question_context_tokens=question_context_tokens,
            question_padding_mask=question_padding_mask,
            lexical_question_tokens=lexical_question_tokens,
        )


__all__ = ["PolicyEncoder", "PreparedPolicyContext"]
