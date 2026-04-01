from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.graph import TrajectoryBatch, build_graph_batch
from src.models.components import EmbeddingBackbone, StateFlowHead
from src.models.components.embedding import BackboneInput
from src.utils.precision_utils import align_float_input_dtype

from .prepared_batch import SubgraphPreparedBatch, build_subgraph_prepared_batch
from .state import (
    SubgraphAnalysis,
    SubgraphRolloutBatch,
    SubgraphState,
    _dedup_preserve_order,
)


def _build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(max(int(num_layers) - 1, 0)):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _bit_count(bits: int) -> int:
    return int(int(bits).bit_count())


def _normalize_state_indices(
    *,
    state_indices: list[int] | tuple[int, ...] | torch.Tensor | None,
    num_states: int,
) -> list[int]:
    if state_indices is None:
        return list(range(int(num_states)))
    if torch.is_tensor(state_indices):
        values = state_indices.detach().cpu().view(-1).tolist()
    else:
        values = list(state_indices)
    normalized = [int(state_idx) for state_idx in values]
    for state_idx in normalized:
        if state_idx < 0 or state_idx >= int(num_states):
            raise IndexError(f"state_idx out of range: {state_idx}.")
    return normalized


def _question_attention_pool(
    *,
    token_table: torch.Tensor,
    token_ids: tuple[int, ...],
    query_token: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not token_ids:
        return token_table.new_zeros(
            (int(token_table.size(-1)),), dtype=token_table.dtype
        )
    indices = torch.tensor(token_ids, device=device, dtype=torch.long)
    tokens = token_table.index_select(0, indices)
    scores = torch.matmul(
        tokens.to(dtype=torch.float32), query_token.to(dtype=torch.float32)
    )
    weights = torch.softmax(scores, dim=0).to(dtype=tokens.dtype)
    return (tokens * weights.unsqueeze(-1)).sum(dim=0)


def _frontier_node_ids(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis: SubgraphAnalysis,
) -> tuple[int, ...]:
    frontier_node_ids: list[int] = []
    for graph_node_id in analysis.selected_node_ids:
        outgoing = prepared_batch.graph_outgoing_edge_ids[int(graph_idx)].get(
            int(graph_node_id), ()
        )
        if any(not state.contains_edge(int(edge_id)) for edge_id in outgoing):
            frontier_node_ids.append(int(graph_node_id))
    return tuple(_dedup_preserve_order(frontier_node_ids))


def _answer_ready_node_ids(
    *, analysis: SubgraphAnalysis, full_mask: int
) -> tuple[int, ...]:
    return tuple(
        int(node_id)
        for node_id in analysis.selected_node_ids
        if int(analysis.reachability_bits.get(int(node_id), 0)) == int(full_mask)
    )


def _selected_relation_pool(
    *,
    prepared_batch: SubgraphPreparedBatch,
    edge_ids: tuple[int, ...],
    query_token: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if not edge_ids:
        return prepared_batch.relation_tokens.new_zeros(
            (int(prepared_batch.relation_tokens.size(-1)),),
            dtype=prepared_batch.relation_tokens.dtype,
        )
    edge_indices = torch.tensor(edge_ids, device=device, dtype=torch.long)
    relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_indices)
    relation_tokens = prepared_batch.relation_tokens.index_select(0, relation_ids)
    relation_scores = torch.matmul(
        relation_tokens.to(dtype=torch.float32), query_token.to(dtype=torch.float32)
    )
    relation_weights = torch.softmax(relation_scores, dim=0).to(
        dtype=relation_tokens.dtype
    )
    return (relation_tokens * relation_weights.unsqueeze(-1)).sum(dim=0)


def _coverage_features(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    state: SubgraphState,
    analysis: SubgraphAnalysis,
    full_mask: int,
) -> torch.Tensor:
    max_anchor_count = max(_bit_count(full_mask), 1)
    if analysis.selected_node_ids:
        coverages = [
            float(_bit_count(int(analysis.reachability_bits.get(int(node_id), 0))))
            / float(max_anchor_count)
            for node_id in analysis.selected_node_ids
        ]
        mean_coverage = float(sum(coverages)) / float(len(coverages))
        max_coverage = float(max(coverages))
    else:
        mean_coverage = 0.0
        max_coverage = 0.0
    frontier_edge_count = 0
    for graph_node_id in analysis.selected_node_ids:
        frontier_edge_count += sum(
            1
            for edge_id in prepared_batch.graph_outgoing_edge_ids[int(graph_idx)].get(
                int(graph_node_id), ()
            )
            if not state.contains_edge(int(edge_id))
        )
    redundancy_edges = max(
        int(analysis.num_selected_edges)
        - max(
            int(len(analysis.selected_node_ids)) - int(analysis.anchor_component_count),
            0,
        ),
        0,
    )
    full_coverage_nodes = float(
        sum(
            1
            for node_id in analysis.selected_node_ids
            if int(analysis.reachability_bits.get(int(node_id), 0)) == int(full_mask)
        )
    )
    return torch.tensor(
        [
            float(len(analysis.selected_node_ids)),
            float(analysis.num_selected_edges),
            float(analysis.anchor_component_count),
            float(mean_coverage),
            float(max_coverage),
            float(frontier_edge_count),
            float(redundancy_edges),
            float(full_coverage_nodes),
        ],
        dtype=torch.float32,
    )


class SubgraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        backbone: dict[str, Any],
        state_encoder: dict[str, Any],
        flow_head: dict[str, Any],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(backbone["hidden_dim"])
        state_struct_dim = 8
        self.backbone = EmbeddingBackbone(**backbone)
        self.state_encoder_norm = nn.LayerNorm((6 * self.hidden_dim) + state_struct_dim)
        self.state_encoder = _build_mlp(
            input_dim=(6 * self.hidden_dim) + state_struct_dim,
            output_dim=self.hidden_dim,
            hidden_dim=int(state_encoder["hidden_dim"]),
            num_layers=int(state_encoder["num_layers"]),
            dropout=float(state_encoder["dropout"]),
        )
        self.flow_head = StateFlowHead(
            node_dim=self.hidden_dim,
            question_dim=self.hidden_dim,
            hidden_dim=int(flow_head["hidden_dim"]),
            num_layers=int(flow_head["num_layers"]),
            dropout=float(flow_head["dropout"]),
            conditioning=str(flow_head["conditioning"]),
        )

    def prepare_batch(self, batch: TrajectoryBatch) -> SubgraphPreparedBatch:
        batch.require_raw_features()
        topology, observation = build_graph_batch(batch, validate=False)
        encoded = self.backbone.encode(
            BackboneInput(
                node_features=observation.node_features,
                relation_features=observation.relation_features,
                question_embedding=observation.question_embedding,
                question_context=observation.question_context,
                edge_index=topology.edge_index,
                edge_relations=topology.edge_type,
                num_nodes=topology.num_nodes,
                node_graph_index=topology.all_node_graph_index(
                    device=topology.edge_index.device
                ),
            )
        )
        return build_subgraph_prepared_batch(
            batch=batch,
            topology=topology,
            observation=observation,
            encoded=encoded,
        )

    def encode_states(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | Mapping[int, SubgraphAnalysis],
        state_indices: list[int] | tuple[int, ...] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        device = prepared_batch.device
        selected_state_indices = _normalize_state_indices(
            state_indices=state_indices,
            num_states=len(rollout_batch.states),
        )
        for state_idx in selected_state_indices:
            state = rollout_batch.states[int(state_idx)]
            graph_idx = int(rollout_batch.graph_ids[state_idx].item())
            if isinstance(analyses, Mapping):
                analysis = analyses[int(state_idx)]
            else:
                analysis = analyses[int(state_idx)]
            question_pool = prepared_batch.question_tokens[graph_idx]
            node_pool = _question_attention_pool(
                token_table=prepared_batch.node_tokens,
                token_ids=analysis.selected_node_ids,
                query_token=question_pool,
                device=device,
            )
            frontier_pool = _question_attention_pool(
                token_table=prepared_batch.node_tokens,
                token_ids=_frontier_node_ids(
                    prepared_batch=prepared_batch,
                    graph_idx=graph_idx,
                    state=state,
                    analysis=analysis,
                ),
                query_token=question_pool,
                device=device,
            )
            answer_ready_pool = _question_attention_pool(
                token_table=prepared_batch.node_tokens,
                token_ids=_answer_ready_node_ids(
                    analysis=analysis,
                    full_mask=int(prepared_batch.graph_anchor_full_mask[graph_idx]),
                ),
                query_token=question_pool,
                device=device,
            )
            anchor_pool = _question_attention_pool(
                token_table=prepared_batch.node_tokens,
                token_ids=prepared_batch.graph_anchor_abs_node_ids[graph_idx],
                query_token=question_pool,
                device=device,
            )
            relation_pool = _selected_relation_pool(
                prepared_batch=prepared_batch,
                edge_ids=state.edge_ids,
                query_token=question_pool,
                device=device,
            )
            struct = _coverage_features(
                prepared_batch=prepared_batch,
                graph_idx=graph_idx,
                state=state,
                analysis=analysis,
                full_mask=int(prepared_batch.graph_anchor_full_mask[graph_idx]),
            ).to(device=device)
            features.append(
                torch.cat(
                    (
                        node_pool,
                        frontier_pool,
                        answer_ready_pool,
                        anchor_pool,
                        relation_pool,
                        question_pool,
                        struct,
                    ),
                    dim=0,
                )
            )
        if not features:
            return prepared_batch.node_tokens.new_empty((0, self.hidden_dim))
        stacked = torch.stack(features, dim=0)
        stacked = align_float_input_dtype(stacked, module=self.state_encoder_norm)
        stacked = self.state_encoder_norm(stacked)
        stacked = align_float_input_dtype(stacked, module=self.state_encoder[0])
        return self.state_encoder(stacked)

    def compute_log_flows(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_log_flows_for_graph_ids(
            prepared_batch=prepared_batch,
            graph_ids=rollout_batch.graph_ids,
            state_features=state_features,
            done_mask=rollout_batch.done_mask,
        )

    def compute_log_flows_for_graph_ids(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_ids: torch.Tensor,
        state_features: torch.Tensor,
        done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(state_features.size(0)) != int(graph_ids.numel()):
            raise ValueError(
                "state_features must align with graph_ids in SubgraphEncoder.compute_log_flows_for_graph_ids."
            )
        question_features = prepared_batch.question_tokens.index_select(
            0, graph_ids.to(device=prepared_batch.device, dtype=torch.long)
        )
        log_flows = self.flow_head(state_features, question_features).to(
            dtype=torch.float32
        )
        if done_mask is None:
            return log_flows
        if tuple(done_mask.shape) != tuple(graph_ids.shape):
            raise ValueError(
                "done_mask must align with graph_ids in SubgraphEncoder.compute_log_flows_for_graph_ids."
            )
        return torch.where(
            done_mask.to(device=log_flows.device, dtype=torch.bool),
            torch.zeros_like(log_flows, dtype=torch.float32),
            log_flows,
        )


__all__ = ["SubgraphEncoder"]
