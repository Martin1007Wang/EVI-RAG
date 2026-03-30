from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.graph import TrajectoryBatch, build_graph_batch
from src.models.components import EmbeddingBackbone, StateFlowHead
from src.models.components.embedding import BackboneInput
from src.utils.precision_utils import align_float_input_dtype

from .prepared_batch import SubgraphPreparedBatch, build_subgraph_prepared_batch
from .state import SubgraphAnalysis, SubgraphRolloutBatch


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


def _selected_node_pool(
    *,
    node_tokens: torch.Tensor,
    selected_nodes: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not selected_nodes:
        return node_tokens.new_zeros(
            (int(node_tokens.size(-1)),), dtype=node_tokens.dtype
        )
    indices = torch.tensor(selected_nodes, device=device, dtype=torch.long)
    return node_tokens.index_select(0, indices).mean(dim=0)


def _selected_relation_pool(
    *,
    prepared_batch: SubgraphPreparedBatch,
    edge_ids: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not edge_ids:
        return prepared_batch.relation_tokens.new_zeros(
            (int(prepared_batch.relation_tokens.size(-1)),),
            dtype=prepared_batch.relation_tokens.dtype,
        )
    edge_indices = torch.tensor(edge_ids, device=device, dtype=torch.long)
    relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_indices)
    return prepared_batch.relation_tokens.index_select(0, relation_ids).mean(dim=0)


def _coverage_features(
    *,
    graph_idx: int,
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
    return torch.tensor(
        [
            float(graph_idx),
            float(len(analysis.selected_node_ids)),
            float(analysis.num_selected_edges),
            float(analysis.anchor_component_count),
            float(mean_coverage),
            float(max_coverage),
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
        state_struct_dim = 6
        self.backbone = EmbeddingBackbone(**backbone)
        self.state_encoder_norm = nn.LayerNorm((3 * self.hidden_dim) + state_struct_dim)
        self.state_encoder = _build_mlp(
            input_dim=(3 * self.hidden_dim) + state_struct_dim,
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
        analyses: tuple[SubgraphAnalysis, ...],
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        device = prepared_batch.device
        for state_idx, state in enumerate(rollout_batch.states):
            graph_idx = int(rollout_batch.graph_ids[state_idx].item())
            analysis = analyses[state_idx]
            node_pool = _selected_node_pool(
                node_tokens=prepared_batch.node_tokens,
                selected_nodes=analysis.selected_node_ids,
                device=device,
            )
            relation_pool = _selected_relation_pool(
                prepared_batch=prepared_batch,
                edge_ids=state.edge_ids,
                device=device,
            )
            question_pool = prepared_batch.question_tokens[graph_idx]
            struct = _coverage_features(
                graph_idx=graph_idx,
                analysis=analysis,
                full_mask=int(prepared_batch.graph_anchor_full_mask[graph_idx]),
            ).to(device=device)
            features.append(
                torch.cat((node_pool, relation_pool, question_pool, struct), dim=0)
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
        question_features = prepared_batch.question_tokens.index_select(
            0, rollout_batch.graph_ids
        )
        log_flows = self.flow_head(state_features, question_features).to(
            dtype=torch.float32
        )
        return torch.where(
            rollout_batch.done_mask,
            torch.zeros_like(log_flows, dtype=torch.float32),
            log_flows,
        )


__all__ = ["SubgraphEncoder"]
