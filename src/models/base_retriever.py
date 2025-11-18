from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from .components import DDE, EmbeddingProjector, FeatureFusion, GroupRanker
from .outputs import BaseModelOutput

logger = logging.getLogger(__name__)


class BaseRetriever(nn.Module, ABC):
    """Shared projection, graph encoding, and ranking utilities for retrievers."""

    def __init__(
        self,
        *,
        emb_dim: int,
        topic_pe: bool = True,
        num_topics: int = 2,
        dde_cfg: Optional[Dict[str, Any]] = None,
        fusion_method: str = "film",
        dropout_p: float = 0.1,
        kge_interaction: str = "transe",
        enable_entity_finetune: bool = False,
        enable_query_finetune: bool = False,
        enable_relation_finetune: bool = False,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.topic_pe = bool(topic_pe)
        self.num_topics = int(num_topics)
        self.dropout_p = float(dropout_p)
        self.kge_interaction = kge_interaction.strip().lower()
        self.enable_entity_finetune = enable_entity_finetune
        self.enable_query_finetune = enable_query_finetune
        self.enable_relation_finetune = enable_relation_finetune

        dde_cfg = dict(dde_cfg or {})
        self.dde = DDE(
            num_rounds=int(dde_cfg.get("num_rounds", 2)),
            num_reverse_rounds=int(dde_cfg.get("num_reverse_rounds", 2)),
        )
        structure_dim = 0
        if self.topic_pe:
            structure_dim += self.num_topics
        structure_dim += (len(self.dde.layers) + len(self.dde.reverse_layers)) * self.num_topics
        self.feature_fusion = FeatureFusion(
            fusion_method=fusion_method,
            semantic_dim=self.emb_dim,
            structure_dim=structure_dim,
        )
        self.entity_projector = EmbeddingProjector(
            self.emb_dim,
            finetune=self.enable_entity_finetune,
        )
        self.query_projector = EmbeddingProjector(
            self.emb_dim,
            finetune=self.enable_query_finetune,
        )
        self.relation_projector = EmbeddingProjector(
            self.emb_dim,
            finetune=self.enable_relation_finetune,
        )
        self.ranker = GroupRanker()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_feature_dim(self) -> int:
        query_dim = self.emb_dim
        entity_dim = self.feature_fusion.get_output_dim()
        if self.kge_interaction == "concat":
            triple_dim = entity_dim * 2 + self.emb_dim
        else:
            triple_dim = entity_dim
        return query_dim + triple_dim

    def build_enhanced_entity_features(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if base_semantic_features_override is not None:
            semantic = base_semantic_features_override.to(self.device)
        else:
            semantic = self.entity_projector(data.node_embeddings)

        structure_inputs = []
        topic_input = data.topic_one_hot.float()
        if self.topic_pe:
            structure_inputs.append(topic_input)
        dde_outputs = self.dde(topic_input, data.edge_index, getattr(data, "reverse_edge_index", None))
        structure_inputs.extend(dde_outputs)
        structure = torch.cat(structure_inputs, dim=1)
        return self.feature_fusion(semantic, structure)

    def _prepare_embeddings(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
        *,
        align_q_to_edges: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_emb = data.question_emb
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        projected_query = self.query_projector(q_emb)

        enhanced_entities = self.build_enhanced_entity_features(
            data,
            base_semantic_features_override=base_semantic_features_override,
        )
        edge_index = data.edge_index
        h_emb = enhanced_entities[edge_index[0]]
        t_emb = enhanced_entities[edge_index[1]]
        r_emb = self.relation_projector(data.edge_embeddings)
        query_ids = data.batch[edge_index[0]]
        query_emb = projected_query[query_ids] if align_q_to_edges else projected_query
        return query_emb, h_emb, r_emb, t_emb, query_ids

    def _compute_triple_representation(
        self,
        h_emb: torch.Tensor,
        r_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.kge_interaction == "transe":
            return h_emb + r_emb - t_emb
        if self.kge_interaction == "distmult":
            return h_emb * r_emb * t_emb
        if self.kge_interaction == "rotate":
            h_real, h_imag = torch.chunk(h_emb, 2, dim=-1)
            t_real, t_imag = torch.chunk(t_emb, 2, dim=-1)
            r_real, r_imag = torch.chunk(r_emb, 2, dim=-1)
            hr_real = h_real * r_real - h_imag * r_imag
            hr_imag = h_real * r_imag + h_imag * r_real
            return torch.cat([hr_real - t_real, hr_imag - t_imag], dim=-1)
        if self.kge_interaction == "concat":
            return torch.cat([h_emb, r_emb, t_emb], dim=-1)
        raise NotImplementedError(f"KGE interaction '{self.kge_interaction}' not implemented.")

    def forward(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        return self._score_triples(data, base_semantic_features_override=base_semantic_features_override, **kwargs)

    def group_and_rank(
        self,
        ranking_scores: torch.Tensor,
        query_ids: torch.Tensor,
        *,
        k: Optional[int] = None,
    ) -> Tuple[list[torch.Tensor], list[torch.Tensor]]:
        groups = self.ranker.group(query_ids)
        orders = self.ranker.rank(ranking_scores, groups, k=k)
        return groups, orders

    @abstractmethod
    def _score_triples(
        self,
        data: Batch,
        base_semantic_features_override: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BaseModelOutput:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, data: Batch, k: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


__all__ = ["BaseRetriever"]
