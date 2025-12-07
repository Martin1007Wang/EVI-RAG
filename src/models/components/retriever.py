from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .fusion import FeatureFusion
from .graph import DDE
from .heads import DenseFeatureExtractor, DeterministicHead
from .projections import EmbeddingProjector


@dataclass
class RetrieverOutput:
    """Minimal container for retriever outputs."""
    scores: torch.Tensor
    logits: torch.Tensor
    query_ids: torch.Tensor
    relation_ids: torch.Tensor | None = None

    def detach(self) -> "RetrieverOutput":
        return RetrieverOutput(
            scores=self.scores.detach(),
            logits=self.logits.detach(),
            query_ids=self.query_ids.detach(),
            relation_ids=self.relation_ids.detach() if self.relation_ids is not None else None,
        )


class Retriever(nn.Module):
    """Retriever built from static question/entity/relation representations."""

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        kge_interaction: str = "concat",
        topic_pe: bool = True,
        num_topics: int = 2,
        dde_cfg: Optional[Dict[str, int]] = None,
        fusion_method: str = "film",
        dropout_p: float = 0.0,
        enable_entity_finetune: bool = False,
        enable_query_finetune: bool = False,
        enable_relation_finetune: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.kge_interaction = str(kge_interaction).strip().lower()
        self.use_topic_pe = bool(topic_pe)
        self.num_topics = int(num_topics)

        self.entity_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=enable_entity_finetune)
        self.relation_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=enable_relation_finetune)
        self.query_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=enable_query_finetune)

        self.dde = DDE(**(dde_cfg or {})) if self.use_topic_pe else None

        semantic_dim = self._semantic_dim()
        if self.use_topic_pe:
            self.fusion = FeatureFusion(
                fusion_method=fusion_method,
                semantic_dim=semantic_dim,
                structure_dim=self.num_topics,
            )
            fusion_dim = self.fusion.get_output_dim()
        else:
            self.fusion = None
            fusion_dim = semantic_dim

        self.feature_extractor = DenseFeatureExtractor(
            input_dim=fusion_dim,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            dropout_p=dropout_p,
        )
        self.head = DeterministicHead(hidden_dim=self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, batch) -> RetrieverOutput:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for scoring.")
        head_idx, tail_idx = edge_index
        if head_idx.numel() == 0:
            empty = head_idx.new_empty(0)
            return RetrieverOutput(scores=empty, logits=empty, query_ids=empty, relation_ids=None)

        question_emb = getattr(batch, "question_emb", None)
        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if question_emb is None or node_embeddings is None or edge_embeddings is None:
            raise ValueError("Batch must provide question_emb, node_embeddings, and edge_embeddings.")

        query_ids = self._compute_query_ids(batch, head_idx)
        query_repr = self.query_proj(question_emb)[query_ids]

        node_repr = self.entity_proj(node_embeddings)
        head_repr = node_repr[head_idx]
        tail_repr = node_repr[tail_idx]
        relation_repr = self.relation_proj(edge_embeddings)

        semantic = self._compose_semantic(query_repr, head_repr, relation_repr, tail_repr)
        structure = self._build_structure_features(batch, head_idx, tail_idx) if self.use_topic_pe else None
        fused = self.fusion(semantic, structure) if (self.fusion is not None and structure is not None) else semantic

        features = self.feature_extractor(self.dropout(fused))
        logits = self.head(features)
        scores = torch.sigmoid(logits)
        relation_ids = getattr(batch, "edge_attr", None)
        return RetrieverOutput(scores=scores, logits=logits, query_ids=query_ids, relation_ids=relation_ids)

    def _semantic_dim(self) -> int:
        if self.kge_interaction == "concat":
            return self.emb_dim * 4
        if self.kge_interaction in {"add", "mul", "sub"}:
            return self.emb_dim
        raise ValueError(f"Unsupported kge_interaction={self.kge_interaction}")

    def _compose_semantic(
        self,
        query: torch.Tensor,
        head: torch.Tensor,
        relation: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
        if self.kge_interaction == "concat":
            return torch.cat([query, head, relation, tail], dim=-1)
        if self.kge_interaction == "add":
            return query + head + relation + tail
        if self.kge_interaction == "mul":
            return query * head * tail
        if self.kge_interaction == "sub":
            return query + head - tail
        raise ValueError(f"Unsupported kge_interaction={self.kge_interaction}")

    def _build_structure_features(
        self,
        batch,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
    ) -> torch.Tensor:
        topic = getattr(batch, "topic_one_hot", None)
        if topic is None:
            raise ValueError("topic_pe is enabled but batch.topic_one_hot is missing.")
        if topic.dim() == 1:
            topic = topic.unsqueeze(-1)
        features = [topic]
        if self.dde is not None:
            features.extend(self.dde(topic, batch.edge_index, getattr(batch, "reverse_edge_index", None)))
        stacked = torch.stack(features, dim=-1)
        node_struct = stacked.mean(dim=-1)
        head_struct = node_struct[head_idx]
        tail_struct = node_struct[tail_idx]
        return 0.5 * (head_struct + tail_struct)

    def _compute_query_ids(self, batch, head_idx: torch.Tensor) -> torch.Tensor:
        node_batch = getattr(batch, "batch", None)
        if node_batch is not None:
            return node_batch[head_idx]
        ptr = getattr(batch, "ptr", None)
        if ptr is not None:
            return torch.bucketize(head_idx, ptr[1:], right=False)
        return head_idx.new_zeros(head_idx.numel(), dtype=head_idx.dtype)


__all__ = ["Retriever"]
