from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

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
        topic_pe: bool = True,
        num_topics: int = 2,
        dde_cfg: Optional[Dict[str, int]] = None,
        dropout_p: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_topic_pe = bool(topic_pe)
        self.num_topics = int(num_topics)

        # Projectors are always trainable: deterministic projection is part of the retriever operator.
        self.entity_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        self.relation_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        self.query_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        # SubgraphRAG parity: trainable non-text entity representation for embedding_id=0.
        self.non_text_entity_emb = nn.Embedding(1, self.emb_dim)

        self.dde = DDE(**(dde_cfg or {})) if self.use_topic_pe else None

        semantic_dim = self.emb_dim * 4  # concat [q,h,r,t]
        struct_dim = 0
        if self.use_topic_pe:
            num_rounds = int(self.dde.num_rounds) if self.dde is not None else 0
            num_rev = int(self.dde.num_reverse_rounds) if self.dde is not None else 0
            struct_dim = 2 * self.num_topics * (1 + num_rounds + num_rev)  # head+tail concatenation
        else:
            num_rounds = 0
            num_rev = 0
        fusion_dim = semantic_dim + struct_dim

        # Minimal metadata persisted in checkpoints for downstream consumers (e.g., GFlowNet embedder).
        # [use_topic_pe, num_topics, num_rounds, num_reverse_rounds]
        self.register_buffer(
            "parity_meta",
            torch.tensor(
                [int(self.use_topic_pe), int(self.num_topics), int(num_rounds), int(num_rev)],
                dtype=torch.long,
            ),
        )

        self.feature_extractor = DenseFeatureExtractor(
            input_dim=fusion_dim,
            emb_dim=self.emb_dim,
            hidden_dim=self.hidden_dim,
            dropout_p=dropout_p,
        )
        self.head = DeterministicHead(hidden_dim=self.hidden_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, batch) -> RetrieverOutput:
        output, _ = self._forward_impl(batch, return_features=False)
        return output

    def extract_edge_tokens(self, batch: Any) -> torch.Tensor:
        """Return per-edge dense tokens aligned with DenseFeatureExtractor output (pre-head)."""
        _, features = self._forward_impl(batch, return_features=True)
        if features is None:
            raise RuntimeError("extract_edge_tokens expected non-empty features but got None.")
        return features

    def _forward_impl(self, batch: Any, *, return_features: bool) -> tuple[RetrieverOutput, Optional[torch.Tensor]]:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for scoring.")
        head_idx, tail_idx = edge_index
        if head_idx.numel() == 0:
            param = next(self.parameters(), None)
            dtype = param.dtype if param is not None else torch.float32
            device = head_idx.device
            empty_scores = torch.empty(0, device=device, dtype=dtype)
            empty_ids = head_idx.new_empty(0)
            output = RetrieverOutput(scores=empty_scores, logits=empty_scores, query_ids=empty_ids, relation_ids=None)
            if return_features:
                features = torch.empty((0, self.hidden_dim), device=device, dtype=dtype)
                return output, features
            return output, None

        question_emb = getattr(batch, "question_emb", None)
        node_embeddings = getattr(batch, "node_embeddings", None)
        node_embedding_ids = getattr(batch, "node_embedding_ids", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if question_emb is None or node_embeddings is None or node_embedding_ids is None or edge_embeddings is None:
            raise ValueError("Batch must provide question_emb, node_embeddings, node_embedding_ids, and edge_embeddings.")

        query_ids = self._compute_query_ids(batch, head_idx)
        query_repr = self.query_proj(question_emb)[query_ids]

        node_repr = self.entity_proj(node_embeddings)
        non_text_proj = self.entity_proj(self.non_text_entity_emb.weight)[0].to(dtype=node_repr.dtype)
        if not isinstance(node_embedding_ids, torch.Tensor):
            node_embedding_ids = torch.as_tensor(node_embedding_ids, dtype=torch.long, device=node_repr.device)
        else:
            node_embedding_ids = node_embedding_ids.to(device=node_repr.device, dtype=torch.long)
        non_text_mask = node_embedding_ids == 0
        if non_text_mask.any():
            node_repr = torch.where(non_text_mask.unsqueeze(-1), non_text_proj.unsqueeze(0), node_repr)
        head_repr = node_repr[head_idx]
        tail_repr = node_repr[tail_idx]
        relation_repr = self.relation_proj(edge_embeddings)

        semantic = torch.cat([query_repr, head_repr, relation_repr, tail_repr], dim=-1)
        structure = self._build_structure_features(batch, head_idx, tail_idx) if self.use_topic_pe else None
        fused = torch.cat([semantic, structure], dim=-1) if structure is not None else semantic

        features = self.feature_extractor(self.dropout(fused))
        logits = self.head(features)
        scores = torch.sigmoid(logits)
        relation_ids = getattr(batch, "edge_attr", None)
        output = RetrieverOutput(scores=scores, logits=logits, query_ids=query_ids, relation_ids=relation_ids)
        return (output, features) if return_features else (output, None)

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
        features = [topic]  # [N, num_topics]
        if self.dde is not None:
            features.extend(self.dde(topic, batch.edge_index, getattr(batch, "reverse_edge_index", None)))
        stacked = torch.stack(features, dim=-1)
        node_struct = stacked.reshape(stacked.size(0), -1)
        head_struct = node_struct[head_idx]
        tail_struct = node_struct[tail_idx]
        return torch.cat([head_struct, tail_struct], dim=-1)

    def _compute_query_ids(self, batch, head_idx: torch.Tensor) -> torch.Tensor:
        node_batch = getattr(batch, "batch", None)
        if node_batch is not None:
            return node_batch[head_idx]
        ptr = getattr(batch, "ptr", None)
        if ptr is not None:
            return torch.bucketize(head_idx, ptr[1:], right=False)
        return head_idx.new_zeros(head_idx.numel(), dtype=head_idx.dtype)


__all__ = ["Retriever"]
