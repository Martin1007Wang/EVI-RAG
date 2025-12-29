from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from .graph import DDE
from .projections import EmbeddingProjector


@dataclass
class RetrieverOutput:
    """Minimal container for retriever outputs."""

    logits: torch.Tensor
    query_ids: torch.Tensor
    relation_ids: torch.Tensor | None = None
    logits_fwd: torch.Tensor | None = None
    logits_bwd: torch.Tensor | None = None
    edge_embeddings: torch.Tensor | None = None

    def detach(self) -> "RetrieverOutput":
        return RetrieverOutput(
            logits=self.logits.detach(),
            query_ids=self.query_ids.detach(),
            relation_ids=self.relation_ids.detach() if self.relation_ids is not None else None,
            logits_fwd=self.logits_fwd.detach() if self.logits_fwd is not None else None,
            logits_bwd=self.logits_bwd.detach() if self.logits_bwd is not None else None,
            edge_embeddings=self.edge_embeddings.detach() if self.edge_embeddings is not None else None,
        )


class Retriever(nn.Module):
    """Retriever with configurable scoring heads."""

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        topic_pe: bool = True,
        num_topics: int = 2,
        dde_cfg: Optional[Dict[str, int]] = None,
        dropout_p: float = 0.1,
        core_mode: str = "geometry",
        direction_mode: str = "bidirectional",
        **_: Any,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_topic_pe = bool(topic_pe)
        if not self.use_topic_pe:
            raise ValueError("topic_pe must be enabled; retriever requires topic_one_hot + DDE.")
        self.num_topics = int(num_topics)
        if self.num_topics <= 0:
            raise ValueError(f"num_topics must be positive, got {self.num_topics}")
        _ = self._normalize_core_mode(core_mode)
        self.direction_mode = self._normalize_direction_mode(direction_mode)

        self.entity_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        self.relation_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        self.query_proj = EmbeddingProjector(self.emb_dim, input_dim=self.emb_dim, finetune=True)
        self.non_text_entity_emb = nn.Embedding(1, self.emb_dim)

        dde_cfg = dde_cfg or {}
        self.dde = DDE(**dde_cfg)
        num_rounds = int(self.dde.num_rounds)
        num_rev = int(self.dde.num_reverse_rounds)
        self._topic_struct_dim = self.num_topics * (1 + num_rounds + num_rev)
        struct_raw_dim = 2 * self._topic_struct_dim

        self._use_transe = True
        self._use_distmult = True
        self.register_buffer(
            "parity_meta",
            torch.tensor(
                [
                    int(self.use_topic_pe),
                    int(self.num_topics),
                    int(num_rounds),
                    int(num_rev),
                ],
                dtype=torch.long,
            ),
        )

        self.q_gate = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.Sigmoid())
        self.q_bias = nn.Sequential(nn.Linear(self.emb_dim, self.emb_dim), nn.Tanh())

        self.struct_proj = nn.Sequential(
            nn.Linear(struct_raw_dim, self.emb_dim),
            nn.LayerNorm(self.emb_dim),
            nn.GELU(),
        )
        self.struct_gate_net = nn.Sequential(
            nn.Linear(self.emb_dim, 1),
            nn.Sigmoid(),
        )

        fusion_in_dim = self.emb_dim
        if self._use_distmult:
            fusion_in_dim += self.emb_dim
        if self._use_transe:
            fusion_in_dim += self.emb_dim + 1
        self.state_net = nn.Sequential(
            nn.Linear(fusion_in_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.score_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, batch: Any) -> RetrieverOutput:
        output, _ = self._forward_impl(batch, return_features=False)
        return output

    def extract_edge_tokens(self, batch: Any) -> torch.Tensor:
        _, features = self._forward_impl(batch, return_features=True)
        if features is None:
            raise RuntimeError("extract_edge_tokens expected non-empty features but got None.")
        return features

    def _forward_impl(
        self,
        batch: Any,
        *,
        return_features: bool,
    ) -> tuple[RetrieverOutput, Optional[torch.Tensor]]:
        edge_index = getattr(batch, "edge_index", None)
        if edge_index is None:
            raise ValueError("Batch missing edge_index required for scoring.")
        edge_index = edge_index.to(device=next(self.parameters()).device)
        head_idx, tail_idx = edge_index
        if head_idx.numel() == 0:
            return self._empty_output(head_idx=head_idx, return_features=return_features)

        (
            query_ids,
            query_repr,
            head_repr,
            tail_repr,
            relation_repr,
            struct_fwd,
            struct_bwd,
        ) = self._prepare_edge_inputs(
            batch,
            head_idx=head_idx,
            tail_idx=tail_idx,
            edge_index=edge_index,
        )

        logits_fwd = None
        features_fwd = None
        if self.direction_mode in {"forward", "bidirectional"}:
            logits_fwd, features_fwd = self._score_edges(
                query_repr=query_repr,
                head_repr=head_repr,
                relation_repr=relation_repr,
                tail_repr=tail_repr,
                struct_feat_raw=struct_fwd,
            )
        logits_bwd = None
        features_bwd = None
        if self.direction_mode in {"backward", "bidirectional"}:
            logits_bwd, features_bwd = self._score_edges(
                query_repr=query_repr,
                head_repr=tail_repr,
                relation_repr=relation_repr,
                tail_repr=head_repr,
                struct_feat_raw=struct_bwd,
            )

        if self.direction_mode == "bidirectional":
            if logits_fwd is None or logits_bwd is None or features_fwd is None or features_bwd is None:
                raise RuntimeError("bidirectional mode requires both forward and backward scores.")
            logits, edge_embeddings = self._combine_directional_outputs(
                logits_fwd=logits_fwd,
                logits_bwd=logits_bwd,
                features_fwd=features_fwd,
                features_bwd=features_bwd,
            )
        elif self.direction_mode == "forward":
            if logits_fwd is None or features_fwd is None:
                raise RuntimeError("forward mode requires forward scores.")
            logits = logits_fwd
            edge_embeddings = features_fwd
        else:
            if logits_bwd is None or features_bwd is None:
                raise RuntimeError("backward mode requires backward scores.")
            logits = logits_bwd
            edge_embeddings = features_bwd
        relation_ids = getattr(batch, "edge_attr", None)

        output = RetrieverOutput(
            logits=logits,
            query_ids=query_ids,
            relation_ids=relation_ids,
            logits_fwd=logits_fwd,
            logits_bwd=logits_bwd,
            edge_embeddings=edge_embeddings,
        )
        if return_features:
            return output, edge_embeddings
        return output, None

    @staticmethod
    def _combine_directional_outputs(
        *,
        logits_fwd: torch.Tensor,
        logits_bwd: torch.Tensor,
        features_fwd: torch.Tensor,
        features_bwd: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([logits_fwd, logits_bwd], dim=0)
        weights = torch.softmax(stacked, dim=0)
        logits = (weights * stacked).sum(dim=0)
        edge_embeddings = weights[0].unsqueeze(-1) * features_fwd + weights[1].unsqueeze(-1) * features_bwd
        return logits, edge_embeddings

    def _empty_output(self, *, head_idx: torch.Tensor, return_features: bool) -> tuple[RetrieverOutput, Optional[torch.Tensor]]:
        param = next(self.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        device = head_idx.device
        empty_logits = torch.empty(0, device=device, dtype=dtype)
        empty_ids = head_idx.new_empty(0)
        feature_dim = self.hidden_dim
        features = torch.empty((0, feature_dim), device=device, dtype=dtype)
        output = RetrieverOutput(
            logits=empty_logits,
            query_ids=empty_ids,
            relation_ids=None,
            logits_fwd=empty_logits,
            logits_bwd=empty_logits,
            edge_embeddings=features,
        )
        if return_features:
            return output, features
        return output, None

    def _prepare_edge_inputs(
        self,
        batch: Any,
        *,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        question_emb = getattr(batch, "question_emb", None)
        node_embedding_ids = getattr(batch, "node_embedding_ids", None)
        edge_attr = getattr(batch, "edge_attr", None)
        if question_emb is None or node_embedding_ids is None or edge_attr is None:
            raise ValueError("Batch must provide question_emb, node_embedding_ids, and edge_attr.")

        device = next(self.parameters()).device
        question_emb = question_emb.to(device=device)

        node_embeddings = getattr(batch, "node_embeddings", None)
        edge_embeddings = getattr(batch, "edge_embeddings", None)
        if node_embeddings is None or edge_embeddings is None:
            raise ValueError("Batch must provide node_embeddings and edge_embeddings.")
        node_embeddings = node_embeddings.to(device=device, non_blocking=True)
        edge_embeddings = edge_embeddings.to(device=device, non_blocking=True)

        query_ids = self._compute_query_ids(batch, head_idx)
        query_repr = self.query_proj(question_emb)[query_ids]

        node_repr = self._project_nodes(node_embeddings, node_embedding_ids)
        head_repr = node_repr[head_idx]
        tail_repr = node_repr[tail_idx]
        relation_repr = self.relation_proj(edge_embeddings)

        struct_fwd = self._build_structure_features(
            batch,
            head_idx,
            tail_idx,
            edge_index=edge_index,
            num_nodes=node_repr.size(0),
        )
        struct_bwd = self._build_structure_features(
            batch,
            tail_idx,
            head_idx,
            edge_index=edge_index,
            num_nodes=node_repr.size(0),
        )

        return query_ids, query_repr, head_repr, tail_repr, relation_repr, struct_fwd, struct_bwd

    def _score_edges(
        self,
        *,
        query_repr: torch.Tensor,
        head_repr: torch.Tensor,
        relation_repr: torch.Tensor,
        tail_repr: torch.Tensor,
        struct_feat_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_gate is None or self.q_bias is None or self.state_net is None or self.score_head is None:
            raise RuntimeError("Geometry scoring requires q_gate/q_bias/state_net/score_head to be initialized.")
        r_ctx = relation_repr * self.q_gate(query_repr) + self.q_bias(query_repr)

        struct_ctx, nav_gate = self._encode_structure(
            struct_feat_raw=struct_feat_raw,
        )

        combined_parts = []
        if self._use_distmult:
            interaction_vec = head_repr * r_ctx * tail_repr
            modulated_interaction = interaction_vec * nav_gate
            combined_parts.append(modulated_interaction)
        combined_parts.append(struct_ctx)
        if self._use_transe:
            error_vec = head_repr + r_ctx - tail_repr
            dist_scalar = -torch.norm(error_vec, p=2, dim=-1, keepdim=True)
            combined_parts.append(error_vec)
            combined_parts.append(dist_scalar)
        combined = torch.cat(combined_parts, dim=-1)
        features = self.state_net(combined)
        logits = self.score_head(features).squeeze(-1)
        return logits, features

    def _encode_structure(
        self,
        *,
        struct_feat_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if struct_feat_raw is None:
            raise ValueError("struct_feat_raw is required; ensure topic_one_hot + DDE are enabled.")
        struct_ctx = self.struct_proj(struct_feat_raw)
        nav_gate = self.struct_gate_net(struct_ctx)
        return struct_ctx, nav_gate

    def _project_nodes(self, node_embeddings: torch.Tensor, node_embedding_ids: torch.Tensor) -> torch.Tensor:
        node_repr = self.entity_proj(node_embeddings)
        non_text_proj = self.entity_proj(self.non_text_entity_emb.weight)[0].to(dtype=node_repr.dtype)
        if not isinstance(node_embedding_ids, torch.Tensor):
            node_embedding_ids = torch.as_tensor(node_embedding_ids, dtype=torch.long, device=node_repr.device)
        else:
            node_embedding_ids = node_embedding_ids.to(device=node_repr.device, dtype=torch.long)
        non_text_mask = node_embedding_ids == 0
        if non_text_mask.any():
            node_repr = torch.where(non_text_mask.unsqueeze(-1), non_text_proj.unsqueeze(0), node_repr)
        return node_repr

    def _build_structure_features(
        self,
        batch,
        head_idx: torch.Tensor,
        tail_idx: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        topic_one_hot = getattr(batch, "topic_one_hot", None)
        if topic_one_hot is None:
            raise ValueError("topic_one_hot is required for DDE-based structure features.")
        if not torch.is_tensor(topic_one_hot):
            topic_one_hot = torch.as_tensor(topic_one_hot, dtype=torch.float32, device=edge_index.device)
        else:
            topic_one_hot = topic_one_hot.to(device=edge_index.device, dtype=torch.float32)
        if topic_one_hot.dim() == 1:
            topic_one_hot = topic_one_hot.unsqueeze(-1)
        if topic_one_hot.dim() != 2:
            raise ValueError(f"topic_one_hot must be 2D (N, C), got shape {tuple(topic_one_hot.shape)}")
        if topic_one_hot.size(0) != int(num_nodes):
            raise ValueError(f"topic_one_hot first dim {topic_one_hot.size(0)} != num_nodes {int(num_nodes)}")
        if topic_one_hot.size(-1) < self.num_topics:
                raise ValueError(
                    f"topic_one_hot feature dim {topic_one_hot.size(-1)} < num_topics={self.num_topics}; "
                    "rebuild g_retrieval caches or update configs/build_retrieval_pipeline.yaml."
                )
        if topic_one_hot.size(-1) != self.num_topics:
            topic_one_hot = topic_one_hot[..., : self.num_topics]
        feats: list[torch.Tensor] = [topic_one_hot]
        reverse_edge_index = getattr(batch, "reverse_edge_index", None)
        if reverse_edge_index is not None and torch.is_tensor(reverse_edge_index):
            reverse_edge_index = reverse_edge_index.to(device=edge_index.device)
        feats.extend(self.dde(topic_one_hot, edge_index, reverse_edge_index))
        stacked = torch.stack(feats, dim=-1)
        node_struct = stacked.reshape(stacked.size(0), -1)
        head_struct = node_struct[head_idx]
        tail_struct = node_struct[tail_idx]
        return torch.cat([head_struct, tail_struct], dim=-1)

    @staticmethod
    def _normalize_core_mode(mode: str) -> str:
        mode_clean = str(mode or "").strip().lower()
        if mode_clean in {"geometry", "structured", "full"}:
            return "geometry"
        raise ValueError(f"core_mode must be 'geometry' for DDE-based retriever, got {mode!r}")

    @staticmethod
    def _normalize_direction_mode(mode: str) -> str:
        mode_clean = str(mode or "").strip().lower()
        if mode_clean in {"bidirectional", "forward", "backward"}:
            return mode_clean
        raise ValueError(
            "direction_mode must be one of {'bidirectional', 'forward', 'backward'}, "
            f"got {mode!r}."
        )

    def _compute_query_ids(self, batch, head_idx: torch.Tensor) -> torch.Tensor:
        edge_batch = getattr(batch, "edge_batch", None)
        if edge_batch is None:
            raise ValueError("Batch missing edge_batch; provide explicit edge-to-graph mapping in the DataLoader.")
        if not torch.is_tensor(edge_batch):
            edge_batch = torch.as_tensor(edge_batch, dtype=torch.long, device=head_idx.device)
        else:
            edge_batch = edge_batch.to(device=head_idx.device, dtype=torch.long)
        edge_batch = edge_batch.view(-1)
        if edge_batch.numel() != head_idx.numel():
            raise ValueError(f"edge_batch length mismatch: {edge_batch.numel()} vs edges {head_idx.numel()}")
        return edge_batch


__all__ = ["Retriever"]
