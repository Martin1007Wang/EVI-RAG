from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schema import RetrievalBatch


def l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1, eps=eps)


class EntityEmbeddingLayer(nn.Module):
    """
    Resolve entity catalog ids into node embeddings.

    Text entities use precomputed PLM embeddings.
    Non-text entities share one learnable embedding.

    Output is L2-normalized so text and non-text entities stay in the same
    cosine-geometry convention.
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        non_text_init_std: float = 0.02,
        normalize: bool = False,
    ) -> None:
        super().__init__()

        text = entity_text_embeddings.to(dtype=torch.float32).contiguous()
        if normalize:
            text = l2_normalize(text)

        self.register_buffer("_entity_text_embeddings", text)
        self.register_buffer(
            "_entity_embedding_map",
            entity_embedding_map.to(dtype=torch.long).contiguous(),
        )

        dim = int(text.size(-1))
        self.non_text_embedding = nn.Parameter(
            torch.randn(dim, dtype=torch.float32) * float(non_text_init_std)
        )
        self.normalize = bool(normalize)

    @property
    def entity_text_embeddings(self) -> torch.Tensor:
        return cast(torch.Tensor, self.get_buffer("_entity_text_embeddings"))

    @property
    def entity_embedding_map(self) -> torch.Tensor:
        return cast(torch.Tensor, self.get_buffer("_entity_embedding_map"))

    @property
    def embedding_dim(self) -> int:
        return int(self.entity_text_embeddings.size(-1))

    def forward(self, entity_ids: torch.Tensor) -> torch.Tensor:
        table = self.entity_text_embeddings
        mapping = self.entity_embedding_map

        entity_ids = entity_ids.to(device=mapping.device, dtype=torch.long).view(-1)
        text_ids = mapping.index_select(0, entity_ids)
        has_text = text_ids.ge(0)

        out = table.new_empty(entity_ids.numel(), self.embedding_dim)

        if bool(has_text.any()):
            idx = has_text.nonzero(as_tuple=False).view(-1)
            out.index_copy_(
                0,
                idx,
                table.index_select(0, text_ids.index_select(0, idx)),
            )

        if bool((~has_text).any()):
            idx = (~has_text).nonzero(as_tuple=False).view(-1)
            non_text = self.non_text_embedding.to(device=out.device, dtype=out.dtype)
            if self.normalize:
                non_text = l2_normalize(non_text.view(1, -1)).view(-1)
            out.index_copy_(0, idx, non_text.view(1, -1).expand(idx.numel(), -1))

        return l2_normalize(out) if self.normalize else out


class RoleProjection(nn.Module):
    """
    Lightweight role-specific projection from semantic space to model space.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")

        self.proj = nn.Linear(self.dim, self.dim, bias=False)
        self.norm = nn.LayerNorm(self.dim)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


@dataclass(frozen=True)
class FeatureBank:
    """
    Static graph/query features.

    These are independent of rollout state.
    """

    # Model-space features for readout, residual, stop, and flow modules.
    node_h: torch.Tensor  # [num_nodes, H]
    rel_h: torch.Tensor  # [num_edges, H]
    query_h: torch.Tensor  # [num_graphs, H]

    # Semantic-space features for cosine prior diagnostics/scoring only.
    node_sem_h: torch.Tensor  # [num_nodes, H]
    rel_sem_h: torch.Tensor  # [num_edges, H]
    query_sem_h: torch.Tensor  # [num_graphs, H]

    anchor_forward_bucket: torch.Tensor
    anchor_backward_bucket: torch.Tensor
    anchor_mask: torch.Tensor  # [num_nodes]

    node_is_non_text: torch.Tensor | None = None


class SemanticFeatureEncoder(nn.Module):
    """
    Static feature resolver.

    It maps:
        node entity catalog ids -> node_sem_h -> node_h
        edge relation catalog ids -> rel_sem_h -> rel_h
        question_emb -> query_sem_h -> query_h
        anchor distances -> discrete buckets
        anchor ids -> anchor mask

    It must not encode active subgraphs or rollout decisions.
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        relation_embeddings: torch.Tensor,
        embedding_dim: int = 1024,
        hidden_dim: int = 1024,
        anchor_distance_max: int = 3,
        non_text_init_std: float = 0.02,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if int(hidden_dim) != int(embedding_dim):
            raise ValueError(
                "SemanticFeatureEncoder assumes hidden_dim == embedding_dim; "
                f"got hidden_dim={hidden_dim}, embedding_dim={embedding_dim}."
            )

        self.hidden_dim = int(hidden_dim)
        self.anchor_distance_max = int(anchor_distance_max)
        self.normalize = bool(normalize)

        self.entity_embedding = EntityEmbeddingLayer(
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            non_text_init_std=non_text_init_std,
            normalize=normalize,
        )

        rel = relation_embeddings.to(dtype=torch.float32).contiguous()
        if normalize:
            rel = l2_normalize(rel)

        self.register_buffer("_relation_embeddings", rel)

        self.query_projection = RoleProjection(self.hidden_dim)
        self.node_projection = RoleProjection(self.hidden_dim)
        self.rel_projection = RoleProjection(self.hidden_dim)

    @property
    def relation_embedding_table(self) -> torch.Tensor:
        return cast(torch.Tensor, self.get_buffer("_relation_embeddings"))

    def forward(self, batch: RetrievalBatch) -> FeatureBank:
        rel_table = self.relation_embedding_table
        device = rel_table.device

        node_sem_h = self.entity_embedding(
            batch.node_entity_catalog_ids.to(device=device, dtype=torch.long)
        )

        edge_rel_ids = batch.edge_relation_catalog_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)

        rel_sem_h = rel_table.index_select(0, edge_rel_ids)

        query_sem_h = batch.question_emb.to(device=device, dtype=node_sem_h.dtype)
        if query_sem_h.ndim == 1:
            query_sem_h = query_sem_h.view(1, -1)
        if self.normalize:
            query_sem_h = l2_normalize(query_sem_h)

        node_h = self.node_projection(node_sem_h)
        rel_h = self.rel_projection(rel_sem_h)
        query_h = self.query_projection(query_sem_h)

        num_nodes = int(node_sem_h.size(0))

        return FeatureBank(
            node_h=node_h,
            rel_h=rel_h,
            query_h=query_h,
            node_sem_h=node_sem_h,
            rel_sem_h=rel_sem_h,
            query_sem_h=query_sem_h,
            anchor_forward_bucket=self._bucket(
                batch.anchor_node_forward_distances_flat,
                num_nodes=num_nodes,
                device=device,
            ),
            anchor_backward_bucket=self._bucket(
                batch.anchor_node_backward_distances_flat,
                num_nodes=num_nodes,
                device=device,
            ),
            anchor_mask=self._anchor_mask(
                batch=batch,
                num_nodes=num_nodes,
                device=device,
            ),
            node_is_non_text=self._node_is_non_text(
                batch=batch,
                num_nodes=num_nodes,
                device=device,
            ),
        )

    def _bucket(
        self,
        distance: torch.Tensor,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        distance = distance.to(device=device, dtype=torch.long).view(num_nodes)

        bucket = torch.zeros(num_nodes, dtype=torch.long, device=device)
        reachable = distance.ge(0)

        bucket[reachable] = (
            distance[reachable]
            .clamp(
                min=0,
                max=self.anchor_distance_max,
            )
            .add(1)
        )

        return bucket

    @staticmethod
    def _anchor_mask(
        *,
        batch: RetrievalBatch,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        anchor_ids = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)

        mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if anchor_ids.numel() > 0:
            mask[anchor_ids] = True

        return mask

    @staticmethod
    def _node_is_non_text(
        *,
        batch: RetrievalBatch,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if hasattr(batch, "non_text_node_mask"):
            return batch.non_text_node_mask.to(
                device=device,
                dtype=torch.bool,
            ).view(num_nodes)

        if hasattr(batch, "is_non_text_entity"):
            return batch.is_non_text_entity.to(
                device=device,
                dtype=torch.bool,
            ).view(num_nodes)

        return None


__all__ = [
    "EntityEmbeddingLayer",
    "FeatureBank",
    "RoleProjection",
    "SemanticFeatureEncoder",
    "l2_normalize",
]
