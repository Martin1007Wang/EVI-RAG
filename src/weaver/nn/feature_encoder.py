from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from src.data.schema import RetrievalBatch
from src.utils.nn_utils import init_xavier

from .dde import DirectionalDDE
from .static_graph_features import edge_relation_log_frequency, node_log_degree


class EntityEmbeddingLayer(nn.Module):
    """
    Resolve entity catalog ids into semantic-space node embeddings.

    Text entities use precomputed PLM embeddings.
    Non-text entities use one shared learnable embedding.
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        non_text_init_std: float = 0.02,
    ) -> None:
        super().__init__()

        text = entity_text_embeddings.to(dtype=torch.float32).contiguous()
        if text.ndim != 2:
            raise ValueError(
                f"entity_text_embeddings must have shape [N, D], got {tuple(text.shape)}."
            )
        if text.size(0) <= 0 or text.size(1) <= 0:
            raise ValueError("entity_text_embeddings must be non-empty.")

        mapping = entity_embedding_map.to(dtype=torch.long).contiguous()
        if mapping.ndim != 1:
            raise ValueError(
                f"entity_embedding_map must have shape [num_entities], got {tuple(mapping.shape)}."
            )

        self.register_buffer("_entity_text_embeddings", text)
        self.register_buffer("_entity_embedding_map", mapping)

        self.non_text_embedding = nn.Parameter(
            torch.randn(text.size(1), dtype=torch.float32) * float(non_text_init_std)
        )

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

        text_h = table.index_select(0, text_ids.clamp_min(0))

        non_text_h = self.non_text_embedding.to(device=table.device, dtype=table.dtype)
        return torch.where(
            has_text.unsqueeze(-1),
            text_h,
            non_text_h.view(1, -1).expand_as(text_h),
        )


class RoleProjection(nn.Module):
    """
    Role-specific projection from semantic space to model space.

    Without position bias:

        h = LN(Wx)

    With additive position bias:

        h = LN(Wx + b)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        init: str = "identity",
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)

        init = str(init)
        if init == "identity" and self.input_dim == self.hidden_dim:
            nn.init.eye_(self.proj.weight)
        elif init in {"identity", "xavier"}:
            init_xavier(self.proj)
        else:
            raise ValueError(
                f"role_projection_init must be 'identity' or 'xavier', got {init!r}."
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.proj(x)

        if bias is not None:
            if bias.shape != h.shape:
                raise ValueError(
                    f"bias must have shape {tuple(h.shape)}, got {tuple(bias.shape)}."
                )
            h = h + bias

        return self.norm(h)


@dataclass(frozen=True)
class FeatureBank:
    """
    Static graph/query features independent of rollout state.

    Model-space features are consumed by state readout, flow, Stop gate,
    and residual edge scoring.

    Semantic-space features are consumed by semantic edge priors.
    """

    node_h: torch.Tensor  # [num_nodes, H]
    rel_h: torch.Tensor  # [num_edges, H]
    query_h: torch.Tensor  # [num_graphs, H]

    node_sem_h: torch.Tensor  # [num_nodes, D]
    rel_sem_h: torch.Tensor  # [num_edges, D]
    query_sem_h: torch.Tensor  # [num_graphs, D]

    node_dde: torch.Tensor | None = None  # [num_nodes, D_dde]
    node_is_non_text: torch.Tensor | None = None
    node_log_degree: torch.Tensor | None = None  # [num_nodes]
    edge_relation_log_frequency: torch.Tensor | None = None  # [num_edges]


class FeatureEncoder(nn.Module):
    """
    Resolve static batch features.

    It maps:
        entity ids         -> node_sem_h  -> node_h
        relation ids       -> rel_sem_h   -> rel_h
        question embedding -> query_sem_h -> query_h

    Anchor-relative structure is injected only into node_h:

        node_h = LN(W_node node_sem_h + p_anchor)

    It does not encode rollout state, active subgraphs, rewards, labels,
    target-path teacher signals, or terminal outcomes.
    """

    def __init__(
        self,
        *,
        entity_text_embeddings: torch.Tensor,
        entity_embedding_map: torch.Tensor,
        relation_embeddings: torch.Tensor,
        embedding_dim: int | None = None,
        hidden_dim: int = 1024,
        dde: dict[str, object] | None = None,
        non_text_init_std: float = 0.02,
        role_projection: str | None = None,
        role_projection_init: str = "identity",
    ) -> None:
        super().__init__()

        rel = relation_embeddings.to(dtype=torch.float32).contiguous()
        if rel.ndim != 2:
            raise ValueError(
                f"relation_embeddings must have shape [R, D], got {tuple(rel.shape)}."
            )

        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")

        self.entity_embedding = EntityEmbeddingLayer(
            entity_text_embeddings=entity_text_embeddings,
            entity_embedding_map=entity_embedding_map,
            non_text_init_std=non_text_init_std,
        )

        self.embedding_dim = self.entity_embedding.embedding_dim
        if embedding_dim is not None and int(embedding_dim) != self.embedding_dim:
            raise ValueError(
                f"embedding_dim must match entity embedding dim: {embedding_dim} "
                f"!= {self.embedding_dim}."
            )
        if int(rel.size(-1)) != self.embedding_dim:
            raise ValueError(
                "relation embedding dim must match entity embedding dim: "
                f"got relation={rel.size(-1)}, entity={self.embedding_dim}."
            )
        if role_projection not in (None, "linear_layernorm"):
            raise ValueError(
                "role_projection must be 'linear_layernorm' when provided, "
                f"got {role_projection!r}."
            )

        self.register_buffer("_relation_embeddings", rel)

        dde_cfg = dict(dde or {})
        dde_enabled = bool(dde_cfg.pop("enabled", True))
        if dde_cfg.pop("type", "directional") != "directional":
            raise ValueError("feature_encoder.dde.type must be 'directional'.")
        self.dde: DirectionalDDE | None = None
        self.dde_projection: nn.Linear | None = None
        self.dde_dim = 0
        if dde_enabled:
            self.dde = DirectionalDDE(**dde_cfg)
            self.dde_dim = self.dde.output_dim
            self.dde_projection = nn.Linear(self.dde_dim, self.hidden_dim, bias=False)
            init_xavier(self.dde_projection)

        self.node_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            init=role_projection_init,
        )
        self.rel_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            init=role_projection_init,
        )
        self.query_projection = RoleProjection(
            input_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            init=role_projection_init,
        )

    @property
    def relation_embedding_table(self) -> torch.Tensor:
        return cast(torch.Tensor, self.get_buffer("_relation_embeddings"))

    def forward(self, batch: RetrievalBatch) -> FeatureBank:
        rel_table = self.relation_embedding_table
        device = rel_table.device

        node_sem_h = self.entity_embedding(
            batch.node_entity_catalog_ids.to(device=device, dtype=torch.long)
        )
        num_nodes = int(node_sem_h.size(0))

        rel_ids = batch.edge_relation_catalog_ids.to(
            device=device,
            dtype=torch.long,
        ).view(-1)
        rel_sem_h = rel_table.index_select(0, rel_ids)
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)

        query_sem_h = batch.question_emb.to(device=device, dtype=node_sem_h.dtype)
        if query_sem_h.ndim == 1:
            query_sem_h = query_sem_h.view(1, -1)
        if query_sem_h.ndim != 2:
            raise ValueError(
                f"question_emb must have shape [B, D], got {tuple(query_sem_h.shape)}."
            )
        if query_sem_h.size(-1) != self.embedding_dim:
            raise ValueError(
                f"question_emb dim must be {self.embedding_dim}, got {query_sem_h.size(-1)}."
            )

        node_dde = self._node_dde(
            batch=batch,
            num_nodes=num_nodes,
            device=device,
            dtype=node_sem_h.dtype,
        )
        structural_bias = self._structural_bias(
            node_dde=node_dde,
            device=device,
            dtype=node_sem_h.dtype,
        )

        return FeatureBank(
            node_h=self.node_projection(node_sem_h, bias=structural_bias),
            rel_h=self.rel_projection(rel_sem_h),
            query_h=self.query_projection(query_sem_h),
            node_sem_h=node_sem_h,
            rel_sem_h=rel_sem_h,
            query_sem_h=query_sem_h,
            node_dde=node_dde,
            node_is_non_text=self._node_is_non_text(
                batch=batch,
                num_nodes=num_nodes,
                device=device,
                entity_embedding_map=self.entity_embedding.entity_embedding_map,
            ),
            node_log_degree=node_log_degree(
                edge_index=edge_index,
                num_nodes=num_nodes,
                dtype=node_sem_h.dtype,
            ),
            edge_relation_log_frequency=self._edge_relation_log_frequency(
                batch=batch,
                rel_ids=rel_ids,
                device=device,
                dtype=node_sem_h.dtype,
            ),
        )

    def _node_dde(
        self,
        *,
        batch: RetrievalBatch,
        num_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.dde is None:
            return torch.zeros((num_nodes, 0), device=device, dtype=dtype)

        return self.dde(
            edge_index=batch.edge_index.to(device=device, dtype=torch.long),
            anchor_node_ids=batch.anchor_node_ids,
            num_nodes=num_nodes,
        ).to(device=device, dtype=dtype)

    def _structural_bias(
        self,
        *,
        node_dde: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        bias: torch.Tensor | None = None

        if self.dde_projection is not None:
            bias = self.dde_projection(node_dde.to(device=device, dtype=dtype))

        return bias

    @staticmethod
    def _node_is_non_text(
        *,
        batch: RetrievalBatch,
        num_nodes: int,
        device: torch.device,
        entity_embedding_map: torch.Tensor,
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

        node_entity_catalog_ids = batch.node_entity_catalog_ids.to(
            device=entity_embedding_map.device,
            dtype=torch.long,
        ).view(num_nodes)
        return entity_embedding_map.index_select(0, node_entity_catalog_ids).lt(0).to(
            device=device,
            dtype=torch.bool,
        )

    @staticmethod
    def _edge_relation_log_frequency(
        *,
        batch: RetrievalBatch,
        rel_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not hasattr(batch, "edge_batch"):
            return None

        return edge_relation_log_frequency(
            relation_ids=rel_ids.to(device=device, dtype=torch.long),
            edge_batch=batch.edge_batch.to(device=device, dtype=torch.long),
            dtype=dtype,
        )


__all__ = [
    "DirectionalDDE",
    "EntityEmbeddingLayer",
    "FeatureBank",
    "FeatureEncoder",
    "RoleProjection",
]
