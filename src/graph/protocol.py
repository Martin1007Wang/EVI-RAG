from __future__ import annotations

from typing import Protocol, Sequence

import torch


class GraphBatchProtocol(Protocol):
    @property
    def num_graphs(self) -> int: ...

    @property
    def node_ptr(self) -> torch.Tensor: ...

    @property
    def edge_index(self) -> torch.Tensor: ...

    @property
    def edge_rel_global(self) -> torch.Tensor: ...

    @property
    def node_embeddings(self) -> torch.Tensor: ...

    @property
    def edge_embeddings(self) -> torch.Tensor | None: ...

    @property
    def relation_embeddings(self) -> torch.Tensor | None: ...

    @property
    def edge_rel_local(self) -> torch.Tensor | None: ...

    @property
    def question_emb(self) -> torch.Tensor: ...

    @property
    def question_ctx(self) -> torch.Tensor: ...

    @property
    def question_ctx_mask(self) -> torch.Tensor: ...

    @property
    def q_local_indices(self) -> torch.Tensor: ...

    @property
    def q_ptr(self) -> torch.Tensor: ...

    @property
    def node_entity_ids(self) -> torch.Tensor: ...

    @property
    def sample_ids(self) -> Sequence[str]: ...


__all__ = ["GraphBatchProtocol"]
