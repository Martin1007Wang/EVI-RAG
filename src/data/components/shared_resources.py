"""Hydra-friendly component that manages shared vocab/embedding stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.data.schema.constants import _NON_TEXT_EMBEDDING_ID, _ONE, _ZERO
from .embeddings import GlobalEmbeddingStore


class SharedDataResources:
    """Handle shared heavy-weight stores for multiple datasets.

    Lightning's :class:`~lightning.LightningDataModule` may construct several
    :class:`GRetrievalDataset` instances (train/val/test). Each dataset used to
    build its own :class:`GlobalEmbeddingStore`, which duplicates expensive
    embedding loads. ``SharedDataResources`` keeps a single copy that can be
    injected wherever needed, aligning with the lightning-hydra-template style
    ``components`` that Hydra can instantiate.
    """

    def __init__(self, *, entity_vocab_path: Path, embeddings_dir: Path, embeddings_device: Optional[str] = None) -> None:
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.embeddings_dir = Path(embeddings_dir).expanduser().resolve()
        self.embeddings_device = None if embeddings_device is None else str(embeddings_device)
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._entity_embedding_map: Optional[torch.Tensor] = None

    @property
    def global_embeddings(self) -> GlobalEmbeddingStore:
        if self._global_embeddings is None:
            self._global_embeddings = GlobalEmbeddingStore(
                embeddings_dir=self.embeddings_dir,
                entity_vocab_path=self.entity_vocab_path,
                device=self.embeddings_device,
            )
        return self._global_embeddings

    @property
    def entity_embedding_map(self) -> torch.Tensor:
        if self._entity_embedding_map is None:
            self._entity_embedding_map = _load_entity_embedding_map(self.entity_vocab_path)
        return self._entity_embedding_map

    def clear(self) -> None:
        """Drop cached stores so new configs can be applied safely."""

        if self._global_embeddings is not None:
            self._global_embeddings.clear_device_cache()
        self._global_embeddings = None
        self._entity_embedding_map = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_global_embeddings"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def _load_entity_embedding_map(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"entity_vocab.parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to load entity_vocab.parquet.") from exc
    table = pq.read_table(path, columns=["entity_id", "embedding_id"])
    entity_ids = torch.as_tensor(table.column("entity_id").to_numpy(), dtype=torch.long)
    embedding_ids = torch.as_tensor(table.column("embedding_id").to_numpy(), dtype=torch.long)
    if entity_ids.numel() == _ZERO:
        raise ValueError("entity_vocab.parquet is empty.")
    if int(entity_ids.min().detach().tolist()) < _ZERO:
        raise ValueError("entity_vocab.parquet contains negative entity_id values.")
    max_id = int(entity_ids.max().detach().tolist())
    mapping = torch.full((max_id + _ONE,), _NON_TEXT_EMBEDDING_ID, dtype=torch.long)
    mapping[entity_ids] = embedding_ids
    return mapping
