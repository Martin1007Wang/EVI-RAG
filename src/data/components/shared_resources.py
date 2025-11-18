"""Hydra-friendly component that manages shared vocabulary/embedding stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .graph_store import GraphStore
from .embedding_store import GlobalEmbeddingStore


class SharedDataResources:
    """Handle shared heavy-weight stores for multiple datasets.

    Lightning's :class:`~lightning.LightningDataModule` may construct several
    :class:`RetrievalDataset` instances (train/val/test). Each dataset used to
    build its own :class:`GraphStore` and :class:`GlobalEmbeddingStore`, which
    duplicates expensive vocabulary/embedding loads. ``SharedDataResources``
    keeps a single copy that can be injected wherever needed, aligning with the
    lightning-hydra-template style ``components`` that Hydra can instantiate.
    """

    def __init__(self, *, vocabulary_path: Path, embeddings_dir: Path) -> None:
        self.vocabulary_path = Path(vocabulary_path).expanduser().resolve()
        self.embeddings_dir = Path(embeddings_dir).expanduser().resolve()
        self._graph_store: Optional[GraphStore] = None
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None

    @property
    def graph_store(self) -> GraphStore:
        if self._graph_store is None:
            self._graph_store = GraphStore(vocabulary_path=str(self.vocabulary_path))
        return self._graph_store

    @property
    def global_embeddings(self) -> GlobalEmbeddingStore:
        if self._global_embeddings is None:
            self._global_embeddings = GlobalEmbeddingStore(
                embeddings_dir=self.embeddings_dir,
                vocabulary_path=self.vocabulary_path,
            )
        return self._global_embeddings

    def clear(self) -> None:
        """Drop cached stores so new configs can be applied safely."""

        self._graph_store = None
        if self._global_embeddings is not None:
            self._global_embeddings.clear_device_cache()
        self._global_embeddings = None
