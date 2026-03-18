"""Hydra-friendly component that manages shared vocab/embedding stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.data.schema.constants import _NON_TEXT_EMBEDDING_ID
from .embeddings import GlobalEmbeddingStore


class SharedDataResources:
    """Handle shared heavy-weight stores for multiple datasets.

    Lightning's :class:`~lightning.LightningDataModule` may construct several
    :class:`GraphRetrievalDataset` instances (train/val/test). Each dataset used to
    build its own :class:`GlobalEmbeddingStore`, which duplicates expensive
    embedding loads. ``SharedDataResources`` keeps a single copy that can be
    injected wherever needed, aligning with the lightning-hydra-template style
    ``components`` that Hydra can instantiate.
    """

    def __init__(
        self,
        *,
        entity_vocab_path: Path,
        relation_vocab_path: Path,
        embeddings_dir: Path,
        embeddings_device: Optional[str] = None,
        heuristic_log_v_path: Optional[Path] = None,
    ) -> None:
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve()
        self.embeddings_dir = Path(embeddings_dir).expanduser().resolve()
        self.embeddings_device = (
            None if embeddings_device is None else str(embeddings_device)
        )
        self.heuristic_log_v_path = (
            None
            if heuristic_log_v_path is None
            else Path(heuristic_log_v_path).expanduser().resolve()
        )
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._entity_embedding_map: Optional[torch.Tensor] = None
        self._cvt_mask: Optional[torch.Tensor] = None
        self._heuristic_log_v: Optional[torch.Tensor] = None

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
            self._entity_embedding_map = _load_entity_embedding_map(
                self.entity_vocab_path
            )
        return self._entity_embedding_map

    @property
    def cvt_mask(self) -> torch.Tensor:
        if self._cvt_mask is None:
            self._cvt_mask = _load_cvt_mask(self.entity_vocab_path)
        return self._cvt_mask

    @property
    def heuristic_log_v(self) -> Optional[torch.Tensor]:
        if self.heuristic_log_v_path is None:
            return None
        if self._heuristic_log_v is None:
            self._heuristic_log_v = _load_heuristic_log_v(self.heuristic_log_v_path)
        return self._heuristic_log_v

    def clear(self) -> None:
        """Drop cached stores so new configs can be applied safely."""

        if self._global_embeddings is not None:
            self._global_embeddings.clear_device_cache()
        self._global_embeddings = None
        self._entity_embedding_map = None
        self._cvt_mask = None
        self._heuristic_log_v = None

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
        raise ModuleNotFoundError(
            "pyarrow is required to load entity_vocab.parquet."
        ) from exc
    table = pq.read_table(path, columns=["entity_id", "embedding_id"])
    entity_ids = torch.as_tensor(table.column("entity_id").to_numpy(), dtype=torch.long)
    embedding_ids = torch.as_tensor(
        table.column("embedding_id").to_numpy(), dtype=torch.long
    )
    if entity_ids.numel() == 0:
        raise ValueError("entity_vocab.parquet is empty.")
    if int(entity_ids.min().detach().tolist()) < 0:
        raise ValueError("entity_vocab.parquet contains negative entity_id values.")
    max_id = int(entity_ids.max().detach().tolist())
    mapping = torch.full((max_id + 1,), _NON_TEXT_EMBEDDING_ID, dtype=torch.long)
    mapping[entity_ids] = embedding_ids
    return mapping


def _load_cvt_mask(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"entity_vocab.parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required to load entity_vocab.parquet."
        ) from exc
    table = pq.read_table(path, columns=["entity_id", "is_cvt"])
    entity_ids = torch.as_tensor(table.column("entity_id").to_numpy(), dtype=torch.long)
    is_cvt = torch.as_tensor(table.column("is_cvt").to_numpy(), dtype=torch.bool)
    if entity_ids.numel() == 0:
        raise ValueError("entity_vocab.parquet is empty.")
    if entity_ids.numel() != is_cvt.numel():
        raise ValueError("entity_vocab.parquet entity_id/is_cvt length mismatch.")
    max_id = int(entity_ids.max().detach().tolist())
    if max_id < 0:
        raise ValueError("entity_vocab.parquet contains negative entity_id values.")
    mask = torch.zeros((max_id + 1,), dtype=torch.bool)
    mask[entity_ids] = is_cvt
    return mask


def _load_heuristic_log_v(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"heuristic_log_v tensor not found: {path}")
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"heuristic_log_v must be a torch.Tensor, got {type(tensor)!r}")
    if tensor.dim() != 1:
        raise ValueError("heuristic_log_v must be a 1D tensor.")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"heuristic_log_v must be floating point, got {tensor.dtype}")
    return tensor
