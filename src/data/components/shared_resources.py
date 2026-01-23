"""Hydra-friendly component that manages shared vocab/embedding stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from src.data.schema.constants import _INVERSE_RELATION_SUFFIX_DEFAULT, _NON_TEXT_EMBEDDING_ID, _ONE, _ZERO
from .embeddings import GlobalEmbeddingStore

_INVALID_RELATION_ID = -1


class SharedDataResources:
    """Handle shared heavy-weight stores for multiple datasets.

    Lightning's :class:`~lightning.LightningDataModule` may construct several
    :class:`GRetrievalDataset` instances (train/val/test). Each dataset used to
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
    ) -> None:
        self.entity_vocab_path = Path(entity_vocab_path).expanduser().resolve()
        self.relation_vocab_path = Path(relation_vocab_path).expanduser().resolve()
        self.embeddings_dir = Path(embeddings_dir).expanduser().resolve()
        self.embeddings_device = None if embeddings_device is None else str(embeddings_device)
        self._global_embeddings: Optional[GlobalEmbeddingStore] = None
        self._entity_embedding_map: Optional[torch.Tensor] = None
        self._cvt_mask: Optional[torch.Tensor] = None
        self._relation_inverse_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

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

    @property
    def cvt_mask(self) -> torch.Tensor:
        if self._cvt_mask is None:
            self._cvt_mask = _load_cvt_mask(self.entity_vocab_path)
        return self._cvt_mask

    def relation_inverse_assets(self, *, suffix: Optional[str] = None) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_val = _INVERSE_RELATION_SUFFIX_DEFAULT if suffix is None else str(suffix)
        suffix_val = suffix_val.strip()
        if not suffix_val:
            raise ValueError("inverse_relation_suffix must be a non-empty string.")
        cached = self._relation_inverse_cache.get(suffix_val)
        if cached is not None:
            return cached
        assets = _load_relation_inverse_assets(self.relation_vocab_path, suffix=suffix_val)
        self._relation_inverse_cache[suffix_val] = assets
        return assets

    def clear(self) -> None:
        """Drop cached stores so new configs can be applied safely."""

        if self._global_embeddings is not None:
            self._global_embeddings.clear_device_cache()
        self._global_embeddings = None
        self._entity_embedding_map = None
        self._cvt_mask = None
        self._relation_inverse_cache = {}

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


def _load_cvt_mask(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"entity_vocab.parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to load entity_vocab.parquet.") from exc
    table = pq.read_table(path, columns=["entity_id", "is_cvt"])
    entity_ids = torch.as_tensor(table.column("entity_id").to_numpy(), dtype=torch.long)
    is_cvt = torch.as_tensor(table.column("is_cvt").to_numpy(), dtype=torch.bool)
    if entity_ids.numel() == _ZERO:
        raise ValueError("entity_vocab.parquet is empty.")
    if entity_ids.numel() != is_cvt.numel():
        raise ValueError("entity_vocab.parquet entity_id/is_cvt length mismatch.")
    max_id = int(entity_ids.max().detach().tolist())
    if max_id < _ZERO:
        raise ValueError("entity_vocab.parquet contains negative entity_id values.")
    mask = torch.zeros((max_id + _ONE,), dtype=torch.bool)
    mask[entity_ids] = is_cvt
    return mask


def _load_relation_inverse_assets(path: Path, *, suffix: str) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"relation_vocab.parquet not found: {path}")
    suffix_val = str(suffix).strip()
    if not suffix_val:
        raise ValueError("inverse_relation_suffix must be a non-empty string.")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to load relation_vocab.parquet.") from exc
    table = pq.read_table(path, columns=["relation_id", "kg_id"])
    relation_ids = torch.as_tensor(table.column("relation_id").to_numpy(), dtype=torch.long)
    kg_ids = [str(val) for val in table.column("kg_id").to_pylist()]
    if relation_ids.numel() == _ZERO:
        raise ValueError("relation_vocab.parquet is empty.")
    max_id = int(relation_ids.max().detach().tolist())
    if max_id < _ZERO:
        raise ValueError("relation_vocab.parquet contains negative relation_id values.")
    vocab_size = max_id + _ONE
    id_lookup = {kg_id: int(rel_id) for rel_id, kg_id in zip(relation_ids.tolist(), kg_ids)}
    inverse_map = torch.full((vocab_size,), _INVALID_RELATION_ID, dtype=torch.long)
    inverse_mask = torch.zeros((vocab_size,), dtype=torch.bool)
    for rel_id, kg_id in zip(relation_ids.tolist(), kg_ids):
        inv_key = kg_id[: -len(suffix_val)] if kg_id.endswith(suffix_val) else f"{kg_id}{suffix_val}"
        inv_id = id_lookup.get(inv_key)
        if inv_id is not None:
            inverse_map[int(rel_id)] = int(inv_id)
        if kg_id.endswith(suffix_val):
            inverse_mask[int(rel_id)] = True
    return inverse_map, inverse_mask
