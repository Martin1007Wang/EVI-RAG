"""Hydra-friendly component that manages shared vocab/embedding stores."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

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

    def relation_inverse_assets(
        self,
        *,
        suffix: Optional[str] = None,
        mapping_path: Optional[Path] = None,
        prefix: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        suffix_val = _INVERSE_RELATION_SUFFIX_DEFAULT if suffix is None else str(suffix)
        suffix_val = suffix_val.strip()
        if not suffix_val:
            raise ValueError("inverse_relation_suffix must be a non-empty string.")
        map_path = mapping_path
        if map_path is None:
            candidate = self.relation_vocab_path.parent / "inverse_relations.json"
            if candidate.exists():
                map_path = candidate
        prefix_val = "" if prefix is None else str(prefix).strip()
        cache_key = (str(map_path) if map_path is not None else "", prefix_val, suffix_val)
        cached = self._relation_inverse_cache.get(cache_key)
        if cached is not None:
            return cached
        if map_path is not None and map_path.exists():
            assets = _load_relation_inverse_assets_from_mapping(
                self.relation_vocab_path,
                mapping_path=map_path,
                prefix=prefix_val,
                suffix=suffix_val,
            )
        else:
            assets = _load_relation_inverse_assets(self.relation_vocab_path, suffix=suffix_val)
        self._relation_inverse_cache[cache_key] = assets
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


def _load_relation_inverse_assets_from_mapping(
    relation_vocab_path: Path,
    *,
    mapping_path: Path,
    prefix: str,
    suffix: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not relation_vocab_path.exists():
        raise FileNotFoundError(f"relation_vocab.parquet not found: {relation_vocab_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"inverse_relations.json not found: {mapping_path}")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyarrow is required to load relation_vocab.parquet.") from exc
    table = pq.read_table(relation_vocab_path, columns=["relation_id", "kg_id"])
    relation_ids = torch.as_tensor(table.column("relation_id").to_numpy(), dtype=torch.long)
    kg_ids = [str(val) for val in table.column("kg_id").to_pylist()]
    if relation_ids.numel() == _ZERO:
        raise ValueError("relation_vocab.parquet is empty.")
    max_id = int(relation_ids.max().detach().tolist())
    if max_id < _ZERO:
        raise ValueError("relation_vocab.parquet contains negative relation_id values.")
    vocab_size = max_id + _ONE
    id_lookup = {kg_id: int(rel_id) for rel_id, kg_id in zip(relation_ids.tolist(), kg_ids)}

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    inner = payload.get("inverse_relations", payload)
    entries: list[dict[str, str]] = []
    if isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("forward") and item.get("inverse_relation"):
                entries.append(
                    {
                        "forward": str(item["forward"]),
                        "inverse_relation": str(item["inverse_relation"]),
                    }
                )
    elif isinstance(inner, dict):
        for key, value in inner.items():
            if isinstance(value, dict):
                inv = value.get("inverse_relation")
                if inv:
                    entries.append({"forward": str(key), "inverse_relation": str(inv)})
    else:
        raise ValueError("inverse_relations payload must be a dict or list.")
    if not entries:
        raise ValueError("inverse_relations mapping is empty.")

    forward_set = {entry["forward"] for entry in entries}
    kg_id_set = set(kg_ids)

    inverse_map = torch.full((vocab_size,), _INVALID_RELATION_ID, dtype=torch.long)
    inverse_mask = torch.zeros((vocab_size,), dtype=torch.bool)

    pair_seen: set[tuple[str, str]] = set()
    for entry in entries:
        fwd = entry["forward"]
        inv = entry["inverse_relation"]
        if inv == fwd:
            candidate = f"{prefix}{fwd}" if prefix else None
            if candidate and candidate in kg_id_set:
                inv = candidate
            else:
                matches = [
                    kg for kg in kg_id_set if kg != fwd and kg.endswith(fwd) and (not prefix or kg.startswith(prefix))
                ]
                if len(matches) == 1:
                    inv = matches[0]
                else:
                    continue
        f_id = id_lookup.get(fwd)
        inv_id = id_lookup.get(inv)
        if f_id is None or inv_id is None:
            candidate = f"{prefix}{fwd}" if prefix else None
            if candidate and candidate in id_lookup:
                inv = candidate
                inv_id = id_lookup.get(inv)
            if f_id is None or inv_id is None:
                raise ValueError(f"inverse_relations mapping not found in relation_vocab: {fwd!r} -> {inv!r}")
        if int(f_id) == int(inv_id):
            continue
        if inverse_map[int(f_id)] not in (_INVALID_RELATION_ID, int(inv_id)):
            raise ValueError(f"inverse_relations conflict for {fwd!r}.")
        if inverse_map[int(inv_id)] not in (_INVALID_RELATION_ID, int(f_id)):
            raise ValueError(f"inverse_relations conflict for {inv!r}.")
        inverse_map[int(f_id)] = int(inv_id)
        inverse_map[int(inv_id)] = int(f_id)
        if inv not in forward_set:
            inverse_mask[int(inv_id)] = True
            continue
        a, b = (fwd, inv) if fwd < inv else (inv, fwd)
        if (a, b) in pair_seen:
            continue
        pair_seen.add((a, b))
        inverse_choice = b
        inv_choice_id = id_lookup.get(inverse_choice)
        if inv_choice_id is not None:
            inverse_mask[int(inv_choice_id)] = True

    return inverse_map, inverse_mask
