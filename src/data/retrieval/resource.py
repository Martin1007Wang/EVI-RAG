"""Hydra-friendly component that manages shared vocab/embedding stores."""

from __future__ import annotations

from pathlib import Path
import torch

from .embedding_store import EmbeddingStore


class DataResource:
    def __init__(self, entity_metadata_path: Path, embeddings_dir: Path) -> None:
        self.embedding_store = EmbeddingStore(embeddings_dir)
        self.entity_embedding_map, self.cvt_mask = _load_entity_metadata(
            entity_metadata_path
        )


def _load_entity_metadata(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"entity metadata not found: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Entity metadata at {path} must decode to a dict.")
    entity_embedding_map = payload.get("entity_embedding_map")
    cvt_mask = payload.get("cvt_mask")
    if (
        not torch.is_tensor(entity_embedding_map)
        or entity_embedding_map.dtype != torch.long
    ):
        raise TypeError("entity_embedding_map must be a torch.long tensor.")
    if not torch.is_tensor(cvt_mask) or cvt_mask.dtype != torch.bool:
        raise TypeError("cvt_mask must be a torch.bool tensor.")
    if entity_embedding_map.dim() != 1 or cvt_mask.dim() != 1:
        raise ValueError("entity metadata tensors must be 1D.")
    if int(entity_embedding_map.numel()) != int(cvt_mask.numel()):
        raise ValueError("entity_embedding_map/cvt_mask length mismatch.")
    return entity_embedding_map, cvt_mask
