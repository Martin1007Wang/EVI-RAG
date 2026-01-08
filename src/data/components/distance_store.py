from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

DISTANCE_CACHE_PT = "pt"
DISTANCE_CACHE_ALLOWED = {DISTANCE_CACHE_PT}


class DistanceStore:
    def load(self, sample_id: str, num_nodes: int) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class DistancePTPayload:
    sample_ids: Sequence[str]
    ptr: torch.Tensor
    values: torch.Tensor


class DistancePTStore(DistanceStore):
    def __init__(self, pt_path: Path) -> None:
        self.path = Path(pt_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Distance PT cache not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("Distance PT cache must be a dict payload.")
        sample_ids = payload.get("sample_ids")
        ptr = payload.get("ptr")
        values = payload.get("values")
        if not isinstance(sample_ids, (list, tuple)):
            raise ValueError("Distance PT cache missing sample_ids list.")
        if not torch.is_tensor(ptr) or not torch.is_tensor(values):
            raise ValueError("Distance PT cache missing ptr/values tensors.")
        ptr = ptr.to(dtype=torch.long)
        values = values.to(dtype=torch.int16)
        if int(ptr.numel()) != len(sample_ids) + 1:
            raise ValueError("Distance PT cache ptr length mismatch with sample_ids.")
        self._payload = DistancePTPayload(
            sample_ids=[str(sid) for sid in sample_ids],
            ptr=ptr,
            values=values,
        )
        self._index: Dict[str, int] = {sid: idx for idx, sid in enumerate(self._payload.sample_ids)}
        LOGGER.info(
            "Loaded distance PT cache: %s samples=%d values=%d",
            self.path,
            len(self._payload.sample_ids),
            int(self._payload.values.numel()),
        )

    def load(self, sample_id: str, num_nodes: int) -> torch.Tensor:
        idx = self._index.get(sample_id)
        if idx is None:
            raise KeyError(f"Distance PT cache missing sample_id={sample_id}.")
        start = int(self._payload.ptr[idx].item())
        end = int(self._payload.ptr[idx + 1].item())
        if end - start != int(num_nodes):
            raise ValueError(
                f"Distance PT length mismatch for {sample_id}: "
                f"expected {int(num_nodes)} got {end - start}."
            )
        return self._payload.values[start:end]

    def close(self) -> None:
        return None
