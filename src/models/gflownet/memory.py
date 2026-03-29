from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.models.configs import PrefixMemoryConfig

from .prefix import PrefixKey


_CUDA_MAX_SIMILARITY_VALUES = 8_388_608
_CPU_MAX_SIMILARITY_VALUES = 33_554_432


def _memory_query_chunk_size(
    *, device: torch.device, total_queries: int, num_keys: int
) -> int:
    if total_queries <= 0:
        return 1
    budget = (
        _CUDA_MAX_SIMILARITY_VALUES
        if device.type == "cuda"
        else _CPU_MAX_SIMILARITY_VALUES
    )
    return max(1, min(int(total_queries), budget // max(int(num_keys), 1)))


@dataclass(frozen=True)
class PrefixMemoryEntry:
    prefix_key: PrefixKey
    key_vector: torch.Tensor
    value_vector: torch.Tensor
    success: bool
    remaining_steps: int
    terminal_log_reward: float


class PrefixMemoryBank:
    def __init__(self, *, config: PrefixMemoryConfig, value_dim: int) -> None:
        self.config = config
        self.value_dim = int(value_dim)
        self._entries: deque[PrefixMemoryEntry] = deque()
        self._key_matrix_cache: torch.Tensor | None = None
        self._value_matrix_cache: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    @property
    def ready(self) -> bool:
        return self.enabled and len(self) >= int(self.config.min_entries)

    def clear(self) -> None:
        self._entries.clear()
        self._invalidate_cache()

    def add_entries(self, entries: list[PrefixMemoryEntry]) -> int:
        if not self.enabled or not entries:
            return 0
        added = 0
        for entry in entries:
            if tuple(entry.key_vector.shape) != (self.value_dim,):
                raise ValueError(
                    "PrefixMemoryEntry.key_vector must match the configured value_dim. "
                    f"expected={(self.value_dim,)} got={tuple(entry.key_vector.shape)}."
                )
            if tuple(entry.value_vector.shape) != (self.value_dim,):
                raise ValueError(
                    "PrefixMemoryEntry.value_vector must match the configured value_dim. "
                    f"expected={(self.value_dim,)} got={tuple(entry.value_vector.shape)}."
                )
            self._entries.append(entry)
            added += 1
            while len(self._entries) > int(self.config.capacity):
                self._entries.popleft()
        if added > 0:
            self._invalidate_cache()
        return added

    def retrieve(self, query_vectors: torch.Tensor) -> torch.Tensor:
        if int(query_vectors.numel()) == 0:
            return query_vectors.new_empty((0, self.value_dim), dtype=torch.float32)
        if not self.ready or len(self._entries) == 0:
            return query_vectors.new_zeros(
                (int(query_vectors.size(0)), self.value_dim), dtype=torch.float32
            )
        key_matrix, value_matrix = self._materialize_cache()
        key_matrix = key_matrix.to(device=query_vectors.device, dtype=torch.float32)
        value_matrix = value_matrix.to(device=query_vectors.device, dtype=torch.float32)
        normalized_keys = F.normalize(key_matrix, dim=-1)
        top_k = min(int(self.config.top_k), int(key_matrix.size(0)))
        total_queries = int(query_vectors.size(0))
        chunk_size = _memory_query_chunk_size(
            device=query_vectors.device,
            total_queries=total_queries,
            num_keys=int(key_matrix.size(0)),
        )
        retrieved = query_vectors.new_empty(
            (total_queries, self.value_dim), dtype=torch.float32
        )
        for chunk_start in range(0, total_queries, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_queries)
            chunk_queries = F.normalize(
                query_vectors[chunk_start:chunk_end].to(dtype=torch.float32),
                dim=-1,
            )
            similarity = chunk_queries @ normalized_keys.transpose(0, 1)
            top_similarity, top_indices = torch.topk(
                similarity,
                k=top_k,
                dim=-1,
                largest=True,
                sorted=True,
            )
            scaled_similarity = top_similarity / float(self.config.temperature)
            weights = torch.softmax(scaled_similarity, dim=-1)
            gathered_values = value_matrix.index_select(
                0, top_indices.reshape(-1)
            ).view(
                chunk_end - chunk_start,
                top_k,
                self.value_dim,
            )
            retrieved[chunk_start:chunk_end] = (
                weights.unsqueeze(-1) * gathered_values
            ).sum(dim=1)
        return retrieved

    def _invalidate_cache(self) -> None:
        self._key_matrix_cache = None
        self._value_matrix_cache = None

    def _materialize_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._key_matrix_cache is None or self._value_matrix_cache is None:
            self._key_matrix_cache = torch.stack(
                [
                    entry.key_vector.detach().cpu().to(dtype=torch.float32)
                    for entry in self._entries
                ],
                dim=0,
            )
            self._value_matrix_cache = torch.stack(
                [
                    entry.value_vector.detach().cpu().to(dtype=torch.float32)
                    for entry in self._entries
                ],
                dim=0,
            )
        return self._key_matrix_cache, self._value_matrix_cache


__all__ = [
    "PrefixMemoryBank",
    "PrefixMemoryEntry",
    "_memory_query_chunk_size",
]
