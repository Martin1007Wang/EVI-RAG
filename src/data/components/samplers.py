from __future__ import annotations

from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler

_ZERO = 0


class CycleSampler(Sampler[int]):
    """Samples a fixed-size epoch by cycling through a permutation without replacement.

    - If `shuffle=True`, each full pass generates a fresh random permutation.
    - If `epoch_size < len(dataset)`, then consecutive epochs cover disjoint slices
      of the same permutation, guaranteeing full coverage after enough epochs.
    """

    def __init__(
        self,
        data_source: Sized,
        *,
        epoch_size: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(data_source)
        self._n = int(len(data_source))
        if self._n <= _ZERO:
            raise ValueError("CycleSampler requires a non-empty dataset.")
        self._epoch_size = int(epoch_size)
        if self._epoch_size <= _ZERO:
            raise ValueError("epoch_size must be a positive integer.")
        if self._epoch_size > self._n:
            raise ValueError(f"epoch_size must be <= len(dataset). Got epoch_size={self._epoch_size} > {self._n}.")
        self._shuffle = bool(shuffle)
        self._generator = generator
        self._order: Optional[torch.Tensor] = None
        self._offset = _ZERO

    def __len__(self) -> int:
        return self._epoch_size

    def __iter__(self) -> Iterator[int]:
        remaining = self._epoch_size
        while remaining > _ZERO:
            order = self._ensure_order()
            take = min(remaining, int(order.numel()) - int(self._offset))
            if take <= _ZERO:
                self._reset_order()
                continue
            chunk = order.narrow(0, int(self._offset), int(take))
            self._offset += int(take)
            remaining -= int(take)
            yield from (int(idx) for idx in chunk.tolist())

    def _ensure_order(self) -> torch.Tensor:
        if self._order is None:
            self._reset_order()
        if self._order is None:
            raise RuntimeError("CycleSampler failed to initialize order.")
        if self._offset >= int(self._order.numel()):
            self._reset_order()
        if self._order is None:
            raise RuntimeError("CycleSampler failed to refresh order.")
        return self._order

    def _reset_order(self) -> None:
        self._offset = _ZERO
        if not self._shuffle:
            self._order = torch.arange(self._n, dtype=torch.long)
            return
        if self._generator is None:
            self._order = torch.randperm(self._n, dtype=torch.long)
            return
        self._order = torch.randperm(self._n, generator=self._generator, dtype=torch.long)

