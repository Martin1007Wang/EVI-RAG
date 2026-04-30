from __future__ import annotations

import torch
from torch_scatter import scatter_logsumexp


def has_segment(
    *,
    batch_index: torch.Tensor,
    num_segments: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_index.numel() == 0:
        return torch.zeros(num_segments, dtype=torch.bool, device=device)

    return torch.bincount(
        batch_index.to(device=device, dtype=torch.long),
        minlength=num_segments,
    ).gt(0)


def segment_logsumexp(
    *,
    values: torch.Tensor,
    batch_index: torch.Tensor,
    num_segments: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    empty = torch.finfo(dtype).min

    if values.numel() == 0:
        return torch.full((num_segments,), empty, dtype=dtype, device=device)

    batch_index = batch_index.to(device=device, dtype=torch.long)

    out = scatter_logsumexp(
        values.to(device=device, dtype=dtype),
        batch_index,
        dim=0,
        dim_size=num_segments,
    )

    count = torch.bincount(batch_index, minlength=num_segments)
    return out.masked_fill(count == 0, empty)


__all__ = [
    "has_segment",
    "segment_logsumexp",
]