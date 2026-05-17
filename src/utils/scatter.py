from __future__ import annotations

import torch

try:
    from torch_scatter import scatter_sum as _torch_scatter_sum
except ImportError:  # pragma: no cover - exercised in environments without torch_scatter
    _torch_scatter_sum = None


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    *,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    if _torch_scatter_sum is not None:
        return _torch_scatter_sum(src, index, dim=dim, dim_size=dim_size)

    if dim != 0:
        raise NotImplementedError(
            "Fallback scatter_sum only supports dim=0 without torch_scatter."
        )

    index = index.to(device=src.device, dtype=torch.long)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    out_shape = list(src.shape)
    out_shape[dim] = int(dim_size)
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    if src.numel() > 0:
        out.index_add_(dim, index, src)
    return out


__all__ = ["scatter_sum"]
