from __future__ import annotations

import pytest
import sys
import types
import torch

if "torch_scatter" not in sys.modules:
    torch_scatter_stub = types.ModuleType("torch_scatter")

    def _scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del dim
        size = int(dim_size or (int(index.max().item()) + 1 if index.numel() else 0))
        return torch.full((size,), -torch.inf), torch.full((size,), -1, dtype=torch.long)

    def _scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(dim_size or (int(index.max().item()) + 1 if index.numel() else 0))
        out = torch.zeros((size,) + tuple(src.shape[1:]), dtype=src.dtype, device=src.device)
        for row, dest in enumerate(index.tolist()):
            out[dest] += src[row]
        return out

    def _scatter_logsumexp(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = 0,
        dim_size: int | None = None,
    ) -> torch.Tensor:
        if dim != 0:
            raise NotImplementedError("test stub only supports dim=0")
        size = int(dim_size or (int(index.max().item()) + 1 if index.numel() else 0))
        out = torch.full((size,), -torch.inf, dtype=src.dtype, device=src.device)
        for dest in range(size):
            mask = index == dest
            if bool(mask.any()):
                out[dest] = torch.logsumexp(src[mask], dim=0)
        return out

    torch_scatter_stub.scatter_max = _scatter_max
    torch_scatter_stub.scatter_sum = _scatter_sum
    torch_scatter_stub.scatter_logsumexp = _scatter_logsumexp
    sys.modules["torch_scatter"] = torch_scatter_stub

from src.weaver.module import build_loss


def test_subtb_objective_is_removed_from_active_factory() -> None:
    with pytest.raises(ValueError, match="budgeted_flow_distill"):
        build_loss({"type": "subtb"})


def test_te_bfm_objective_is_removed_from_active_factory() -> None:
    with pytest.raises(ValueError, match="budgeted_flow_distill"):
        build_loss({"type": "te_bfm"})
