from __future__ import annotations

import torch
from torch import nn


def resolve_module_float_dtype(module: nn.Module) -> torch.dtype | None:
    for tensor in module.parameters(recurse=False):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    for tensor in module.buffers(recurse=False):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return None


def align_float_input_dtype(tensor: torch.Tensor, *, module: nn.Module) -> torch.Tensor:
    if not torch.is_floating_point(tensor):
        return tensor
    target_dtype = resolve_module_float_dtype(module)
    if target_dtype is None or tensor.dtype == target_dtype:
        return tensor
    if torch.is_autocast_enabled():
        return tensor
    return tensor.to(dtype=target_dtype)


__all__ = ["align_float_input_dtype", "resolve_module_float_dtype"]
