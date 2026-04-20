from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def init_xavier(layer: nn.Linear) -> None:
    """Xavier uniform initialization for a linear layer."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def init_negative_identity(layer: nn.Linear) -> None:
    """Initialize a square linear layer as −I.

    Raises
    ------
    ValueError
        If the weight matrix is not square.
    """
    if layer.weight.shape[0] != layer.weight.shape[1]:
        raise ValueError(
            "Inverse relation projection must be square, got "
            f"{tuple(layer.weight.shape)}."
        )
    with torch.no_grad():
        layer.weight.copy_(
            -torch.eye(
                layer.weight.shape[0],
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )
        )
        if layer.bias is not None:
            layer.bias.zero_()


def zero_last_linear(module: nn.Sequential | nn.Linear) -> None:
    """Zero-initialize the last ``nn.Linear`` in *module*.

    Raises
    ------
    TypeError
        If no ``nn.Linear`` is found in *module*.
    """
    target = (
        module
        if isinstance(module, nn.Linear)
        else next(
            (layer for layer in reversed(module) if isinstance(layer, nn.Linear)),
            None,
        )
    )
    if target is None:
        raise TypeError("No nn.Linear found in module.")
    nn.init.zeros_(target.weight)
    if target.bias is not None:
        nn.init.zeros_(target.bias)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    """Build a fully-connected MLP with GELU activations.

    Raises
    ------
    ValueError
        If ``num_layers < 1``.
    """
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def require_finite(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Assert all values in *tensor* are finite and return it unchanged.

    Raises
    ------
    ValueError
        If any value is NaN or ±inf.
    """
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def cosine_scores(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity, result in [-1, 1]."""
    normalized_left = F.normalize(
        require_finite(left, name="cosine_left"), dim=-1, eps=1e-8
    )
    normalized_right = F.normalize(
        require_finite(right, name="cosine_right"), dim=-1, eps=1e-8
    )
    return (normalized_left * normalized_right).sum(dim=-1)


def validate_bool_mask(
    mask: torch.Tensor,
    expected_len: int,
    name: str,
    ref_device: torch.device,
) -> torch.Tensor:
    """Validate shape, dtype, and device of a 1-D boolean mask.

    Parameters
    ----------
    mask         : tensor to validate
    expected_len : required number of elements
    name         : field name used in error messages
    ref_device   : device the mask must reside on

    Returns
    -------
    The input *mask* unchanged (for inline use).

    Raises
    ------
    TypeError  : if ``mask.dtype != torch.bool``
    ValueError : if shape or device does not match
    """
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be torch.bool.")
    if mask.dim() != 1 or mask.numel() != expected_len:
        raise ValueError(
            f"{name}: expected ({expected_len},), got {tuple(mask.shape)}."
        )
    if mask.device != ref_device:
        raise ValueError(
            f"{name} and reference tensor must be on the same device."
        )
    return mask


__all__ = [
    "build_mlp",
    "cosine_scores",
    "init_negative_identity",
    "init_xavier",
    "require_finite",
    "validate_bool_mask",
    "zero_last_linear",
]