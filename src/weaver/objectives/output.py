from __future__ import annotations

from dataclasses import dataclass

import torch

from src.utils.scalars import Scalar, detach_scalar, require_scalar_tensor, validate_scalar


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, Scalar]
    num_states: int
    per_unit_loss: torch.Tensor | None = None

    def __post_init__(self) -> None:
        require_scalar_tensor(self.loss, name="ObjectiveOutput.loss")
        if int(self.num_states) < 0:
            raise ValueError("ObjectiveOutput.num_states must be nonnegative.")
        for key, value in self.metrics.items():
            validate_scalar(value, name=f"ObjectiveOutput.metrics[{key!r}]")
        if self.per_unit_loss is not None and self.per_unit_loss.ndim == 0:
            raise ValueError("ObjectiveOutput.per_unit_loss must be non-scalar or None.")

    def require_loss(self) -> torch.Tensor:
        """Return the live loss Tensor for .backward()."""
        return self.loss

    def detached_loss(self) -> float:
        """Return loss as a Python float, detached from the graph."""
        return float(self.loss.detach())

    def detached_metrics(self) -> dict[str, Scalar]:
        return {key: detach_scalar(value, name=f"ObjectiveOutput.metrics[{key!r}]") for key, value in self.metrics.items()}
