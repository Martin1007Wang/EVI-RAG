from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]
    num_states: int
    per_unit_loss: torch.Tensor | None = None

    @staticmethod
    def aggregate(outputs: Sequence[ObjectiveOutput]) -> ObjectiveOutput:
        loss = torch.stack([x.loss for x in outputs]).mean()
        keys = set(outputs[0].metrics)
        for x in outputs[1:]:
            keys &= set(x.metrics)

        per_unit = [x.per_unit_loss.detach().flatten() for x in outputs if x.per_unit_loss is not None]

        return ObjectiveOutput(
            loss=loss,
            metrics={k: torch.stack([x.metrics[k] for x in outputs]).mean() for k in sorted(keys)},
            num_states=sum(x.num_states for x in outputs),
            per_unit_loss=torch.cat(per_unit) if per_unit else None,
        )


__all__ = [
    "ObjectiveOutput",
]
