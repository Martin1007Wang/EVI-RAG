from __future__ import annotations

from torch import nn


class SubTrajectoryBalanceLoss(nn.Module):
    # REMOVED: trajectory-level SubTB objective — see methodology.md §3.9
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__()
        raise RuntimeError(
            "SubTrajectoryBalanceLoss was removed; use BudgetedDAGDetailedBalanceLoss."
        )


__all__ = ["SubTrajectoryBalanceLoss"]
