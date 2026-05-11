from __future__ import annotations

from torch import nn


class TruncatedExactBFMLoss(nn.Module):
    # REMOVED: backup-derived TE-BFM objective/logits — see methodology.md §3.6
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super().__init__()
        raise RuntimeError(
            "TruncatedExactBFMLoss was removed; use BudgetedDAGDetailedBalanceLoss."
        )


__all__ = ["TruncatedExactBFMLoss"]
