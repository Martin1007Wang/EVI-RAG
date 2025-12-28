from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torchmetrics import Metric


class FeatureMonitor(Metric):
    """Track score separation and feature norm stability."""

    full_state_update: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("pos_score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pos_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neg_score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("neg_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("feat_norm_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("feat_count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        preds = preds.detach().sigmoid()
        target = target.detach()

        pos_mask = target > 0.5
        neg_mask = ~pos_mask

        if bool(pos_mask.any().item()):
            self.pos_score_sum += preds[pos_mask].sum()
            self.pos_count += pos_mask.sum()
        if bool(neg_mask.any().item()):
            self.neg_score_sum += preds[neg_mask].sum()
            self.neg_count += neg_mask.sum()

        if features is not None:
            norms = torch.norm(features.detach(), p=2, dim=-1)
            if norms.numel() > 0:
                self.feat_norm_sum += norms.sum()
                self.feat_count += norms.numel()

    def compute(self) -> Dict[str, torch.Tensor]:
        pos_avg = self.pos_score_sum / self.pos_count.clamp(min=1.0)
        neg_avg = self.neg_score_sum / self.neg_count.clamp(min=1.0)
        norm_avg = self.feat_norm_sum / self.feat_count.clamp(min=1.0)

        return {
            "features/pos_prob_avg": pos_avg,
            "features/neg_prob_avg": neg_avg,
            "features/separation_gap": pos_avg - neg_avg,
            "features/norm_avg": norm_avg,
        }
