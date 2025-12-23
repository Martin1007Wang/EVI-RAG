from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.retriever import RetrieverOutput


@dataclass
class LossOutput:
    loss: torch.Tensor
    components: Dict[str, float]
    metrics: Dict[str, float]


def _apply_edge_mask(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    edge_batch: Optional[torch.Tensor],
    edge_mask: Optional[torch.Tensor],
    name: str,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, float]]:
    metrics: Dict[str, float] = {}
    if edge_mask is None:
        return logits, targets, edge_batch, metrics

    mask = edge_mask.view(-1).to(dtype=torch.bool, device=logits.device)
    if mask.numel() != logits.numel():
        raise ValueError(f"{name} edge_mask shape mismatch: {mask.shape} vs logits {logits.shape}")
    if not bool(mask.any().item()):
        raise ValueError(f"{name} edge_mask has no True entries.")

    metrics[f"{name}_mask_frac"] = float(mask.float().mean().item())
    pos_mask_all = targets > 0.5
    if bool(pos_mask_all.any().item()):
        pos_in_mask = (mask & pos_mask_all).float().sum().item()
        pos_total = pos_mask_all.float().sum().item()
        metrics[f"{name}_mask_pos_frac"] = float(pos_in_mask / pos_total)

    logits = logits[mask]
    targets = targets[mask]
    if edge_batch is not None:
        edge_batch = edge_batch[mask]
    return logits, targets, edge_batch, metrics


class RetrieverReachabilityLoss(nn.Module):
    """Reachability-aware loss with soft labels.

    Uses a soft listwise objective plus optional BCE-on-soft-targets for calibration.
    """

    def __init__(
        self,
        *,
        temperature: float,
        listwise_weight: float,
        bce_weight: float,
        soft_label_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if not (self.temperature > 0.0):
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        self.listwise_weight = float(listwise_weight)
        self.bce_weight = float(bce_weight)
        self.soft_label_power = float(soft_label_power)
        if self.soft_label_power <= 0.0:
            raise ValueError(f"soft_label_power must be > 0, got {self.soft_label_power}")

    def forward(
        self,
        output: RetrieverOutput,
        targets: torch.Tensor,
        training_step: int = 0,
        *,
        edge_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        del training_step
        logits = output.logits
        if logits is None:
            raise ValueError("RetrieverReachabilityLoss requires output.logits.")
        if edge_batch is None:
            raise ValueError("RetrieverReachabilityLoss requires edge_batch to define per-graph groups.")

        logits = logits.view(-1)
        targets = targets.view(-1).float()
        edge_batch = edge_batch.view(-1).to(dtype=torch.long)
        if logits.numel() != targets.numel() or logits.numel() != edge_batch.numel():
            raise ValueError(
                f"logits/targets/edge_batch shape mismatch: {logits.shape} vs {targets.shape} vs {edge_batch.shape}"
            )

        logits, targets, edge_batch, mask_metrics = _apply_edge_mask(
            logits=logits,
            targets=targets,
            edge_batch=edge_batch,
            edge_mask=edge_mask,
            name="edge",
        )

        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 0
        num_graphs = int(num_graphs)
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")

        weights = targets.clamp(min=0.0, max=1.0)
        if self.soft_label_power != 1.0:
            weights = weights.pow(self.soft_label_power)

        scaled = logits / float(self.temperature)
        max_per = scaled.new_full((num_graphs,), -torch.inf)
        max_per.scatter_reduce_(0, edge_batch, scaled, reduce="amax", include_self=True)
        max_b = max_per[edge_batch]

        exp_scaled = torch.exp(scaled - max_b)
        denom_sum = exp_scaled.new_zeros(num_graphs)
        denom_sum.scatter_add_(0, edge_batch, exp_scaled)
        weighted = exp_scaled * weights
        numerator_sum = weighted.new_zeros(num_graphs)
        numerator_sum.scatter_add_(0, edge_batch, weighted)

        tiny = torch.finfo(exp_scaled.dtype).tiny
        log_denom = max_per + torch.log(denom_sum.clamp_min(tiny))
        log_num = max_per + torch.log(numerator_sum.clamp_min(tiny))

        has_pos = numerator_sum > 0
        listwise_per = -(log_num - log_denom)
        listwise_loss = listwise_per[has_pos].mean() if bool(has_pos.any().item()) else logits.new_zeros(())

        bce_loss = logits.new_zeros(())
        if self.bce_weight != 0.0:
            edge_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            per_graph_sum = edge_loss.new_zeros(num_graphs)
            per_graph_sum.scatter_add_(0, edge_batch, edge_loss)
            per_graph_cnt = edge_loss.new_zeros(num_graphs)
            per_graph_cnt.scatter_add_(0, edge_batch, torch.ones_like(edge_loss))
            bce_loss = (per_graph_sum / per_graph_cnt.clamp_min(1.0)).mean()

        total = self.listwise_weight * listwise_loss + self.bce_weight * bce_loss

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_mask = targets > 0.5
            pos_avg = probs[pos_mask].mean() if bool(pos_mask.any().item()) else probs.new_tensor(0.0)
            neg_avg = probs[~pos_mask].mean() if bool((~pos_mask).any().item()) else probs.new_tensor(0.0)

        return LossOutput(
            loss=total,
            components={
                "listwise": float(listwise_loss.detach().cpu().item()),
                "bce": float(bce_loss.detach().cpu().item()),
            },
            metrics={
                "pos_prob": float(pos_avg.item()),
                "neg_prob": float(neg_avg.item()),
                "separation": float((pos_avg - neg_avg).item()),
                "num_pos_graphs": float(has_pos.to(torch.float32).sum().item()),
                **mask_metrics,
            },
        )
