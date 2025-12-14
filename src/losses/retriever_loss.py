from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.retriever import RetrieverOutput

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import DictConfig


@dataclass
class LossOutput:
    loss: torch.Tensor
    components: Dict[str, float]
    metrics: Dict[str, float]


class RetrieverBCELoss(nn.Module):
    """Edge-wise BCE-with-logits with optional graph-normalized reduction."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        pos_weight_value = float(pos_weight)
        self._use_pos_weight = pos_weight_value != 1.0
        self.register_buffer("pos_weight", torch.tensor(pos_weight_value, dtype=torch.float32))

    def forward(
        self,
        output: RetrieverOutput,
        targets: torch.Tensor,
        training_step: int = 0,
        *,
        edge_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> LossOutput:
        logits = output.logits
        if logits is None:
            raise ValueError("RetrieverBCELoss requires output.logits (scores-only outputs are unsupported).")

        logits = logits.view(-1)
        targets = targets.view(-1).float()
        if logits.numel() != targets.numel():
            raise ValueError(f"logits/targets shape mismatch: {logits.shape} vs {targets.shape}")

        pos_weight = self.pos_weight.to(dtype=logits.dtype) if self._use_pos_weight else None
        edge_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )

        if edge_batch is not None:
            edge_batch = edge_batch.view(-1).to(dtype=torch.long)
            if edge_batch.numel() != edge_loss.numel():
                raise ValueError(f"edge_batch shape mismatch: {edge_batch.shape} vs logits {logits.shape}")
            if num_graphs is None:
                num_graphs = int(edge_batch.max().item()) + 1
            if num_graphs <= 0:
                raise ValueError(f"num_graphs must be positive, got {num_graphs}")

            per_graph_sum = edge_loss.new_zeros(num_graphs)
            per_graph_sum.scatter_add_(0, edge_batch, edge_loss)
            per_graph_cnt = edge_loss.new_zeros(num_graphs)
            per_graph_cnt.scatter_add_(0, edge_batch, torch.ones_like(edge_loss))
            loss = (per_graph_sum / per_graph_cnt.clamp_min(1.0)).mean()
        else:
            loss = edge_loss.mean()

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_mask = targets > 0.5
            pos_avg = probs[pos_mask].mean() if bool(pos_mask.any().item()) else probs.new_tensor(0.0)
            neg_avg = probs[~pos_mask].mean() if bool((~pos_mask).any().item()) else probs.new_tensor(0.0)

        return LossOutput(
            loss=loss,
            components={},
            metrics={
                "pos_prob": float(pos_avg.item()),
                "neg_prob": float(neg_avg.item()),
                "separation": float((pos_avg - neg_avg).item()),
            },
        )


def create_loss_function(cfg: Mapping[str, Any] | "DictConfig") -> RetrieverBCELoss:
    try:
        from omegaconf import DictConfig, OmegaConf  # type: ignore

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ModuleNotFoundError:  # pragma: no cover
        pass

    pos_weight = float(cfg.get("pos_weight", 1.0))
    return RetrieverBCELoss(pos_weight=pos_weight)

