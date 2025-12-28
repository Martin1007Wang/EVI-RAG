from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.retriever import RetrieverOutput

_NO_PATH_WEIGHT = 0.0
_NO_PATH_WARMUP = 0
_DEFAULT_INFONCE_WEIGHT = 1.0
_DEFAULT_BCE_WEIGHT = 0.0
_LABEL_POSITIVE_THRESHOLD = 0.5

@dataclass
class LossOutput:
    loss: torch.Tensor
    components: Dict[str, float]
    metrics: Dict[str, float]


class RetrieverLoss(nn.Module):
    """Composite loss: multi-positive InfoNCE (triple-only supervision)."""

    def __init__(
        self,
        *,
        path_weight: float = _NO_PATH_WEIGHT,
        path_warmup_steps: int = _NO_PATH_WARMUP,
        infonce_temperature: float = 1.0,
        infonce_weight: float = _DEFAULT_INFONCE_WEIGHT,
        bce_weight: float = _DEFAULT_BCE_WEIGHT,
    ) -> None:
        super().__init__()
        self.path_weight = float(path_weight)
        if self.path_weight != _NO_PATH_WEIGHT:
            raise ValueError(
                "RetrieverLoss forbids path supervision; set path_weight=0 and keep path_edge_indices unset."
            )
        self.path_warmup_steps = int(path_warmup_steps)
        if self.path_warmup_steps != _NO_PATH_WARMUP:
            raise ValueError("RetrieverLoss forbids path warmup; set path_warmup_steps=0.")
        self.infonce_temperature = float(infonce_temperature)
        if self.infonce_temperature <= 0.0:
            raise ValueError(f"infonce_temperature must be positive, got {self.infonce_temperature}")
        self.infonce_weight = float(infonce_weight)
        self.bce_weight = float(bce_weight)
        if self.infonce_weight < 0.0 or self.bce_weight < 0.0:
            raise ValueError("infonce_weight and bce_weight must be non-negative.")
        if self.infonce_weight == 0.0 and self.bce_weight == 0.0:
            raise ValueError("RetrieverLoss requires at least one non-zero loss weight.")

    def _infonce_loss(
        self,
        *,
        targets: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pos_mask = targets > _LABEL_POSITIVE_THRESHOLD
        neg_mask = ~pos_mask
        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())
        if pos_count == 0 or neg_count == 0:
            return logits.new_zeros(()), {
                "infonce_pos_edges": float(pos_count),
                "infonce_neg_edges": float(neg_count),
                "infonce_graphs": 0.0,
            }

        scores = logits / self.infonce_temperature
        dtype = scores.dtype
        device = scores.device
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)

        max_all = torch.full((num_graphs,), neg_inf, device=device, dtype=dtype)
        max_all.scatter_reduce_(0, edge_batch, scores, reduce="amax", include_self=True)

        scores_pos = scores.masked_fill(~pos_mask, neg_inf)
        max_pos = torch.full((num_graphs,), neg_inf, device=device, dtype=dtype)
        max_pos.scatter_reduce_(0, edge_batch, scores_pos, reduce="amax", include_self=True)

        exp_all = torch.exp(scores - max_all[edge_batch])
        sum_all = torch.zeros(num_graphs, device=device, dtype=dtype)
        sum_all.scatter_add_(0, edge_batch, exp_all)

        exp_pos = torch.zeros_like(scores)
        pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
        if pos_idx.numel() > 0:
            exp_pos[pos_idx] = torch.exp(scores[pos_idx] - max_pos[edge_batch[pos_idx]])
        sum_pos = torch.zeros(num_graphs, device=device, dtype=dtype)
        sum_pos.scatter_add_(0, edge_batch, exp_pos)

        logsumexp_all = max_all + torch.log(sum_all.clamp_min(1e-12))
        logsumexp_pos = max_pos + torch.log(sum_pos.clamp_min(1e-12))

        pos_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
        pos_counts.scatter_add_(0, edge_batch, pos_mask.to(dtype))
        edge_counts = torch.zeros(num_graphs, device=device, dtype=dtype)
        edge_counts.scatter_add_(0, edge_batch, torch.ones_like(edge_batch, dtype=dtype))
        neg_counts = edge_counts - pos_counts
        valid = (pos_counts > 0) & (neg_counts > 0)
        if not bool(valid.any().item()):
            return logits.new_zeros(()), {
                "infonce_pos_edges": float(pos_count),
                "infonce_neg_edges": float(neg_count),
                "infonce_graphs": 0.0,
                "infonce_graphs_no_pos": float((pos_counts == 0).sum().item()),
                "infonce_graphs_no_neg": float((neg_counts == 0).sum().item()),
            }

        loss = (logsumexp_all - logsumexp_pos)[valid].mean()
        return loss, {
            "infonce_pos_edges": float(pos_count),
            "infonce_neg_edges": float(neg_count),
            "infonce_graphs": float(valid.sum().item()),
            "infonce_graphs_no_pos": float((pos_counts == 0).sum().item()),
            "infonce_graphs_no_neg": float((neg_counts == 0).sum().item()),
        }

    def _bce_loss(
        self,
        *,
        targets: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        per_edge = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss_sum = torch.zeros(num_graphs, device=logits.device, dtype=per_edge.dtype)
        loss_sum.scatter_add_(0, edge_batch, per_edge)
        edge_counts = torch.zeros(num_graphs, device=logits.device, dtype=per_edge.dtype)
        edge_counts.scatter_add_(0, edge_batch, torch.ones_like(per_edge, dtype=per_edge.dtype))
        valid = edge_counts > 0
        if not bool(valid.any().item()):
            return logits.new_zeros(()), {
                "bce_graphs": 0.0,
                "bce_edges": float(per_edge.numel()),
            }
        loss = (loss_sum[valid] / edge_counts[valid]).mean()
        return loss, {
            "bce_graphs": float(valid.sum().item()),
            "bce_edges": float(per_edge.numel()),
        }

    def _prepare_loss_inputs(
        self,
        *,
        output: RetrieverOutput,
        targets: torch.Tensor,
        edge_batch: Optional[torch.Tensor],
        num_graphs: Optional[int],
        path_edge_indices: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        logits = output.logits
        if logits is None:
            raise ValueError("RetrieverLoss requires output.logits.")
        if edge_batch is None:
            raise ValueError("RetrieverLoss requires edge_batch to define per-graph groups.")
        if path_edge_indices is not None:
            raise ValueError("RetrieverLoss forbids path_edge_indices; retriever is triple-only.")

        logits = logits.view(-1)
        targets = targets.view(-1).float()
        edge_batch = edge_batch.view(-1).to(dtype=torch.long)
        if logits.numel() == 0:
            raise ValueError("RetrieverLoss received empty logits/targets; check dataset filtering.")
        if logits.numel() != targets.numel() or logits.numel() != edge_batch.numel():
            raise ValueError(
                f"logits/targets/edge_batch shape mismatch: {logits.shape} vs {targets.shape} vs {edge_batch.shape}"
            )

        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 0
        num_graphs = int(num_graphs)
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")
        return logits, targets, edge_batch, num_graphs

    @staticmethod
    def _compute_separation_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_mask = targets > _LABEL_POSITIVE_THRESHOLD
            pos_avg = probs[pos_mask].mean() if bool(pos_mask.any().item()) else probs.new_tensor(0.0)
            neg_avg = probs[~pos_mask].mean() if bool((~pos_mask).any().item()) else probs.new_tensor(0.0)
        return {
            "pos_prob": float(pos_avg.item()),
            "neg_prob": float(neg_avg.item()),
            "separation": float((pos_avg - neg_avg).item()),
        }

    def forward(
        self,
        output: RetrieverOutput,
        targets: torch.Tensor,
        training_step: int = 0,
        *,
        edge_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
        path_edge_indices: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        logits, targets, edge_batch, num_graphs = self._prepare_loss_inputs(
            output=output,
            targets=targets,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            path_edge_indices=path_edge_indices,
        )

        infonce_loss, infonce_metrics = self._infonce_loss(
            targets=targets,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
            logits=logits,
        )
        bce_loss = logits.new_zeros(())
        bce_metrics = {"bce_graphs": 0.0, "bce_edges": 0.0}
        if self.bce_weight > 0.0:
            bce_loss, bce_metrics = self._bce_loss(
                targets=targets,
                edge_batch=edge_batch,
                num_graphs=num_graphs,
                logits=logits,
            )
        path_loss = logits.new_zeros(())
        path_weight = _NO_PATH_WEIGHT
        total = self.infonce_weight * infonce_loss + self.bce_weight * bce_loss
        separation_metrics = self._compute_separation_metrics(logits, targets)

        return LossOutput(
            loss=total,
            components={
                "infonce": float(infonce_loss.detach().cpu().item()),
                "infonce_weight": float(self.infonce_weight),
                "bce": float(bce_loss.detach().cpu().item()),
                "bce_weight": float(self.bce_weight),
                "path": float(path_loss.detach().cpu().item()),
                "path_weight": float(path_weight),
            },
            metrics={
                **separation_metrics,
                **infonce_metrics,
                **bce_metrics,
                "path_graphs": 0.0,
            },
        )
