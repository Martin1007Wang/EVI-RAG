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


class RetrieverListwiseHardNegLoss(nn.Module):
    """Per-graph listwise softmax loss with optional hard-negative pairwise term.

    This loss aligns training with top-k ranking objectives by optimizing, per graph:

    - Listwise: -log(sum_{pos} exp(z/tau) / sum_{all} exp(z/tau))
    - Pairwise (hard negatives): E_{p in P, n in topK(N)} softplus(margin + z_n - z_p)
    """

    def __init__(
        self,
        *,
        temperature: float,
        listwise_weight: float,
        hard_neg_k: int,
        pairwise_margin: float,
        pairwise_weight: float,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        if not (self.temperature > 0.0):
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        self.listwise_weight = float(listwise_weight)
        self.hard_neg_k = int(max(0, hard_neg_k))
        self.pairwise_margin = float(pairwise_margin)
        self.pairwise_weight = float(pairwise_weight)

    @staticmethod
    def _segment_logsumexp(
        values: torch.Tensor,
        index: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        """Stable segment-wise logsumexp over 1D values grouped by `index`."""
        if values.numel() == 0:
            return values.new_full((num_segments,), -torch.inf)
        if values.dim() != 1 or index.dim() != 1:
            raise ValueError(f"segment_logsumexp expects 1D tensors, got {values.shape=} {index.shape=}")
        if values.numel() != index.numel():
            raise ValueError(f"segment_logsumexp shape mismatch: {values.shape} vs {index.shape}")

        num_segments = int(num_segments)
        if num_segments <= 0:
            raise ValueError(f"num_segments must be positive, got {num_segments}")

        max_per = values.new_full((num_segments,), -torch.inf)
        max_per.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        max_b = max_per[index]

        exp_sum = values.new_zeros(num_segments)
        # Under autocast, torch.exp may promote to float32; force dtype alignment before scatter.
        exp_term = torch.exp(values - max_b).to(dtype=exp_sum.dtype)
        exp_sum.scatter_add_(0, index, exp_term)

        tiny = torch.finfo(values.dtype).tiny
        return max_per + torch.log(exp_sum.clamp_min(tiny))

    def forward(
        self,
        output: RetrieverOutput,
        targets: torch.Tensor,
        training_step: int = 0,
        *,
        edge_batch: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None,
    ) -> LossOutput:
        del training_step
        logits = output.logits
        if logits is None:
            raise ValueError("RetrieverListwiseHardNegLoss requires output.logits.")
        if edge_batch is None:
            raise ValueError("RetrieverListwiseHardNegLoss requires edge_batch to define per-graph groups.")

        logits = logits.view(-1)
        targets = targets.view(-1).float()
        edge_batch = edge_batch.view(-1).to(dtype=torch.long)
        if logits.numel() != targets.numel() or logits.numel() != edge_batch.numel():
            raise ValueError(
                f"logits/targets/edge_batch shape mismatch: {logits.shape} vs {targets.shape} vs {edge_batch.shape}"
            )
        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 0
        num_graphs = int(num_graphs)
        if num_graphs <= 0:
            raise ValueError(f"num_graphs must be positive, got {num_graphs}")

        # --- Listwise term (vectorized; graphs without positives contribute 0) ---
        scaled = logits / float(self.temperature)
        log_denom = self._segment_logsumexp(scaled, edge_batch, num_segments=num_graphs)
        edge_cnt = torch.bincount(edge_batch, minlength=num_graphs).to(dtype=torch.long)
        has_edge = edge_cnt > 0
        # Avoid -inf - -inf in empty-edge graphs.
        log_denom = torch.where(has_edge, log_denom, log_denom.new_zeros(log_denom.shape))

        pos_mask = targets > 0.5
        pos_batch = edge_batch[pos_mask]
        pos_scaled = scaled[pos_mask]
        log_num = self._segment_logsumexp(pos_scaled, pos_batch, num_segments=num_graphs)

        pos_cnt = torch.bincount(pos_batch, minlength=num_graphs).to(dtype=scaled.dtype)
        has_pos = pos_cnt > 0
        # Avoid inf/NaN from empty-positive graphs by defining their loss contribution as 0.
        log_num_safe = torch.where(has_pos, log_num, log_denom)
        listwise_per = -(log_num_safe - log_denom)
        listwise_sum = listwise_per.sum()
        listwise_den = has_pos.to(dtype=scaled.dtype).sum().clamp_min(1.0)
        listwise_loss = listwise_sum / listwise_den

        # --- Hard-negative pairwise term (per-graph top-k negatives) ---
        pairwise_loss_sum = logits.new_zeros(())
        pairwise_graphs = logits.new_zeros(())
        if self.hard_neg_k > 0 and float(self.pairwise_weight) != 0.0:
            # NOTE: avoid per-graph `.item()` GPU sync by slicing via contiguous edge ranges.
            # PyG Batch concatenates edges per graph, so edge_batch is expected to be non-decreasing.
            if edge_batch.numel() > 1 and not bool((edge_batch[1:] >= edge_batch[:-1]).all().item()):
                raise ValueError(
                    "edge_batch must be non-decreasing for hard-negative selection. "
                    "Ensure the batch is built via PyG Batch/DataLoader without shuffling edges."
                )

            edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
            edge_ptr = torch.cat([edge_counts.new_zeros(1), edge_counts.cumsum(0)], dim=0)
            edge_ptr_cpu = edge_ptr.detach().cpu()

            for gid in range(num_graphs):
                start = int(edge_ptr_cpu[gid].item())
                end = int(edge_ptr_cpu[gid + 1].item())
                if end <= start:
                    continue
                logits_g = logits[start:end]
                targets_g = targets[start:end]
                pos_logits = logits_g[targets_g > 0.5]
                neg_logits = logits_g[targets_g <= 0.5]
                if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                    continue
                k = min(self.hard_neg_k, int(neg_logits.numel()))
                hard_neg = torch.topk(neg_logits, k=k, largest=True, sorted=False).values
                pairwise = F.softplus(self.pairwise_margin + hard_neg.unsqueeze(0) - pos_logits.unsqueeze(1)).mean()
                pairwise_loss_sum = pairwise_loss_sum + pairwise
                pairwise_graphs = pairwise_graphs + 1.0
        pairwise_loss = pairwise_loss_sum / pairwise_graphs.clamp_min(1.0)

        total = self.listwise_weight * listwise_loss + self.pairwise_weight * pairwise_loss

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pos_avg = probs[pos_mask].mean() if bool(pos_mask.any().item()) else probs.new_tensor(0.0)
            neg_avg = probs[~pos_mask].mean() if bool((~pos_mask).any().item()) else probs.new_tensor(0.0)

        return LossOutput(
            loss=total,
            components={
                "listwise": float(listwise_loss.detach().cpu().item()),
                "pairwise": float(pairwise_loss.detach().cpu().item()),
            },
            metrics={
                "pos_prob": float(pos_avg.item()),
                "neg_prob": float(neg_avg.item()),
                "separation": float((pos_avg - neg_avg).item()),
                "num_pos_graphs": float(has_pos.to(torch.float32).sum().item()),
                "num_pairwise_graphs": float(pairwise_graphs.detach().cpu().item()),
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
