from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.models.outputs import DeterministicOutput, ProbabilisticOutput

from .base import BaseLoss
from .evidential import Type2Loss
from .output import LossOutput
from .registry import register_loss

try:  # Optional dependency for InfoNCE
    from .deterministic import DeterministicInfoNCELoss
except Exception:  # pragma: no cover
    DeterministicInfoNCELoss = None  # type: ignore[assignment]


@register_loss(
    "hybrid_vera",
    description="Hybrid loss combining InfoNCE ranking with evidential calibration",
    family="evidential",
)
class HybridVERALoss(BaseLoss):
    """Implements ranking + candidate evidential + query-level VERA losses with consistency regularization."""

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        hard_negative_mining: bool = True,
        hard_negative_ratio: float = 0.5,
        rank_weight: float = 1.0,
        cand_weight: float = 0.2,
        query_weight: float = 0.5,
        consistency_weight: float = 0.1,
        alpha: float = 0.7,
        beta: float = 1.0,
        top_k: int = 1,
        clamp_min: float = 1e-6,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if DeterministicInfoNCELoss is None:
            raise RuntimeError("hybrid_vera loss requires torch_scatter for InfoNCE.")
        self.rank_component = DeterministicInfoNCELoss(
            temperature=temperature,
            hard_negative_mining=hard_negative_mining,
            hard_negative_ratio=hard_negative_ratio,
        )
        self.type2_component = Type2Loss(clamp_min=clamp_min)
        self.rank_weight = float(rank_weight)
        self.cand_weight = float(cand_weight)
        self.query_weight = float(query_weight)
        self.consistency_weight = float(consistency_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.top_k = max(1, int(top_k))
        self._eps = float(max(eps, 1e-9))

    def __call__(self, model_output: ProbabilisticOutput, targets: torch.Tensor, **kwargs) -> LossOutput:
        return self.forward(model_output, targets, **kwargs)

    def forward(self, model_output: ProbabilisticOutput, targets: torch.Tensor, **kwargs) -> LossOutput:
        if DeterministicInfoNCELoss is None:
            raise RuntimeError("hybrid_vera loss requires torch_scatter for InfoNCE.")
        model_output.ensure_moments(self._eps)
        scores = model_output.scores
        if scores is None or model_output.logits is None:
            raise ValueError("HybridVERALoss requires retriever to provide ranking scores and logits.")
        query_ids = kwargs.get("query_ids", None)
        if query_ids is None:
            query_ids = model_output.query_ids
        components: Dict[str, float] = {}
        metrics: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=targets.device)

        if self.rank_weight > 0:
            det_output = DeterministicOutput(scores=scores, logits=model_output.logits, query_ids=query_ids)
            rank_out = self.rank_component(det_output, targets, query_ids=query_ids)
            total_loss = total_loss + self.rank_weight * rank_out.loss
            components["rank"] = float(rank_out.loss.detach())
            metrics.update(rank_out.metrics)

        if self.cand_weight > 0:
            type2_out = self.type2_component(model_output, targets, **kwargs)
            total_loss = total_loss + self.cand_weight * type2_out.loss
            components["candidate"] = float(type2_out.loss.detach())

        if self.query_weight > 0:
            query_loss, query_stats = self._query_level_loss(
                model_output, targets, query_ids
            )
            total_loss = total_loss + self.query_weight * query_loss
            components["query"] = float(query_loss.detach())
            metrics.update(query_stats)

        if self.consistency_weight > 0 and model_output.posterior_mean is not None:
            mu = model_output.posterior_mean
            cons_loss = F.mse_loss(scores, mu)
            total_loss = total_loss + self.consistency_weight * cons_loss
            components["consistency"] = float(cons_loss.detach())

        return LossOutput(loss=total_loss, components=components, metrics=metrics)

    def _query_level_loss(
        self,
        model_output: ProbabilisticOutput,
        targets: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits = model_output.logits
        mu = model_output.posterior_mean
        if logits is None or mu is None:
            raise ValueError("Query-level VERA requires logits and posterior mean.")
        if query_ids is None:
            raise ValueError("Query IDs are required for query-level aggregation.")
        if logits.numel() == 0:
            zero = torch.tensor(0.0, device=targets.device)
            return zero, {}

        num_queries = int(query_ids.max().item()) + 1
        coverages = []
        targets_bin = []
        for q in range(num_queries):
            mask = query_ids == q
            if not torch.any(mask):
                continue
            q_logits = logits[mask]
            q_mu = mu[mask]
            q_targets = targets[mask]
            k = min(self.top_k, q_logits.numel())
            if k <= 0:
                continue
            values, idx = torch.topk(q_logits, k=k, largest=True)
            mu_topk = q_mu[idx]
            tgt_topk = q_targets[idx]
            z = 1.0 if torch.any(tgt_topk > 0.5) else 0.0
            c = 1.0 - torch.prod(1.0 - mu_topk)
            coverages.append(torch.tensor(c, device=logits.device))
            targets_bin.append(torch.tensor(z, device=logits.device))

        if not coverages:
            zero = torch.tensor(0.0, device=targets.device)
            return zero, {}

        q_tensor = torch.stack(coverages)
        z_tensor = torch.stack(targets_bin)

        eps = self._eps
        p = torch.clamp(z_tensor * (1.0 - 2.0 * eps) + eps, eps, 1.0 - eps)
        q = torch.clamp(q_tensor, eps, 1.0 - eps)
        kl_pq = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
        kl_qp = q * torch.log(q / p) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - p))
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        h_norm = entropy / math.log(2.0)
        loss = self.alpha * kl_pq + (1.0 - self.alpha) * kl_qp + self.beta * (1.0 - h_norm) * kl_pq
        metrics = {
            "query/kl_pq": float(kl_pq.mean().detach()),
            "query/kl_qp": float(kl_qp.mean().detach()),
            "query/h_norm": float(h_norm.mean().detach()),
        }
        return loss.mean(), metrics


__all__ = ["HybridVERALoss"]
