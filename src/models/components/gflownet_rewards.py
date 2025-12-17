from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import torch
from torch import nn


def _safe_f1(precision: torch.Tensor, recall: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return 2 * precision * recall / (precision + recall + eps)


def _compute_path_metrics(
    *,
    selected_mask: torch.Tensor,
    path_mask: Optional[torch.Tensor],
    edge_batch: torch.Tensor,
    path_exists: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = selected_mask.device
    if path_mask is None or not path_mask.any():
        zeros = torch.zeros_like(path_exists, dtype=torch.float32, device=device)
        return zeros, zeros, zeros, zeros

    num_graphs = int(path_exists.numel())
    tp = torch.bincount(edge_batch, weights=(selected_mask & path_mask).float(), minlength=num_graphs)
    pred = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs).clamp(min=1.0)
    pos = torch.bincount(edge_batch, weights=path_mask.float(), minlength=num_graphs)

    precision = torch.where(pos > 0, tp / pred, torch.zeros_like(tp))
    recall = torch.where(pos > 0, tp / pos.clamp(min=1.0), torch.zeros_like(tp))
    f1 = _safe_f1(precision, recall)
    full_hit = (tp == pos) & (pos > 0)
    return precision, recall, f1, full_hit.float()


def _compute_pos_metrics(
    *,
    selected_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-graph precision/recall/f1 against positive edge labels."""
    device = selected_mask.device
    zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
    if positive_mask.numel() == 0 or num_graphs == 0:
        return zeros, zeros, zeros

    tp = torch.bincount(edge_batch, weights=(selected_mask & positive_mask).float(), minlength=num_graphs)
    pred = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs).clamp(min=1.0)
    pos = torch.bincount(edge_batch, weights=positive_mask.float(), minlength=num_graphs).clamp(min=1.0)

    precision = torch.where(pred > 0, tp / pred, zeros)
    recall = torch.where(pos > 0, tp / pos, zeros)
    f1 = _safe_f1(precision, recall)
    return precision, recall, f1


@dataclass
class RewardOutput:
    reward: torch.Tensor
    log_reward: torch.Tensor
    success: torch.Tensor
    answer_reach_frac: torch.Tensor
    pos_precision: torch.Tensor
    pos_recall: torch.Tensor
    pos_f1: torch.Tensor
    answer_precision: torch.Tensor
    answer_recall: torch.Tensor
    answer_f1: torch.Tensor
    path_exists: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


@dataclass
class _AnswerStats:
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    has_answers: torch.Tensor
    contains_answer: torch.Tensor
    hits: torch.Tensor
    totals: torch.Tensor


class AnswerOnlyReward(nn.Module):
    """二元结果奖励：命中答案 vs 未命中（无 shaping）。"""

    def __init__(
        self,
        *,
        success_reward: float,
        failure_reward: float,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        if self.success_reward <= 0.0 or self.failure_reward <= 0.0:
            raise ValueError(f"Rewards must be positive; got success_reward={self.success_reward}, failure_reward={self.failure_reward}.")

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        edge_index: torch.Tensor,  # [2, E_total] local node indices (collated)
        node_ptr: torch.Tensor,  # [B+1]
        answer_node_locals: torch.Tensor,  # [A_total] local node indices (collated)
        answer_node_ptr: torch.Tensor,  # [B+1]
        path_exists: Optional[torch.Tensor],
        reach_success: torch.Tensor,  # [B]
        reach_fraction: torch.Tensor,  # [B]
        **_: Any,
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(node_ptr.numel() - 1)
        stats = self._answer_metrics_vectorized(
            selected_mask=selected_mask,
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_ptr=node_ptr,
            answer_node_locals=answer_node_locals,
            answer_node_ptr=answer_node_ptr,
        )
        pos_prec, pos_rec, pos_f1 = _compute_pos_metrics(
            selected_mask=selected_mask.bool(),
            positive_mask=edge_labels > 0.5,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        # Keep reward self-contained: success/coverage are determined by (selected edges, answer nodes),
        # not by external flags that may drift across refactors.
        reach_fraction = stats.recall.clamp(min=0.0, max=1.0)
        success_mask = stats.hits > 0
        log_reward = torch.where(
            success_mask,
            torch.full_like(reach_fraction, float(math.log(self.success_reward))),
            torch.full_like(reach_fraction, float(math.log(self.failure_reward))),
        )
        reward = log_reward.exp()

        path_exists_tensor = (
            path_exists.to(device).bool()
            if path_exists is not None
            else torch.zeros(num_graphs, dtype=torch.bool, device=device)
        )
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=success_mask.float(),
            answer_reach_frac=reach_fraction.float(),
            pos_precision=pos_prec,
            pos_recall=pos_rec,
            pos_f1=pos_f1,
            answer_precision=stats.precision,
            answer_recall=stats.recall,
            answer_f1=stats.f1,
            path_exists=path_exists_tensor,
        )

    @staticmethod
    def _answer_metrics_vectorized(
        *,
        selected_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        node_ptr: torch.Tensor,
        answer_node_locals: torch.Tensor,
        answer_node_ptr: torch.Tensor,
    ) -> _AnswerStats:
        device = selected_mask.device
        num_graphs = int(answer_node_ptr.numel() - 1)
        zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        if answer_node_locals.numel() == 0 or num_graphs == 0:
            no_ans = torch.zeros(num_graphs, dtype=torch.bool, device=device)
            return _AnswerStats(zeros, zeros, zeros, no_ans, no_ans, zeros, zeros)

        num_nodes = int(node_ptr[-1].item())
        node_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), node_ptr[1:] - node_ptr[:-1])

        is_answer_node = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        is_answer_node[answer_node_locals] = True

        sel_mask = selected_mask.bool()
        hit_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if sel_mask.any():
            heads_local = edge_index[0][sel_mask]
            tails_local = edge_index[1][sel_mask]
            hit_nodes[heads_local] = True
            hit_nodes[tails_local] = True

        answer_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), answer_node_ptr[1:] - answer_node_ptr[:-1])
        hit_answers = hit_nodes[answer_node_locals].float()
        hits = torch.bincount(answer_batch, weights=hit_answers, minlength=num_graphs)
        totals = torch.bincount(answer_batch, minlength=num_graphs).float()

        # 使用被访问的唯一节点数作为分母，更符合“预测为答案的节点数”语义，避免 answer_density 偏差。
        visited_counts = torch.bincount(node_batch[hit_nodes], minlength=num_graphs).float()
        precision, recall, f1 = AnswerOnlyReward._precision_recall_f1(
            hits=hits,
            predicted_total=visited_counts,
            target_total=totals,
        )
        has_answers = totals > 0
        contains_answer = torch.bincount(node_batch[is_answer_node], minlength=num_graphs) > 0
        return _AnswerStats(precision, recall, f1, has_answers, contains_answer, hits, totals)

    @staticmethod
    def _precision_recall_f1(
        hits: torch.Tensor,
        predicted_total: torch.Tensor,
        target_total: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        precision_hits = torch.minimum(hits, predicted_total)
        recall_hits = torch.minimum(hits, target_total)
        precision = torch.where(predicted_total > 0, precision_hits / predicted_total.clamp(min=1.0), torch.zeros_like(hits))
        recall = torch.where(target_total > 0, recall_hits / target_total.clamp(min=1.0), torch.zeros_like(hits))
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall).clamp(min=1e-12),
            torch.zeros_like(precision),
        )
        return precision, recall, f1


class AnswerDiffusionReward(AnswerOnlyReward):
    """别名保留：AnswerDiffusionReward 已去掉 answer_gravity，等价于 AnswerOnlyReward。"""
    pass


class AnswerFractionReward(nn.Module):
    """
    Coverage-shaped reward based on answer recall fraction.

    Let f in [0,1] be the fraction of answer entities visited by the trajectory.
    We interpolate rewards geometrically:
      log R = log(failure_reward) + f * log(success_reward / failure_reward)
    so that f=0 -> failure_reward and f=1 -> success_reward.
    """

    def __init__(
        self,
        *,
        success_reward: float,
        failure_reward: float,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        if self.success_reward <= 0.0 or self.failure_reward <= 0.0:
            raise ValueError(f"Rewards must be positive; got success_reward={self.success_reward}, failure_reward={self.failure_reward}.")

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        edge_index: torch.Tensor,  # [2, E_total] local node indices (collated)
        node_ptr: torch.Tensor,  # [B+1]
        answer_node_locals: torch.Tensor,  # [A_total] local node indices (collated)
        answer_node_ptr: torch.Tensor,  # [B+1]
        path_exists: Optional[torch.Tensor],
        **_: Any,
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(node_ptr.numel() - 1)
        stats = AnswerOnlyReward._answer_metrics_vectorized(
            selected_mask=selected_mask,
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_ptr=node_ptr,
            answer_node_locals=answer_node_locals,
            answer_node_ptr=answer_node_ptr,
        )
        pos_prec, pos_rec, pos_f1 = _compute_pos_metrics(
            selected_mask=selected_mask.bool(),
            positive_mask=edge_labels > 0.5,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        reach_fraction = stats.recall.clamp(min=0.0, max=1.0)
        success_mask = stats.hits > 0
        log_ratio = float(math.log(self.success_reward / self.failure_reward))
        log_reward = torch.full_like(reach_fraction, float(math.log(self.failure_reward))) + reach_fraction * log_ratio
        reward = log_reward.exp()

        path_exists_tensor = (
            path_exists.to(device).bool()
            if path_exists is not None
            else torch.zeros(num_graphs, dtype=torch.bool, device=device)
        )
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=success_mask.float(),
            answer_reach_frac=reach_fraction.float(),
            pos_precision=pos_prec,
            pos_recall=pos_rec,
            pos_f1=pos_f1,
            answer_precision=stats.precision,
            answer_recall=stats.recall,
            answer_f1=stats.f1,
            path_exists=path_exists_tensor,
        )


__all__ = [
    "RewardOutput",
    "AnswerOnlyReward",
    "AnswerFractionReward",
    "AnswerDiffusionReward",
]
