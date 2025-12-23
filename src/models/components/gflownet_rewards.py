from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
from torch import nn


def _safe_f1(precision: torch.Tensor, recall: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    return 2 * precision * recall / (precision + recall + eps)


def _log_failure_from_graph(
    *,
    node_ptr: torch.Tensor,
    edge_batch: torch.Tensor,
    failure_action_topk: Optional[int],
    failure_path_length: float,
    log_failure_bias: float,
) -> torch.Tensor:
    """Compute log R_failure(g) = -log|V_g| - L*log K_eff + bias (per graph)."""
    device = node_ptr.device
    num_graphs = int(node_ptr.numel() - 1)
    if num_graphs <= 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    node_counts = (node_ptr[1:] - node_ptr[:-1]).to(dtype=torch.float32).clamp(min=1.0)
    if edge_batch.numel() == 0:
        edge_counts = torch.zeros(num_graphs, device=device, dtype=torch.float32)
    else:
        edge_counts = torch.bincount(edge_batch, minlength=num_graphs).to(dtype=torch.float32)

    avg_out_degree = edge_counts / node_counts
    k_eff = avg_out_degree
    if failure_action_topk is not None:
        k_eff = torch.minimum(k_eff, torch.full_like(k_eff, float(failure_action_topk)))
    k_eff = k_eff.clamp(min=1.0)

    log_failure = -torch.log(node_counts) - float(failure_path_length) * torch.log(k_eff)
    if log_failure_bias != 0.0:
        log_failure = log_failure + float(log_failure_bias)
    return log_failure



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
    answer_hit: torch.Tensor

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
    """Binary answer-hit reward with graph-scaled failure in log-domain."""

    def __init__(
        self,
        *,
        success_reward: float,
        failure_action_topk: Optional[int],
        failure_path_length: float,
        log_failure_bias: float,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_action_topk = None if failure_action_topk is None else int(failure_action_topk)
        self.failure_path_length = float(failure_path_length)
        self.log_failure_bias = float(log_failure_bias)
        if self.success_reward <= 0.0:
            raise ValueError(f"success_reward must be positive; got {self.success_reward}.")
        if self.failure_action_topk is not None and self.failure_action_topk <= 0:
            raise ValueError(f"failure_action_topk must be positive when set; got {self.failure_action_topk}.")
        if self.failure_path_length <= 0.0:
            raise ValueError(f"failure_path_length must be positive; got {self.failure_path_length}.")

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        edge_index: torch.Tensor,  # [2, E_total] local node indices (collated)
        node_ptr: torch.Tensor,  # [B+1]
        start_node_locals: torch.Tensor,  # [S_total] local node indices (collated)
        answer_node_locals: torch.Tensor,  # [A_total] local node indices (collated)
        answer_node_ptr: torch.Tensor,  # [B+1]
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(node_ptr.numel() - 1)
        stats = self._answer_metrics_vectorized(
            selected_mask=selected_mask,
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_ptr=node_ptr,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            answer_node_ptr=answer_node_ptr,
        )
        pos_prec, pos_rec, pos_f1 = _compute_pos_metrics(
            selected_mask=selected_mask.bool(),
            positive_mask=edge_labels > 0.5,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        # Keep reward self-contained: success/coverage are determined by (selected edges, answer nodes).
        reach_fraction = stats.recall.clamp(min=0.0, max=1.0)
        success_mask = stats.hits > 0
        log_failure = _log_failure_from_graph(
            node_ptr=node_ptr,
            edge_batch=edge_batch,
            failure_action_topk=self.failure_action_topk,
            failure_path_length=self.failure_path_length,
            log_failure_bias=self.log_failure_bias,
        ).to(dtype=reach_fraction.dtype)
        log_reward = torch.where(
            success_mask,
            torch.full_like(reach_fraction, float(math.log(self.success_reward))),
            log_failure,
        )
        reward = log_reward.exp()

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
            answer_hit=success_mask.float(),
        )

    @staticmethod
    def _answer_metrics_vectorized(
        *,
        selected_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        node_ptr: torch.Tensor,
        start_node_locals: torch.Tensor,
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
        if start_node_locals.numel() > 0:
            hit_nodes[start_node_locals.long()] = True

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


class AnswerAndPositiveEdgeF1Reward(nn.Module):
    """结果奖励 + 集合监督稠密 shaping（基于 positive edge set 的 F1）。

    记轨迹选择的边集合为 S(τ)，正边集合为 P（由 edge_labels>0.5 给出）。
    令 f1(S,P) 为边级 F1，则本奖励在 log-domain 中定义为：

        log R(τ) = log R_base(τ) + α · f1(S(τ), P)

    其中 R_base 是二元的 answer-hit 奖励（success_reward / scaled failure），α>=0 控制稠密项强度。
    该设计保持“集合监督”不变性：不同但等价的最短路径并集成员不会因 tie-break 被惩罚。
    """

    def __init__(
        self,
        *,
        success_reward: float,
        failure_action_topk: Optional[int],
        failure_path_length: float,
        log_failure_bias: float,
        pos_f1_coef: float,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_action_topk = None if failure_action_topk is None else int(failure_action_topk)
        self.failure_path_length = float(failure_path_length)
        self.log_failure_bias = float(log_failure_bias)
        self.pos_f1_coef = float(pos_f1_coef)
        if self.success_reward <= 0.0:
            raise ValueError(f"success_reward must be positive; got {self.success_reward}.")
        if self.failure_action_topk is not None and self.failure_action_topk <= 0:
            raise ValueError(f"failure_action_topk must be positive when set; got {self.failure_action_topk}.")
        if self.failure_path_length <= 0.0:
            raise ValueError(f"failure_path_length must be positive; got {self.failure_path_length}.")
        if self.pos_f1_coef < 0.0:
            raise ValueError(f"pos_f1_coef must be >= 0, got {self.pos_f1_coef}.")

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        edge_index: torch.Tensor,  # [2, E_total] local node indices (collated)
        node_ptr: torch.Tensor,  # [B+1]
        start_node_locals: torch.Tensor,  # [S_total] local node indices (collated)
        answer_node_locals: torch.Tensor,  # [A_total] local node indices (collated)
        answer_node_ptr: torch.Tensor,  # [B+1]
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(node_ptr.numel() - 1)
        stats = AnswerOnlyReward._answer_metrics_vectorized(
            selected_mask=selected_mask,
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_ptr=node_ptr,
            start_node_locals=start_node_locals,
            answer_node_locals=answer_node_locals,
            answer_node_ptr=answer_node_ptr,
        )
        pos_prec, pos_rec, pos_f1 = _compute_pos_metrics(
            selected_mask=selected_mask.bool(),
            positive_mask=edge_labels > 0.5,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )

        success_mask = stats.hits > 0
        log_failure = _log_failure_from_graph(
            node_ptr=node_ptr,
            edge_batch=edge_batch,
            failure_action_topk=self.failure_action_topk,
            failure_path_length=self.failure_path_length,
            log_failure_bias=self.log_failure_bias,
        )
        base_log_reward = torch.where(
            success_mask,
            torch.full((num_graphs,), float(math.log(self.success_reward)), device=device, dtype=torch.float32),
            log_failure.to(dtype=torch.float32),
        )
        log_reward = base_log_reward + (pos_f1.to(dtype=base_log_reward.dtype) * self.pos_f1_coef)
        reward = torch.exp(log_reward)
        reach_fraction = stats.recall.clamp(min=0.0, max=1.0)

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
            answer_hit=success_mask.float(),
        )


__all__ = [
    "RewardOutput",
    "AnswerOnlyReward",
    "AnswerAndPositiveEdgeF1Reward",
]
