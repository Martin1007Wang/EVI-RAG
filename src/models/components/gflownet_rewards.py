from __future__ import annotations

from dataclasses import dataclass
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
    log_reward_struct: torch.Tensor
    recall: torch.Tensor
    success: torch.Tensor
    semantic_only_success: torch.Tensor
    fallback_to_system1: torch.Tensor
    pos_precision: torch.Tensor
    pos_recall: torch.Tensor
    pos_f1: torch.Tensor
    answer_precision: torch.Tensor
    answer_recall: torch.Tensor
    answer_f1: torch.Tensor
    gt_path_precision: torch.Tensor
    gt_path_recall: torch.Tensor
    gt_path_f1: torch.Tensor
    gt_path_exists: torch.Tensor
    gt_path_full_hit: torch.Tensor
    answer_reach_frac: torch.Tensor
    path_exists: torch.Tensor
    reward_connectivity: torch.Tensor
    reward_path_term: torch.Tensor
    semantic_score: torch.Tensor
    struct_phi_len: torch.Tensor
    struct_phi_score: torch.Tensor
    struct_phi_gt: torch.Tensor
    struct_phi_answer_div: torch.Tensor

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
    """最小化的奖励：命中答案 vs 未命中，shaping 仅通过 log-domain 可控项注入。"""

    def __init__(
        self,
        *,
        success_reward: float = 1.0,
        failure_reward: float = 0.1,
        lambda_reach: float = 0.0,
        gamma_len: float = 0.0,
        gamma_score: float = 0.0,
        gamma_gt: float = 0.0,
        gamma_answer_div: float = 0.0,
        score_clip_max: float = 1.0,
        score_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.lambda_reach = float(lambda_reach)
        self.gamma_len = float(gamma_len)
        self.gamma_score = float(gamma_score)
        self.gamma_gt = float(gamma_gt)
        self.gamma_answer_div = float(gamma_answer_div)
        self.score_clip_max = float(score_clip_max)
        self.score_eps = float(score_eps)

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,  # [E_total]
        edge_labels: torch.Tensor,  # [E_total]
        edge_batch: torch.Tensor,  # [E_total]
        edge_heads: torch.Tensor,  # [E_total]
        edge_tails: torch.Tensor,  # [E_total]
        edge_index: torch.Tensor,  # [2, E_total] local node indices (collated)
        node_ptr: torch.Tensor,  # [B+1]
        answer_entity_ids: torch.Tensor,  # [A_total]
        answer_ptr: torch.Tensor,  # [B+1]
        answer_node_locals: torch.Tensor,  # [A_total] local node indices (collated)
        answer_node_ptr: torch.Tensor,  # [B+1]
        path_mask: Optional[torch.Tensor],
        path_exists: Optional[torch.Tensor],
        reach_success: torch.Tensor,  # [B]
        reach_fraction: torch.Tensor,  # [B]
        edge_scores: Optional[torch.Tensor] = None,  # [E_total], retriever/confidence scores
        **_: Any,
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(answer_ptr.numel() - 1)
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

        success_mask = reach_success.bool()
        base_log_reward = torch.where(
            success_mask,
            torch.log(torch.full_like(reach_fraction, self.success_reward)),
            torch.log(torch.full_like(reach_fraction, self.failure_reward)),
        )

        log_reward = base_log_reward
        if self.lambda_reach != 0.0:
            reach_bonus = self.lambda_reach * reach_fraction * success_mask.float()
            log_reward = log_reward + reach_bonus
        else:
            reach_bonus = torch.zeros_like(log_reward)

        # 结构项：长度惩罚 / 置信度 / GT 覆盖
        zeros_struct = torch.zeros(num_graphs, dtype=log_reward.dtype, device=device)
        phi_len = zeros_struct
        if self.gamma_len != 0.0:
            length = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)
            phi_len = -length

        phi_score = zeros_struct
        if self.gamma_score != 0.0 and edge_scores is not None:
            scores = torch.clamp(edge_scores.to(device), min=self.score_eps, max=self.score_clip_max)
            log_scores = scores.log()
            sel = selected_mask.float()
            sum_scores = torch.bincount(edge_batch, weights=log_scores * sel, minlength=num_graphs)
            counts = torch.bincount(edge_batch, weights=sel, minlength=num_graphs)
            phi_score = torch.where(counts > 0, sum_scores / counts.clamp(min=1.0), zeros_struct)

        gt_prec, gt_rec, gt_f1, gt_full_hit = _compute_path_metrics(
            selected_mask=selected_mask.bool(),
            path_mask=path_mask.bool() if path_mask is not None else None,
            edge_batch=edge_batch,
            path_exists=(path_exists.to(device) if path_exists is not None else stats.contains_answer).bool(),
        )
        phi_gt = gt_f1
        phi_answer_div = stats.hits

        struct_bonus = (
            self.gamma_len * phi_len
            + self.gamma_score * phi_score
            + self.gamma_gt * phi_gt
            + self.gamma_answer_div * phi_answer_div
        )
        log_reward_struct = log_reward + struct_bonus
        log_reward = log_reward_struct

        # 归一化：对齐最大可能奖励到 1，减少对 logZ 初值依赖。
        log_max_reward = torch.log(torch.tensor(self.success_reward, device=log_reward.device, dtype=log_reward.dtype))
        if self.lambda_reach > 0.0:
            log_max_reward = log_max_reward + self.lambda_reach
        if self.gamma_gt > 0.0:
            log_max_reward = log_max_reward + self.gamma_gt
        if self.gamma_answer_div > 0.0 and stats.totals is not None:
            log_max_reward = log_max_reward + self.gamma_answer_div * stats.totals
        log_reward = log_reward - log_max_reward
        reward = log_reward.exp().clamp(min=1e-8)

        zeros = torch.zeros_like(reach_fraction)
        ones = torch.ones_like(reach_fraction)
        path_exists_tensor = path_exists.to(reach_fraction.device) if path_exists is not None else stats.contains_answer
        selected_per_graph = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)
        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            log_reward_struct=log_reward_struct,
            recall=stats.recall,
            success=success_mask.float(),
            semantic_only_success=zeros,
            fallback_to_system1=(selected_per_graph == 0).float(),
            pos_precision=pos_prec,
            pos_recall=pos_rec,
            pos_f1=pos_f1,
            answer_precision=stats.precision,
            answer_recall=stats.recall,
            answer_f1=stats.f1,
            gt_path_precision=gt_prec,
            gt_path_recall=gt_rec,
            gt_path_f1=gt_f1,
            gt_path_exists=path_exists_tensor.bool(),
            gt_path_full_hit=gt_full_hit,
            answer_reach_frac=reach_fraction.float(),
            path_exists=path_exists_tensor,
            reward_connectivity=reach_fraction.float(),
            reward_path_term=ones,
            semantic_score=ones,
            struct_phi_len=phi_len,
            struct_phi_score=phi_score,
            struct_phi_gt=phi_gt,
            struct_phi_answer_div=phi_answer_div,
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


__all__ = [
    "RewardOutput",
    "AnswerOnlyReward",
    "AnswerDiffusionReward",
]
