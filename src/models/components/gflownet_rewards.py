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
    path_prefix_len: torch.Tensor
    path_prefix_ratio: torch.Tensor
    path_full_hit: torch.Tensor
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
        zeros = torch.zeros_like(reach_fraction, dtype=torch.float32)
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
            path_prefix_len=zeros,
            path_prefix_ratio=zeros,
            path_full_hit=zeros,
            answer_hit=success_mask.float(),
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
        zeros = torch.zeros_like(reach_fraction, dtype=torch.float32)
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
            path_prefix_len=zeros,
            path_prefix_ratio=zeros,
            path_full_hit=zeros,
            answer_hit=success_mask.float(),
        )


class GTPathAlignedReward(nn.Module):
    """
    奖励对齐 GT 路径：前缀一致性 + 命中答案 − 长度惩罚（log 域）。

    log R = log(failure_reward) + s * log(success_reward / failure_reward) - lambda_len * norm_len
    s = (alpha_prefix * prefix_ratio + beta_answer * answer_hit) / (alpha_prefix + beta_answer)
    - prefix_ratio: 动作序列与 GT 路径的最长前缀匹配比例
    - answer_hit: 轨迹是否访问到任一答案节点
    - norm_len: 轨迹长度 / max_steps
    """

    def __init__(
        self,
        *,
        alpha_prefix: float,
        beta_answer: float,
        answer_gate_full_path: bool,
        lambda_len: float,
        success_reward: float,
        failure_reward: float,
    ) -> None:
        super().__init__()
        if alpha_prefix < 0 or beta_answer < 0:
            raise ValueError("alpha_prefix and beta_answer must be non-negative.")
        if alpha_prefix + beta_answer <= 0:
            raise ValueError("alpha_prefix + beta_answer must be positive.")
        if lambda_len < 0:
            raise ValueError("lambda_len must be >= 0.")
        if success_reward <= 0.0 or failure_reward <= 0.0:
            raise ValueError(
                f"Rewards must be positive; got success_reward={success_reward}, failure_reward={failure_reward}."
            )
        if success_reward <= failure_reward:
            raise ValueError("success_reward must be larger than failure_reward.")
        self.alpha_prefix = float(alpha_prefix)
        self.beta_answer = float(beta_answer)
        self.answer_gate_full_path = bool(answer_gate_full_path)
        self.lambda_len = float(lambda_len)
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)

    def forward(
        self,
        *,
        actions_seq: torch.Tensor,           # [B, T_action]
        edge_ptr: torch.Tensor,              # [B+1]
        selected_mask: torch.Tensor,          # [E_total] bool
        selection_order: torch.Tensor,        # [E_total] long (-1 for unselected)
        edge_batch: torch.Tensor,             # [E_total] graph ids
        path_mask: Optional[torch.Tensor],    # [E_total] bool (unused; kept for signature parity)
        path_exists: torch.Tensor,            # [B] bool
        length: torch.Tensor,                 # [B] number of selected edges
        max_steps: torch.Tensor,              # scalar tensor (same device)
        gt_path_edge_local_ids: torch.Tensor, # [P_total]
        gt_path_ptr: torch.Tensor,            # [B+1]
        reach_success: torch.Tensor,          # [B]
        **_: Any,
    ) -> RewardOutput:
        device = selected_mask.device
        if selection_order.shape != selected_mask.shape:
            raise ValueError("selection_order shape must match selected_mask.")
        if edge_batch.shape != selected_mask.shape:
            raise ValueError("edge_batch shape must match selected_mask.")
        num_graphs = int(path_exists.numel())
        if length.numel() != num_graphs:
            raise ValueError("length must align with batch graphs.")

        if gt_path_ptr.numel() != num_graphs + 1:
            raise ValueError("gt_path_ptr length mismatch; expected one offset per graph.")
        if actions_seq.dim() != 2 or actions_seq.size(0) != num_graphs:
            raise ValueError("actions_seq must be [B,T_action] and aligned to batch size.")

        max_steps_scalar = float(max_steps.item()) if max_steps.numel() > 0 else 1.0
        norm_len = length.float() / max(max_steps_scalar, 1.0)

        gt_counts = gt_path_ptr[1:] - gt_path_ptr[:-1]
        max_gt = int(gt_counts.max().item()) if gt_counts.numel() > 0 else 0
        if max_gt > 0:
            gt_path = torch.full((num_graphs, max_gt), -1, dtype=torch.long, device=device)
            if gt_path_edge_local_ids.numel() > 0:
                if edge_ptr.numel() != num_graphs + 1:
                    raise ValueError("edge_ptr length mismatch; expected one offset per graph.")
                gt_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), gt_counts)
                gt_pos = torch.arange(gt_path_edge_local_ids.numel(), device=device) - gt_path_ptr[gt_batch]
                gt_edges = gt_path_edge_local_ids.to(device=device, dtype=torch.long)
                edge_start = edge_ptr[:-1].to(device=device)
                edge_end = edge_ptr[1:].to(device=device)
                in_range = (gt_edges >= edge_start[gt_batch]) & (gt_edges < edge_end[gt_batch])
                if not bool(in_range.all().item()):
                    bad = torch.nonzero(~in_range, as_tuple=False).view(-1)
                    preview = bad[:5].detach().cpu().tolist()
                    raise ValueError(
                        "gt_path_edge_local_ids must be batch-global edge indices; "
                        f"found out-of-range entries at positions={preview}."
                    )
                local_edges = gt_edges - edge_start[gt_batch]
                gt_path[gt_batch, gt_pos] = local_edges
        else:
            gt_path = torch.empty(num_graphs, 0, dtype=torch.long, device=device)

        stop_indices = edge_ptr[1:].view(-1, 1).to(device=device)
        actions = actions_seq.to(device=device, dtype=torch.long)
        is_stop = actions == stop_indices
        actions_local = actions - edge_ptr[:-1].view(-1, 1).to(device=device)
        actions_local = torch.where(is_stop, torch.full_like(actions_local, -1), actions_local)

        max_compare = min(actions_local.size(1), gt_path.size(1))
        if max_compare > 0:
            actions_cmp = actions_local[:, :max_compare]
            gt_cmp = gt_path[:, :max_compare]
            valid = gt_cmp >= 0
            match = (actions_cmp == gt_cmp) & valid & (actions_cmp >= 0)
            prefix_mask = match.float().cumprod(dim=1)
            prefix_len = prefix_mask.sum(dim=1)
        else:
            prefix_len = torch.zeros(num_graphs, device=device, dtype=torch.float32)

        gt_len = gt_counts.to(dtype=prefix_len.dtype)
        prefix_ratio = torch.zeros_like(prefix_len)
        has_gt = gt_counts > 0
        if has_gt.any():
            prefix_ratio[has_gt] = prefix_len[has_gt] / gt_len[has_gt].clamp(min=1.0)

        prefix_len_long = prefix_len.to(dtype=torch.long)
        full_hit = (gt_counts > 0) & (prefix_len_long == gt_counts)
        if full_hit.any():
            next_idx = gt_counts.clamp(min=0)
            has_next = next_idx < actions_local.size(1)
            stop_after = torch.zeros_like(full_hit, dtype=torch.bool)
            if has_next.any():
                next_actions = actions_local[has_next].gather(1, next_idx[has_next].view(-1, 1)).view(-1)
                stop_after[has_next] = next_actions < 0
            stop_after[~has_next] = True
            full_hit = full_hit & stop_after

        answer_hit = reach_success.to(device=device, dtype=torch.float32).clamp(min=0.0, max=1.0)
        if self.answer_gate_full_path:
            answer_hit = answer_hit * full_hit.to(dtype=answer_hit.dtype)
        denom = self.alpha_prefix + self.beta_answer
        score = (self.alpha_prefix * prefix_ratio + self.beta_answer * answer_hit) / denom
        score = score.clamp(min=0.0, max=1.0)
        log_ratio = float(math.log(self.success_reward / self.failure_reward))
        log_reward = torch.full_like(score, float(math.log(self.failure_reward))) + score * log_ratio
        if self.lambda_len > 0:
            log_reward = log_reward - self.lambda_len * norm_len
        reward = log_reward.exp()
        success = answer_hit

        path_exists_tensor = path_exists.to(device).bool()
        zeros = torch.zeros_like(reward)

        return RewardOutput(
            reward=reward,
            log_reward=log_reward,
            success=success,
            answer_reach_frac=answer_hit,
            pos_precision=zeros,
            pos_recall=zeros,
            pos_f1=zeros,
            answer_precision=zeros,
            answer_recall=zeros,
            answer_f1=zeros,
            path_exists=path_exists_tensor,
            path_prefix_len=prefix_len.to(dtype=reward.dtype),
            path_prefix_ratio=prefix_ratio.to(dtype=reward.dtype),
            path_full_hit=full_hit.float(),
            answer_hit=answer_hit,
        )


__all__ = [
    "RewardOutput",
    "GTPathAlignedReward",
]
