from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn


@dataclass
class RewardOutput:
    reward: torch.Tensor
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

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return self.__dict__


class System1GuidedReward(nn.Module):
    """扁平版 System1 reward（无 dense padding，按图 scatter/gather）。"""

    def __init__(
        self,
        *,
        positive_threshold: float,
        epsilon: float = 0.05,
        alpha_semantic: float = 1.0,
        base_success_reward: float = 10.0,
        terminal_bonus: float = 1.0,
        illegal_reward: float = 1e-8,
        length_penalty: float = 0.9,
        length_penalty_retriever: Optional[float] = None,
        semantic_success_threshold: float = 0.8,
        allow_semantic_only_success: bool = False,
        require_full_path_hit: bool = True,
        path_f1_power: float = 1.0,
        path_success_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.positive_threshold = float(positive_threshold)
        self.epsilon = float(epsilon)
        self.alpha_semantic = float(alpha_semantic)
        self.base_success_reward = float(base_success_reward)
        self.terminal_bonus = float(terminal_bonus)
        self.illegal_reward = float(illegal_reward)
        self.length_penalty = float(length_penalty)
        self.length_penalty_retriever = float(length_penalty_retriever or length_penalty)
        self.semantic_success_threshold = float(semantic_success_threshold)
        self.allow_semantic_only_success = bool(allow_semantic_only_success)
        self.require_full_path_hit = bool(require_full_path_hit)
        self.path_f1_power = float(path_f1_power)
        self.path_success_threshold = float(path_success_threshold)

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,     # [E_total] bool
        edge_labels: torch.Tensor,       # [E_total] float
        edge_scores: torch.Tensor,       # [E_total] float
        edge_batch: torch.Tensor,        # [E_total] long
        edge_heads: torch.Tensor,        # [E_total] global entity id
        edge_tails: torch.Tensor,        # [E_total] global entity id
        answer_entity_ids: torch.Tensor, # [A_total]
        answer_ptr: torch.Tensor,        # [B+1]
        path_mask: Optional[torch.Tensor],   # [E_total] bool or None
        path_exists: Optional[torch.Tensor], # [B] bool or None
        reach_success: torch.Tensor,     # [B] float/bool
        reach_fraction: torch.Tensor,    # [B] float
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(answer_ptr.numel() - 1)

        selected_total = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)
        fallback_to_system1 = selected_total == 0

        pos_mask_eval = edge_labels > self.positive_threshold
        pos_total_eval = torch.bincount(edge_batch, weights=pos_mask_eval.float(), minlength=num_graphs)
        selected_pos_eval = torch.bincount(edge_batch, weights=(selected_mask & pos_mask_eval).float(), minlength=num_graphs)
        pos_precision, pos_recall, pos_f1 = self._precision_recall_f1(selected_pos_eval, selected_total, pos_total_eval)

        pos_mask_reward = edge_labels > self.positive_threshold
        pos_total_reward = torch.bincount(edge_batch, weights=pos_mask_reward.float(), minlength=num_graphs)
        selected_pos_reward = torch.bincount(edge_batch, weights=(selected_mask & pos_mask_reward).float(), minlength=num_graphs)
        label_recall = torch.where(
            pos_total_reward > 0,
            selected_pos_reward / pos_total_reward.clamp(min=1.0),
            torch.zeros_like(selected_pos_reward),
        )

        answer_precision, answer_recall, answer_f1, has_answers, contains_answer = self._answer_metrics_flat(
            selected_edges=selected_mask,
            edge_heads=edge_heads,
            edge_tails=edge_tails,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
            edge_batch=edge_batch,
        )
        recall = torch.where(has_answers, answer_recall, label_recall)

        path_precision, path_recall, path_f1, has_gt_path, path_full_hit = self._path_metrics_flat(
            path_mask=path_mask,
            path_exists=path_exists,
            selected_edges=selected_mask,
            edge_batch=edge_batch,
            num_graphs=num_graphs,
        )
        recall = torch.where(has_gt_path, path_recall, recall)

        success = reach_success.bool()
        if has_gt_path.any():
            if self.require_full_path_hit:
                success = success & path_full_hit
            else:
                success = success & (path_f1 >= self.path_success_threshold)
        reach_fraction = reach_fraction.float()
        connectivity_likelihood = (reach_fraction + self.epsilon).clamp(min=1e-6)

        effective_scores = edge_scores.clamp(min=1e-8, max=1.0)
        selected_scores_sum = torch.bincount(edge_batch, weights=effective_scores * selected_mask.float(), minlength=num_graphs)
        semantic_mean = torch.where(
            selected_total > 0,
            selected_scores_sum / selected_total.clamp(min=1.0),
            torch.zeros_like(selected_scores_sum),
        ).clamp(min=1e-8, max=1.0)
        semantic_score = semantic_mean.pow(self.alpha_semantic)

        gamma = torch.full_like(selected_total, self.length_penalty_retriever)
        reward_path_term = gamma.pow(selected_total)

        path_term = torch.ones_like(reward_path_term)
        if has_gt_path.any():
            path_term = torch.where(
                has_gt_path,
                path_f1.clamp(min=1e-3).pow(self.path_f1_power),
                path_term,
            )

        reward = torch.where(
            success,
            self.base_success_reward * reward_path_term * path_term * semantic_score * connectivity_likelihood,
            torch.full_like(connectivity_likelihood, self.illegal_reward),
        )
        if self.allow_semantic_only_success:
            reward = torch.where(
                (semantic_score >= self.semantic_success_threshold) & contains_answer,
                self.base_success_reward * reward_path_term * semantic_score,
                reward,
            )
        if self.require_full_path_hit and has_gt_path.any():
            reward = torch.where(path_full_hit, reward * (1.0 + self.terminal_bonus), reward)

        reward = reward.clamp(min=self.illegal_reward)

        return RewardOutput(
            reward=reward,
            recall=recall,
            success=success.float(),
            semantic_only_success=torch.zeros_like(recall),
            fallback_to_system1=fallback_to_system1.float(),
            pos_precision=pos_precision,
            pos_recall=pos_recall,
            pos_f1=pos_f1,
            answer_precision=answer_precision,
            answer_recall=answer_recall,
            answer_f1=answer_f1,
            gt_path_precision=path_precision,
            gt_path_recall=path_recall,
            gt_path_f1=path_f1,
            gt_path_exists=has_gt_path,
            gt_path_full_hit=path_full_hit.float(),
            answer_reach_frac=reach_fraction.float(),
            path_exists=path_exists if path_exists is not None else torch.zeros_like(success, dtype=torch.bool),
            reward_connectivity=reach_fraction.float(),
            reward_path_term=reward_path_term,
            semantic_score=semantic_score,
        )

    def _path_metrics_flat(
        self,
        path_mask: Optional[torch.Tensor],    # [E_total] bool or None
        path_exists: Optional[torch.Tensor],  # [B] bool or None
        selected_edges: torch.Tensor,         # [E_total] bool
        edge_batch: torch.Tensor,             # [E_total] long
        num_graphs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = selected_edges.device
        if path_mask is None:
            zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
            mask = path_exists.to(device) if path_exists is not None else torch.zeros(num_graphs, dtype=torch.bool, device=device)
            return zeros, zeros, zeros, mask, torch.zeros(num_graphs, dtype=torch.bool, device=device)
        pos_total = torch.bincount(edge_batch, weights=path_mask.float(), minlength=num_graphs)
        hits = torch.bincount(edge_batch, weights=(selected_edges & path_mask).float(), minlength=num_graphs)
        precision, recall, f1 = self._precision_recall_f1(
            hits=hits,
            predicted_total=torch.bincount(edge_batch, weights=selected_edges.float(), minlength=num_graphs),
            target_total=pos_total,
        )
        has_path = pos_total > 0
        path_full_hit = hits == pos_total
        if path_exists is not None:
            has_path = path_exists.to(device).bool()
        return precision, recall, f1, has_path, path_full_hit

    @staticmethod
    def _precision_recall_f1(
        hits: torch.Tensor,
        predicted_total: torch.Tensor,
        target_total: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _answer_metrics_flat(
        *,
        selected_edges: torch.Tensor,
        edge_heads: torch.Tensor,
        edge_tails: torch.Tensor,
        answer_entity_ids: torch.Tensor,
        answer_ptr: torch.Tensor,
        edge_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = selected_edges.device
        num_graphs = int(answer_ptr.numel() - 1)
        zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        if answer_entity_ids.numel() == 0:
            mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
            return zeros, zeros, zeros, mask, mask

        selected_heads = torch.where(selected_edges, edge_heads, torch.full_like(edge_heads, -1))
        selected_tails = torch.where(selected_edges, edge_tails, torch.full_like(edge_tails, -1))

        hits = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        totals = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        contains_answer = torch.zeros(num_graphs, dtype=torch.bool, device=device)

        for g in range(num_graphs):
            a_start, a_end = int(answer_ptr[g].item()), int(answer_ptr[g + 1].item())
            if a_start == a_end:
                continue
            answers = answer_entity_ids[a_start:a_end]
            mask = (edge_batch == g)
            sel_h = selected_heads[mask]
            sel_t = selected_tails[mask]
            cand_h = edge_heads[mask]
            cand_t = edge_tails[mask]
            any_hits = ((sel_h.unsqueeze(1) == answers) | (sel_t.unsqueeze(1) == answers)).any(dim=(0, 1))
            hits[g] = any_hits.sum().float()
            totals[g] = float(answers.numel())
            contains_answer[g] = ((cand_h.unsqueeze(1) == answers) | (cand_t.unsqueeze(1) == answers)).any()

        precision, recall, f1 = System1GuidedReward._precision_recall_f1(
            hits=hits,
            predicted_total=torch.bincount(edge_batch, weights=selected_edges.float(), minlength=num_graphs),
            target_total=totals,
        )
        has_answers = totals > 0
        return precision, recall, f1, has_answers, contains_answer


@dataclass
class _AnswerStats:
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    has_answers: torch.Tensor
    contains_answer: torch.Tensor


class AnswerOnlyReward(nn.Module):
    """最简奖励：命中答案实体则奖励，可选软 shaping 项。"""

    def __init__(
        self,
        *,
        success_reward: float = 1.0,
        failure_reward: float = 1.0e-8,
        beta_reach: float = 0.0,
        beta_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)
        self.beta_reach: float = float(beta_reach)
        self.beta_score: float = float(beta_score)

    def forward(
        self,
        *,
        selected_mask: torch.Tensor,    # [E_total]
        edge_labels: torch.Tensor,      # [E_total]
        edge_scores: torch.Tensor,      # [E_total]
        edge_batch: torch.Tensor,       # [E_total]
        edge_heads: torch.Tensor,       # [E_total]
        edge_tails: torch.Tensor,       # [E_total]
        answer_entity_ids: torch.Tensor,# [A_total]
        answer_ptr: torch.Tensor,       # [B+1]
        path_mask: Optional[torch.Tensor],
        path_exists: Optional[torch.Tensor],
        reach_success: torch.Tensor,    # [B]
        reach_fraction: torch.Tensor,   # [B]
    ) -> RewardOutput:
        device = selected_mask.device
        num_graphs = int(answer_ptr.numel() - 1)
        stats = self._answer_metrics_flat(
            selected_mask=selected_mask,
            edge_heads=edge_heads,
            edge_tails=edge_tails,
            edge_batch=edge_batch,
            answer_entity_ids=answer_entity_ids,
            answer_ptr=answer_ptr,
        )

        success_mask = reach_success.bool()
        success_reward = torch.full_like(reach_fraction, self.success_reward)
        failure_reward = torch.full_like(reach_fraction, self.failure_reward)
        reward = torch.where(success_mask, success_reward, failure_reward).clamp(min=1e-8)

        if self.beta_reach != 0.0:
            reward = reward * torch.exp(self.beta_reach * reach_fraction)

        if self.beta_score != 0.0:
            safe_scores = torch.nan_to_num(edge_scores, nan=0.0, posinf=0.0, neginf=0.0)
            normalized_scores = self._standardize_scores_flat(safe_scores, edge_batch, num_graphs)
            selected_scores_sum = torch.bincount(
                edge_batch, weights=normalized_scores * selected_mask.float(), minlength=num_graphs
            )
            selected_count = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs).clamp(min=1.0)
            score_mean = (selected_scores_sum / selected_count).clamp(min=-4.0, max=4.0)
            reward = reward * torch.exp(self.beta_score * score_mean)

        reward = reward.clamp(min=1e-8)

        zeros = torch.zeros_like(reach_fraction)
        ones = torch.ones_like(reach_fraction)
        path_exists_tensor = (
            path_exists.to(reach_fraction.device).float() if path_exists is not None else stats.contains_answer.float()
        )

        selected_per_graph = torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs)

        return RewardOutput(
            reward=reward,
            recall=stats.recall,
            success=success_mask.float(),
            semantic_only_success=zeros,
            fallback_to_system1=(selected_per_graph == 0).float(),
            pos_precision=zeros,
            pos_recall=zeros,
            pos_f1=zeros,
            answer_precision=stats.precision,
            answer_recall=stats.recall,
            answer_f1=stats.f1,
            gt_path_precision=zeros,
            gt_path_recall=zeros,
            gt_path_f1=zeros,
            gt_path_exists=path_exists.to(reach_fraction.device) if path_exists is not None else torch.zeros_like(success_mask, dtype=torch.bool),
            gt_path_full_hit=zeros,
            answer_reach_frac=reach_fraction.float(),
            path_exists=path_exists_tensor,
            reward_connectivity=reach_fraction.float(),
            reward_path_term=ones,
            semantic_score=ones,
        )

    @staticmethod
    def _answer_metrics_flat(
        *,
        selected_mask: torch.Tensor,
        edge_heads: torch.Tensor,
        edge_tails: torch.Tensor,
        edge_batch: torch.Tensor,
        answer_entity_ids: torch.Tensor,
        answer_ptr: torch.Tensor,
    ) -> _AnswerStats:
        device = selected_mask.device
        num_graphs = int(answer_ptr.numel() - 1)
        zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        if answer_entity_ids.numel() == 0:
            mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)
            return _AnswerStats(zeros, zeros, zeros, mask, mask)

        selected_heads = torch.where(selected_mask, edge_heads, torch.full_like(edge_heads, -1))
        selected_tails = torch.where(selected_mask, edge_tails, torch.full_like(edge_tails, -1))

        hits = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        totals = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        contains_answer = torch.zeros(num_graphs, dtype=torch.bool, device=device)

        for g in range(num_graphs):
            a_start, a_end = int(answer_ptr[g].item()), int(answer_ptr[g + 1].item())
            if a_start == a_end:
                continue
            answers = answer_entity_ids[a_start:a_end]
            mask = (edge_batch == g)
            sel_h = selected_heads[mask]
            sel_t = selected_tails[mask]
            cand_h = edge_heads[mask]
            cand_t = edge_tails[mask]
            any_hits = ((sel_h.unsqueeze(1) == answers) | (sel_t.unsqueeze(1) == answers)).any(dim=(0, 1))
            hits[g] = any_hits.sum().float()
            totals[g] = float(answers.numel())
            contains_answer[g] = ((cand_h.unsqueeze(1) == answers) | (cand_t.unsqueeze(1) == answers)).any()

        precision, recall, f1 = AnswerOnlyReward._precision_recall_f1(
            hits=hits,
            predicted_total=torch.bincount(edge_batch, weights=selected_mask.float(), minlength=num_graphs),
            target_total=totals,
        )
        has_answers = totals > 0
        return _AnswerStats(precision, recall, f1, has_answers, contains_answer)

    @staticmethod
    def _precision_recall_f1(
        hits: torch.Tensor,
        predicted_total: torch.Tensor,
        target_total: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _standardize_scores_flat(scores: torch.Tensor, edge_batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        device = scores.device
        mean = torch.zeros(num_graphs, device=device)
        var = torch.zeros(num_graphs, device=device)
        counts = torch.bincount(edge_batch, minlength=num_graphs).clamp(min=1).float()
        mean.scatter_add_(0, edge_batch, scores)
        mean = mean / counts
        diff = scores - mean[edge_batch]
        var.scatter_add_(0, edge_batch, diff * diff)
        var = var / counts
        std = torch.sqrt(var).clamp(min=1e-6)
        return (scores - mean[edge_batch]) / std[edge_batch]


__all__ = ["RewardOutput", "System1GuidedReward", "AnswerOnlyReward"]
