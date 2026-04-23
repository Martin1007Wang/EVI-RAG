"""
sampling.py — GFlowNet 边采样与行为采样 (极简纯净版)
专注于 Online Gumbel 探索与 Teacher Forcing 引导。
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.models.policy import CandidateEdges
from src.models.state import State
from src.models.guidance import TeacherGuidance


@dataclass(frozen=True)
class EdgeSampleResult:
    chosen_edges: torch.Tensor
    chosen_local_ids: torch.Tensor
    edge_log_prob: torch.Tensor


def scatter_log_softmax(
    logits: torch.Tensor, batch_idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    max_logits = scatter_max(logits, batch_idx, dim=0, dim_size=num_segments)[0]
    max_per_item = max_logits.gather(0, batch_idx)
    shifted = logits - max_per_item
    sum_exp = scatter_sum(shifted.exp(), batch_idx, dim=0, dim_size=num_segments)
    log_z = max_logits + sum_exp.clamp_min(torch.finfo(logits.dtype).eps).log()
    return logits - log_z.gather(0, batch_idx)


def segmented_gumbel_sample(
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
    active_segments: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    eps = torch.finfo(logits.dtype).tiny
    u = torch.rand_like(logits).clamp(min=eps)
    noisy_logits = logits / temperature - torch.log(-torch.log(u))
    _, full_sampled_idx = scatter_max(
        noisy_logits, batch_idx, dim=0, dim_size=num_segments
    )
    valid_sampled_idx = full_sampled_idx[active_segments]
    log_probs = scatter_log_softmax(logits, batch_idx, num_segments)
    return valid_sampled_idx, log_probs[valid_sampled_idx]


def _compute_edge_log_probs(
    candidates: CandidateEdges, chosen_edges: torch.Tensor, batch_size: int
) -> torch.Tensor:
    log_probs = scatter_log_softmax(
        candidates.expand_logits, candidates.batch_index, num_segments=batch_size
    )
    max_id = (
        int(candidates.edge_ids.max().item()) if candidates.edge_ids.numel() > 0 else 0
    )
    inv_map = torch.full(
        (max_id + 1,), -1, dtype=torch.long, device=chosen_edges.device
    )
    inv_map[candidates.edge_ids] = torch.arange(
        candidates.edge_ids.numel(), device=chosen_edges.device
    )
    pos = inv_map[chosen_edges]
    return log_probs[pos]


class _OnlineSampler:
    def __init__(self, batch_size: int, device: torch.device, edge_ptr: torch.Tensor):
        self._batch_size = batch_size
        self._device = device
        self._edge_ptr = edge_ptr

    def sample_expand_edge(
        self,
        candidates: CandidateEdges,
        expand_graph_ids: torch.Tensor,
        temperature: float,
    ) -> EdgeSampleResult:
        sampled_global_idx, log_probs = segmented_gumbel_sample(
            logits=candidates.expand_logits,
            batch_idx=candidates.batch_index,
            num_segments=self._batch_size,
            active_segments=expand_graph_ids,
            temperature=temperature,
        )
        chosen_edges = candidates.edge_ids[sampled_global_idx]
        chosen_local_ids = chosen_edges - self._edge_ptr[expand_graph_ids]
        return EdgeSampleResult(chosen_edges, chosen_local_ids, log_probs)


class _TeacherSampler:
    def __init__(
        self,
        guidance: TeacherGuidance,
        batch_size: int,
        edge_ptr: torch.Tensor,
        expand_budget: int,
    ):
        self._guidance = guidance
        self._batch_size = batch_size
        self._edge_ptr = edge_ptr
        self._expand_budget = expand_budget

    def sample_expand_edge(
        self,
        candidates: CandidateEdges,
        expand_graph_ids: torch.Tensor,
        num_expands: int,
        retrieval_batch: RetrievalBatch,
        state: State,
    ) -> EdgeSampleResult:
        remaining = self._expand_budget - (num_expands + 1)
        valid_mask, teacher_scores = self._guidance.candidate_scores(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            remaining_expand_budget=remaining,
        )
        chosen_pos = []
        for graph_id in expand_graph_ids.tolist():
            graph_mask = candidates.batch_index.eq(int(graph_id)) & valid_mask
            matched = torch.nonzero(graph_mask, as_tuple=False).view(-1)
            scores = teacher_scores[matched].clamp_min(
                torch.finfo(teacher_scores.dtype).eps
            )
            chosen = matched[torch.multinomial(scores, num_samples=1)[0]]
            chosen_pos.append(int(chosen.item()))

        chosen_edges = candidates.edge_ids[
            torch.tensor(chosen_pos, dtype=torch.long, device=expand_graph_ids.device)
        ]
        chosen_local_ids = chosen_edges - self._edge_ptr[expand_graph_ids]
        log_probs = _compute_edge_log_probs(candidates, chosen_edges, self._batch_size)
        return EdgeSampleResult(chosen_edges, chosen_local_ids, log_probs)


class ActionSampler:
    def __init__(
        self,
        *,
        teacher_guidance: TeacherGuidance | None,
        teacher_force_prob: float,
        edge_ptr: torch.Tensor,
        batch_size: int,
        device: torch.device,
        expand_budget: int,
    ) -> None:
        self.batch_size = batch_size
        self.device = device
        self.force_prob = float(teacher_force_prob)
        self.expand_budget = expand_budget
        self.guidance = teacher_guidance

        self._online = _OnlineSampler(
            batch_size=batch_size, device=device, edge_ptr=edge_ptr
        )
        self._teacher = (
            _TeacherSampler(
                guidance=teacher_guidance,
                batch_size=batch_size,
                edge_ptr=edge_ptr,
                expand_budget=expand_budget,
            )
            if teacher_guidance is not None
            else None
        )

        self._teacher_action_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self._teacher_expand_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

    def teacher_action_counts(self) -> torch.Tensor:
        return self._teacher_action_counts.detach().clone()

    def sample_action_types(
        self,
        behavior_logits: torch.Tensor,
        target_logits: torch.Tensor,
        step_mask: torch.Tensor,
        num_expands: int,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_type = torch.distributions.Categorical(
            logits=behavior_logits.detach()
        ).sample()
        self._teacher_expand_mask.zero_()
        if self.guidance is not None and self.force_prob > 0.0:
            is_teacher = step_mask & (
                torch.rand(self.batch_size, device=self.device) < self.force_prob
            )
            if is_teacher.any():
                should_stop = self.guidance.graph_should_stop(
                    retrieval_batch=retrieval_batch,
                    state=state,
                    candidates=candidates,
                    remaining_expand_budget=self.expand_budget - (num_expands + 1),
                    num_graphs=self.batch_size,
                )
                force_stop = is_teacher & should_stop
                force_expand = is_teacher & ~should_stop
                action_type[force_stop] = 1
                action_type[force_expand] = 0
                self._teacher_expand_mask = force_expand
                self._teacher_action_counts[force_stop | force_expand] += 1
        log_prob = torch.distributions.Categorical(logits=target_logits).log_prob(
            action_type
        )
        return action_type, log_prob

    def sample_expand_edge(
        self,
        candidates: CandidateEdges,
        expand_graph_ids: torch.Tensor,
        temperature: float,
        num_expands: int,
        retrieval_batch: RetrievalBatch,
        state: State,
    ) -> EdgeSampleResult:
        is_teacher = self._teacher_expand_mask[expand_graph_ids]
        teacher_ids = expand_graph_ids[is_teacher]
        online_ids = expand_graph_ids[~is_teacher]
        num_expand = expand_graph_ids.numel()
        chosen_edges = torch.empty(num_expand, dtype=torch.long, device=self.device)
        chosen_local_ids = torch.empty(num_expand, dtype=torch.long, device=self.device)
        edge_log_prob = torch.empty(num_expand, dtype=torch.float32, device=self.device)
        if online_ids.numel() > 0:
            r_on = self._online.sample_expand_edge(
                candidates=candidates,
                expand_graph_ids=online_ids,
                temperature=temperature,
            )
            chosen_edges[~is_teacher] = r_on.chosen_edges
            chosen_local_ids[~is_teacher] = r_on.chosen_local_ids
            edge_log_prob[~is_teacher] = r_on.edge_log_prob
        if teacher_ids.numel() > 0 and self._teacher is not None:
            r_tc = self._teacher.sample_expand_edge(
                candidates=candidates,
                expand_graph_ids=teacher_ids,
                num_expands=num_expands,
                retrieval_batch=retrieval_batch,
                state=state,
            )
            chosen_edges[is_teacher] = r_tc.chosen_edges
            chosen_local_ids[is_teacher] = r_tc.chosen_local_ids
            edge_log_prob[is_teacher] = r_tc.edge_log_prob
        return EdgeSampleResult(chosen_edges, chosen_local_ids, edge_log_prob)


__all__ = [
    "ActionSampler",
    "EdgeSampleResult",
    "scatter_log_softmax",
    "segmented_gumbel_sample",
]
