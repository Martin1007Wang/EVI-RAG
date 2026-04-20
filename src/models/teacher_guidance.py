from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.models.policy import CandidateEdges
from src.models.state import State


@dataclass(frozen=True)
class TeacherGuidanceConfig:
    enabled: bool = False
    mode: str = "bounded_path"
    score_exponent: float = 0.5
    fallback_to_policy: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"shortest_path", "bounded_path"}:
            raise ValueError(
                f"Unsupported teacher.mode {self.mode!r}; expected 'shortest_path' or 'bounded_path'."
            )
        if self.score_exponent < 0.0:
            raise ValueError(
                "teacher.score_exponent must be >= 0, got "
                f"{self.score_exponent}."
            )


class TeacherGuidance:
    """Static teacher support over the current dynamic state.

    ``mode='shortest_path'`` uses the preprocess-time shortest-path union and
    shortest suffix counts. ``mode='bounded_path'`` uses a budget-conditioned
    suffix tensor ``bounded_suffix_count[budget, node]`` to score frontier
    actions that can still reach an answer within the remaining budget.
    """

    def __init__(
        self,
        *,
        mode: str,
        score_exponent: float,
        undirected: bool,
        fallback_to_policy: bool,
    ) -> None:
        if mode not in {"shortest_path", "bounded_path"}:
            raise ValueError(
                f"Unsupported teacher guidance mode {mode!r}."
            )
        if score_exponent < 0.0:
            raise ValueError(
                f"score_exponent must be >= 0, got {score_exponent}."
            )
        self.mode = str(mode)
        self.score_exponent = float(score_exponent)
        self.undirected = bool(undirected)
        self.fallback_to_policy = bool(fallback_to_policy)

    def candidate_scores(
        self,
        *,
        base_graph: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (valid_mask, teacher_scores) aligned with ``candidates``."""

        num_candidates = len(candidates)
        device = candidates.edge_ids.device
        valid_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        scores = torch.zeros(num_candidates, dtype=torch.float32, device=device)
        if num_candidates == 0 or remaining_expand_budget < 0:
            return valid_mask, scores

        if self.mode == "shortest_path":
            return self._shortest_candidate_scores(
                base_graph=base_graph,
                state=state,
                candidates=candidates,
                remaining_expand_budget=remaining_expand_budget,
                valid_mask=valid_mask,
                scores=scores,
            )
        return self._bounded_candidate_scores(
            base_graph=base_graph,
            state=state,
            candidates=candidates,
            remaining_expand_budget=remaining_expand_budget,
            valid_mask=valid_mask,
            scores=scores,
        )

    def _shortest_candidate_scores(
        self,
        *,
        base_graph: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
        valid_mask: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = candidates.edge_ids.long()
        src = base_graph.edge_index[0].index_select(0, edge_ids)
        dst = base_graph.edge_index[1].index_select(0, edge_ids)
        positive = base_graph.positive_edge_mask.index_select(0, edge_ids).bool()
        src_active = state.active_nodes.index_select(0, src)
        dst_active = state.active_nodes.index_select(0, dst)
        src_dist = base_graph.node_to_target_distance.index_select(0, src)
        dst_dist = base_graph.node_to_target_distance.index_select(0, dst)
        src_suffix = base_graph.shortest_suffix_count.index_select(0, src).float()
        dst_suffix = base_graph.shortest_suffix_count.index_select(0, dst).float()

        expand_src_to_dst = (
            positive
            & src_active
            & ~dst_active
            & src_dist.ge(1)
            & dst_dist.ge(0)
            & src_dist.eq(dst_dist + 1)
            & dst_dist.le(remaining_expand_budget)
        )
        expand_dst_to_src = (
            positive
            & dst_active
            & ~src_active
            & dst_dist.ge(1)
            & src_dist.ge(0)
            & dst_dist.eq(src_dist + 1)
            & src_dist.le(remaining_expand_budget)
        )
        if not self.undirected:
            expand_dst_to_src = torch.zeros_like(expand_dst_to_src)

        valid_mask = expand_src_to_dst | expand_dst_to_src
        if not bool(valid_mask.any().item()):
            return valid_mask, scores

        next_suffix = torch.zeros_like(scores)
        next_suffix[expand_src_to_dst] = dst_suffix[expand_src_to_dst]
        next_suffix[expand_dst_to_src] = src_suffix[expand_dst_to_src]
        if self.score_exponent == 0.0:
            scores[valid_mask] = 1.0
        else:
            scores[valid_mask] = next_suffix[valid_mask].clamp_min(1.0).pow(
                self.score_exponent
            )
        return valid_mask, scores

    def _bounded_candidate_scores(
        self,
        *,
        base_graph: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
        valid_mask: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounded_suffix_count = getattr(base_graph, "bounded_suffix_count", None)
        if bounded_suffix_count is None:
            raise RuntimeError(
                "Teacher mode 'bounded_path' requires batch.bounded_suffix_count. "
                "Rebuild preprocess outputs with teacher_budget_max_steps aligned to model.max_steps."
            )
        if bounded_suffix_count.ndim != 2:
            raise ValueError(
                "bounded_suffix_count must be 2D, got shape "
                f"{tuple(bounded_suffix_count.shape)}."
            )
        max_budget = int(bounded_suffix_count.size(0)) - 1
        if remaining_expand_budget > max_budget:
            raise RuntimeError(
                "Teacher bounded suffix tensor does not cover the current rollout budget: "
                f"remaining_expand_budget={remaining_expand_budget}, max_precomputed_budget={max_budget}. "
                "Rebuild preprocess outputs with a larger teacher_budget_max_steps."
            )

        edge_ids = candidates.edge_ids.long()
        src = base_graph.edge_index[0].index_select(0, edge_ids)
        dst = base_graph.edge_index[1].index_select(0, edge_ids)
        src_active = state.active_nodes.index_select(0, src)
        dst_active = state.active_nodes.index_select(0, dst)

        expand_src_to_dst = src_active & ~dst_active
        expand_dst_to_src = dst_active & ~src_active
        next_node = torch.full_like(edge_ids, -1)
        next_node[expand_src_to_dst] = dst[expand_src_to_dst]
        next_node[expand_dst_to_src] = src[expand_dst_to_src]

        next_suffix = torch.zeros_like(scores)
        valid_next = next_node.ge(0)
        if bool(valid_next.any().item()):
            next_suffix[valid_next] = bounded_suffix_count[
                remaining_expand_budget,
                next_node[valid_next],
            ].float()
        valid_mask = valid_next & next_suffix.gt(0.0)
        if not bool(valid_mask.any().item()):
            return valid_mask, scores

        if self.score_exponent == 0.0:
            scores[valid_mask] = 1.0
        else:
            scores[valid_mask] = next_suffix[valid_mask].pow(self.score_exponent)
        return valid_mask, scores

    def graph_has_teacher_expand(
        self,
        *,
        base_graph: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        remaining_expand_budget: int,
        num_graphs: int,
    ) -> torch.Tensor:
        valid_mask, _ = self.candidate_scores(
            base_graph=base_graph,
            state=state,
            candidates=candidates,
            remaining_expand_budget=remaining_expand_budget,
        )
        if not bool(valid_mask.any().item()):
            return torch.zeros(num_graphs, dtype=torch.bool, device=state.device)
        counts = torch.bincount(
            candidates.batch_index[valid_mask], minlength=num_graphs
        ).to(device=state.device)
        return counts.gt(0)

    @staticmethod
    def graph_has_terminal_target(
        *,
        base_graph: RetrievalBatch,
        state: State,
        num_graphs: int,
    ) -> torch.Tensor:
        active_answers = (state.active_nodes & base_graph.is_target_mask).to(torch.int32)
        counts = torch.bincount(
            base_graph.batch[active_answers.bool()], minlength=num_graphs
        ).to(device=state.device)
        return counts.gt(0)


__all__ = ["TeacherGuidance", "TeacherGuidanceConfig"]
