from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.weaver.policy import CandidateEdges
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.state import State


@dataclass(frozen=True)
class CoverageProposalScores:
    """
    Candidate-edge coverage proposal scores aligned with CandidateEdges.

    continue_mask:
        Boolean mask over candidate edges. True means the candidate edge is a
        budget-feasible shortest-path progress action toward at least one
        uncovered reachable target.

    expand_score:
        Coverage proposal score used for edge sampling:
            coverage_gain + path_count_tiebreak_weight * path_count_bonus

    coverage_gain:
        Number of uncovered reachable targets for which this candidate edge is
        a valid budget-feasible shortest-path progress action.

    path_count_bonus:
        Tie-breaker based on shortest-path multiplicity after taking the edge.
    """

    continue_mask: torch.Tensor
    expand_score: torch.Tensor
    coverage_gain: torch.Tensor
    path_count_bonus: torch.Tensor


@dataclass(frozen=True)
class CoverageStopDecision:
    """
    Per-graph coverage proposal stop decision.

    The three reason masks are mutually exclusive by construction.

    Priority:
        1. all_targets_covered
        2. no_expand_budget
        3. no_budget_feasible_target

    should_stop:
        True if any mutually exclusive stop reason is true.

    all_targets_covered:
        True if all reachable guide targets are already active.

    no_expand_budget:
        True if there are uncovered targets but no expand action can still be
        taken.

    no_budget_feasible_target:
        True if there are uncovered targets and remaining expand budget, but no
        uncovered target is reachable from the current active set within that
        budget.

    remaining_potential:
        Number of uncovered reachable targets still reachable from some active
        node within the per-graph expand budget available before acting.
    """

    should_stop: torch.Tensor
    all_targets_covered: torch.Tensor
    no_budget_feasible_target: torch.Tensor
    no_expand_budget: torch.Tensor
    remaining_potential: torch.Tensor


@dataclass(frozen=True)
class CoverageProposalContext:
    """
    Step-level cache for coverage proposal decisions.

    Computed once per rollout step before executing the action.

    Shared by:
    - action-type proposal intervention;
    - coverage edge proposal;
    - optional auxiliary edge objective.
    """

    stop_decision: CoverageStopDecision
    scores: CoverageProposalScores
    has_valid_expand: torch.Tensor
    teacher_decision: TeacherDecision | None = None


@dataclass(frozen=True)
class TeacherDecision:
    """
    Marginal-utility teacher decision aligned with terminal reward.

    current_log_value is J(s), the same terminal log reward used by SubTB.
    expand_gain is candidate-aligned Delta_J(e | s) = J(s + e) - J(s).
    """

    should_stop: torch.Tensor
    current_utility: torch.Tensor
    current_log_value: torch.Tensor
    best_expand_gain: torch.Tensor
    best_expand_edge: torch.Tensor
    expand_gain: torch.Tensor
    valid_expand_mask: torch.Tensor


@dataclass(frozen=True)
class CoverageGuideBatchMeta:
    """
    Static per-batch metadata for CoveragePathGuide.

    These values depend only on RetrievalBatch, not on rollout State or
    CandidateEdges. They are cached by CoveragePathGuide to avoid recomputing
    target pointers and flattened offsets at every rollout step.
    """

    num_graphs: int
    guide_target_ids: torch.Tensor

    ptr: torch.Tensor
    edge_ptr: torch.Tensor
    node_counts: torch.Tensor
    edge_counts: torch.Tensor
    target_counts: torch.Tensor

    target_ptr_cpu: tuple[int, ...]
    node_flat_ptr_cpu: tuple[int, ...]
    edge_flat_ptr_cpu: tuple[int, ...]
    node_counts_cpu: tuple[int, ...]
    edge_counts_cpu: tuple[int, ...]
    ptr_cpu: tuple[int, ...]
    edge_ptr_cpu: tuple[int, ...]


@dataclass(frozen=True)
class CandidateGraphGroups:
    """
    Candidate-edge grouping by graph.

    order:
        Candidate positions sorted by graph id.

    candidate_ptr_cpu:
        Prefix sums over candidate counts per graph. Candidate positions for
        graph g are:
            order[candidate_ptr_cpu[g] : candidate_ptr_cpu[g + 1]]
    """

    order: torch.Tensor
    candidate_ptr_cpu: tuple[int, ...]


class MinimalSufficiencyTeacher:
    """
    Reward-aligned one-step teacher for minimal sufficient evidence.

    Stop is proposed iff the current state already has positive verified
    utility and no candidate expansion improves the same terminal objective by
    more than gain_margin:

        U(s) > min_utility and max_e[J(s + e) - J(s)] <= gain_margin

    Expansion scores are a soft distribution over marginal reward gains. The
    returned context intentionally preserves the same shape as
    CoveragePathGuide so rollout sampling can keep recording target-policy
    log-probabilities while only changing behavior sampling.
    """

    def __init__(
        self,
        *,
        min_utility: float = 1.0e-6,
        gain_margin: float = 0.02,
        expand_temperature: float = 1.0,
    ) -> None:
        self.min_utility = float(min_utility)
        self.gain_margin = float(gain_margin)
        self.expand_temperature = float(expand_temperature)

        if self.min_utility < 0.0:
            raise ValueError(f"min_utility must be >= 0, got {min_utility}.")
        if self.gain_margin < 0.0:
            raise ValueError(f"gain_margin must be >= 0, got {gain_margin}.")
        if self.expand_temperature <= 0.0:
            raise ValueError(
                f"expand_temperature must be > 0, got {expand_temperature}."
            )

    @torch.no_grad()
    def build_context(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        expand_budget_before_action: int | torch.Tensor,
        num_graphs: int,
        reward_model: RewardModel,
        current_reward: TerminalRewardOutput | None = None,
    ) -> CoverageProposalContext:
        num_graphs = int(num_graphs)
        device = candidates.edge_ids.device

        if current_reward is None:
            current_reward = reward_model.evaluate_terminal_state(
                retrieval_batch=retrieval_batch,
                active_nodes=state.active_nodes,
                active_edges=state.active_edges,
                state=state,
            )

        budget_per_graph = _per_graph_budget_tensor(
            expand_budget_before_action,
            num_graphs=num_graphs,
            device=device,
            name="expand_budget_before_action",
        )

        expand_gain, valid_expand_mask = self.score_expands(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            reward_model=reward_model,
            current_reward=current_reward,
            budget_per_graph=budget_per_graph,
            num_graphs=num_graphs,
        )

        best_expand_gain, best_expand_edge = _best_candidate_gain_by_graph(
            candidates=candidates,
            expand_gain=expand_gain,
            valid_expand_mask=valid_expand_mask,
            num_graphs=num_graphs,
        )

        current_utility = current_reward.utility.to(device=device, dtype=torch.float32)
        current_log_value = current_reward.log_reward.to(
            device=device, dtype=torch.float32
        )

        should_stop = current_utility.gt(self.min_utility) & best_expand_gain.le(
            self.gain_margin
        )

        teacher_decision = TeacherDecision(
            should_stop=should_stop,
            current_utility=current_utility,
            current_log_value=current_log_value,
            best_expand_gain=best_expand_gain,
            best_expand_edge=best_expand_edge,
            expand_gain=expand_gain,
            valid_expand_mask=valid_expand_mask,
        )

        scores = self._proposal_scores(
            expand_gain=expand_gain,
            valid_expand_mask=valid_expand_mask,
        )
        has_valid_expand = CoveragePathGuide.has_valid_expand_from_scores(
            candidates=candidates,
            scores=scores,
            num_graphs=num_graphs,
        )

        stop_decision = _teacher_stop_decision(
            should_stop=should_stop,
            has_valid_expand=has_valid_expand,
            best_expand_gain=best_expand_gain,
        )

        return CoverageProposalContext(
            stop_decision=stop_decision,
            scores=scores,
            has_valid_expand=has_valid_expand,
            teacher_decision=teacher_decision,
        )

    @torch.no_grad()
    def evaluate_state(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        reward_model: RewardModel,
    ) -> TerminalRewardOutput:
        return reward_model.evaluate_terminal_state(
            retrieval_batch=retrieval_batch,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            state=state,
        )

    @torch.no_grad()
    def score_expands(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        reward_model: RewardModel,
        current_reward: TerminalRewardOutput,
        budget_per_graph: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_graphs

        num_candidates = len(candidates)
        device = candidates.edge_ids.device
        expand_gain = torch.full(
            (num_candidates,),
            float("-inf"),
            dtype=torch.float32,
            device=device,
        )
        valid_expand_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)

        if num_candidates == 0:
            return expand_gain, valid_expand_mask

        candidate_graph_ids = candidates.batch_index.to(device=device, dtype=torch.long)
        candidate_valid = budget_per_graph.index_select(0, candidate_graph_ids).gt(0)
        if not bool(candidate_valid.any()):
            return expand_gain, valid_expand_mask

        edge_ids = candidates.edge_ids.to(device=device, dtype=torch.long)
        edge_index = retrieval_batch.edge_index.to(device=device, dtype=torch.long)
        current_log_value = current_reward.log_reward.to(
            device=device, dtype=torch.float32
        )

        for candidate_pos in candidate_valid.nonzero(as_tuple=False).view(-1).tolist():
            candidate_pos = int(candidate_pos)
            graph_id = int(candidate_graph_ids[candidate_pos].item())
            edge_id = edge_ids[candidate_pos].view(1)

            next_active_nodes = state.active_nodes.detach().clone()
            next_active_edges = state.active_edges.detach().clone()

            src = edge_index[0].index_select(0, edge_id)
            dst = edge_index[1].index_select(0, edge_id)
            next_active_edges[edge_id] = True
            next_active_nodes[src] = True
            next_active_nodes[dst] = True

            next_reward = reward_model.evaluate_terminal_state(
                retrieval_batch=retrieval_batch,
                active_nodes=next_active_nodes,
                active_edges=next_active_edges,
                state=state,
            )
            next_log_value = next_reward.log_reward.to(
                device=device,
                dtype=torch.float32,
            )
            expand_gain[candidate_pos] = (
                next_log_value[graph_id] - current_log_value[graph_id]
            )
            valid_expand_mask[candidate_pos] = True

        return expand_gain, valid_expand_mask

    def decide_stop(
        self,
        *,
        current_utility: torch.Tensor,
        best_expand_gain: torch.Tensor,
    ) -> torch.Tensor:
        return current_utility.gt(self.min_utility) & best_expand_gain.le(
            self.gain_margin
        )

    def _proposal_scores(
        self,
        *,
        expand_gain: torch.Tensor,
        valid_expand_mask: torch.Tensor,
    ) -> CoverageProposalScores:
        weights = torch.zeros_like(expand_gain, dtype=torch.float32)
        if bool(valid_expand_mask.any()):
            scaled = expand_gain[valid_expand_mask] / self.expand_temperature
            scaled = scaled - scaled.max()
            weights[valid_expand_mask] = scaled.exp().clamp_min(
                torch.finfo(weights.dtype).eps
            )

        return CoverageProposalScores(
            continue_mask=valid_expand_mask,
            expand_score=weights,
            coverage_gain=expand_gain,
            path_count_bonus=torch.zeros_like(expand_gain, dtype=torch.float32),
        )


class CoveragePathGuide:
    """
    Coverage-oriented target-path guide for sparse-reward KGQA GFlowNet training.

    This guide is not a semantic oracle. It is a structural proposal scaffold
    built from target-conditioned shortest-path statistics.

    Required RetrievalBatch fields:
    - reachable_target_node_ids
    - target_node_distances_flat
    - target_shortest_path_count_flat
    - target_shortest_path_edge_mask_flat
    """

    def __init__(
        self,
        *,
        path_count_tiebreak_weight: float = 0.0,
    ) -> None:
        if path_count_tiebreak_weight < 0.0:
            raise ValueError(
                "path_count_tiebreak_weight must be >= 0, "
                f"got {path_count_tiebreak_weight}."
            )

        self.path_count_tiebreak_weight = float(path_count_tiebreak_weight)
        self._cached_meta_key: tuple[Any, ...] | None = None
        self._cached_meta: CoverageGuideBatchMeta | None = None

    def build_context(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        expand_budget_before_action: int | torch.Tensor,
        num_graphs: int,
        reward_model: RewardModel | None = None,
        current_reward: TerminalRewardOutput | None = None,
    ) -> CoverageProposalContext:
        """
        Build all coverage proposal signals for one rollout step.

        This is the preferred entry point. It computes edge scores once and
        derives has_valid_expand from those scores.
        """
        del reward_model, current_reward

        meta = self._get_batch_meta(retrieval_batch)

        stop_decision = self.should_stop(
            retrieval_batch=retrieval_batch,
            state=state,
            expand_budget_before_action=expand_budget_before_action,
            num_graphs=int(num_graphs),
            meta=meta,
        )

        scores = self.score_candidate_expands(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            expand_budget_before_action=expand_budget_before_action,
            meta=meta,
        )

        has_valid_expand = self.has_valid_expand_from_scores(
            candidates=candidates,
            scores=scores,
            num_graphs=int(num_graphs),
        )

        return CoverageProposalContext(
            stop_decision=stop_decision,
            scores=scores,
            has_valid_expand=has_valid_expand,
        )

    def score_candidate_expands(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        expand_budget_before_action: int | torch.Tensor,
        meta: CoverageGuideBatchMeta | None = None,
    ) -> CoverageProposalScores:
        """
        Score candidate expand edges.

        A candidate edge e = u -> v is coverage-valid for target y iff:
        1. y is an uncovered reachable target.
        2. u is active.
        3. v is inactive.
        4. e lies on a shortest path toward y.
        5. dist_y(u) = dist_y(v) + 1.
        6. after consuming the current expand action, y is still reachable
           from v within the remaining budget.

        If expand_budget_before_action is B, the post-action budget is B - 1:

            dist_y(v) <= B - 1
        """

        num_candidates = len(candidates)
        device = candidates.edge_ids.device

        continue_mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        expand_score = torch.zeros(num_candidates, dtype=torch.float32, device=device)
        coverage_gain = torch.zeros(num_candidates, dtype=torch.float32, device=device)
        path_count_bonus = torch.zeros(
            num_candidates,
            dtype=torch.float32,
            device=device,
        )

        if meta is None:
            meta = self._get_batch_meta(retrieval_batch)

        return self._score_candidate_expands_impl(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            expand_budget_before_action=expand_budget_before_action,
            meta=meta,
            continue_mask=continue_mask,
            expand_score=expand_score,
            coverage_gain=coverage_gain,
            path_count_bonus=path_count_bonus,
        )

    def _score_candidate_expands_impl(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        expand_budget_before_action: int | torch.Tensor,
        meta: CoverageGuideBatchMeta,
        continue_mask: torch.Tensor,
        expand_score: torch.Tensor,
        coverage_gain: torch.Tensor,
        path_count_bonus: torch.Tensor,
    ) -> CoverageProposalScores:
        num_candidates = len(candidates)
        budget_per_graph = _per_graph_budget_tensor(
            expand_budget_before_action,
            num_graphs=meta.num_graphs,
            device=continue_mask.device,
            name="expand_budget_before_action",
        )

        if (
            num_candidates == 0
            or meta.guide_target_ids.numel() == 0
            or not bool(budget_per_graph.gt(0).any())
        ):
            return CoverageProposalScores(
                continue_mask=continue_mask,
                expand_score=expand_score,
                coverage_gain=coverage_gain,
                path_count_bonus=path_count_bonus,
            )

        candidate_groups = _group_candidates_by_graph(
            candidates=candidates,
            num_graphs=meta.num_graphs,
        )

        candidate_edge_ids = candidates.edge_ids.long()
        candidate_src = retrieval_batch.edge_index[0].index_select(
            0,
            candidate_edge_ids,
        )
        candidate_dst = retrieval_batch.edge_index[1].index_select(
            0,
            candidate_edge_ids,
        )

        for graph_id in range(meta.num_graphs):
            graph_budget_before_action = int(budget_per_graph[graph_id].item())
            if graph_budget_before_action <= 0:
                continue

            budget_after_action = graph_budget_before_action - 1

            candidate_lo = candidate_groups.candidate_ptr_cpu[graph_id]
            candidate_hi = candidate_groups.candidate_ptr_cpu[graph_id + 1]
            if candidate_lo == candidate_hi:
                continue

            target_lo = meta.target_ptr_cpu[graph_id]
            target_hi = meta.target_ptr_cpu[graph_id + 1]
            if target_lo == target_hi:
                continue

            graph_node_count = meta.node_counts_cpu[graph_id]
            graph_edge_count = meta.edge_counts_cpu[graph_id]

            if graph_node_count <= 0 or graph_edge_count <= 0:
                continue

            matched = candidate_groups.order[candidate_lo:candidate_hi]

            graph_target_ids = meta.guide_target_ids[target_lo:target_hi]
            uncovered_target_mask = ~state.active_nodes.index_select(
                0,
                graph_target_ids,
            )

            if not bool(uncovered_target_mask.any()):
                continue

            node_flat_lo = meta.node_flat_ptr_cpu[graph_id]
            node_flat_hi = meta.node_flat_ptr_cpu[graph_id + 1]
            edge_flat_lo = meta.edge_flat_ptr_cpu[graph_id]
            edge_flat_hi = meta.edge_flat_ptr_cpu[graph_id + 1]

            node_to_target_dist_all = retrieval_batch.target_node_distances_flat[
                node_flat_lo:node_flat_hi
            ].view(-1, graph_node_count)

            node_to_target_count_all = retrieval_batch.target_shortest_path_count_flat[
                node_flat_lo:node_flat_hi
            ].view(-1, graph_node_count)

            edge_on_target_path_all = (
                retrieval_batch.target_shortest_path_edge_mask_flat[
                    edge_flat_lo:edge_flat_hi
                ].view(-1, graph_edge_count)
            )

            node_to_target_dist = node_to_target_dist_all[uncovered_target_mask]
            node_to_target_count = node_to_target_count_all[uncovered_target_mask]
            edge_on_target_path = edge_on_target_path_all[uncovered_target_mask]

            global_edge_ids = candidate_edge_ids.index_select(0, matched)
            local_edge_ids = global_edge_ids - meta.edge_ptr_cpu[graph_id]

            global_src = candidate_src.index_select(0, matched)
            global_dst = candidate_dst.index_select(0, matched)

            local_src = global_src - meta.ptr_cpu[graph_id]
            local_dst = global_dst - meta.ptr_cpu[graph_id]

            src_active = state.active_nodes.index_select(0, global_src)
            dst_active = state.active_nodes.index_select(0, global_dst)

            edge_is_on_target_path = edge_on_target_path.index_select(
                1,
                local_edge_ids,
            )
            src_dist = node_to_target_dist.index_select(1, local_src)
            dst_dist = node_to_target_dist.index_select(1, local_dst)
            dst_path_count = node_to_target_count.index_select(1, local_dst).float()

            valid_for_target = (
                edge_is_on_target_path
                & src_active.unsqueeze(0)
                & ~dst_active.unsqueeze(0)
                & src_dist.ge(1)
                & dst_dist.ge(0)
                & src_dist.eq(dst_dist + 1)
                & dst_dist.le(budget_after_action)
            )

            local_coverage_gain = valid_for_target.float().sum(dim=0)
            local_path_count_support = (dst_path_count * valid_for_target.float()).sum(
                dim=0
            )

            local_path_count_bonus = torch.log1p(local_path_count_support)
            local_expand_score = (
                local_coverage_gain
                + self.path_count_tiebreak_weight * local_path_count_bonus
            )
            local_expand_mask = local_coverage_gain.gt(0.0)

            matched_valid = matched[local_expand_mask]

            continue_mask[matched_valid] = True
            coverage_gain[matched_valid] = local_coverage_gain[local_expand_mask]
            path_count_bonus[matched_valid] = local_path_count_bonus[local_expand_mask]
            expand_score[matched_valid] = local_expand_score[local_expand_mask]

        return CoverageProposalScores(
            continue_mask=continue_mask,
            expand_score=expand_score,
            coverage_gain=coverage_gain,
            path_count_bonus=path_count_bonus,
        )

    def count_budget_feasible_uncovered_targets(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        expand_budget_before_action: int | torch.Tensor,
        num_graphs: int | None = None,
        meta: CoverageGuideBatchMeta | None = None,
    ) -> torch.Tensor:
        """
        Count per graph how many uncovered reachable targets are still feasible.

        A target y is budget-feasible under state s iff:

            exists active node v such that
            0 <= dist_y(v) <= expand_budget_before_action
        """

        if meta is None:
            meta = self._get_batch_meta(retrieval_batch)

        if num_graphs is None:
            num_graphs = meta.num_graphs

        device = meta.ptr.device
        potential = torch.zeros(num_graphs, dtype=torch.long, device=device)
        budget_per_graph = _per_graph_budget_tensor(
            expand_budget_before_action,
            num_graphs=int(num_graphs),
            device=device,
            name="expand_budget_before_action",
        )

        if meta.guide_target_ids.numel() == 0 or not bool(budget_per_graph.ge(0).any()):
            return potential

        limit = min(int(num_graphs), meta.num_graphs)

        for graph_id in range(limit):
            graph_budget_before_action = int(budget_per_graph[graph_id].item())
            if graph_budget_before_action < 0:
                continue

            target_lo = meta.target_ptr_cpu[graph_id]
            target_hi = meta.target_ptr_cpu[graph_id + 1]
            if target_lo == target_hi:
                continue

            graph_node_count = meta.node_counts_cpu[graph_id]
            if graph_node_count <= 0:
                continue

            graph_target_ids = meta.guide_target_ids[target_lo:target_hi]
            uncovered_target_mask = ~state.active_nodes.index_select(
                0,
                graph_target_ids,
            )

            if not bool(uncovered_target_mask.any()):
                continue

            node_lo = meta.ptr_cpu[graph_id]
            node_hi = meta.ptr_cpu[graph_id + 1]

            graph_active_nodes = torch.nonzero(
                state.active_nodes[node_lo:node_hi],
                as_tuple=False,
            ).view(-1)

            if graph_active_nodes.numel() == 0:
                continue

            flat_lo = meta.node_flat_ptr_cpu[graph_id]
            flat_hi = meta.node_flat_ptr_cpu[graph_id + 1]

            node_to_target_dist_all = retrieval_batch.target_node_distances_flat[
                flat_lo:flat_hi
            ].view(-1, graph_node_count)

            node_to_target_dist = node_to_target_dist_all[uncovered_target_mask]

            active_dist = node_to_target_dist.index_select(
                1,
                graph_active_nodes,
            )

            feasible_target_mask = active_dist.ge(0) & active_dist.le(
                graph_budget_before_action
            )
            feasible_targets = feasible_target_mask.any(dim=1)

            potential[graph_id] = feasible_targets.long().sum()

        return potential

    def should_stop(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        expand_budget_before_action: int | torch.Tensor,
        num_graphs: int | None = None,
        meta: CoverageGuideBatchMeta | None = None,
    ) -> CoverageStopDecision:
        """
        Decide whether each graph should stop according to the coverage guide.

        This is candidate-independent. Candidate-dependent Stop can falsely stop
        when the candidate generator prunes too aggressively.

        Stop reasons are mutually exclusive by priority:
            all_targets_covered
            no_expand_budget
            no_budget_feasible_target
        """

        if meta is None:
            meta = self._get_batch_meta(retrieval_batch)

        if num_graphs is None:
            num_graphs = meta.num_graphs

        num_graphs = int(num_graphs)
        device = meta.ptr.device
        budget_per_graph = _per_graph_budget_tensor(
            expand_budget_before_action,
            num_graphs=num_graphs,
            device=device,
            name="expand_budget_before_action",
        )

        uncovered_targets = torch.zeros(num_graphs, dtype=torch.long, device=device)

        limit = min(num_graphs, meta.num_graphs)

        if meta.guide_target_ids.numel() > 0 and limit > 0:
            for graph_id in range(limit):
                target_lo = meta.target_ptr_cpu[graph_id]
                target_hi = meta.target_ptr_cpu[graph_id + 1]
                if target_lo == target_hi:
                    continue

                graph_target_ids = meta.guide_target_ids[target_lo:target_hi]
                uncovered = ~state.active_nodes.index_select(
                    0,
                    graph_target_ids,
                )
                uncovered_targets[graph_id] = uncovered.long().sum()

        all_targets_covered = uncovered_targets.eq(0)

        remaining_potential = self.count_budget_feasible_uncovered_targets(
            retrieval_batch=retrieval_batch,
            state=state,
            expand_budget_before_action=budget_per_graph,
            num_graphs=num_graphs,
            meta=meta,
        )

        raw_no_expand_budget = budget_per_graph <= 0

        no_expand_budget = ~all_targets_covered & raw_no_expand_budget

        no_budget_feasible_target = (
            ~all_targets_covered
            & ~no_expand_budget
            & uncovered_targets.gt(0)
            & remaining_potential.eq(0)
        )

        should_stop = all_targets_covered | no_expand_budget | no_budget_feasible_target

        _validate_exclusive_stop_reasons(
            all_targets_covered=all_targets_covered,
            no_expand_budget=no_expand_budget,
            no_budget_feasible_target=no_budget_feasible_target,
        )

        return CoverageStopDecision(
            should_stop=should_stop,
            all_targets_covered=all_targets_covered,
            no_budget_feasible_target=no_budget_feasible_target,
            no_expand_budget=no_expand_budget,
            remaining_potential=remaining_potential,
        )

    def has_valid_expand(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
        expand_budget_before_action: int | torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Compatibility helper. Prefer build_context(...) in rollout code.
        """

        scores = self.score_candidate_expands(
            retrieval_batch=retrieval_batch,
            state=state,
            candidates=candidates,
            expand_budget_before_action=expand_budget_before_action,
        )
        return self.has_valid_expand_from_scores(
            candidates=candidates,
            scores=scores,
            num_graphs=int(num_graphs),
        )

    @staticmethod
    def has_valid_expand_from_scores(
        *,
        candidates: CandidateEdges,
        scores: CoverageProposalScores,
        num_graphs: int,
    ) -> torch.Tensor:
        if len(candidates) == 0:
            return torch.zeros(
                num_graphs,
                dtype=torch.bool,
                device=candidates.edge_ids.device,
            )

        counts = torch.bincount(
            candidates.batch_index[scores.continue_mask],
            minlength=int(num_graphs),
        )
        return counts.gt(0)

    def _get_batch_meta(
        self,
        retrieval_batch: RetrievalBatch,
    ) -> CoverageGuideBatchMeta:
        self._validate_batch(retrieval_batch)

        key = self._batch_meta_key(retrieval_batch)
        if self._cached_meta_key == key and self._cached_meta is not None:
            return self._cached_meta

        meta = self._build_batch_meta(retrieval_batch)
        self._cached_meta_key = key
        self._cached_meta = meta
        return meta

    def _build_batch_meta(
        self,
        retrieval_batch: RetrievalBatch,
    ) -> CoverageGuideBatchMeta:
        ptr = retrieval_batch.ptr.long()
        edge_ptr = retrieval_batch.edge_ptr.long()

        num_graphs = int(ptr.numel()) - 1

        guide_target_ids = retrieval_batch.reachable_target_node_ids.long()

        node_counts = ptr[1:] - ptr[:-1]
        edge_counts = edge_ptr[1:] - edge_ptr[:-1]

        if guide_target_ids.numel() > 0 and num_graphs > 0:
            target_batch = retrieval_batch.batch.index_select(0, guide_target_ids)
            target_counts = torch.bincount(
                target_batch,
                minlength=num_graphs,
            ).long()
        else:
            target_counts = torch.zeros(
                num_graphs,
                dtype=torch.long,
                device=ptr.device,
            )

        target_ptr = _exclusive_cumsum(target_counts)
        node_flat_ptr = _exclusive_cumsum(target_counts * node_counts)
        edge_flat_ptr = _exclusive_cumsum(target_counts * edge_counts)

        return CoverageGuideBatchMeta(
            num_graphs=num_graphs,
            guide_target_ids=guide_target_ids,
            ptr=ptr,
            edge_ptr=edge_ptr,
            node_counts=node_counts,
            edge_counts=edge_counts,
            target_counts=target_counts,
            target_ptr_cpu=tuple(int(x) for x in target_ptr.detach().cpu().tolist()),
            node_flat_ptr_cpu=tuple(
                int(x) for x in node_flat_ptr.detach().cpu().tolist()
            ),
            edge_flat_ptr_cpu=tuple(
                int(x) for x in edge_flat_ptr.detach().cpu().tolist()
            ),
            node_counts_cpu=tuple(int(x) for x in node_counts.detach().cpu().tolist()),
            edge_counts_cpu=tuple(int(x) for x in edge_counts.detach().cpu().tolist()),
            ptr_cpu=tuple(int(x) for x in ptr.detach().cpu().tolist()),
            edge_ptr_cpu=tuple(int(x) for x in edge_ptr.detach().cpu().tolist()),
        )

    @staticmethod
    def _batch_meta_key(retrieval_batch: RetrievalBatch) -> tuple[Any, ...]:
        """
        Cache key for static RetrievalBatch guide metadata.

        The cache is intentionally shallow: it avoids recomputing static pointer
        metadata for the current batch, but it does not try to be a global LRU.
        """

        return (
            id(retrieval_batch),
            retrieval_batch.ptr.data_ptr(),
            retrieval_batch.edge_ptr.data_ptr(),
            retrieval_batch.reachable_target_node_ids.data_ptr(),
            retrieval_batch.target_node_distances_flat.data_ptr(),
            retrieval_batch.target_shortest_path_count_flat.data_ptr(),
            retrieval_batch.target_shortest_path_edge_mask_flat.data_ptr(),
        )

    @staticmethod
    def _validate_batch(retrieval_batch: RetrievalBatch) -> None:
        required = (
            "ptr",
            "batch",
            "edge_index",
            "edge_ptr",
            "reachable_target_node_ids",
            "target_node_distances_flat",
            "target_shortest_path_count_flat",
            "target_shortest_path_edge_mask_flat",
        )
        missing = [name for name in required if not hasattr(retrieval_batch, name)]
        if missing:
            raise RuntimeError(
                "RetrievalBatch is missing fields required by CoveragePathGuide: "
                f"{missing}. Rebuild preprocessing/materialization artifacts."
            )


def _group_candidates_by_graph(
    *,
    candidates: CandidateEdges,
    num_graphs: int,
) -> CandidateGraphGroups:
    """
    Group candidate positions by graph id.

    This replaces the old per-graph pattern:

        candidates.batch_index.eq(graph_id)
        torch.nonzero(...)

    which scans the entire candidate list once per graph.
    """

    device = candidates.edge_ids.device
    num_candidates = len(candidates)

    if num_candidates == 0:
        empty_order = torch.empty(0, dtype=torch.long, device=device)
        return CandidateGraphGroups(
            order=empty_order,
            candidate_ptr_cpu=tuple(0 for _ in range(num_graphs + 1)),
        )

    batch_index = candidates.batch_index.to(device=device, dtype=torch.long)

    order = torch.argsort(batch_index)
    sorted_batch = batch_index.index_select(0, order)

    candidate_counts = torch.bincount(
        sorted_batch,
        minlength=int(num_graphs),
    ).long()

    candidate_ptr = _exclusive_cumsum(candidate_counts)

    return CandidateGraphGroups(
        order=order,
        candidate_ptr_cpu=tuple(int(x) for x in candidate_ptr.detach().cpu().tolist()),
    )


def _per_graph_budget_tensor(
    value: int | torch.Tensor,
    *,
    num_graphs: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(value, int):
        return torch.full(
            (int(num_graphs),),
            int(value),
            dtype=torch.long,
            device=device,
        )

    value = value.to(device=device, dtype=torch.long)
    if value.ndim == 0:
        return torch.full(
            (int(num_graphs),),
            int(value.item()),
            dtype=torch.long,
            device=device,
        )
    if value.shape != (int(num_graphs),):
        raise ValueError(
            f"{name} must be scalar or shape [{int(num_graphs)}], got {tuple(value.shape)}."
        )
    return value


def _best_candidate_gain_by_graph(
    *,
    candidates: CandidateEdges,
    expand_gain: torch.Tensor,
    valid_expand_mask: torch.Tensor,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = candidates.edge_ids.device
    best_gain = torch.full(
        (int(num_graphs),),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    best_edge = torch.full((int(num_graphs),), -1, dtype=torch.long, device=device)

    if len(candidates) == 0 or not bool(valid_expand_mask.any()):
        return best_gain, best_edge

    candidate_graph_ids = candidates.batch_index.to(device=device, dtype=torch.long)
    candidate_edge_ids = candidates.edge_ids.to(device=device, dtype=torch.long)
    gain = expand_gain.to(device=device, dtype=torch.float32)

    for graph_id in range(int(num_graphs)):
        graph_mask = candidate_graph_ids.eq(graph_id) & valid_expand_mask
        if not bool(graph_mask.any()):
            continue

        graph_pos = graph_mask.nonzero(as_tuple=False).view(-1)
        graph_gain = gain.index_select(0, graph_pos)
        local_best = int(torch.argmax(graph_gain).item())
        candidate_pos = graph_pos[local_best]

        best_gain[graph_id] = gain[candidate_pos]
        best_edge[graph_id] = candidate_edge_ids[candidate_pos]

    return best_gain, best_edge


def _teacher_stop_decision(
    *,
    should_stop: torch.Tensor,
    has_valid_expand: torch.Tensor,
    best_expand_gain: torch.Tensor,
) -> CoverageStopDecision:
    device = should_stop.device
    should_stop = should_stop.to(device=device, dtype=torch.bool)
    has_valid_expand = has_valid_expand.to(device=device, dtype=torch.bool)
    best_expand_gain = best_expand_gain.to(device=device, dtype=torch.float32)

    no_expand_budget = ~should_stop & ~has_valid_expand
    no_budget_feasible_target = torch.zeros_like(should_stop)

    return CoverageStopDecision(
        should_stop=should_stop | no_expand_budget,
        all_targets_covered=should_stop,
        no_budget_feasible_target=no_budget_feasible_target,
        no_expand_budget=no_expand_budget,
        remaining_potential=best_expand_gain.gt(float("-inf")).long(),
    )


def _exclusive_cumsum(values: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        values.numel() + 1,
        dtype=values.dtype,
        device=values.device,
    )
    if values.numel() > 0:
        out[1:] = torch.cumsum(values, dim=0)
    return out


def _validate_exclusive_stop_reasons(
    *,
    all_targets_covered: torch.Tensor,
    no_expand_budget: torch.Tensor,
    no_budget_feasible_target: torch.Tensor,
) -> None:
    overlap = (
        all_targets_covered.long()
        + no_expand_budget.long()
        + no_budget_feasible_target.long()
    )

    if bool((overlap > 1).any()):
        bad = torch.nonzero(overlap > 1, as_tuple=False).view(-1)
        raise RuntimeError(
            "Coverage stop reasons must be mutually exclusive. "
            f"Bad graph ids: {bad.tolist()}."
        )


__all__ = [
    "CoverageGuideBatchMeta",
    "CoveragePathGuide",
    "CoverageProposalContext",
    "CoverageProposalScores",
    "CoverageStopDecision",
    "MinimalSufficiencyTeacher",
    "TeacherDecision",
]
