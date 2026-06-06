from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import NodeSelection, StateBatch

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceStateScoreOutput:
    """Structured output of EvidenceStateScorer.

    Reward decomposition
    --------------------
    state_potential is the dense FL potential:
        Ψ(z) = Cov(z) - α * |S_z| / B

    log_reward is terminal-only:
        log R(z) = Rec(z) - α * |S_z| / B - fail_cost * 1[h(z)=0]

    remaining_log_reward is derived, not independently designed:
        ρ(z) = log R(z) - Ψ(z)

    h(z) and n_Y exclude anchor-target overlap.
    """

    # --- core reward fields ---
    state_potential: Tensor
    remaining_log_reward: Tensor
    log_reward: Tensor
    raw_log_reward: Tensor

    # --- state statistics ---
    answer_count: Tensor
    candidate_count: Tensor
    target_count: Tensor
    target_recall: Tensor
    target_precision: Tensor
    terminal_quality: Tensor
    edge_count: Tensor

    # --- masks ---
    valid_target_mask: Tensor
    nonempty_mask: Tensor
    success_mask: Tensor
    terminal_valid_mask: Tensor

    # --- diagnostics ---
    metrics: dict[str, Tensor] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scorer module
# ---------------------------------------------------------------------------


class EvidenceStateScorer(nn.Module):
    """Compute Ψ(z) and terminal log R(z) for evidence subgraph sampling.

    Ψ(z) = Cov(z) - α * |S_z| / B drives EDGE exploration:
        ΔΨ > 0 for bridge / structural edges that move active nodes closer to
        reachable targets.
        The compactness term is paid on every expansion step through telescope.

    log R(z) = Rec(z) - α * |S_z| / B - fail_cost * 1[h(z)=0] drives STOP quality.
    Because selected-edge states are root-reachable by construction, every
    active node is already anchor-reachable; reward only needs to exclude
    anchor nodes from retrieval credit.

    Parameters
    ----------
    fail_cost : float
        c_fail — terminal penalty when h(z)=0 (no answer hit at STOP).
    alpha : float
        Linear compactness penalty weight.
    budget : int
        Rollout edge budget B used to normalize the compactness penalty.
    """

    def __init__(
        self,
        *,
        fail_cost: float = 1.0,
        alpha: float = 0.1,
        budget: int,
    ) -> None:
        super().__init__()
        self.fail_cost = float(fail_cost)
        self.alpha = float(alpha)
        self.budget = int(budget)
        if self.alpha < 0.0:
            raise ValueError("alpha must be nonnegative.")
        if self.budget <= 0:
            raise ValueError("budget must be positive.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        *,
        state: StateBatch,
        target_context: TargetContext,
        graph_context: GraphContext,
        active: NodeSelection | None = None,
        require_terminal: bool = False,
    ) -> EvidenceStateScoreOutput:
        """
        Parameters
        ----------
        state : StateBatch
        target_context : TargetContext
        graph_context : GraphContext
        active : NodeSelection | None
            Pre-computed active node selection; recomputed if None.
        require_terminal : bool
            If True, raises ValueError when any state is not terminal-valid
            (i.e. empty subgraph or missing targets). Use during training loss
            computation to catch incorrect usage early. Default False.
        """
        device = state.device

        # ------------------------------------------------------------------ #
        # 1.  Active node statistics: h(z) and |X_z|                          #
        # ------------------------------------------------------------------ #
        if active is None:
            active = state.active_node_index(graph_context)

        answer_count = torch.zeros(state.num_states, dtype=torch.float32, device=device)
        candidate_count = torch.zeros_like(answer_count)

        if int(active.node_ids.numel()) > 0:
            # Anchor-target overlap should not receive retrieval credit.
            non_anchor_target_mask = target_context.target_mask & ~graph_context.anchor_mask
            target_hits = non_anchor_target_mask.index_select(0, active.node_ids).float()
            answer_count.scatter_add_(0, active.row_ids, target_hits)
            candidate_count.scatter_add_(
                0,
                active.row_ids,
                torch.ones_like(active.row_ids, dtype=torch.float32),
            )

        # ------------------------------------------------------------------ #
        # 2.  Target / graph-level counts                                      #
        # ------------------------------------------------------------------ #
        effective_target_count = (
            target_context.target_count_by_graph - target_context.anchor_target_count_by_graph
        ).clamp_min(0)
        target_count = effective_target_count.index_select(0, state.graph_ids).float()
        edge_count = state.edge_count.float()

        # ------------------------------------------------------------------ #
        # 3.  Masks                                                            #
        # ------------------------------------------------------------------ #
        valid_target = target_count.gt(0)  # non-anchor reachable targets exist
        nonempty = edge_count.gt(0)  # |S_z| > 0
        success = answer_count.gt(0) & valid_target  # h(z) > 0 and valid
        terminal_valid = valid_target & nonempty  # can legally STOP

        # ------------------------------------------------------------------ #
        # 4.  Guard: if caller insists on terminal validity, enforce it        #
        # ------------------------------------------------------------------ #
        if require_terminal and not bool(terminal_valid.all()):
            n_invalid = int((~terminal_valid).sum().item())
            raise ValueError(
                f"Terminal reward requested but {n_invalid} state(s) are not "
                "terminal-valid (empty subgraph or missing target context). "
                "Check that STOP is only allowed for nonempty states with "
                "valid target context."
            )

        recall = answer_count / target_count.clamp_min(1.0)  # Rec(z)
        precision = answer_count / edge_count.clamp_min(1.0)  # Prec(z)
        compactness_penalty = self.alpha * edge_count / float(self.budget)

        coverage_potential = torch.zeros_like(recall)
        if int(active.node_ids.numel()) > 0:
            active_non_anchor = ~graph_context.anchor_mask.index_select(0, active.node_ids)
            active_distances = target_context.node_target_distance.index_select(0, active.node_ids).float()
            d_max = target_context.target_max_distance_by_graph.index_select(0, state.graph_ids).float()
            active_d_max = d_max.index_select(0, active.row_ids)
            dist_score = (1.0 - active_distances / (active_d_max + 1.0)).clamp_min(0.0)
            dist_score = dist_score * active_non_anchor.float()
            coverage_potential.scatter_add_(0, active.row_ids, dist_score)
            coverage_potential = coverage_potential / target_count.clamp_min(1.0)

        # Empty states keep the FL boundary Ψ(z0)=0. Non-empty states rely on
        # root-reachable state invariants, so every active node is already
        # anchor-reachable and only anchor exclusion is needed here.
        state_potential = torch.where(
            nonempty,
            coverage_potential - compactness_penalty,
            torch.zeros_like(recall),
        )

        terminal_quality = recall - compactness_penalty
        zero_hit = answer_count.eq(0)
        log_reward = terminal_quality - self.fail_cost * zero_hit.float()
        remaining_log_reward = log_reward - state_potential

        # ------------------------------------------------------------------ #
        # 7.  Diagnostics / metrics                                            #
        # ------------------------------------------------------------------ #
        metrics = _build_metrics(
            log_reward=log_reward,
            state_potential=state_potential,
            remaining_log_reward=remaining_log_reward,
            recall=recall,
            precision=precision,
            terminal_quality=terminal_quality,
            compactness_penalty=compactness_penalty,
            coverage_potential=coverage_potential,
            edge_count=edge_count,
            answer_count=answer_count,
            success=success,
            valid_target=valid_target,
            terminal_valid=terminal_valid,
        )

        return EvidenceStateScoreOutput(
            state_potential=state_potential,
            remaining_log_reward=remaining_log_reward,
            log_reward=log_reward,
            raw_log_reward=log_reward,
            answer_count=answer_count,
            candidate_count=candidate_count,
            target_count=target_count,
            target_recall=recall,
            target_precision=precision,
            terminal_quality=terminal_quality,
            edge_count=edge_count,
            valid_target_mask=valid_target,
            nonempty_mask=nonempty,
            success_mask=success,
            terminal_valid_mask=terminal_valid,
            metrics=metrics,
        )


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def _build_metrics(
    *,
    log_reward: Tensor,
    state_potential: Tensor,
    remaining_log_reward: Tensor,
    recall: Tensor,
    precision: Tensor,
    terminal_quality: Tensor,
    compactness_penalty: Tensor,
    coverage_potential: Tensor,
    edge_count: Tensor,
    answer_count: Tensor,
    success: Tensor,
    valid_target: Tensor,
    terminal_valid: Tensor,
) -> dict[str, Tensor]:
    return {
        # --- terminal reward diagnostics ---
        "reward/log_reward_mean": _mean(log_reward),
        "reward/log_reward_std": _std(log_reward),
        "reward/log_reward_min": _min(log_reward),
        "reward/log_reward_max": _max(log_reward),
        # --- FL potential diagnostics ---
        "reward/potential_mean": _mean(state_potential),
        "reward/potential_std": _std(state_potential),
        # --- terminal remainder diagnostics ---
        "reward/residual_mean": _mean(remaining_log_reward),
        "reward/residual_std": _std(remaining_log_reward),
        # --- recall / precision / compactness-adjusted coverage ---
        "reward/recall_mean": _masked_mean(recall, valid_target),
        "reward/precision_mean": _masked_mean(precision, valid_target),
        "reward/coverage_potential_mean": _masked_mean(coverage_potential, valid_target),
        "reward/compactness_penalty_mean": _masked_mean(compactness_penalty, valid_target),
        "reward/terminal_quality_mean": _masked_mean(terminal_quality, valid_target),
        # --- structure ---
        "reward/edge_count_mean": _masked_mean(edge_count, valid_target),
        # --- rates ---
        "reward/hit_rate": _mean(success.float()),
        "reward/valid_rate": _mean(valid_target.float()),
        "reward/terminal_valid_rate": _mean(terminal_valid.float()),
        # --- efficiency ---
        "reward/target_hit_per_edge": _masked_mean(
            answer_count / edge_count.clamp_min(1.0),
            valid_target,
        ),
    }


# ---------------------------------------------------------------------------
# Scalar aggregation helpers
# ---------------------------------------------------------------------------


def _mean(value: Tensor) -> Tensor:
    return value.mean().detach() if int(value.numel()) else value.new_zeros(())


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].mean().detach() if bool(mask.any()) else value.new_zeros(())


def _std(value: Tensor) -> Tensor:
    return value.std(unbiased=False).detach() if int(value.numel()) else value.new_zeros(())


def _min(value: Tensor) -> Tensor:
    return value.min().detach() if int(value.numel()) else value.new_zeros(())


def _max(value: Tensor) -> Tensor:
    return value.max().detach() if int(value.numel()) else value.new_zeros(())


__all__ = [
    "EvidenceStateScorer",
    "EvidenceStateScoreOutput",
]
