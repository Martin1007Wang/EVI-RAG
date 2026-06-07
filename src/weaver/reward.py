from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import nn

from src.weaver.context import GraphContext, TargetContext
from src.weaver.state import NodeSelection, StateBatch

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class EvidenceStateScoreOutput:
    """
    Structured output of EvidenceStateScorer.

    Terminal base utility:

        U_R(z)
        = log(eps + Rec(z)) - log(1 + eps)
          - lambda * |S_z| / B

    Terminal reward:

        log R(z)
        = beta * U_R(z)

    Optional graph-wise centering:

        log R_centered(z)
        = log R(z)
          - max_{z' in same graph, terminal-valid} log R(z')

    Dense FL potential:

        U_Psi(z)
        = Prox(z) - lambda * |S_z| / B

        Psi(z)
        = beta * U_Psi(z)

    where:

        Prox(z)
        = max_{x in X_z \\ anchors}
          [1 - d(x, Y) / (d_max(g) + 1)]_+

    remaining_log_reward is derived:

        rho(z) = log R_centered(z) - Psi(z)

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


class EvidenceStateScorer(nn.Module):
    """
    Compute dense FL potential Psi(z) and terminal log reward log R(z).

    This version removes fail_cost.

    Zero-hit states are penalized by the log utility itself:

        log(eps + Rec(z))

    Therefore when eps=1e-4, zero-hit states receive approximately -9.21
    before beta scaling and compactness cost.

    Parameters
    ----------
    reward_beta:
        Inverse temperature of the terminal distribution.
        beta > 1 sharpens P(z) proportional to R(z)^beta.

    edge_cost_lambda:
        Compactness penalty in base utility.
        Because log R(z) = beta * [U_rec(z) - lambda * |S_z| / B],
        lambda controls the recall-vs-compactness tradeoff independently of beta.

    reward_epsilon:
        Fixed reward floor for zero-hit states. Usually keep at 1e-4.

    center_log_reward:
        If True, subtract graph-wise max terminal log reward from log_reward.
        This preserves within-graph relative reward ratios but improves flow scale.

    budget:
        Rollout edge budget B.
    """

    def __init__(
        self,
        *,
        reward_beta: float = 3.0,
        edge_cost_lambda: float = 0.5,
        reward_epsilon: float = 1.0e-4,
        center_log_reward: bool = True,
        budget: int,
    ) -> None:
        super().__init__()

        self.reward_beta = float(reward_beta)
        self.edge_cost_lambda = float(edge_cost_lambda)
        self.reward_epsilon = float(reward_epsilon)
        self.center_log_reward = bool(center_log_reward)
        self.budget = int(budget)

        if self.reward_beta <= 0.0:
            raise ValueError("reward_beta must be positive.")
        if self.edge_cost_lambda < 0.0:
            raise ValueError("edge_cost_lambda must be nonnegative.")
        if not (0.0 < self.reward_epsilon < 1.0):
            raise ValueError("reward_epsilon must be in (0, 1).")
        if self.budget <= 0:
            raise ValueError("budget must be positive.")

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
        device = state.device

        if active is None:
            active = state.active_node_index(graph_context)

        # ------------------------------------------------------------
        # 1. Active node statistics
        # ------------------------------------------------------------
        answer_count = torch.zeros(
            state.num_states,
            dtype=torch.float32,
            device=device,
        )
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

        # ------------------------------------------------------------
        # 2. Target statistics
        # ------------------------------------------------------------
        effective_target_count = (target_context.target_count_by_graph - target_context.anchor_target_count_by_graph).clamp_min(0)

        target_count = effective_target_count.index_select(
            0,
            state.graph_ids,
        ).float()

        edge_count = state.edge_count.float()

        valid_target = target_count.gt(0)
        nonempty = edge_count.gt(0)
        success = answer_count.gt(0) & valid_target
        terminal_valid = valid_target & nonempty

        if require_terminal and not bool(terminal_valid.all()):
            n_invalid = int((~terminal_valid).sum().item())
            raise ValueError(
                f"Terminal reward requested but {n_invalid} state(s) are not " "terminal-valid: empty subgraph or missing non-anchor targets."
            )

        # ------------------------------------------------------------
        # 3. Terminal reward
        # ------------------------------------------------------------
        recall = answer_count / target_count.clamp_min(1.0)
        precision = answer_count / edge_count.clamp_min(1.0)

        compactness_penalty = self.edge_cost_lambda * edge_count / float(self.budget)

        # U_rec(z) = log(eps + Rec(z)) - log(1 + eps)
        #
        # Full recall has utility approximately 0.
        # Zero recall has utility approximately log(eps).
        recall_log_utility = torch.log(self.reward_epsilon + recall) - math.log1p(self.reward_epsilon)

        # Base terminal utility:
        #
        # U_R(z) = U_rec(z) - lambda * |S_z| / B
        #
        # lambda now controls recall-vs-compactness.
        # beta only controls target distribution sharpness.
        terminal_base_utility = recall_log_utility - compactness_penalty

        # Uncentered target log reward:
        #
        # log R(z) = beta * U_R(z)
        uncentered_log_reward = self.reward_beta * terminal_base_utility

        # Only nonempty states with valid target context are legal terminal objects.
        # Keep invalid rows finite and neutral.
        uncentered_log_reward = torch.where(
            terminal_valid,
            uncentered_log_reward,
            torch.zeros_like(uncentered_log_reward),
        )

        # Graph-wise max-centering:
        #
        # log R_centered(z)
        # = log R(z) - max_{terminal-valid z' in same graph} log R(z')
        #
        # This preserves within-graph target distribution:
        #
        # P(z | graph) proportional to R(z).
        if self.center_log_reward:
            reward_shift = _masked_graph_max(
                value=uncentered_log_reward,
                graph_ids=state.graph_ids,
                mask=terminal_valid,
            )
            log_reward = torch.where(
                terminal_valid,
                uncentered_log_reward - reward_shift,
                torch.zeros_like(uncentered_log_reward),
            )
        else:
            reward_shift = torch.zeros_like(uncentered_log_reward)
            log_reward = uncentered_log_reward

        # raw_log_reward means beta-scaled but uncentered log reward.
        raw_log_reward = uncentered_log_reward

        # ------------------------------------------------------------
        # 4. Dense FL potential
        # ------------------------------------------------------------
        proximity_potential = _bounded_target_proximity_potential(
            state=state,
            target_context=target_context,
            graph_context=graph_context,
            active=active,
        )

        # Base shaping utility:
        #
        # U_Psi(z) = Prox(z) - lambda * |S_z| / B
        #
        # Keep lambda semantics aligned with terminal reward.
        potential_base_utility = proximity_potential - compactness_penalty

        # Psi(z) = beta * U_Psi(z)
        #
        # Do not graph-center state_potential. It is a shaping term used in
        # transition residuals; shifting it would alter intermediate equations.
        state_potential = self.reward_beta * potential_base_utility

        # Root states keep Psi(z0)=0.
        # Invalid-target states receive no shaping.
        state_potential = torch.where(
            terminal_valid,
            state_potential,
            torch.zeros_like(state_potential),
        )

        terminal_quality = terminal_base_utility
        remaining_log_reward = log_reward - state_potential

        metrics = _build_metrics(
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
            uncentered_log_reward=uncentered_log_reward,
            reward_shift=reward_shift,
            state_potential=state_potential,
            remaining_log_reward=remaining_log_reward,
            recall=recall,
            precision=precision,
            recall_log_utility=recall_log_utility,
            terminal_base_utility=terminal_base_utility,
            potential_base_utility=potential_base_utility,
            terminal_quality=terminal_quality,
            compactness_penalty=compactness_penalty,
            proximity_potential=proximity_potential,
            edge_count=edge_count,
            answer_count=answer_count,
            success=success,
            valid_target=valid_target,
            terminal_valid=terminal_valid,
            reward_beta=log_reward.new_tensor(self.reward_beta),
            edge_cost_lambda=log_reward.new_tensor(self.edge_cost_lambda),
            reward_epsilon=log_reward.new_tensor(self.reward_epsilon),
            center_log_reward=log_reward.new_tensor(float(self.center_log_reward)),
        )

        return EvidenceStateScoreOutput(
            state_potential=state_potential,
            remaining_log_reward=remaining_log_reward,
            log_reward=log_reward,
            raw_log_reward=raw_log_reward,
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


def _bounded_target_proximity_potential(
    *,
    state: StateBatch,
    target_context: TargetContext,
    graph_context: GraphContext,
    active: NodeSelection,
) -> Tensor:
    """
    Bounded dense proxy for target proximity.

        Prox(z)
        = max_{x in X_z \\ anchors}
          [1 - d(x, Y) / (d_max(g) + 1)]_+

    This avoids rewarding evidence breadth merely because more active nodes
    were added.
    """
    device = state.device
    out = torch.zeros(state.num_states, dtype=torch.float32, device=device)

    if int(active.node_ids.numel()) == 0:
        return out

    non_anchor = ~graph_context.anchor_mask.index_select(0, active.node_ids)

    distances = target_context.node_target_distance.index_select(
        0,
        active.node_ids,
    ).float()

    d_max_by_state = target_context.target_max_distance_by_graph.index_select(
        0,
        state.graph_ids,
    ).float()

    d_max_active = d_max_by_state.index_select(0, active.row_ids)

    score = (1.0 - distances / (d_max_active + 1.0).clamp_min(1.0)).clamp(min=0.0, max=1.0)

    score = score * non_anchor.float()

    out.scatter_reduce_(
        0,
        active.row_ids,
        score,
        reduce="amax",
        include_self=True,
    )

    return out


def _build_metrics(
    *,
    log_reward: Tensor,
    raw_log_reward: Tensor,
    uncentered_log_reward: Tensor,
    reward_shift: Tensor,
    state_potential: Tensor,
    remaining_log_reward: Tensor,
    recall: Tensor,
    precision: Tensor,
    recall_log_utility: Tensor,
    terminal_base_utility: Tensor,
    potential_base_utility: Tensor,
    terminal_quality: Tensor,
    compactness_penalty: Tensor,
    proximity_potential: Tensor,
    edge_count: Tensor,
    answer_count: Tensor,
    success: Tensor,
    valid_target: Tensor,
    terminal_valid: Tensor,
    reward_beta: Tensor,
    edge_cost_lambda: Tensor,
    reward_epsilon: Tensor,
    center_log_reward: Tensor,
) -> dict[str, Tensor]:
    hit_terminal = terminal_valid & success
    nohit_terminal = terminal_valid & (~success)

    hit_log_reward_mean = _masked_mean(log_reward, hit_terminal)
    nohit_log_reward_mean = _masked_mean(log_reward, nohit_terminal)

    hit_uncentered_log_reward_mean = _masked_mean(
        uncentered_log_reward,
        hit_terminal,
    )
    nohit_uncentered_log_reward_mean = _masked_mean(
        uncentered_log_reward,
        nohit_terminal,
    )

    return {
        # --- configuration ---
        "reward/beta": reward_beta.detach(),
        "reward/edge_cost_lambda": edge_cost_lambda.detach(),
        "reward/epsilon": reward_epsilon.detach(),
        "reward/center_log_reward": center_log_reward.detach(),
        # --- centered terminal reward diagnostics ---
        "reward/log_reward_mean": _masked_mean(log_reward, terminal_valid),
        "reward/log_reward_std": _masked_std(log_reward, terminal_valid),
        "reward/log_reward_min": _masked_min(log_reward, terminal_valid),
        "reward/log_reward_max": _masked_max(log_reward, terminal_valid),
        # --- uncentered terminal reward diagnostics ---
        "reward/log_reward_uncentered_mean": _masked_mean(
            uncentered_log_reward,
            terminal_valid,
        ),
        "reward/log_reward_uncentered_std": _masked_std(
            uncentered_log_reward,
            terminal_valid,
        ),
        "reward/log_reward_uncentered_min": _masked_min(
            uncentered_log_reward,
            terminal_valid,
        ),
        "reward/log_reward_uncentered_max": _masked_max(
            uncentered_log_reward,
            terminal_valid,
        ),
        # compatibility alias: beta-scaled, uncentered log reward
        "reward/raw_log_reward_mean": _masked_mean(
            raw_log_reward,
            terminal_valid,
        ),
        # centering diagnostics
        "reward/reward_shift_mean": _masked_mean(reward_shift, terminal_valid),
        "reward/reward_shift_max": _masked_max(reward_shift, terminal_valid),
        # --- beta-free utilities ---
        "reward/recall_log_utility_mean": _masked_mean(
            recall_log_utility,
            terminal_valid,
        ),
        "reward/terminal_base_utility_mean": _masked_mean(
            terminal_base_utility,
            terminal_valid,
        ),
        "reward/potential_base_utility_mean": _masked_mean(
            potential_base_utility,
            terminal_valid,
        ),
        # --- hit / no-hit separation, centered ---
        "reward/log_reward_hit_mean": hit_log_reward_mean,
        "reward/log_reward_nohit_mean": nohit_log_reward_mean,
        "reward/log_reward_hit_nohit_gap": (hit_log_reward_mean - nohit_log_reward_mean).detach(),
        "reward/nohit_mass_fraction": _mass_fraction(
            log_value=log_reward,
            numerator_mask=nohit_terminal,
            denominator_mask=terminal_valid,
        ),
        # --- hit / no-hit separation, uncentered ---
        "reward/log_reward_uncentered_hit_mean": hit_uncentered_log_reward_mean,
        "reward/log_reward_uncentered_nohit_mean": nohit_uncentered_log_reward_mean,
        "reward/log_reward_uncentered_hit_nohit_gap": (hit_uncentered_log_reward_mean - nohit_uncentered_log_reward_mean).detach(),
        "reward/nohit_mass_fraction_uncentered": _mass_fraction(
            log_value=uncentered_log_reward,
            numerator_mask=nohit_terminal,
            denominator_mask=terminal_valid,
        ),
        # --- FL potential diagnostics ---
        "reward/potential_mean": _masked_mean(state_potential, terminal_valid),
        "reward/potential_std": _masked_std(state_potential, terminal_valid),
        "reward/proximity_potential_mean": _masked_mean(
            proximity_potential,
            terminal_valid,
        ),
        "reward/proximity_potential_max": _masked_max(
            proximity_potential,
            terminal_valid,
        ),
        # --- terminal remainder diagnostics ---
        "reward/residual_mean": _masked_mean(remaining_log_reward, terminal_valid),
        "reward/residual_std": _masked_std(remaining_log_reward, terminal_valid),
        # --- quality / compactness ---
        "reward/recall_mean": _masked_mean(recall, valid_target),
        "reward/terminal_recall_mean": _masked_mean(recall, terminal_valid),
        "reward/precision_mean": _masked_mean(precision, valid_target),
        "reward/terminal_precision_mean": _masked_mean(precision, terminal_valid),
        "reward/compactness_penalty_mean": _masked_mean(
            compactness_penalty,
            terminal_valid,
        ),
        "reward/terminal_quality_mean": _masked_mean(
            terminal_quality,
            terminal_valid,
        ),
        # --- structure ---
        "reward/edge_count_mean": _masked_mean(edge_count, valid_target),
        "reward/terminal_edge_count_mean": _masked_mean(edge_count, terminal_valid),
        # --- rates ---
        "reward/hit_rate": _mean(success.float()),
        "reward/terminal_hit_rate": _masked_mean(success.float(), terminal_valid),
        "reward/valid_rate": _mean(valid_target.float()),
        "reward/terminal_valid_rate": _mean(terminal_valid.float()),
        # --- efficiency ---
        "reward/target_hit_per_edge": _masked_mean(
            answer_count / edge_count.clamp_min(1.0),
            valid_target,
        ),
        "reward/terminal_target_hit_per_edge": _masked_mean(
            answer_count / edge_count.clamp_min(1.0),
            terminal_valid,
        ),
    }


def _masked_graph_max(
    *,
    value: Tensor,
    graph_ids: Tensor,
    mask: Tensor,
) -> Tensor:
    """
    For each state row i, return:

        max_{j: graph_ids[j] == graph_ids[i] and mask[j]} value[j]

    If a graph has no masked row, its shift is 0.

    This is used for graph-wise log-reward centering.
    """
    if int(value.numel()) == 0:
        return value.new_zeros(value.shape)

    graph_ids = graph_ids.to(device=value.device, dtype=torch.long).view(-1)
    mask = mask.to(device=value.device, dtype=torch.bool).view(-1)

    if int(graph_ids.numel()) != int(value.numel()):
        raise ValueError("graph_ids must have one value per state.")
    if int(mask.numel()) != int(value.numel()):
        raise ValueError("mask must have one value per state.")

    if not bool(mask.any()):
        return value.new_zeros(value.shape)

    _, inverse_graph_ids = torch.unique(
        graph_ids,
        sorted=True,
        return_inverse=True,
    )

    num_graphs = int(inverse_graph_ids.max().item()) + 1

    graph_max = value.new_full(
        (num_graphs,),
        torch.finfo(value.dtype).min,
    )
    graph_count = torch.zeros(
        num_graphs,
        dtype=torch.long,
        device=value.device,
    )

    masked_inverse = inverse_graph_ids[mask]

    graph_max.scatter_reduce_(
        0,
        masked_inverse,
        value[mask],
        reduce="amax",
        include_self=True,
    )

    graph_count.scatter_add_(
        0,
        masked_inverse,
        torch.ones_like(masked_inverse, dtype=torch.long),
    )

    graph_max = torch.where(
        graph_count.gt(0),
        graph_max,
        graph_max.new_zeros(graph_max.shape),
    )

    return graph_max.index_select(0, inverse_graph_ids)


def _mass_fraction(
    *,
    log_value: Tensor,
    numerator_mask: Tensor,
    denominator_mask: Tensor,
) -> Tensor:
    if not bool(denominator_mask.any()):
        return log_value.new_zeros(())
    if not bool(numerator_mask.any()):
        return log_value.new_zeros(())

    numerator = torch.logsumexp(log_value[numerator_mask], dim=0)
    denominator = torch.logsumexp(log_value[denominator_mask], dim=0)

    return torch.exp(numerator - denominator).detach()


def _mean(value: Tensor) -> Tensor:
    return value.mean().detach() if int(value.numel()) else value.new_zeros(())


def _std(value: Tensor) -> Tensor:
    return value.std(unbiased=False).detach() if int(value.numel()) else value.new_zeros(())


def _min(value: Tensor) -> Tensor:
    return value.min().detach() if int(value.numel()) else value.new_zeros(())


def _max(value: Tensor) -> Tensor:
    return value.max().detach() if int(value.numel()) else value.new_zeros(())


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].mean().detach() if bool(mask.any()) else value.new_zeros(())


def _masked_std(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].std(unbiased=False).detach() if bool(mask.any()) else value.new_zeros(())


def _masked_min(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].min().detach() if bool(mask.any()) else value.new_zeros(())


def _masked_max(value: Tensor, mask: Tensor) -> Tensor:
    return value[mask].max().detach() if bool(mask.any()) else value.new_zeros(())


__all__ = [
    "EvidenceStateScorer",
    "EvidenceStateScoreOutput",
]
