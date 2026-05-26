from __future__ import annotations

import torch
from torch import nn

from src.graph.segments import segment_logsumexp
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import ForwardPolicy
from src.weaver.reward import count_answers_in_state
from src.weaver.rollout.replay import WeakReplayBatch

from .output import ObjectiveOutput

Tensor = torch.Tensor


class WeakReplayLoss(nn.Module):
    """
    Weak shortest-path edge supervision over legal frontier actions.

    For each replay prefix state, all frontier edges marked by
    TargetContext.shortest_path_edge_mask are positives. The loss maximizes the
    total probability mass assigned to that positive set.
    """

    def __init__(
        self,
        *,
        weight: float = 1.0,
        gate_sufficient: bool = False,
        sufficient_recall_threshold: float = 1.0,
    ) -> None:
        super().__init__()
        if float(weight) < 0.0:
            raise ValueError("weight must be nonnegative.")
        if float(sufficient_recall_threshold) < 0.0 or float(sufficient_recall_threshold) > 1.0:
            raise ValueError("sufficient_recall_threshold must be in [0, 1].")
        self.weight = float(weight)
        self.gate_sufficient = bool(gate_sufficient)
        self.sufficient_recall_threshold = float(sufficient_recall_threshold)

    def forward(
        self,
        *,
        policy: ForwardPolicy,
        features: FeatureBank,
        graph_context: GraphContext,
        target_context: TargetContext,
        weak_replay: WeakReplayBatch,
    ) -> ObjectiveOutput:
        if weak_replay.num_states <= 0 or self.weight <= 0.0:
            zero = torch.zeros((), dtype=torch.float32, device=graph_context.device)
            return ObjectiveOutput(
                loss=zero,
                metrics={
                    "weak_replay/loss": zero,
                    "weak_replay/active_state_count": zero,
                    "weak_replay/active_state_count_before_gate": zero,
                    "weak_replay/gated_state_count": zero,
                    "weak_replay/sufficient_state_fraction": zero,
                    "weak_replay/positive_edge_count": zero,
                    "weak_replay/positive_mass": zero,
                },
                num_states=0,
                per_unit_loss=None,
            )

        state = weak_replay.state
        action_space = state.action_space(graph_context)
        if action_space.num_expansions <= 0:
            return _empty_output(graph_context.device)

        policy_out = policy(
            features=features,
            state=state,
            context=graph_context,
            action_space=action_space,
        )

        positive_edge = target_context.shortest_path_edge_mask.index_select(
            0,
            action_space.expand_edge_ids,
        )
        if not bool(positive_edge.any()):
            return _empty_output(graph_context.device)

        positive_rows = action_space.expand_state_ids[positive_edge]
        positive_log_prob = policy_out.edge_log_prob[positive_edge]
        positive_log_mass = segment_logsumexp(
            values=positive_log_prob,
            segment_ids=positive_rows,
            num_segments=int(state.num_states),
        )
        active = torch.isfinite(positive_log_mass)
        active_before_gate = active
        if self.gate_sufficient:
            sufficient = sufficient_state_mask(
                state=state,
                graph_context=graph_context,
                target_context=target_context,
                threshold=self.sufficient_recall_threshold,
            )
            active = active & ~sufficient
        else:
            sufficient = active.new_zeros(active.shape)

        if bool(active.any()):
            units = -positive_log_mass[active]
            loss = units.mean() * float(self.weight)
            positive_mass = positive_log_mass[active].exp().mean()
        else:
            units = positive_log_mass.new_empty((0,))
            loss = positive_log_mass.new_zeros(())
            positive_mass = positive_log_mass.new_zeros(())

        active_count = active.to(dtype=torch.float32).sum()
        active_count_before_gate = active_before_gate.to(dtype=torch.float32).sum()
        sufficient_count = (sufficient & active_before_gate).to(dtype=torch.float32).sum()
        positive_count = positive_edge.to(dtype=torch.float32).sum()

        return ObjectiveOutput(
            loss=loss,
            metrics={
                "weak_replay/loss": loss.detach(),
                "weak_replay/unit_loss": (units.mean() if units.numel() > 0 else loss.new_zeros(())).detach(),
                "weak_replay/active_state_count": active_count.detach(),
                "weak_replay/active_state_count_before_gate": active_count_before_gate.detach(),
                "weak_replay/gated_state_count": (active_count_before_gate - active_count).detach(),
                "weak_replay/sufficient_state_fraction": _safe_divide(
                    sufficient_count,
                    active_count_before_gate,
                ).detach(),
                "weak_replay/positive_edge_count": positive_count.detach(),
                "weak_replay/positive_mass": positive_mass.detach(),
            },
            num_states=int(active_count.item()),
            per_unit_loss=units.detach(),
        )


def _empty_output(device: torch.device) -> ObjectiveOutput:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return ObjectiveOutput(
        loss=zero,
        metrics={
            "weak_replay/loss": zero,
            "weak_replay/active_state_count": zero,
            "weak_replay/active_state_count_before_gate": zero,
            "weak_replay/gated_state_count": zero,
            "weak_replay/sufficient_state_fraction": zero,
            "weak_replay/positive_edge_count": zero,
            "weak_replay/positive_mass": zero,
        },
        num_states=0,
        per_unit_loss=None,
    )


def sufficient_state_mask(
    *,
    state,
    graph_context: GraphContext,
    target_context: TargetContext,
    threshold: float,
) -> Tensor:
    target_count = target_context.target_count_by_graph.index_select(
        0,
        state.graph_ids,
    ).to(dtype=torch.float)
    answer_count = count_answers_in_state(
        state=state,
        graph=graph_context,
        target_mask=target_context.target_mask,
    ).to(dtype=torch.float)
    recall = answer_count / target_count.clamp_min(1.0)
    return target_count.gt(0) & recall.ge(float(threshold))


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    if not bool(denominator.gt(0)):
        return numerator.new_zeros(())
    return numerator.float() / denominator.float()


__all__ = [
    "WeakReplayLoss",
]
