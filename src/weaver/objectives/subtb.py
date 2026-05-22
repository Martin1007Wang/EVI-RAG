from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.policy import ForwardPolicyOutput
from src.weaver.transition import (
    ExpansionBatch,
    SRC_POLICY,
    SRC_REPLAY,
    SRC_UNKNOWN,
    TerminalBatch,
)
from src.weaver.utility import RewardOutput

from .common import ObjectiveOutput


@dataclass(frozen=True, slots=True)
class SubTBInput:
    events: SubTBEventBatch


@dataclass(frozen=True, slots=True)
class SubTBEventBatch:
    trajectory_ids: torch.Tensor
    step_ids: torch.Tensor
    source_ids: torch.Tensor
    parent_state_log_flow: torch.Tensor
    child_state_log_flow: torch.Tensor
    action_log_prob: torch.Tensor
    action_log_flow: torch.Tensor
    backward_log_prob: torch.Tensor
    terminal_log_reward: torch.Tensor
    terminal: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.trajectory_ids.numel())
        fields = {
            "step_ids": self.step_ids,
            "source_ids": self.source_ids,
            "parent_state_log_flow": self.parent_state_log_flow,
            "child_state_log_flow": self.child_state_log_flow,
            "action_log_prob": self.action_log_prob,
            "action_log_flow": self.action_log_flow,
            "backward_log_prob": self.backward_log_prob,
            "terminal_log_reward": self.terminal_log_reward,
            "terminal": self.terminal,
        }
        for name, value in fields.items():
            if int(value.numel()) != count:
                raise ValueError(f"{name} must match trajectory_ids length.")

    @property
    def num_items(self) -> int:
        return int(self.trajectory_ids.numel())


@dataclass(frozen=True, slots=True)
class SubTBEvent:
    trajectory_id: int
    step_id: int
    source_id: torch.Tensor
    parent_state_log_flow: torch.Tensor
    child_state_log_flow: torch.Tensor
    action_log_prob: torch.Tensor
    action_log_flow: torch.Tensor
    backward_log_prob: torch.Tensor
    terminal_log_reward: torch.Tensor
    terminal: bool


@dataclass(frozen=True, slots=True)
class SubTBTerms:
    residual: torch.Tensor
    weight: torch.Tensor
    source_ids: torch.Tensor
    terminal: torch.Tensor
    length: torch.Tensor


class SubTBLoss(nn.Module):
    def __init__(
        self,
        *,
        subtb_lambda: float = 0.9,
        max_len: int | None = None,
        alpha_replay: float = 1.0,
        residual_loss: str = "huber",
        huber_delta: float = 2.0,
    ) -> None:
        super().__init__()
        self.subtb_lambda = float(subtb_lambda)
        self.max_len = None if max_len is None else int(max_len)
        self.alpha_replay = float(alpha_replay)
        self.residual_loss = residual_loss
        self.huber_delta = float(huber_delta)

    def forward(self, x: SubTBInput) -> ObjectiveOutput:
        terms = subtrajectory_terms(
            x,
            subtb_lambda=self.subtb_lambda,
            max_len=self.max_len,
        )
        units = residual_loss_units(
            terms.residual,
            loss_type=self.residual_loss,
            huber_delta=self.huber_delta,
        )
        loss = weighted_source_balanced_mean(
            values=units,
            weights=terms.weight,
            source_ids=terms.source_ids,
            alpha_replay=self.alpha_replay,
        )

        terminal = terms.terminal
        nonterminal = ~terminal
        metrics = {
            "subtb_residual_mean": mean_or_zero(terms.residual).detach(),
            "subtb_residual_abs_mean": mean_or_zero(terms.residual.abs()).detach(),
            "subtb_terminal_residual_mean": masked_mean_or_zero(terms.residual, terminal).detach(),
            "subtb_nonterminal_residual_mean": masked_mean_or_zero(terms.residual, nonterminal).detach(),
            "subtb_num_segments": terms.residual.new_tensor(float(terms.residual.numel())).detach(),
            "subtb_length_mean": mean_or_zero(terms.length.float()).detach(),
            "subtb_terminal_fraction": masked_fraction(
                terminal,
                torch.ones_like(terminal, dtype=torch.bool),
            ).detach(),
            "subtb_lambda": terms.residual.new_tensor(float(self.subtb_lambda)).detach(),
            "source_policy_num_segments": terms.residual.new_tensor(
                float(
                    (
                        terms.source_ids.eq(SRC_POLICY)
                        | terms.source_ids.eq(SRC_UNKNOWN)
                    ).sum().item()
                )
            ).detach(),
            "source_replay_num_segments": terms.residual.new_tensor(
                float(terms.source_ids.eq(SRC_REPLAY).sum().item())
            ).detach(),
        }
        metrics.update(branch_decomposition_metrics(x))
        metrics.update(source_subtb_metrics(
            terms=terms,
            units=units,
        ))
        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_states=x.events.num_items,
            per_unit_loss=units.detach(),
        )


def build_subtb_input(
    *,
    parent_out: ForwardPolicyOutput,
    child_out: ForwardPolicyOutput,
    terminal_out: ForwardPolicyOutput,
    reward_out: RewardOutput,
    backward_log_prob: torch.Tensor,
    expansions: ExpansionBatch,
    terminals: TerminalBatch,
) -> SubTBInput:
    expansion_rows = torch.arange(
        expansions.num_items,
        dtype=torch.long,
        device=expansions.device,
    )
    expansion_parent_state_log_flow = parent_out.state_log_flow.float()
    expansion_child_state_log_flow = child_out.state_log_flow.float()
    expansion_action_log_prob = parent_out.gather_log_prob(
        row_ids=expansion_rows,
        edge_ids=expansions.edge_ids,
    ).float()
    expansion_action_log_flow = parent_out.gather_action_log_flow(
        row_ids=expansion_rows,
        edge_ids=expansions.edge_ids,
    ).float()
    expansion_terminal_log_reward = expansion_parent_state_log_flow.new_zeros(expansions.num_items)
    expansion_terminal = torch.zeros(
        expansions.num_items,
        dtype=torch.bool,
        device=expansions.device,
    )

    terminal_parent_state_log_flow = terminal_out.state_log_flow.float()
    terminal_child_state_log_flow = terminal_parent_state_log_flow.new_zeros(terminals.num_items)
    terminal_action_log_prob = terminal_out.stop_log_prob.float()
    terminal_action_log_flow = terminal_out.stop_log_flow.float()
    terminal_backward_log_prob = terminal_parent_state_log_flow.new_zeros(terminals.num_items)
    terminal_terminal = torch.ones(
        terminals.num_items,
        dtype=torch.bool,
        device=terminals.device,
    )

    return SubTBInput(
        events=SubTBEventBatch(
            trajectory_ids=torch.cat(
                [
                    expansions.meta.trajectory_ids,
                    terminals.meta.trajectory_ids,
                ],
                dim=0,
            ),
            step_ids=torch.cat(
                [
                    expansions.meta.step_ids,
                    terminals.meta.step_ids,
                ],
                dim=0,
            ),
            source_ids=torch.cat(
                [
                    expansions.meta.source_ids,
                    terminals.meta.source_ids,
                ],
                dim=0,
            ),
            parent_state_log_flow=torch.cat(
                [
                    expansion_parent_state_log_flow,
                    terminal_parent_state_log_flow,
                ],
                dim=0,
            ),
            child_state_log_flow=torch.cat(
                [
                    expansion_child_state_log_flow,
                    terminal_child_state_log_flow,
                ],
                dim=0,
            ),
            action_log_prob=torch.cat(
                [
                    expansion_action_log_prob,
                    terminal_action_log_prob,
                ],
                dim=0,
            ),
            action_log_flow=torch.cat(
                [
                    expansion_action_log_flow,
                    terminal_action_log_flow,
                ],
                dim=0,
            ),
            backward_log_prob=torch.cat(
                [
                    backward_log_prob.float(),
                    terminal_backward_log_prob,
                ],
                dim=0,
            ),
            terminal_log_reward=torch.cat(
                [
                    expansion_terminal_log_reward,
                    reward_out.log_reward.float(),
                ],
                dim=0,
            ),
            terminal=torch.cat(
                [
                    expansion_terminal,
                    terminal_terminal,
                ],
                dim=0,
            ),
        ),
    )


def single_step_branch_losses(
    x: SubTBInput,
    *,
    loss_type: str,
    huber_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    exp_residual = expansion_event_residual(x.events)
    terminal_residual = terminal_event_residual(x.events)
    exp_mask = ~x.events.terminal
    terminal_mask = x.events.terminal
    exp_units = residual_loss_units(
        exp_residual,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )
    terminal_units = residual_loss_units(
        terminal_residual,
        loss_type=loss_type,
        huber_delta=huber_delta,
    )
    return (
        masked_mean_or_zero(exp_units, exp_mask),
        masked_mean_or_zero(terminal_units, terminal_mask),
    )


def expansion_event_residual(events: SubTBEventBatch) -> torch.Tensor:
    return (
        events.action_log_flow
        - events.child_state_log_flow
        - events.backward_log_prob
    )


def terminal_event_residual(events: SubTBEventBatch) -> torch.Tensor:
    return events.action_log_flow - events.terminal_log_reward


def branch_decomposition_metrics(x: SubTBInput) -> dict[str, torch.Tensor]:
    events = x.events
    exp = ~events.terminal
    terminal = events.terminal
    exp_residual = expansion_event_residual(events)
    terminal_residual = terminal_event_residual(events)

    return {
        "subtb_exp_parent_state_log_flow_mean": masked_mean_or_zero(events.parent_state_log_flow, exp).detach(),
        "subtb_exp_child_state_log_flow_mean": masked_mean_or_zero(events.child_state_log_flow, exp).detach(),
        "subtb_exp_action_log_flow_mean": masked_mean_or_zero(events.action_log_flow, exp).detach(),
        "subtb_exp_action_log_prob_mean": masked_mean_or_zero(events.action_log_prob, exp).detach(),
        "subtb_exp_backward_log_prob_mean": masked_mean_or_zero(events.backward_log_prob, exp).detach(),
        "subtb_exp_db_residual_mean": masked_mean_or_zero(exp_residual, exp).detach(),
        "subtb_exp_db_residual_abs_mean": masked_mean_or_zero(exp_residual.abs(), exp).detach(),
        "subtb_terminal_parent_state_log_flow_mean": masked_mean_or_zero(events.parent_state_log_flow, terminal).detach(),
        "subtb_terminal_action_log_flow_mean": masked_mean_or_zero(events.action_log_flow, terminal).detach(),
        "subtb_terminal_log_reward_mean": masked_mean_or_zero(events.terminal_log_reward, terminal).detach(),
        "subtb_terminal_db_residual_mean": masked_mean_or_zero(terminal_residual, terminal).detach(),
        "subtb_terminal_db_residual_abs_mean": masked_mean_or_zero(terminal_residual.abs(), terminal).detach(),
    }


def source_subtb_metrics(
    *,
    terms: SubTBTerms,
    units: torch.Tensor,
) -> dict[str, torch.Tensor]:
    policy = terms.source_ids.eq(SRC_POLICY) | terms.source_ids.eq(SRC_UNKNOWN)
    replay = terms.source_ids.eq(SRC_REPLAY)
    policy_loss = weighted_mean_or_zero(units, terms.weight, policy)
    replay_loss = weighted_mean_or_zero(units, terms.weight, replay)
    return {
        "subtb_policy_residual_abs_mean": masked_mean_or_zero(terms.residual.abs(), policy).detach(),
        "subtb_replay_residual_abs_mean": masked_mean_or_zero(terms.residual.abs(), replay).detach(),
        "subtb_policy_terminal_fraction": masked_fraction(terms.terminal & policy, policy).detach(),
        "subtb_replay_terminal_fraction": masked_fraction(terms.terminal & replay, replay).detach(),
        "subtb_policy_loss": policy_loss.detach(),
        "subtb_replay_loss": replay_loss.detach(),
    }


def subtrajectory_terms(
    x: SubTBInput,
    *,
    subtb_lambda: float,
    max_len: int | None,
) -> SubTBTerms:
    events = assemble_events(x)
    if not events:
        return empty_terms(x.events.parent_state_log_flow, x.events.source_ids)

    residuals: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    sources: list[torch.Tensor] = []
    terminal_flags: list[bool] = []
    lengths: list[int] = []

    by_trajectory: dict[int, list[SubTBEvent]] = {}
    for event in events:
        by_trajectory.setdefault(event.trajectory_id, []).append(event)

    for event_list in by_trajectory.values():
        event_list.sort(key=lambda item: item.step_id)
        n = len(event_list)

        for start_pos in range(n):
            start = event_list[start_pos]
            running_forward = x.events.parent_state_log_flow.new_zeros(())
            running_backward = x.events.parent_state_log_flow.new_zeros(())
            previous_step: int | None = None
            end_limit = n if max_len is None else min(n, start_pos + int(max_len))

            for end_pos in range(start_pos, end_limit):
                event = event_list[end_pos]
                step = event.step_id
                if previous_step is not None and step != previous_step + 1:
                    break
                previous_step = step

                running_forward = running_forward + event.action_log_prob
                is_terminal = event.terminal
                if not is_terminal:
                    running_backward = running_backward + event.backward_log_prob

                length = end_pos - start_pos + 1
                weight = x.events.parent_state_log_flow.new_tensor(float(subtb_lambda) ** float(length - 1))

                if is_terminal:
                    residual = (
                        start.parent_state_log_flow
                        + running_forward
                        - event.terminal_log_reward
                        - running_backward
                    )
                else:
                    residual = (
                        start.parent_state_log_flow
                        + running_forward
                        - event.child_state_log_flow
                        - running_backward
                    )

                residuals.append(residual)
                weights.append(weight)
                sources.append(start.source_id)
                terminal_flags.append(is_terminal)
                lengths.append(length)

                if is_terminal:
                    break

    if not residuals:
        return empty_terms(x.events.parent_state_log_flow, x.events.source_ids)

    return SubTBTerms(
        residual=torch.stack(residuals),
        weight=torch.stack(weights),
        source_ids=torch.stack(sources),
        terminal=torch.tensor(terminal_flags, dtype=torch.bool, device=x.events.parent_state_log_flow.device),
        length=torch.tensor(lengths, dtype=torch.long, device=x.events.parent_state_log_flow.device),
    )


def assemble_events(x: SubTBInput) -> list[SubTBEvent]:
    events: list[SubTBEvent] = []
    batch = x.events
    for idx in range(batch.num_items):
        events.append(
            SubTBEvent(
                trajectory_id=int(batch.trajectory_ids[idx].item()),
                step_id=int(batch.step_ids[idx].item()),
                source_id=batch.source_ids[idx],
                parent_state_log_flow=batch.parent_state_log_flow[idx],
                child_state_log_flow=batch.child_state_log_flow[idx],
                action_log_prob=batch.action_log_prob[idx],
                action_log_flow=batch.action_log_flow[idx],
                backward_log_prob=batch.backward_log_prob[idx],
                terminal_log_reward=batch.terminal_log_reward[idx],
                terminal=bool(batch.terminal[idx].item()),
            )
        )

    return events


def empty_terms(
    values: torch.Tensor,
    source_ids: torch.Tensor,
) -> SubTBTerms:
    return SubTBTerms(
        residual=values.new_empty((0,)),
        weight=values.new_empty((0,)),
        source_ids=source_ids.new_empty((0,)),
        terminal=torch.empty(0, dtype=torch.bool, device=values.device),
        length=torch.empty(0, dtype=torch.long, device=values.device),
    )


def residual_loss_units(
    residual: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    if loss_type == "mse":
        return residual.square()

    delta = float(huber_delta)
    abs_residual = residual.abs()
    return torch.where(
        abs_residual <= delta,
        0.5 * residual.square(),
        delta * (abs_residual - 0.5 * delta),
    )


def weighted_source_balanced_mean(
    *,
    values: torch.Tensor,
    weights: torch.Tensor,
    source_ids: torch.Tensor,
    alpha_replay: float,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())

    valid = weights.gt(0)
    policy_mask = (
        source_ids.eq(SRC_POLICY)
        | source_ids.eq(SRC_UNKNOWN)
    ) & valid
    replay_mask = source_ids.eq(SRC_REPLAY) & valid

    policy_loss = weighted_mean_or_zero(values, weights, policy_mask)
    replay_loss = weighted_mean_or_zero(values, weights, replay_mask)

    if not bool(policy_mask.any()):
        return float(alpha_replay) * replay_loss
    if not bool(replay_mask.any()):
        return policy_loss
    return policy_loss + float(alpha_replay) * replay_loss


def weighted_mean_or_zero(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not bool(mask.any()):
        return values.new_zeros(())
    selected_weights = weights[mask]
    total_weight = selected_weights.sum()
    if not bool(total_weight.gt(0)):
        return values.new_zeros(())
    return (values[mask] * selected_weights).sum() / total_weight


def masked_mean_or_zero(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.numel() == 0 or not bool(mask.any()):
        return values.new_zeros(())
    return values[mask].float().mean()


def masked_fraction(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    count = denominator.float().sum()
    if not bool(count.gt(0)):
        return count.new_zeros(())
    return numerator.float().sum() / count


def mean_or_zero(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros(())
    return values.float().mean()


__all__ = [
    "build_subtb_input",
    "SubTBEvent",
    "SubTBEventBatch",
    "SubTBInput",
    "SubTBLoss",
    "SubTBTerms",
    "mean_or_zero",
    "masked_fraction",
    "masked_mean_or_zero",
    "residual_loss_units",
    "single_step_branch_losses",
    "subtrajectory_terms",
    "weighted_source_balanced_mean",
]
