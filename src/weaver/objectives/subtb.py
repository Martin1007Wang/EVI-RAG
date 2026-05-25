from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.policy import PolicyOutput
from src.weaver.policy.backward import deterministic_backward_log_prob
from src.weaver.rollout.trajectory import (
    SRC_POLICY,
    SRC_REPLAY,
    TrajectoryBatch,
)
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import ForwardPolicy
from src.weaver.state import ActionSpace
from src.weaver.objectives.prefix import (
    ExpansionPrefixBatch,
    PrefixBatch,
    TerminalPrefixBatch,
    build_prefix_batch,
)
from src.weaver.reward import RewardOutput
from src.weaver.reward import TerminalRecallReward

from .output import ObjectiveOutput

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class SubTBInput:
    events: SubTBEventBatch
    metrics: dict[str, Tensor] | None = None


@dataclass(frozen=True, slots=True)
class SubTBEventBatch:
    """
    Sparse event batch for SubTB.

    Each item is either:

        expansion:
            z_t --e_t--> z_{t+1}

        stop:
            z_t --STOP--> terminal reward R(z_t)

    Stored quantities:
    - parent_state_log_flow = Φ(z_t)
    - child_state_log_flow  = Φ(z_{t+1}) for expansion, unused for STOP
    - action_log_flow       = G(z_t, a_t)
    - action_log_prob       = G(z_t, a_t) - Φ(z_t), derived
    - backward_log_prob     = log P_B(z_t | z_{t+1}) for expansion, 0 for STOP
    - terminal_log_reward   = log R(z_t) for STOP, 0 for expansion
    """

    trajectory_ids: Tensor  # [N]
    step_ids: Tensor  # [N]
    source_ids: Tensor  # [N]

    parent_state_log_flow: Tensor  # [N]
    child_state_log_flow: Tensor  # [N]
    action_log_flow: Tensor  # [N]
    backward_log_prob: Tensor  # [N]
    terminal_log_reward: Tensor  # [N]

    is_stop: Tensor  # [N], bool

    @property
    def device(self) -> torch.device:
        return self.trajectory_ids.device

    @property
    def num_items(self) -> int:
        return int(self.trajectory_ids.numel())

    @property
    def action_log_prob(self) -> Tensor:
        return self.action_log_flow - self.parent_state_log_flow


@dataclass(frozen=True, slots=True)
class SubTBTerms:
    residual: Tensor
    weight: Tensor
    source_ids: Tensor
    is_stop: Tensor


class SubTBLoss(nn.Module):
    """
    Subtrajectory Balance loss over reconstructed rollout events.

    Residual:

        non-terminal:
            Φ(z_i) + Σ log P_F(a_t | z_t)
            - Φ(z_j)
            - Σ log P_B(z_t | z_{t+1})

        terminal:
            Φ(z_i) + Σ log P_F(a_t | z_t)
            - log R(z_j)
            - Σ log P_B(z_t | z_{t+1})

    DB is recovered by max_len=1.
    TB is approximated by max_len=None with full trajectory segments.
    """

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

        if residual_loss not in {"huber", "mse"}:
            raise ValueError(f"Unsupported residual_loss: {residual_loss!r}")

        self.subtb_lambda = float(subtb_lambda)
        self.max_len = None if max_len is None else int(max_len)
        self.alpha_replay = float(alpha_replay)
        self.residual_loss = str(residual_loss)
        self.huber_delta = float(huber_delta)

    def forward(self, x: SubTBInput) -> ObjectiveOutput:
        terms = subtrajectory_terms(
            x.events,
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

        active = terms.weight.gt(0)
        exp = (~terms.is_stop) & active
        stop = terms.is_stop & active

        metrics = {
            "subtb/loss": loss.detach(),
            "subtb/unit_loss": weighted_mean_or_zero(
                units,
                terms.weight,
                active,
            ).detach(),
            "subtb/residual_abs": weighted_mean_or_zero(
                terms.residual.abs(),
                terms.weight,
                active,
            ).detach(),
            "subtb/exp_residual_abs": weighted_mean_or_zero(
                terms.residual.abs(),
                terms.weight,
                exp,
            ).detach(),
            "subtb/stop_residual_abs": weighted_mean_or_zero(
                terms.residual.abs(),
                terms.weight,
                stop,
            ).detach(),
        }

        metrics.update(single_step_metrics(x.events))
        metrics.update(source_metrics(terms=terms, units=units))

        if x.metrics:
            metrics.update(x.metrics)

        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_states=x.events.num_items,
            per_unit_loss=units.detach(),
        )


class SubTBObjective(nn.Module):
    """
    High-level objective wrapper from trajectories to SubTB loss.

    Rollout owns trajectory sampling. This module owns loss-facing prefix
    reconstruction and all policy/reward evaluations required by SubTB.
    """

    def __init__(
        self,
        *,
        subtb_lambda: float = 0.9,
        max_len: int | None = None,
        alpha_replay: float = 1.0,
        residual_loss: str = "huber",
        huber_delta: float = 2.0,
        validate_backward: bool = False,
    ) -> None:
        super().__init__()
        self.loss = SubTBLoss(
            subtb_lambda=subtb_lambda,
            max_len=max_len,
            alpha_replay=alpha_replay,
            residual_loss=residual_loss,
            huber_delta=huber_delta,
        )
        self.validate_backward = bool(validate_backward)

    def forward(
        self,
        *,
        policy: ForwardPolicy,
        reward_model: TerminalRecallReward,
        features: FeatureBank,
        graph_context: GraphContext,
        target_context: TargetContext,
        trajectories: TrajectoryBatch,
    ) -> ObjectiveOutput:
        prefix = build_prefix_batch(trajectories)
        subtb_input = build_subtb_input_from_prefix(
            policy=policy,
            reward_model=reward_model,
            features=features,
            graph_context=graph_context,
            target_context=target_context,
            prefix=prefix,
            validate_backward=self.validate_backward,
        )
        return self.loss(subtb_input)


def build_subtb_input_from_prefix(
    *,
    policy: ForwardPolicy,
    reward_model: TerminalRecallReward,
    features: FeatureBank,
    graph_context: GraphContext,
    target_context: TargetContext,
    prefix: PrefixBatch,
    validate_backward: bool = False,
) -> SubTBInput:
    expansions = prefix.expansions
    terminals = prefix.terminals

    if expansions.num_items > 0:
        parent_out = policy(
            features=features,
            state=expansions.parent,
            context=graph_context,
            action_space=expansions.parent.action_space(graph_context),
        )
        child_out = policy(
            features=features,
            state=expansions.child,
            context=graph_context,
            action_space=expansions.child.action_space(graph_context),
        )
        backward_log_prob = deterministic_backward_log_prob(
            parent_state=expansions.parent,
            child_state=expansions.child,
            action_edge_ids=expansions.edge_ids,
            validate=bool(validate_backward),
        )
    else:
        parent_out = _empty_policy_output(device=terminals.device)
        child_out = _empty_policy_output(device=terminals.device)
        backward_log_prob = torch.empty(
            0,
            dtype=torch.float32,
            device=terminals.device,
        )

    if terminals.num_items > 0:
        terminal_out = policy(
            features=features,
            state=terminals.state,
            context=graph_context,
            action_space=terminals.state.action_space(graph_context),
        )
        terminal_reward = reward_model(
            state=terminals.state,
            graph_context=graph_context,
            target_context=target_context,
        )
    else:
        terminal_out = _empty_policy_output(device=expansions.device)
        terminal_reward = _empty_reward_output(device=expansions.device)

    subtb_input = build_subtb_input(
        parent_out=parent_out,
        child_out=child_out,
        terminal_out=terminal_out,
        terminal_reward=terminal_reward,
        backward_log_prob=backward_log_prob,
        expansions=expansions,
        terminals=terminals,
    )

    return SubTBInput(
        events=subtb_input.events,
        metrics=terminal_metrics(
            terminal_reward=terminal_reward,
            terminals=terminals,
        ),
    )


def build_subtb_input(
    *,
    parent_out: PolicyOutput,
    child_out: PolicyOutput,
    terminal_out: PolicyOutput,
    terminal_reward: RewardOutput,
    backward_log_prob: Tensor,
    expansions: ExpansionPrefixBatch,
    terminals: TerminalPrefixBatch,
) -> SubTBInput:
    """
    Build SubTB events from transition states and policy outputs.

    STOP uses terminal_out.stop_log_flow directly.
    Do not represent STOP as a fake KG edge id.
    """

    exp_events = _build_expansion_events(
        parent_out=parent_out,
        child_out=child_out,
        backward_log_prob=backward_log_prob,
        expansions=expansions,
    )

    stop_events = _build_terminal_events(
        terminal_out=terminal_out,
        terminal_reward=terminal_reward,
        terminals=terminals,
    )

    return SubTBInput(
        events=_cat_event_batches(
            [exp_events, stop_events],
            device=_event_device(expansions, terminals),
        )
    )


def _build_expansion_events(
    *,
    parent_out: PolicyOutput,
    child_out: PolicyOutput,
    backward_log_prob: Tensor,
    expansions: ExpansionPrefixBatch,
) -> SubTBEventBatch:
    device = expansions.device
    n = expansions.num_items

    rows = torch.arange(
        n,
        dtype=torch.long,
        device=device,
    )

    parent_state_log_flow = parent_out.state_log_flow.float()
    child_state_log_flow = child_out.state_log_flow.float()

    action_log_flow = parent_out.gather_action_log_flow(
        row_ids=rows,
        edge_ids=expansions.edge_ids,
    ).float()

    return SubTBEventBatch(
        trajectory_ids=expansions.traj_ids,
        step_ids=expansions.step_ids,
        source_ids=expansions.source,
        parent_state_log_flow=parent_state_log_flow,
        child_state_log_flow=child_state_log_flow,
        action_log_flow=action_log_flow,
        backward_log_prob=backward_log_prob.to(device=device).float().view(-1),
        terminal_log_reward=torch.zeros(
            n,
            dtype=parent_state_log_flow.dtype,
            device=device,
        ),
        is_stop=torch.zeros(
            n,
            dtype=torch.bool,
            device=device,
        ),
    )


def _build_terminal_events(
    *,
    terminal_out: PolicyOutput,
    terminal_reward: RewardOutput,
    terminals: TerminalPrefixBatch,
) -> SubTBEventBatch:
    device = terminals.device
    n_all = terminals.num_items

    if n_all == 0:
        return _empty_events(device=device)

    valid = terminal_reward.valid_mask.to(
        device=device,
        dtype=torch.bool,
    ).view(-1)

    if int(valid.numel()) != n_all:
        raise ValueError("terminal_reward.valid_mask must have one item per terminal action.")

    rows = valid.nonzero(as_tuple=False).flatten()
    n = int(rows.numel())

    if n == 0:
        return _empty_events(device=device)

    parent_state_log_flow = terminal_out.state_log_flow.index_select(
        0,
        rows,
    ).float()

    action_log_flow = terminal_out.stop_log_flow.index_select(
        0,
        rows,
    ).float()

    return SubTBEventBatch(
        trajectory_ids=terminals.traj_ids.index_select(0, rows),
        step_ids=terminals.step_ids.index_select(0, rows),
        source_ids=terminals.source.index_select(0, rows),
        parent_state_log_flow=parent_state_log_flow,
        child_state_log_flow=torch.zeros(
            n,
            dtype=parent_state_log_flow.dtype,
            device=device,
        ),
        action_log_flow=action_log_flow,
        backward_log_prob=torch.zeros(
            n,
            dtype=parent_state_log_flow.dtype,
            device=device,
        ),
        terminal_log_reward=terminal_reward.log_reward.index_select(
            0,
            rows,
        ).float(),
        is_stop=torch.ones(
            n,
            dtype=torch.bool,
            device=device,
        ),
    )


def subtrajectory_terms(
    events: SubTBEventBatch,
    *,
    subtb_lambda: float,
    max_len: int | None,
) -> SubTBTerms:
    """
    Vectorized SubTB segment construction.

    Events are sparse, but trajectory ids and step ids define a dense
    trajectory-step table. Since budget is small, the loop is over segment
    length, not over individual Python event objects.
    """

    if events.num_items == 0:
        return _empty_terms(events)

    num_trajectories = int(events.trajectory_ids.max().item()) + 1
    num_steps = int(events.step_ids.max().item()) + 1

    device = events.device
    dtype = events.parent_state_log_flow.dtype
    shape = (num_trajectories, num_steps)

    has_event = torch.zeros(shape, dtype=torch.bool, device=device)
    is_stop = torch.zeros(shape, dtype=torch.bool, device=device)
    source_ids = torch.full(shape, -1, dtype=torch.long, device=device)

    parent_flow = torch.zeros(shape, dtype=dtype, device=device)
    child_flow = torch.zeros(shape, dtype=dtype, device=device)
    action_log_prob = torch.zeros(shape, dtype=dtype, device=device)
    backward_log_prob = torch.zeros(shape, dtype=dtype, device=device)
    terminal_log_reward = torch.zeros(shape, dtype=dtype, device=device)

    traj = events.trajectory_ids
    step = events.step_ids

    has_event[traj, step] = True
    is_stop[traj, step] = events.is_stop
    source_ids[traj, step] = events.source_ids
    parent_flow[traj, step] = events.parent_state_log_flow
    child_flow[traj, step] = events.child_state_log_flow
    action_log_prob[traj, step] = events.action_log_prob
    backward_log_prob[traj, step] = events.backward_log_prob
    terminal_log_reward[traj, step] = events.terminal_log_reward

    action_prefix = _exclusive_prefix_sum(action_log_prob)
    backward_prefix = _exclusive_prefix_sum(backward_log_prob)
    event_prefix = _exclusive_prefix_sum(has_event.long()).long()

    residuals: list[Tensor] = []
    weights: list[Tensor] = []
    sources: list[Tensor] = []
    stop_flags: list[Tensor] = []

    max_segment_len = num_steps if max_len is None else min(num_steps, int(max_len))

    for length in range(1, max_segment_len + 1):
        start = torch.arange(
            0,
            num_steps - length + 1,
            dtype=torch.long,
            device=device,
        )
        end = start + length - 1

        forward_sum = action_prefix[:, end + 1] - action_prefix[:, start]
        backward_sum = backward_prefix[:, end + 1] - backward_prefix[:, start]
        event_count = event_prefix[:, end + 1] - event_prefix[:, start]

        valid = event_count.eq(length) & has_event[:, start]

        if not bool(valid.any()):
            continue

        tail = torch.where(
            is_stop[:, end],
            terminal_log_reward[:, end],
            child_flow[:, end],
        )

        residual = parent_flow[:, start] + forward_sum - tail - backward_sum
        weight = residual.new_full(
            residual.shape,
            float(subtb_lambda) ** float(length - 1),
        )

        residuals.append(residual[valid])
        weights.append(weight[valid])
        sources.append(source_ids[:, start][valid])
        stop_flags.append(is_stop[:, end][valid])

    if not residuals:
        return _empty_terms(events)

    return SubTBTerms(
        residual=torch.cat(residuals, dim=0),
        weight=torch.cat(weights, dim=0),
        source_ids=torch.cat(sources, dim=0),
        is_stop=torch.cat(stop_flags, dim=0),
    )


def single_step_metrics(events: SubTBEventBatch) -> dict[str, Tensor]:
    exp = ~events.is_stop
    stop = events.is_stop

    exp_residual = expansion_db_residual(events)
    stop_residual = stop_db_residual(events)

    return {
        "db/exp_residual_abs": masked_mean_or_zero(
            exp_residual.abs(),
            exp,
        ).detach(),
        "db/stop_residual_abs": masked_mean_or_zero(
            stop_residual.abs(),
            stop,
        ).detach(),
        "flow/terminal_log_reward": masked_mean_or_zero(
            events.terminal_log_reward,
            stop,
        ).detach(),
    }


def expansion_db_residual(events: SubTBEventBatch) -> Tensor:
    """
    One-step DB residual for expansion:

        G(z,e) - Φ(z') - log P_B(z | z')
    """

    return events.action_log_flow - events.child_state_log_flow - events.backward_log_prob


def stop_db_residual(events: SubTBEventBatch) -> Tensor:
    """
    One-step terminal residual for STOP:

        G(z,STOP) - log R(z)
    """

    return events.action_log_flow - events.terminal_log_reward


def source_metrics(
    *,
    terms: SubTBTerms,
    units: Tensor,
) -> dict[str, Tensor]:
    active = terms.weight.gt(0)
    policy = terms.source_ids.eq(SRC_POLICY) & active
    replay = terms.source_ids.eq(SRC_REPLAY) & active

    return {
        "source/policy_loss": weighted_mean_or_zero(
            units,
            terms.weight,
            policy,
        ).detach(),
        "source/replay_loss": weighted_mean_or_zero(
            units,
            terms.weight,
            replay,
        ).detach(),
        "source/policy_residual_abs": weighted_mean_or_zero(
            terms.residual.abs(),
            terms.weight,
            policy,
        ).detach(),
        "source/replay_residual_abs": weighted_mean_or_zero(
            terms.residual.abs(),
            terms.weight,
            replay,
        ).detach(),
    }


def terminal_metrics(
    *,
    terminal_reward: RewardOutput,
    terminals: TerminalPrefixBatch,
) -> dict[str, Tensor]:
    valid = terminal_reward.valid_mask.to(
        device=terminals.device,
        dtype=torch.bool,
    )

    replay = terminals.source.eq(SRC_REPLAY) & valid

    hit = terminal_reward.answer_count.gt(0).to(dtype=torch.float)

    return {
        "terminal/hit": masked_mean_or_zero(
            hit,
            valid,
        ).detach(),
        "terminal/recall": masked_mean_or_zero(
            terminal_reward.target_recall,
            valid,
        ).detach(),
        "terminal/replay_hit": masked_mean_or_zero(
            hit,
            replay,
        ).detach(),
        "terminal/replay_recall": masked_mean_or_zero(
            terminal_reward.target_recall,
            replay,
        ).detach(),
    }


def weighted_source_balanced_mean(
    *,
    values: Tensor,
    weights: Tensor,
    source_ids: Tensor,
    alpha_replay: float,
) -> Tensor:
    if values.numel() == 0:
        return values.new_zeros(())

    valid = weights.gt(0)

    policy = source_ids.eq(SRC_POLICY) & valid
    replay = source_ids.eq(SRC_REPLAY) & valid

    parts: list[Tensor] = []

    if bool(policy.any()):
        parts.append(weighted_mean_or_zero(values, weights, policy))

    if bool(replay.any()):
        parts.append(float(alpha_replay) * weighted_mean_or_zero(values, weights, replay))

    if not parts:
        return values.new_zeros(())

    return torch.stack(parts).sum()


def residual_loss_units(
    residual: Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> Tensor:
    if loss_type == "mse":
        return residual.square()

    if loss_type != "huber":
        raise ValueError(f"Unsupported residual loss: {loss_type!r}")

    delta = float(huber_delta)
    abs_residual = residual.abs()

    return torch.where(
        abs_residual <= delta,
        0.5 * residual.square(),
        delta * (abs_residual - 0.5 * delta),
    )


def weighted_mean_or_zero(
    values: Tensor,
    weights: Tensor,
    mask: Tensor,
) -> Tensor:
    if values.numel() == 0:
        return values.new_zeros(())

    mask = mask.to(device=values.device, dtype=torch.bool).view(-1)

    if not bool(mask.any()):
        return values.new_zeros(())

    masked_weights = weights[mask].float()
    denom = masked_weights.sum()

    if not bool(denom.gt(0)):
        return values.new_zeros(())

    return (values[mask].float() * masked_weights).sum() / denom


def masked_mean_or_zero(
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    if values.numel() == 0:
        return values.new_zeros(())

    mask = mask.to(device=values.device, dtype=torch.bool).view(-1)

    if not bool(mask.any()):
        return values.new_zeros(())

    return values.float()[mask].mean()


def _exclusive_prefix_sum(x: Tensor) -> Tensor:
    zero = x.new_zeros((x.size(0), 1))
    return torch.cat(
        [
            zero,
            torch.cumsum(x, dim=1),
        ],
        dim=1,
    )


def _cat_event_batches(
    batches: list[SubTBEventBatch],
    *,
    device: torch.device,
) -> SubTBEventBatch:
    non_empty = [batch for batch in batches if batch.num_items > 0]

    if not non_empty:
        return _empty_events(device=device)

    return SubTBEventBatch(
        trajectory_ids=torch.cat([x.trajectory_ids for x in non_empty], dim=0),
        step_ids=torch.cat([x.step_ids for x in non_empty], dim=0),
        source_ids=torch.cat([x.source_ids for x in non_empty], dim=0),
        parent_state_log_flow=torch.cat(
            [x.parent_state_log_flow for x in non_empty],
            dim=0,
        ),
        child_state_log_flow=torch.cat(
            [x.child_state_log_flow for x in non_empty],
            dim=0,
        ),
        action_log_flow=torch.cat([x.action_log_flow for x in non_empty], dim=0),
        backward_log_prob=torch.cat(
            [x.backward_log_prob for x in non_empty],
            dim=0,
        ),
        terminal_log_reward=torch.cat(
            [x.terminal_log_reward for x in non_empty],
            dim=0,
        ),
        is_stop=torch.cat([x.is_stop for x in non_empty], dim=0),
    )


def _empty_events(
    *,
    device: torch.device,
) -> SubTBEventBatch:
    long = torch.empty(0, dtype=torch.long, device=device)
    float_ = torch.empty(0, dtype=torch.float32, device=device)
    bool_ = torch.empty(0, dtype=torch.bool, device=device)

    return SubTBEventBatch(
        trajectory_ids=long,
        step_ids=long,
        source_ids=long,
        parent_state_log_flow=float_,
        child_state_log_flow=float_,
        action_log_flow=float_,
        backward_log_prob=float_,
        terminal_log_reward=float_,
        is_stop=bool_,
    )


def _empty_policy_output(
    *,
    device: torch.device,
) -> PolicyOutput:
    float_ = torch.empty(0, dtype=torch.float32, device=device)
    return PolicyOutput(
        action_space=ActionSpace.empty(num_states=0, device=device),
        state_log_flow=float_,
        stop_log_flow=float_,
        continue_log_flow=float_,
        stop_log_prob=float_,
        continue_log_prob=float_,
        edge_log_flow=float_,
        edge_log_prob=float_,
        conditional_edge_log_prob=float_,
        edge_raw_score=float_,
    )


def _empty_reward_output(
    *,
    device: torch.device,
) -> RewardOutput:
    float_ = torch.empty(0, dtype=torch.float32, device=device)
    bool_ = torch.empty(0, dtype=torch.bool, device=device)

    return RewardOutput(
        log_reward=float_,
        raw_log_reward=float_,
        answer_count=float_,
        target_count=float_,
        target_recall=float_,
        target_proximity=float_,
        path_edge_count=float_,
        path_edge_precision=float_,
        edge_count=float_,
        valid_mask=bool_,
        success_mask=bool_,
        metrics={},
    )


def _empty_terms(events: SubTBEventBatch) -> SubTBTerms:
    return SubTBTerms(
        residual=events.parent_state_log_flow.new_empty((0,)),
        weight=events.parent_state_log_flow.new_empty((0,)),
        source_ids=events.source_ids.new_empty((0,)),
        is_stop=torch.empty(0, dtype=torch.bool, device=events.device),
    )


def _event_device(
    expansions: ExpansionPrefixBatch,
    terminals: TerminalPrefixBatch,
) -> torch.device:
    if expansions.num_items > 0:
        return expansions.device
    return terminals.device


__all__ = [
    "SubTBEventBatch",
    "SubTBInput",
    "SubTBObjective",
    "SubTBLoss",
    "SubTBTerms",
    "build_subtb_input_from_prefix",
    "build_subtb_input",
    "expansion_db_residual",
    "masked_mean_or_zero",
    "residual_loss_units",
    "single_step_metrics",
    "source_metrics",
    "stop_db_residual",
    "subtrajectory_terms",
    "terminal_metrics",
    "weighted_source_balanced_mean",
]
