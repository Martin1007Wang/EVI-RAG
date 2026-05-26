from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.weaver.policy import PolicyOutput
from src.weaver.policy.backward import canonical_backward_log_prob
from src.weaver.rollout.trajectory import (
    BUDGET,
    NO_FRONTIER,
    POLICY_STOP,
    SRC_POLICY,
    SRC_REPLAY,
    TrajectoryBatch,
)
from src.weaver.context import GraphContext, TargetContext
from src.weaver.nn.feature_encoder import FeatureBank
from src.weaver.policy import ForwardPolicy
from src.weaver.state import ActionSpace, StateBatch, cat_state_batches
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
TERMINAL_REASON_NONE = -1


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

        terminal:
            z_t --STOP--> terminal reward R(z_t), for policy STOP
            z_t --BOUNDARY--> terminal reward R(z_t), for forced boundaries

    Stored quantities:
    - parent_state_log_flow = Φ(z_t)
    - child_state_log_flow  = Φ(z_{t+1}) for expansion, unused for STOP
    - action_log_flow       = G(z_t, a_t) or Φ(z_t) for forced boundaries
    - action_log_prob       = G(z_t, a_t) - Φ(z_t), derived
    - backward_log_prob     = log P_B(z_t | z_{t+1}) for expansion, 0 for terminal
    - terminal_log_reward   = log R(z_t) for terminal, 0 for expansion
    - terminal_reason       = rollout stop reason for terminal, -1 for expansion
    """

    trajectory_ids: Tensor  # [N]
    step_ids: Tensor  # [N]
    source_ids: Tensor  # [N]

    parent_state_log_flow: Tensor  # [N]
    child_state_log_flow: Tensor  # [N]
    action_log_flow: Tensor  # [N]
    backward_log_prob: Tensor  # [N]
    terminal_log_reward: Tensor  # [N]
    terminal_reason: Tensor  # [N]

    is_terminal: Tensor  # [N], bool

    @property
    def device(self) -> torch.device:
        return self.trajectory_ids.device

    @property
    def num_items(self) -> int:
        return int(self.trajectory_ids.numel())

    @property
    def action_log_prob(self) -> Tensor:
        return self.action_log_flow - self.parent_state_log_flow

    @property
    def is_stop(self) -> Tensor:
        return self.is_terminal


@dataclass(frozen=True, slots=True)
class SubTBTerms:
    residual: Tensor
    weight: Tensor
    source_ids: Tensor
    is_terminal: Tensor
    terminal_reason: Tensor

    @property
    def is_stop(self) -> Tensor:
        return self.is_terminal


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
        exp = (~terms.is_terminal) & active
        terminal = terms.is_terminal & active

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
                terminal,
            ).detach(),
            "subtb/terminal_residual_abs": weighted_mean_or_zero(
                terms.residual.abs(),
                terms.weight,
                terminal,
            ).detach(),
        }

        metrics.update(single_step_metrics(x.events))
        metrics.update(source_metrics(terms=terms, units=units))
        metrics.update(subtrajectory_reason_metrics(terms=terms))

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
        prefix_calibration_weight: float = 0.0,
        prefix_calibration_huber_delta: float = 2.0,
        sufficient_stop_margin_weight: float = 0.0,
        sufficient_stop_margin: float = 0.0,
        sufficient_recall_threshold: float = 1.0,
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
        self.prefix_calibration_weight = float(prefix_calibration_weight)
        self.prefix_calibration_huber_delta = float(prefix_calibration_huber_delta)
        self.sufficient_stop_margin_weight = float(sufficient_stop_margin_weight)
        self.sufficient_stop_margin = float(sufficient_stop_margin)
        self.sufficient_recall_threshold = float(sufficient_recall_threshold)

        if self.prefix_calibration_weight < 0.0:
            raise ValueError("prefix_calibration_weight must be nonnegative.")
        if self.prefix_calibration_huber_delta <= 0.0:
            raise ValueError("prefix_calibration_huber_delta must be positive.")
        if self.sufficient_stop_margin_weight < 0.0:
            raise ValueError("sufficient_stop_margin_weight must be nonnegative.")
        if self.sufficient_recall_threshold < 0.0 or self.sufficient_recall_threshold > 1.0:
            raise ValueError("sufficient_recall_threshold must be in [0, 1].")

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
        out = self.loss(subtb_input)
        aux = prefix_terminal_objective(
            policy=policy,
            reward_model=reward_model,
            features=features,
            graph_context=graph_context,
            target_context=target_context,
            prefix=prefix,
            prefix_calibration_weight=self.prefix_calibration_weight,
            prefix_calibration_huber_delta=self.prefix_calibration_huber_delta,
            sufficient_stop_margin_weight=self.sufficient_stop_margin_weight,
            sufficient_stop_margin=self.sufficient_stop_margin,
            sufficient_recall_threshold=self.sufficient_recall_threshold,
        )
        if aux is None:
            return out

        metrics = dict(out.metrics)
        metrics.update(aux.metrics)
        return ObjectiveOutput(
            loss=out.loss + aux.loss,
            metrics=metrics,
            num_states=out.num_states,
            per_unit_loss=out.per_unit_loss,
        )


def prefix_terminal_objective(
    *,
    policy: ForwardPolicy,
    reward_model: TerminalRecallReward,
    features: FeatureBank,
    graph_context: GraphContext,
    target_context: TargetContext,
    prefix: PrefixBatch,
    prefix_calibration_weight: float,
    prefix_calibration_huber_delta: float,
    sufficient_stop_margin_weight: float,
    sufficient_stop_margin: float,
    sufficient_recall_threshold: float,
) -> ObjectiveOutput | None:
    if prefix_calibration_weight <= 0.0 and sufficient_stop_margin_weight <= 0.0:
        return None

    state = _prefix_supervision_state(prefix)
    if state.num_states == 0:
        zero = torch.zeros((), dtype=torch.float32, device=graph_context.device)
        return ObjectiveOutput(
            loss=zero,
            metrics=prefix_terminal_metrics_empty(device=graph_context.device),
            num_states=0,
            per_unit_loss=None,
        )

    unique_state, inverse = _deduplicate_canonical_states(state)
    action_space = unique_state.action_space(graph_context)
    policy_out = policy(
        features=features,
        state=unique_state,
        context=graph_context,
        action_space=action_space,
    )
    unique_reward = reward_model(
        state=unique_state,
        graph_context=graph_context,
        target_context=target_context,
    )
    reward = _select_reward_output(unique_reward, inverse)

    valid = reward.valid_mask.to(device=state.device, dtype=torch.bool)
    sufficient = valid & reward.target_recall.ge(float(sufficient_recall_threshold))

    stop_log_flow = policy_out.stop_log_flow.index_select(0, inverse).float()
    continue_log_flow = policy_out.continue_log_flow.index_select(0, inverse).float()
    stop_log_prob = policy_out.stop_log_prob.index_select(0, inverse).float()

    residual = stop_log_flow - reward.log_reward.detach().float()
    calib_units = residual_loss_units(
        residual,
        loss_type="huber",
        huber_delta=float(prefix_calibration_huber_delta),
    )
    calib_loss = masked_mean_or_zero(calib_units, valid)

    has_frontier = torch.isfinite(continue_log_flow)
    margin_value = (stop_log_flow - continue_log_flow).float()
    margin_active = sufficient & has_frontier
    margin_units = torch.relu(float(sufficient_stop_margin) + continue_log_flow - stop_log_flow)
    margin_loss = masked_mean_or_zero(margin_units, margin_active)

    loss = (
        float(prefix_calibration_weight) * calib_loss
        + float(sufficient_stop_margin_weight) * margin_loss
    )

    metrics = prefix_terminal_metrics(
        reward=reward,
        stop_log_prob=stop_log_prob,
        residual=residual,
        margin=margin_value,
        valid=valid,
        sufficient=sufficient,
        margin_active=margin_active,
        calib_loss=calib_loss,
        margin_loss=margin_loss,
    )

    return ObjectiveOutput(
        loss=loss,
        metrics=metrics,
        num_states=state.num_states,
        per_unit_loss=None,
    )


def _prefix_supervision_state(prefix: PrefixBatch) -> StateBatch:
    states: list[StateBatch] = []
    if prefix.expansions.num_items > 0:
        states.append(prefix.expansions.parent)
    if prefix.terminals.num_items > 0:
        states.append(prefix.terminals.state)
    if not states:
        return _empty_state(device=prefix.terminals.device, budget=int(prefix.terminals.state.budget))
    if len(states) == 1:
        return states[0]
    return cat_state_batches(states)


def _deduplicate_canonical_states(state: StateBatch) -> tuple[StateBatch, Tensor]:
    if state.num_states == 0:
        return state, torch.empty(0, dtype=torch.long, device=state.device)

    key = _canonical_state_key(state)
    unique_key, inverse = torch.unique(
        key,
        dim=0,
        sorted=True,
        return_inverse=True,
    )

    return (
        StateBatch(
            graph_ids=unique_key[:, 0],
            edge_count=unique_key[:, 1],
            edge_ids=unique_key[:, 2:],
            budget=int(state.budget),
        ),
        inverse.to(device=state.device, dtype=torch.long),
    )


def _canonical_state_key(state: StateBatch) -> Tensor:
    edge_ids = state.edge_ids.to(device=state.device, dtype=torch.long)
    if int(state.budget) > 0:
        canonical = edge_ids.new_full(edge_ids.shape, -1)
        for row in range(state.num_states):
            count = int(state.edge_count[row].item())
            if count <= 0:
                continue
            canonical[row, :count] = torch.sort(edge_ids[row, :count]).values
        edge_ids = canonical

    return torch.cat(
        (
            state.graph_ids.to(dtype=torch.long).view(-1, 1),
            state.edge_count.to(dtype=torch.long).view(-1, 1),
            edge_ids,
        ),
        dim=1,
    )


def _deduplicate_ordered_states(state: StateBatch) -> tuple[StateBatch, Tensor]:
    """
    Backward-compatible alias; state equality is now canonical edge-set equality.
    """

    return _deduplicate_canonical_states(state)


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

    all_states = cat_state_batches(
        [
            expansions.parent,
            expansions.child,
            terminals.state,
        ]
    )
    unique_states, inverse = _deduplicate_canonical_states(all_states)

    if unique_states.num_states > 0:
        policy_out = policy(
            features=features,
            state=unique_states,
            context=graph_context,
            action_space=unique_states.action_space(graph_context),
        )
    else:
        policy_out = _empty_policy_output(device=all_states.device)

    n_exp = expansions.num_items
    n_term = terminals.num_items
    parent_rows = inverse[:n_exp]
    child_rows = inverse[n_exp : 2 * n_exp]
    terminal_rows = inverse[2 * n_exp : 2 * n_exp + n_term]

    if expansions.num_items > 0:
        backward_log_prob = canonical_backward_log_prob(
            parent_state=expansions.parent,
            child_state=expansions.child,
            action_edge_ids=expansions.edge_ids,
            graph_context=graph_context,
            validate=bool(validate_backward),
        )
    else:
        backward_log_prob = torch.empty(
            0,
            dtype=torch.float32,
            device=terminals.device,
        )

    if terminals.num_items > 0:
        unique_terminal_state, terminal_inverse = _deduplicate_canonical_states(terminals.state)
        unique_terminal_reward = reward_model(
            state=unique_terminal_state,
            graph_context=graph_context,
            target_context=target_context,
        )
        terminal_reward = _select_reward_output(unique_terminal_reward, terminal_inverse)
    else:
        terminal_reward = _empty_reward_output(device=expansions.device)

    exp_events = _build_expansion_events_from_policy_rows(
        policy_out=policy_out,
        parent_rows=parent_rows,
        child_rows=child_rows,
        backward_log_prob=backward_log_prob,
        expansions=expansions,
    )
    stop_events = _build_terminal_events_from_policy_rows(
        policy_out=policy_out,
        terminal_rows=terminal_rows,
        terminal_reward=terminal_reward,
        terminals=terminals,
    )

    return SubTBInput(
        events=_cat_event_batches(
            [exp_events, stop_events],
            device=_event_device(expansions, terminals),
        ),
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

    Policy STOP uses terminal_out.stop_log_flow directly. Forced terminal
    boundaries use terminal_out.state_log_flow so they do not act like sampled
    STOP actions.
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
        terminal_reason=torch.full(
            (n,),
            int(TERMINAL_REASON_NONE),
            dtype=torch.long,
            device=device,
        ),
        is_terminal=torch.zeros(
            n,
            dtype=torch.bool,
            device=device,
        ),
    )


def _build_expansion_events_from_policy_rows(
    *,
    policy_out: PolicyOutput,
    parent_rows: Tensor,
    child_rows: Tensor,
    backward_log_prob: Tensor,
    expansions: ExpansionPrefixBatch,
) -> SubTBEventBatch:
    device = expansions.device
    n = expansions.num_items

    if n == 0:
        return _empty_events(device=device)

    parent_rows = parent_rows.to(device=device, dtype=torch.long).view(-1)
    child_rows = child_rows.to(device=device, dtype=torch.long).view(-1)

    parent_state_log_flow = policy_out.state_log_flow.index_select(0, parent_rows).float()
    child_state_log_flow = policy_out.state_log_flow.index_select(0, child_rows).float()

    action_log_flow = policy_out.gather_action_log_flow(
        row_ids=parent_rows,
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
        terminal_reason=torch.full(
            (n,),
            int(TERMINAL_REASON_NONE),
            dtype=torch.long,
            device=device,
        ),
        is_terminal=torch.zeros(
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

    reason = terminals.reason.index_select(0, rows).to(
        device=device,
        dtype=torch.long,
    )
    stop_log_flow = terminal_out.stop_log_flow.index_select(
        0,
        rows,
    ).float()
    boundary_log_flow = parent_state_log_flow
    action_log_flow = torch.where(
        reason.eq(int(POLICY_STOP)),
        stop_log_flow,
        boundary_log_flow,
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
        terminal_reason=reason,
        is_terminal=torch.ones(
            n,
            dtype=torch.bool,
            device=device,
        ),
    )


def _build_terminal_events_from_policy_rows(
    *,
    policy_out: PolicyOutput,
    terminal_rows: Tensor,
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

    terminal_rows = terminal_rows.to(device=device, dtype=torch.long).view(-1)
    policy_rows = terminal_rows.index_select(0, rows)

    parent_state_log_flow = policy_out.state_log_flow.index_select(
        0,
        policy_rows,
    ).float()

    reason = terminals.reason.index_select(0, rows).to(
        device=device,
        dtype=torch.long,
    )
    stop_log_flow = policy_out.stop_log_flow.index_select(
        0,
        policy_rows,
    ).float()
    boundary_log_flow = parent_state_log_flow
    action_log_flow = torch.where(
        reason.eq(int(POLICY_STOP)),
        stop_log_flow,
        boundary_log_flow,
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
        terminal_reason=reason,
        is_terminal=torch.ones(
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
    is_terminal = torch.zeros(shape, dtype=torch.bool, device=device)
    source_ids = torch.full(shape, -1, dtype=torch.long, device=device)
    terminal_reason = torch.full(
        shape,
        int(TERMINAL_REASON_NONE),
        dtype=torch.long,
        device=device,
    )

    parent_flow = torch.zeros(shape, dtype=dtype, device=device)
    child_flow = torch.zeros(shape, dtype=dtype, device=device)
    action_log_prob = torch.zeros(shape, dtype=dtype, device=device)
    backward_log_prob = torch.zeros(shape, dtype=dtype, device=device)
    terminal_log_reward = torch.zeros(shape, dtype=dtype, device=device)

    traj = events.trajectory_ids
    step = events.step_ids

    has_event[traj, step] = True
    is_terminal[traj, step] = events.is_terminal
    source_ids[traj, step] = events.source_ids
    terminal_reason[traj, step] = events.terminal_reason
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
    terminal_flags: list[Tensor] = []
    terminal_reasons: list[Tensor] = []

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
            is_terminal[:, end],
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
        terminal_flags.append(is_terminal[:, end][valid])
        terminal_reasons.append(terminal_reason[:, end][valid])

    if not residuals:
        return _empty_terms(events)

    return SubTBTerms(
        residual=torch.cat(residuals, dim=0),
        weight=torch.cat(weights, dim=0),
        source_ids=torch.cat(sources, dim=0),
        is_terminal=torch.cat(terminal_flags, dim=0),
        terminal_reason=torch.cat(terminal_reasons, dim=0),
    )


def single_step_metrics(events: SubTBEventBatch) -> dict[str, Tensor]:
    exp = ~events.is_terminal
    terminal = events.is_terminal
    policy_stop = events.terminal_reason.eq(int(POLICY_STOP))
    no_frontier = events.terminal_reason.eq(int(NO_FRONTIER))
    budget = events.terminal_reason.eq(int(BUDGET))

    exp_residual = expansion_db_residual(events)
    terminal_residual = terminal_db_residual(events)

    return {
        "db/exp_residual_abs": masked_mean_or_zero(
            exp_residual.abs(),
            exp,
        ).detach(),
        "db/stop_residual_abs": masked_mean_or_zero(
            terminal_residual.abs(),
            terminal,
        ).detach(),
        "db/terminal_residual_abs": masked_mean_or_zero(
            terminal_residual.abs(),
            terminal,
        ).detach(),
        "db/policy_stop_residual_abs": masked_mean_or_zero(
            terminal_residual.abs(),
            policy_stop,
        ).detach(),
        "db/no_frontier_boundary_residual_abs": masked_mean_or_zero(
            terminal_residual.abs(),
            no_frontier,
        ).detach(),
        "db/budget_boundary_residual_abs": masked_mean_or_zero(
            terminal_residual.abs(),
            budget,
        ).detach(),
        "flow/terminal_log_reward": masked_mean_or_zero(
            events.terminal_log_reward,
            terminal,
        ).detach(),
    }


def expansion_db_residual(events: SubTBEventBatch) -> Tensor:
    """
    One-step DB residual for expansion:

        G(z,e) - Φ(z') - log P_B(z | z')
    """

    return events.action_log_flow - events.child_state_log_flow - events.backward_log_prob


def terminal_db_residual(events: SubTBEventBatch) -> Tensor:
    """
    One-step terminal residual:

        G(z,STOP) - log R(z), for policy STOP
        Φ(z) - log R(z), for forced boundaries
    """

    return events.action_log_flow - events.terminal_log_reward


def stop_db_residual(events: SubTBEventBatch) -> Tensor:
    """
    Backward-compatible alias for terminal_db_residual.
    """

    return terminal_db_residual(events)


def subtrajectory_reason_metrics(
    *,
    terms: SubTBTerms,
) -> dict[str, Tensor]:
    active = terms.weight.gt(0)
    policy_stop = terms.terminal_reason.eq(int(POLICY_STOP)) & active
    no_frontier = terms.terminal_reason.eq(int(NO_FRONTIER)) & active
    budget = terms.terminal_reason.eq(int(BUDGET)) & active

    return {
        "subtb/policy_stop_residual_abs": weighted_mean_or_zero(
            terms.residual.abs(),
            terms.weight,
            policy_stop,
        ).detach(),
        "subtb/no_frontier_boundary_residual_abs": weighted_mean_or_zero(
            terms.residual.abs(),
            terms.weight,
            no_frontier,
        ).detach(),
        "subtb/budget_boundary_residual_abs": weighted_mean_or_zero(
            terms.residual.abs(),
            terms.weight,
            budget,
        ).detach(),
    }


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
    policy_stop = terminals.reason.eq(int(POLICY_STOP)) & valid
    no_frontier = terminals.reason.eq(int(NO_FRONTIER)) & valid
    budget = terminals.reason.eq(int(BUDGET)) & valid

    hit = terminal_reward.answer_count.gt(0).to(dtype=torch.float)
    valid_count = valid.to(dtype=torch.float).sum()
    policy_stop_count = policy_stop.to(dtype=torch.float).sum()
    no_frontier_count = no_frontier.to(dtype=torch.float).sum()
    budget_count = budget.to(dtype=torch.float).sum()

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
        "terminal/policy_stop_count": policy_stop_count.detach(),
        "terminal/no_frontier_count": no_frontier_count.detach(),
        "terminal/budget_count": budget_count.detach(),
        "terminal/policy_stop_fraction": _safe_divide(
            policy_stop_count,
            valid_count,
        ).detach(),
        "terminal/no_frontier_fraction": _safe_divide(
            no_frontier_count,
            valid_count,
        ).detach(),
        "terminal/budget_fraction": _safe_divide(
            budget_count,
            valid_count,
        ).detach(),
    }


def prefix_terminal_metrics(
    *,
    reward: RewardOutput,
    stop_log_prob: Tensor,
    residual: Tensor,
    margin: Tensor,
    valid: Tensor,
    sufficient: Tensor,
    margin_active: Tensor,
    calib_loss: Tensor,
    margin_loss: Tensor,
) -> dict[str, Tensor]:
    zero_recall = valid & reward.target_recall.le(0.0)
    partial_recall = valid & reward.target_recall.gt(0.0) & ~sufficient
    stop_prob = stop_log_prob.float().exp()

    valid_count = valid.to(dtype=torch.float).sum()
    sufficient_count = sufficient.to(dtype=torch.float).sum()
    margin_count = margin_active.to(dtype=torch.float).sum()

    return {
        "prefix_calib/loss": calib_loss.detach(),
        "prefix_calib/residual_abs": masked_mean_or_zero(
            residual.abs(),
            valid,
        ).detach(),
        "prefix_calib/residual_abs_zero": masked_mean_or_zero(
            residual.abs(),
            zero_recall,
        ).detach(),
        "prefix_calib/residual_abs_partial": masked_mean_or_zero(
            residual.abs(),
            partial_recall,
        ).detach(),
        "prefix_calib/residual_abs_sufficient": masked_mean_or_zero(
            residual.abs(),
            sufficient,
        ).detach(),
        "sufficient/stop_margin_loss": margin_loss.detach(),
        "sufficient/stop_margin_mean": masked_mean_or_zero(
            margin,
            margin_active,
        ).detach(),
        "full_cover/stop_margin_mean": masked_mean_or_zero(
            margin,
            margin_active,
        ).detach(),
        "sufficient/stop_prob_mean": masked_mean_or_zero(
            stop_prob,
            sufficient,
        ).detach(),
        "sufficient/active_fraction": _safe_divide(
            sufficient_count,
            valid_count,
        ).detach(),
        "sufficient/margin_active_fraction": _safe_divide(
            margin_count,
            valid_count,
        ).detach(),
    }


def prefix_terminal_metrics_empty(*, device: torch.device) -> dict[str, Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return {
        "prefix_calib/loss": zero,
        "prefix_calib/residual_abs": zero,
        "prefix_calib/residual_abs_zero": zero,
        "prefix_calib/residual_abs_partial": zero,
        "prefix_calib/residual_abs_sufficient": zero,
        "sufficient/stop_margin_loss": zero,
        "sufficient/stop_margin_mean": zero,
        "full_cover/stop_margin_mean": zero,
        "sufficient/stop_prob_mean": zero,
        "sufficient/active_fraction": zero,
        "sufficient/margin_active_fraction": zero,
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


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    if not bool(denominator.gt(0)):
        return numerator.new_zeros(())
    return numerator.float() / denominator.float()


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
        terminal_reason=torch.cat(
            [x.terminal_reason for x in non_empty],
            dim=0,
        ),
        is_terminal=torch.cat([x.is_terminal for x in non_empty], dim=0),
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
        terminal_reason=long,
        is_terminal=bool_,
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


def _empty_state(
    *,
    device: torch.device,
    budget: int,
) -> StateBatch:
    return StateBatch.initial(
        graph_ids=torch.empty(0, dtype=torch.long, device=device),
        budget=int(budget),
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


def _select_reward_output(reward: RewardOutput, rows: Tensor) -> RewardOutput:
    rows = rows.to(device=reward.log_reward.device, dtype=torch.long).view(-1)

    return RewardOutput(
        log_reward=reward.log_reward.index_select(0, rows),
        raw_log_reward=reward.raw_log_reward.index_select(0, rows),
        answer_count=reward.answer_count.index_select(0, rows),
        target_count=reward.target_count.index_select(0, rows),
        target_recall=reward.target_recall.index_select(0, rows),
        target_proximity=reward.target_proximity.index_select(0, rows),
        path_edge_count=reward.path_edge_count.index_select(0, rows),
        path_edge_precision=reward.path_edge_precision.index_select(0, rows),
        edge_count=reward.edge_count.index_select(0, rows),
        valid_mask=reward.valid_mask.index_select(0, rows),
        success_mask=reward.success_mask.index_select(0, rows),
        metrics={},
    )


def _empty_terms(events: SubTBEventBatch) -> SubTBTerms:
    return SubTBTerms(
        residual=events.parent_state_log_flow.new_empty((0,)),
        weight=events.parent_state_log_flow.new_empty((0,)),
        source_ids=events.source_ids.new_empty((0,)),
        is_terminal=torch.empty(0, dtype=torch.bool, device=events.device),
        terminal_reason=torch.empty(0, dtype=torch.long, device=events.device),
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
    "prefix_terminal_objective",
    "residual_loss_units",
    "single_step_metrics",
    "source_metrics",
    "stop_db_residual",
    "subtrajectory_terms",
    "subtrajectory_reason_metrics",
    "terminal_db_residual",
    "terminal_metrics",
    "weighted_source_balanced_mean",
]
