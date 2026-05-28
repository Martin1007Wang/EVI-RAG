# src/weaver/objectives/
# .py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from src.weaver.context import GraphContext, TargetContext
from src.weaver.feature import FeatureBank
from src.weaver.objectives.transition_batch import (
    NonterminalTransitionBatch,
    TerminalTransitionBatch,
)
from src.weaver.objectives.output import ObjectiveOutput
from src.weaver.state import ActionSpace, StateBatch

Tensor = torch.Tensor

ResidualLoss = Literal["huber", "l2", "l1"]
BackwardKernel = Literal["uniform_parent", "deterministic"]


@dataclass(frozen=True, slots=True)
class MatchingTerms:
    """
    Per-unit residuals and losses for diagnostics.
    """

    residual: Tensor  # [N]
    loss_units: Tensor  # [N]
    valid_mask: Tensor  # [N]

    @property
    def num_units(self) -> int:
        return int(self.residual.numel())

    @property
    def num_valid(self) -> int:
        return int(self.valid_mask.sum().item())


class EdgeFlowMatchingObjective(nn.Module):
    """
    Edge Flow Matching objective with explicit terminating action.

    This replaces:

    - SubTB over expansion prefixes, if you choose one-step EFM/DB;
    - prefix_calibration_loss, because STOP reward matching is now a first-class
      terminal-edge residual.

    Objective:

        nonterminal edge:
            Q_theta(z, e) = Phi_theta(z + e) + log P_B(z | z + e)

        terminal edge:
            rho_theta(z) = log R(z)

    where:

        Phi_theta(z) = logsumexp(
            stop_log_flow(z),
            edge_log_flow(z, e) for e in C(z)
        )

    Required policy output fields:
        - stop_log_flow: Tensor [S]
        - edge_log_flow: Tensor [F]
        - optionally action_space.
          If absent, state.action_space(graph_context) is used, and the policy
          must have produced edge_log_flow in that same order.

    Required reward output fields:
        - log_reward: Tensor [S]
        - optionally valid_mask: Tensor [S]
    """

    def __init__(
        self,
        *,
        nonterminal_weight: float = 1.0,
        terminal_weight: float = 1.0,
        residual_loss: ResidualLoss = "huber",
        huber_delta: float = 2.0,
        backward_kernel: BackwardKernel = "uniform_parent",
        policy_state_chunk_size: int | None = None,
    ) -> None:
        super().__init__()

        self.nonterminal_weight = float(nonterminal_weight)
        self.terminal_weight = float(terminal_weight)
        self.residual_loss = residual_loss
        self.huber_delta = float(huber_delta)
        self.backward_kernel = backward_kernel
        self.policy_state_chunk_size = _optional_positive_int(
            policy_state_chunk_size,
            "policy_state_chunk_size",
        )

        if self.nonterminal_weight < 0.0:
            raise ValueError("nonterminal_weight must be nonnegative.")
        if self.terminal_weight < 0.0:
            raise ValueError("terminal_weight must be nonnegative.")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if residual_loss not in {"huber", "l2", "l1"}:
            raise ValueError(f"Unknown residual_loss: {residual_loss!r}.")
        if backward_kernel not in {"uniform_parent", "deterministic"}:
            raise ValueError(f"Unknown backward_kernel: {backward_kernel!r}.")

    def forward(
        self,
        *,
        policy: nn.Module,
        reward_model: nn.Module,
        features: FeatureBank,
        graph_context: GraphContext,
        target_context: TargetContext,
        nonterminal: NonterminalTransitionBatch | None = None,
        terminal: TerminalTransitionBatch | None = None,
    ) -> ObjectiveOutput:
        device = _infer_device(nonterminal=nonterminal, terminal=terminal)
        dtype = torch.float32

        pieces: list[Tensor] = []
        per_unit: list[Tensor] = []

        metrics: dict[str, float] = {}

        if nonterminal is not None and nonterminal.num_transitions > 0 and self.nonterminal_weight > 0.0:
            nt = nonterminal_edge_flow_matching(
                policy=policy,
                features=features,
                graph_context=graph_context,
                target_context=target_context,
                transitions=nonterminal,
                residual_loss=self.residual_loss,
                huber_delta=self.huber_delta,
                backward_kernel=self.backward_kernel,
                policy_state_chunk_size=self.policy_state_chunk_size,
            )
            nt_loss = masked_mean_or_zero(nt.loss_units, nt.valid_mask)
            pieces.append(self.nonterminal_weight * nt_loss)
            per_unit.append(self.nonterminal_weight * nt.loss_units[nt.valid_mask])

            metrics.update(
                _term_metrics(
                    prefix="objective/nonterminal_edge_flow_matching",
                    terms=nt,
                    loss=nt_loss,
                )
            )
        else:
            metrics.update(_empty_metrics("objective/nonterminal_edge_flow_matching"))

        if terminal is not None and terminal.num_transitions > 0 and self.terminal_weight > 0.0:
            tt = terminal_edge_reward_matching(
                policy=policy,
                reward_model=reward_model,
                features=features,
                graph_context=graph_context,
                target_context=target_context,
                transitions=terminal,
                residual_loss=self.residual_loss,
                huber_delta=self.huber_delta,
                policy_state_chunk_size=self.policy_state_chunk_size,
            )
            tt_loss = masked_mean_or_zero(tt.loss_units, tt.valid_mask)
            pieces.append(self.terminal_weight * tt_loss)
            per_unit.append(self.terminal_weight * tt.loss_units[tt.valid_mask])

            metrics.update(
                _term_metrics(
                    prefix="objective/terminal_edge_reward_matching",
                    terms=tt,
                    loss=tt_loss,
                )
            )
        else:
            metrics.update(_empty_metrics("objective/terminal_edge_reward_matching"))

        if pieces:
            loss = torch.stack(pieces).sum()
        else:
            loss = torch.zeros((), dtype=dtype, device=device)

        if per_unit:
            per_unit_loss = torch.cat(per_unit, dim=0)
        else:
            per_unit_loss = None

        metrics["objective/loss"] = _to_float(loss)
        metrics["objective/nonterminal_weight"] = self.nonterminal_weight
        metrics["objective/terminal_weight"] = self.terminal_weight

        num_states = 0
        if nonterminal is not None:
            num_states += int(nonterminal.parent_state.num_states)
        if terminal is not None:
            num_states += int(terminal.state.num_states)

        return ObjectiveOutput(
            loss=loss,
            metrics=metrics,
            num_states=num_states,
            per_unit_loss=per_unit_loss,
        )

def nonterminal_edge_flow_matching(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    target_context: TargetContext,
    transitions: NonterminalTransitionBatch,
    residual_loss: ResidualLoss = "huber",
    huber_delta: float = 2.0,
    backward_kernel: BackwardKernel = "uniform_parent",
    policy_state_chunk_size: int | None = None,
) -> MatchingTerms:
    """
    Nonterminal EXPAND edge flow matching.

    For transition:

        z --e--> z'

    train:

        Q_theta(z, e) = Phi_theta(z') + log P_B(z | z')

    residual:

        r = Q_theta(z, e) - Phi_theta(z') - log P_B(z | z')

    Under uniform-parent backward kernel:

        P_B(z | z') = 1 / number_of_legal_parents(z')

    so:

        log P_B = -log(number_of_legal_parents(z'))
    """

    parent_state_ids = _as_1d_long(
        transitions.parent_state_ids,
        device=transitions.device,
        name="transitions.parent_state_ids",
    )
    edge_ids = _as_1d_long(
        transitions.edge_ids,
        device=transitions.device,
        name="transitions.edge_ids",
    )

    if int(parent_state_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("parent_state_ids and edge_ids must have the same length.")

    if int(edge_ids.numel()) == 0:
        empty = torch.empty(0, dtype=torch.float32, device=transitions.device)
        valid = torch.empty(0, dtype=torch.bool, device=transitions.device)
        return MatchingTerms(residual=empty, loss_units=empty, valid_mask=valid)

    child_state = transitions.materialize_child_state()

    if int(child_state.num_states) != int(edge_ids.numel()):
        raise ValueError(
            "child_state must have one row per nonterminal transition. "
            f"Got child_state.num_states={child_state.num_states}, "
            f"num_transitions={int(edge_ids.numel())}."
        )

    q_parent_edge = _gather_edge_log_flow_chunked(
        policy=policy,
        features=features,
        graph_context=graph_context,
        state=transitions.parent_state,
        state_ids=parent_state_ids,
        edge_ids=edge_ids,
        policy_state_chunk_size=policy_state_chunk_size,
    )

    child_log_flow = _state_log_flow_chunked(
        policy=policy,
        features=features,
        graph_context=graph_context,
        state=child_state,
        policy_state_chunk_size=policy_state_chunk_size,
    )

    if transitions.log_backward is not None:
        log_backward = transitions.log_backward.to(
            device=transitions.device,
            dtype=q_parent_edge.dtype,
        ).view(-1)
        if int(log_backward.numel()) != int(edge_ids.numel()):
            raise ValueError("transitions.log_backward must have one value per transition.")
    else:
        log_backward = compute_log_backward(
            child_state=child_state,
            graph_context=graph_context,
            backward_kernel=backward_kernel,
            dtype=q_parent_edge.dtype,
        )

    residual = q_parent_edge - child_log_flow - log_backward
    valid_mask = torch.isfinite(residual)

    loss_units = residual_loss_units(
        residual=residual,
        residual_loss=residual_loss,
        huber_delta=huber_delta,
    )

    return MatchingTerms(
        residual=residual,
        loss_units=loss_units,
        valid_mask=valid_mask,
    )


def terminal_edge_reward_matching(
    *,
    policy: nn.Module,
    reward_model: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    target_context: TargetContext,
    transitions: TerminalTransitionBatch,
    residual_loss: ResidualLoss = "huber",
    huber_delta: float = 2.0,
    policy_state_chunk_size: int | None = None,
) -> MatchingTerms:
    """
    Terminal STOP edge reward matching.

    For transition:

        z --STOP--> terminal(z)

    train:

        stop_log_flow_theta(z) = log R(z)

    residual:

        r = stop_log_flow_theta(z) - log_reward(z)

    This is the mathematically clean replacement for prefix_calibration_loss.
    """

    if transitions.num_transitions == 0:
        empty = torch.empty(0, dtype=torch.float32, device=transitions.device)
        valid = torch.empty(0, dtype=torch.bool, device=transitions.device)
        return MatchingTerms(residual=empty, loss_units=empty, valid_mask=valid)

    stop_log_flow = _stop_log_flow_chunked(
        policy=policy,
        features=features,
        graph_context=graph_context,
        state=transitions.state,
        policy_state_chunk_size=policy_state_chunk_size,
    )

    reward_output = _call_reward(
        reward_model=reward_model,
        graph_context=graph_context,
        target_context=target_context,
        state=transitions.state,
    )

    log_reward = _get_log_reward(reward_output).to(
        device=stop_log_flow.device,
        dtype=stop_log_flow.dtype,
    )

    if int(stop_log_flow.numel()) != transitions.num_transitions:
        raise ValueError("policy_output.stop_log_flow must have one value per terminal state.")
    if int(log_reward.numel()) != transitions.num_transitions:
        raise ValueError("reward_output.log_reward must have one value per terminal state.")

    reward_valid = _get_reward_valid_mask(
        reward_output=reward_output,
        like=log_reward,
    )

    residual = stop_log_flow - log_reward
    valid_mask = reward_valid & torch.isfinite(residual)

    loss_units = residual_loss_units(
        residual=residual,
        residual_loss=residual_loss,
        huber_delta=huber_delta,
    )

    return MatchingTerms(
        residual=residual,
        loss_units=loss_units,
        valid_mask=valid_mask,
    )


def state_log_flow_from_policy_output(
    *,
    policy_output: object,
    state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    """
    Compute:

        Phi_theta(z) = logsumexp(
            stop_log_flow(z),
            edge_log_flow(z, e) for e in C(z)
        )

    This is the outgoing state flow induced by STOP plus legal EXPAND actions.
    """

    stop_log_flow = _get_stop_log_flow(policy_output)
    edge_log_flow = _get_edge_log_flow(policy_output)

    action_space = _get_action_space(
        policy_output=policy_output,
        state=state,
        graph_context=graph_context,
    )

    if int(stop_log_flow.numel()) != int(state.num_states):
        raise ValueError("stop_log_flow must have shape [num_states].")
    if int(edge_log_flow.numel()) != int(action_space.num_expansions):
        raise ValueError(
            "edge_log_flow must have one value per legal expansion action. "
            f"Got edge_log_flow={int(edge_log_flow.numel())}, "
            f"num_expansions={action_space.num_expansions}."
        )

    continue_log_flow = segment_logsumexp(
        values=edge_log_flow,
        segment_ids=action_space.expand_state_ids,
        num_segments=int(state.num_states),
    )

    return torch.logaddexp(stop_log_flow, continue_log_flow)


def compute_log_backward(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
    backward_kernel: BackwardKernel,
    dtype: torch.dtype,
) -> Tensor:
    """
    Compute log P_B(parent | child) for each child row.

    For canonical unordered states, deterministic backward is generally wrong
    unless the data generator explicitly imposes a deterministic parent rule.

    uniform_parent:
        P_B(parent | child) = 1 / number_of_legal_parents(child)

    deterministic:
        P_B(parent | child) = 1

    The uniform-parent count is computed under the current StateBatch expansion
    semantics: outgoing edges from covered nodes.
    """

    if backward_kernel == "deterministic":
        return torch.zeros(
            int(child_state.num_states),
            dtype=dtype,
            device=child_state.device,
        )

    if backward_kernel != "uniform_parent":
        raise ValueError(f"Unknown backward_kernel: {backward_kernel!r}.")

    counts = count_legal_backward_parents(
        child_state=child_state,
        graph_context=graph_context,
    ).to(dtype=dtype)

    if bool(counts.le(0).any()):
        raise ValueError("Every non-initial child state must have at least one legal backward parent.")

    return -torch.log(counts)


def count_legal_backward_parents(
    *,
    child_state: StateBatch,
    graph_context: GraphContext,
) -> Tensor:
    """
    Count legal parents for each canonical child state.

    A selected edge e in child state z' defines a candidate parent:

        z = z' \\ {e}

    Under the current forward expansion semantics, z is a legal parent iff:

        src(e) in covered_nodes(z)

    because:
    - e belongs to the same graph by construction;
    - e is not selected in z;
    - z has one more budget slot than z';
    - action_space enumerates outgoing edges from covered nodes.

    This is O(S * B^2), deliberately acceptable because B is the expansion
    budget and should stay small. If this becomes hot, cache log_backward in
    the transition builder.
    """

    device = child_state.device
    out = torch.zeros(
        int(child_state.num_states),
        dtype=torch.long,
        device=device,
    )

    edge_src = graph_context.edge_src
    edge_dst = graph_context.edge_dst
    anchor_ptr = graph_context.anchor_ptr
    anchor_node_ids = graph_context.anchor_node_ids

    for row in range(int(child_state.num_states)):
        graph_id = int(child_state.graph_ids[row].item())
        count = int(child_state.edge_count[row].item())

        if count <= 0:
            out[row] = 0
            continue

        selected = child_state.edge_ids[row, :count].to(dtype=torch.long)

        anchor_start = int(anchor_ptr[graph_id].item())
        anchor_end = int(anchor_ptr[graph_id + 1].item())
        anchors = anchor_node_ids[anchor_start:anchor_end].to(device=device)

        for j in range(count):
            edge_id = selected[j]
            keep = torch.ones(count, dtype=torch.bool, device=device)
            keep[j] = False

            remaining = selected[keep]

            if int(remaining.numel()) == 0:
                covered = anchors
            else:
                covered = torch.cat(
                    (
                        anchors,
                        edge_src.index_select(0, remaining),
                        edge_dst.index_select(0, remaining),
                    ),
                    dim=0,
                ).unique(sorted=False)

            src = edge_src.index_select(0, edge_id.view(1))[0]
            if bool(covered.eq(src).any()):
                out[row] += 1

    return out


def residual_loss_units(
    *,
    residual: Tensor,
    residual_loss: ResidualLoss,
    huber_delta: float,
) -> Tensor:
    if residual_loss == "huber":
        return F.huber_loss(
            residual,
            torch.zeros_like(residual),
            delta=float(huber_delta),
            reduction="none",
        )

    if residual_loss == "l2":
        return 0.5 * residual.square()

    if residual_loss == "l1":
        return residual.abs()

    raise ValueError(f"Unknown residual_loss: {residual_loss!r}.")


def masked_mean_or_zero(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)

    if int(values.numel()) != int(mask.numel()):
        raise ValueError("values and mask must have the same number of elements.")

    if int(values.numel()) == 0 or not bool(mask.any()):
        return torch.zeros((), dtype=values.dtype, device=values.device)

    return values[mask].mean()


def segment_logsumexp(
    *,
    values: Tensor,
    segment_ids: Tensor,
    num_segments: int,
) -> Tensor:
    """
    Segment logsumexp.

    Returns out[s] = logsumexp(values[k] where segment_ids[k] == s).

    Empty segments receive -inf.
    """

    num_segments = int(num_segments)

    if values.ndim != 1:
        raise ValueError(f"values must have shape [N], got {tuple(values.shape)}.")
    if segment_ids.ndim != 1:
        raise ValueError(f"segment_ids must have shape [N], got {tuple(segment_ids.shape)}.")
    if int(values.numel()) != int(segment_ids.numel()):
        raise ValueError("values and segment_ids must have the same length.")

    out = torch.full(
        (num_segments,),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )

    if int(values.numel()) == 0:
        return out

    segment_ids = segment_ids.to(device=values.device, dtype=torch.long)

    max_per_segment = out.clone()
    max_per_segment.scatter_reduce_(
        dim=0,
        index=segment_ids,
        src=values,
        reduce="amax",
        include_self=True,
    )

    shifted = values - max_per_segment.index_select(0, segment_ids)
    exp_shifted = shifted.exp()

    sum_per_segment = torch.zeros(
        (num_segments,),
        dtype=values.dtype,
        device=values.device,
    )
    sum_per_segment.scatter_add_(0, segment_ids, exp_shifted)

    return max_per_segment + sum_per_segment.clamp_min(torch.finfo(values.dtype).tiny).log()


def _gather_edge_log_flow(
    *,
    policy_output: object,
    state: StateBatch,
    graph_context: GraphContext,
    state_ids: Tensor,
    edge_ids: Tensor,
) -> Tensor:
    """
    Gather Q_theta(z, e) from policy_output.edge_log_flow.

    Assumption:
    edge_log_flow is aligned with action_space.expand_state_ids /
    action_space.expand_edge_ids.
    """

    edge_log_flow = _get_edge_log_flow(policy_output)

    action_space = _get_action_space(
        policy_output=policy_output,
        state=state,
        graph_context=graph_context,
    )

    state_ids = state_ids.to(device=edge_log_flow.device, dtype=torch.long).view(-1)
    edge_ids = edge_ids.to(device=edge_log_flow.device, dtype=torch.long).view(-1)

    if int(state_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("state_ids and edge_ids must have the same length.")

    if int(edge_ids.numel()) == 0:
        return torch.empty(0, dtype=edge_log_flow.dtype, device=edge_log_flow.device)

    if int(edge_log_flow.numel()) != int(action_space.num_expansions):
        raise ValueError(
            "edge_log_flow must have one value per legal expansion action. "
            f"Got edge_log_flow={int(edge_log_flow.numel())}, "
            f"num_expansions={action_space.num_expansions}."
        )

    base = int(graph_context.num_edges)

    frontier_state_ids = action_space.expand_state_ids.to(
        device=edge_log_flow.device,
        dtype=torch.long,
    )
    frontier_edge_ids = action_space.expand_edge_ids.to(
        device=edge_log_flow.device,
        dtype=torch.long,
    )

    frontier_keys = frontier_state_ids * base + frontier_edge_ids
    query_keys = state_ids * base + edge_ids

    if int(frontier_keys.numel()) == 0:
        raise ValueError("Parent state action_space contains no legal expansion actions.")

    sorted_keys, order = torch.sort(frontier_keys)
    positions = torch.searchsorted(sorted_keys, query_keys)

    in_range = positions.lt(int(sorted_keys.numel()))
    safe_positions = positions.clamp_max(max(int(sorted_keys.numel()) - 1, 0))
    matched = in_range & sorted_keys.index_select(0, safe_positions).eq(query_keys)

    if not bool(matched.all()):
        missing = query_keys[~matched][:10].detach().cpu().tolist()
        raise ValueError(
            "Some requested expansion actions are not legal under the parent state's action_space. " f"First missing encoded keys: {missing}"
        )

    flow_positions = order.index_select(0, safe_positions)
    return edge_log_flow.index_select(0, flow_positions)


def _call_policy(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    state: StateBatch,
) -> object:
    action_space = state.action_space(graph_context)
    return policy(
        features=features,
        state=state,
        context=graph_context,
        action_space=action_space,
    )


def _gather_edge_log_flow_chunked(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    state: StateBatch,
    state_ids: Tensor,
    edge_ids: Tensor,
    policy_state_chunk_size: int | None,
) -> Tensor:
    state_ids = _as_1d_long(
        state_ids,
        device=state.device,
        name="state_ids",
    )
    edge_ids = _as_1d_long(
        edge_ids,
        device=state.device,
        name="edge_ids",
    )

    if int(state_ids.numel()) != int(edge_ids.numel()):
        raise ValueError("state_ids and edge_ids must have the same length.")

    if int(edge_ids.numel()) == 0:
        return torch.empty(0, dtype=torch.float32, device=state.device)

    if (
        policy_state_chunk_size is None
        or int(state.num_states) <= int(policy_state_chunk_size)
    ):
        policy_output = _call_policy(
            policy=policy,
            features=features,
            graph_context=graph_context,
            state=state,
        )
        return _gather_edge_log_flow(
            policy_output=policy_output,
            state=state,
            graph_context=graph_context,
            state_ids=state_ids,
            edge_ids=edge_ids,
        )

    query_positions = torch.arange(
        int(state_ids.numel()),
        dtype=torch.long,
        device=state.device,
    )
    chunk_positions: list[Tensor] = []
    chunk_values: list[Tensor] = []

    for start, end, chunk_state in _iter_state_chunks(
        state=state,
        chunk_size=policy_state_chunk_size,
    ):
        mask = state_ids.ge(start) & state_ids.lt(end)
        if not bool(mask.any()):
            continue

        policy_output = _call_policy(
            policy=policy,
            features=features,
            graph_context=graph_context,
            state=chunk_state,
        )
        chunk_positions.append(query_positions[mask])
        chunk_values.append(
            _gather_edge_log_flow(
                policy_output=policy_output,
                state=chunk_state,
                graph_context=graph_context,
                state_ids=state_ids[mask] - int(start),
                edge_ids=edge_ids[mask],
            )
        )

    if not chunk_values:
        raise ValueError("No transition rows matched any state chunk.")

    positions = torch.cat(chunk_positions, dim=0)
    values = torch.cat(chunk_values, dim=0)
    order = torch.argsort(positions)
    return values.index_select(0, order)


def _state_log_flow_chunked(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    state: StateBatch,
    policy_state_chunk_size: int | None,
) -> Tensor:
    return _policy_vector_chunked(
        policy=policy,
        features=features,
        graph_context=graph_context,
        state=state,
        policy_state_chunk_size=policy_state_chunk_size,
        extractor=lambda policy_output, chunk_state: state_log_flow_from_policy_output(
            policy_output=policy_output,
            state=chunk_state,
            graph_context=graph_context,
        ),
    )


def _stop_log_flow_chunked(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    state: StateBatch,
    policy_state_chunk_size: int | None,
) -> Tensor:
    return _policy_vector_chunked(
        policy=policy,
        features=features,
        graph_context=graph_context,
        state=state,
        policy_state_chunk_size=policy_state_chunk_size,
        extractor=lambda policy_output, _chunk_state: _get_stop_log_flow(policy_output),
    )


def _policy_vector_chunked(
    *,
    policy: nn.Module,
    features: FeatureBank,
    graph_context: GraphContext,
    state: StateBatch,
    policy_state_chunk_size: int | None,
    extractor: Callable[[object, StateBatch], Tensor],
) -> Tensor:
    pieces: list[Tensor] = []

    for _, _, chunk_state in _iter_state_chunks(
        state=state,
        chunk_size=policy_state_chunk_size,
    ):
        policy_output = _call_policy(
            policy=policy,
            features=features,
            graph_context=graph_context,
            state=chunk_state,
        )
        pieces.append(extractor(policy_output, chunk_state))

    if not pieces:
        return torch.empty(0, dtype=torch.float32, device=state.device)

    return torch.cat(pieces, dim=0)


def _iter_state_chunks(
    *,
    state: StateBatch,
    chunk_size: int | None,
):
    num_states = int(state.num_states)
    if num_states == 0:
        return

    if chunk_size is None or chunk_size >= num_states:
        yield 0, num_states, state
        return

    for start in range(0, num_states, int(chunk_size)):
        end = min(start + int(chunk_size), num_states)
        row_ids = torch.arange(
            start,
            end,
            dtype=torch.long,
            device=state.device,
        )
        yield start, end, state.take(row_ids)


def _call_reward(
    *,
    reward_model: nn.Module,
    graph_context: GraphContext,
    target_context: TargetContext,
    state: StateBatch,
) -> object:
    return reward_model(
        state=state,
        graph_context=graph_context,
        target_context=target_context,
    )


def _get_stop_log_flow(policy_output: object) -> Tensor:
    value = getattr(policy_output, "stop_log_flow", None)
    if value is None:
        raise AttributeError("policy_output must define stop_log_flow.")
    if not isinstance(value, torch.Tensor):
        raise TypeError("policy_output.stop_log_flow must be a Tensor.")
    return value.view(-1)


def _get_edge_log_flow(policy_output: object) -> Tensor:
    value = getattr(policy_output, "edge_log_flow", None)
    if value is None:
        raise AttributeError("policy_output must define edge_log_flow.")
    if not isinstance(value, torch.Tensor):
        raise TypeError("policy_output.edge_log_flow must be a Tensor.")
    return value.view(-1)


def _get_action_space(
    *,
    policy_output: object,
    state: StateBatch,
    graph_context: GraphContext,
) -> ActionSpace:
    value = getattr(policy_output, "action_space", None)
    if value is not None:
        return value
    return state.action_space(graph_context)


def _get_log_reward(reward_output: object) -> Tensor:
    value = getattr(reward_output, "log_reward", None)
    if value is None:
        raise AttributeError("reward_output must define log_reward.")
    if not isinstance(value, torch.Tensor):
        raise TypeError("reward_output.log_reward must be a Tensor.")
    return value.view(-1)


def _get_reward_valid_mask(
    *,
    reward_output: object,
    like: Tensor,
) -> Tensor:
    value = getattr(reward_output, "valid_mask", None)

    if value is None:
        return torch.isfinite(like)

    if not isinstance(value, torch.Tensor):
        raise TypeError("reward_output.valid_mask must be a Tensor if provided.")

    return value.to(device=like.device, dtype=torch.bool).view(-1)


def _as_1d_long(
    value: Tensor,
    *,
    device: torch.device,
    name: str,
) -> Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor.")
    return value.to(device=device, dtype=torch.long).view(-1)


def _infer_device(
    *,
    nonterminal: NonterminalTransitionBatch | None,
    terminal: TerminalTransitionBatch | None,
) -> torch.device:
    if nonterminal is not None:
        return nonterminal.device
    if terminal is not None:
        return terminal.device
    return torch.device("cpu")


def _term_metrics(
    *,
    prefix: str,
    terms: MatchingTerms,
    loss: Tensor,
) -> dict[str, float]:
    valid = terms.valid_mask

    if terms.num_units == 0 or not bool(valid.any()):
        return _empty_metrics(prefix)

    residual = terms.residual[valid].detach()
    loss_units = terms.loss_units[valid].detach()

    return {
        f"{prefix}/loss": _to_float(loss),
        f"{prefix}/num_units": float(terms.num_units),
        f"{prefix}/num_valid": float(int(valid.sum().item())),
        f"{prefix}/residual_mean": _to_float(residual.mean()),
        f"{prefix}/residual_abs_mean": _to_float(residual.abs().mean()),
        f"{prefix}/residual_rmse": _to_float(residual.square().mean().sqrt()),
        f"{prefix}/loss_unit_mean": _to_float(loss_units.mean()),
    }


def _empty_metrics(prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/loss": 0.0,
        f"{prefix}/num_units": 0.0,
        f"{prefix}/num_valid": 0.0,
        f"{prefix}/residual_mean": 0.0,
        f"{prefix}/residual_abs_mean": 0.0,
        f"{prefix}/residual_rmse": 0.0,
        f"{prefix}/loss_unit_mean": 0.0,
    }


def _to_float(value: Tensor) -> float:
    return float(value.detach().float().cpu().item())


def _optional_positive_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive when provided.")
    return int(value)


__all__ = [
    "BackwardKernel",
    "EdgeFlowMatchingObjective",
    "MatchingTerms",
    "NonterminalTransitionBatch",
    "ResidualLoss",
    "TerminalTransitionBatch",
    "compute_log_backward",
    "count_legal_backward_parents",
    "masked_mean_or_zero",
    "nonterminal_edge_flow_matching",
    "residual_loss_units",
    "segment_logsumexp",
    "state_log_flow_from_policy_output",
    "terminal_edge_reward_matching",
]
