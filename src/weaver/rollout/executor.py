from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.graph.ops import (
    build_anchor_induced_edge_mask,
    compute_uniform_nonroot_backward_removals,
)
from src.weaver.policy import PolicyOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.state import RolloutState

from .sampling import CONTINUE_ACTION, STOP_ACTION, sample_policy_actions
from .schema import StepResult


@dataclass(frozen=True, slots=True)
class StepContext:
    """
    Per-step rollout masks shared by executor, diagnostics, and auxiliaries.

    These masks are derived from the current canonical state snapshot. In
    particular, has_candidate and can_expand use the frontier induced by
    V_s = anchors union endpoints(E_s).
    """

    t: int
    active_mask: torch.Tensor
    remaining_budget: torch.Tensor
    has_candidate: torch.Tensor
    budget_exhausted: torch.Tensor
    can_expand: torch.Tensor


@dataclass(frozen=True, slots=True)
class StepGraphContext:
    """
    Physical batched-graph context for one executor.

    Coordinate convention:
        graph ids: physical graph ids in the current RetrievalBatch
        node ids: physical node ids in the current RetrievalBatch
        edge ids: physical edge ids in the current RetrievalBatch
    """

    edge_index: torch.Tensor
    edge_batch: torch.Tensor
    anchor_mask: torch.Tensor
    root_edges: torch.Tensor
    num_nodes: int
    num_edges: int
    num_graphs: int

    @classmethod
    def from_batch(cls, batch: RetrievalBatch) -> StepGraphContext:
        edge_index = batch.edge_index.to(dtype=torch.long)
        device = edge_index.device

        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

        num_nodes = int(batch.num_nodes_total)
        num_edges = int(batch.edge_index.size(1))
        num_graphs = int(batch.num_graphs)

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
            )
        if edge_index.size(1) != num_edges:
            raise ValueError(
                f"edge_index has {edge_index.size(1)} edges, expected {num_edges}."
            )
        if edge_batch.numel() != num_edges:
            raise ValueError(
                f"edge_batch has length {edge_batch.numel()}, expected {num_edges}."
            )

        anchor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        anchor_ids = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)

        if anchor_ids.numel() > 0:
            _validate_node_ids(
                anchor_ids,
                num_nodes=num_nodes,
                name="anchor_node_ids",
            )
            anchor_mask[anchor_ids] = True

        root_edges = build_anchor_induced_edge_mask(
            edge_index=edge_index,
            anchor_mask=anchor_mask,
        )

        if root_edges.shape != (num_edges,):
            raise ValueError(
                f"root_edges must have shape [{num_edges}], got {tuple(root_edges.shape)}."
            )

        return cls(
            edge_index=edge_index,
            edge_batch=edge_batch,
            anchor_mask=anchor_mask,
            root_edges=root_edges,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )


class FusedStepExecutor:
    """
    One-step transition for static-batch / dynamic-rollout execution.

    Static graph ids remain in the original RetrievalBatch. Dynamic rollout rows
    are indexed by state.rollout_to_graph and by PolicyOutput.candidate_batch_ids.
    """

    def __init__(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        validate_frontier: bool = False,
    ) -> None:
        self.batch = retrieval_batch
        self.graph = StepGraphContext.from_batch(retrieval_batch)
        self.reward_model = reward_model
        self.validate_frontier = bool(validate_frontier)

    def execute_step(
        self,
        *,
        step_out: PolicyOutput,
        state: RolloutState,
        step_context: StepContext,
        temperature: float,
        stop_now_reward: TerminalRewardOutput | None = None,
    ) -> StepResult:
        device = self.graph.edge_index.device
        num_rollouts = int(state.num_rollouts)

        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        active = step_context.active_mask.to(device=device, dtype=torch.bool)
        if active.shape != (num_rollouts,):
            raise ValueError(
                f"active must have shape [{num_rollouts}], got {tuple(active.shape)}."
            )

        validate_policy_output(
            step_out,
            num_graphs=num_rollouts,
            num_edges=self.graph.num_edges,
            device=device,
        )
        self._validate_state_invariants(state)

        if self.validate_frontier:
            validate_fused_frontier_candidates(
                candidate_edge_ids=step_out.candidate_edge_ids,
                candidate_batch_ids=step_out.candidate_batch_ids,
                state=state,
                graph=self.graph,
            )

        can_expand = step_context.can_expand.to(device=device, dtype=torch.bool)
        if can_expand.shape != (num_rollouts,):
            raise ValueError(
                f"can_expand must have shape [{num_rollouts}], "
                f"got {tuple(can_expand.shape)}."
            )
        if bool((can_expand & ~active).any()):
            raise ValueError("can_expand cannot be true for inactive rollout rows.")

        action = sample_policy_actions(
            stop_logits=step_out.stop_logits,
            edge_logits=step_out.edge_logits,
            candidate_edge_ids=step_out.candidate_edge_ids,
            candidate_batch_ids=step_out.candidate_batch_ids,
            active=active,
            can_expand=can_expand,
            temperature=float(temperature),
            batch_size=num_rollouts,
        )

        action_type = action.action_type.to(device=device, dtype=torch.long)
        action_log_prob = action.target_log_prob.to(device=device, dtype=torch.float32)
        chosen_edges_by_rollout = action.chosen_edges.to(
            device=device, dtype=torch.long
        )

        _validate_step_vector(action_type, size=num_rollouts, name="action_type")
        _validate_step_vector(
            action_log_prob,
            size=num_rollouts,
            name="action.target_log_prob",
        )
        _validate_step_vector(
            chosen_edges_by_rollout,
            size=num_rollouts,
            name="action.chosen_edges",
        )

        continue_mask = active & action_type.eq(CONTINUE_ACTION)
        stop_mask = active & action_type.eq(STOP_ACTION)

        log_pf = torch.zeros(num_rollouts, dtype=torch.float32, device=device)
        log_pb = torch.zeros_like(log_pf)
        log_pf[active] = action_log_prob[active]

        selected_edge_ids = torch.full(
            (num_rollouts,),
            -1,
            dtype=torch.long,
            device=device,
        )

        terminal = TerminalStepTensors.zeros(
            num_graphs=num_rollouts,
            device=device,
        )

        self._continue_with_edge(
            state=state,
            continue_mask=continue_mask,
            log_pb=log_pb,
            chosen_edges_by_rollout=chosen_edges_by_rollout,
            selected_edge_ids=selected_edge_ids,
        )

        self._stop(
            state=state,
            stop_mask=stop_mask,
            terminal=terminal,
            stop_now_reward=stop_now_reward,
        )

        return StepResult(
            log_pf=log_pf,
            log_pb=log_pb,
            action_type=action_type,
            continue_mask=continue_mask,
            stop_mask=stop_mask,
            selected_edge_ids=selected_edge_ids,
            terminal_log_reward=terminal.log_reward,
            terminal_answer_f1=terminal.answer_f1,
            terminal_complexity_penalty=terminal.complexity_penalty,
            terminal_base_log_reward=terminal.base_log_reward,
            terminal_utility=terminal.utility,
            terminal_expanded_edge_count=terminal.expanded_edge_count,
            terminal_answer_degree_excess=terminal.answer_degree_excess,
        )

    def _continue_with_edge(
        self,
        *,
        state: RolloutState,
        continue_mask: torch.Tensor,
        log_pb: torch.Tensor,
        chosen_edges_by_rollout: torch.Tensor,
        selected_edge_ids: torch.Tensor,
    ) -> None:
        rollout_ids = continue_mask.nonzero(as_tuple=False).flatten()
        if rollout_ids.numel() == 0:
            return

        chosen_edges = chosen_edges_by_rollout.index_select(0, rollout_ids).to(
            device=self.graph.edge_index.device,
            dtype=torch.long,
        )

        if chosen_edges.shape != rollout_ids.shape:
            raise RuntimeError(
                f"sampler returned {chosen_edges.numel()} edges for "
                f"{rollout_ids.numel()} continuing rollout rows."
            )
        if bool((chosen_edges < 0).any()):
            bad = rollout_ids[chosen_edges < 0]
            raise RuntimeError(
                "Continue actions must carry selected edge ids for rollout ids "
                f"{bad.tolist()}."
            )

        _validate_edge_ids(
            chosen_edges,
            num_edges=self.graph.num_edges,
            name="chosen_edges",
        )
        selected_edge_batch = self.graph.edge_batch.index_select(0, chosen_edges)
        expected_graph = state.rollout_to_graph.to(
            device=selected_edge_batch.device,
            dtype=torch.long,
        ).index_select(0, rollout_ids)
        if not torch.equal(selected_edge_batch, expected_graph):
            raise RuntimeError(
                "selected edge ids must belong to their rollout rows' original "
                f"graphs, got edge_batch={selected_edge_batch.tolist()} and "
                f"rollout_to_graph={expected_graph.tolist()}."
            )

        selected_edge_ids[rollout_ids] = chosen_edges

        state.apply_expansion(
            rollout_ids=rollout_ids,
            chosen_edges=chosen_edges,
            edge_index=self.graph.edge_index,
        )

        log_pb[rollout_ids] = self._uniform_log_pb_after_continue(
            state=state,
            rollout_ids=rollout_ids,
            selected_edge_ids=chosen_edges,
        )

    def _uniform_log_pb_after_continue(
        self,
        *,
        state: RolloutState,
        rollout_ids: torch.Tensor,
        selected_edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for rollout_id_tensor, edge_id_tensor in zip(rollout_ids, selected_edge_ids):
            rollout_id = int(rollout_id_tensor.item())
            edge_id = int(edge_id_tensor.item())
            static_graph_id = int(state.rollout_to_graph[rollout_id].item())

            removable_mask, counts = compute_uniform_nonroot_backward_removals(
                active_edges=state.active_edges[rollout_id],
                edge_index=self.graph.edge_index,
                anchor_mask=state.anchor_nodes[rollout_id],
                edge_batch=self.graph.edge_batch,
                num_graphs=self.graph.num_graphs,
                root_edges=state.root_edges[rollout_id],
                validate=False,
            )
            if not bool(removable_mask[edge_id]):
                raise RuntimeError(
                    "Selected Expand edge must be removable from the child under "
                    "the canonical backward parent definition, got edge id="
                    f"{edge_id} for rollout id={rollout_id}."
                )

            count = counts[static_graph_id].to(dtype=torch.float32)
            if bool(count.lt(1.0)):
                raise RuntimeError(
                    "No valid non-root backward removal after Continue for "
                    f"rollout id={rollout_id}, original graph id={static_graph_id}."
                )
            values.append(-torch.log(count))

        if not values:
            return torch.empty(
                0, dtype=torch.float32, device=self.graph.edge_index.device
            )
        return torch.stack(values, dim=0).to(
            device=self.graph.edge_index.device,
            dtype=torch.float32,
        )

    def _stop(
        self,
        *,
        state: RolloutState,
        stop_mask: torch.Tensor,
        terminal: "TerminalStepTensors",
        stop_now_reward: TerminalRewardOutput | None,
    ) -> None:
        rollout_ids = stop_mask.nonzero(as_tuple=False).flatten()
        if rollout_ids.numel() == 0:
            return

        reward = (
            stop_now_reward
            if stop_now_reward is not None
            else self.reward_model.evaluate_terminal_state(
                retrieval_batch=self.batch,
                active_nodes=state.active_nodes,
                active_edges=state.active_edges,
                state=state,
            )
        )

        validate_terminal_reward(reward, num_graphs=state.num_rollouts)
        self._write_terminal_reward(
            terminal=terminal,
            reward=reward,
            rollout_ids=rollout_ids,
        )

    def _write_terminal_reward(
        self,
        *,
        terminal: "TerminalStepTensors",
        reward: TerminalRewardOutput,
        rollout_ids: torch.Tensor,
    ) -> None:
        reward_ids = rollout_ids.to(device=reward.log_reward.device)

        for target, source in (
            (terminal.log_reward, reward.log_reward),
            (terminal.answer_f1, reward.answer_f1),
            (terminal.complexity_penalty, reward.complexity_penalty),
            (terminal.base_log_reward, reward.base_log_reward),
            (terminal.utility, reward.utility),
            (terminal.expanded_edge_count, reward.expanded_edge_count),
            (terminal.answer_degree_excess, reward.answer_degree_excess),
        ):
            target[rollout_ids] = source.index_select(0, reward_ids).to(
                device=target.device,
                dtype=target.dtype,
            )

    def _validate_state_invariants(self, state: RolloutState) -> None:
        device = self.graph.edge_index.device
        num_rollouts = int(state.num_rollouts)

        active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
        active_edges = state.active_edges.to(device=device, dtype=torch.bool)
        root_edges = state.root_edges.to(device=device, dtype=torch.bool)
        anchor_nodes = state.anchor_nodes.to(device=device, dtype=torch.bool)
        rollout_to_graph = state.rollout_to_graph.to(device=device, dtype=torch.long)

        if active_nodes.shape != (num_rollouts, self.graph.num_nodes):
            raise RuntimeError(
                "state.active_nodes shape does not match fused graph nodes: "
                f"{tuple(active_nodes.shape)} != "
                f"{(num_rollouts, self.graph.num_nodes)}."
            )
        if active_edges.shape != (num_rollouts, self.graph.num_edges):
            raise RuntimeError(
                "state.active_edges shape does not match fused graph edges: "
                f"{tuple(active_edges.shape)} != "
                f"{(num_rollouts, self.graph.num_edges)}."
            )
        if root_edges.shape != active_edges.shape:
            raise RuntimeError("state.root_edges shape must match state.active_edges.")
        if anchor_nodes.shape != active_nodes.shape:
            raise RuntimeError(
                "state.anchor_nodes shape must match state.active_nodes."
            )
        if rollout_to_graph.shape != (num_rollouts,):
            raise RuntimeError(
                f"state.rollout_to_graph must have shape [{num_rollouts}], "
                f"got {tuple(rollout_to_graph.shape)}."
            )

        if bool((root_edges & ~active_edges).any()):
            bad = (root_edges & ~active_edges).nonzero(as_tuple=False)
            raise RuntimeError(
                "state.active_edges must keep every root edge active, "
                f"but inactive fused root coordinates were found: {bad.tolist()}."
            )

        edge_belongs = self.graph.edge_batch.view(1, -1).eq(
            rollout_to_graph.view(-1, 1)
        )
        if bool((active_edges & ~edge_belongs).any()):
            bad = (active_edges & ~edge_belongs).nonzero(as_tuple=False)
            raise RuntimeError(
                "RolloutState active_edges must stay inside each row's original "
                f"graph, got fused coordinates={bad.tolist()}."
            )

        src, dst = self.graph.edge_index
        expected_root = (
            anchor_nodes.index_select(1, src)
            & anchor_nodes.index_select(1, dst)
            & edge_belongs
        )
        if not torch.equal(root_edges, expected_root):
            raise RuntimeError("state.root_edges does not match fused anchor roots.")

        expected_nodes = anchor_nodes.clone()
        rollout_ids, edge_ids = active_edges.nonzero(as_tuple=True)
        if edge_ids.numel() > 0:
            expected_nodes[rollout_ids, src.index_select(0, edge_ids)] = True
            expected_nodes[rollout_ids, dst.index_select(0, edge_ids)] = True

        if not torch.equal(active_nodes, expected_nodes):
            extra = (active_nodes & ~expected_nodes).nonzero(as_tuple=False)
            missing = (expected_nodes & ~active_nodes).nonzero(as_tuple=False)
            raise RuntimeError(
                "state.active_nodes must equal anchors union endpoints(active_edges); "
                f"extra coordinates={extra.tolist()}, missing coordinates="
                f"{missing.tolist()}."
            )


@dataclass(slots=True)
class TerminalStepTensors:
    log_reward: torch.Tensor
    answer_f1: torch.Tensor
    complexity_penalty: torch.Tensor
    base_log_reward: torch.Tensor
    utility: torch.Tensor
    expanded_edge_count: torch.Tensor
    answer_degree_excess: torch.Tensor

    @classmethod
    def zeros(
        cls,
        *,
        num_graphs: int,
        device: torch.device,
    ) -> TerminalStepTensors:
        zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)

        return cls(
            log_reward=zeros.clone(),
            answer_f1=zeros.clone(),
            complexity_penalty=zeros.clone(),
            base_log_reward=zeros.clone(),
            utility=zeros.clone(),
            expanded_edge_count=zeros.clone(),
            answer_degree_excess=zeros.clone(),
        )


def has_candidate(
    *,
    candidate_batch_ids: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    candidate_batch_ids = candidate_batch_ids.to(device=device, dtype=torch.long)

    if candidate_batch_ids.numel() == 0:
        return torch.zeros(num_graphs, dtype=torch.bool, device=device)

    _validate_graph_ids(
        candidate_batch_ids,
        num_graphs=int(num_graphs),
        name="candidate_batch_ids",
    )

    return torch.bincount(candidate_batch_ids, minlength=int(num_graphs)).gt(0)


def budget_exhausted_mask(
    remaining_budget: int | torch.Tensor,
    *,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(remaining_budget, int):
        return torch.full(
            (num_graphs,),
            remaining_budget <= 0,
            dtype=torch.bool,
            device=device,
        )

    remaining_budget = remaining_budget.to(device=device)

    if remaining_budget.ndim == 0:
        return torch.full(
            (num_graphs,),
            bool(remaining_budget.item() <= 0),
            dtype=torch.bool,
            device=device,
        )

    if remaining_budget.shape != (num_graphs,):
        raise ValueError(
            f"remaining_budget must be scalar or shape [{num_graphs}], "
            f"got {tuple(remaining_budget.shape)}."
        )

    return remaining_budget <= 0


def validate_policy_output(
    step_out: PolicyOutput,
    *,
    num_graphs: int,
    num_edges: int,
    device: torch.device,
) -> None:
    _validate_step_vector(
        step_out.stop_logits,
        size=num_graphs,
        name="PolicyOutput.stop_logits",
    )
    _validate_step_vector(
        step_out.expand_logits,
        size=num_graphs,
        name="PolicyOutput.expand_logits",
    )

    if step_out.stop_logits.device != device:
        raise ValueError(
            f"PolicyOutput.stop_logits is on {step_out.stop_logits.device}, expected {device}."
        )
    if step_out.expand_logits.device != device:
        raise ValueError(
            f"PolicyOutput.expand_logits is on {step_out.expand_logits.device}, expected {device}."
        )

    validate_candidate_tensors(
        edge_logits=step_out.edge_logits,
        candidate_edge_ids=step_out.candidate_edge_ids,
        candidate_batch_ids=step_out.candidate_batch_ids,
        num_graphs=num_graphs,
        num_edges=num_edges,
        device=device,
    )


def validate_candidate_tensors(
    *,
    edge_logits: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    num_graphs: int,
    num_edges: int,
    device: torch.device,
) -> None:
    if edge_logits.device != device:
        raise ValueError(f"edge_logits is on {edge_logits.device}, expected {device}.")
    if candidate_edge_ids.device != device:
        raise ValueError(
            f"candidate_edge_ids is on {candidate_edge_ids.device}, expected {device}."
        )
    if candidate_batch_ids.device != device:
        raise ValueError(
            f"candidate_batch_ids is on {candidate_batch_ids.device}, expected {device}."
        )

    if edge_logits.ndim != 1:
        raise ValueError(f"edge_logits must be 1D, got {tuple(edge_logits.shape)}.")
    if candidate_edge_ids.ndim != 1:
        raise ValueError(
            f"candidate_edge_ids must be 1D, got {tuple(candidate_edge_ids.shape)}."
        )
    if candidate_batch_ids.ndim != 1:
        raise ValueError(
            f"candidate_batch_ids must be 1D, got {tuple(candidate_batch_ids.shape)}."
        )

    if candidate_edge_ids.dtype != torch.long:
        raise TypeError(
            f"candidate_edge_ids must be torch.long, got {candidate_edge_ids.dtype}."
        )
    if candidate_batch_ids.dtype != torch.long:
        raise TypeError(
            f"candidate_batch_ids must be torch.long, got {candidate_batch_ids.dtype}."
        )

    if not (
        edge_logits.numel() == candidate_edge_ids.numel() == candidate_batch_ids.numel()
    ):
        raise ValueError(
            "candidate tensors must have the same length: "
            f"edge_logits={edge_logits.numel()}, "
            f"candidate_edge_ids={candidate_edge_ids.numel()}, "
            f"candidate_batch_ids={candidate_batch_ids.numel()}."
        )

    _validate_edge_ids(
        candidate_edge_ids,
        num_edges=num_edges,
        name="candidate_edge_ids",
    )
    _validate_graph_ids(
        candidate_batch_ids,
        num_graphs=num_graphs,
        name="candidate_batch_ids",
    )


def validate_fused_frontier_candidates(
    *,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    state: RolloutState,
    graph: StepGraphContext,
) -> None:
    """
    Debug validator for fused frontier legality.

    Every candidate must satisfy, in its rollout row:
        e belongs to rollout_to_graph[row]
        e not in E_s[row]
        and at least one endpoint is in V_s[row].
    """
    if candidate_edge_ids.numel() == 0:
        return

    device = graph.edge_index.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long)
    rollout_ids = candidate_batch_ids.to(device=device, dtype=torch.long)
    _validate_edge_ids(edge_ids, num_edges=graph.num_edges, name="candidate_edge_ids")
    _validate_graph_ids(
        rollout_ids,
        num_graphs=state.num_rollouts,
        name="candidate_batch_ids",
    )

    src = graph.edge_index[0].index_select(0, edge_ids)
    dst = graph.edge_index[1].index_select(0, edge_ids)

    active_nodes = state.active_nodes.to(device=device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=device, dtype=torch.bool)
    rollout_to_graph = state.rollout_to_graph.to(device=device, dtype=torch.long)

    already_active = active_edges[rollout_ids, edge_ids]
    incident = active_nodes[rollout_ids, src] | active_nodes[rollout_ids, dst]
    belongs = graph.edge_batch.index_select(0, edge_ids).eq(
        rollout_to_graph.index_select(0, rollout_ids)
    )

    invalid = already_active | ~incident | ~belongs
    if bool(invalid.any()):
        bad_edges = edge_ids[invalid]
        bad_rows = rollout_ids[invalid]
        raise RuntimeError(
            "Invalid fused frontier candidates: "
            f"rollout_ids={bad_rows.tolist()}, edge_ids={bad_edges.tolist()}."
        )


def validate_terminal_reward(
    reward: TerminalRewardOutput,
    *,
    num_graphs: int,
) -> None:
    for name, tensor in {
        "reward.log_reward": reward.log_reward,
        "reward.answer_f1": reward.answer_f1,
        "reward.complexity_penalty": reward.complexity_penalty,
        "reward.base_log_reward": reward.base_log_reward,
        "reward.utility": reward.utility,
        "reward.supported_answer_precision": reward.supported_answer_precision,
        "reward.supported_answer_f_beta": reward.supported_answer_f_beta,
        "reward.supported_retrieved_count": reward.supported_retrieved_count,
        "reward.expanded_edge_count": reward.expanded_edge_count,
        "reward.answer_degree_excess": reward.answer_degree_excess,
    }.items():
        _validate_step_vector(
            tensor,
            size=num_graphs,
            name=name,
        )


def _validate_step_vector(
    tensor: torch.Tensor,
    *,
    size: int,
    name: str,
) -> None:
    if tensor.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}], got {tuple(tensor.shape)}.")


def _validate_node_ids(
    node_ids: torch.Tensor,
    *,
    num_nodes: int,
    name: str,
) -> None:
    if node_ids.numel() == 0:
        return

    min_id = int(node_ids.min())
    max_id = int(node_ids.max())

    if min_id < 0 or max_id >= int(num_nodes):
        raise ValueError(
            f"{name} must contain physical node ids in current batch: "
            f"min={min_id}, max={max_id}, num_nodes={num_nodes}."
        )


def _validate_edge_ids(
    edge_ids: torch.Tensor,
    *,
    num_edges: int,
    name: str,
) -> None:
    if edge_ids.numel() == 0:
        return

    min_id = int(edge_ids.min())
    max_id = int(edge_ids.max())

    if min_id < 0 or max_id >= int(num_edges):
        raise ValueError(
            f"{name} must contain physical edge ids in current batch: "
            f"min={min_id}, max={max_id}, num_edges={num_edges}."
        )


def _validate_graph_ids(
    graph_ids: torch.Tensor,
    *,
    num_graphs: int,
    name: str,
) -> None:
    if graph_ids.numel() == 0:
        return

    min_id = int(graph_ids.min())
    max_id = int(graph_ids.max())

    if min_id < 0 or max_id >= int(num_graphs):
        raise ValueError(
            f"{name} must contain physical graph ids in current batch: "
            f"min={min_id}, max={max_id}, num_graphs={num_graphs}."
        )


__all__ = [
    "FusedStepExecutor",
    "StepContext",
    "StepGraphContext",
    "TerminalStepTensors",
    "budget_exhausted_mask",
    "has_candidate",
    "validate_candidate_tensors",
    "validate_fused_frontier_candidates",
    "validate_policy_output",
    "validate_terminal_reward",
]
