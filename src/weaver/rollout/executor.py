from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.data.schema import RetrievalBatch
from src.graph.ops import (
    build_anchor_induced_edge_mask,
    compute_uniform_nonroot_backward_removals,
)
from src.weaver.policy import CandidateEdges, PolicyStepOutput
from src.weaver.reward import RewardModel, TerminalRewardOutput
from src.weaver.state import State

from .sampling import CONTINUE_ACTION, STOP_ACTION, sample_policy_actions
from .schema import StepResult


@dataclass(frozen=True)
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
    root_edge_mask: torch.Tensor
    num_nodes: int
    num_edges: int
    num_graphs: int

    @classmethod
    def from_batch(cls, batch: RetrievalBatch) -> StepGraphContext:
        edge_index = batch.edge_index.to(dtype=torch.long)
        device = edge_index.device

        edge_batch = batch.edge_batch.to(device=device, dtype=torch.long)

        num_nodes = int(batch.num_nodes_total)
        num_edges = int(batch.num_edges_total)
        num_graphs = int(batch.num_graphs)

        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}."
            )
        if edge_index.size(1) != num_edges:
            raise ValueError(
                f"edge_index has {edge_index.size(1)} edges, "
                f"but num_edges_total={num_edges}."
            )
        if edge_batch.numel() != num_edges:
            raise ValueError(
                f"edge_batch has length {edge_batch.numel()}, "
                f"but num_edges_total={num_edges}."
            )

        anchor_ids = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        anchor_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        if anchor_ids.numel() > 0:
            _validate_node_ids(
                anchor_ids,
                num_nodes=num_nodes,
                name="anchor_node_ids",
            )
            anchor_mask[anchor_ids] = True

        root_edge_mask = build_anchor_induced_edge_mask(edge_index, anchor_mask)
        if root_edge_mask.shape != (num_edges,):
            raise ValueError(
                f"root_edge_mask must have shape [{num_edges}], "
                f"got {tuple(root_edge_mask.shape)}."
            )

        return cls(
            edge_index=edge_index,
            edge_batch=edge_batch,
            anchor_mask=anchor_mask,
            root_edge_mask=root_edge_mask,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_graphs=num_graphs,
        )


class BackwardPolicy(Protocol):
    def log_prob_after_continue(
        self,
        *,
        state: State,
        graph: StepGraphContext,
        continue_graph_ids: torch.Tensor,
    ) -> torch.Tensor: ...


class UniformRemovalBackwardPolicy:
    """
    Uniform backward policy over removable non-root active edges.

        log P_B(parent | child) = -log |R(child)|

    Here child is the state after one Expand-edge transition.
    """

    def log_prob_after_continue(
        self,
        *,
        state: State,
        graph: StepGraphContext,
        continue_graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, counts = compute_uniform_nonroot_backward_removals(
            active_edges=state.active_edges,
            edge_index=graph.edge_index,
            is_anchor_mask=graph.anchor_mask,
            edge_batch=graph.edge_batch,
            num_graphs=graph.num_graphs,
            root_active_edges=state.root_active_edges.to(
                device=graph.edge_index.device,
                dtype=torch.bool,
            ),
            validate=False,
        )

        continue_graph_ids = continue_graph_ids.to(
            device=counts.device,
            dtype=torch.long,
        )
        _validate_graph_ids(
            continue_graph_ids,
            num_graphs=graph.num_graphs,
            name="continue_graph_ids",
        )

        selected_counts = counts.index_select(0, continue_graph_ids)
        if bool((selected_counts < 1).any()):
            bad = continue_graph_ids[selected_counts < 1]
            raise RuntimeError(
                "No valid non-root backward removal after Continue for "
                f"physical graph ids={bad.tolist()}."
            )

        return -torch.log(selected_counts.to(dtype=torch.float32))


class StepExecutor:
    """
    One-step environment transition operator.

    Action convention:
        CONTINUE_ACTION = 0 means Expand one frontier edge.
        STOP_ACTION     = 1 means terminate the graph.

    Forward policy:
        Stop/Expand compete at option level.
        Expand selects one edge from the conditional frontier-edge policy.

    This executor does not know about coverage guides, proposals, teachers, or
    VIGOR. It samples from the target policy and mutates State only for Expand.
    """

    def __init__(
        self,
        *,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        backward_policy: BackwardPolicy | None = None,
        validate_frontier: bool = False,
    ) -> None:
        self.batch = retrieval_batch
        self.graph = StepGraphContext.from_batch(retrieval_batch)
        self.reward_model = reward_model
        self.backward_policy = backward_policy or UniformRemovalBackwardPolicy()
        self.validate_frontier = bool(validate_frontier)

    def execute_step(
        self,
        *,
        step_out: PolicyStepOutput,
        state: State,
        active: torch.Tensor,
        temperature: float,
        remaining_budget: torch.Tensor | None = None,
        stop_now_reward: TerminalRewardOutput | None = None,
    ) -> StepResult:
        device = self.graph.edge_index.device

        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}.")

        active = active.to(device=device, dtype=torch.bool)
        if active.shape != (self.graph.num_graphs,):
            raise ValueError(
                f"active must have shape [{self.graph.num_graphs}], "
                f"got {tuple(active.shape)}."
            )

        validate_policy_step_output(
            step_out,
            num_graphs=self.graph.num_graphs,
            num_edges=self.graph.num_edges,
            device=device,
        )
        self._validate_state_root(state)

        if self.validate_frontier:
            validate_frontier_candidates(
                candidates=step_out.candidates,
                state=state,
                edge_index=self.graph.edge_index,
            )

        has_edge = has_candidate(
            candidate_batch_index=step_out.candidates.batch_index,
            num_graphs=self.graph.num_graphs,
            device=device,
        )

        if remaining_budget is None:
            remaining_budget = state.remaining_budget_per_graph(
                edge_batch=self.graph.edge_batch,
                num_graphs=self.graph.num_graphs,
            )

        exhausted = budget_exhausted_mask(
            remaining_budget,
            num_graphs=self.graph.num_graphs,
            device=device,
        )

        can_expand = active & has_edge & ~exhausted

        action = sample_policy_actions(
            stop_logits=step_out.stop_logits,
            expand_logits=step_out.expand_logits,
            candidates=step_out.candidates,
            active=active,
            can_expand=can_expand,
            temperature=float(temperature),
            batch_size=self.graph.num_graphs,
        )

        action_type = action.action_type.to(device=device, dtype=torch.long)
        action_log_prob = action.target_log_prob.to(device=device, dtype=torch.float32)
        chosen_edges_by_graph = action.chosen_edges.to(device=device, dtype=torch.long)

        _validate_step_vector(
            action_type,
            size=self.graph.num_graphs,
            name="action_type",
        )
        _validate_step_vector(
            action_log_prob,
            size=self.graph.num_graphs,
            name="action.target_log_prob",
        )
        _validate_step_vector(
            chosen_edges_by_graph,
            size=self.graph.num_graphs,
            name="action.chosen_edges",
        )

        continue_mask = active & action_type.eq(CONTINUE_ACTION)
        stop_mask = active & action_type.eq(STOP_ACTION)

        log_pf = torch.zeros(self.graph.num_graphs, dtype=torch.float32, device=device)
        log_pb = torch.zeros_like(log_pf)
        log_pf[active] = action_log_prob[active]

        selected_edge_ids = torch.full(
            (self.graph.num_graphs,),
            -1,
            dtype=torch.long,
            device=device,
        )

        terminal = TerminalStepTensors.zeros(
            num_graphs=self.graph.num_graphs,
            device=device,
        )

        self._continue_with_edge(
            state=state,
            continue_mask=continue_mask,
            log_pb=log_pb,
            chosen_edges_by_graph=chosen_edges_by_graph,
            selected_edge_ids=selected_edge_ids,
        )

        self._stop(
            state=state,
            stop_mask=stop_mask,
            terminal=terminal,
            stop_now_reward=stop_now_reward,
        )

        # Kept as all-zero compatibility field. Delete it once StepResult,
        # RolloutBuffer, and RolloutTraces no longer expose proposal fields.
        proposal_intervention_mask = torch.zeros(
            self.graph.num_graphs,
            dtype=torch.bool,
            device=device,
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
            proposal_intervention_mask=proposal_intervention_mask,
            terminal_edge_penalty=terminal.edge_penalty,
            terminal_base_log_reward=terminal.base_log_reward,
            terminal_utility=terminal.utility,
            terminal_expanded_edge_count=terminal.expanded_edge_count,
            terminal_minimal_edge_count=terminal.minimal_edge_count,
            terminal_minimality_gap=terminal.minimality_gap,
            terminal_minimality_penalty=terminal.minimality_penalty,
        )

    def _continue_with_edge(
        self,
        *,
        state: State,
        continue_mask: torch.Tensor,
        log_pb: torch.Tensor,
        chosen_edges_by_graph: torch.Tensor,
        selected_edge_ids: torch.Tensor,
    ) -> None:
        graph_ids = continue_mask.nonzero(as_tuple=False).view(-1)
        if graph_ids.numel() == 0:
            return

        chosen_edges = chosen_edges_by_graph.index_select(0, graph_ids).to(
            device=self.graph.edge_index.device,
            dtype=torch.long,
        )

        if chosen_edges.shape != graph_ids.shape:
            raise RuntimeError(
                f"sampler returned {chosen_edges.numel()} edges for "
                f"{graph_ids.numel()} continuing graphs."
            )
        if bool((chosen_edges < 0).any()):
            bad = graph_ids[chosen_edges < 0]
            raise RuntimeError(
                "Continue actions must carry selected edge ids for graph ids "
                f"{bad.tolist()}."
            )

        if self.validate_frontier:
            _validate_edge_ids(
                chosen_edges,
                num_edges=self.graph.num_edges,
                name="chosen_edges",
            )

        selected_edge_ids[graph_ids] = chosen_edges

        state.apply_expansion(
            chosen_edges=chosen_edges,
            edge_index=self.graph.edge_index,
        )

        log_pb[graph_ids] = self.backward_policy.log_prob_after_continue(
            state=state,
            graph=self.graph,
            continue_graph_ids=graph_ids,
        )

    def _stop(
        self,
        *,
        state: State,
        stop_mask: torch.Tensor,
        terminal: "TerminalStepTensors",
        stop_now_reward: TerminalRewardOutput | None,
    ) -> None:
        graph_ids = stop_mask.nonzero(as_tuple=False).view(-1)
        if graph_ids.numel() == 0:
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

        validate_terminal_reward(
            reward,
            num_graphs=self.graph.num_graphs,
        )

        reward_graph_ids = graph_ids.to(device=reward.log_reward.device)

        terminal.log_reward[graph_ids] = reward.log_reward.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.log_reward.device, dtype=torch.float32)

        terminal.answer_f1[graph_ids] = reward.answer_f1.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.answer_f1.device, dtype=torch.float32)

        terminal.edge_penalty[graph_ids] = reward.edge_penalty.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.edge_penalty.device, dtype=torch.float32)

        terminal.base_log_reward[graph_ids] = reward.base_log_reward.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.base_log_reward.device, dtype=torch.float32)

        terminal.utility[graph_ids] = reward.utility.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.utility.device, dtype=torch.float32)

        terminal.expanded_edge_count[graph_ids] = (
            reward.expanded_edge_count.index_select(
                0,
                reward_graph_ids,
            ).to(device=terminal.expanded_edge_count.device, dtype=torch.float32)
        )

        terminal.minimal_edge_count[graph_ids] = reward.minimal_edge_count.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.minimal_edge_count.device, dtype=torch.float32)

        terminal.minimality_gap[graph_ids] = reward.minimality_gap.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.minimality_gap.device, dtype=torch.float32)

        terminal.minimality_penalty[graph_ids] = reward.minimality_penalty.index_select(
            0,
            reward_graph_ids,
        ).to(device=terminal.minimality_penalty.device, dtype=torch.float32)

    def _validate_state_root(self, state: State) -> None:
        root = state.root_active_edges.to(
            device=self.graph.root_edge_mask.device,
            dtype=torch.bool,
        )

        if root.shape != self.graph.root_edge_mask.shape:
            raise RuntimeError(
                "state.root_active_edges shape does not match graph.root_edge_mask: "
                f"{tuple(root.shape)} != {tuple(self.graph.root_edge_mask.shape)}."
            )

        if not torch.equal(root, self.graph.root_edge_mask):
            raise RuntimeError(
                "state.root_active_edges does not match graph.root_edge_mask."
            )


@dataclass
class TerminalStepTensors:
    log_reward: torch.Tensor
    answer_f1: torch.Tensor
    edge_penalty: torch.Tensor
    base_log_reward: torch.Tensor
    utility: torch.Tensor
    expanded_edge_count: torch.Tensor
    minimal_edge_count: torch.Tensor
    minimality_gap: torch.Tensor
    minimality_penalty: torch.Tensor

    @classmethod
    def zeros(
        cls,
        *,
        num_graphs: int,
        device: torch.device,
    ) -> "TerminalStepTensors":
        zeros = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        return cls(
            log_reward=zeros.clone(),
            answer_f1=zeros.clone(),
            edge_penalty=zeros.clone(),
            base_log_reward=zeros.clone(),
            utility=zeros.clone(),
            expanded_edge_count=zeros.clone(),
            minimal_edge_count=zeros.clone(),
            minimality_gap=zeros.clone(),
            minimality_penalty=zeros.clone(),
        )


def has_candidate(
    *,
    candidate_batch_index: torch.Tensor,
    num_graphs: int,
    device: torch.device,
) -> torch.Tensor:
    candidate_batch_index = candidate_batch_index.to(device=device, dtype=torch.long)

    if candidate_batch_index.numel() == 0:
        return torch.zeros(num_graphs, dtype=torch.bool, device=device)

    _validate_graph_ids(
        candidate_batch_index,
        num_graphs=int(num_graphs),
        name="candidate_batch_index",
    )

    return torch.bincount(candidate_batch_index, minlength=int(num_graphs)).gt(0)


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


def validate_policy_step_output(
    step_out: PolicyStepOutput,
    *,
    num_graphs: int,
    num_edges: int,
    device: torch.device,
) -> None:
    if step_out.stop_logits.shape != (num_graphs,):
        raise ValueError(
            f"PolicyStepOutput.stop_logits must have shape [{num_graphs}], "
            f"got {tuple(step_out.stop_logits.shape)}."
        )
    if step_out.stop_logits.device != device:
        raise ValueError(
            f"PolicyStepOutput.stop_logits is on {step_out.stop_logits.device}, "
            f"expected {device}."
        )

    if step_out.expand_logits.shape != (num_graphs,):
        raise ValueError(
            f"PolicyStepOutput.expand_logits must have shape [{num_graphs}], "
            f"got {tuple(step_out.expand_logits.shape)}."
        )
    if step_out.expand_logits.device != device:
        raise ValueError(
            f"PolicyStepOutput.expand_logits is on {step_out.expand_logits.device}, "
            f"expected {device}."
        )

    validate_candidates(
        step_out.candidates,
        num_graphs=num_graphs,
        num_edges=num_edges,
        device=device,
    )


def validate_candidates(
    candidates: CandidateEdges,
    *,
    num_graphs: int,
    num_edges: int,
    device: torch.device,
) -> None:
    """
    CandidateEdges must use current physical batch coordinates.
    """
    edge_ids = candidates.edge_ids
    batch_index = candidates.batch_index
    expand_logits = candidates.expand_logits

    if edge_ids.device != device:
        raise ValueError(
            f"candidates.edge_ids is on {edge_ids.device}, expected {device}."
        )
    if batch_index.device != device:
        raise ValueError(
            f"candidates.batch_index is on {batch_index.device}, expected {device}."
        )
    if expand_logits.device != device:
        raise ValueError(
            f"candidates.expand_logits is on {expand_logits.device}, expected {device}."
        )

    if edge_ids.dtype != torch.long:
        raise TypeError(
            f"candidates.edge_ids must be torch.long, got {edge_ids.dtype}."
        )
    if batch_index.dtype != torch.long:
        raise TypeError(
            f"candidates.batch_index must be torch.long, got {batch_index.dtype}."
        )

    if edge_ids.ndim != 1:
        raise ValueError(
            f"candidates.edge_ids must be 1D, got {tuple(edge_ids.shape)}."
        )
    if batch_index.ndim != 1:
        raise ValueError(
            f"candidates.batch_index must be 1D, got {tuple(batch_index.shape)}."
        )
    if expand_logits.ndim != 1:
        raise ValueError(
            f"candidates.expand_logits must be 1D, got {tuple(expand_logits.shape)}."
        )

    if not (edge_ids.numel() == batch_index.numel() == expand_logits.numel()):
        raise ValueError(
            "CandidateEdges fields must have the same length: "
            f"edge_ids={edge_ids.numel()}, "
            f"batch_index={batch_index.numel()}, "
            f"expand_logits={expand_logits.numel()}."
        )

    _validate_edge_ids(edge_ids, num_edges=num_edges, name="candidates.edge_ids")
    _validate_graph_ids(
        batch_index,
        num_graphs=num_graphs,
        name="candidates.batch_index",
    )


def validate_frontier_candidates(
    *,
    candidates: CandidateEdges,
    state: State,
    edge_index: torch.Tensor,
) -> None:
    """
    Debug validator for forward candidate legality.

    Every candidate edge must satisfy:
        e not in E_s
        and at least one endpoint is in V_s.
    """
    edge_ids = candidates.edge_ids
    if edge_ids.numel() == 0:
        return

    edge_ids = edge_ids.to(device=edge_index.device, dtype=torch.long)
    active_nodes = state.active_nodes.to(device=edge_index.device, dtype=torch.bool)
    active_edges = state.active_edges.to(device=edge_index.device, dtype=torch.bool)

    src = edge_index[0].index_select(0, edge_ids)
    dst = edge_index[1].index_select(0, edge_ids)

    already_active = active_edges.index_select(0, edge_ids)
    not_incident = ~(
        active_nodes.index_select(0, src) | active_nodes.index_select(0, dst)
    )

    invalid = already_active | not_incident
    if bool(invalid.any()):
        bad = edge_ids[invalid]
        raise RuntimeError(f"Invalid frontier candidates: {bad.tolist()}.")


def validate_terminal_reward(
    reward: TerminalRewardOutput,
    *,
    num_graphs: int,
) -> None:
    for name, tensor in {
        "reward.log_reward": reward.log_reward,
        "reward.answer_f1": reward.answer_f1,
        "reward.edge_penalty": reward.edge_penalty,
        "reward.base_log_reward": reward.base_log_reward,
        "reward.utility": reward.utility,
        "reward.expanded_edge_count": reward.expanded_edge_count,
        "reward.minimal_edge_count": reward.minimal_edge_count,
        "reward.minimality_gap": reward.minimality_gap,
        "reward.minimality_penalty": reward.minimality_penalty,
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
    "BackwardPolicy",
    "StepExecutor",
    "StepGraphContext",
    "TerminalStepTensors",
    "UniformRemovalBackwardPolicy",
    "budget_exhausted_mask",
    "has_candidate",
    "validate_candidates",
    "validate_frontier_candidates",
    "validate_policy_step_output",
    "validate_terminal_reward",
]
