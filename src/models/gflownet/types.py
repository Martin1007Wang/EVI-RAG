from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from src.graph_runtime import GraphTopology, SearchObservation

from .path import append_relation_and_node_tokens_inplace, initialize_path_token_ids


@dataclass(frozen=True)
class StartDistribution:
    candidate_nodes_abs: torch.Tensor
    candidate_graph_ids: torch.Tensor
    log_flows: torch.Tensor
    log_probs: torch.Tensor
    graph_log_z: torch.Tensor


@dataclass(frozen=True)
class ForwardActionDistribution:
    edge_logits: torch.Tensor
    edge_agent_batch: torch.Tensor
    edge_ids: torch.Tensor
    target_nodes: torch.Tensor
    out_degrees: torch.Tensor
    is_submit: torch.Tensor | None = None
    current_log_f: torch.Tensor | None = None
    active_agent_count: int = 0
    unique_active_state_count: int = 0
    raw_graph_candidate_count: int = 0
    scored_graph_candidate_count: int = 0
    shortlist_active_state_count: int = 0


@dataclass(frozen=True)
class PreparedSearchBatch:
    """Encoded batch payload shared across search, training, and evaluation."""

    topology: GraphTopology
    observation: SearchObservation
    node_tokens: torch.Tensor
    relation_tokens: torch.Tensor
    question_tokens: torch.Tensor
    question_context_tokens: torch.Tensor
    question_context_mask: torch.Tensor


@dataclass(frozen=True)
class HeuristicCache:
    """Cheap proposal cache used by behavior-policy sampling."""

    node_log_heuristic: torch.Tensor | None = None
    step_node_log_heuristic: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedGFlowNetBatch(PreparedSearchBatch):
    heuristic_cache: HeuristicCache


@dataclass(frozen=True)
class SearchState:
    """Canonical recurrent-prefix search state.

    The environment state still keeps the exact discrete trajectory prefix for
    tree-structured backward transitions, but forward scoring is driven by a
    recurrent control state that compresses question-conditioned prefix history.
    """

    topology: GraphTopology
    observation: SearchObservation
    current_nodes: torch.Tensor
    done_mask: torch.Tensor
    num_steps: torch.Tensor
    path_token_ids: torch.Tensor | None = None
    control_state: torch.Tensor | None = None

    def __post_init__(self) -> None:
        expected_shape = tuple(self.current_nodes.shape)
        if tuple(self.done_mask.shape) != expected_shape:
            raise ValueError(
                "done_mask must match current_nodes shape in SearchState. "
                f"current_nodes={expected_shape} done_mask={tuple(self.done_mask.shape)}."
            )
        if tuple(self.num_steps.shape) != expected_shape:
            raise ValueError(
                "num_steps must match current_nodes shape in SearchState. "
                f"current_nodes={expected_shape} num_steps={tuple(self.num_steps.shape)}."
            )
        if bool((self.num_steps < 0).any().item()):
            raise ValueError("num_steps must be >= 0 in SearchState.")
        if (
            self.path_token_ids is not None
            and tuple(self.path_token_ids.shape[:2]) != expected_shape
        ):
            raise ValueError(
                "path_token_ids batch shape must match current_nodes shape in SearchState. "
                f"current_nodes={expected_shape} path_token_ids={tuple(self.path_token_ids.shape)}."
            )
        if (
            self.control_state is not None
            and tuple(self.control_state.shape[:2]) != expected_shape
        ):
            raise ValueError(
                "control_state batch shape must match current_nodes shape in SearchState. "
                f"current_nodes={expected_shape} control_state={tuple(self.control_state.shape)}."
            )

    @classmethod
    def initialize(
        cls,
        *,
        topology: GraphTopology,
        observation: SearchObservation,
        start_nodes: torch.Tensor,
        max_steps: int | None = None,
    ) -> SearchState:
        if start_nodes.dim() != 2:
            raise ValueError(
                "start_nodes must be 2D [num_graphs, num_rollouts], "
                f"got shape={tuple(start_nodes.shape)}."
            )
        path_token_ids = None
        if max_steps is not None:
            path_token_ids = initialize_path_token_ids(
                start_nodes=start_nodes,
                max_steps=int(max_steps),
            )
        return cls(
            topology=topology,
            observation=observation,
            current_nodes=start_nodes.clone(),
            done_mask=torch.zeros_like(start_nodes, dtype=torch.bool),
            num_steps=torch.zeros_like(start_nodes, dtype=torch.long),
            path_token_ids=path_token_ids,
            control_state=None,
        )

    @classmethod
    def from_edge_path(
        cls,
        *,
        topology: GraphTopology,
        observation: SearchObservation,
        start_node: int,
        edge_ids: tuple[int, ...],
        max_steps: int,
        device: torch.device,
    ) -> SearchState:
        num_steps = len(edge_ids)
        if num_steps > max_steps:
            raise ValueError("edge path length cannot exceed max_steps.")
        current_node = int(start_node)
        current_nodes = torch.tensor([[current_node]], device=device, dtype=torch.long)
        path_token_ids = initialize_path_token_ids(
            start_nodes=current_nodes,
            max_steps=int(max_steps),
        )
        step_tensor = torch.zeros((1, 1), device=device, dtype=torch.long)
        for edge_id in edge_ids:
            edge_id_int = int(edge_id)
            edge_src = int(topology.edge_index[0, edge_id_int].item())
            edge_dst = int(topology.edge_index[1, edge_id_int].item())
            if edge_src != current_node:
                raise ValueError("edge path is not source-contiguous.")
            relation_id = int(topology.edge_type[edge_id_int].item())
            path_token_ids = append_relation_and_node_tokens_inplace(
                path_token_ids=path_token_ids,
                num_steps=step_tensor,
                relation_ids=torch.tensor(
                    [[relation_id]], device=device, dtype=torch.long
                ),
                target_nodes=torch.tensor(
                    [[edge_dst]], device=device, dtype=torch.long
                ),
            )
            step_tensor = step_tensor + 1
            current_node = edge_dst
        return cls(
            topology=topology,
            observation=observation,
            current_nodes=torch.tensor(
                [[current_node]], device=device, dtype=torch.long
            ),
            done_mask=torch.zeros((1, 1), device=device, dtype=torch.bool),
            num_steps=torch.tensor([[num_steps]], device=device, dtype=torch.long),
            path_token_ids=path_token_ids,
            control_state=None,
        )

    def flatten_current_nodes(self) -> torch.Tensor:
        return self.current_nodes.view(-1)

    def flatten_done_mask(self) -> torch.Tensor:
        return self.done_mask.view(-1)

    def flatten_num_steps(self) -> torch.Tensor:
        return self.num_steps.view(-1)

    def path_lengths(self) -> torch.Tensor:
        return (2 * self.num_steps + 1).to(dtype=torch.long)

    def flatten_path_lengths(self) -> torch.Tensor:
        return self.path_lengths().view(-1)

    def resolve_path_token_ids(self, *, max_steps: int) -> torch.Tensor:
        if self.path_token_ids is None:
            if bool((self.num_steps != 0).any().item()):
                raise ValueError(
                    "Non-root SearchState instances must carry exact path_token_ids. "
                    "The search space is defined over discrete trajectory prefixes, so "
                    "path history cannot be reconstructed from (current_node, num_steps) alone."
                )
            return initialize_path_token_ids(
                start_nodes=self.current_nodes,
                max_steps=int(max_steps),
            )
        expected_shape = (*self.current_nodes.shape, (2 * int(max_steps)) + 1)
        if tuple(self.path_token_ids.shape) != tuple(expected_shape):
            raise ValueError(
                "path_token_ids shape mismatch with current_nodes/max_steps in SearchState. "
                f"expected={expected_shape} got={tuple(self.path_token_ids.shape)}."
            )
        return self.path_token_ids

    def flatten_path_token_ids(self, *, max_steps: int) -> torch.Tensor:
        return self.resolve_path_token_ids(max_steps=max_steps).view(
            -1, (2 * int(max_steps)) + 1
        )

    def flatten_control_state(self) -> torch.Tensor:
        if self.control_state is None:
            raise ValueError(
                "SearchState is missing control_state; provide it explicitly or let the policy reconstruct it."
            )
        return self.control_state.view(-1, int(self.control_state.size(-1)))

    def flatten_graph_index(self) -> torch.Tensor:
        return self.topology.graph_index_from_nodes(self.flatten_current_nodes())


class SearchPolicyProtocol(Protocol):
    def prepare_batch(self, batch) -> PreparedSearchBatch: ...

    def build_start_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor: ...

    def compute_next_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        control_states: torch.Tensor,
        next_nodes: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor: ...

    def compute_start_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> StartDistribution: ...

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution: ...

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor: ...

    @staticmethod
    def compute_move_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class GFlowNetPolicyProtocol(SearchPolicyProtocol, Protocol):
    def prepare_batch(self, batch) -> PreparedGFlowNetBatch: ...

    def compute_graph_log_z(
        self, prepared_batch: PreparedGFlowNetBatch
    ) -> torch.Tensor: ...

    def compute_behavior_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> StartDistribution: ...

    def compute_behavior_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution: ...

    def compute_behavior_edge_logits(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor: ...

    def compute_backward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution: ...

    @staticmethod
    def sample_start_nodes(
        distribution: StartDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


__all__ = [
    "ForwardActionDistribution",
    "GFlowNetPolicyProtocol",
    "HeuristicCache",
    "PreparedGFlowNetBatch",
    "PreparedSearchBatch",
    "SearchPolicyProtocol",
    "SearchState",
    "StartDistribution",
]
