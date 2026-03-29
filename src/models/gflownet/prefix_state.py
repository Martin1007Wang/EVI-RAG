from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from src.graph import GraphTopology, SearchObservation

from .path import (
    STOP_TOKEN_ID,
    append_relation_and_node_tokens_inplace,
    append_stop_token_inplace,
    initialize_path_token_ids,
    max_path_tokens,
)


@dataclass(frozen=True)
class RootState:
    """Explicit abstract root boundary state for the prefix-tree search space."""

    topology: GraphTopology
    observation: SearchObservation
    sequence_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class RootActionDistribution:
    """Outgoing action distribution from the abstract root boundary state.

    The root itself is not materialized as a regular ``SearchState`` because its
    outgoing actions are query-dependent start-node selections rather than graph
    edges. This distribution is the explicit runtime object that represents
    those root actions together with the decoupled root-flow and child-state-flow
    quantities used by SubTB.
    """

    candidate_nodes_abs: torch.Tensor
    candidate_graph_ids: torch.Tensor
    log_flows: torch.Tensor
    log_probs: torch.Tensor
    graph_log_z: torch.Tensor
    start_log_rewards: torch.Tensor | None = None
    action_logits: torch.Tensor | None = None
    root_state: RootState | None = None

    @property
    def start_state_log_flows(self) -> torch.Tensor:
        return self.log_flows

    @property
    def root_log_flow(self) -> torch.Tensor:
        return self.graph_log_z


@dataclass(frozen=True)
class ForwardActionDistribution:
    """Per-state branch log-masses over graph moves plus the explicit STOP action."""

    edge_logits: torch.Tensor
    edge_agent_batch: torch.Tensor
    edge_ids: torch.Tensor
    target_nodes: torch.Tensor
    out_degrees: torch.Tensor
    is_stop_action: torch.Tensor | None = None
    is_root_action: torch.Tensor | None = None
    current_log_f: torch.Tensor | None = None
    active_agent_count: int = 0
    unique_active_state_count: int = 0
    raw_graph_candidate_count: int = 0
    scored_graph_candidate_count: int = 0


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
    answer_mask: torch.Tensor
    answer_sink_ids: torch.Tensor
    answer_sink_log_rewards: torch.Tensor
    answer_distance: torch.Tensor | None = None
    shortest_path_edge_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class ActionPriorCache:
    """Cached per-node priors shared across proposal-policy action scoring."""

    node_prior: torch.Tensor | None = None
    answer_distance: torch.Tensor | None = None
    shortest_path_edge_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedGFlowNetBatch(PreparedSearchBatch):
    action_prior_cache: ActionPriorCache = field(default_factory=ActionPriorCache)


@dataclass(frozen=True)
class SearchState:
    """Canonical search state on the legal-prefix state space.

    Non-root states are defined by exact discrete trajectory prefixes rather
    than by ``(current_node, num_steps)`` alone. This is what keeps hard
    legality rules such as entity-level no-repeat Markovian: same node and same
    step with different prefix history are different states. Forward scoring may
    cache a recurrent ``control_state`` summary, but ``path_token_ids`` remain
    the authoritative discrete state identity for legality and tree-structured
    backward semantics. Root selection is modeled separately by
    ``RootActionDistribution`` rather than materializing the abstract root as a
    regular ``SearchState``.
    """

    topology: GraphTopology
    observation: SearchObservation
    current_nodes: torch.Tensor
    done_mask: torch.Tensor
    num_steps: torch.Tensor
    path_token_ids: torch.Tensor | None = None
    control_state: torch.Tensor | None = None
    absorbing_mask: torch.Tensor | None = None

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
        if (
            self.absorbing_mask is not None
            and tuple(self.absorbing_mask.shape) != expected_shape
        ):
            raise ValueError(
                "absorbing_mask must match current_nodes shape in SearchState. "
                f"current_nodes={expected_shape} absorbing_mask={tuple(self.absorbing_mask.shape)}."
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
    def from_sequence_prefix(
        cls,
        *,
        topology: GraphTopology,
        observation: SearchObservation,
        start_nodes: torch.Tensor,
        path_token_ids: torch.Tensor,
        num_steps: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> SearchState:
        return cls(
            topology=topology,
            observation=observation,
            current_nodes=start_nodes,
            done_mask=done_mask,
            num_steps=num_steps,
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

    @property
    def prefix_token_ids(self) -> torch.Tensor | None:
        return self.path_token_ids

    @property
    def is_absorbing_mask(self) -> torch.Tensor:
        if self.absorbing_mask is not None:
            return self.absorbing_mask
        return self.done_mask

    @property
    def hop_counts(self) -> torch.Tensor:
        return self.num_steps

    def flatten_done_mask(self) -> torch.Tensor:
        return self.done_mask.view(-1)

    def flatten_absorbing_mask(self) -> torch.Tensor:
        return self.is_absorbing_mask.view(-1)

    def flatten_num_steps(self) -> torch.Tensor:
        return self.num_steps.view(-1)

    def requires_exact_prefix_history(self) -> bool:
        if bool((self.num_steps != 0).any().item()):
            return True
        if self.absorbing_mask is not None:
            return bool(self.absorbing_mask.any().item())
        return bool(self.done_mask.any().item())

    def path_lengths(self) -> torch.Tensor:
        return (
            2 * self.num_steps
            + 1
            + self.is_absorbing_mask.to(dtype=self.num_steps.dtype)
        ).to(dtype=torch.long)

    def flatten_path_lengths(self) -> torch.Tensor:
        return self.path_lengths().view(-1)

    def resolve_path_token_ids(self, *, max_steps: int) -> torch.Tensor:
        if self.path_token_ids is None:
            if self.requires_exact_prefix_history():
                raise ValueError(
                    "Non-root SearchState instances must carry exact path_token_ids. "
                    "The legal search space is defined over discrete trajectory prefixes, so "
                    "path history cannot be reconstructed from (current_node, num_steps) alone."
                )
            return initialize_path_token_ids(
                start_nodes=self.current_nodes,
                max_steps=int(max_steps),
            )
        expected_shape = (
            *self.current_nodes.shape,
            max_path_tokens(max_steps=int(max_steps)),
        )
        if tuple(self.path_token_ids.shape) != tuple(expected_shape):
            raise ValueError(
                "path_token_ids shape mismatch with current_nodes/max_steps in SearchState. "
                f"expected={expected_shape} got={tuple(self.path_token_ids.shape)}."
            )
        absorbing_mask = self.is_absorbing_mask.to(dtype=torch.bool)
        if bool(absorbing_mask.any().item()):
            stop_positions = (2 * self.num_steps[absorbing_mask] + 1).to(
                dtype=torch.long
            )
            observed_stop_tokens = self.path_token_ids[absorbing_mask, stop_positions]
            if not bool((observed_stop_tokens == STOP_TOKEN_ID).all().item()):
                raise ValueError(
                    "Absorbing SearchState rows must terminate with STOP_TOKEN_ID."
                )
        return self.path_token_ids

    def flatten_path_token_ids(self, *, max_steps: int) -> torch.Tensor:
        return self.resolve_path_token_ids(max_steps=max_steps).view(
            -1, max_path_tokens(max_steps=int(max_steps))
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

    def compute_root_action_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> RootActionDistribution: ...

    def compute_start_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> RootActionDistribution: ...

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

    def compute_proposal_root_action_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        *,
        action_prior_scale: float = 1.0,
    ) -> RootActionDistribution: ...

    def compute_proposal_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        *,
        action_prior_scale: float = 1.0,
    ) -> RootActionDistribution: ...

    def compute_proposal_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        *,
        action_prior_scale: float = 1.0,
        transition_bias_scale: float = 1.0,
    ) -> ForwardActionDistribution: ...

    def compute_proposal_edge_logits(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
        *,
        action_prior_scale: float = 1.0,
        transition_bias_scale: float = 1.0,
    ) -> torch.Tensor: ...

    def compute_backward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution: ...

    @staticmethod
    def sample_start_nodes(
        distribution: RootActionDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


__all__ = [
    "ForwardActionDistribution",
    "GFlowNetPolicyProtocol",
    "ActionPriorCache",
    "PreparedGFlowNetBatch",
    "PreparedSearchBatch",
    "RootState",
    "RootActionDistribution",
    "SearchPolicyProtocol",
    "SearchState",
]
