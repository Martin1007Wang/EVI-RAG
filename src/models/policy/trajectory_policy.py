from __future__ import annotations

import torch
from torch import nn

from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
    StartLogitHead,
)
from src.models.components.embedding import BackboneInput
from src.models.configs import PolicyConfig
from src.graph_runtime import build_graph_batch
from src.utils.segment_ops import segment_logsumexp_1d

from .state import SearchState
from .types import ForwardActionDistribution, PreparedSearchBatch, StartDistribution


def _mask_nonfinite_scores(values: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(values, float("-inf"))
    return torch.where(torch.isfinite(values), values, neg_inf)


class StartDistributionError(ValueError):
    """Base class for recoverable start-distribution failures."""


class EmptyStartCandidatesError(StartDistributionError):
    def __init__(self, *, empty_samples: list[str]) -> None:
        self.empty_samples = tuple(str(sample_id) for sample_id in empty_samples)
        super().__init__(
            "q_local_indices contains empty graphs; cannot build start distribution. "
            f"empty_samples={list(self.empty_samples)}"
        )


class InvalidStartCandidatesError(StartDistributionError):
    def __init__(self, *, invalid_samples: list[str]) -> None:
        self.invalid_samples = tuple(str(sample_id) for sample_id in invalid_samples)
        super().__init__(
            "Each graph must expose at least one finite start candidate. "
            f"invalid_samples={list(self.invalid_samples)}"
        )


def resolve_start_candidates(
    prepared_batch: PreparedSearchBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation = prepared_batch.observation
    candidate_nodes_abs, candidate_graph_ids = (
        prepared_batch.topology.resolve_local_node_indices(
            observation.q_local_indices,
            field_name="q_local_indices",
        )
    )
    candidate_counts = observation.q_local_indices.counts()
    if bool((candidate_counts <= 0).any().item()):
        empty_graphs = (
            torch.nonzero(candidate_counts <= 0, as_tuple=False).view(-1).tolist()
        )
        empty_samples = [observation.sample_ids[idx] for idx in empty_graphs]
        raise EmptyStartCandidatesError(empty_samples=empty_samples)
    return candidate_nodes_abs, candidate_graph_ids


def build_start_distribution_from_logits(
    *,
    prepared_batch: PreparedSearchBatch,
    candidate_nodes_abs: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    logits: torch.Tensor,
) -> StartDistribution:
    num_graphs = int(prepared_batch.topology.num_graphs)
    normalized_logits = _mask_nonfinite_scores(logits.to(dtype=torch.float32))
    lse, has_values = segment_logsumexp_1d(
        values=normalized_logits,
        segment_ids=candidate_graph_ids,
        num_segments=num_graphs,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    if not bool(has_values.all().item()):
        invalid_graphs = torch.nonzero(~has_values, as_tuple=False).view(-1).tolist()
        invalid_samples = [
            prepared_batch.observation.sample_ids[idx] for idx in invalid_graphs
        ]
        raise InvalidStartCandidatesError(invalid_samples=invalid_samples)
    return StartDistribution(
        candidate_nodes_abs=candidate_nodes_abs,
        candidate_graph_ids=candidate_graph_ids,
        log_probs=normalized_logits - lse.index_select(0, candidate_graph_ids),
    )


class TrajectoryPolicy(nn.Module):
    def __init__(
        self,
        config: PolicyConfig,
        *,
        max_steps: int,
        backbone: EmbeddingBackbone,
        state_score_head: NodeFlowHead,
        start_head: StartLogitHead,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        graph_hidden_dim = int(config.backbone.hidden_dim)
        self.state_score_head = state_score_head
        self.start_head = start_head
        self.step_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.remaining_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.state_feature_norm = nn.LayerNorm(graph_hidden_dim)
        self.backbone = backbone

    def prepare_batch(self, batch) -> PreparedSearchBatch:
        topology, observation = build_graph_batch(batch)
        encoded = self.backbone.encode(
            BackboneInput(
                node_features=observation.node_features,
                relation_features=observation.relation_features,
                question_embedding=observation.question_embedding,
                edge_index=topology.edge_index,
                edge_relations=topology.edge_type,
                num_nodes=topology.num_nodes,
            )
        )
        return PreparedSearchBatch(
            topology=topology,
            observation=observation,
            node_tokens=encoded.node_tokens,
            question_tokens=encoded.question_tokens,
        )

    def encode(self, batch) -> PreparedSearchBatch:
        return self.prepare_batch(batch)

    def compute_start_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> StartDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        node_features = prepared_batch.node_tokens.index_select(0, candidate_nodes_abs)
        question_features = prepared_batch.question_tokens.index_select(
            0, candidate_graph_ids
        )
        logits = self.start_head(
            node_features=node_features,
            question_features=question_features,
        )
        return build_start_distribution_from_logits(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            logits=logits,
        )

    def _build_flat_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        flat_nodes: torch.Tensor,
        flat_num_steps: torch.Tensor,
        flat_done_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(prepared_batch.topology.num_nodes)
        safe_nodes = flat_nodes.clamp(min=0, max=max(num_nodes - 1, 0))
        node_features = prepared_batch.node_tokens.index_select(0, safe_nodes)
        step_ids = flat_num_steps.clamp(min=0, max=self.max_steps)
        step_features = self.step_embedding(step_ids).to(dtype=node_features.dtype)
        remaining_ids = (self.max_steps - flat_num_steps).clamp(
            min=0, max=self.max_steps
        )
        remaining_features = self.remaining_embedding(remaining_ids).to(
            dtype=node_features.dtype
        )
        state_features = self.state_feature_norm(
            node_features + step_features + remaining_features
        )
        return torch.where(
            flat_done_mask.unsqueeze(-1),
            torch.zeros_like(state_features),
            state_features,
        )

    def build_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor:
        batch_size, num_rollouts = state.current_nodes.shape
        flat_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=state.flatten_current_nodes(),
            flat_num_steps=state.flatten_num_steps(),
            flat_done_mask=state.flatten_done_mask(),
        )
        return flat_features.view(batch_size, num_rollouts, -1)

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor:
        state_features = self.build_state_features(prepared_batch, state)
        log_state_scores = self._compute_log_state_scores_from_flat_features(
            prepared_batch=prepared_batch,
            flat_state_features=state_features.reshape(
                -1, int(state_features.size(-1))
            ),
            graph_ids=state.flatten_graph_index(),
        ).view_as(state.current_nodes)
        return torch.where(
            state.done_mask,
            torch.zeros_like(log_state_scores),
            log_state_scores,
        )

    def _compute_log_state_scores_from_flat_features(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        flat_state_features: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        question_features = prepared_batch.question_tokens.index_select(0, graph_ids)
        scores = self.state_score_head(flat_state_features, question_features)
        scores = scores.to(dtype=torch.float32)
        return _mask_nonfinite_scores(scores)

    @staticmethod
    def _prepare_agent_state(
        state: SearchState,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        batch_size, num_rollouts = state.current_nodes.shape
        total_agents = batch_size * num_rollouts
        flat_current_nodes = state.flatten_current_nodes()
        active_mask = ~state.flatten_done_mask()
        flat_num_steps = state.flatten_num_steps()
        return (
            total_agents,
            flat_current_nodes,
            active_mask,
            flat_num_steps,
            batch_size,
            num_rollouts,
        )

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        (
            total_agents,
            flat_current_nodes,
            active_mask,
            flat_num_steps,
            batch_size,
            num_rollouts,
        ) = self._prepare_agent_state(state)
        edge_ids, target_nodes, edge_agent_batch, out_degrees = (
            state.topology.gather_outgoing_edges(
                current_nodes=flat_current_nodes,
                active_mask=active_mask,
            )
        )
        edge_logits = torch.empty(
            (0,), device=flat_current_nodes.device, dtype=torch.float32
        )
        if int(edge_ids.numel()) > 0:
            child_graph_ids = state.topology.graph_index_from_nodes(target_nodes)
            child_num_steps = flat_num_steps.index_select(0, edge_agent_batch) + 1
            child_done_mask = torch.zeros_like(child_num_steps, dtype=torch.bool)
            child_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=target_nodes,
                flat_num_steps=child_num_steps,
                flat_done_mask=child_done_mask,
            )
            edge_logits = self._compute_log_state_scores_from_flat_features(
                prepared_batch=prepared_batch,
                flat_state_features=child_state_features,
                graph_ids=child_graph_ids,
            )
        return ForwardActionDistribution(
            edge_logits=edge_logits.to(dtype=prepared_batch.node_tokens.dtype),
            edge_agent_batch=edge_agent_batch,
            edge_ids=edge_ids,
            target_nodes=target_nodes,
            out_degrees=out_degrees.view(batch_size, num_rollouts),
        )

    @staticmethod
    def compute_move_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_agents = int(distribution.out_degrees.numel())
        edge_lse = torch.full(
            (total_agents,),
            fill_value=float("-inf"),
            device=distribution.edge_logits.device,
            dtype=torch.float32,
        )
        has_values = torch.zeros(
            (total_agents,), device=distribution.edge_logits.device, dtype=torch.bool
        )
        if int(distribution.edge_logits.numel()) > 0:
            move_lse, segment_has_values = segment_logsumexp_1d(
                values=distribution.edge_logits.to(dtype=torch.float32),
                segment_ids=distribution.edge_agent_batch,
                num_segments=total_agents,
                dtype=torch.float32,
                ignore_non_finite=True,
                empty_value=float("-inf"),
            )
            edge_lse = torch.where(segment_has_values, move_lse, edge_lse)
            has_values = segment_has_values
        move_log_probs = distribution.edge_logits.to(dtype=torch.float32)
        if int(move_log_probs.numel()) > 0:
            move_log_probs = move_log_probs - edge_lse.index_select(
                0, distribution.edge_agent_batch
            )
        return move_log_probs, edge_lse, has_values


__all__ = [
    "build_start_distribution_from_logits",
    "EmptyStartCandidatesError",
    "ForwardActionDistribution",
    "InvalidStartCandidatesError",
    "resolve_start_candidates",
    "StartDistributionError",
    "StartDistribution",
    "TrajectoryPolicy",
]
