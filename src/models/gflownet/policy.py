from __future__ import annotations

import torch
from torch import nn

from src.graph_runtime import build_graph_batch
from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
    TransitionPolicyHead,
)
from src.models.components.embedding import BackboneInput
from src.models.configs import HeuristicConfig, PolicyConfig
from src.utils.segment_ops import segment_logsumexp_1d

from .heuristics import SearchHeuristic
from .types import (
    ForwardActionDistribution,
    PreparedGFlowNetBatch,
    PreparedSearchBatch,
    SearchState,
    StartDistribution,
)


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


def build_start_distribution_from_log_flows(
    *,
    prepared_batch: PreparedSearchBatch,
    candidate_nodes_abs: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    log_flows: torch.Tensor,
) -> StartDistribution:
    num_graphs = int(prepared_batch.topology.num_graphs)
    log_flows = _mask_nonfinite_scores(log_flows.to(dtype=torch.float32))
    lse, has_values = segment_logsumexp_1d(
        values=log_flows,
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
        log_flows=log_flows,
        log_probs=log_flows - lse.index_select(0, candidate_graph_ids),
        graph_log_z=lse,
    )


class BaseSearchPolicy(nn.Module):
    def __init__(
        self,
        config: PolicyConfig,
        *,
        max_steps: int,
        backbone: EmbeddingBackbone,
        state_score_head: NodeFlowHead,
        forward_policy_head: TransitionPolicyHead,
        backward_policy_head: TransitionPolicyHead,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        graph_hidden_dim = int(config.backbone.hidden_dim)
        self.state_flow_head = state_score_head
        self.state_score_head = self.state_flow_head
        self.forward_policy_head = forward_policy_head
        self.backward_policy_head = backward_policy_head
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
            relation_tokens=encoded.relation_tokens,
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
        log_flows = self.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return build_start_distribution_from_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            log_flows=log_flows,
        )

    def compute_start_log_flows(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        candidate_nodes_abs: torch.Tensor,
    ) -> torch.Tensor:
        flat_num_steps = torch.zeros_like(candidate_nodes_abs, dtype=torch.long)
        flat_done_mask = torch.zeros_like(candidate_nodes_abs, dtype=torch.bool)
        flat_state_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=candidate_nodes_abs,
            flat_num_steps=flat_num_steps,
            flat_done_mask=flat_done_mask,
        )
        return self._compute_log_state_scores_from_flat_features(
            prepared_batch=prepared_batch,
            flat_state_features=flat_state_features,
            graph_ids=prepared_batch.topology.graph_index_from_nodes(
                candidate_nodes_abs
            ),
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
        scores = self.state_flow_head(flat_state_features, question_features)
        scores = scores.to(dtype=torch.float32)
        return _mask_nonfinite_scores(scores)

    def _compute_transition_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        current_state_features: torch.Tensor,
        candidate_state_features: torch.Tensor,
        edge_ids: torch.Tensor,
        graph_ids: torch.Tensor,
        head: TransitionPolicyHead,
    ) -> torch.Tensor:
        relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_ids)
        relation_features = prepared_batch.relation_tokens.index_select(0, relation_ids)
        question_features = prepared_batch.question_tokens.index_select(0, graph_ids)
        logits = head(
            current_state_features.to(dtype=torch.float32),
            candidate_state_features.to(dtype=torch.float32),
            relation_features.to(dtype=torch.float32),
            question_features.to(dtype=torch.float32),
        )
        return _mask_nonfinite_scores(logits.to(dtype=torch.float32))

    def _gather_forward_candidates(
        self,
        state: SearchState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            total_agents,
            flat_current_nodes,
            active_mask,
            flat_num_steps,
            batch_size,
            num_rollouts,
        ) = self._prepare_agent_state(state)
        del total_agents
        edge_ids, target_nodes, edge_agent_batch, out_degrees = (
            state.topology.gather_outgoing_edges(
                current_nodes=flat_current_nodes,
                active_mask=active_mask,
            )
        )
        if int(edge_ids.numel()) > 0:
            child_num_steps = flat_num_steps.index_select(0, edge_agent_batch) + 1
        else:
            child_num_steps = flat_num_steps.new_empty((0,))
        return (
            edge_ids,
            target_nodes,
            edge_agent_batch,
            out_degrees.view(batch_size, num_rollouts),
            child_num_steps,
        )

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

    def _gather_backward_candidates(
        self,
        state: SearchState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            total_agents,
            flat_current_nodes,
            active_mask,
            flat_num_steps,
            batch_size,
            num_rollouts,
        ) = self._prepare_agent_state(state)
        del total_agents
        backward_active_mask = active_mask & (flat_num_steps > 0)
        edge_ids, source_nodes, edge_agent_batch, in_degrees = (
            state.topology.gather_incoming_edges(
                current_nodes=flat_current_nodes,
                active_mask=backward_active_mask,
            )
        )
        if int(edge_ids.numel()) > 0:
            parent_num_steps = flat_num_steps.index_select(0, edge_agent_batch) - 1
        else:
            parent_num_steps = flat_num_steps.new_empty((0,))
        return (
            edge_ids,
            source_nodes,
            edge_agent_batch,
            in_degrees.view(batch_size, num_rollouts),
            parent_num_steps,
        )

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        edge_ids, target_nodes, edge_agent_batch, out_degrees, child_num_steps = (
            self._gather_forward_candidates(state)
        )
        edge_logits = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.float32
        )
        if int(edge_ids.numel()) > 0:
            current_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=state.flatten_current_nodes(),
                flat_num_steps=state.flatten_num_steps(),
                flat_done_mask=state.flatten_done_mask(),
            ).index_select(0, edge_agent_batch)
            child_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=target_nodes,
                flat_num_steps=child_num_steps,
                flat_done_mask=torch.zeros_like(child_num_steps, dtype=torch.bool),
            )
            graph_ids = state.flatten_graph_index().index_select(0, edge_agent_batch)
            edge_logits = self._compute_transition_logits(
                prepared_batch=prepared_batch,
                current_state_features=current_state_features,
                candidate_state_features=child_state_features,
                edge_ids=edge_ids,
                graph_ids=graph_ids,
                head=self.forward_policy_head,
            )
        return ForwardActionDistribution(
            edge_logits=edge_logits.to(dtype=prepared_batch.node_tokens.dtype),
            edge_agent_batch=edge_agent_batch,
            edge_ids=edge_ids,
            target_nodes=target_nodes,
            out_degrees=out_degrees,
        )

    def compute_backward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        edge_ids, source_nodes, edge_agent_batch, in_degrees, parent_num_steps = (
            self._gather_backward_candidates(state)
        )
        edge_logits = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.float32
        )
        if int(edge_ids.numel()) > 0:
            current_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=state.flatten_current_nodes(),
                flat_num_steps=state.flatten_num_steps(),
                flat_done_mask=state.flatten_done_mask(),
            ).index_select(0, edge_agent_batch)
            parent_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=source_nodes,
                flat_num_steps=parent_num_steps,
                flat_done_mask=torch.zeros_like(parent_num_steps, dtype=torch.bool),
            )
            graph_ids = state.flatten_graph_index().index_select(0, edge_agent_batch)
            edge_logits = self._compute_transition_logits(
                prepared_batch=prepared_batch,
                current_state_features=current_state_features,
                candidate_state_features=parent_state_features,
                edge_ids=edge_ids,
                graph_ids=graph_ids,
                head=self.backward_policy_head,
            )
        return ForwardActionDistribution(
            edge_logits=edge_logits.to(dtype=prepared_batch.node_tokens.dtype),
            edge_agent_batch=edge_agent_batch,
            edge_ids=edge_ids,
            target_nodes=source_nodes,
            out_degrees=in_degrees,
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


class GFlowNetPolicy(nn.Module):
    def __init__(
        self,
        *,
        base_policy: BaseSearchPolicy,
        heuristic_cfg: HeuristicConfig,
        search_heuristic: SearchHeuristic,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self._heuristic_cfg = heuristic_cfg
        self.search_heuristic = search_heuristic

    @property
    def heuristic_cfg(self) -> HeuristicConfig:
        return self._heuristic_cfg

    def prepare_batch(self, batch) -> PreparedGFlowNetBatch:
        prepared_batch = self.base_policy.prepare_batch(batch)
        heuristic_cache = self.search_heuristic.build_cache(prepared_batch)
        return PreparedGFlowNetBatch(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            node_tokens=prepared_batch.node_tokens,
            relation_tokens=prepared_batch.relation_tokens,
            question_tokens=prepared_batch.question_tokens,
            heuristic_cache=heuristic_cache,
        )

    def encode(self, batch) -> PreparedGFlowNetBatch:
        return self.prepare_batch(batch)

    def compute_graph_log_z(
        self, prepared_batch: PreparedGFlowNetBatch
    ) -> torch.Tensor:
        start_distribution = self.compute_start_distribution(prepared_batch)
        return start_distribution.graph_log_z.to(dtype=torch.float32)

    @staticmethod
    def _build_state_from_nodes(
        *,
        prepared_batch: PreparedGFlowNetBatch,
        nodes: torch.Tensor,
        num_steps: torch.Tensor,
    ) -> SearchState:
        return SearchState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            current_nodes=nodes.view(-1, 1),
            done_mask=torch.zeros(
                (int(nodes.numel()), 1),
                device=nodes.device,
                dtype=torch.bool,
            ),
            num_steps=num_steps.view(-1, 1),
        )

    def _compute_state_bias(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> torch.Tensor:
        return self.search_heuristic.compute_state_bias(
            prepared_batch=prepared_batch,
            heuristic_cache=prepared_batch.heuristic_cache,
            state=state,
            build_state_features=self.base_policy.build_state_features,
        ).to(dtype=torch.float32)

    def _compute_behavior_forward_logits(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor:
        edge_logits = distribution.edge_logits.to(dtype=torch.float32)
        if int(edge_logits.numel()) == 0 or float(self.heuristic_cfg.beta) == 0.0:
            return edge_logits
        child_num_steps = (
            state.flatten_num_steps().index_select(0, distribution.edge_agent_batch) + 1
        )
        child_state = self._build_state_from_nodes(
            prepared_batch=prepared_batch,
            nodes=distribution.target_nodes,
            num_steps=child_num_steps,
        )
        child_bias = self._compute_state_bias(
            prepared_batch=prepared_batch,
            state=child_state,
        ).view(-1)
        return _mask_nonfinite_scores(
            edge_logits + float(self.heuristic_cfg.beta) * child_bias
        )

    def compute_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> StartDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        log_flows = self.base_policy.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return build_start_distribution_from_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            log_flows=log_flows,
        )

    def compute_behavior_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> StartDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        start_state = self._build_state_from_nodes(
            prepared_batch=prepared_batch,
            nodes=candidate_nodes_abs,
            num_steps=torch.zeros_like(candidate_nodes_abs, dtype=torch.long),
        )
        log_flows = self.base_policy.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_bias = self._compute_state_bias(
            prepared_batch=prepared_batch,
            state=start_state,
        ).view(-1)
        return build_start_distribution_from_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            log_flows=log_flows + float(self.heuristic_cfg.beta) * start_bias,
        )

    @staticmethod
    def sample_start_nodes(
        distribution: StartDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = int(distribution.candidate_graph_ids.max().item()) + 1
        selected_nodes: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        selected_log_flows: list[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            mask = distribution.candidate_graph_ids == graph_idx
            graph_nodes = distribution.candidate_nodes_abs[mask]
            graph_log_probs = distribution.log_probs[mask]
            graph_log_flows = distribution.log_flows[mask]
            if int(graph_nodes.numel()) == 0:
                raise ValueError("Each graph must expose at least one start candidate.")
            if deterministic:
                order = torch.argsort(graph_log_probs, descending=True)
                graph_nodes = graph_nodes.index_select(0, order)
                graph_log_probs = graph_log_probs.index_select(0, order)
                graph_log_flows = graph_log_flows.index_select(0, order)
                if int(graph_nodes.numel()) < num_rollouts:
                    repeat_idx = torch.remainder(
                        torch.arange(num_rollouts, device=graph_nodes.device),
                        int(graph_nodes.numel()),
                    )
                    graph_nodes = graph_nodes.index_select(0, repeat_idx)
                    graph_log_probs = graph_log_probs.index_select(0, repeat_idx)
                    graph_log_flows = graph_log_flows.index_select(0, repeat_idx)
                else:
                    graph_nodes = graph_nodes[:num_rollouts]
                    graph_log_probs = graph_log_probs[:num_rollouts]
                    graph_log_flows = graph_log_flows[:num_rollouts]
            else:
                probs = torch.softmax(graph_log_probs, dim=0)
                sample_idx = torch.multinomial(
                    probs,
                    num_samples=num_rollouts,
                    replacement=True,
                )
                graph_nodes = graph_nodes.index_select(0, sample_idx)
                graph_log_probs = graph_log_probs.index_select(0, sample_idx)
                graph_log_flows = graph_log_flows.index_select(0, sample_idx)
            selected_nodes.append(graph_nodes)
            selected_log_probs.append(graph_log_probs)
            selected_log_flows.append(graph_log_flows)
        return (
            torch.stack(selected_nodes, dim=0),
            torch.stack(selected_log_probs, dim=0),
            torch.stack(selected_log_flows, dim=0),
        )

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> torch.Tensor:
        return self.base_policy.compute_log_state_scores(
            prepared_batch,
            state,
        ).to(dtype=torch.float32)

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        return self.base_policy.compute_forward_distribution(
            prepared_batch,
            state,
        )

    def compute_backward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        return self.base_policy.compute_backward_distribution(
            prepared_batch,
            state,
        )

    def compute_behavior_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        distribution = self.base_policy.compute_forward_distribution(
            prepared_batch,
            state,
        )
        return ForwardActionDistribution(
            edge_logits=self._compute_behavior_forward_logits(
                prepared_batch=prepared_batch,
                state=state,
                distribution=distribution,
            ),
            edge_agent_batch=distribution.edge_agent_batch,
            edge_ids=distribution.edge_ids,
            target_nodes=distribution.target_nodes,
            out_degrees=distribution.out_degrees,
        )

    @staticmethod
    def compute_move_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return BaseSearchPolicy.compute_move_log_probs(distribution)


__all__ = [
    "BaseSearchPolicy",
    "EmptyStartCandidatesError",
    "ForwardActionDistribution",
    "GFlowNetPolicy",
    "InvalidStartCandidatesError",
    "PreparedGFlowNetBatch",
    "StartDistribution",
    "StartDistributionError",
    "build_start_distribution_from_log_flows",
    "resolve_start_candidates",
]
