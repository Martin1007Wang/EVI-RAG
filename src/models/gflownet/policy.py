from __future__ import annotations

import math

import torch
from torch import nn

from src.graph import SearchObservation, TrajectoryBatch, build_graph_batch
from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
    TransitionPolicyHead,
)
from src.models.components.embedding import BackboneInput
from src.models.configs import HeuristicConfig, PolicyConfig
from src.utils.nn_init import init_linear_xavier
from src.utils.segment_ops import sample_segmented_one_1d, segment_logsumexp_1d

from .heuristics import SearchHeuristic
from .repetition import build_entity_revisit_mask_from_flat_state
from .types import (
    ForwardActionDistribution,
    PreparedGFlowNetBatch,
    PreparedSearchBatch,
    RootState,
    SearchState,
    RootActionDistribution,
)


def _mask_nonfinite_scores(values: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(values, float("-inf"))
    return torch.where(torch.isfinite(values), values, neg_inf)


_FORWARD_EDGE_CHUNK_SIZE = 1024
_TRANSITION_LOGIT_CHUNK_SIZE = 1024


class RootActionDistributionError(ValueError):
    """Base class for recoverable root-action-distribution failures."""


class EmptyStartCandidatesError(RootActionDistributionError):
    def __init__(self, *, empty_samples: list[str]) -> None:
        self.empty_samples = tuple(str(sample_id) for sample_id in empty_samples)
        super().__init__(
            "q_local_indices contains empty graphs; cannot build start distribution. "
            f"empty_samples={list(self.empty_samples)}"
        )


class InvalidStartCandidatesError(RootActionDistributionError):
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
    (
        candidate_nodes_abs,
        candidate_graph_ids,
    ) = prepared_batch.topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )
    candidate_counts = observation.q_local_indices.counts()
    if bool((candidate_counts <= 0).any().item()):
        empty_graphs = (
            torch.nonzero(candidate_counts <= 0, as_tuple=False).view(-1).tolist()
        )
        empty_samples = [observation.sample_ids[idx] for idx in empty_graphs]
        raise EmptyStartCandidatesError(empty_samples=empty_samples)
    return candidate_nodes_abs, candidate_graph_ids


def build_root_action_distribution(
    *,
    prepared_batch: PreparedSearchBatch,
    candidate_nodes_abs: torch.Tensor,
    candidate_graph_ids: torch.Tensor,
    action_logits: torch.Tensor,
    start_state_log_flows: torch.Tensor,
    graph_log_z: torch.Tensor,
) -> RootActionDistribution:
    num_graphs = int(prepared_batch.topology.num_graphs)
    action_logits = _mask_nonfinite_scores(action_logits.to(dtype=torch.float32))
    lse, has_values = segment_logsumexp_1d(
        values=action_logits,
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
    return RootActionDistribution(
        candidate_nodes_abs=candidate_nodes_abs,
        candidate_graph_ids=candidate_graph_ids,
        log_flows=_mask_nonfinite_scores(start_state_log_flows.to(dtype=torch.float32)),
        log_probs=action_logits - lse.index_select(0, candidate_graph_ids),
        graph_log_z=_mask_nonfinite_scores(graph_log_z.to(dtype=torch.float32)),
        action_logits=action_logits,
        root_state=RootState(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
        ),
    )


def _temper_segmented_log_probs(
    *,
    log_probs: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"sampling temperature must be > 0, got {temperature!r}.")
    tempered = _mask_nonfinite_scores(log_probs.to(dtype=torch.float32))
    if int(tempered.numel()) == 0 or float(temperature) == 1.0:
        return tempered
    tempered = tempered / float(temperature)
    lse, _ = segment_logsumexp_1d(
        values=tempered,
        segment_ids=segment_ids,
        num_segments=num_segments,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    return tempered - lse.index_select(0, segment_ids)


class BaseSearchPolicy(nn.Module):
    def __init__(
        self,
        config: PolicyConfig,
        *,
        max_steps: int,
        backbone: EmbeddingBackbone,
        state_score_head: NodeFlowHead,
        forward_policy_head: TransitionPolicyHead,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        graph_hidden_dim = int(config.backbone.hidden_dim)
        base_state_dim = graph_hidden_dim * 3
        state_feature_input_dim = base_state_dim + graph_hidden_dim
        root_feature_dim = graph_hidden_dim * 3 + 4
        self.state_flow_head = state_score_head
        self.state_score_head = self.state_flow_head
        self.forward_policy_head = forward_policy_head
        self.root_flow_input_norm = nn.LayerNorm(root_feature_dim)
        self.root_flow_hidden = nn.Linear(root_feature_dim, graph_hidden_dim)
        self.root_flow_activation = nn.GELU()
        self.root_flow_head = nn.Linear(graph_hidden_dim, 1)
        self.root_action_head = nn.Linear(graph_hidden_dim, 1)
        self.step_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.remaining_embedding = nn.Embedding(self.max_steps + 1, graph_hidden_dim)
        self.control_query = nn.Linear(graph_hidden_dim, graph_hidden_dim, bias=False)
        self.control_input_norm = nn.LayerNorm(graph_hidden_dim * 3)
        self.control_update = nn.GRUCell(
            input_size=graph_hidden_dim * 3,
            hidden_size=graph_hidden_dim,
        )
        self.control_norm = nn.LayerNorm(graph_hidden_dim)
        self.control_dropout = nn.Dropout(float(config.prefix_controller.dropout))
        self.state_feature_input_norm = nn.LayerNorm(state_feature_input_dim)
        state_hidden_dim = int(config.state_score_head.hidden_dim)
        state_layers: list[nn.Module] = [
            nn.Linear(state_feature_input_dim, state_hidden_dim),
            nn.GELU(),
        ]
        if float(config.state_score_head.dropout) > 0.0:
            state_layers.append(nn.Dropout(float(config.state_score_head.dropout)))
        for _ in range(max(int(config.state_score_head.num_layers) - 2, 0)):
            state_layers.extend(
                [
                    nn.Linear(state_hidden_dim, state_hidden_dim),
                    nn.GELU(),
                ]
            )
            if float(config.state_score_head.dropout) > 0.0:
                state_layers.append(nn.Dropout(float(config.state_score_head.dropout)))
        state_layers.append(nn.Linear(state_hidden_dim, graph_hidden_dim))
        self.state_feature_mlp = nn.Sequential(*state_layers)
        self.state_feature_norm = nn.LayerNorm(graph_hidden_dim)
        self.start_relation_feature = nn.Parameter(torch.zeros(graph_hidden_dim))
        self.register_buffer(
            "stop_action_relation_feature", torch.zeros(graph_hidden_dim)
        )
        self.backbone = backbone
        init_linear_xavier(self.root_flow_hidden)
        nn.init.normal_(self.root_flow_head.weight, mean=0.0, std=1.0e-2)
        if self.root_flow_head.bias is not None:
            nn.init.zeros_(self.root_flow_head.bias)

    def prepare_batch(self, batch) -> PreparedSearchBatch:
        if isinstance(batch, TrajectoryBatch):
            batch.require_raw_features()
        topology, observation = build_graph_batch(batch, validate=False)
        encoded = self.backbone.encode(
            BackboneInput(
                node_features=observation.node_features,
                relation_features=observation.relation_features,
                question_embedding=observation.question_embedding,
                question_context=observation.question_context,
                edge_index=topology.edge_index,
                edge_relations=topology.edge_type,
                num_nodes=topology.num_nodes,
            )
        )
        return PreparedSearchBatch(
            topology=topology,
            observation=SearchObservation.from_graph_observation(observation),
            node_tokens=encoded.node_tokens,
            relation_tokens=encoded.relation_tokens,
            question_tokens=encoded.question_tokens,
            question_context_tokens=encoded.question_context_tokens,
            question_context_mask=observation.question_valid_mask,
        )

    def encode(self, batch) -> PreparedSearchBatch:
        return self.prepare_batch(batch)

    def compute_root_action_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> RootActionDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        start_state_log_flows = self.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        action_logits = self.compute_root_action_logits(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return build_root_action_distribution(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_logits=action_logits,
            start_state_log_flows=start_state_log_flows,
            graph_log_z=self.compute_graph_log_z(
                prepared_batch,
                candidate_nodes_abs=candidate_nodes_abs,
                candidate_graph_ids=candidate_graph_ids,
            ),
        )

    def compute_start_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> RootActionDistribution:
        return self.compute_root_action_distribution(prepared_batch)

    def _build_start_state_features(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        candidate_nodes_abs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_num_steps = torch.zeros_like(candidate_nodes_abs, dtype=torch.long)
        flat_done_mask = torch.zeros_like(candidate_nodes_abs, dtype=torch.bool)
        flat_control_states = self.build_start_control_states(
            prepared_batch,
            candidate_nodes_abs.view(-1, 1),
        ).view(int(candidate_nodes_abs.numel()), -1)
        flat_state_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=candidate_nodes_abs,
            flat_num_steps=flat_num_steps,
            flat_done_mask=flat_done_mask,
            flat_control_states=flat_control_states,
        )
        graph_ids = prepared_batch.topology.graph_index_from_nodes(candidate_nodes_abs)
        return flat_state_features, graph_ids

    def compute_start_log_flows(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        candidate_nodes_abs: torch.Tensor,
    ) -> torch.Tensor:
        flat_state_features, graph_ids = self._build_start_state_features(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return self._compute_log_state_scores_from_flat_features(
            prepared_batch=prepared_batch,
            flat_state_features=flat_state_features,
            graph_ids=graph_ids,
        )

    def compute_root_action_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        candidate_nodes_abs: torch.Tensor,
    ) -> torch.Tensor:
        flat_state_features, _ = self._build_start_state_features(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        action_logits = self.root_action_head(
            flat_state_features.to(dtype=torch.float32)
        ).squeeze(-1)
        return _mask_nonfinite_scores(action_logits)

    @staticmethod
    def _scatter_mean_by_graph(
        *,
        values: torch.Tensor,
        graph_ids: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        values = values.to(dtype=torch.float32)
        hidden_dim = int(values.size(-1))
        pooled = values.new_zeros((num_graphs, hidden_dim))
        if int(values.numel()) == 0:
            return pooled
        pooled.scatter_add_(
            0,
            graph_ids.unsqueeze(-1).expand(-1, hidden_dim),
            values,
        )
        counts = values.new_zeros((num_graphs, 1))
        counts.scatter_add_(
            0,
            graph_ids.unsqueeze(-1),
            torch.ones((int(graph_ids.numel()), 1), device=values.device),
        )
        return pooled / counts.clamp_min(1.0)

    def _pool_graph_embeddings(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        candidate_nodes_abs: torch.Tensor | None = None,
        candidate_graph_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(prepared_batch.topology.num_graphs)
        node_tokens = prepared_batch.node_tokens
        all_graph_ids = prepared_batch.topology.all_node_graph_index(
            device=node_tokens.device
        )
        pooled_all_nodes = self._scatter_mean_by_graph(
            values=node_tokens,
            graph_ids=all_graph_ids,
            num_graphs=num_graphs,
        )
        if candidate_nodes_abs is None or candidate_graph_ids is None:
            candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
                prepared_batch
            )
        pooled_start_nodes = self._scatter_mean_by_graph(
            values=node_tokens.index_select(0, candidate_nodes_abs),
            graph_ids=candidate_graph_ids,
            num_graphs=num_graphs,
        )
        return pooled_all_nodes, pooled_start_nodes

    def _build_root_size_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        candidate_graph_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        topology = prepared_batch.topology
        device = prepared_batch.question_tokens.device
        node_counts = (
            topology.graph_node_offsets[1:] - topology.graph_node_offsets[:-1]
        ).to(
            device=device,
            dtype=torch.float32,
        )
        edge_counts = torch.zeros(
            (int(topology.num_graphs),),
            device=device,
            dtype=torch.float32,
        )
        if int(topology.edge_index.size(1)) > 0:
            edge_graph_ids = topology.graph_index_from_nodes(
                topology.edge_index[0].to(device=device)
            )
            edge_counts.scatter_add_(
                0,
                edge_graph_ids,
                torch.ones_like(edge_graph_ids, dtype=torch.float32),
            )
        if candidate_graph_ids is None:
            start_counts = prepared_batch.observation.q_local_indices.counts().to(
                device=device,
                dtype=torch.float32,
            )
        else:
            start_counts = torch.zeros_like(node_counts)
            start_counts.scatter_add_(
                0,
                candidate_graph_ids.to(device=device),
                torch.ones_like(
                    candidate_graph_ids, device=device, dtype=torch.float32
                ),
            )
        horizon_feature = torch.full_like(
            node_counts,
            fill_value=math.log1p(float(self.max_steps)),
        )
        return torch.stack(
            (
                torch.log1p(node_counts),
                torch.log1p(edge_counts),
                torch.log1p(start_counts),
                horizon_feature,
            ),
            dim=-1,
        )

    def _build_root_flow_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        candidate_nodes_abs: torch.Tensor | None = None,
        candidate_graph_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if candidate_nodes_abs is None or candidate_graph_ids is None:
            candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
                prepared_batch
            )
        pooled_all_nodes, pooled_start_nodes = self._pool_graph_embeddings(
            prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
        )
        return torch.cat(
            (
                prepared_batch.question_tokens.to(dtype=torch.float32),
                pooled_all_nodes,
                pooled_start_nodes,
                self._build_root_size_features(
                    prepared_batch,
                    candidate_graph_ids=candidate_graph_ids,
                ),
            ),
            dim=-1,
        )

    def compute_graph_log_z(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        candidate_nodes_abs: torch.Tensor | None = None,
        candidate_graph_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        root_inputs = self._build_root_flow_features(
            prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
        )
        hidden = self.root_flow_activation(
            self.root_flow_hidden(self.root_flow_input_norm(root_inputs))
        )
        root_log_z = self.root_flow_head(hidden).squeeze(-1)
        return _mask_nonfinite_scores(root_log_z)

    def _build_flat_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        flat_nodes: torch.Tensor,
        flat_num_steps: torch.Tensor,
        flat_done_mask: torch.Tensor,
        flat_control_states: torch.Tensor,
    ) -> torch.Tensor:
        base_state_features = self._build_base_state_features(
            prepared_batch,
            flat_nodes=flat_nodes,
            flat_num_steps=flat_num_steps,
        )
        step_ids = flat_num_steps.clamp(min=0, max=self.max_steps)
        step_features = self.step_embedding(step_ids).to(
            dtype=base_state_features.dtype
        )
        remaining_ids = (self.max_steps - flat_num_steps).clamp(
            min=0, max=self.max_steps
        )
        remaining_features = self.remaining_embedding(remaining_ids).to(
            dtype=base_state_features.dtype
        )
        state_features = self.state_feature_norm(
            self.state_feature_mlp(
                self.state_feature_input_norm(
                    torch.cat(
                        (
                            base_state_features,
                            step_features,
                            remaining_features,
                            flat_control_states.to(dtype=base_state_features.dtype),
                        ),
                        dim=-1,
                    )
                )
            ).to(dtype=base_state_features.dtype)
        )
        return torch.where(
            flat_done_mask.unsqueeze(-1),
            torch.zeros_like(state_features),
            state_features,
        )

    def _build_base_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        flat_nodes: torch.Tensor,
        flat_num_steps: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = int(prepared_batch.topology.num_nodes)
        safe_nodes = flat_nodes.clamp(min=0, max=max(num_nodes - 1, 0))
        del flat_num_steps
        return prepared_batch.node_tokens.index_select(0, safe_nodes)

    def build_local_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        flat_nodes: torch.Tensor,
        flat_num_steps: torch.Tensor,
        flat_done_mask: torch.Tensor,
    ) -> torch.Tensor:
        graph_ids = prepared_batch.topology.graph_index_from_nodes(flat_nodes)
        local_control_states = prepared_batch.question_tokens.index_select(0, graph_ids)
        return self._build_flat_state_features(
            prepared_batch,
            flat_nodes=flat_nodes,
            flat_num_steps=flat_num_steps,
            flat_done_mask=flat_done_mask,
            flat_control_states=local_control_states,
        )

    def _attend_question_context(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        control_states: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        total_states = int(control_states.size(0))
        if total_states == 0:
            return control_states.new_empty((0, int(control_states.size(-1))))
        context_query = self.control_query(control_states).to(dtype=torch.float32)
        context_summary = torch.zeros_like(context_query)
        unique_graph_ids = torch.unique(graph_ids, sorted=True)
        for graph_id_tensor in unique_graph_ids:
            graph_id = int(graph_id_tensor.item())
            graph_mask = graph_ids == graph_id
            question_context = prepared_batch.question_context_tokens[graph_id]
            question_mask = prepared_batch.question_context_mask[graph_id]
            if not bool(question_mask.any().item()):
                raise ValueError(
                    "question_context_mask contains rows without valid tokens when updating control states."
                )
            context_matrix = question_context.to(dtype=torch.float32)
            attention_scores = torch.matmul(
                context_query[graph_mask],
                context_matrix.transpose(0, 1),
            )
            attention_scores = attention_scores / math.sqrt(
                float(context_matrix.size(-1))
            )
            attention_scores = attention_scores.masked_fill(
                ~question_mask.unsqueeze(0),
                float("-inf"),
            )
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_weights = torch.where(
                torch.isfinite(attention_weights),
                attention_weights,
                torch.zeros_like(attention_weights),
            )
            context_summary[graph_mask] = torch.matmul(
                attention_weights.to(dtype=context_matrix.dtype),
                context_matrix,
            ).to(dtype=torch.float32)
        return context_summary.to(dtype=control_states.dtype)

    def _update_control_states_from_features(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        previous_control_states: torch.Tensor,
        node_features: torch.Tensor,
        relation_features: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        attended_question = self._attend_question_context(
            prepared_batch=prepared_batch,
            control_states=previous_control_states,
            graph_ids=graph_ids,
        )
        update_inputs = self.control_input_norm(
            torch.cat(
                (
                    attended_question.to(dtype=previous_control_states.dtype),
                    relation_features.to(dtype=previous_control_states.dtype),
                    node_features.to(dtype=previous_control_states.dtype),
                ),
                dim=-1,
            )
        )
        next_control = self.control_update(
            self.control_dropout(update_inputs).to(dtype=torch.float32),
            previous_control_states.to(dtype=torch.float32),
        )
        return self.control_norm(next_control).to(dtype=previous_control_states.dtype)

    def build_start_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        flat_start_nodes = start_nodes.reshape(-1)
        if int(flat_start_nodes.numel()) == 0:
            hidden_dim = int(prepared_batch.question_tokens.size(-1))
            return prepared_batch.question_tokens.new_empty(
                (*start_nodes.shape, hidden_dim)
            )
        graph_ids = prepared_batch.topology.graph_index_from_nodes(flat_start_nodes)
        root_control_states = prepared_batch.question_tokens.index_select(0, graph_ids)
        start_node_features = prepared_batch.node_tokens.index_select(
            0, flat_start_nodes
        )
        start_relation_features = (
            self.start_relation_feature.to(dtype=start_node_features.dtype)
            .unsqueeze(0)
            .expand(int(flat_start_nodes.numel()), -1)
        )
        start_control_states = self._update_control_states_from_features(
            prepared_batch=prepared_batch,
            previous_control_states=root_control_states,
            node_features=start_node_features,
            relation_features=start_relation_features,
            graph_ids=graph_ids,
        )
        return start_control_states.view(*start_nodes.shape, -1)

    def compute_next_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        control_states: torch.Tensor,
        next_nodes: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(control_states.shape[:-1]) != tuple(next_nodes.shape):
            raise ValueError(
                "control_states must align with next_nodes when updating the recurrent controller. "
                f"control_states={tuple(control_states.shape)} next_nodes={tuple(next_nodes.shape)}."
            )
        if tuple(relation_ids.shape) != tuple(next_nodes.shape):
            raise ValueError(
                "relation_ids must align with next_nodes when updating the recurrent controller. "
                f"relation_ids={tuple(relation_ids.shape)} next_nodes={tuple(next_nodes.shape)}."
            )
        flat_next_nodes = next_nodes.reshape(-1)
        if int(flat_next_nodes.numel()) == 0:
            return control_states
        graph_ids = prepared_batch.topology.graph_index_from_nodes(flat_next_nodes)
        next_node_features = prepared_batch.node_tokens.index_select(0, flat_next_nodes)
        relation_features = prepared_batch.relation_tokens.index_select(
            0, relation_ids.reshape(-1)
        )
        next_control_states = self._update_control_states_from_features(
            prepared_batch=prepared_batch,
            previous_control_states=control_states.view(
                -1, int(control_states.size(-1))
            ),
            node_features=next_node_features,
            relation_features=relation_features,
            graph_ids=graph_ids,
        )
        return next_control_states.view_as(control_states)

    def _reconstruct_flat_control_states(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        flat_path_token_ids: torch.Tensor,
        flat_num_steps: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        del graph_ids
        if int(flat_path_token_ids.size(0)) == 0:
            hidden_dim = int(prepared_batch.question_tokens.size(-1))
            return prepared_batch.question_tokens.new_empty((0, hidden_dim))
        control_states = self.build_start_control_states(
            prepared_batch,
            flat_path_token_ids[:, 0].view(-1, 1),
        ).view(int(flat_path_token_ids.size(0)), -1)
        max_num_steps = (
            int(flat_num_steps.max().item()) if int(flat_num_steps.numel()) > 0 else 0
        )
        for step_idx in range(1, max_num_steps + 1):
            active_mask = flat_num_steps >= step_idx
            if not bool(active_mask.any().item()):
                break
            relation_ids = flat_path_token_ids[active_mask, (2 * step_idx) - 1]
            next_nodes = flat_path_token_ids[active_mask, 2 * step_idx]
            control_states[active_mask] = self.compute_next_control_states(
                prepared_batch,
                control_states=control_states[active_mask],
                next_nodes=next_nodes,
                relation_ids=relation_ids,
            )
        return control_states

    def _resolve_flat_control_states(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor:
        if state.control_state is not None:
            return state.flatten_control_state().to(
                dtype=prepared_batch.node_tokens.dtype
            )
        flat_num_steps = state.flatten_num_steps()
        if state.path_token_ids is None:
            if bool((flat_num_steps != 0).any().item()):
                raise ValueError(
                    "Non-root state features require exact path_token_ids or control_state. "
                    "The recurrent controller cannot reconstruct prefix history from (current_node, num_steps) alone."
                )
            return self.build_start_control_states(
                prepared_batch,
                state.current_nodes,
            ).view(-1, int(prepared_batch.question_tokens.size(-1)))
        return self._reconstruct_flat_control_states(
            prepared_batch=prepared_batch,
            flat_path_token_ids=state.flatten_path_token_ids(max_steps=self.max_steps),
            flat_num_steps=flat_num_steps,
            graph_ids=state.flatten_graph_index(),
        )

    def build_state_features(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> torch.Tensor:
        batch_size, num_rollouts = state.current_nodes.shape
        flat_control_states = self._resolve_flat_control_states(prepared_batch, state)
        flat_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=state.flatten_current_nodes(),
            flat_num_steps=state.flatten_num_steps(),
            flat_done_mask=state.flatten_done_mask(),
            flat_control_states=flat_control_states,
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
        del graph_ids
        total_candidates = int(edge_ids.numel())
        if total_candidates == 0:
            return torch.empty((0,), device=edge_ids.device, dtype=torch.float32)
        relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_ids)
        relation_features = prepared_batch.relation_tokens.index_select(0, relation_ids)
        chunk_size = total_candidates
        if edge_ids.device.type == "cuda":
            chunk_size = min(total_candidates, _TRANSITION_LOGIT_CHUNK_SIZE)
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, total_candidates, max(chunk_size, 1)):
            end = min(start + max(chunk_size, 1), total_candidates)
            logits_chunks.append(
                head(
                    current_state_features=current_state_features[start:end],
                    candidate_state_features=candidate_state_features[start:end],
                    relation_features=relation_features[start:end],
                )
            )
        return _mask_nonfinite_scores(torch.cat(logits_chunks, dim=0))

    def _compute_stop_action_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        current_state_features: torch.Tensor,
        current_node_features: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        total_candidates = int(graph_ids.numel())
        if total_candidates == 0:
            return torch.empty((0,), device=graph_ids.device, dtype=torch.float32)
        stop_action_relation = self.stop_action_relation_feature.to(
            dtype=current_state_features.dtype
        )
        chunk_size = total_candidates
        if graph_ids.device.type == "cuda":
            chunk_size = min(total_candidates, _TRANSITION_LOGIT_CHUNK_SIZE)
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, total_candidates, max(chunk_size, 1)):
            end = min(start + max(chunk_size, 1), total_candidates)
            chunk_size_current = end - start
            logits_chunks.append(
                self.forward_policy_head(
                    current_state_features=current_state_features[start:end],
                    candidate_state_features=current_node_features[start:end],
                    relation_features=stop_action_relation.unsqueeze(0).expand(
                        chunk_size_current, -1
                    ),
                )
            )
        return _mask_nonfinite_scores(torch.cat(logits_chunks, dim=0))

    @staticmethod
    def _distribution_stop_mask(
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor:
        if distribution.is_stop_action is None:
            return torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
        return distribution.is_stop_action.to(dtype=torch.bool)

    @staticmethod
    def _filter_edge_candidate_tensors(
        *,
        edge_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_agent_batch: torch.Tensor,
        child_num_steps: torch.Tensor,
        out_degrees: torch.Tensor,
        keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(keep_mask.shape) != tuple(edge_ids.shape):
            raise ValueError(
                "keep_mask must align with edge_ids when filtering forward candidates. "
                f"keep_mask={tuple(keep_mask.shape)} edge_ids={tuple(edge_ids.shape)}."
            )
        if int(edge_ids.numel()) == 0 or bool(keep_mask.all().item()):
            return (
                edge_ids,
                target_nodes,
                edge_agent_batch,
                child_num_steps,
                out_degrees,
            )
        filtered_edge_ids = edge_ids[keep_mask]
        filtered_target_nodes = target_nodes[keep_mask]
        filtered_edge_agent_batch = edge_agent_batch[keep_mask]
        filtered_child_num_steps = child_num_steps[keep_mask]
        filtered_out_degrees = torch.zeros_like(out_degrees).view(-1)
        if int(filtered_edge_agent_batch.numel()) > 0:
            filtered_out_degrees.scatter_add_(
                0,
                filtered_edge_agent_batch,
                torch.ones_like(
                    filtered_edge_agent_batch, dtype=filtered_out_degrees.dtype
                ),
            )
        return (
            filtered_edge_ids,
            filtered_target_nodes,
            filtered_edge_agent_batch,
            filtered_child_num_steps,
            filtered_out_degrees.view_as(out_degrees),
        )

    def _compute_forward_edge_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        flat_state_features: torch.Tensor,
        edge_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_agent_batch: torch.Tensor,
        child_num_steps: torch.Tensor,
        graph_ids: torch.Tensor,
    ) -> torch.Tensor:
        _ = child_num_steps
        if int(edge_ids.numel()) == 0:
            return torch.empty((0,), device=edge_ids.device, dtype=torch.float32)
        total_edges = int(edge_ids.numel())
        chunk_size = total_edges
        if edge_ids.device.type == "cuda":
            chunk_size = min(total_edges, _FORWARD_EDGE_CHUNK_SIZE)
        edge_logits_chunks: list[torch.Tensor] = []
        for start in range(0, total_edges, max(chunk_size, 1)):
            end = min(start + max(chunk_size, 1), total_edges)
            chunk_edge_agent_batch = edge_agent_batch[start:end]
            chunk_edge_ids = edge_ids[start:end]
            chunk_target_nodes = target_nodes[start:end]
            chunk_child_node_features = prepared_batch.node_tokens.index_select(
                0, chunk_target_nodes
            )
            edge_logits_chunks.append(
                self._compute_transition_logits(
                    prepared_batch=prepared_batch,
                    current_state_features=flat_state_features.index_select(
                        0, chunk_edge_agent_batch
                    ),
                    candidate_state_features=chunk_child_node_features,
                    edge_ids=chunk_edge_ids,
                    graph_ids=graph_ids[start:end],
                    head=self.forward_policy_head,
                )
            )
        return torch.cat(edge_logits_chunks, dim=0)

    def _expected_backward_transitions(
        self,
        *,
        state: SearchState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_num_steps = state.flatten_num_steps()
        flat_path_token_ids = state.flatten_path_token_ids(max_steps=self.max_steps)
        parent_step_ids = (flat_num_steps - 1).clamp_min(0)
        parent_node_positions = (2 * parent_step_ids).to(dtype=torch.long)
        parent_relation_positions = (parent_node_positions + 1).to(dtype=torch.long)
        row_idx = torch.arange(
            int(flat_num_steps.numel()),
            device=flat_num_steps.device,
            dtype=torch.long,
        )
        expected_parent_nodes = flat_path_token_ids[row_idx, parent_node_positions]
        expected_relation_ids = flat_path_token_ids[row_idx, parent_relation_positions]
        return expected_parent_nodes, expected_relation_ids

    def _compute_tree_backward_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        edge_ids: torch.Tensor,
        source_nodes: torch.Tensor,
        edge_agent_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the unique parent edge on the prefix-tree state space."""

        edge_logits = torch.full(
            (int(edge_ids.numel()),),
            fill_value=float("-inf"),
            device=edge_ids.device,
            dtype=torch.float32,
        )
        if int(edge_ids.numel()) == 0:
            return edge_logits

        (
            expected_parent_nodes,
            expected_relation_ids,
        ) = self._expected_backward_transitions(state=state)
        relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_ids)
        valid_edges = (
            source_nodes == expected_parent_nodes.index_select(0, edge_agent_batch)
        ) & (relation_ids == expected_relation_ids.index_select(0, edge_agent_batch))
        edge_logits = torch.where(
            valid_edges, torch.zeros_like(edge_logits), edge_logits
        )

        valid_counts = torch.zeros(
            (int(state.current_nodes.numel()),),
            device=edge_ids.device,
            dtype=torch.long,
        )
        if int(valid_edges.numel()) > 0:
            valid_counts.scatter_add_(
                0, edge_agent_batch, valid_edges.to(dtype=torch.long)
            )
        active_mask = (~state.flatten_done_mask()) & (state.flatten_num_steps() > 0)
        if not bool((valid_counts[active_mask] > 0).all().item()):
            invalid_agents = torch.nonzero(
                active_mask & (valid_counts <= 0), as_tuple=False
            ).view(-1)
            raise RuntimeError(
                "Backward distribution could not recover a parent edge from the encoded path. "
                f"invalid_agents={invalid_agents.tolist()}"
            )
        return edge_logits

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
        forward_active_mask = active_mask & (flat_num_steps < self.max_steps)
        (
            edge_ids,
            target_nodes,
            edge_agent_batch,
            out_degrees,
        ) = state.topology.gather_outgoing_edges(
            current_nodes=flat_current_nodes,
            active_mask=forward_active_mask,
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

    @staticmethod
    def _deduplicate_active_forward_states(
        *,
        state: SearchState,
        flat_path_token_ids: torch.Tensor | None,
        flat_control_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        active_agents = torch.nonzero(~state.flatten_done_mask(), as_tuple=False).view(
            -1
        )
        if int(active_agents.numel()) == 0:
            empty_long = state.current_nodes.new_empty((0,), dtype=torch.long)
            empty_control = flat_control_states.new_empty(
                (0, int(flat_control_states.size(-1))),
            )
            return (
                active_agents,
                empty_long,
                empty_long,
                empty_long,
                empty_control,
                None,
            )
        active_nodes = state.flatten_current_nodes().index_select(0, active_agents)
        active_num_steps = state.flatten_num_steps().index_select(0, active_agents)
        active_control_states = flat_control_states.index_select(0, active_agents)
        if int(active_agents.numel()) == 1:
            active_path_token_rows = (
                None
                if flat_path_token_ids is None
                else flat_path_token_ids.index_select(0, active_agents)
            )
            return (
                active_agents,
                torch.zeros_like(active_agents),
                active_nodes,
                active_num_steps,
                active_control_states,
                active_path_token_rows,
            )
        if flat_path_token_ids is None:
            if bool((active_num_steps != 0).any().item()):
                unique_index = torch.arange(
                    int(active_agents.numel()),
                    device=active_agents.device,
                    dtype=torch.long,
                )
                return (
                    active_agents,
                    unique_index,
                    active_nodes,
                    active_num_steps,
                    active_control_states,
                    None,
                )
            unique_state_rows, active_to_unique = torch.unique(
                torch.stack((active_nodes, active_num_steps), dim=1),
                dim=0,
                return_inverse=True,
            )
            sorted_inverse = torch.argsort(active_to_unique, stable=True)
            _, inverse_counts = torch.unique_consecutive(
                active_to_unique.index_select(0, sorted_inverse),
                return_counts=True,
            )
            inverse_offsets = inverse_counts.cumsum(0) - inverse_counts
            representative_positions = sorted_inverse.index_select(0, inverse_offsets)
            return (
                active_agents,
                active_to_unique,
                unique_state_rows[:, 0].to(dtype=torch.long),
                unique_state_rows[:, 1].to(dtype=torch.long),
                active_control_states.index_select(0, representative_positions),
                None,
            )
        active_path_token_ids = flat_path_token_ids.index_select(0, active_agents)
        # Prefix history is part of the state definition, so deduplication must
        # key on the full encoded trajectory prefix instead of node/time alone.
        unique_state_rows, active_to_unique = torch.unique(
            torch.cat(
                (
                    active_nodes.unsqueeze(1),
                    active_num_steps.unsqueeze(1),
                    active_path_token_ids,
                ),
                dim=1,
            ),
            dim=0,
            return_inverse=True,
        )
        sorted_inverse = torch.argsort(active_to_unique, stable=True)
        _, inverse_counts = torch.unique_consecutive(
            active_to_unique.index_select(0, sorted_inverse),
            return_counts=True,
        )
        inverse_offsets = inverse_counts.cumsum(0) - inverse_counts
        representative_positions = sorted_inverse.index_select(0, inverse_offsets)
        return (
            active_agents,
            active_to_unique,
            unique_state_rows[:, 0].to(dtype=torch.long),
            unique_state_rows[:, 1].to(dtype=torch.long),
            active_control_states.index_select(0, representative_positions),
            active_path_token_ids.index_select(0, representative_positions),
        )

    @staticmethod
    def _expand_unique_edge_candidates(
        *,
        active_agents: torch.Tensor,
        active_to_unique: torch.Tensor,
        unique_edge_ids: torch.Tensor,
        unique_target_nodes: torch.Tensor,
        unique_edge_agent_batch: torch.Tensor,
        unique_edge_logits: torch.Tensor,
        unique_out_degrees: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if int(active_agents.numel()) == 0 or int(unique_edge_ids.numel()) == 0:
            empty_logits = unique_edge_logits.new_empty((0,))
            empty_long = unique_edge_ids.new_empty((0,))
            return empty_logits, empty_long, empty_long, empty_long
        del unique_edge_agent_batch
        unique_edge_ptr = torch.zeros(
            (int(unique_out_degrees.numel()) + 1),
            device=unique_out_degrees.device,
            dtype=torch.long,
        )
        unique_edge_ptr[1:] = unique_out_degrees.to(dtype=torch.long).cumsum(0)
        active_edge_counts = unique_out_degrees.index_select(0, active_to_unique).to(
            dtype=torch.long
        )
        expanded_edge_agent_batch = active_agents.repeat_interleave(active_edge_counts)
        if int(expanded_edge_agent_batch.numel()) == 0:
            empty_logits = unique_edge_logits.new_empty((0,))
            empty_long = unique_edge_ids.new_empty((0,))
            return empty_logits, empty_long, empty_long, empty_long
        base_index = unique_edge_ptr[:-1].index_select(0, active_to_unique)
        base_index = base_index.repeat_interleave(active_edge_counts)
        segment_starts = active_edge_counts.cumsum(0) - active_edge_counts
        expanded_positions = base_index + (
            torch.arange(
                int(expanded_edge_agent_batch.numel()),
                device=unique_edge_ids.device,
                dtype=torch.long,
            )
            - segment_starts.repeat_interleave(active_edge_counts)
        )
        return (
            unique_edge_logits.index_select(0, expanded_positions),
            expanded_edge_agent_batch,
            unique_edge_ids.index_select(0, expanded_positions),
            unique_target_nodes.index_select(0, expanded_positions),
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
        (
            edge_ids,
            source_nodes,
            edge_agent_batch,
            in_degrees,
        ) = state.topology.gather_incoming_edges(
            current_nodes=flat_current_nodes,
            active_mask=backward_active_mask,
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

    def _compute_forward_distribution_impl(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution:
        _ = required_edge_ids
        total_agents = int(state.current_nodes.numel())
        flat_current_nodes = state.flatten_current_nodes()
        flat_done_mask = state.flatten_done_mask()
        flat_num_steps = state.flatten_num_steps()
        flat_path_token_ids = None
        if state.path_token_ids is not None:
            flat_path_token_ids = state.flatten_path_token_ids(max_steps=self.max_steps)
        flat_control_states = self._resolve_flat_control_states(prepared_batch, state)
        (
            active_agents,
            active_to_unique,
            unique_current_nodes,
            unique_num_steps,
            unique_control_states,
            unique_path_token_ids,
        ) = self._deduplicate_active_forward_states(
            state=state,
            flat_path_token_ids=flat_path_token_ids,
            flat_control_states=flat_control_states,
        )
        active_agent_count = int(active_agents.numel())
        unique_active_state_count = int(unique_current_nodes.numel())
        hidden_dim = int(prepared_batch.node_tokens.size(-1))
        unique_state_features = prepared_batch.node_tokens.new_empty((0, hidden_dim))
        unique_graph_ids = prepared_batch.node_tokens.new_empty((0,), dtype=torch.long)
        if int(unique_current_nodes.numel()) > 0:
            unique_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=unique_current_nodes,
                flat_num_steps=unique_num_steps,
                flat_done_mask=torch.zeros_like(unique_num_steps, dtype=torch.bool),
                flat_control_states=unique_control_states,
            )
            unique_graph_ids = state.topology.graph_index_from_nodes(
                unique_current_nodes
            )
        flat_state_features = torch.zeros(
            (total_agents, hidden_dim),
            device=state.current_nodes.device,
            dtype=prepared_batch.node_tokens.dtype,
        )
        if int(active_agents.numel()) > 0:
            flat_state_features.index_copy_(
                0,
                active_agents,
                unique_state_features.index_select(0, active_to_unique),
            )
        unique_forward_active_mask = unique_num_steps < self.max_steps
        (
            unique_edge_ids,
            unique_target_nodes,
            unique_edge_agent_batch,
            unique_out_degrees,
        ) = state.topology.gather_outgoing_edges(
            current_nodes=unique_current_nodes,
            active_mask=unique_forward_active_mask,
        )
        if int(unique_edge_ids.numel()) > 0:
            unique_child_num_steps = (
                unique_num_steps.index_select(0, unique_edge_agent_batch) + 1
            )
            legal_fresh_entity_mask = ~build_entity_revisit_mask_from_flat_state(
                flat_current_abs_nodes=unique_current_nodes,
                flat_num_steps=unique_num_steps,
                flat_path_token_ids=unique_path_token_ids,
                node_entity_ids_by_abs_node=state.observation.node_entity_ids,
                num_nodes=int(state.topology.num_nodes),
                candidate_target_abs_nodes=unique_target_nodes,
                candidate_agent_indices=unique_edge_agent_batch,
            )
            (
                unique_edge_ids,
                unique_target_nodes,
                unique_edge_agent_batch,
                unique_child_num_steps,
                unique_out_degrees,
            ) = self._filter_edge_candidate_tensors(
                edge_ids=unique_edge_ids,
                target_nodes=unique_target_nodes,
                edge_agent_batch=unique_edge_agent_batch,
                child_num_steps=unique_child_num_steps,
                out_degrees=unique_out_degrees,
                keep_mask=legal_fresh_entity_mask,
            )
        else:
            unique_child_num_steps = unique_num_steps.new_empty((0,))
        raw_graph_candidate_count = int(unique_edge_ids.numel())
        scored_graph_candidate_count = int(unique_edge_ids.numel())
        current_log_f_flat = torch.zeros(
            (total_agents,), device=state.current_nodes.device, dtype=torch.float32
        )
        if int(active_agents.numel()) > 0:
            unique_current_log_f = self._compute_log_state_scores_from_flat_features(
                prepared_batch=prepared_batch,
                flat_state_features=unique_state_features,
                graph_ids=unique_graph_ids,
            )
            current_log_f_flat.index_copy_(
                0,
                active_agents,
                unique_current_log_f.index_select(0, active_to_unique),
            )
        current_log_f = current_log_f_flat.view_as(state.current_nodes)
        edge_logits = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.float32
        )
        edge_agent_batch = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        edge_ids = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        target_nodes = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        stop_action_logits = torch.empty_like(edge_logits)
        stop_action_agent_batch = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        stop_action_target_nodes = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        stop_action_edge_ids = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.long
        )
        flat_graph_out_degrees = torch.zeros(
            (total_agents,), device=state.current_nodes.device, dtype=torch.long
        )
        if int(active_agents.numel()) > 0:
            flat_graph_out_degrees.index_copy_(
                0,
                active_agents,
                unique_out_degrees.index_select(0, active_to_unique),
            )
        if int(unique_edge_ids.numel()) > 0:
            unique_edge_logits = self._compute_forward_edge_logits(
                prepared_batch=prepared_batch,
                flat_state_features=unique_state_features,
                edge_ids=unique_edge_ids,
                target_nodes=unique_target_nodes,
                edge_agent_batch=unique_edge_agent_batch,
                child_num_steps=unique_child_num_steps,
                graph_ids=unique_graph_ids.index_select(0, unique_edge_agent_batch),
            )
            (
                edge_logits,
                edge_agent_batch,
                edge_ids,
                target_nodes,
            ) = self._expand_unique_edge_candidates(
                active_agents=active_agents,
                active_to_unique=active_to_unique,
                unique_edge_ids=unique_edge_ids,
                unique_target_nodes=unique_target_nodes,
                unique_edge_agent_batch=unique_edge_agent_batch,
                unique_edge_logits=unique_edge_logits,
                unique_out_degrees=unique_out_degrees,
            )
        if int(active_agents.numel()) > 0:
            stop_action_agent_batch = active_agents
            stop_action_target_nodes = flat_current_nodes.index_select(0, active_agents)
            stop_action_edge_ids = torch.full_like(
                stop_action_target_nodes, fill_value=-1
            )
            unique_stop_action_logits = self._compute_stop_action_logits(
                prepared_batch=prepared_batch,
                current_state_features=unique_state_features,
                current_node_features=prepared_batch.node_tokens.index_select(
                    0, unique_current_nodes
                ),
                graph_ids=unique_graph_ids,
            )
            stop_action_logits = unique_stop_action_logits.index_select(
                0, active_to_unique
            )
        combined_edge_logits = torch.cat((edge_logits, stop_action_logits), dim=0)
        combined_edge_agent_batch = torch.cat(
            (edge_agent_batch, stop_action_agent_batch), dim=0
        )
        combined_edge_ids = torch.cat((edge_ids, stop_action_edge_ids), dim=0)
        combined_target_nodes = torch.cat(
            (target_nodes, stop_action_target_nodes), dim=0
        )
        is_stop_action = torch.cat(
            (
                torch.zeros_like(edge_ids, dtype=torch.bool),
                torch.ones_like(stop_action_edge_ids, dtype=torch.bool),
            ),
            dim=0,
        )
        if int(combined_edge_agent_batch.numel()) > 0:
            order = torch.argsort(combined_edge_agent_batch, stable=True)
            combined_edge_logits = combined_edge_logits.index_select(0, order)
            combined_edge_agent_batch = combined_edge_agent_batch.index_select(0, order)
            combined_edge_ids = combined_edge_ids.index_select(0, order)
            combined_target_nodes = combined_target_nodes.index_select(0, order)
            is_stop_action = is_stop_action.index_select(0, order)
        out_degrees = flat_graph_out_degrees.view_as(state.current_nodes)
        combined_out_degrees = out_degrees + (~flat_done_mask).view_as(out_degrees).to(
            dtype=out_degrees.dtype
        )
        return ForwardActionDistribution(
            edge_logits=combined_edge_logits.to(dtype=prepared_batch.node_tokens.dtype),
            edge_agent_batch=combined_edge_agent_batch,
            edge_ids=combined_edge_ids,
            target_nodes=combined_target_nodes,
            out_degrees=combined_out_degrees,
            is_stop_action=is_stop_action,
            current_log_f=current_log_f,
            active_agent_count=active_agent_count,
            unique_active_state_count=unique_active_state_count,
            raw_graph_candidate_count=raw_graph_candidate_count,
            scored_graph_candidate_count=scored_graph_candidate_count,
        )

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution:
        return self._compute_forward_distribution_impl(
            prepared_batch,
            state,
            required_edge_ids=required_edge_ids,
        )

    def compute_backward_distribution(
        self,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        flat_done_mask = state.flatten_done_mask()
        flat_absorbing_mask = state.flatten_absorbing_mask()
        flat_num_steps = state.flatten_num_steps()
        flat_current_nodes = state.flatten_current_nodes()
        active_non_root = (~flat_done_mask) & (flat_num_steps > 0)
        active_start = (~flat_done_mask) & (flat_num_steps == 0)
        absorbing_mask = flat_absorbing_mask
        if bool(active_non_root.any().item()):
            state.flatten_path_token_ids(max_steps=self.max_steps)
        (
            edge_ids,
            source_nodes,
            edge_agent_batch,
            in_degrees,
            _,
        ) = self._gather_backward_candidates(state)
        if bool(active_non_root.any().item()) and int(edge_ids.numel()) == 0:
            invalid_agents = torch.nonzero(active_non_root, as_tuple=False).view(-1)
            raise RuntimeError(
                "Backward distribution could not recover a parent edge from the encoded path. "
                f"invalid_agents={invalid_agents.tolist()}"
            )
        edge_logits = torch.empty(
            (0,), device=state.current_nodes.device, dtype=torch.float32
        )
        is_stop_action = torch.zeros_like(edge_ids, dtype=torch.bool)
        is_root_action = torch.zeros_like(edge_ids, dtype=torch.bool)
        if int(edge_ids.numel()) > 0:
            edge_logits = self._compute_tree_backward_logits(
                prepared_batch=prepared_batch,
                state=state,
                edge_ids=edge_ids,
                source_nodes=source_nodes,
                edge_agent_batch=edge_agent_batch,
            )
        if bool(active_start.any().item()):
            root_agents = torch.nonzero(active_start, as_tuple=False).view(-1)
            root_edge_ids = torch.full_like(root_agents, fill_value=-2)
            root_source_nodes = flat_current_nodes.index_select(0, root_agents)
            root_logits = torch.zeros_like(root_agents, dtype=torch.float32)
            edge_ids = torch.cat((edge_ids, root_edge_ids), dim=0)
            source_nodes = torch.cat((source_nodes, root_source_nodes), dim=0)
            edge_agent_batch = torch.cat((edge_agent_batch, root_agents), dim=0)
            edge_logits = torch.cat((edge_logits, root_logits), dim=0)
            is_stop_action = torch.cat(
                (is_stop_action, torch.zeros_like(root_agents, dtype=torch.bool)), dim=0
            )
            is_root_action = torch.cat(
                (is_root_action, torch.ones_like(root_agents, dtype=torch.bool)), dim=0
            )
        if bool(absorbing_mask.any().item()):
            stop_agents = torch.nonzero(absorbing_mask, as_tuple=False).view(-1)
            stop_edge_ids = torch.full_like(stop_agents, fill_value=-1)
            stop_source_nodes = flat_current_nodes.index_select(0, stop_agents)
            stop_logits = torch.zeros_like(stop_agents, dtype=torch.float32)
            edge_ids = torch.cat((edge_ids, stop_edge_ids), dim=0)
            source_nodes = torch.cat((source_nodes, stop_source_nodes), dim=0)
            edge_agent_batch = torch.cat((edge_agent_batch, stop_agents), dim=0)
            edge_logits = torch.cat((edge_logits, stop_logits), dim=0)
            is_stop_action = torch.cat(
                (is_stop_action, torch.ones_like(stop_agents, dtype=torch.bool)), dim=0
            )
            is_root_action = torch.cat(
                (is_root_action, torch.zeros_like(stop_agents, dtype=torch.bool)), dim=0
            )
        if int(edge_agent_batch.numel()) > 0:
            order = torch.argsort(edge_agent_batch, stable=True)
            edge_ids = edge_ids.index_select(0, order)
            source_nodes = source_nodes.index_select(0, order)
            edge_agent_batch = edge_agent_batch.index_select(0, order)
            edge_logits = edge_logits.index_select(0, order)
            is_stop_action = is_stop_action.index_select(0, order)
            is_root_action = is_root_action.index_select(0, order)
        if bool(active_start.any().item()):
            in_degrees.view(-1)[active_start] = 1
        if bool(absorbing_mask.any().item()):
            in_degrees.view(-1)[absorbing_mask] = 1
        return ForwardActionDistribution(
            edge_logits=edge_logits.to(dtype=prepared_batch.node_tokens.dtype),
            edge_agent_batch=edge_agent_batch,
            edge_ids=edge_ids,
            target_nodes=source_nodes,
            out_degrees=in_degrees,
            is_stop_action=is_stop_action,
            is_root_action=is_root_action,
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
        with torch.no_grad():
            heuristic_cache = self.search_heuristic.build_cache(
                prepared_batch,
                build_local_state_features=self.base_policy.build_local_state_features,
                max_steps=self.base_policy.max_steps,
            )
        return PreparedGFlowNetBatch(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            node_tokens=prepared_batch.node_tokens,
            relation_tokens=prepared_batch.relation_tokens,
            question_tokens=prepared_batch.question_tokens,
            question_context_tokens=prepared_batch.question_context_tokens,
            question_context_mask=prepared_batch.question_context_mask,
            heuristic_cache=heuristic_cache,
        )

    def encode(self, batch) -> PreparedGFlowNetBatch:
        return self.prepare_batch(batch)

    def compute_graph_log_z(
        self, prepared_batch: PreparedGFlowNetBatch
    ) -> torch.Tensor:
        return self.base_policy.compute_graph_log_z(prepared_batch).to(
            dtype=torch.float32
        )

    def _compute_state_bias(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        node_abs_indices: torch.Tensor,
        num_steps: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.search_heuristic.compute_cached_bias(
            heuristic_cache=prepared_batch.heuristic_cache,
            node_abs_indices=node_abs_indices,
            num_steps=num_steps,
            done_mask=done_mask,
        )

    def compute_guidance_logits(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        *,
        detach_features: bool = False,
    ) -> torch.Tensor:
        return self.search_heuristic.compute_state_logits(
            prepared_batch=prepared_batch,
            state=state,
            build_state_features=self.base_policy.build_state_features,
            detach_features=detach_features,
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
        stop_action_mask = self.base_policy._distribution_stop_mask(distribution)
        if bool(stop_action_mask.all().item()):
            return edge_logits
        graph_mask = ~stop_action_mask
        child_target_nodes = distribution.target_nodes[graph_mask]
        child_num_steps = (
            state.flatten_num_steps().index_select(
                0, distribution.edge_agent_batch[graph_mask]
            )
            + 1
        )
        child_bias = self._compute_state_bias(
            prepared_batch=prepared_batch,
            node_abs_indices=child_target_nodes,
            num_steps=child_num_steps,
            done_mask=torch.zeros_like(child_num_steps, dtype=torch.bool),
        )
        adjusted_logits = edge_logits.clone()
        adjusted_logits[graph_mask] = _mask_nonfinite_scores(
            adjusted_logits[graph_mask] + float(self.heuristic_cfg.beta) * child_bias
        )
        return adjusted_logits

    def compute_behavior_edge_logits(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor:
        return self._compute_behavior_forward_logits(
            prepared_batch=prepared_batch,
            state=state,
            distribution=distribution,
        )

    def build_start_control_states(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        return self.base_policy.build_start_control_states(prepared_batch, start_nodes)

    def compute_next_control_states(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        *,
        control_states: torch.Tensor,
        next_nodes: torch.Tensor,
        relation_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.base_policy.compute_next_control_states(
            prepared_batch,
            control_states=control_states,
            next_nodes=next_nodes,
            relation_ids=relation_ids,
        )

    def compute_root_action_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> RootActionDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        start_state_log_flows = self.base_policy.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        action_logits = self.base_policy.compute_root_action_logits(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return build_root_action_distribution(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_logits=action_logits,
            start_state_log_flows=start_state_log_flows,
            graph_log_z=self.base_policy.compute_graph_log_z(prepared_batch),
        )

    def compute_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> RootActionDistribution:
        return self.compute_root_action_distribution(prepared_batch)

    def compute_behavior_root_action_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> RootActionDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        start_state_log_flows = self.base_policy.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        action_logits = self.base_policy.compute_root_action_logits(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_bias = self._compute_state_bias(
            prepared_batch=prepared_batch,
            node_abs_indices=candidate_nodes_abs,
            num_steps=torch.zeros_like(candidate_nodes_abs, dtype=torch.long),
            done_mask=torch.zeros_like(candidate_nodes_abs, dtype=torch.bool),
        )
        return build_root_action_distribution(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_logits=action_logits + float(self.heuristic_cfg.beta) * start_bias,
            start_state_log_flows=start_state_log_flows,
            graph_log_z=self.base_policy.compute_graph_log_z(prepared_batch),
        )

    def compute_behavior_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> RootActionDistribution:
        return self.compute_behavior_root_action_distribution(prepared_batch)

    @staticmethod
    def sample_root_start_nodes(
        distribution: RootActionDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return GFlowNetPolicy.sample_start_nodes(
            distribution,
            num_rollouts=num_rollouts,
            deterministic=deterministic,
            temperature=temperature,
        )

    @staticmethod
    def sample_start_nodes(
        distribution: RootActionDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = int(distribution.graph_log_z.numel())
        selected_nodes: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        selected_log_flows: list[torch.Tensor] = []
        sampling_log_probs = _temper_segmented_log_probs(
            log_probs=distribution.log_probs,
            segment_ids=distribution.candidate_graph_ids,
            num_segments=num_graphs,
            temperature=float(temperature),
        )
        if num_graphs < 1:
            empty = distribution.candidate_nodes_abs.new_empty((0, num_rollouts))
            empty_scores = sampling_log_probs.new_empty((0, num_rollouts))
            return empty, empty_scores, empty_scores
        for graph_idx in range(num_graphs):
            mask = distribution.candidate_graph_ids == graph_idx
            graph_nodes = distribution.candidate_nodes_abs[mask]
            graph_log_probs = sampling_log_probs[mask]
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
            selected_nodes.append(graph_nodes)
            selected_log_probs.append(graph_log_probs)
            selected_log_flows.append(graph_log_flows)
        if deterministic:
            return (
                torch.stack(selected_nodes, dim=0),
                torch.stack(selected_log_probs, dim=0),
                torch.stack(selected_log_flows, dim=0),
            )
        sampled_nodes: list[torch.Tensor] = []
        sampled_log_probs: list[torch.Tensor] = []
        sampled_log_flows: list[torch.Tensor] = []
        for _ in range(num_rollouts):
            sampled_positions, sampled_probs, has_values = sample_segmented_one_1d(
                logits=sampling_log_probs,
                segment_ids=distribution.candidate_graph_ids,
                num_segments=num_graphs,
                temperature=1.0,
            )
            if not bool(has_values.all().item()):
                raise ValueError("Each graph must expose at least one start candidate.")
            sampled_nodes.append(
                distribution.candidate_nodes_abs.index_select(0, sampled_positions)
            )
            sampled_log_probs.append(sampled_probs)
            sampled_log_flows.append(
                distribution.log_flows.index_select(0, sampled_positions)
            )
        return (
            torch.stack(sampled_nodes, dim=1),
            torch.stack(sampled_log_probs, dim=1),
            torch.stack(sampled_log_flows, dim=1),
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
        *,
        required_edge_ids: torch.Tensor | None = None,
    ) -> ForwardActionDistribution:
        return self.base_policy.compute_forward_distribution(
            prepared_batch,
            state,
            required_edge_ids=required_edge_ids,
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
            is_stop_action=distribution.is_stop_action,
            is_root_action=distribution.is_root_action,
            current_log_f=distribution.current_log_f,
            active_agent_count=distribution.active_agent_count,
            unique_active_state_count=distribution.unique_active_state_count,
            raw_graph_candidate_count=distribution.raw_graph_candidate_count,
            scored_graph_candidate_count=distribution.scored_graph_candidate_count,
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
    "RootActionDistribution",
    "RootActionDistributionError",
    "StartDistributionError",
    "StartDistribution",
    "build_root_action_distribution",
    "resolve_start_candidates",
]

# Backward-compatible aliases for older imports.
StartDistributionError = RootActionDistributionError
StartDistribution = RootActionDistribution
# Backward-compatible alias for older imports.
build_root_action_distribution_from_log_flows = build_root_action_distribution
build_start_distribution_from_log_flows = build_root_action_distribution
