from __future__ import annotations

import math

import torch
from torch import nn

from src.graph import SearchObservation, TrajectoryBatch, build_graph_batch
from src.data.preprocess.labels.edge_retrieval import (
    compute_forward_answer_distances,
    compute_forward_shortest_path_edge_mask,
)
from src.models.components import (
    EmbeddingBackbone,
    NodeFlowHead,
    TransitionPolicyHead,
)
from src.models.components.embedding import BackboneInput
from src.models.configs import ActionPriorConfig, PolicyConfig, PotentialRewardConfig
from src.utils.nn_init import init_linear_xavier
from src.utils.precision_utils import (
    align_float_input_dtype,
    masked_softmax_in_float32,
)
from src.utils.segment_ops import sample_segmented_one_1d, segment_logsumexp_1d

from .answer_supervision import build_answer_mask, build_node_answer_sink_tensors
from .backward import compute_policy_backward_distribution
from .heuristics import SearchActionPrior
from .legality import build_unique_forward_candidate_keep_mask
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


_CANDIDATE_SCORING_CHUNK_SIZE = 1024
_CONTROL_ATTENTION_CHUNK_SIZE = 256


def _candidate_chunk_size(*, device: torch.device, total_candidates: int) -> int:
    if device.type != "cuda":
        return max(total_candidates, 1)
    return max(min(total_candidates, _CANDIDATE_SCORING_CHUNK_SIZE), 1)


def _control_attention_chunk_size(*, device: torch.device, total_states: int) -> int:
    if device.type != "cuda":
        return max(total_states, 1)
    return max(min(total_states, _CONTROL_ATTENTION_CHUNK_SIZE), 1)


def _build_forward_guidance_tensors(
    batch: TrajectoryBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_nodes = int(batch.num_nodes_total)
    total_edges = int(batch.edge_index.size(1))
    answer_distance = torch.full(
        (total_nodes,), fill_value=-1, device=batch.node_ptr.device, dtype=torch.long
    )
    shortest_path_edge_mask = torch.zeros(
        (total_edges,), device=batch.edge_index.device, dtype=torch.bool
    )
    edge_ptr = batch.edge_ptr
    for graph_idx in range(batch.num_graphs):
        node_start = int(batch.node_ptr[graph_idx].item())
        node_end = int(batch.node_ptr[graph_idx + 1].item())
        edge_start = int(edge_ptr[graph_idx].item())
        edge_end = int(edge_ptr[graph_idx + 1].item())
        q_start = int(batch.q_ptr[graph_idx].item())
        q_end = int(batch.q_ptr[graph_idx + 1].item())
        a_start = int(batch.a_ptr[graph_idx].item())
        a_end = int(batch.a_ptr[graph_idx + 1].item())
        local_edge_index = batch.edge_index[:, edge_start:edge_end] - node_start
        local_q = batch.q_local_indices[q_start:q_end]
        local_a = batch.a_local_indices[a_start:a_end]
        local_num_nodes = max(node_end - node_start, 0)
        answer_distance[node_start:node_end] = compute_forward_answer_distances(
            edge_index=local_edge_index,
            a_local_indices=local_a,
            num_nodes=local_num_nodes,
        ).to(device=batch.node_ptr.device)
        shortest_path_edge_mask[edge_start:edge_end] = (
            compute_forward_shortest_path_edge_mask(
                edge_index=local_edge_index,
                q_local_indices=local_q,
                a_local_indices=local_a,
                num_nodes=local_num_nodes,
            ).to(device=batch.edge_index.device)
        )
    return answer_distance, shortest_path_edge_mask


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
    graph_log_z: torch.Tensor | None = None,
    start_log_rewards: torch.Tensor | None = None,
) -> RootActionDistribution:
    num_graphs = int(prepared_batch.topology.num_graphs)
    action_logits = _mask_nonfinite_scores(action_logits.to(dtype=torch.float32))
    start_state_log_flows = _mask_nonfinite_scores(
        start_state_log_flows.to(dtype=torch.float32)
    )
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
    _, has_finite_flows = segment_logsumexp_1d(
        values=start_state_log_flows,
        segment_ids=candidate_graph_ids,
        num_segments=num_graphs,
        dtype=torch.float32,
        ignore_non_finite=True,
        empty_value=float("-inf"),
    )
    if not bool(has_finite_flows.all().item()):
        invalid_graphs = (
            torch.nonzero(~has_finite_flows, as_tuple=False).view(-1).tolist()
        )
        invalid_samples = [
            prepared_batch.observation.sample_ids[idx] for idx in invalid_graphs
        ]
        raise InvalidStartCandidatesError(invalid_samples=invalid_samples)
    return RootActionDistribution(
        candidate_nodes_abs=candidate_nodes_abs,
        candidate_graph_ids=candidate_graph_ids,
        log_flows=start_state_log_flows,
        log_probs=action_logits - lse.index_select(0, candidate_graph_ids),
        graph_log_z=_mask_nonfinite_scores(
            lse if graph_log_z is None else graph_log_z.to(dtype=torch.float32)
        ),
        start_log_rewards=(
            None
            if start_log_rewards is None
            else _mask_nonfinite_scores(start_log_rewards.to(dtype=torch.float32))
        ),
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
        transition_policy_head: TransitionPolicyHead | None,
        step_log_penalty: float,
        non_gold_terminal_log_reward: float,
        answer_stop_log_reward_bonus: float,
        answer_quotient_allocate_stop_mass: bool,
        answer_quotient_gold_reward_mode: str,
        potential_reward_cfg: PotentialRewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if float(step_log_penalty) > 0.0:
            raise ValueError("step_log_penalty must be <= 0 for BaseSearchPolicy.")
        if float(non_gold_terminal_log_reward) > 0.0:
            raise ValueError(
                "non_gold_terminal_log_reward must be <= 0 for BaseSearchPolicy."
            )
        if float(answer_stop_log_reward_bonus) < 0.0:
            raise ValueError(
                "answer_stop_log_reward_bonus must be >= 0 for BaseSearchPolicy."
            )
        if answer_quotient_gold_reward_mode not in {"shared", "unit"}:
            raise ValueError(
                "answer_quotient_gold_reward_mode must be one of {'shared', 'unit'}."
            )

        graph_hidden_dim = int(config.backbone.hidden_dim)
        base_state_dim = graph_hidden_dim * 3
        state_feature_input_dim = base_state_dim + graph_hidden_dim
        root_feature_dim = graph_hidden_dim * 3 + 4
        self.state_flow_head = state_score_head
        self.state_score_head = self.state_flow_head
        self.transition_proposal_head = transition_policy_head
        self.step_log_penalty = float(step_log_penalty)
        self.non_gold_terminal_log_reward = float(non_gold_terminal_log_reward)
        self.answer_stop_log_reward_bonus = float(answer_stop_log_reward_bonus)
        self.answer_quotient_allocate_stop_mass = bool(
            answer_quotient_allocate_stop_mass
        )
        self.answer_quotient_gold_reward_mode = str(answer_quotient_gold_reward_mode)
        if potential_reward_cfg is None:
            potential_reward_cfg = PotentialRewardConfig()
        self.potential_reward_cfg = potential_reward_cfg
        self.answer_distance_potential_weight = float(
            potential_reward_cfg.answer_distance_weight
        )
        self.answer_distance_potential_unreachable_distance = (
            None
            if potential_reward_cfg.unreachable_distance is None
            else int(potential_reward_cfg.unreachable_distance)
        )
        # Keep the legacy root-flow modules registered so older checkpoints still
        # load cleanly; strict successor-flow root mass is now derived from start
        # action log-masses in compute_graph_log_z/compute_root_action_distribution.
        self.root_flow_input_norm = nn.LayerNorm(root_feature_dim)
        self.root_flow_hidden = nn.Linear(root_feature_dim, graph_hidden_dim)
        self.root_flow_activation = nn.GELU()
        self.root_flow_head = nn.Linear(graph_hidden_dim, 1)
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
        self.stop_allocation_input_norm = nn.LayerNorm(graph_hidden_dim)
        self.stop_allocation_head = nn.Linear(graph_hidden_dim, 1)
        self.start_relation_feature = nn.Parameter(torch.zeros(graph_hidden_dim))
        self.backbone = backbone
        init_linear_xavier(self.root_flow_hidden)
        init_linear_xavier(self.stop_allocation_head)
        nn.init.normal_(self.root_flow_head.weight, mean=0.0, std=1.0e-2)
        if self.root_flow_head.bias is not None:
            nn.init.zeros_(self.root_flow_head.bias)

    def prepare_batch(self, batch) -> PreparedSearchBatch:
        if isinstance(batch, TrajectoryBatch):
            batch.require_raw_features()
        topology, observation = build_graph_batch(batch, validate=False)
        answer_mask = torch.zeros(
            (int(topology.num_nodes),),
            device=observation.node_entity_ids.device,
            dtype=torch.bool,
        )
        answer_sink_ids = torch.zeros(
            (int(topology.num_nodes),),
            device=observation.node_entity_ids.device,
            dtype=torch.long,
        )
        answer_sink_log_rewards = torch.full(
            (int(topology.num_nodes),),
            fill_value=self.non_gold_terminal_log_reward,
            device=observation.node_entity_ids.device,
            dtype=torch.float32,
        )
        answer_distance = None
        shortest_path_edge_mask = None
        if all(
            hasattr(batch, field_name)
            for field_name in (
                "node_ptr",
                "node_entity_ids",
                "answer_entity_ids",
                "answer_ptr",
            )
        ):
            answer_mask = build_answer_mask(
                node_ptr=getattr(batch, "node_ptr"),
                node_entity_ids=getattr(batch, "node_entity_ids"),
                answer_entity_ids=getattr(batch, "answer_entity_ids"),
                answer_ptr=getattr(batch, "answer_ptr"),
            )
            (_, answer_sink_ids, answer_sink_log_rewards, _) = (
                build_node_answer_sink_tensors(
                    node_ptr=getattr(batch, "node_ptr"),
                    node_entity_ids=getattr(batch, "node_entity_ids"),
                    answer_entity_ids=getattr(batch, "answer_entity_ids"),
                    answer_ptr=getattr(batch, "answer_ptr"),
                    non_gold_terminal_log_reward=self.non_gold_terminal_log_reward,
                    gold_reward_mode=self.answer_quotient_gold_reward_mode,
                )
            )
        if isinstance(batch, TrajectoryBatch):
            answer_distance, shortest_path_edge_mask = _build_forward_guidance_tensors(
                batch
            )
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
            answer_mask=answer_mask,
            answer_sink_ids=answer_sink_ids,
            answer_sink_log_rewards=answer_sink_log_rewards,
            answer_distance=answer_distance,
            shortest_path_edge_mask=shortest_path_edge_mask,
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
        start_log_rewards = self.compute_start_log_rewards(
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
            start_log_rewards=start_log_rewards,
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
        start_state_log_flows = self.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_log_rewards = self.compute_start_log_rewards(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return start_state_log_flows + start_log_rewards

    def _resolve_answer_distance_values(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        node_abs_indices: torch.Tensor,
    ) -> torch.Tensor:
        if prepared_batch.answer_distance is None:
            return torch.full_like(node_abs_indices, fill_value=self.max_steps + 1)
        answer_distance = prepared_batch.answer_distance.index_select(
            0, node_abs_indices
        ).to(dtype=torch.long)
        unreachable_distance = self.answer_distance_potential_unreachable_distance
        if unreachable_distance is None:
            unreachable_distance = self.max_steps + 1
        return torch.where(
            answer_distance >= 0,
            answer_distance,
            torch.full_like(answer_distance, fill_value=int(unreachable_distance)),
        )

    def compute_node_log_potentials(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        node_abs_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.potential_reward_cfg.active):
            return torch.zeros_like(node_abs_indices, dtype=torch.float32)
        safe_distance = self._resolve_answer_distance_values(
            prepared_batch=prepared_batch,
            node_abs_indices=node_abs_indices,
        )
        return -float(self.answer_distance_potential_weight) * safe_distance.to(
            dtype=torch.float32
        )

    def compute_start_log_rewards(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        candidate_nodes_abs: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_node_log_potentials(
            prepared_batch=prepared_batch,
            node_abs_indices=candidate_nodes_abs,
        )

    def compute_move_log_reward_shaping(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        current_nodes: torch.Tensor,
        next_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.potential_reward_cfg.active):
            return torch.zeros_like(current_nodes, dtype=torch.float32)
        current_potential = self.compute_node_log_potentials(
            prepared_batch=prepared_batch,
            node_abs_indices=current_nodes,
        )
        next_potential = self.compute_node_log_potentials(
            prepared_batch=prepared_batch,
            node_abs_indices=next_nodes,
        )
        return next_potential - current_potential

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
        if candidate_nodes_abs is None or candidate_graph_ids is None:
            candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
                prepared_batch
            )
        start_state_log_flows = self.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_log_rewards = self.compute_start_log_rewards(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        root_log_z, has_values = segment_logsumexp_1d(
            values=(start_state_log_flows + start_log_rewards).to(dtype=torch.float32),
            segment_ids=candidate_graph_ids,
            num_segments=int(prepared_batch.topology.num_graphs),
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        if not bool(has_values.all().item()):
            invalid_graphs = (
                torch.nonzero(~has_values, as_tuple=False).view(-1).tolist()
            )
            invalid_samples = [
                prepared_batch.observation.sample_ids[idx] for idx in invalid_graphs
            ]
            raise InvalidStartCandidatesError(invalid_samples=invalid_samples)
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
        state_dtype = base_state_features.dtype
        step_ids = flat_num_steps.clamp(min=0, max=self.max_steps)
        step_features = self.step_embedding(step_ids).to(dtype=state_dtype)
        remaining_ids = (self.max_steps - flat_num_steps).clamp(
            min=0, max=self.max_steps
        )
        remaining_features = self.remaining_embedding(remaining_ids).to(
            dtype=state_dtype
        )
        state_inputs = torch.cat(
            (
                base_state_features,
                step_features,
                remaining_features,
                flat_control_states.to(dtype=state_dtype),
            ),
            dim=-1,
        )
        state_inputs = align_float_input_dtype(
            state_inputs, module=self.state_feature_input_norm
        )
        state_inputs = self.state_feature_input_norm(state_inputs)
        first_state_layer = self.state_feature_mlp[0]
        state_inputs = align_float_input_dtype(state_inputs, module=first_state_layer)
        state_features = self.state_feature_mlp(state_inputs)
        state_features = align_float_input_dtype(
            state_features, module=self.state_feature_norm
        )
        state_features = self.state_feature_norm(state_features).to(dtype=state_dtype)
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
        control_states = align_float_input_dtype(
            control_states, module=self.control_query
        )
        graph_ids = graph_ids.to(device=control_states.device, dtype=torch.long)
        context_query = self.control_query(control_states)
        context_summary = torch.zeros_like(context_query)
        query_scale = math.sqrt(float(prepared_batch.question_context_tokens.size(-1)))
        chunk_size = _control_attention_chunk_size(
            device=context_query.device,
            total_states=total_states,
        )
        for chunk_start in range(0, total_states, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_states)
            chunk_graph_ids = graph_ids[chunk_start:chunk_end]
            chunk_query = context_query[chunk_start:chunk_end]
            question_context = prepared_batch.question_context_tokens.index_select(
                0, chunk_graph_ids
            )
            question_mask = prepared_batch.question_context_mask.index_select(
                0, chunk_graph_ids
            )
            if bool((~question_mask.any(dim=1)).any().item()):
                raise ValueError(
                    "question_context_mask contains rows without valid tokens when updating control states."
                )
            attention_scores = torch.bmm(
                chunk_query.to(dtype=torch.float32).unsqueeze(1),
                question_context.to(dtype=torch.float32).transpose(1, 2),
            ).squeeze(1)
            attention_scores = attention_scores / query_scale
            attention_weights = masked_softmax_in_float32(
                attention_scores,
                mask=question_mask,
                dim=-1,
                output_dtype=context_summary.dtype,
            )
            context_summary[chunk_start:chunk_end] = torch.bmm(
                attention_weights.unsqueeze(1),
                question_context.to(dtype=attention_weights.dtype),
            ).squeeze(1)
        return context_summary

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
        control_dtype = attended_question.dtype
        update_inputs = torch.cat(
            (
                attended_question,
                relation_features.to(dtype=control_dtype),
                node_features.to(dtype=control_dtype),
            ),
            dim=-1,
        )
        update_inputs = align_float_input_dtype(
            update_inputs, module=self.control_input_norm
        )
        update_inputs = self.control_input_norm(update_inputs)
        update_inputs = align_float_input_dtype(
            update_inputs, module=self.control_update
        )
        previous_control_states = align_float_input_dtype(
            previous_control_states, module=self.control_update
        )
        next_control = self.control_update(
            self.control_dropout(update_inputs),
            previous_control_states,
        )
        next_control = align_float_input_dtype(next_control, module=self.control_norm)
        return self.control_norm(next_control).to(dtype=control_dtype)

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

    def _compute_stop_branch_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        current_nodes: torch.Tensor,
        state_features: torch.Tensor | None = None,
        graph_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total_candidates = int(current_nodes.numel())
        if total_candidates == 0:
            return torch.empty((0,), device=current_nodes.device, dtype=torch.float32)
        is_gold = prepared_batch.answer_mask.index_select(0, current_nodes).to(
            dtype=torch.bool
        )
        terminal_log_rewards = prepared_batch.answer_sink_log_rewards.index_select(
            0, current_nodes
        ).to(dtype=torch.float32)
        stop_step_rewards = is_gold.to(dtype=torch.float32) * float(
            self.answer_stop_log_reward_bonus
        )
        stop_logits = terminal_log_rewards + stop_step_rewards
        if not self.answer_quotient_allocate_stop_mass:
            return _mask_nonfinite_scores(stop_logits)
        if state_features is None or graph_ids is None:
            raise ValueError(
                "Answer-allocated STOP logits require state_features and graph_ids."
            )
        stop_allocation_scores = self._compute_stop_allocation_scores(state_features)
        sink_ids = prepared_batch.answer_sink_ids.index_select(0, current_nodes).to(
            dtype=torch.long
        )
        stop_allocation_log_probs = self._compute_sink_allocation_log_probs(
            allocation_scores=stop_allocation_scores,
            graph_ids=graph_ids,
            sink_ids=sink_ids,
        )
        return _mask_nonfinite_scores(stop_logits + stop_allocation_log_probs)

    def _compute_stop_allocation_scores(
        self,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        if int(state_features.numel()) == 0:
            return torch.empty((0,), device=state_features.device, dtype=torch.float32)
        allocation_inputs = align_float_input_dtype(
            state_features, module=self.stop_allocation_input_norm
        )
        allocation_inputs = self.stop_allocation_input_norm(allocation_inputs)
        allocation_inputs = align_float_input_dtype(
            allocation_inputs, module=self.stop_allocation_head
        )
        return _mask_nonfinite_scores(
            self.stop_allocation_head(allocation_inputs)
            .squeeze(-1)
            .to(dtype=torch.float32)
        )

    @staticmethod
    def _compute_sink_allocation_log_probs(
        *,
        allocation_scores: torch.Tensor,
        graph_ids: torch.Tensor,
        sink_ids: torch.Tensor,
    ) -> torch.Tensor:
        if int(allocation_scores.numel()) == 0:
            return torch.empty(
                (0,), device=allocation_scores.device, dtype=torch.float32
            )
        if tuple(allocation_scores.shape) != tuple(graph_ids.shape) or tuple(
            allocation_scores.shape
        ) != tuple(sink_ids.shape):
            raise ValueError(
                "STOP allocation inputs must share the same shape. "
                f"scores={tuple(allocation_scores.shape)} graph_ids={tuple(graph_ids.shape)} "
                f"sink_ids={tuple(sink_ids.shape)}."
            )
        key_base = int(sink_ids.max().item()) + 1
        sink_group_ids = graph_ids.to(dtype=torch.long) * key_base + sink_ids.to(
            dtype=torch.long
        )
        sink_lse, sink_has_values = segment_logsumexp_1d(
            values=allocation_scores.to(dtype=torch.float32),
            segment_ids=sink_group_ids,
            num_segments=int(sink_group_ids.max().item()) + 1,
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        allocation_log_probs = torch.zeros_like(allocation_scores, dtype=torch.float32)
        valid_groups = sink_has_values.index_select(0, sink_group_ids)
        if bool(valid_groups.any().item()):
            valid_indices = torch.nonzero(valid_groups, as_tuple=False).view(-1)
            allocation_log_probs[valid_indices] = allocation_scores.index_select(
                0, valid_indices
            ).to(dtype=torch.float32) - sink_lse.index_select(
                0, sink_group_ids.index_select(0, valid_indices)
            )
        return allocation_log_probs

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
        unique_current_nodes: torch.Tensor,
        unique_control_states: torch.Tensor,
        unique_graph_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_agent_batch: torch.Tensor,
        child_num_steps: torch.Tensor,
    ) -> torch.Tensor:
        if int(edge_ids.numel()) == 0:
            return torch.empty((0,), device=edge_ids.device, dtype=torch.float32)
        total_edges = int(edge_ids.numel())
        chunk_size = _candidate_chunk_size(
            device=edge_ids.device, total_candidates=total_edges
        )
        edge_logits_chunks: list[torch.Tensor] = []
        for start in range(0, total_edges, chunk_size):
            end = min(start + chunk_size, total_edges)
            chunk_edge_agent_batch = edge_agent_batch[start:end]
            chunk_edge_ids = edge_ids[start:end]
            chunk_target_nodes = target_nodes[start:end]
            chunk_child_num_steps = child_num_steps[start:end]
            chunk_relation_ids = prepared_batch.topology.edge_type.index_select(
                0, chunk_edge_ids
            )
            chunk_next_control_states = self.compute_next_control_states(
                prepared_batch,
                control_states=unique_control_states.index_select(
                    0, chunk_edge_agent_batch
                ),
                next_nodes=chunk_target_nodes,
                relation_ids=chunk_relation_ids,
            )
            chunk_child_state_features = self._build_flat_state_features(
                prepared_batch,
                flat_nodes=chunk_target_nodes,
                flat_num_steps=chunk_child_num_steps,
                flat_done_mask=torch.zeros_like(chunk_target_nodes, dtype=torch.bool),
                flat_control_states=chunk_next_control_states,
            )
            chunk_child_log_flows = self._compute_log_state_scores_from_flat_features(
                prepared_batch=prepared_batch,
                flat_state_features=chunk_child_state_features,
                graph_ids=unique_graph_ids.index_select(0, chunk_edge_agent_batch),
            )
            chunk_move_log_rewards = self._compute_move_action_log_rewards(
                prepared_batch=prepared_batch,
                current_nodes=unique_current_nodes.index_select(
                    0, chunk_edge_agent_batch
                ),
                next_nodes=chunk_target_nodes,
            )
            edge_logits_chunks.append(chunk_child_log_flows + chunk_move_log_rewards)
        return torch.cat(edge_logits_chunks, dim=0)

    def _compute_move_action_log_rewards(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        current_nodes: torch.Tensor,
        next_nodes: torch.Tensor,
    ) -> torch.Tensor:
        if int(current_nodes.numel()) == 0:
            return torch.empty((0,), device=current_nodes.device, dtype=torch.float32)
        move_log_rewards = torch.full(
            current_nodes.shape,
            fill_value=float(self.step_log_penalty),
            device=current_nodes.device,
            dtype=torch.float32,
        )
        move_log_rewards = move_log_rewards + self.compute_move_log_reward_shaping(
            prepared_batch=prepared_batch,
            current_nodes=current_nodes,
            next_nodes=next_nodes,
        )
        return _mask_nonfinite_scores(move_log_rewards)

    def compute_transition_proposal_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor:
        edge_logits = distribution.edge_logits.to(dtype=torch.float32)
        if self.transition_proposal_head is None or int(edge_logits.numel()) == 0:
            return torch.zeros_like(edge_logits)
        stop_mask = self._distribution_stop_mask(distribution)
        move_indices = torch.nonzero(~stop_mask, as_tuple=False).view(-1)
        if int(move_indices.numel()) == 0:
            return torch.zeros_like(edge_logits)
        move_agent_batch = distribution.edge_agent_batch.index_select(0, move_indices)
        flat_current_nodes = state.flatten_current_nodes()
        flat_num_steps = state.flatten_num_steps()
        flat_control_states = self._resolve_flat_control_states(prepared_batch, state)
        current_nodes = flat_current_nodes.index_select(0, move_agent_batch)
        current_num_steps = flat_num_steps.index_select(0, move_agent_batch)
        current_control_states = flat_control_states.index_select(0, move_agent_batch)
        current_state_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=current_nodes,
            flat_num_steps=current_num_steps,
            flat_done_mask=torch.zeros_like(current_nodes, dtype=torch.bool),
            flat_control_states=current_control_states,
        )
        edge_ids = distribution.edge_ids.index_select(0, move_indices)
        relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_ids)
        next_nodes = distribution.target_nodes.index_select(0, move_indices)
        next_control_states = self.compute_next_control_states(
            prepared_batch,
            control_states=current_control_states,
            next_nodes=next_nodes,
            relation_ids=relation_ids,
        )
        child_state_features = self._build_flat_state_features(
            prepared_batch,
            flat_nodes=next_nodes,
            flat_num_steps=current_num_steps + 1,
            flat_done_mask=torch.zeros_like(next_nodes, dtype=torch.bool),
            flat_control_states=next_control_states,
        )
        relation_features = prepared_batch.relation_tokens.index_select(0, relation_ids)
        transition_bias = self.transition_proposal_head(
            current_state_features=current_state_features,
            candidate_state_features=child_state_features,
            relation_features=relation_features,
        ).to(dtype=torch.float32)
        full_transition_bias = torch.zeros_like(edge_logits)
        full_transition_bias.index_copy_(
            0,
            move_indices,
            _mask_nonfinite_scores(transition_bias),
        )
        return full_transition_bias

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
        raw_graph_candidate_count = int(unique_edge_ids.numel())
        if int(unique_edge_ids.numel()) > 0:
            unique_child_num_steps = (
                unique_num_steps.index_select(0, unique_edge_agent_batch) + 1
            )
            legal_fresh_entity_mask = build_unique_forward_candidate_keep_mask(
                flat_current_abs_nodes=unique_current_nodes,
                flat_num_steps=unique_num_steps,
                flat_path_token_ids=unique_path_token_ids,
                node_entity_ids_by_abs_node=state.observation.node_entity_ids,
                num_nodes=int(state.topology.num_nodes),
                candidate_target_abs_nodes=unique_target_nodes,
                candidate_agent_indices=unique_edge_agent_batch,
                child_num_steps=unique_child_num_steps,
                max_steps=self.max_steps,
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
                unique_current_nodes=unique_current_nodes,
                unique_control_states=unique_control_states,
                unique_graph_ids=unique_graph_ids,
                edge_ids=unique_edge_ids,
                target_nodes=unique_target_nodes,
                edge_agent_batch=unique_edge_agent_batch,
                child_num_steps=unique_child_num_steps,
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
            unique_stop_action_logits = self._compute_stop_branch_logits(
                prepared_batch=prepared_batch,
                current_nodes=unique_current_nodes,
                state_features=unique_state_features,
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
            edge_logits=combined_edge_logits.to(dtype=torch.float32),
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
        return compute_policy_backward_distribution(
            prepared_batch=prepared_batch,
            state=state,
            max_steps=self.max_steps,
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
        action_prior_cfg: ActionPriorConfig,
        search_action_prior: SearchActionPrior,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self._action_prior_cfg = action_prior_cfg
        self.search_action_prior = search_action_prior

    @property
    def action_prior_cfg(self) -> ActionPriorConfig:
        return self._action_prior_cfg

    def prepare_batch(self, batch) -> PreparedGFlowNetBatch:
        prepared_batch = self.base_policy.prepare_batch(batch)
        with torch.no_grad():
            action_prior_cache = self.search_action_prior.build_cache(prepared_batch)
        return PreparedGFlowNetBatch(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            node_tokens=prepared_batch.node_tokens,
            relation_tokens=prepared_batch.relation_tokens,
            question_tokens=prepared_batch.question_tokens,
            question_context_tokens=prepared_batch.question_context_tokens,
            question_context_mask=prepared_batch.question_context_mask,
            answer_mask=prepared_batch.answer_mask,
            answer_sink_ids=prepared_batch.answer_sink_ids,
            answer_sink_log_rewards=prepared_batch.answer_sink_log_rewards,
            answer_distance=prepared_batch.answer_distance,
            shortest_path_edge_mask=prepared_batch.shortest_path_edge_mask,
            action_prior_cache=action_prior_cache,
        )

    def encode(self, batch) -> PreparedGFlowNetBatch:
        return self.prepare_batch(batch)

    def compute_graph_log_z(
        self, prepared_batch: PreparedGFlowNetBatch
    ) -> torch.Tensor:
        return self.base_policy.compute_graph_log_z(prepared_batch).to(
            dtype=torch.float32
        )

    def _compute_proposal_forward_logits(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
        action_prior_scale: float = 1.0,
    ) -> torch.Tensor:
        edge_logits = distribution.edge_logits.to(dtype=torch.float32)
        if int(edge_logits.numel()) == 0:
            return edge_logits
        transition_bias = self.base_policy.compute_transition_proposal_logits(
            prepared_batch=prepared_batch,
            state=state,
            distribution=distribution,
        )
        if int(transition_bias.numel()) > 0:
            edge_logits = _mask_nonfinite_scores(edge_logits + transition_bias)
        if not self.action_prior_cfg.enabled or float(action_prior_scale) <= 0.0:
            return edge_logits
        action_prior = self.search_action_prior.score_forward_actions(
            prepared_batch=prepared_batch,
            state=state,
            distribution=distribution,
            action_prior_scale=action_prior_scale,
        )
        return _mask_nonfinite_scores(edge_logits + action_prior)

    def compute_proposal_edge_logits(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
        *,
        action_prior_scale: float = 1.0,
    ) -> torch.Tensor:
        return self._compute_proposal_forward_logits(
            prepared_batch=prepared_batch,
            state=state,
            distribution=distribution,
            action_prior_scale=action_prior_scale,
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
        start_log_rewards = self.base_policy.compute_start_log_rewards(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        return build_root_action_distribution(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_logits=start_state_log_flows + start_log_rewards,
            start_state_log_flows=start_state_log_flows,
            start_log_rewards=start_log_rewards,
        )

    def compute_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> RootActionDistribution:
        return self.compute_root_action_distribution(prepared_batch)

    def compute_proposal_root_action_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        *,
        action_prior_scale: float = 1.0,
    ) -> RootActionDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        start_state_log_flows = self.base_policy.compute_start_log_flows(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_log_rewards = self.base_policy.compute_start_log_rewards(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
        )
        start_bias = self.search_action_prior.score_root_actions(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_prior_scale=action_prior_scale,
        )
        return build_root_action_distribution(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            action_logits=start_state_log_flows + start_log_rewards + start_bias,
            start_state_log_flows=start_state_log_flows,
            graph_log_z=self.base_policy.compute_graph_log_z(prepared_batch),
            start_log_rewards=start_log_rewards,
        )

    def compute_proposal_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        *,
        action_prior_scale: float = 1.0,
    ) -> RootActionDistribution:
        return self.compute_proposal_root_action_distribution(
            prepared_batch,
            action_prior_scale=action_prior_scale,
        )

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

    def compute_proposal_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        *,
        action_prior_scale: float = 1.0,
    ) -> ForwardActionDistribution:
        distribution = self.base_policy.compute_forward_distribution(
            prepared_batch,
            state,
        )
        return ForwardActionDistribution(
            edge_logits=self._compute_proposal_forward_logits(
                prepared_batch=prepared_batch,
                state=state,
                distribution=distribution,
                action_prior_scale=action_prior_scale,
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
    "build_root_action_distribution",
    "resolve_start_candidates",
]
