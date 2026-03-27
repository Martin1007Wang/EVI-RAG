from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.configs import ActionPriorConfig
from src.utils.segment_ops import segment_mean_1d

from .types import ActionPriorCache, ForwardActionDistribution, PreparedGFlowNetBatch
from .types import PreparedSearchBatch, SearchState


def compute_topology_node_prior(
    *,
    topology,
    observation,
    restart_prob: float,
    num_iters: int,
    eps: float,
) -> torch.Tensor:
    device = topology.edge_index.device
    num_nodes_total = int(topology.num_nodes)
    log_prior = torch.full((num_nodes_total,), fill_value=float("-inf"), device=device)
    q_abs, q_graph_ids = topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )
    if int(q_abs.numel()) == 0:
        return log_prior.to(dtype=torch.float32)

    num_graphs = int(topology.num_graphs)
    graph_ids = topology.all_node_graph_index(device=device)
    seed_counts = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
    seed_counts.scatter_add_(
        0, q_graph_ids, torch.ones_like(q_graph_ids, dtype=torch.float32)
    )
    seeded_graphs = seed_counts > 0

    seeds = torch.zeros((num_nodes_total,), device=device, dtype=torch.float32)
    per_seed_mass = seed_counts.index_select(0, q_graph_ids).reciprocal()
    seeds.scatter_(0, q_abs, per_seed_mass)

    edge_ids = torch.arange(
        int(topology.edge_index.size(1)), device=device, dtype=torch.long
    )
    edge_graph_ids = topology.graph_index_from_edges(edge_ids)
    graph_has_edges = torch.zeros((num_graphs,), device=device, dtype=torch.bool)
    if int(edge_graph_ids.numel()) > 0:
        graph_has_edges.scatter_(0, edge_graph_ids, True)
    seeded_without_edges = seeded_graphs & (~graph_has_edges)
    if bool(seeded_without_edges.any().item()):
        no_edge_nodes = seeded_without_edges.index_select(0, graph_ids)
        log_prior[no_edge_nodes] = (seeds[no_edge_nodes] + float(eps)).log()

    seeded_edge_mask = seeded_graphs.index_select(0, edge_graph_ids)
    if not bool(seeded_edge_mask.any().item()):
        return log_prior.to(dtype=torch.float32)

    src = topology.edge_index[0].index_select(
        0, seeded_edge_mask.nonzero(as_tuple=False).view(-1)
    )
    dst = topology.edge_index[1].index_select(
        0, seeded_edge_mask.nonzero(as_tuple=False).view(-1)
    )
    active_graphs = seeded_graphs & graph_has_edges
    active_nodes = active_graphs.index_select(0, graph_ids)
    mass = seeds.clone()
    out_degree = torch.zeros((num_nodes_total,), device=device, dtype=torch.float32)
    out_degree.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
    norm = out_degree.index_select(0, src).clamp_min(1.0)
    restart_mass = float(restart_prob)
    continue_mass = 1.0 - restart_mass
    for _ in range(int(num_iters)):
        propagated = torch.zeros_like(mass)
        propagated.scatter_add_(0, dst, mass.index_select(0, src) / norm)
        mass = restart_mass * seeds + continue_mass * propagated
    log_prior[active_nodes] = (mass[active_nodes] + float(eps)).log()
    return log_prior.to(dtype=torch.float32)


def compute_question_node_prior(
    *,
    topology,
    node_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    graph_ids = topology.all_node_graph_index(device=node_tokens.device)
    question = question_tokens.index_select(0, graph_ids)
    cosine = torch.nn.functional.cosine_similarity(node_tokens, question, dim=-1)
    return cosine.to(dtype=torch.float32) / float(temperature)


def compute_question_relation_prior(
    *,
    relation_features: torch.Tensor,
    question_features: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if tuple(relation_features.shape[:-1]) != tuple(question_features.shape[:-1]):
        raise ValueError(
            "question_features must match relation_features batch shape when scoring relation priors."
        )
    cosine = torch.nn.functional.cosine_similarity(
        relation_features,
        question_features,
        dim=-1,
    )
    return cosine.to(dtype=torch.float32) / float(temperature)


def _group_center(
    *,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    centered = values.to(dtype=torch.float32)
    if int(centered.numel()) == 0:
        return centered.clone()
    group_ids = group_ids.to(device=centered.device, dtype=torch.long)
    group_mean = segment_mean_1d(
        values=centered,
        segment_ids=group_ids,
        num_segments=int(num_groups),
        dtype=torch.float32,
    )
    return centered - group_mean.index_select(0, group_ids)


def _group_standardize(
    *,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    standardized = values.to(dtype=torch.float32)
    if int(standardized.numel()) == 0:
        return standardized.clone()
    group_ids = group_ids.to(device=standardized.device, dtype=torch.long)
    group_mean = segment_mean_1d(
        values=standardized,
        segment_ids=group_ids,
        num_segments=int(num_groups),
        dtype=torch.float32,
    )
    centered = standardized - group_mean.index_select(0, group_ids)
    squared = centered.square()
    group_var = segment_mean_1d(
        values=squared,
        segment_ids=group_ids,
        num_segments=int(num_groups),
        dtype=torch.float32,
    )
    group_scale = group_var.sqrt().clamp_min(float(eps))
    return centered / group_scale.index_select(0, group_ids)


class SearchActionPrior(nn.Module):
    def __init__(self, *, config: ActionPriorConfig) -> None:
        super().__init__()
        self._config = config

    @property
    def config(self) -> ActionPriorConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @staticmethod
    def _resolve_strength_scale(*, action_prior_scale: float) -> float:
        return max(float(action_prior_scale), 0.0)

    def _build_node_prior(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> torch.Tensor | None:
        topology = prepared_batch.topology
        num_graphs = int(topology.num_graphs)
        node_graph_ids = topology.all_node_graph_index(
            device=prepared_batch.node_tokens.device
        )
        components: list[torch.Tensor] = []
        weights: list[float] = []
        if self.config.node_topology_weight > 0.0:
            topology_prior = compute_topology_node_prior(
                topology=topology,
                observation=prepared_batch.observation,
                restart_prob=float(self.config.topology_restart_prob),
                num_iters=int(self.config.topology_num_iters),
                eps=float(self.config.topology_eps),
            )
            components.append(
                _group_standardize(
                    values=topology_prior,
                    group_ids=node_graph_ids,
                    num_groups=num_graphs,
                )
            )
            weights.append(float(self.config.node_topology_weight))
        if self.config.node_embedding_weight > 0.0:
            embedding_prior = compute_question_node_prior(
                topology=topology,
                node_tokens=prepared_batch.node_tokens,
                question_tokens=prepared_batch.question_tokens,
                temperature=float(self.config.embedding_temperature),
            )
            components.append(
                _group_standardize(
                    values=embedding_prior,
                    group_ids=node_graph_ids,
                    num_groups=num_graphs,
                )
            )
            weights.append(float(self.config.node_embedding_weight))
        if not components:
            return None
        node_prior = torch.zeros_like(components[0], dtype=torch.float32)
        for component, weight in zip(components, weights):
            node_prior = node_prior + float(weight) * component
        return node_prior

    def build_cache(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> ActionPriorCache:
        if not self.enabled:
            return ActionPriorCache(
                node_prior=None,
                answer_distance=prepared_batch.answer_distance,
                shortest_path_edge_mask=prepared_batch.shortest_path_edge_mask,
            )
        return ActionPriorCache(
            node_prior=self._build_node_prior(prepared_batch),
            answer_distance=prepared_batch.answer_distance,
            shortest_path_edge_mask=prepared_batch.shortest_path_edge_mask,
        )

    @staticmethod
    def _lookup_node_prior(
        *,
        action_prior_cache: ActionPriorCache,
        node_abs_indices: torch.Tensor,
    ) -> torch.Tensor:
        if action_prior_cache.node_prior is None:
            return torch.zeros_like(node_abs_indices, dtype=torch.float32)
        return action_prior_cache.node_prior.index_select(0, node_abs_indices)

    @staticmethod
    def _lookup_answer_distance(
        *,
        action_prior_cache: ActionPriorCache,
        node_abs_indices: torch.Tensor,
    ) -> torch.Tensor:
        if action_prior_cache.answer_distance is None:
            return torch.full_like(node_abs_indices, fill_value=-1, dtype=torch.long)
        return action_prior_cache.answer_distance.index_select(0, node_abs_indices).to(
            dtype=torch.long
        )

    @staticmethod
    def _lookup_shortest_path_edge_mask(
        *,
        action_prior_cache: ActionPriorCache,
        edge_ids: torch.Tensor,
    ) -> torch.Tensor:
        if action_prior_cache.shortest_path_edge_mask is None:
            return torch.zeros_like(edge_ids, dtype=torch.float32)
        return action_prior_cache.shortest_path_edge_mask.index_select(0, edge_ids).to(
            dtype=torch.float32
        )

    def compute_cached_bias(
        self,
        *,
        action_prior_cache: ActionPriorCache,
        node_abs_indices: torch.Tensor,
        num_steps: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        del num_steps
        node_prior = self._lookup_node_prior(
            action_prior_cache=action_prior_cache,
            node_abs_indices=node_abs_indices,
        )
        return torch.where(
            done_mask.to(dtype=torch.bool),
            torch.zeros_like(node_prior),
            node_prior,
        )

    def score_root_actions(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        candidate_nodes_abs: torch.Tensor,
        candidate_graph_ids: torch.Tensor,
        action_prior_scale: float = 1.0,
    ) -> torch.Tensor:
        strength_scale = self._resolve_strength_scale(
            action_prior_scale=action_prior_scale
        )
        root_beta = float(self.config.root_beta or 0.0) * strength_scale
        if root_beta == 0.0:
            return torch.zeros_like(candidate_nodes_abs, dtype=torch.float32)
        node_prior = self._lookup_node_prior(
            action_prior_cache=prepared_batch.action_prior_cache,
            node_abs_indices=candidate_nodes_abs,
        )
        node_prior = _group_center(
            values=node_prior,
            group_ids=candidate_graph_ids,
            num_groups=int(prepared_batch.topology.num_graphs),
        )
        return root_beta * node_prior

    def _score_relation_prior(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        relation_ids: torch.Tensor,
        graph_ids: torch.Tensor,
        action_group_ids: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        if float(self.config.relation_embedding_weight) == 0.0:
            return torch.zeros_like(relation_ids, dtype=torch.float32)
        relation_features = prepared_batch.relation_tokens.index_select(0, relation_ids)
        question_features = prepared_batch.question_tokens.index_select(0, graph_ids)
        relation_prior = compute_question_relation_prior(
            relation_features=relation_features,
            question_features=question_features,
            temperature=float(self.config.embedding_temperature),
        )
        return _group_standardize(
            values=relation_prior,
            group_ids=action_group_ids,
            num_groups=num_groups,
        )

    def _score_intent_action_prior(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        relation_ids: torch.Tensor,
        target_nodes: torch.Tensor,
        action_group_ids: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        if float(self.config.intent_alignment_weight) == 0.0:
            return torch.zeros_like(relation_ids, dtype=torch.float32)
        if state.control_state is None:
            return torch.zeros_like(relation_ids, dtype=torch.float32)
        relation_weight = float(self.config.intent_relation_weight)
        target_weight = float(self.config.intent_target_weight)
        if relation_weight == 0.0 and target_weight == 0.0:
            return torch.zeros_like(relation_ids, dtype=torch.float32)

        relation_features = prepared_batch.relation_tokens.index_select(0, relation_ids)
        target_features = prepared_batch.node_tokens.index_select(0, target_nodes)
        action_features = relation_weight * relation_features.to(
            dtype=torch.float32
        ) + target_weight * target_features.to(dtype=torch.float32)
        control_states = state.flatten_control_state().index_select(0, action_group_ids)
        intent_similarity = F.cosine_similarity(
            action_features,
            control_states.to(dtype=action_features.dtype),
            dim=-1,
            eps=1.0e-8,
        )
        return _group_standardize(
            values=intent_similarity,
            group_ids=action_group_ids,
            num_groups=num_groups,
        )

    def score_forward_actions(
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
        strength_scale = self._resolve_strength_scale(
            action_prior_scale=action_prior_scale
        )
        total_agents = int(state.current_nodes.numel())
        stop_action_mask = (
            distribution.is_stop_action.to(dtype=torch.bool)
            if distribution.is_stop_action is not None
            else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
        )
        action_prior = torch.zeros_like(edge_logits, dtype=torch.float32)
        graph_action_mask = ~stop_action_mask
        edge_beta = float(self.config.edge_beta or 0.0) * strength_scale
        if bool(graph_action_mask.any().item()) and edge_beta > 0.0:
            graph_edge_ids = distribution.edge_ids[graph_action_mask]
            graph_target_nodes = distribution.target_nodes[graph_action_mask]
            graph_action_groups = distribution.edge_agent_batch[graph_action_mask]
            current_nodes = state.flatten_current_nodes().index_select(
                0, graph_action_groups
            )
            graph_ids = prepared_batch.topology.graph_index_from_nodes(current_nodes)
            relation_ids = prepared_batch.topology.edge_type.index_select(
                0, graph_edge_ids
            )
            target_prior = self._lookup_node_prior(
                action_prior_cache=prepared_batch.action_prior_cache,
                node_abs_indices=graph_target_nodes,
            )
            current_prior = self._lookup_node_prior(
                action_prior_cache=prepared_batch.action_prior_cache,
                node_abs_indices=current_nodes,
            )
            relation_prior = self._score_relation_prior(
                prepared_batch=prepared_batch,
                relation_ids=relation_ids,
                graph_ids=graph_ids,
                action_group_ids=graph_action_groups,
                num_groups=total_agents,
            )
            intent_prior = self._score_intent_action_prior(
                prepared_batch=prepared_batch,
                state=state,
                relation_ids=relation_ids,
                target_nodes=graph_target_nodes,
                action_group_ids=graph_action_groups,
                num_groups=total_agents,
            )
            edge_prior = (
                float(self.config.relation_embedding_weight) * relation_prior
                + float(self.config.target_node_weight) * target_prior
                + float(self.config.progress_weight) * (target_prior - current_prior)
                + float(self.config.intent_alignment_weight) * intent_prior
            )
            shortest_path_weight = float(self.config.shortest_path_edge_weight)
            if shortest_path_weight > 0.0:
                shortest_path_bonus = self._lookup_shortest_path_edge_mask(
                    action_prior_cache=prepared_batch.action_prior_cache,
                    edge_ids=graph_edge_ids,
                )
                edge_prior = edge_prior + shortest_path_weight * _group_standardize(
                    values=shortest_path_bonus,
                    group_ids=graph_action_groups,
                    num_groups=total_agents,
                )
            answer_distance_weight = float(self.config.answer_distance_weight)
            if answer_distance_weight > 0.0:
                current_distance = self._lookup_answer_distance(
                    action_prior_cache=prepared_batch.action_prior_cache,
                    node_abs_indices=current_nodes,
                )
                target_distance = self._lookup_answer_distance(
                    action_prior_cache=prepared_batch.action_prior_cache,
                    node_abs_indices=graph_target_nodes,
                )
                valid_distance = (current_distance >= 0) & (target_distance >= 0)
                distance_progress = torch.zeros_like(edge_prior, dtype=torch.float32)
                if bool(valid_distance.any().item()):
                    progress = (
                        current_distance[valid_distance]
                        - target_distance[valid_distance]
                    ).to(dtype=torch.float32)
                    distance_progress[valid_distance] = progress.clamp_min(0.0)
                edge_prior = edge_prior + answer_distance_weight * _group_standardize(
                    values=distance_progress,
                    group_ids=graph_action_groups,
                    num_groups=total_agents,
                )
            action_prior[graph_action_mask] = edge_beta * edge_prior
        stop_beta = float(self.config.stop_beta) * strength_scale
        if bool(stop_action_mask.any().item()) and stop_beta > 0.0:
            stop_nodes = distribution.target_nodes[stop_action_mask]
            stop_prior = self._lookup_node_prior(
                action_prior_cache=prepared_batch.action_prior_cache,
                node_abs_indices=stop_nodes,
            )
            action_prior[stop_action_mask] = (
                stop_beta * float(self.config.stop_node_weight) * stop_prior
            )
        return _group_center(
            values=action_prior,
            group_ids=distribution.edge_agent_batch,
            num_groups=total_agents,
        )


compute_topology_log_heuristic = compute_topology_node_prior
compute_embedding_log_heuristic = compute_question_node_prior


__all__ = [
    "SearchActionPrior",
    "compute_embedding_log_heuristic",
    "compute_question_node_prior",
    "compute_question_relation_prior",
    "compute_topology_log_heuristic",
    "compute_topology_node_prior",
]
