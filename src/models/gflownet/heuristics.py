from __future__ import annotations

import torch
from torch import nn

from src.models.configs import ActionPriorConfig

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
    log_prior = torch.full(
        (topology.num_nodes,), fill_value=float("-inf"), device=device
    )
    q_abs, q_graph_ids = topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )
    graph_offsets = topology.graph_node_offsets.to(device=device)
    edge_ids = torch.arange(
        int(topology.edge_index.size(1)), device=device, dtype=torch.long
    )
    edge_graph_ids = topology.graph_index_from_edges(edge_ids)
    for graph_idx in range(int(topology.num_graphs)):
        node_start = int(graph_offsets[graph_idx].item())
        node_end = int(graph_offsets[graph_idx + 1].item())
        num_nodes = node_end - node_start
        if num_nodes <= 0:
            continue
        seed_nodes = q_abs[q_graph_ids == graph_idx] - node_start
        if int(seed_nodes.numel()) == 0:
            continue
        seeds = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
        seeds.scatter_(0, seed_nodes, 1.0 / float(seed_nodes.numel()))
        mass = seeds.clone()
        edge_mask = edge_graph_ids == graph_idx
        local_edge_index = topology.edge_index[:, edge_mask] - node_start
        if int(local_edge_index.numel()) == 0:
            log_prior[node_start:node_end] = (seeds + float(eps)).log()
            continue
        src = local_edge_index[0]
        dst = local_edge_index[1]
        out_degree = torch.bincount(src, minlength=num_nodes).to(dtype=torch.float32)
        norm = out_degree.index_select(0, src).clamp_min(1.0)
        for _ in range(int(num_iters)):
            propagated = torch.zeros_like(mass)
            propagated.scatter_add_(0, dst, mass.index_select(0, src) / norm)
            mass = (
                float(restart_prob) * seeds + (1.0 - float(restart_prob)) * propagated
            )
        log_prior[node_start:node_end] = (mass + float(eps)).log()
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
    centered = values.to(dtype=torch.float32).clone()
    if int(centered.numel()) == 0:
        return centered
    for group_idx in range(int(num_groups)):
        group_mask = group_ids == group_idx
        if not bool(group_mask.any().item()):
            continue
        group_values = centered[group_mask]
        centered[group_mask] = group_values - group_values.mean()
    return centered


def _group_standardize(
    *,
    values: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    standardized = values.to(dtype=torch.float32).clone()
    if int(standardized.numel()) == 0:
        return standardized
    for group_idx in range(int(num_groups)):
        group_mask = group_ids == group_idx
        if not bool(group_mask.any().item()):
            continue
        group_values = standardized[group_mask]
        centered = group_values - group_values.mean()
        scale = centered.square().mean().sqrt().clamp_min(float(eps))
        standardized[group_mask] = centered / scale
    return standardized


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

    def _build_node_prior(
        self,
        prepared_batch: PreparedSearchBatch,
    ) -> torch.Tensor | None:
        if self.config.kind == "none":
            return None
        topology = prepared_batch.topology
        num_graphs = int(topology.num_graphs)
        node_graph_ids = topology.all_node_graph_index(
            device=prepared_batch.node_tokens.device
        )
        components: list[torch.Tensor] = []
        weights: list[float] = []
        if (
            self.config.kind in {"topology", "hybrid"}
            and self.config.node_topology_weight > 0.0
        ):
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
        if (
            self.config.kind in {"embedding", "hybrid"}
            and self.config.node_embedding_weight > 0.0
        ):
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
            return ActionPriorCache(node_prior=None)
        return ActionPriorCache(node_prior=self._build_node_prior(prepared_batch))

    @staticmethod
    def _lookup_node_prior(
        *,
        action_prior_cache: ActionPriorCache,
        node_abs_indices: torch.Tensor,
    ) -> torch.Tensor:
        if action_prior_cache.node_prior is None:
            return torch.zeros_like(node_abs_indices, dtype=torch.float32)
        return action_prior_cache.node_prior.index_select(0, node_abs_indices)

    def compute_cached_bias(
        self,
        *,
        heuristic_cache: ActionPriorCache,
        node_abs_indices: torch.Tensor,
        num_steps: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        del num_steps
        node_prior = self._lookup_node_prior(
            action_prior_cache=heuristic_cache,
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
    ) -> torch.Tensor:
        if float(self.config.root_beta or 0.0) == 0.0:
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
        return float(self.config.root_beta or 0.0) * node_prior

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

    def score_forward_actions(
        self,
        *,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
        distribution: ForwardActionDistribution,
    ) -> torch.Tensor:
        edge_logits = distribution.edge_logits.to(dtype=torch.float32)
        if int(edge_logits.numel()) == 0:
            return edge_logits
        total_agents = int(state.current_nodes.numel())
        stop_action_mask = (
            distribution.is_stop_action.to(dtype=torch.bool)
            if distribution.is_stop_action is not None
            else torch.zeros_like(distribution.edge_ids, dtype=torch.bool)
        )
        action_prior = torch.zeros_like(edge_logits, dtype=torch.float32)
        graph_action_mask = ~stop_action_mask
        if (
            bool(graph_action_mask.any().item())
            and float(self.config.edge_beta or 0.0) > 0.0
        ):
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
            edge_prior = (
                float(self.config.relation_embedding_weight) * relation_prior
                + float(self.config.target_node_weight) * target_prior
                + float(self.config.progress_weight) * (target_prior - current_prior)
            )
            action_prior[graph_action_mask] = (
                float(self.config.edge_beta or 0.0) * edge_prior
            )
        if bool(stop_action_mask.any().item()) and float(self.config.stop_beta) > 0.0:
            stop_nodes = distribution.target_nodes[stop_action_mask]
            stop_prior = self._lookup_node_prior(
                action_prior_cache=prepared_batch.action_prior_cache,
                node_abs_indices=stop_nodes,
            )
            action_prior[stop_action_mask] = (
                float(self.config.stop_beta)
                * float(self.config.stop_node_weight)
                * stop_prior
            )
        return _group_center(
            values=action_prior,
            group_ids=distribution.edge_agent_batch,
            num_groups=total_agents,
        )


# Backward-compatible aliases for older names.
SearchHeuristic = SearchActionPrior
compute_topology_log_heuristic = compute_topology_node_prior
compute_embedding_log_heuristic = compute_question_node_prior


__all__ = [
    "SearchActionPrior",
    "SearchHeuristic",
    "compute_embedding_log_heuristic",
    "compute_question_node_prior",
    "compute_question_relation_prior",
    "compute_topology_log_heuristic",
    "compute_topology_node_prior",
]
