from __future__ import annotations

from typing import Callable, Protocol

import torch
from torch import nn

from src.graph import GraphTopology, SearchObservation
from src.models.components.heuristic_heads import LearnedHeuristicHead
from src.models.configs import HeuristicConfig

from .types import (
    ForwardActionDistribution,
    HeuristicCache,
    PreparedSearchBatch,
    SearchState,
)


def compute_topology_log_heuristic(
    *,
    topology: GraphTopology,
    observation: SearchObservation,
    restart_prob: float,
    num_iters: int,
    eps: float,
) -> torch.Tensor:
    device = topology.edge_index.device
    log_heuristic = torch.full(
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
            log_heuristic[node_start:node_end] = (seeds + float(eps)).log()
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
        log_heuristic[node_start:node_end] = (mass + float(eps)).log()
    return log_heuristic


def compute_embedding_log_heuristic(
    *,
    topology: GraphTopology,
    node_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    graph_ids = topology.all_node_graph_index(device=node_tokens.device)
    question = question_tokens.index_select(0, graph_ids)
    cosine = torch.nn.functional.cosine_similarity(node_tokens, question, dim=-1)
    return cosine.to(dtype=torch.float32) / float(temperature)


StateFeatureBuilder = Callable[[PreparedSearchBatch, SearchState], torch.Tensor]


class LocalStateFeatureBuilder(Protocol):
    def __call__(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        flat_nodes: torch.Tensor,
        flat_num_steps: torch.Tensor,
        flat_done_mask: torch.Tensor,
    ) -> torch.Tensor: ...


_LEARNED_PROPOSAL_CACHE_CHUNK_SIZE = 4096


class SearchHeuristic(nn.Module):
    def __init__(
        self,
        *,
        config: HeuristicConfig,
        learned_head: LearnedHeuristicHead | None = None,
    ) -> None:
        super().__init__()
        self._config = config
        self.learned_head = learned_head
        if self.uses_learned_head and self.learned_head is None:
            raise ValueError("learned heuristic requires a LearnedHeuristicHead.")
        if not self.uses_learned_head and self.learned_head is not None:
            raise ValueError(
                "learned_head should only be provided when heuristic.kind is 'learned'."
            )

    @property
    def config(self) -> HeuristicConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self.config.kind != "none" and float(self.config.beta) > 0.0

    @property
    def uses_learned_head(self) -> bool:
        return self.config.kind == "learned"

    def build_cache(
        self,
        prepared_batch: PreparedSearchBatch,
        *,
        build_local_state_features: LocalStateFeatureBuilder | None = None,
        max_steps: int | None = None,
    ) -> HeuristicCache:
        # Behavior sampling only needs a cheap proposal prior, so we cache
        # per-node or per-(step, node) scores once per prepared batch.
        if not self.enabled:
            return HeuristicCache(node_log_heuristic=None, step_node_log_heuristic=None)
        if self.config.kind == "topology":
            return HeuristicCache(
                node_log_heuristic=compute_topology_log_heuristic(
                    topology=prepared_batch.topology,
                    observation=prepared_batch.observation,
                    restart_prob=float(self.config.topology_restart_prob),
                    num_iters=int(self.config.topology_num_iters),
                    eps=float(self.config.topology_eps),
                )
            )
        if self.config.kind == "embedding":
            return HeuristicCache(
                node_log_heuristic=compute_embedding_log_heuristic(
                    topology=prepared_batch.topology,
                    node_tokens=prepared_batch.node_tokens,
                    question_tokens=prepared_batch.question_tokens,
                    temperature=float(self.config.embedding_temperature),
                )
            )
        if build_local_state_features is None or max_steps is None:
            raise ValueError(
                "learned heuristic cache requires build_local_state_features and max_steps."
            )
        return HeuristicCache(
            node_log_heuristic=None,
            step_node_log_heuristic=self._build_step_node_log_heuristic(
                prepared_batch=prepared_batch,
                build_local_state_features=build_local_state_features,
                max_steps=int(max_steps),
            ),
        )

    @staticmethod
    def _node_log_heuristic(
        *, node_abs_indices: torch.Tensor, cache: HeuristicCache
    ) -> torch.Tensor:
        if cache.node_log_heuristic is None:
            return torch.zeros_like(node_abs_indices, dtype=torch.float32)
        return cache.node_log_heuristic.index_select(0, node_abs_indices)

    def _build_step_node_log_heuristic(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        build_local_state_features: LocalStateFeatureBuilder,
        max_steps: int,
    ) -> torch.Tensor:
        if self.learned_head is None:
            raise RuntimeError(
                "learned heuristic cache requires a LearnedHeuristicHead."
            )
        num_nodes = int(prepared_batch.topology.num_nodes)
        device = prepared_batch.node_tokens.device
        node_abs_indices = torch.arange(num_nodes, device=device, dtype=torch.long)
        node_graph_ids = prepared_batch.topology.all_node_graph_index(device=device)
        cache = torch.empty(
            (max_steps + 1, num_nodes),
            device=device,
            dtype=torch.float32,
        )
        if num_nodes == 0:
            return cache
        chunk_size = num_nodes
        if device.type == "cuda":
            chunk_size = min(num_nodes, _LEARNED_PROPOSAL_CACHE_CHUNK_SIZE)
        for step in range(max_steps + 1):
            chunk_logits: list[torch.Tensor] = []
            for start in range(0, num_nodes, max(chunk_size, 1)):
                end = min(start + max(chunk_size, 1), num_nodes)
                chunk_nodes = node_abs_indices[start:end]
                chunk_steps = torch.full_like(chunk_nodes, fill_value=step)
                chunk_done = torch.zeros_like(chunk_nodes, dtype=torch.bool)
                local_state_features = build_local_state_features(
                    prepared_batch,
                    flat_nodes=chunk_nodes,
                    flat_num_steps=chunk_steps,
                    flat_done_mask=chunk_done,
                )
                question_features = prepared_batch.question_tokens.index_select(
                    0, node_graph_ids[start:end]
                )
                chunk_logits.append(
                    self._state_log_heuristic(
                        state_features=local_state_features,
                        question_features=question_features,
                    )
                )
            cache[step] = torch.cat(chunk_logits, dim=0)
        return cache

    def compute_cached_bias(
        self,
        *,
        heuristic_cache: HeuristicCache,
        node_abs_indices: torch.Tensor,
        num_steps: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        safe_steps = num_steps.to(dtype=torch.long)
        if heuristic_cache.step_node_log_heuristic is not None:
            safe_node_abs_indices = node_abs_indices.to(dtype=torch.long).clamp(
                min=0,
                max=max(int(heuristic_cache.step_node_log_heuristic.size(1)) - 1, 0),
            )
            max_step = int(heuristic_cache.step_node_log_heuristic.size(0)) - 1
            safe_steps = safe_steps.clamp(min=0, max=max(max_step, 0))
            bias = heuristic_cache.step_node_log_heuristic[
                safe_steps, safe_node_abs_indices
            ]
        else:
            bias = self._node_log_heuristic(
                node_abs_indices=node_abs_indices,
                cache=heuristic_cache,
            )
        return torch.where(
            done_mask.to(dtype=torch.bool),
            torch.zeros_like(bias),
            bias.to(dtype=torch.float32),
        )

    def _state_log_heuristic(
        self,
        *,
        state_features: torch.Tensor,
        question_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.learned_head is None:
            return torch.zeros(
                (int(state_features.size(0)),),
                device=state_features.device,
                dtype=torch.float32,
            )
        logits = self.learned_head(
            state_features=state_features,
            question_features=question_features,
        )
        return torch.nn.functional.logsigmoid(logits)

    def compute_state_logits(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        state: SearchState,
        build_state_features: StateFeatureBuilder,
        detach_features: bool = False,
    ) -> torch.Tensor:
        if not self.uses_learned_head:
            return torch.zeros(
                state.current_nodes.shape,
                device=state.current_nodes.device,
                dtype=torch.float32,
            )
        state_features = build_state_features(prepared_batch, state).view(
            int(state.current_nodes.numel()), -1
        )
        question_features = prepared_batch.question_tokens.index_select(
            0, state.flatten_graph_index()
        )
        if detach_features:
            state_features = state_features.detach()
            question_features = question_features.detach()
        if self.learned_head is None:
            raise RuntimeError(
                "learned heuristic logits require a LearnedHeuristicHead."
            )
        logits = self.learned_head(
            state_features=state_features,
            question_features=question_features,
        ).view_as(state.current_nodes)
        return torch.where(state.done_mask, torch.zeros_like(logits), logits)

    def compute_state_bias(
        self,
        *,
        prepared_batch: PreparedSearchBatch,
        heuristic_cache: HeuristicCache,
        state: SearchState,
        build_state_features: StateFeatureBuilder,
    ) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros(
                state.current_nodes.shape,
                device=state.current_nodes.device,
                dtype=torch.float32,
            )
        flat_nodes = state.flatten_current_nodes()
        if self.uses_learned_head:
            flat_bias = torch.nn.functional.logsigmoid(
                self.compute_state_logits(
                    prepared_batch=prepared_batch,
                    state=state,
                    build_state_features=build_state_features,
                ).view(-1)
            )
        else:
            flat_bias = self._node_log_heuristic(
                node_ids=flat_nodes,
                cache=heuristic_cache,
            )
        bias = flat_bias.view_as(state.current_nodes)
        return torch.where(state.done_mask, torch.zeros_like(bias), bias)


__all__ = [
    "SearchHeuristic",
    "StateFeatureBuilder",
    "compute_embedding_log_heuristic",
    "compute_topology_log_heuristic",
]
