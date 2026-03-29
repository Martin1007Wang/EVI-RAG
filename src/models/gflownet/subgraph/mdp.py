from __future__ import annotations

from collections import deque
import math
from typing import TYPE_CHECKING

import torch

from src.graph import TrajectoryBatch
from src.models.components import EmbeddingBackbone
from src.models.configs.gflownet_training import SubgraphRewardConfig

from .answers import resolve_subgraph_answer_entities
from .prepared_batch import (
    UNREACHABLE_DISTANCE,
    SubgraphPreparedBatch,
    build_subgraph_prepared_batch,
)
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch, SubgraphState

if TYPE_CHECKING:
    from src.graph import GraphTopology


def _sorted_edge_records(
    *,
    topology: GraphTopology,
    edge_ids: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], ...]:
    records: list[tuple[int, int, int, int]] = []
    for edge_id in edge_ids:
        edge_idx = int(edge_id)
        records.append(
            (
                edge_idx,
                int(topology.edge_index[0, edge_idx].item()),
                int(topology.edge_type[edge_idx].item()),
                int(topology.edge_index[1, edge_idx].item()),
            )
        )
    records.sort(key=lambda item: item[0])
    return tuple(records)


class SubgraphEnv:
    def __init__(self, *, max_steps: int, reward_cfg: SubgraphRewardConfig) -> None:
        self.max_steps = int(max_steps)
        self.reward_cfg = reward_cfg

    def prepare_batch(
        self,
        *,
        batch: TrajectoryBatch,
        backbone: EmbeddingBackbone,
    ) -> SubgraphPreparedBatch:
        return build_subgraph_prepared_batch(batch=batch, backbone=backbone)

    def initial_state(self) -> SubgraphState:
        return SubgraphState()

    def initialize_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        num_rollouts: int,
    ) -> SubgraphRolloutBatch:
        if int(num_rollouts) < 1:
            raise ValueError(
                "num_rollouts must be >= 1 for subgraph state initialization."
            )
        graph_ids = torch.arange(
            prepared_batch.num_graphs,
            device=prepared_batch.device,
            dtype=torch.long,
        ).repeat_interleave(int(num_rollouts))
        states = tuple(
            self.initial_state()
            for _ in range(prepared_batch.num_graphs * int(num_rollouts))
        )
        return SubgraphRolloutBatch(
            graph_ids=graph_ids,
            states=states,
            done_mask=torch.zeros_like(graph_ids, dtype=torch.bool),
            view_shape=(prepared_batch.num_graphs, int(num_rollouts)),
        )

    def analyze_state(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        state: SubgraphState,
    ) -> SubgraphAnalysis:
        anchors = prepared_batch.graph_anchor_abs_nodes[int(graph_idx)]
        selected_nodes = set(int(anchor) for anchor in anchors)
        directed_adj: dict[int, list[int]] = {}
        undirected_adj: dict[int, list[int]] = {}
        edge_records = _sorted_edge_records(
            topology=prepared_batch.topology,
            edge_ids=state.edge_ids,
        )
        for _, src, _, dst in edge_records:
            selected_nodes.add(int(src))
            selected_nodes.add(int(dst))
            directed_adj.setdefault(int(src), []).append(int(dst))
            undirected_adj.setdefault(int(src), []).append(int(dst))
            undirected_adj.setdefault(int(dst), []).append(int(src))
        ordered_nodes = tuple(sorted(int(node_id) for node_id in selected_nodes))
        reachability_bits = {int(node_id): 0 for node_id in ordered_nodes}
        queue: deque[int] = deque()
        for bit_idx, anchor in enumerate(anchors):
            anchor_node = int(anchor)
            anchor_bits = reachability_bits.get(anchor_node, 0) | (1 << int(bit_idx))
            reachability_bits[anchor_node] = anchor_bits
            queue.append(anchor_node)
        while queue:
            current = int(queue.popleft())
            current_bits = int(reachability_bits.get(current, 0))
            for neighbor in directed_adj.get(current, []):
                updated_bits = int(reachability_bits.get(neighbor, 0)) | current_bits
                if updated_bits == int(reachability_bits.get(neighbor, 0)):
                    continue
                reachability_bits[int(neighbor)] = updated_bits
                queue.append(int(neighbor))
        component_labels: dict[int, int] = {}
        next_component = 0
        for node_id in ordered_nodes:
            node = int(node_id)
            if node in component_labels:
                continue
            stack = [node]
            component_labels[node] = int(next_component)
            while stack:
                current = int(stack.pop())
                for neighbor in undirected_adj.get(current, []):
                    neighbor = int(neighbor)
                    if neighbor in component_labels:
                        continue
                    component_labels[neighbor] = int(next_component)
                    stack.append(neighbor)
            next_component += 1
        anchor_components = {
            int(component_labels[int(anchor)])
            for anchor in anchors
            if int(anchor) in component_labels
        }
        return SubgraphAnalysis(
            selected_node_ids=ordered_nodes,
            reachability_bits=reachability_bits,
            component_labels=component_labels,
            anchor_component_count=int(len(anchor_components)),
            num_selected_edges=int(state.num_edges),
        )

    def analyze_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
    ) -> tuple[SubgraphAnalysis, ...]:
        return tuple(
            self.analyze_state(
                prepared_batch=prepared_batch,
                graph_idx=int(rollout_batch.graph_ids[state_idx].item()),
                state=rollout_batch.states[state_idx],
            )
            for state_idx in range(rollout_batch.num_states)
        )

    def count_gold_answers(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[int, bool]:
        gold_answers = prepared_batch.graph_answer_entities[int(graph_idx)]
        answer_entities = {
            int(entity_id)
            for entity_id in resolve_subgraph_answer_entities(
                prepared_batch=prepared_batch,
                graph_idx=int(graph_idx),
                analysis=analysis,
            )
            if int(entity_id) in gold_answers
        }
        answer_count = int(len(answer_entities))
        return answer_count, bool(answer_count > 0)

    def transition(
        self,
        *,
        rollout_batch: SubgraphRolloutBatch,
        chosen_actions: tuple[SubgraphAction, ...],
    ) -> SubgraphRolloutBatch:
        if len(chosen_actions) != rollout_batch.num_states:
            raise ValueError("chosen_actions must align with rollout states.")
        next_states: list[SubgraphState] = []
        next_done_mask = rollout_batch.done_mask.clone()
        for state_idx, action in enumerate(chosen_actions):
            state = rollout_batch.states[state_idx]
            if bool(next_done_mask[state_idx].item()):
                next_states.append(state)
                continue
            if action.is_stop:
                next_done_mask[state_idx] = True
                next_states.append(state)
                continue
            next_states.append(state.with_edge(int(action.edge_id)))
        return SubgraphRolloutBatch(
            graph_ids=rollout_batch.graph_ids,
            states=tuple(next_states),
            done_mask=next_done_mask,
            view_shape=rollout_batch.view_shape,
        )

    def compute_expand_log_reward(
        self,
        *,
        current_analysis: SubgraphAnalysis,
        next_analysis: SubgraphAnalysis,
    ) -> float:
        return -float(self.reward_cfg.c_step) + float(
            self.reward_cfg.lambda_conn
        ) * float(
            max(
                int(current_analysis.anchor_component_count)
                - int(next_analysis.anchor_component_count),
                0,
            )
        )

    def compute_stop_log_reward(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> tuple[float, int, bool]:
        answer_count, hit = self.count_gold_answers(
            prepared_batch=prepared_batch,
            graph_idx=int(graph_idx),
            analysis=analysis,
        )
        premature = int(analysis.num_selected_edges) < int(
            self.reward_cfg.min_stop_edges
        ) or int(analysis.anchor_component_count) == len(
            prepared_batch.graph_anchor_abs_nodes[int(graph_idx)]
        )
        reward = 0.0
        if hit:
            reward += float(self.reward_cfg.beta_hit)
            reward += float(self.reward_cfg.beta_cnt) * math.log1p(float(answer_count))
        elif premature:
            reward -= float(self.reward_cfg.beta_early)
        return float(reward), answer_count, bool(hit)

    def oracle_distance(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        analysis: SubgraphAnalysis,
    ) -> int:
        oracle_distance_map = prepared_batch.graph_oracle_answer_distance[
            int(graph_idx)
        ]
        return min(
            (
                int(oracle_distance_map[node_id])
                for node_id in analysis.selected_node_ids
                if int(node_id) in oracle_distance_map
            ),
            default=UNREACHABLE_DISTANCE,
        )


__all__ = ["SubgraphEnv"]
