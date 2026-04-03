from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import SubgraphAction


@dataclass(frozen=True)
class HierarchicalEdgeChoice:
    action: SubgraphAction
    edge_id: int
    source_graph_node: int
    relation_id: int
    target_graph_node: int
    logit: torch.Tensor
    current_component_count: int
    next_component_count: int
    question_similarity: float
    semantic_overlap: float
    action_new_bit_gain: int
    answer_candidate_count: int
    current_best_answer_distance: float
    target_answer_distance: float


@dataclass(frozen=True)
class HierarchicalRelationChoice:
    relation_id: int
    logit: torch.Tensor
    edges: tuple[HierarchicalEdgeChoice, ...]


@dataclass(frozen=True)
class HierarchicalNodeChoice:
    graph_node_id: int
    logit: torch.Tensor
    relations: tuple[HierarchicalRelationChoice, ...]


@dataclass(frozen=True)
class AnswerStopChoice:
    action: SubgraphAction
    answer_entity_id: int | None
    logit: torch.Tensor
    support_node_count: int


@dataclass(frozen=True)
class HierarchicalStateActionDistribution:
    flat_state_index: int
    stop_logit: torch.Tensor
    continue_logit: torch.Tensor
    stop_choice_logits: torch.Tensor
    stop_choice_answer_entity_ids: torch.Tensor
    stop_choice_support_node_counts: torch.Tensor
    node_choice_graph_node_ids: torch.Tensor
    node_choice_logits: torch.Tensor
    relation_choice_relation_ids: torch.Tensor
    relation_choice_logits: torch.Tensor
    relation_choice_node_choice_indices: torch.Tensor
    edge_choice_edge_ids: torch.Tensor
    edge_choice_source_graph_nodes: torch.Tensor
    edge_choice_relation_ids: torch.Tensor
    edge_choice_target_graph_nodes: torch.Tensor
    edge_choice_logits: torch.Tensor
    edge_choice_next_component_counts: torch.Tensor
    edge_choice_question_similarity: torch.Tensor
    edge_choice_semantic_overlap: torch.Tensor
    edge_choice_action_new_bit_gain: torch.Tensor
    edge_choice_answer_candidate_counts: torch.Tensor
    edge_choice_target_answer_distance: torch.Tensor
    edge_choice_relation_choice_indices: torch.Tensor
    node_relation_ptr: torch.Tensor
    relation_edge_ptr: torch.Tensor
    current_component_count: int
    current_answer_candidate_count: int
    current_oracle_distance: float

    def _stop_answer_entity_id(self, stop_choice_idx: int) -> int | None:
        del stop_choice_idx
        return None

    def relation_slice(self, node_choice_idx: int) -> slice:
        start = int(self.node_relation_ptr[int(node_choice_idx)].item())
        end = int(self.node_relation_ptr[int(node_choice_idx) + 1].item())
        return slice(start, end)

    def edge_slice(self, relation_choice_idx: int) -> slice:
        start = int(self.relation_edge_ptr[int(relation_choice_idx)].item())
        end = int(self.relation_edge_ptr[int(relation_choice_idx) + 1].item())
        return slice(start, end)

    def build_stop_action(self, stop_choice_idx: int) -> SubgraphAction:
        del stop_choice_idx
        return SubgraphAction.stop()

    def build_edge_action(self, edge_choice_idx: int) -> SubgraphAction:
        edge_choice_idx = int(edge_choice_idx)
        return SubgraphAction.add_edge(
            int(self.edge_choice_edge_ids[edge_choice_idx].item()),
            source_graph_node=int(
                self.edge_choice_source_graph_nodes[edge_choice_idx].item()
            ),
            relation_id=int(self.edge_choice_relation_ids[edge_choice_idx].item()),
            target_graph_node=int(
                self.edge_choice_target_graph_nodes[edge_choice_idx].item()
            ),
        )

    def edge_next_component_count(self, edge_choice_idx: int) -> int:
        return int(self.edge_choice_next_component_counts[int(edge_choice_idx)].item())

    @property
    def stop_choices(self) -> tuple[AnswerStopChoice, ...]:
        return tuple(
            AnswerStopChoice(
                action=self.build_stop_action(stop_choice_idx),
                answer_entity_id=self._stop_answer_entity_id(stop_choice_idx),
                logit=self.stop_choice_logits[int(stop_choice_idx)],
                support_node_count=int(
                    self.stop_choice_support_node_counts[int(stop_choice_idx)].item()
                ),
            )
            for stop_choice_idx in range(int(self.stop_choice_logits.numel()))
        )

    @property
    def node_choices(self) -> tuple[HierarchicalNodeChoice, ...]:
        node_choices: list[HierarchicalNodeChoice] = []
        for node_choice_idx in range(int(self.node_choice_logits.numel())):
            relation_choices: list[HierarchicalRelationChoice] = []
            relation_slice = self.relation_slice(node_choice_idx)
            for relation_choice_idx in range(relation_slice.start, relation_slice.stop):
                edge_choices: list[HierarchicalEdgeChoice] = []
                edge_slice = self.edge_slice(relation_choice_idx)
                for edge_choice_idx in range(edge_slice.start, edge_slice.stop):
                    edge_choices.append(
                        HierarchicalEdgeChoice(
                            action=self.build_edge_action(edge_choice_idx),
                            edge_id=int(
                                self.edge_choice_edge_ids[edge_choice_idx].item()
                            ),
                            source_graph_node=int(
                                self.edge_choice_source_graph_nodes[
                                    edge_choice_idx
                                ].item()
                            ),
                            relation_id=int(
                                self.edge_choice_relation_ids[edge_choice_idx].item()
                            ),
                            target_graph_node=int(
                                self.edge_choice_target_graph_nodes[
                                    edge_choice_idx
                                ].item()
                            ),
                            logit=self.edge_choice_logits[edge_choice_idx],
                            current_component_count=int(self.current_component_count),
                            next_component_count=self.edge_next_component_count(
                                edge_choice_idx
                            ),
                            question_similarity=float(
                                self.edge_choice_question_similarity[
                                    edge_choice_idx
                                ].item()
                            ),
                            semantic_overlap=float(
                                self.edge_choice_semantic_overlap[
                                    edge_choice_idx
                                ].item()
                            ),
                            action_new_bit_gain=int(
                                self.edge_choice_action_new_bit_gain[
                                    edge_choice_idx
                                ].item()
                            ),
                            answer_candidate_count=int(
                                self.edge_choice_answer_candidate_counts[
                                    edge_choice_idx
                                ].item()
                            ),
                            current_best_answer_distance=float(
                                self.current_oracle_distance
                            ),
                            target_answer_distance=float(
                                self.edge_choice_target_answer_distance[
                                    edge_choice_idx
                                ].item()
                            ),
                        )
                    )
                relation_choices.append(
                    HierarchicalRelationChoice(
                        relation_id=int(
                            self.relation_choice_relation_ids[
                                relation_choice_idx
                            ].item()
                        ),
                        logit=self.relation_choice_logits[relation_choice_idx],
                        edges=tuple(edge_choices),
                    )
                )
            node_choices.append(
                HierarchicalNodeChoice(
                    graph_node_id=int(
                        self.node_choice_graph_node_ids[node_choice_idx].item()
                    ),
                    logit=self.node_choice_logits[node_choice_idx],
                    relations=tuple(relation_choices),
                )
            )
        return tuple(node_choices)


@dataclass(frozen=True)
class SubgraphActionDistribution:
    flat_state_indices: torch.Tensor
    state_features: torch.Tensor
    state_distributions: tuple[HierarchicalStateActionDistribution, ...]


@dataclass(frozen=True)
class _ActionDistributionBuildStats:
    raw_candidate_count: int
    stop_choice_count: int


__all__ = [
    "AnswerStopChoice",
    "HierarchicalEdgeChoice",
    "HierarchicalNodeChoice",
    "HierarchicalRelationChoice",
    "HierarchicalStateActionDistribution",
    "SubgraphActionDistribution",
]
