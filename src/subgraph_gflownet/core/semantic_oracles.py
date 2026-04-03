from __future__ import annotations

from dataclasses import dataclass

from .state import SubgraphAnalysis
from .subgraph_batch import UNREACHABLE_DISTANCE, SubgraphBatch


@dataclass(frozen=True)
class AdmissibleAnswerSet:
    entities: tuple[int, ...]
    gold_entities: tuple[int, ...]
    full_anchor_mask: int

    @property
    def count(self) -> int:
        return int(len(self.entities))

    @property
    def gold_count(self) -> int:
        return int(len(self.gold_entities))

    def contains(self, entity_id: int) -> bool:
        return int(entity_id) in self.entities

    def is_gold(self, entity_id: int) -> bool:
        return int(entity_id) in self.gold_entities


@dataclass(frozen=True)
class RewardWeights:
    hit_bonus: float = 5.0
    frontier_bonus: float = 1.0
    coverage_bonus: float = 0.2
    size_penalty: float = 0.1
    component_penalty: float = 0.5


@dataclass(frozen=True)
class TerminalRewardSummary:
    log_reward: float
    hit: bool
    answer_set: AdmissibleAnswerSet
    gold_answer_entities_in_graph: tuple[int, ...]
    frontier_hit: bool
    anchor_coverage: float
    utility: float
    redundancy_edges: int

    @property
    def answer_entities(self) -> tuple[int, ...]:
        return tuple(int(entity_id) for entity_id in self.answer_set.entities)

    @property
    def gold_answer_count(self) -> int:
        return int(len(self.gold_answer_entities_in_graph))


def resolve_admissible_answer_set(
    *,
    batch: SubgraphBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> AdmissibleAnswerSet:
    graph = batch.graph(graph_idx)
    full_mask = int(graph.anchor_full_mask)
    if full_mask <= 0:
        return AdmissibleAnswerSet(
            entities=(), gold_entities=(), full_anchor_mask=full_mask
        )
    admissible_entities = tuple(
        sorted(
            int(entity_id)
            for entity_id, bits in analysis.entity_reachability_bits.items()
            if int(bits) == int(full_mask)
        )
    )
    gold_entities = tuple(
        int(entity_id)
        for entity_id in admissible_entities
        if int(entity_id) in graph.answer_entities
    )
    return AdmissibleAnswerSet(
        entities=admissible_entities,
        gold_entities=gold_entities,
        full_anchor_mask=full_mask,
    )


def gold_answer_entities_in_graph(
    *, batch: SubgraphBatch, graph_idx: int, analysis: SubgraphAnalysis
) -> tuple[int, ...]:
    gold_answers = batch.graph(graph_idx).answer_entities
    if not gold_answers:
        return ()
    return tuple(
        sorted(
            int(entity_id)
            for entity_id in set(
                int(entity_id) for entity_id in analysis.state_node_entity_ids
            )
            if int(entity_id) in gold_answers
        )
    )


def frontier_hits_gold_answer(
    *, batch: SubgraphBatch, graph_idx: int, analysis: SubgraphAnalysis
) -> bool:
    graph = batch.graph(graph_idx)
    gold_answers = graph.answer_entities
    if not gold_answers:
        return False
    selected_entities = {int(entity_id) for entity_id in analysis.state_node_entity_ids}
    if selected_entities.intersection(gold_answers):
        return False
    for node_id in analysis.selected_node_ids:
        for edge_id in graph.outgoing_edge_ids.get(int(node_id), ()):
            target_node_id = int(batch.topology.edge_index[1, int(edge_id)].item())
            target_entity_id = int(batch.node_entity_ids[target_node_id].item())
            if int(target_entity_id) in gold_answers:
                return True
    return False


def anchor_coverage_ratio(
    *, batch: SubgraphBatch, graph_idx: int, analysis: SubgraphAnalysis
) -> float:
    graph = batch.graph(graph_idx)
    anchor_node_ids = {int(node_id) for node_id in graph.anchor_abs_node_ids}
    anchor_count = max(int(graph.anchor_full_mask).bit_count(), 1)
    covered_anchor_bits = 0
    for anchor_idx in range(anchor_count):
        bit = 1 << int(anchor_idx)
        if any(
            int(node_id) not in anchor_node_ids
            and (int(analysis.reachability_bits.get(int(node_id), 0)) & int(bit)) > 0
            for node_id in analysis.selected_node_ids
        ):
            covered_anchor_bits |= int(bit)
    return float(int(covered_anchor_bits).bit_count()) / float(anchor_count)


def redundancy_edge_count(analysis: SubgraphAnalysis) -> int:
    selected_nodes = int(max(analysis.num_state_nodes, len(analysis.selected_node_ids)))
    minimal_forest_edges = max(selected_nodes - int(analysis.anchor_component_count), 0)
    return max(int(analysis.num_selected_edges) - int(minimal_forest_edges), 0)


def compute_terminal_reward(
    *,
    batch: SubgraphBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
    weights: RewardWeights,
) -> TerminalRewardSummary:
    answer_set = resolve_admissible_answer_set(
        batch=batch, graph_idx=graph_idx, analysis=analysis
    )
    gold_entities = gold_answer_entities_in_graph(
        batch=batch, graph_idx=graph_idx, analysis=analysis
    )
    frontier_hit = frontier_hits_gold_answer(
        batch=batch, graph_idx=graph_idx, analysis=analysis
    )
    anchor_coverage = anchor_coverage_ratio(
        batch=batch, graph_idx=graph_idx, analysis=analysis
    )
    redundancy_edges = redundancy_edge_count(analysis)
    structure_penalty = float(weights.size_penalty) * float(
        analysis.num_selected_edges
    ) + float(weights.component_penalty) * float(
        max(int(analysis.anchor_component_count) - 1, 0)
    )
    hit_value = 1.0 if gold_entities else 0.0
    frontier_value = 1.0 if frontier_hit and not gold_entities else 0.0
    utility = (
        float(weights.hit_bonus) * float(hit_value)
        + float(weights.frontier_bonus) * float(frontier_value)
        + float(weights.coverage_bonus) * float(anchor_coverage)
    )
    return TerminalRewardSummary(
        log_reward=float(utility - structure_penalty),
        hit=bool(gold_entities),
        answer_set=answer_set,
        gold_answer_entities_in_graph=gold_entities,
        frontier_hit=bool(frontier_hit),
        anchor_coverage=float(anchor_coverage),
        utility=float(utility),
        redundancy_edges=int(redundancy_edges),
    )


def oracle_distance(
    *, batch: SubgraphBatch, graph_idx: int, analysis: SubgraphAnalysis
) -> int:
    oracle_distance_map = batch.graph(graph_idx).oracle_answer_distance
    return min(
        (
            int(oracle_distance_map[node_id])
            for node_id in analysis.selected_node_ids
            if int(node_id) in oracle_distance_map
        ),
        default=UNREACHABLE_DISTANCE,
    )


__all__ = [
    "AdmissibleAnswerSet",
    "RewardWeights",
    "TerminalRewardSummary",
    "anchor_coverage_ratio",
    "compute_terminal_reward",
    "frontier_hits_gold_answer",
    "gold_answer_entities_in_graph",
    "oracle_distance",
    "redundancy_edge_count",
    "resolve_admissible_answer_set",
]
