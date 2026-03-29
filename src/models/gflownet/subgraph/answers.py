from __future__ import annotations

from .prepared_batch import SubgraphPreparedBatch
from .state import SubgraphAnalysis


def resolve_subgraph_answer_entities(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> tuple[int, ...]:
    full_mask = int(prepared_batch.graph_anchor_full_mask[int(graph_idx)])
    if full_mask <= 0:
        return ()
    answer_entities: set[int] = set()
    for node_id in analysis.selected_node_ids:
        node_bits = int(analysis.reachability_bits.get(int(node_id), 0))
        if node_bits != full_mask:
            continue
        entity_id = int(prepared_batch.node_entity_ids[int(node_id)].item())
        answer_entities.add(entity_id)
    return tuple(sorted(answer_entities))


__all__ = ["resolve_subgraph_answer_entities"]
