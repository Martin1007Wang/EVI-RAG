from __future__ import annotations

from src.data.reasoner_path_dataset import ReasonerPathDataset


def test_reasoner_path_dataset_uses_traversal_direction() -> None:
    # Bypass __init__ (it requires filesystem parquet caches); we only validate formatting logic.
    dataset = ReasonerPathDataset.__new__(ReasonerPathDataset)
    dataset._ent_map = {1: "A", 2: "B"}
    dataset._rel_map = {0: "r0"}

    # Graph-defined direction: head=1 -> tail=2, but the agent traversed reverse: 2 -> 1.
    edge = {
        "head_entity_id": 1,
        "tail_entity_id": 2,
        "relation_id": 0,
        "src_entity_id": 2,
        "dst_entity_id": 1,
    }
    assert dataset._fmt_edge(edge) == "B -[r0]-> A"
