from __future__ import annotations

import pyarrow as pa

from src.data.preprocess.stages.step3_lmdb import (
    _expected_entity_embedding_rows,
    _take_graph_batch_columns,
)


def test_expected_entity_embedding_rows_keeps_reserved_zero_row() -> None:
    assert _expected_entity_embedding_rows({"embedding_id": []}) == 1
    assert _expected_entity_embedding_rows({"embedding_id": [0, 0, 2]}) == 3


def test_take_graph_batch_columns_reads_only_requested_rows() -> None:
    table = pa.table(
        {
            "node_entity_ids": [[0], [1], [2]],
            "node_embedding_ids": [[0], [1], [2]],
            "edge_src": [[0], [0], [0]],
            "edge_dst": [[0], [0], [0]],
            "edge_relation_ids": [[0], [1], [2]],
        }
    )

    row_lookup, graph_cols = _take_graph_batch_columns(table, [2, 0, 2])

    assert row_lookup == {2: 0, 0: 1}
    assert graph_cols["node_entity_ids"] == [[2], [0]]
    assert graph_cols["edge_relation_ids"] == [[2], [0]]
