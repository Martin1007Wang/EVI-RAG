from __future__ import annotations

import pytest
import torch

from src.weaver.state import State
from src.weaver.transition import (
    ExpansionBatch,
    SampleMeta,
    SRC_UNKNOWN,
    TerminalBatch,
    TrainingBatch,
)


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("trajectory_ids", None),
        ("step_ids", None),
        ("source_ids", None),
    ],
)
def test_sample_meta_rejects_missing_metadata(
    field_name: str,
    field_value: torch.Tensor | None,
) -> None:
    kwargs = _meta_kwargs()
    kwargs[field_name] = field_value

    with pytest.raises(ValueError, match=rf"{field_name} cannot be None\."):
        SampleMeta(**kwargs)


def test_expansion_batch_rejects_row_mismatch_via_tensor_indexing() -> None:
    parent = _state(num_rows=1)
    child = _state(num_rows=2)
    edge_ids = torch.tensor([0], dtype=torch.long)
    meta = SampleMeta(
        trajectory_ids=torch.tensor([0], dtype=torch.long),
        step_ids=torch.tensor([0], dtype=torch.long),
        source_ids=torch.tensor([SRC_UNKNOWN], dtype=torch.long),
    )

    batch = ExpansionBatch(
        parent=parent,
        child=child,
        edge_ids=edge_ids,
        meta=meta,
    )

    with pytest.raises(IndexError):
        batch.select_rows(torch.tensor([1], dtype=torch.long))


def test_training_batch_counts_expansions_and_terminals() -> None:
    state = _state(num_rows=1)
    meta = SampleMeta(
        trajectory_ids=torch.tensor([0], dtype=torch.long),
        step_ids=torch.tensor([0], dtype=torch.long),
        source_ids=torch.tensor([SRC_UNKNOWN], dtype=torch.long),
    )
    batch = TrainingBatch(
        expansions=ExpansionBatch(
            parent=state,
            child=state.clone(),
            edge_ids=torch.tensor([0], dtype=torch.long),
            meta=meta,
        ),
        terminals=TerminalBatch(
            state=state,
            meta=meta,
        ),
    )

    assert batch.num_expansions == 1
    assert batch.num_terminals == 1
    assert batch.num_items == 2


def _meta_kwargs() -> dict[str, object]:
    return {
        "trajectory_ids": torch.tensor([0], dtype=torch.long),
        "step_ids": torch.tensor([0], dtype=torch.long),
        "source_ids": torch.tensor([SRC_UNKNOWN], dtype=torch.long),
    }


def _state(
    *,
    num_rows: int,
) -> State:
    return State(
        graph_ids=torch.zeros(num_rows, dtype=torch.long),
        selected_edge_mask=torch.zeros((num_rows, 1), dtype=torch.bool),
        active_node_mask=torch.zeros((num_rows, 1), dtype=torch.bool),
        step=torch.zeros(num_rows, dtype=torch.long),
    )
