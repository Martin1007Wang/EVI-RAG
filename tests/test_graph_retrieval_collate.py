from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from src.datasets.graph_retrieval_collate import (
    _InstrumentedDataLoader,
    _expand_answer_samples,
)


@dataclass
class _CloneableSample:
    q_local_indices: torch.Tensor
    a_local_indices: torch.Tensor
    answer_entity_ids: torch.Tensor
    node_entity_ids: torch.Tensor
    sample_id: str

    def clone(self) -> "_CloneableSample":
        return _CloneableSample(
            q_local_indices=self.q_local_indices.clone(),
            a_local_indices=self.a_local_indices.clone(),
            answer_entity_ids=self.answer_entity_ids.clone(),
            node_entity_ids=self.node_entity_ids.clone(),
            sample_id=self.sample_id,
        )


def test_expand_answer_samples_filters_only_zero_hop_answers_when_not_expanding() -> (
    None
):
    sample = _CloneableSample(
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="sample-1",
    )

    expanded = _expand_answer_samples(
        [sample],
        expand_multi_answer=False,
        filter_zero_hop=True,
    )

    assert len(expanded) == 1
    assert torch.equal(expanded[0].a_local_indices, torch.tensor([1], dtype=torch.long))
    assert torch.equal(
        expanded[0].answer_entity_ids,
        torch.tensor([101], dtype=torch.long),
    )
    assert expanded[0].sample_id == "sample-1"


def test_instrumented_dataloader_logs_iterator_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def _capture(_logger, event: str, **fields: object) -> None:  # type: ignore[no-untyped-def]
        events.append((event, dict(fields)))

    monkeypatch.setattr("src.datasets.graph_retrieval_collate.log_event", _capture)

    loader = _InstrumentedDataLoader(
        [1, 2, 3],
        batch_size=2,
        num_workers=0,
        loader_name="probe",
        multiprocessing_context_name=None,
    )

    iterator = iter(loader)
    batch = next(iterator)

    assert torch.equal(batch, torch.tensor([1, 2]))
    assert [event for event, _ in events] == [
        "retrieval_dataloader_iter_start",
        "retrieval_dataloader_iter_ready",
        "retrieval_dataloader_first_batch_ready",
    ]
    assert all(fields["loader_name"] == "probe" for _, fields in events)
