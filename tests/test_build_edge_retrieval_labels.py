from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.data.retrieval import build_edge_retrieval_labels as label_builder


class _DummyLabelDataset:
    def __init__(self, items: list[SimpleNamespace]) -> None:
        self._items = items
        self.closed = False

    def __len__(self) -> int:
        return len(self._items)

    def get(self, idx: int) -> SimpleNamespace:
        return self._items[idx]

    def close(self) -> None:
        self.closed = True


def _make_sample(
    sample_id: str,
    *,
    edge_index: torch.Tensor,
    anchor_local_indices: torch.Tensor,
    a_local_indices: torch.Tensor,
    num_nodes: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        sample_id=sample_id,
        edge_index=edge_index,
        anchor_local_indices=anchor_local_indices,
        a_local_indices=a_local_indices,
        num_nodes=num_nodes,
    )


def test_build_split_writes_expected_payload_and_statistics(
    tmp_path, monkeypatch
) -> None:
    dataset = _DummyLabelDataset(
        [
            _make_sample(
                "sample-path",
                edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
                anchor_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([2], dtype=torch.long),
                num_nodes=3,
            ),
            _make_sample(
                "sample-zero-hop",
                edge_index=torch.empty((2, 0), dtype=torch.long),
                anchor_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([0], dtype=torch.long),
                num_nodes=1,
            ),
            _make_sample(
                "sample-no-path",
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                anchor_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([3], dtype=torch.long),
                num_nodes=4,
            ),
            _make_sample(
                "",
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                anchor_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([1], dtype=torch.long),
                num_nodes=2,
            ),
        ]
    )

    monkeypatch.setattr(
        label_builder,
        "create_graph_retrieval_dataset",
        lambda **kwargs: dataset,
    )

    def _compute_shortest_path_labels(**kwargs):  # type: ignore[no-untyped-def]
        num_nodes = int(kwargs["num_nodes"])
        if num_nodes == 3:
            return SimpleNamespace(
                num_edges=2,
                positive_edge_ids=torch.tensor([0, 1], dtype=torch.long),
                max_path_length=2,
            )
        if num_nodes == 1:
            return SimpleNamespace(
                num_edges=0,
                positive_edge_ids=torch.tensor([], dtype=torch.long),
                max_path_length=0,
            )
        if num_nodes == 4:
            return SimpleNamespace(
                num_edges=1,
                positive_edge_ids=torch.tensor([], dtype=torch.long),
                max_path_length=None,
            )
        raise AssertionError(f"unexpected num_nodes={num_nodes}")

    monkeypatch.setattr(
        label_builder,
        "compute_shortest_path_labels",
        _compute_shortest_path_labels,
    )

    out_path = label_builder._build_split(
        {"dataset": {"name": "webqsp-sub"}},
        split="train",
        output_dir=tmp_path,
        overwrite=True,
    )

    payload = torch.load(out_path, weights_only=False)

    assert dataset.closed is True
    assert payload["meta"] == {
        "algo": "edge_retrieval_shortest_paths_strict_v1",
        "split": "train",
        "num_samples": 3,
        "no_path_samples": 1,
        "zero_hop_samples": 1,
    }
    assert set(payload["entries"]) == {
        "sample-path",
        "sample-zero-hop",
        "sample-no-path",
    }
    assert payload["entries"]["sample-path"]["num_edges"] == 2
    assert torch.equal(
        payload["entries"]["sample-path"]["positive_edge_ids"],
        torch.tensor([0, 1], dtype=torch.long),
    )
    assert payload["entries"]["sample-path"]["max_path_length"] == 2
    assert payload["entries"]["sample-zero-hop"]["num_edges"] == 0
    assert payload["entries"]["sample-zero-hop"]["max_path_length"] == 0
    assert payload["entries"]["sample-no-path"]["num_edges"] == 1
    assert payload["entries"]["sample-no-path"]["max_path_length"] is None


def test_build_split_closes_dataset_on_label_failure(tmp_path, monkeypatch) -> None:
    dataset = _DummyLabelDataset(
        [
            _make_sample(
                "sample-path",
                edge_index=torch.tensor([[0], [1]], dtype=torch.long),
                anchor_local_indices=torch.tensor([0], dtype=torch.long),
                a_local_indices=torch.tensor([1], dtype=torch.long),
                num_nodes=2,
            )
        ]
    )

    monkeypatch.setattr(
        label_builder,
        "create_graph_retrieval_dataset",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        label_builder,
        "compute_shortest_path_labels",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("label failure")),
    )

    with pytest.raises(RuntimeError, match="label failure"):
        label_builder._build_split(
            {"dataset": {"name": "webqsp-sub"}},
            split="train",
            output_dir=tmp_path,
            overwrite=True,
        )

    assert dataset.closed is True
    assert not (tmp_path / "train.pt").exists()
