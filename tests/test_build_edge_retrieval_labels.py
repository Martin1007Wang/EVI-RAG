from __future__ import annotations

from types import SimpleNamespace

import torch

from src.datasets import build_edge_retrieval_labels as label_builder


class _DummyLabelDataset:
    def __init__(self) -> None:
        self.closed = False

    def __len__(self) -> int:
        return 1

    def get(self, idx: int) -> SimpleNamespace:
        assert idx == 0
        return SimpleNamespace(
            sample_id="sample-1",
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            anchor_local_indices=torch.tensor([0], dtype=torch.long),
            a_local_indices=torch.tensor([1], dtype=torch.long),
            num_nodes=2,
        )

    def close(self) -> None:
        self.closed = True


def test_build_split_closes_dataset(tmp_path, monkeypatch) -> None:
    dataset = _DummyLabelDataset()

    monkeypatch.setattr(
        label_builder,
        "create_graph_retrieval_dataset",
        lambda **kwargs: dataset,
    )
    monkeypatch.setattr(
        label_builder,
        "compute_shortest_path_labels",
        lambda **kwargs: SimpleNamespace(
            num_edges=1,
            positive_edge_ids=torch.tensor([0], dtype=torch.long),
            max_path_length=1,
        ),
    )

    out_path = label_builder._build_split(
        {"dataset": {"name": "webqsp-sub"}},
        split="train",
        output_dir=tmp_path,
        overwrite=True,
    )

    assert out_path.exists()
    assert dataset.closed is True
