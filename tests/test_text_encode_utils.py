from __future__ import annotations

from pathlib import Path

import torch

from text_encode_utils import encode_to_memmap


class _FakeEncoder:
    hidden_size = 3
    progress_bar = False

    def encode(self, texts, batch_size, *, show_progress=False, desc=None):  # type: ignore[no-untyped-def]
        del batch_size, show_progress, desc
        rows = []
        for index, _ in enumerate(texts, start=1):
            rows.append([float(index), float(index + 1), float(index + 2)])
        return torch.tensor(rows, dtype=torch.float32)


def test_encode_to_memmap_writes_embedding_rows_by_id(tmp_path: Path) -> None:
    out_path = tmp_path / "entity_embeddings.pt"

    result = encode_to_memmap(
        encoder=_FakeEncoder(),
        texts=["alpha", "beta"],
        emb_ids=[2, 4],
        batch_size=8,
        max_embedding_id=4,
        out_path=out_path,
    )

    assert result == out_path
    stored = torch.load(out_path, map_location="cpu")
    assert tuple(stored.shape) == (5, 3)
    assert torch.equal(stored[2], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(stored[4], torch.tensor([2.0, 3.0, 4.0]))
    assert torch.equal(stored[0], torch.zeros(3))
