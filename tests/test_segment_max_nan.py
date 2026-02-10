from __future__ import annotations

import torch

from src.utils.tensor_ops import neg_inf_value, segment_max


def test_segment_max_ignores_nan_for_argmax() -> None:
    src = torch.tensor([float("nan"), 1.0, float("nan"), 2.0], dtype=torch.float32)
    segment_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    max_per, argmax = segment_max(src, segment_ids, num_segments=2)

    assert torch.allclose(max_per, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert argmax.tolist() == [1, 3]


def test_segment_max_all_nan_segment_falls_back_to_neg_inf() -> None:
    src = torch.tensor([float("nan"), float("nan"), 5.0], dtype=torch.float32)
    segment_ids = torch.tensor([0, 0, 1], dtype=torch.long)

    max_per, argmax = segment_max(src, segment_ids, num_segments=2)

    assert max_per[0].item() == neg_inf_value(src)
    assert max_per[1].item() == 5.0
    assert argmax.tolist() == [0, 2]

