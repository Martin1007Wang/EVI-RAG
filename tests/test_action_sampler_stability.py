from __future__ import annotations

import torch

from src.models.rollout import ActionSampler


def test_action_sampler_sanitizes_invalid_rows_before_sampling() -> None:
    sampler = ActionSampler()
    policy_output = {
        "edge_logits": torch.tensor(
            [1.0, float("nan"), float("-inf")], dtype=torch.float32
        ),
        "out_degrees": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "stop_logits": torch.tensor([[-1.0, -1.0, float("-inf")]], dtype=torch.float32),
        "edge_ids": torch.tensor([0, 1, 2], dtype=torch.long),
        "target_nodes": torch.tensor([1, 2, 3], dtype=torch.long),
    }
    action_info = sampler(
        policy_output,
        is_training=True,
        deterministic=False,
        sampling_mode="gumbel",
        sampling_temperature=1.0,
        eval_sampling_temperature=1.0,
        eval_sample_without_replacement=False,
    )
    assert bool(torch.isfinite(action_info["log_prob"]).all().item())
    assert bool(torch.isfinite(action_info["log_partition"]).all().item())
    # The invalid third row is forced to STOP after sanitization.
    assert bool(action_info["is_stop"][2].item())
    assert int(action_info["chosen_edge_ids"][2].item()) == -1
