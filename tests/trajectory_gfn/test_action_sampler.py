from __future__ import annotations

import torch

from src.models.trajectory_gfn.action_sampler import ActionSampler


def test_action_sampler_masks_invalid_rows_to_stop() -> None:
    sampler = ActionSampler()
    policy_output = {
        "edge_logits": torch.tensor(
            [1.0, float("nan"), float("-inf")],
            dtype=torch.float32,
        ),
        "out_degrees": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "stop_logits": torch.tensor(
            [[-1.0, -1.0, float("-inf")]],
            dtype=torch.float32,
        ),
        "edge_ids": torch.tensor([0, 1, 2], dtype=torch.long),
        "target_nodes": torch.tensor([1, 2, 3], dtype=torch.long),
    }
    action_info = sampler(
        policy_output,
        is_training=True,
        sampling_temperature=1.0,
        invalid_logits_policy="mask",
    )
    assert bool(torch.isfinite(action_info["log_prob"]).all().item())
    assert bool(torch.isfinite(action_info["log_partition"]).all().item())
    assert bool(action_info["is_stop"][2].item())
    assert int(action_info["chosen_edge_ids"][2].item()) == -1
