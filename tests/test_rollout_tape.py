from __future__ import annotations

import torch

from src.weaver.rollout.tape import NO_STEP, RolloutTape


def test_rollout_tape_initializes_1d_buffers_with_tuple_shapes() -> None:
    tape = RolloutTape(
        R=3,
        T=4,
        device=torch.device("cpu"),
    )

    assert tape.selected_edge_ids.shape == (3, 4)
    assert tape.policy_action_log_prob.shape == (3, 4)
    assert tape.behavior_action_log_prob.shape == (3, 4)
    assert tape.terminal_step.shape == (3,)
    assert tape.stop_reason.shape == (3,)
    assert tape.is_stopped.shape == (3,)
    assert torch.equal(
        tape.stop_reason,
        torch.full((3,), NO_STEP, dtype=torch.long),
    )
