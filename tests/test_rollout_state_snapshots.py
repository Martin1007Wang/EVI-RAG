from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rollout.engine import RolloutEngine
from src.models.rollout.executor import StepExecutor
from src.models.state import State


class _RecordingPolicy:
    def __init__(self) -> None:
        self.last_state: State | None = None

    def __call__(
        self,
        base_graph: object,
        state: State,
        backbone_static_context: object | None = None,
    ) -> object:
        self.last_state = state
        return SimpleNamespace(
            base_graph=base_graph,
            state=state,
            backbone_static_context=backbone_static_context,
        )


class _TerminalBackwardPolicy:
    def __init__(self) -> None:
        self.last_state: State | None = None

    def terminal_backward_log_prob(
        self,
        *,
        base_graph: object,
        state: State,
        step_output: object,
    ) -> torch.Tensor:
        self.last_state = state
        return torch.zeros(1, dtype=torch.float32)


def _build_state() -> State:
    return State(
        root_active_edges=torch.tensor([True, False, False], dtype=torch.bool),
        active_nodes=torch.tensor([True, False, False], dtype=torch.bool),
        active_edges=torch.tensor([True, False, False], dtype=torch.bool),
    )


def test_policy_forward_uses_detached_state_snapshot() -> None:
    policy = _RecordingPolicy()
    state = _build_state()

    RolloutEngine._policy_forward(policy, object(), state, backbone_ctx=None)

    captured = policy.last_state
    assert captured is not None
    assert captured is not state
    assert captured.active_nodes.data_ptr() != state.active_nodes.data_ptr()
    assert captured.active_edges.data_ptr() != state.active_edges.data_ptr()

    state.apply_expansion(
        chosen_edges=torch.tensor([1], dtype=torch.long),
        src=torch.tensor([0, 0, 1], dtype=torch.long),
        dst=torch.tensor([1, 2, 2], dtype=torch.long),
    )

    assert captured.active_nodes.tolist() == [True, False, False]
    assert captured.active_edges.tolist() == [True, False, False]


def test_terminal_backward_log_prob_uses_detached_state_snapshot() -> None:
    policy = _TerminalBackwardPolicy()
    state = _build_state()
    executor = StepExecutor(max_steps=1, terminal_backward_mode="policy")
    base_graph = SimpleNamespace(
        ptr=torch.tensor([0, 1], dtype=torch.long),
        node_tokens=torch.zeros(1, dtype=torch.long),
    )

    output = executor._terminal_backward_log_prob(
        policy=policy,
        base_graph=base_graph,
        state=state,
        step_output=object(),
    )

    captured = policy.last_state
    assert captured is not None
    assert output.shape == (1,)
    assert captured is not state
    assert captured.active_nodes.data_ptr() != state.active_nodes.data_ptr()
    assert captured.active_edges.data_ptr() != state.active_edges.data_ptr()

    state.apply_expansion(
        chosen_edges=torch.tensor([1], dtype=torch.long),
        src=torch.tensor([0, 0, 1], dtype=torch.long),
        dst=torch.tensor([1, 2, 2], dtype=torch.long),
    )

    assert captured.active_nodes.tolist() == [True, False, False]
    assert captured.active_edges.tolist() == [True, False, False]
