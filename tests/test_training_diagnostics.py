from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.diagnostics import TrainingDiagnosticsCollector


@dataclass
class FakeLossOutput:
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


def test_training_diagnostics_collects_core_loss_metrics_only() -> None:
    collector = TrainingDiagnosticsCollector(debug=False)
    loss_output = FakeLossOutput(
        loss=torch.tensor(1.0),
        metrics={
            "loss/total": torch.tensor(1.0),
            "reward/log_reward_mean": torch.tensor(2.0),
            "prob/step_log_pf_mean": torch.tensor(3.0),
        },
    )

    assert collector.collect(loss_output=loss_output) == {
        "train/loss/total": 1.0,
        "train/reward/log_reward_mean": 2.0,
    }


def test_training_diagnostics_includes_debug_loss_metrics() -> None:
    collector = TrainingDiagnosticsCollector(debug=True)
    loss_output = FakeLossOutput(
        loss=torch.tensor(1.0),
        metrics={
            "loss/total": torch.tensor(1.0),
            "prob/step_log_pf_mean": torch.tensor(3.0),
        },
    )

    assert collector.collect(loss_output=loss_output) == {
        "train/loss/total": 1.0,
        "train/debug/step_log_pf_mean": 3.0,
    }
