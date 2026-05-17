from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.data.collate import RetrievalCollator
from src.data.dataset import _build_retrieval_data
from src.data.schema.fields import SampleFields
from src.training.config import (
    EvalRuntimeConfig,
    OptimizationRuntimeConfig,
    OptimizerRuntimeConfig,
)
from src.weaver.module import WeaverModule
from src.weaver.reward import EvidenceLogReward
from src.weaver.rollout.result import RolloutResult


def _record_two_paths() -> dict[str, torch.Tensor]:
    return {
        SampleFields.EDGE_INDEX: torch.tensor(
            [[0, 1, 0, 2], [1, 3, 2, 3]],
            dtype=torch.long,
        ),
        SampleFields.NODE_ENTITY_CATALOG_IDS: torch.tensor([0, 1, 2, 3], dtype=torch.long),
        SampleFields.EDGE_RELATION_CATALOG_IDS: torch.tensor([0, 0, 0, 0], dtype=torch.long),
        SampleFields.NUM_NODES: torch.tensor(4, dtype=torch.long),
        SampleFields.NUM_EDGES: torch.tensor(4, dtype=torch.long),
        SampleFields.ANCHOR_NODE_IDS: torch.tensor([0], dtype=torch.long),
        SampleFields.TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.REACHABLE_TARGET_NODE_IDS: torch.tensor([3], dtype=torch.long),
        SampleFields.ANCHOR_NODE_FORWARD_DISTANCE_FLAT: torch.tensor([0, 1, 1, 2], dtype=torch.long),
        SampleFields.ANCHOR_NODE_BACKWARD_DISTANCE_FLAT: torch.tensor([0, -1, -1, -1], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCE: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_DISTANCES_FLAT: torch.tensor([2, 1, 1, 0], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_COUNT_FLAT: torch.tensor(
            [2.0, 1.0, 1.0, 1.0],
            dtype=torch.float32,
        ),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES: torch.tensor([0, 1, 2, 3], dtype=torch.long),
        SampleFields.NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES: torch.ones(4, dtype=torch.float32),
    }


def _batch() -> object:
    data = _build_retrieval_data(
        raw=_record_two_paths(),
        sample_id="two-paths",
        question_emb=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    return RetrievalCollator()([data])


def _rollout_result(
    *,
    source_graph_id: list[int],
    expand_steps: dict[int, dict[int, int]],
    max_steps: int,
    stop_steps: dict[int, int] | None = None,
    forced_stop_steps: dict[int, int] | None = None,
) -> RolloutResult:
    num_rows = len(source_graph_id)
    stop_steps = stop_steps or {}
    forced_stop_steps = forced_stop_steps or {}

    expand_mask = torch.zeros((num_rows, max_steps), dtype=torch.bool)
    stop_mask = torch.zeros((num_rows, max_steps), dtype=torch.bool)
    forced_stop_mask = torch.zeros((num_rows, max_steps), dtype=torch.bool)
    valid_mask = torch.zeros((num_rows, max_steps), dtype=torch.bool)
    selected_edge_ids = torch.full((num_rows, max_steps), -1, dtype=torch.long)
    traj_len = torch.zeros(num_rows, dtype=torch.long)
    terminal_step = torch.full((num_rows,), -1, dtype=torch.long)

    for row in range(num_rows):
        last_step = -1
        for step, edge_id in expand_steps.get(row, {}).items():
            expand_mask[row, step] = True
            valid_mask[row, step] = True
            selected_edge_ids[row, step] = int(edge_id)
            last_step = max(last_step, int(step))
        if row in stop_steps:
            step = int(stop_steps[row])
            stop_mask[row, step] = True
            valid_mask[row, step] = True
            last_step = max(last_step, step)
        if row in forced_stop_steps:
            step = int(forced_stop_steps[row])
            stop_mask[row, step] = True
            forced_stop_mask[row, step] = True
            valid_mask[row, step] = True
            last_step = max(last_step, step)
        traj_len[row] = last_step + 1 if last_step >= 0 else 0
        terminal_step[row] = last_step

    return RolloutResult(
        source_graph_id=torch.tensor(source_graph_id, dtype=torch.long),
        traj_len=traj_len,
        terminal_step=terminal_step,
        terminal_stop_log_prob=torch.zeros(num_rows, dtype=torch.float32),
        valid_mask=valid_mask,
        expand_mask=expand_mask,
        stop_mask=stop_mask,
        forced_stop_mask=forced_stop_mask,
        action_type=torch.zeros((num_rows, max_steps), dtype=torch.long),
        selected_edge_ids=selected_edge_ids,
        expand_budget=max_steps - 1,
    )


class _FeatureEncoderStub(nn.Module):
    def forward(self, batch) -> object:
        del batch
        return object()


class _PolicyStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))


def test_validation_step_logs_rollout_metrics_instead_of_objective_loss(monkeypatch) -> None:
    batch = _batch()
    rollouts = (
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 0, 1: 1}},
            stop_steps={0: 2},
            max_steps=3,
        ),
        _rollout_result(
            source_graph_id=[0],
            expand_steps={0: {0: 2}},
            forced_stop_steps={0: 1},
            max_steps=3,
        ),
    )

    module = WeaverModule(
        feature_encoder=_FeatureEncoderStub(),
        policy=_PolicyStub(),
        reward_model=EvidenceLogReward(),
        runner=SimpleNamespace(
            progress_fn=None,
            eval_num_rollout=2,
            engine=SimpleNamespace(prepare_context=lambda **_: object()),
            eval_rollouts=lambda **_: rollouts,
        ),
        optimization=OptimizationRuntimeConfig(
            optimizer=OptimizerRuntimeConfig(
                type="adamw",
                lr=1.0e-4,
                weight_decay=0.0,
                betas=(0.9, 0.999),
                no_decay_on_bias_and_norm=True,
            ),
            scheduler=None,
        ),
        evaluation=EvalRuntimeConfig(
            best_of_k_values=(1, 2, 4, 8),
            utility_k=8,
            utility_lambda=0.02,
            exclude_anchors_from_retrieved=True,
            use_reachable_targets=True,
        ),
    )

    logged: dict[str, float] = {}

    def _capture(name: str, value, **kwargs) -> None:
        del kwargs
        logged[name] = float(value.item()) if hasattr(value, "item") else float(value)

    monkeypatch.setattr(module, "log", _capture)

    module.validation_step(batch, batch_idx=0)

    assert "val/main/utility_at_8" in logged
    assert "val/best_of_k/target_recall_at_2" in logged
    assert "val/sample/target_f1_mean" in logged
    assert "val/objective_epoch_loss" not in logged
