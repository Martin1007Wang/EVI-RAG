from __future__ import annotations

import torch

from src.models.configs.policy import (
    BackboneConfig,
    FlowHeadConfig,
    PolicyConfig,
    PriorityHeadConfig,
)
from src.models.configs.training import OptimizerConfig, SchedulerConfig
from src.models.configs.trajectory_gfn import (
    HorizonConfig,
    TrajectoryInferenceConfig,
    TrajectoryTrainingConfig,
)
from src.models.trajectory_gfn.module import TrajectoryGFlowNetModule

from .conftest import make_batch_from_graph


def _make_module() -> TrajectoryGFlowNetModule:
    return TrajectoryGFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2, min_stop_steps=1),
        policy_cfg=PolicyConfig(
            backbone=BackboneConfig(
                embedding_dim=8,
                hidden_dim=8,
                gnn_layers=1,
                gnn_dropout=0.0,
                use_adapter=True,
                adapter_dim=4,
                adapter_dropout=0.0,
            ),
            flow_head=FlowHeadConfig(hidden_dim=16, dropout=0.0, relation_low_rank=2),
            priority_head=PriorityHeadConfig(
                hidden_dim=8,
                num_layers=2,
                dropout=0.0,
            ),
            stop_bias_init=-0.5,
            stop_delta_scale=2.0,
            stop_delta_temperature=1.0,
            doob_h_alpha=0.0,
            doob_h_node_temperature=1.0,
        ),
        training_cfg=TrajectoryTrainingConfig(rollout_batch_size=1),
        inference_cfg=TrajectoryInferenceConfig(
            mode="sampled",
            answer_mass_threshold=0.9,
            support_mass_threshold=0.9,
            rollout_chunk_size=1,
            max_rollouts=2,
            answer_top_ks=(1, 5),
            max_expansions=32,
            max_frontier_size=32,
        ),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
    )


def test_predict_step_emits_placeholder_for_invalid_start_support() -> None:
    torch.manual_seed(31)
    batch = make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([1], dtype=torch.long),
        a_local_indices=torch.empty((0,), dtype=torch.long),
        answer_entity_ids=torch.empty((0,), dtype=torch.long),
        node_global_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="invalid-start-support",
    )
    module = _make_module()

    module.on_predict_epoch_start()
    outputs = module.predict_step(batch, batch_idx=0)
    module.on_predict_batch_end(outputs, batch, batch_idx=0)
    module.on_predict_epoch_end()

    assert len(outputs) == 1
    result = outputs[0]
    assert result.sample_id == "invalid-start-support"
    assert result.stop_reason == "invalid_start_support"
    assert result.inference_mode == "sampled"
    assert result.window_size == 0
    assert result.covered_mass == 0.0
    assert result.tail_rollout_mass == 1.0
    assert result.start_entity_ids == [101]
    assert result.answer_posterior == []
    assert module.predict_metrics["invalid_start_count"] == 1
    assert module.predict_metrics["invalid_start_rate"] == 1.0
    assert module.predict_metrics["num_samples"] == 1.0
    assert module.predict_labels[0].sample_id == "invalid-start-support"
    assert module.predict_labels[0].start_entity_ids == [101]
