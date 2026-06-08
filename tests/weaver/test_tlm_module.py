from __future__ import annotations

from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from src.data.schema import ReplayBankBatch, RetrievalBatch
from src.weaver.feature import FeatureEncoder, StateEncoder
from src.weaver.module import WeaverModule
from src.weaver.objectives.subtb.loss import ForwardLookingSubTBObjective
from src.weaver.policy import BackwardPolicy, BackwardScoringModel, FlowEstimator, ForwardPolicy, StateFlowHead
from src.weaver.reward import EvidenceStateScorer
from src.weaver.rollout.engine import RolloutEngine
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch


def _batch() -> RetrievalBatch:
    return RetrievalBatch(
        question_id=["q0"],
        question=["who"],
        answers=[["a"]],
        question_emb=torch.randn(1, 4),
        node_text=["n0", "n1", "n2"],
        node_entity_ids=["e0", "e1", "e2"],
        node_entity_catalog_ids=torch.tensor([0, 1, 2], dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        edge_relation_ids=["r0", "r1"],
        edge_relation_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_graph_ids=torch.tensor([0, 0], dtype=torch.long),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([2], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        reachable_target_max_distance=torch.tensor([2], dtype=torch.long),
        node_target_distance=torch.tensor([2, 1, 0], dtype=torch.long),
        edge_on_shortest_path=torch.tensor([True, True], dtype=torch.bool),
        batch=torch.tensor([0, 0, 0], dtype=torch.long),
        num_graphs_total=1,
        num_nodes_total=3,
        num_edges_total=2,
        replay_bank=ReplayBankBatch(
            edge_ids=torch.full((1, 1, 2), -1, dtype=torch.long),
            edge_count=torch.zeros((1, 1), dtype=torch.long),
            priority=torch.zeros((1, 1), dtype=torch.float32),
        ),
    )


def _feature_encoder() -> FeatureEncoder:
    return FeatureEncoder(
        entity_text_semantic_table=torch.randn(3, 4),
        text_row_by_entity_id=torch.tensor([0, 1, 2], dtype=torch.long),
        entity_relation_neighborhood_semantic_table=torch.randn(1, 4),
        relation_neighborhood_row_by_entity_id=torch.tensor([-1, -1, -1], dtype=torch.long),
        relation_semantic_table=torch.randn(2, 4),
        sem_dim=4,
        hidden_dim=4,
    )


def _module() -> WeaverModule:
    forward_feature_encoder = _feature_encoder()
    backward_feature_encoder = _feature_encoder()
    forward_policy = ForwardPolicy(
        state_encoder=StateEncoder(hidden_dim=4, num_heads=1),
        flow_estimator=FlowEstimator(hidden_dim=4),
        state_flow_head=StateFlowHead(state_dim=4),
    )
    backward_policy = BackwardScoringModel(
        state_encoder=StateEncoder(hidden_dim=4, num_heads=1),
        backward_policy=BackwardPolicy(hidden_dim=4),
    )
    return WeaverModule(
        budget=2,
        hidden_dim=4,
        forward_feature_encoder=forward_feature_encoder,
        backward_feature_encoder=backward_feature_encoder,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        reward_model=EvidenceStateScorer(budget=2),
        objective=ForwardLookingSubTBObjective(),
        runner=RolloutRunner(
            engine=RolloutEngine(),
            train_policy_rollouts=1,
            replay_source=None,
            eval_rollouts=1,
        ),
        optimization=OmegaConf.create(
            {
                "forward": {
                    "optimizer": {
                        "type": "adamw",
                        "lr": 1.0e-3,
                        "weight_decay": 0.0,
                        "betas": [0.9, 0.999],
                        "eps": 1.0e-8,
                        "no_decay_on_bias_and_norm": True,
                    },
                    "scheduler": None,
                },
                "backward": {
                    "optimizer": {
                        "type": "adamw",
                        "lr": 1.0e-3,
                        "weight_decay": 0.0,
                        "betas": [0.9, 0.999],
                        "eps": 1.0e-8,
                        "no_decay_on_bias_and_norm": True,
                    },
                    "scheduler": None,
                },
                "target_ema_decay": 0.5,
            }
        ),
        evaluation=OmegaConf.create(
            {
                "exclude_anchors_from_retrieved": True,
                "use_reachable_targets": True,
                "k_windows": [1],
                "enable_terminal_diagnostics": False,
                "diversity_edge_penalty": 0.0,
            }
        ),
    )


def test_tlm_training_step_updates_online_backward_and_target() -> None:
    torch.manual_seed(0)
    module = _module()
    batch = _batch()
    module._trainer = SimpleNamespace(max_steps=10, estimated_stepping_batches=10, max_epochs=1)
    optimizers = module.configure_optimizers()
    assert isinstance(optimizers, list)
    module.optimizers = lambda: optimizers
    module.lr_schedulers = lambda: []
    module.manual_backward = lambda loss: loss.backward()
    module.log = lambda *args, **kwargs: None
    module.log_dict = lambda *args, **kwargs: None
    module.runner.train_rollouts = lambda **kwargs: TrainRolloutBatch(
        trajectories=TrajectoryBatch(
            graph_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
            edge_logp=torch.zeros((1, 2), dtype=torch.float32),
            edge_count=torch.tensor([2], dtype=torch.long),
            stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
            stop_logp=torch.zeros(1, dtype=torch.float32),
            source=torch.tensor([False]),
        ),
        metrics={},
    )

    before_online = [param.detach().clone() for param in module.backward_policy.parameters()]
    before_target = [param.detach().clone() for param in module.backward_target.parameters()]

    module.training_step(batch, 0)

    after_online = list(module.backward_policy.parameters())
    after_target = list(module.backward_target.parameters())

    assert any(not torch.allclose(before, after.detach()) for before, after in zip(before_online, after_online, strict=True))
    assert any(not torch.allclose(before, after.detach()) for before, after in zip(before_target, after_target, strict=True))


def test_log_train_does_not_duplicate_tlm_loss_key() -> None:
    module = _module()
    batch = _batch()
    logged: list[str] = []
    logged_dicts: list[dict[str, float]] = []
    module.log = lambda name, *args, **kwargs: logged.append(name)
    module.log_dict = lambda metrics, *args, **kwargs: logged_dicts.append(dict(metrics))

    output = SimpleNamespace(
        loss=torch.tensor(1.0),
        detached_metrics=lambda: {"objective": 0.5},
    )
    rollout = TrainRolloutBatch(
        trajectories=TrajectoryBatch(
            graph_ids=torch.tensor([0], dtype=torch.long),
            edge_ids=torch.tensor([[0, 1]], dtype=torch.long),
            edge_logp=torch.zeros((1, 2), dtype=torch.float32),
            edge_count=torch.tensor([2], dtype=torch.long),
            stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
            stop_logp=torch.zeros(1, dtype=torch.float32),
            source=torch.tensor([False]),
        ),
        metrics={},
    )

    module._log_train(
        batch=batch,
        output=output,
        rollout=rollout,
        tlm_loss=torch.tensor(0.25),
        tlm_metrics={
            "train/tlm_loss": 0.25,
            "train/tlm_step_count": 2.0,
        },
    )

    assert logged.count("train/tlm_loss") == 1
    assert all("train/tlm_loss" not in metrics for metrics in logged_dicts)
