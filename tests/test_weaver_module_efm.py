from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from src.weaver.context import GraphContext, TargetContext
from src.weaver.module import WeaverModule
from src.weaver.objectives import ObjectiveOutput
from src.weaver.objectives.transition_batch import (
    NonterminalTransitionBatch,
    TransitionSource,
)
from src.weaver.rollout.runner import TrainRolloutBatch
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch


def test_weaver_module_training_step_uses_edge_flow_matching_batches() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    features = SimpleNamespace()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[0, -1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 2), dtype=torch.float32),
        edge_count=torch.tensor([1], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )

    module = WeaverModule(
        budget=2,
        hidden_dim=4,
        feature_encoder=_FeatureEncoderStub(features=features),
        policy=_PolicyStub(),
        reward_model=_RewardStub(),
        objective=_ObjectiveStub(),
        runner=_RunnerStub(
            rollout=TrainRolloutBatch(
                trajectories=trajectories,
                replay_transitions=None,
                metrics={},
            )
        ),
        optimization=SimpleNamespace(),
        evaluation=SimpleNamespace(
            exclude_anchors_from_retrieved=True,
            use_reachable_targets=True,
            k_windows=(1,),
            enable_terminal_diagnostics=False,
        ),
    )

    module._build_inputs = lambda _: SimpleNamespace(graph=graph, target=target, features=features)  # type: ignore[method-assign]
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    loss = module.training_step(batch, 0)

    assert torch.allclose(loss, torch.tensor(2.5))
    assert module.objective.seen_nonterminal is not None
    assert module.objective.seen_terminal is not None
    assert module.objective.seen_nonterminal.edge_ids.tolist() == [0]
    assert module.objective.seen_terminal.state.edge_count.tolist() == [0, 1]


def test_weaver_module_training_step_merges_replay_transitions() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    features = SimpleNamespace()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([[-1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((1, 2), dtype=torch.float32),
        edge_count=torch.tensor([0], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(1, dtype=torch.float32),
        source=torch.zeros(1, dtype=torch.bool),
    )
    root = StateBatch.initial(
        graph_ids=torch.tensor([0], dtype=torch.long),
        budget=2,
    )
    replay = NonterminalTransitionBatch(
        parent_state=root,
        parent_state_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        child_state=root.branch(
            ExpansionBatch(
                state_ids=torch.tensor([0], dtype=torch.long),
                edge_ids=torch.tensor([0], dtype=torch.long),
            )
        ),
        source=torch.tensor([int(TransitionSource.WEAK_REPLAY)], dtype=torch.long),
    )

    module = WeaverModule(
        budget=2,
        hidden_dim=4,
        feature_encoder=_FeatureEncoderStub(features=features),
        policy=_PolicyStub(),
        reward_model=_RewardStub(),
        objective=_ObjectiveStub(),
        runner=_RunnerStub(
            rollout=TrainRolloutBatch(
                trajectories=trajectories,
                replay_transitions=replay,
                metrics={},
            )
        ),
        optimization=SimpleNamespace(),
        evaluation=SimpleNamespace(
            exclude_anchors_from_retrieved=True,
            use_reachable_targets=True,
            k_windows=(1,),
            enable_terminal_diagnostics=False,
        ),
    )

    module._build_inputs = lambda _: SimpleNamespace(graph=graph, target=target, features=features)  # type: ignore[method-assign]
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    loss = module.training_step(batch, 0)

    assert torch.allclose(loss, torch.tensor(2.5))
    assert module.objective.seen_nonterminal is not None
    assert module.objective.seen_nonterminal.edge_ids.tolist() == [0]
    assert module.objective.seen_nonterminal.source.tolist() == [int(TransitionSource.WEAK_REPLAY)]
    assert module.objective.seen_terminal is not None
    assert module.objective.seen_terminal.source.tolist() == [0, int(TransitionSource.WEAK_REPLAY), int(TransitionSource.WEAK_REPLAY)]


class _FeatureEncoderStub(torch.nn.Module):
    def __init__(self, *, features: object) -> None:
        super().__init__()
        self.features = features

    def forward(self, batch):
        del batch
        return self.features


class _PolicyStub(torch.nn.Module):
    pass


class _RewardStub(torch.nn.Module):
    pass


class _ObjectiveStub(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_nonterminal = None
        self.seen_terminal = None

    def forward(
        self,
        *,
        policy,
        reward_model,
        features,
        graph_context,
        target_context,
        nonterminal,
        terminal,
    ) -> ObjectiveOutput:
        del policy, reward_model, features, graph_context, target_context
        self.seen_nonterminal = nonterminal
        self.seen_terminal = terminal
        return ObjectiveOutput(
            loss=torch.tensor(2.5),
            metrics={},
            num_states=1,
        )


class _RunnerStub:
    def __init__(self, *, rollout: TrainRolloutBatch) -> None:
        self.rollout = rollout

    def train_rollouts(self, *, policy, context, target_context, features, budget):
        del policy, context, target_context, features, budget
        return self.rollout

    def eval_rollouts(self, *, policy, context, features, budget, num_rollouts=None):
        del policy, context, features, budget, num_rollouts
        return self.rollout.trajectories


@dataclass
class _Batch:
    edge_index: torch.Tensor
    batch: torch.Tensor
    ptr: torch.Tensor
    num_nodes: int
    num_graphs: int
    node_entity_catalog_ids: torch.Tensor
    edge_relation_catalog_ids: torch.Tensor
    question_emb: torch.Tensor
    anchor_node_ids: torch.Tensor
    target_node_ids: torch.Tensor
    reachable_target_node_ids: torch.Tensor
    node_target_distance: torch.Tensor
    weak_replay_edge_ids: torch.Tensor
    weak_replay_edge_ids_batch: torch.Tensor
    weak_replay_edge_weight: torch.Tensor
    witness_path_edge_ids: torch.Tensor
    witness_path_edge_ids_batch: torch.Tensor
    witness_path_edge_path_ids: torch.Tensor
    witness_path_target_node_ids: torch.Tensor

    @property
    def num_nodes_total(self) -> int:
        return int(self.num_nodes)

    @property
    def num_edges_total(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def num_graphs_total(self) -> int:
        return int(self.num_graphs)


def _batch() -> _Batch:
    return _Batch(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        batch=torch.tensor([0, 0], dtype=torch.long),
        ptr=torch.tensor([0, 2], dtype=torch.long),
        num_nodes=2,
        num_graphs=1,
        node_entity_catalog_ids=torch.tensor([0, 1], dtype=torch.long),
        edge_relation_catalog_ids=torch.tensor([0], dtype=torch.long),
        question_emb=torch.ones((1, 4), dtype=torch.float32),
        anchor_node_ids=torch.tensor([0], dtype=torch.long),
        target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
        node_target_distance=torch.tensor([1, 0], dtype=torch.long),
        weak_replay_edge_ids=torch.empty(0, dtype=torch.long),
        weak_replay_edge_ids_batch=torch.empty(0, dtype=torch.long),
        weak_replay_edge_weight=torch.empty(0, dtype=torch.float32),
        witness_path_edge_ids=torch.empty(0, dtype=torch.long),
        witness_path_edge_ids_batch=torch.empty(0, dtype=torch.long),
        witness_path_edge_path_ids=torch.empty(0, dtype=torch.long),
        witness_path_target_node_ids=torch.empty(0, dtype=torch.long),
    )
