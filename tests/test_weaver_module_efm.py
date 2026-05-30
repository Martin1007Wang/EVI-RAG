from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from src.data.schema import ReplayProgramBatch
from src.weaver.context import GraphContext, ReplayContext, TargetContext
from src.weaver.module import WeaverModule
from src.weaver.objectives import ObjectiveOutput
from src.weaver.objectives.transition_batch import (
    NonterminalTransitionBatch,
    TransitionSource,
)
from src.weaver.rollout.runner import RolloutRunner, TrainRolloutBatch, _policy_prefix_states
from src.weaver.rollout.trajectory import POLICY_STOP, TrajectoryBatch
from src.weaver.state import ExpansionBatch, StateBatch


def test_weaver_module_training_step_uses_edge_flow_matching_batches() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)
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

    module._build_inputs = lambda _: SimpleNamespace(graph=graph, target=target, replay=replay, features=features)  # type: ignore[method-assign]
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
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)
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
        graph_context=graph,
    )
    replay = NonterminalTransitionBatch(
        parent_state=root,
        parent_state_ids=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        child_state=root.branch(
            ExpansionBatch(
                state_ids=torch.tensor([0], dtype=torch.long),
                edge_ids=torch.tensor([0], dtype=torch.long),
            ),
            graph_context=graph,
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

    module._build_inputs = lambda _: SimpleNamespace(graph=graph, target=target, replay=replay_context, features=features)  # type: ignore[method-assign]
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]

    loss = module.training_step(batch, 0)

    assert torch.allclose(loss, torch.tensor(2.5))
    assert module.objective.seen_nonterminal is not None
    assert module.objective.seen_nonterminal.edge_ids.tolist() == [0]
    assert module.objective.seen_nonterminal.source.tolist() == [int(TransitionSource.WEAK_REPLAY)]
    assert module.objective.seen_terminal is not None
    assert module.objective.seen_terminal.source.tolist() == [0, int(TransitionSource.WEAK_REPLAY), int(TransitionSource.WEAK_REPLAY)]


def test_policy_prefix_states_preserve_visitation_order() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[0, -1], [-1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((2, 2), dtype=torch.float32),
        edge_count=torch.tensor([1, 0], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(2, dtype=torch.float32),
        source=torch.zeros(2, dtype=torch.bool),
    )

    prefix = _policy_prefix_states(
        trajectories=trajectories,
        graph_context=graph,
    )

    assert prefix.num_states == 1
    assert prefix.edge_count.tolist() == [0]
    assert prefix.graph_ids.tolist() == [0]


def test_rollout_runner_collects_replay_from_policy_prefix_states() -> None:
    batch = _batch()
    graph = GraphContext.from_batch(batch)
    target = TargetContext.from_batch(batch=batch, graph_context=graph)
    replay_context = ReplayContext.from_batch(batch=batch, graph_context=graph, target_context=target)
    features = SimpleNamespace()
    trajectories = TrajectoryBatch(
        graph_ids=torch.tensor([0, 0], dtype=torch.long),
        edge_ids=torch.tensor([[0, -1], [-1, -1]], dtype=torch.long),
        edge_logp=torch.zeros((2, 2), dtype=torch.float32),
        edge_count=torch.tensor([1, 0], dtype=torch.long),
        stop_reason=torch.tensor([POLICY_STOP, POLICY_STOP], dtype=torch.uint8),
        stop_logp=torch.zeros(2, dtype=torch.float32),
        source=torch.zeros(2, dtype=torch.bool),
    )
    replay_source = _ReplaySourceRecorder()
    runner = RolloutRunner(
        engine=_EngineStub(trajectories=trajectories),
        train_policy_rollouts=2,
        replay_source=replay_source,
        eval_rollouts=1,
    )

    rollout = runner.train_rollouts(
        policy=_PolicyStub(),
        context=graph,
        target_context=target,
        replay_context=replay_context,
        features=features,
        budget=2,
    )

    assert rollout.trajectories is trajectories
    assert replay_source.seen_initial_state is not None
    assert replay_source.seen_initial_state.num_states == 1
    assert replay_source.seen_initial_state.edge_count.tolist() == [0]
    assert replay_source.seen_initial_state.graph_ids.tolist() == [0]
    assert replay_source.seen_initial_state.selected_edge_ids.tolist() == [[-1, -1]]


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

    def train_rollouts(self, *, policy, context, target_context, replay_context, features, budget):
        del policy, context, target_context, replay_context, features, budget
        return self.rollout

    def eval_rollouts(self, *, policy, context, features, budget, num_rollouts=None):
        del policy, context, features, budget, num_rollouts
        return self.rollout.trajectories


class _EngineStub:
    def __init__(self, *, trajectories: TrajectoryBatch) -> None:
        self.trajectories = trajectories

    def sample(self, *, policy, context, features, graph_ids, budget):
        del policy, context, features, graph_ids, budget
        return self.trajectories


class _ReplaySourceRecorder:
    def __init__(self) -> None:
        self.seen_initial_state = None

    def collect(self, *, graph_context, target_context, replay_context, initial_state):
        del graph_context, target_context, replay_context
        self.seen_initial_state = initial_state
        return SimpleNamespace(nonterminal=None, stats=None)


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
    replay_program: ReplayProgramBatch
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
        replay_program=ReplayProgramBatch(
            candidate_edge_ids=torch.empty(0, dtype=torch.long),
            candidate_ptr=torch.zeros(1, dtype=torch.long),
            candidate_target_positions=torch.empty(0, dtype=torch.long),
            candidate_target_ptr=torch.zeros(1, dtype=torch.long),
            edge_to_candidate_ids=torch.empty(0, dtype=torch.long),
            edge_to_candidate_ptr=torch.zeros(2, dtype=torch.long),
            candidate_graph_ptr=torch.zeros(2, dtype=torch.long),
            path_truncated_by_graph=torch.zeros(1, dtype=torch.long),
        ),
    )
