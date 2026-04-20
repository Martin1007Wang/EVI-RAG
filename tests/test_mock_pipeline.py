from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collate import RetrievalCollator
from src.models.losses import SubTrajectoryBalanceLoss
from src.models.modules.backbone import NBFBackbone
from src.models.modules.heads import (
    ActionHead,
    ExpandEdgeScorer,
    build_edge_scorer_inputs,
)
from src.models.gflownet import GFlowNetModule
from src.models.policy import Policy
from src.models.replay import TrajectoryTrace
from src.models.reward import RewardModel
from src.models.rollout.engine import RolloutEngine
from src.models.state import State
from src.models.teacher_guidance import TeacherGuidance
from src.utils.nn_utils import cosine_scores


class _DummyEmbeddingStore:
    def __init__(self, *, embedding_dim: int) -> None:
        entity_embeddings = torch.arange(17 * embedding_dim, dtype=torch.float32)
        relation_embeddings = torch.arange(8 * embedding_dim, dtype=torch.float32)

        self._entity_embeddings = entity_embeddings.view(17, embedding_dim) / 100.0
        self._entity_embeddings[0].zero_()
        self._relation_embeddings = relation_embeddings.view(8, embedding_dim) / 50.0

    def get_entity_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self._entity_embeddings.index_select(0, ids.long())

    def get_relation_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        return self._relation_embeddings.index_select(0, ids.long())


class _DummyDataResource:
    def __init__(self, *, embedding_dim: int) -> None:
        entity_embedding_map = torch.arange(16, dtype=torch.long) + 1
        cvt_mask = torch.zeros(16, dtype=torch.bool)

        for entity_id in (4, 8):
            entity_embedding_map[entity_id] = 0
            cvt_mask[entity_id] = True

        self.embedding_store = _DummyEmbeddingStore(embedding_dim=embedding_dim)
        self.entity_embedding_map = entity_embedding_map
        self.cvt_mask = cvt_mask


def _build_mock_batch(*, embedding_dim: int = 8):
    graph_a = Data(
        sample_id="mock/graph-a",
        num_nodes=5,
        edge_index=torch.tensor(
            [[0, 0, 1, 3], [1, 3, 2, 4]],
            dtype=torch.long,
        ),
        edge_relation_ids_global=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        node_entity_ids_global=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
        question_emb=torch.linspace(0.1, 0.8, embedding_dim, dtype=torch.float32),
        is_anchor_mask=torch.tensor([True, False, False, False, False]),
        is_target_mask=torch.tensor([False, False, True, False, False]),
        anchor_signed_distance=torch.tensor([0, 1, 2, 1, 2], dtype=torch.long),
        answer_entity_ids_global=torch.tensor([3], dtype=torch.long),
        positive_edge_mask=torch.tensor([True, False, True, False]),
        node_to_target_distance=torch.tensor([2, 1, 0, -1, -1], dtype=torch.long),
        shortest_suffix_count=torch.tensor(
            [1.0, 1.0, 1.0, 0.0, 0.0],
            dtype=torch.float32,
        ),
        bounded_suffix_count=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        max_path_length=torch.tensor(2, dtype=torch.long),
    )
    graph_b = Data(
        sample_id="mock/graph-b",
        num_nodes=3,
        edge_index=torch.tensor(
            [[0, 0], [1, 2]],
            dtype=torch.long,
        ),
        edge_relation_ids_global=torch.tensor([1, 4], dtype=torch.long),
        node_entity_ids_global=torch.tensor([6, 7, 8], dtype=torch.long),
        question_emb=torch.linspace(1.1, 1.8, embedding_dim, dtype=torch.float32),
        is_anchor_mask=torch.tensor([True, False, False]),
        is_target_mask=torch.tensor([False, True, False]),
        anchor_signed_distance=torch.tensor([0, 1, 1], dtype=torch.long),
        answer_entity_ids_global=torch.tensor([7], dtype=torch.long),
        positive_edge_mask=torch.tensor([True, False]),
        node_to_target_distance=torch.tensor([1, 0, -1], dtype=torch.long),
        shortest_suffix_count=torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        bounded_suffix_count=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        max_path_length=torch.tensor(1, dtype=torch.long),
    )

    data_resource = _DummyDataResource(embedding_dim=embedding_dim)
    batch = RetrievalCollator(data_resource)([graph_a, graph_b])
    return batch, data_resource


def _run_training_step_smoke(
    module: GFlowNetModule,
    *,
    batch,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    optimizer = module.configure_optimizers()
    if isinstance(optimizer, dict):
        optimizer = optimizer["optimizer"]

    captured_logs: dict[str, float] = {}

    module._trainer = SimpleNamespace(
        accumulate_grad_batches=1,
        num_training_batches=1,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        lr_scheduler_configs=[],
        is_global_zero=False,
        world_size=1,
    )
    module.optimizers = lambda: optimizer
    module.manual_backward = lambda loss: loss.backward()
    module.log_dict = lambda metrics, **kwargs: captured_logs.update(
        {
            name: float(value.detach().cpu().item())
            if torch.is_tensor(value)
            else float(value)
            for name, value in metrics.items()
        }
    )

    target_param = module.policy.backbone.nbf_layers[0].fwd_msg_mlp[0].weight
    before = target_param.detach().clone()
    module.train()
    module.training_step(batch, batch_idx=0)
    after = target_param.detach().clone()
    return captured_logs, before, after


def test_mock_retrieval_batch_drives_backbone_and_heads() -> None:
    torch.manual_seed(0)
    batch, _ = _build_mock_batch()

    assert batch.num_graphs == 2
    assert batch.question_emb.shape == (2, 8)
    assert batch.node_tokens.shape == (8, 8)
    assert batch.relation_tokens.shape == (6, 8)
    assert batch.non_text_node_mask.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert batch.is_cvt.tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert batch.edge_batch.tolist() == [0, 0, 0, 0, 1, 1]
    assert batch.edge_ptr.tolist() == [0, 4, 6]

    backbone = NBFBackbone(
        embedding_dim=8,
        hidden_dim=8,
        gnn_num_layers=2,
        gnn_dropout=0.0,
    )
    backbone.eval()
    backbone_out = backbone(batch)

    assert backbone_out.node_h.shape == (8, 8)
    assert backbone_out.rel_h.shape == (6, 8)
    assert backbone_out.query_h.shape == (2, 8)
    assert backbone_out.edge_state_ids.tolist() == [0, 0, 0, 0, 0, 0]
    assert torch.isfinite(backbone_out.node_h).all()
    assert torch.isfinite(backbone_out.rel_h).all()
    assert torch.isfinite(backbone_out.query_h).all()

    edge_scorer = ExpandEdgeScorer(hidden_dim=8)
    edge_inputs = build_edge_scorer_inputs(
        backbone_out=backbone_out,
        edge_index=batch.edge_index,
        edge_batch_index=batch.edge_batch,
    )
    edge_scores = edge_scorer(edge_inputs)

    assert edge_scores.shape == (batch.edge_index.size(1),)
    assert torch.isfinite(edge_scores).all()
    assert float(edge_scores.std(unbiased=False)) > 0.0

    action_head = ActionHead(
        hidden_dim=8,
        zero_init_type_output=False,
        dropout=0.0,
    )
    type_logits = action_head(state_h=backbone_out.query_h)["type_logits"]

    assert type_logits.shape == (batch.num_graphs, 2)
    assert torch.isfinite(type_logits).all()
    assert float(type_logits.std(unbiased=False)) > 0.0


def test_expand_edge_scorer_starts_as_relation_only_baseline() -> None:
    batch, _ = _build_mock_batch()
    backbone = NBFBackbone(
        embedding_dim=8,
        hidden_dim=8,
        gnn_num_layers=2,
        gnn_dropout=0.0,
    )
    backbone.eval()
    backbone_out = backbone(batch)

    edge_scorer = ExpandEdgeScorer(hidden_dim=8)
    edge_inputs = build_edge_scorer_inputs(
        backbone_out=backbone_out,
        edge_index=batch.edge_index,
        edge_batch_index=batch.edge_batch,
    )

    breakdown = edge_scorer(edge_inputs, return_breakdown=True)
    expected_scores = edge_scorer.prior_scale * cosine_scores(
        backbone_out.query_h.index_select(0, batch.edge_batch),
        backbone_out.rel_h,
    )

    assert torch.allclose(breakdown.relation_only_logits, expected_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(breakdown.residual_logits, torch.zeros_like(expected_scores))
    assert torch.allclose(breakdown.final_logits, expected_scores, atol=1e-6, rtol=1e-6)


def test_expand_edge_scorer_breakdown_matches_final_logits() -> None:
    batch, _ = _build_mock_batch()
    backbone = NBFBackbone(
        embedding_dim=8,
        hidden_dim=8,
        gnn_num_layers=2,
        gnn_dropout=0.0,
    )
    backbone.eval()
    backbone_out = backbone(batch)

    edge_scorer = ExpandEdgeScorer(hidden_dim=8)
    edge_inputs = build_edge_scorer_inputs(
        backbone_out=backbone_out,
        edge_index=batch.edge_index,
        edge_batch_index=batch.edge_batch,
    )

    breakdown = edge_scorer(edge_inputs, return_breakdown=True)
    assert breakdown.final_logits.shape == (batch.edge_index.size(1),)
    assert torch.allclose(
        breakdown.final_logits,
        breakdown.relation_only_logits + breakdown.residual_logits,
        atol=1e-6,
        rtol=1e-6,
    )


def test_expand_edge_scorer_can_freeze_prior_scale() -> None:
    edge_scorer = ExpandEdgeScorer(hidden_dim=8, prior_scale_trainable=False)

    assert edge_scorer.prior_scale.requires_grad is False


def test_teacher_guidance_forces_first_teacher_edge() -> None:
    torch.manual_seed(0)
    batch, _ = _build_mock_batch()
    policy = Policy(
        backbone_cfg={
            "embedding_dim": 8,
            "hidden_dim": 8,
            "gnn_num_layers": 1,
            "gnn_dropout": 0.0,
        },
        hidden_dim=8,
        max_steps=2,
        action_head_cfg={"dropout": 0.0},
    )
    reward_model = RewardModel(relation_shaping_scale=1.0)
    rollout_engine = RolloutEngine(max_steps=2)
    teacher_guidance = TeacherGuidance(
        mode="bounded_path",
        score_exponent=0.5,
        undirected=False,
        fallback_to_policy=True,
    )

    rollout = rollout_engine.run_exploration(
        policy=policy,
        base_graph=batch,
        reward_model=reward_model,
        num_rollouts=1,
        temperature=0.7,
        collect_terminal_state=False,
        teacher_guidance=teacher_guidance,
        teacher_force_prob=1.0,
    )[0]

    assert rollout.selected_edge_ids is not None
    first_edges = rollout.selected_edge_ids[:, 0]
    positive_first = batch.positive_edge_mask.index_select(0, first_edges)
    assert positive_first.tolist() == [True, True]
    assert rollout.trajectory_traces is not None
    assert [trace.source for trace in rollout.trajectory_traces] == ["teacher", "teacher"]


def test_reward_step_shaping_uses_frontier_relation_potential() -> None:
    batch, _ = _build_mock_batch()
    backbone = NBFBackbone(
        embedding_dim=8,
        hidden_dim=8,
        gnn_num_layers=0,
        gnn_dropout=0.0,
    )
    backbone.eval()
    feature_bank = backbone.project(batch)
    reward_model = RewardModel(relation_shaping_scale=2.0)

    state = State.create_initial(batch)
    edges_before = state.active_edges.clone()
    state.apply_expansion(
        chosen_edges=torch.tensor([0, 4], dtype=torch.long),
        src=batch.edge_index[0],
        dst=batch.edge_index[1],
    )
    edges_after = state.active_edges.clone()

    shaping = reward_model.step_shaping(
        batch,
        edges_before,
        edges_after,
        query_h=feature_bank.query_h,
        rel_h=feature_bank.rel_h,
    )

    def _phi(active_edges: torch.Tensor) -> torch.Tensor:
        src = batch.edge_index[0]
        dst = batch.edge_index[1]
        active_nodes = batch.is_anchor_mask.clone()
        if bool(active_edges.any()):
            active_nodes[src[active_edges]] = True
            active_nodes[dst[active_edges]] = True
        frontier = (~active_edges) & (active_nodes[src] | active_nodes[dst])
        expected = torch.zeros(batch.num_graphs, dtype=torch.float32)
        for graph_id in range(batch.num_graphs):
            graph_frontier = frontier & (batch.edge_batch == graph_id)
            if not bool(graph_frontier.any()):
                continue
            edge_ids = torch.nonzero(graph_frontier, as_tuple=False).view(-1)
            graph_scores = cosine_scores(
                feature_bank.query_h[graph_id].expand(edge_ids.numel(), -1),
                feature_bank.rel_h.index_select(0, edge_ids),
            )
            expected[graph_id] = graph_scores.max()
        return expected

    expected_shaping = _phi(edges_after) - _phi(edges_before)
    assert torch.allclose(shaping, expected_shaping, atol=1e-6, rtol=1e-6)


def test_reward_forward_uses_log_recall_with_floor() -> None:
    batch, _ = _build_mock_batch()
    reward_model = RewardModel(log_r_min=-5.0, semantic_fallback_scale=0.0)

    active_nodes = batch.is_anchor_mask.clone()
    active_nodes[2] = True
    active_edges = torch.zeros(batch.edge_index.size(1), dtype=torch.bool)

    log_reward = reward_model(batch, active_nodes, active_edges)

    assert torch.allclose(log_reward, torch.tensor([0.0, -5.0]), atol=1e-6, rtol=1e-6)


def test_reward_forward_adds_semantic_fallback_for_zero_recall() -> None:
    batch, _ = _build_mock_batch(embedding_dim=2)
    reward_model = RewardModel(log_r_min=-5.0, semantic_fallback_scale=0.25)

    batch.node_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.70710677, 0.70710677],
        ],
        dtype=torch.float32,
    )
    batch.non_text_node_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)

    active_nodes = batch.is_anchor_mask.clone()
    active_nodes[7] = True
    active_edges = torch.zeros(batch.edge_index.size(1), dtype=torch.bool)

    log_reward = reward_model(batch, active_nodes, active_edges)

    expected_bonus_graph_b = 0.25 * cosine_scores(
        batch.node_tokens[7].view(1, -1),
        batch.node_tokens[6].view(1, -1),
    ).item()
    expected = torch.tensor([-5.0, -5.0 + expected_bonus_graph_b], dtype=torch.float32)
    assert torch.allclose(log_reward, expected, atol=1e-6, rtol=1e-6)


def test_mock_rollout_loss_and_backward_produce_signal() -> None:
    torch.manual_seed(0)
    batch, _ = _build_mock_batch()

    policy = Policy(
        backbone_cfg={
            "embedding_dim": 8,
            "hidden_dim": 8,
            "gnn_num_layers": 2,
            "gnn_dropout": 0.0,
        },
        hidden_dim=8,
        max_steps=2,
        action_head_cfg={
            "dropout": 0.0,
        },
    )
    reward_model = RewardModel(relation_shaping_scale=1.0)
    rollout_engine = RolloutEngine(max_steps=2)
    traces = (
        TrajectoryTrace(
            sample_id="mock/graph-a",
            edge_trace_local=(0, 2),
            traj_len=3,
            terminal_log_reward=1.0,
            priority=1.0,
            insert_step=0,
        ),
        TrajectoryTrace(
            sample_id="mock/graph-b",
            edge_trace_local=(0,),
            traj_len=2,
            terminal_log_reward=1.0,
            priority=1.0,
            insert_step=0,
        ),
    )

    rollout = rollout_engine.replay_trajectories(
        policy=policy,
        base_graph=batch,
        reward_model=reward_model,
        traces=traces,
        collect_terminal_state=True,
    )

    assert rollout.traj_len.tolist() == [3, 2]
    assert rollout.trajectory_traces is not None
    assert [trace.sample_id for trace in rollout.trajectory_traces] == [
        "mock/graph-a",
        "mock/graph-b",
    ]
    assert torch.isfinite(rollout.root_log_z).all()
    assert torch.isfinite(rollout.trajectory_log_pf).all()
    assert torch.isfinite(rollout.trajectory_log_pb).all()
    assert torch.isfinite(rollout.terminal_log_rewards).all()
    assert torch.all(rollout.terminal_log_rewards >= 0.0)
    assert rollout.terminal_active_nodes is not None
    assert rollout.terminal_active_edges is not None
    assert rollout.terminal_active_nodes[2].item() is True
    assert rollout.terminal_active_nodes[6].item() is True
    assert rollout.terminal_active_edges[0].item() is True
    assert rollout.terminal_active_edges[2].item() is True
    assert rollout.terminal_active_edges[4].item() is True

    loss_fn = SubTrajectoryBalanceLoss(
        max_trajectory_len=3,
        reward_matching_coef=0.5,
    )
    policy.zero_grad(set_to_none=True)
    loss_out = loss_fn(rollout)
    loss_out.loss.backward()

    assert torch.isfinite(loss_out.loss)
    assert loss_out.loss.item() > 0.0
    assert loss_out.metric("log_reward_mean").item() >= 0.0

    grad_sums = {
        name: float(param.grad.abs().sum().item())
        for name, param in policy.named_parameters()
        if param.grad is not None
    }

    assert grad_sums["backbone.nbf_layers.0.fwd_msg_mlp.0.weight"] > 0.0
    assert grad_sums["expand_edge_scorer.prior_scale"] > 0.0
    assert grad_sums["action_head.type_scorer.2.weight"] > 0.0
    assert grad_sums["z_head.q_proj.weight"] > 0.0


def test_gflownet_training_step_smoke_on_mock_batch() -> None:
    torch.manual_seed(0)
    batch, _ = _build_mock_batch()

    module = GFlowNetModule(
        max_steps=2,
        num_rollout=1,
        eval_num_rollout=1,
        rollout_chunk_size=1,
        eval_rollout_chunk_size=1,
        temperature=0.7,
        backbone={
            "embedding_dim": 8,
            "hidden_dim": 8,
            "gnn_num_layers": 2,
            "gnn_dropout": 0.0,
        },
        policy_hidden_dim=8,
        action_head={
            "dropout": 0.0,
        },
        reward={
            "relation_shaping_scale": 0.0,
        },
        loss={
            "reward_matching_coef": 0.5,
        },
        replay={
            "enabled": False,
        },
        optimizer_cfg={
            "type": "adamw",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
            "log_z_head_lr_multiplier": 1.0,
        },
        scheduler_cfg=None,
    )

    logs, before, after = _run_training_step_smoke(module, batch=batch)

    assert logs["train/loss"] > 0.0
    assert torch.isfinite(torch.tensor(logs["train/log_reward_mean"]))
    assert logs["train/trajectory_length_mean"] >= 1.0
    assert abs(logs["train/lr"] - 1.0e-3) < 1.0e-8
    assert not torch.equal(before, after)
