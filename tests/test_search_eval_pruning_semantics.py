from __future__ import annotations

import pytest
import torch

from src.graph import TrajectoryBatch
from src.metrics.search_eval_utils import (
    format_search_eval_answer_posterior,
    normalize_search_eval_cfg,
)
from src.subgraph_gflownet.application.evaluation.answer_search_runtime import (
    SubgraphAnswerSearchRuntime,
)
from src.subgraph_gflownet.core.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from src.subgraph_gflownet.core.sampler import (
    SubgraphSampler,
    SubgraphTrajectorySampleBatch,
)
from src.subgraph_gflownet.core.subgraph_batch import SubgraphBatchBuildOptions


def _make_subgraph_policy(*, max_steps: int = 2) -> SubgraphPolicy:
    return SubgraphPolicy(
        state_mode=SUBGRAPH_STATE_MODE,
        backbone={
            "embedding_dim": 8,
            "hidden_dim": 8,
            "use_adapter": True,
            "adapter_dim": 4,
            "adapter_dropout": 0.0,
        },
        flow_head={
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
            "conditioning": "concat",
        },
        state_encoder={
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
        },
        actor={
            "hidden_dim": 8,
            "num_layers": 2,
            "dropout": 0.0,
        },
        answer_reward={
            "hit_bonus": 5.0,
            "frontier_bonus": 1.0,
            "coverage_bonus": 0.2,
            "size_penalty": 0.1,
            "component_penalty": 0.5,
        },
        proposal_prior={
            "oracle_answer_distance_weight": 0.0,
            "prior_question_similarity_weight": 0.0,
        },
        max_steps=max_steps,
    )


def _make_bridge_batch() -> TrajectoryBatch:
    emb_dim = 8
    edge_index = torch.tensor([[0, 2], [1, 1]], dtype=torch.long)
    return TrajectoryBatch(
        num_graphs=1,
        node_ptr=torch.tensor([0, 3], dtype=torch.long),
        edge_index=edge_index,
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        edge_batch=torch.zeros((edge_index.size(1),), dtype=torch.long),
        node_batch=torch.zeros((3,), dtype=torch.long),
        node_embeddings=torch.randn(3, emb_dim),
        edge_embeddings=torch.randn(edge_index.size(1), emb_dim),
        question_emb=torch.randn(1, emb_dim),
        question_ctx=torch.randn(1, 2, emb_dim),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        anchor_local_indices=torch.tensor([0, 2], dtype=torch.long),
        anchor_ptr=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_ids=["bridge-subgraph"],
        questions=["toy question"],
        dataset_scope="sub",
    )


def test_normalize_search_eval_cfg_allows_disabling_eval_action_pruning() -> None:
    cfg = normalize_search_eval_cfg(
        {"monte_carlo": {"action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0}}}
    )

    assert cfg["monte_carlo"]["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }
    assert "prune=off" in format_search_eval_answer_posterior(cfg)


def test_normalize_search_eval_cfg_canonicalizes_answer_aggregation_backend() -> None:
    cfg = normalize_search_eval_cfg(
        {
            "monte_carlo": {
                "answer_aggregation": {"backend": "reward"},
                "early_stop": {
                    "enabled": True,
                    "min_rollouts": 32,
                    "stability_top_k": 1,
                },
            }
        }
    )

    assert cfg["monte_carlo"]["answer_aggregation"] == {
        "backend": "terminal_reward",
    }
    assert cfg["monte_carlo"]["early_stop"]["enabled"] is False
    assert "aggregation=terminal_reward" in format_search_eval_answer_posterior(cfg)


def test_normalize_search_eval_cfg_rejects_negative_eval_action_pruning() -> None:
    with pytest.raises(
        ValueError,
        match="evaluation\.monte_carlo\.action_pruning\.per_node_top_k",
    ):
        normalize_search_eval_cfg(
            {"monte_carlo": {"action_pruning": {"per_node_top_k": -1}}}
        )
    with pytest.raises(
        ValueError,
        match="evaluation\.monte_carlo\.action_pruning\.per_state_top_k",
    ):
        normalize_search_eval_cfg(
            {"monte_carlo": {"action_pruning": {"per_state_top_k": -1}}}
        )


def test_runtime_passes_disabled_eval_action_pruning_to_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _make_bridge_batch()
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=normalize_search_eval_cfg(
            {
                "report_profile": "rank_only",
                "monte_carlo": {
                    "rollouts": 2,
                    "batch_rollouts": 2,
                    "confidence": 0.95,
                    "action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0},
                },
            }
        ),
        policy=_make_subgraph_policy(max_steps=2),
        sampler=SubgraphSampler(max_steps=2),
    )
    seen_calls: list[dict[str, object]] = []
    seen_build_options: list[SubgraphBatchBuildOptions | None] = []

    original_prepare_batch = runtime.policy.prepare_batch

    def _tracked_prepare_batch(
        batch_arg: TrajectoryBatch,
        *,
        build_options: SubgraphBatchBuildOptions | None = None,
    ):
        seen_build_options.append(build_options)
        return original_prepare_batch(batch_arg, build_options=build_options)

    def _fake_sample(**kwargs: object) -> SubgraphTrajectorySampleBatch:
        seen_calls.append(dict(kwargs))
        return SubgraphTrajectorySampleBatch(
            state_log_flows=torch.zeros((1, 2, 3), dtype=torch.float32),
            log_pf_actions=torch.zeros((1, 2, 3), dtype=torch.float32),
            log_pb_actions=torch.zeros((1, 2, 3), dtype=torch.float32),
            log_reward_actions=torch.zeros((1, 2, 3), dtype=torch.float32),
            action_mask=torch.zeros((1, 2, 3), dtype=torch.bool),
            termination_action_steps=torch.tensor([[3, 3]], dtype=torch.long),
            chosen_edge_ids=torch.full((1, 2, 2), -1, dtype=torch.long),
            stop_actions=torch.zeros((1, 2, 3), dtype=torch.bool),
            terminal_answer_candidate_counts=torch.tensor([[0, 0]], dtype=torch.long),
            terminal_gold_answer_counts=torch.tensor([[0, 0]], dtype=torch.long),
            terminal_hit_mask=torch.tensor([[False, False]], dtype=torch.bool),
            terminal_component_counts=torch.tensor([[2, 2]], dtype=torch.long),
            terminal_edge_ids=((0,), (0,)),
            terminal_node_ids=((0, 2), (0, 2)),
            terminal_reachability_bits=({0: 1, 2: 2}, {0: 1, 2: 2}),
            terminal_answer_set_entity_ids=((), ()),
            sample_ids=("bridge-subgraph",),
            question_ids=("bridge-subgraph",),
            num_graphs=1,
            num_rollouts=2,
        )

    monkeypatch.setattr(runtime.policy, "prepare_batch", _tracked_prepare_batch)
    monkeypatch.setattr(runtime.sampler, "sample", _fake_sample)

    result = runtime._predict_single_graph(batch=batch, include_answer_support=False)

    assert seen_build_options == [
        SubgraphBatchBuildOptions(
            include_edge_question_similarity=False,
            include_oracle_distance=False,
            include_teacher_banks=False,
        )
    ]
    assert seen_calls[0]["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }
    assert result["requested_rollout_count"] == 2
    assert result["rollout_count"] == 2


def test_runtime_enables_edge_similarity_when_eval_pruning_active() -> None:
    runtime = SubgraphAnswerSearchRuntime(
        eval_cfg=normalize_search_eval_cfg(
            {
                "report_profile": "rank_only",
                "monte_carlo": {
                    "rollouts": 2,
                    "batch_rollouts": 2,
                    "confidence": 0.95,
                    "action_pruning": {"per_node_top_k": 3, "per_state_top_k": 0},
                },
            }
        ),
        policy=_make_subgraph_policy(max_steps=2),
        sampler=SubgraphSampler(max_steps=2),
    )

    assert runtime._eval_batch_build_options() == SubgraphBatchBuildOptions(
        include_edge_question_similarity=True,
        include_oracle_distance=False,
        include_teacher_banks=False,
    )
