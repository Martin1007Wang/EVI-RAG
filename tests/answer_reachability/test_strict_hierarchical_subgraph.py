from __future__ import annotations

import math

import pytest
import torch

from src.metrics.search_eval_utils import normalize_search_eval_cfg
from src.metrics.runtime_factory import GraphTaskRuntimeFactory
from src.models.gflownet.config_utils import normalize_training_cfg
from src.models.gflownet.policy import SUBGRAPH_STATE_MODE, SubgraphPolicy
from src.models.gflownet.sampler import SubgraphSampler
from src.models.gflownet.state import SubgraphAction, SubgraphState
from src.models.gflownet_module import GFlowNetModule

from .conftest import make_batch_from_graph


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
            "gold_answer_bonus": 2.0,
            "wrong_answer_penalty": 2.0,
            "failure_penalty": 4.0,
            "size_penalty": 0.1,
            "redundancy_penalty": 0.25,
            "component_penalty": 0.5,
        },
        proposal_prior={
            "oracle_answer_distance_weight": 0.0,
            "prior_question_similarity_weight": 0.0,
            "prior_component_merge_weight": 0.0,
            "stop_hit_bias": 0.0,
        },
        max_steps=max_steps,
    )


def _make_bridge_batch() -> object:
    return make_batch_from_graph(
        num_nodes=3,
        edge_index=torch.tensor([[0, 2], [1, 1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0, 1], dtype=torch.long),
        anchor_local_indices=torch.tensor([0, 2], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        sample_id="bridge-subgraph",
    )


def test_strict_state_starts_with_anchor_nodes() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    rollout_batch = policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    state = rollout_batch.states[0]
    analysis = policy.analyze_state(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=state,
    )

    assert state.edge_ids == ()
    assert analysis.selected_node_ids == (0, 2)
    assert analysis.reachability_bits == {0: 1, 2: 2}
    assert analysis.anchor_component_count == 2


def test_strict_stop_exposes_answer_set_from_terminal_topology() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    rollout_batch = policy.initialize_rollout_batch(
        prepared_batch=prepared_batch,
        num_rollouts=1,
    )
    rollout_batch = policy.transition(
        rollout_batch=rollout_batch,
        chosen_actions=(SubgraphAction.add_edge(0),),
    )
    rollout_batch = policy.transition(
        rollout_batch=rollout_batch,
        chosen_actions=(SubgraphAction.add_edge(1),),
    )
    distribution = policy.compute_action_distribution(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
    )
    state_distribution = distribution.state_distributions[0]

    assert len(state_distribution.stop_choices) == 1
    assert state_distribution.stop_choices[0].answer_entity_id is None
    analysis = policy.analyze_rollout_batch(
        prepared_batch=prepared_batch,
        rollout_batch=rollout_batch,
    )[0]
    answer_set = policy.admissible_answer_set(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=analysis,
    )

    assert answer_set.entities == (101,)
    assert answer_set.gold_entities == (101,)
    reward, commit_count, gold_count, hit = policy.compute_stop_log_reward(
        prepared_batch=prepared_batch,
        graph_idx=0,
        analysis=analysis,
    )
    assert hit is True
    assert commit_count == 1
    assert gold_count == 1
    assert reward > 0.0


def test_backward_policy_is_uniform_over_forward_valid_parent_edges() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    state = SubgraphState(edge_ids=(0, 1))

    removable = policy.compute_backward_log_prob(
        prepared_batch=prepared_batch,
        graph_idx=0,
        state=state,
    )

    assert policy.backward_policy_name() == "uniform_forward_valid_edge_deletion"
    assert removable == pytest.approx(-math.log(2.0))


def test_teacher_force_marks_gold_answer_state_when_available() -> None:
    batch = _make_bridge_batch()
    policy = _make_subgraph_policy(max_steps=2)
    prepared_batch = policy.prepare_batch(batch)
    sampler = SubgraphSampler(max_steps=2)

    sample_batch = sampler.teacher_force(
        policy=policy,
        prepared_batch=prepared_batch,
        edge_sequences=((0, 1),),
    )

    assert sample_batch.chosen_edge_ids[0, 0].tolist() == [0, 1]
    assert sample_batch.terminal_hit_mask[0, 0].item() is True
    assert sample_batch.terminal_commit_candidate_counts[0, 0].item() == 1
    assert sample_batch.terminal_gold_answer_counts[0, 0].item() == 1
    assert sample_batch.terminal_answer_entity_ids == ((101,),)


def test_strict_paper_defaults_disable_action_pruning() -> None:
    training_cfg = normalize_training_cfg({})
    eval_cfg = normalize_search_eval_cfg({})

    assert training_cfg["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }
    assert eval_cfg["monte_carlo"]["action_pruning"] == {
        "per_node_top_k": 0,
        "per_state_top_k": 0,
    }


def test_gflownet_module_trains_with_answer_committed_policy() -> None:
    batch = _make_bridge_batch()
    module = GFlowNetModule(
        horizon_cfg={"max_steps": 2},
        training_cfg=normalize_training_cfg(
            {
                "rollouts_per_graph": 2,
                "sampling_temperature": 1.0,
                "action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0},
            }
        ),
        policy_cfg={
            "state_mode": SUBGRAPH_STATE_MODE,
            "backbone": {
                "embedding_dim": 8,
                "hidden_dim": 8,
                "use_adapter": True,
                "adapter_dim": 4,
                "adapter_dropout": 0.0,
            },
            "flow_head": {
                "hidden_dim": 8,
                "num_layers": 2,
                "dropout": 0.0,
                "conditioning": "concat",
            },
            "state_encoder": {
                "hidden_dim": 8,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "actor": {"hidden_dim": 8, "num_layers": 2, "dropout": 0.0},
        },
        eval_cfg=normalize_search_eval_cfg(
            {
                "report_profile": "rank_only",
                "monte_carlo": {
                    "rollouts": 4,
                    "batch_rollouts": 4,
                    "action_pruning": {"per_node_top_k": 0, "per_state_top_k": 0},
                },
            }
        ),
        optimizer_cfg={"type": "adamw", "lr": 1.0e-4, "weight_decay": 0.0},
        scheduler_cfg={"type": "cosine", "interval": "step", "t_max": 8},
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )

    loss = module.training_step(batch, 0)

    assert torch.isfinite(loss).item()
