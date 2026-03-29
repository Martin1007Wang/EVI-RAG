from __future__ import annotations

from dataclasses import replace
from typing import cast

import torch
import torch.nn.functional as F

from src.graph import TrajectoryBatch
import src.models.gflownet.memory as memory_module
import src.models.gflownet.prefix_policy as gflownet_policy_impl
from src.models.configs import (
    ActionPriorConfig,
    GFlowNetTrainingConfig,
    HorizonConfig,
    OptimizerConfig,
    PrefixMemoryConfig,
    SchedulerConfig,
    SearchEvalConfig,
)
from src.models.gflownet.prefix_sampler import ForwardTrajectoryGFNSampler
from src.models.gflownet.prefix_state import SearchState
from src.models.gflownet.memory import PrefixMemoryBank, PrefixMemoryEntry
from src.models.gflownet.prefix import PrefixKey
from src.models.gflownet_module import GFlowNetModule
from src.metrics.runtime_factory import GraphTaskRuntimeFactory

from .conftest import make_batch_from_graph, make_policy_config


def _make_memory_module(*, prefix_memory: PrefixMemoryConfig) -> GFlowNetModule:
    policy_cfg = replace(make_policy_config(), prefix_memory=prefix_memory)
    return GFlowNetModule(
        horizon_cfg=HorizonConfig(max_steps=2),
        training_cfg=GFlowNetTrainingConfig(
            rollouts_per_graph=1,
            sampling_temperature=1.0,
            force_stop_on_answer_hit=True,
        ),
        action_prior_cfg=ActionPriorConfig(
            node_topology_weight=0.0,
            node_embedding_weight=0.0,
        ),
        policy_cfg=policy_cfg,
        eval_cfg=SearchEvalConfig(report_profile="rank_only"),
        optimizer_cfg=OptimizerConfig(type="adamw", lr=1.0e-4, weight_decay=0.0),
        scheduler_cfg=SchedulerConfig(type="cosine", interval="step", t_max=8),
        metric_runtime_factory=GraphTaskRuntimeFactory(),
    )


def _force_graph_moves(module: GFlowNetModule) -> None:
    original_forward = module.policy.compute_forward_distribution

    def _prefer_graph_moves(prepared_batch, state, **kwargs):  # noqa: ANN001
        distribution = original_forward(prepared_batch, state, **kwargs)
        logits = distribution.edge_logits.detach().clone().to(dtype=torch.float32)
        if distribution.is_stop_action is not None:
            logits[distribution.is_stop_action.to(dtype=torch.bool)] = -1.0e9
        return replace(distribution, edge_logits=logits)

    module.policy.compute_forward_distribution = _prefer_graph_moves  # type: ignore[method-assign]


def _make_memory_batch() -> TrajectoryBatch:
    return make_batch_from_graph(
        num_nodes=2,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_rel_global=torch.tensor([0], dtype=torch.long),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([1], dtype=torch.long),
        answer_entity_ids=torch.tensor([101], dtype=torch.long),
        node_entity_ids=torch.tensor([100, 101], dtype=torch.long),
        sample_id="prefix-memory",
    )


def _make_prefix_entry(
    *, idx: int, key_vector: torch.Tensor, value_vector: torch.Tensor
) -> PrefixMemoryEntry:
    return PrefixMemoryEntry(
        prefix_key=PrefixKey(
            sample_id=f"sample-{idx}",
            current_node=idx,
            num_steps=0,
            is_absorbing=False,
            token_ids=(idx,),
            visited_entity_ids=(1000 + idx,),
        ),
        key_vector=key_vector,
        value_vector=value_vector,
        success=bool(idx % 2 == 0),
        remaining_steps=idx,
        terminal_log_reward=float(-idx),
    )


def test_prefix_memory_context_changes_state_features_after_recording() -> None:
    module = _make_memory_module(
        prefix_memory=PrefixMemoryConfig(
            enabled=True,
            min_entries=1,
            capacity=32,
            top_k=1,
            temperature=0.1,
        )
    )
    batch = _make_memory_batch()
    _force_graph_moves(module)
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0]], dtype=torch.long),
        max_steps=2,
    )

    features_before = module.policy.base_policy.build_state_features(
        prepared_batch, state
    )
    added = module.policy.record_sampled_prefix_experience(prepared_batch, sample_batch)
    features_after = module.policy.base_policy.build_state_features(
        prepared_batch, state
    )

    assert added > 0
    assert module.policy.prefix_memory_size > 0
    assert module.policy.prefix_memory_ready is True
    assert not torch.allclose(features_before, features_after)


def test_prefix_memory_can_store_failure_only_experience() -> None:
    module = _make_memory_module(
        prefix_memory=PrefixMemoryConfig(
            enabled=True,
            min_entries=1,
            capacity=32,
            top_k=1,
            temperature=0.1,
            store_successes=False,
            store_failures=True,
        )
    )
    batch = _make_memory_batch()
    _force_graph_moves(module)
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    failed_sample_batch = replace(
        sample_batch,
        success_mask=torch.zeros_like(sample_batch.success_mask, dtype=torch.bool),
        terminal_log_rewards=torch.full_like(
            sample_batch.terminal_log_rewards, fill_value=-3.0
        ),
    )

    added = module.policy.record_sampled_prefix_experience(
        prepared_batch,
        failed_sample_batch,
    )

    assert added > 0
    assert module.policy.prefix_memory_size > 0
    assert module.policy.prefix_memory_ready is True


def test_prefix_memory_retrieval_chunks_large_query_batches(monkeypatch) -> None:
    monkeypatch.setattr(memory_module, "_CPU_MAX_SIMILARITY_VALUES", 6)
    bank = PrefixMemoryBank(
        config=PrefixMemoryConfig(
            enabled=True,
            min_entries=1,
            capacity=8,
            top_k=2,
            temperature=0.5,
        ),
        value_dim=3,
    )
    key_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    value_vectors = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    added = bank.add_entries(
        [
            _make_prefix_entry(
                idx=idx,
                key_vector=key_vectors[idx],
                value_vector=value_vectors[idx],
            )
            for idx in range(3)
        ]
    )
    query_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    chunk_size = memory_module._memory_query_chunk_size(
        device=query_vectors.device,
        total_queries=int(query_vectors.size(0)),
        num_keys=int(key_vectors.size(0)),
    )
    retrieved = bank.retrieve(query_vectors)
    normalized_queries = F.normalize(query_vectors, dim=-1)
    normalized_keys = F.normalize(key_vectors, dim=-1)
    similarity = normalized_queries @ normalized_keys.transpose(0, 1)
    top_similarity, top_indices = torch.topk(
        similarity,
        k=2,
        dim=-1,
        largest=True,
        sorted=True,
    )
    weights = torch.softmax(top_similarity / 0.5, dim=-1)
    gathered_values = value_vectors.index_select(0, top_indices.reshape(-1)).view(
        int(query_vectors.size(0)),
        2,
        3,
    )
    expected = (weights.unsqueeze(-1) * gathered_values).sum(dim=1)

    assert added == 3
    assert chunk_size < int(query_vectors.size(0))
    assert torch.allclose(retrieved, expected)


def test_prefix_memory_state_features_match_under_aggressive_state_chunking() -> None:
    module = _make_memory_module(
        prefix_memory=PrefixMemoryConfig(
            enabled=True,
            min_entries=1,
            capacity=32,
            top_k=1,
            temperature=0.1,
        )
    )
    batch = _make_memory_batch()
    _force_graph_moves(module)
    prepared_batch = module.policy.prepare_batch(batch)
    sampler = cast(ForwardTrajectoryGFNSampler, module.sampler)
    sample_batch = sampler.sample(
        batch=batch.without_raw_features(),
        policy=module.policy,
        prepared_batch=prepared_batch,
        rollouts_per_graph=1,
        temperature=1.0,
    )
    added = module.policy.record_sampled_prefix_experience(prepared_batch, sample_batch)
    state = SearchState.initialize(
        topology=prepared_batch.topology,
        observation=prepared_batch.observation,
        start_nodes=torch.tensor([[0, 0, 0]], dtype=torch.long),
        max_steps=2,
    )

    baseline = module.policy.base_policy.build_state_features(prepared_batch, state)
    original_chunk = gflownet_policy_impl._STATE_SCORING_CHUNK_SIZE
    gflownet_policy_impl._STATE_SCORING_CHUNK_SIZE = 1
    try:
        chunked = module.policy.base_policy.build_state_features(prepared_batch, state)
    finally:
        gflownet_policy_impl._STATE_SCORING_CHUNK_SIZE = original_chunk

    assert added > 0
    assert torch.allclose(chunked, baseline, atol=1.0e-6)
