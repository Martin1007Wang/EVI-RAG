from __future__ import annotations

import torch

from src.models.components.backward_prior import StructuralBackwardPrior
from src.models.components.rollout_types import RolloutResult
from src.models.configs.environment import EnvironmentConfig
from src.models.configs.objective import SubTBConfig
from src.models.configs.policy import BackboneConfig, FlowHeadConfig, PolicyConfig, PriorityHeadConfig
from src.models.configs.search import BeamSearchConfig, RolloutConfig
from src.models.configs.training import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.models.dual_flow_module import DualFlowModule
from src.models.environment.contracts import CsrAdjacency, GraphEnvContext


def _build_csr(*, row: torch.Tensor, col: torch.Tensor, edge_ids: torch.Tensor, num_nodes: int) -> CsrAdjacency:
    if int(row.numel()) == 0:
        return CsrAdjacency(
            crow=torch.zeros((num_nodes + 1,), dtype=torch.long),
            col=col,
            edge_ids=edge_ids,
            size=(num_nodes, num_nodes),
        )
    order = torch.argsort(row)
    row_sorted = row.index_select(0, order)
    col_sorted = col.index_select(0, order)
    edge_sorted = edge_ids.index_select(0, order)
    crow = torch.searchsorted(
        row_sorted,
        torch.arange(num_nodes + 1, dtype=torch.long, device=row.device),
        right=False,
    )
    return CsrAdjacency(
        crow=crow,
        col=col_sorted,
        edge_ids=edge_sorted,
        size=(num_nodes, num_nodes),
    )


def _build_context(*, backward_start_local_indices: torch.Tensor | None = None) -> GraphEnvContext:
    num_nodes = 3
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)
    adj_t_fwd = _build_csr(row=edge_index[0], col=edge_index[1], edge_ids=edge_ids, num_nodes=num_nodes)
    adj_t_bwd = _build_csr(row=edge_index[1], col=edge_index[0], edge_ids=edge_ids, num_nodes=num_nodes)
    edge_rel = torch.zeros((edge_index.size(1),), dtype=torch.long)
    return GraphEnvContext(
        num_graphs=1,
        num_nodes_total=num_nodes,
        node_ptr=torch.tensor([0, num_nodes], dtype=torch.long),
        edge_index=edge_index,
        edge_relations=edge_rel,
        edge_rel_global=edge_rel.clone(),
        edge_batch=torch.zeros((edge_index.size(1),), dtype=torch.long),
        node_batch=torch.zeros((num_nodes,), dtype=torch.long),
        adj_t_fwd=adj_t_fwd,
        adj_t_bwd=adj_t_bwd,
        node_embeddings=torch.zeros((num_nodes, 4), dtype=torch.float32),
        node_tokens=torch.zeros((num_nodes, 4), dtype=torch.float32),
        relation_tokens=torch.zeros((1, 4), dtype=torch.float32),
        question_emb=torch.zeros((1, 4), dtype=torch.float32),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([2], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.arange(num_nodes, dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
        start_local_indices=None,
        backward_start_local_indices=backward_start_local_indices,
    )


def _build_module() -> DualFlowModule:
    env_cfg = EnvironmentConfig(super_source_enabled=False)
    policy_cfg = PolicyConfig(
        backbone=BackboneConfig(
            embedding_dim=4,
            hidden_dim=4,
            gnn_layers=0,
            gnn_dropout=0.0,
            use_adapter=False,
            adapter_dim=2,
            adapter_dropout=0.0,
            use_positional_encoding=False,
            use_film=False,
        ),
        flow_head=FlowHeadConfig(
            hidden_dim=8,
            num_layers=2,
            dropout=0.0,
            relation_low_rank=2,
            relation_low_rank_edge_chunk_size=64,
        ),
        priority_head=PriorityHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
    )
    sampling_cfg = RolloutConfig(
        num_rollouts=1,
        max_steps=2,
        stop_min_steps=0,
        sampling_temperature=1.0,
        sampling_mode="greedy",
        eval_sampling_temperature=0.5,
        eval_sample_without_replacement=True,
    )
    eval_cfg = BeamSearchConfig(beam_size=2, max_steps=2, require_done=False)
    subtb_cfg = SubTBConfig(backward_weight=1.0)
    module = DualFlowModule(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
        sampling_cfg=sampling_cfg,
        eval_cfg=eval_cfg,
        subtb_cfg=subtb_cfg,
        training_cfg=TrainingConfig(),
        optimizer_cfg=OptimizerConfig(),
        scheduler_cfg=SchedulerConfig(),
    )
    module.eval()
    return module


def test_uniform_in_degree_prior_uses_exact_in_degree_without_stop_pseudocount() -> None:
    context = _build_context()
    prior = StructuralBackwardPrior(mode="uniform_in_degree")
    log_pb = prior.log_prob_edges(
        env_context=context,
        source_nodes=torch.tensor([0], dtype=torch.long),
        target_nodes=torch.tensor([1], dtype=torch.long),
        edge_graph_ids=torch.tensor([0], dtype=torch.long),
        dtype=torch.float32,
    )
    assert torch.allclose(log_pb, torch.zeros_like(log_pb), atol=1.0e-8)


def test_backward_rollout_uses_reversed_topology_and_answer_start_nodes() -> None:
    module = _build_module()
    base_context = _build_context(backward_start_local_indices=torch.tensor([1], dtype=torch.long))
    encoded_context = (
        torch.zeros((base_context.num_nodes_total, 4), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((base_context.num_graphs, 4), dtype=torch.float32),
    )
    captured: dict[str, GraphEnvContext] = {}

    def _fake_sample_forward(
        env_context: GraphEnvContext,
        policy,
        *,
        deterministic: bool = False,
        encoded_context=None,
    ) -> RolloutResult:
        del policy, deterministic, encoded_context
        captured["context"] = env_context
        return RolloutResult(
            log_pf_sum=torch.zeros((1, 1), dtype=torch.float32),
            stop_nodes=torch.tensor([[0]], dtype=torch.long),
            num_moves=torch.zeros((1, 1), dtype=torch.long),
            num_steps=torch.zeros((1, 1), dtype=torch.long),
            stop_reason=torch.zeros((1, 1), dtype=torch.long),
            valid_mask=torch.ones((1, 1), dtype=torch.bool),
        )

    def _fake_compute_rewards(
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        reward_beta: float | None = None,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del context, reward_beta, target_local_indices, target_ptr, target_field_name
        return torch.ones_like(stop_nodes_abs, dtype=torch.float32), {}

    def _fake_compute_hit_mask(
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
    ) -> torch.Tensor:
        del context, target_local_indices, target_ptr, target_field_name
        return torch.zeros_like(stop_nodes_abs, dtype=torch.bool)

    module.sampler.sample_forward = _fake_sample_forward  # type: ignore[assignment]
    module.compute_rewards = _fake_compute_rewards  # type: ignore[assignment]
    module.compute_hit_mask = _fake_compute_hit_mask  # type: ignore[assignment]

    _, _, _, _, valid_ratio = module._sample_backward_rollout(
        base_context=base_context,
        encoded_context=encoded_context,
        current_beta=1.0,
    )
    bwd_context = captured["context"]
    assert bwd_context.adj_t_fwd is base_context.adj_t_bwd
    assert bwd_context.adj_t_bwd is base_context.adj_t_fwd
    assert bwd_context.start_local_indices is not None
    assert torch.equal(bwd_context.start_local_indices, torch.tensor([2], dtype=torch.long))
    assert float(valid_ratio.item()) == 1.0
