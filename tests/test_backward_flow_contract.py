from __future__ import annotations

import torch

from src.models.rollout import (
    STOP_REASON_ACTION,
    STOP_REASON_MAX_STEPS_REACHED,
    RolloutResult,
    StructuralBackwardPrior,
)
from src.models.configs.environment import EnvironmentConfig
from src.models.configs.objective import SubTBConfig
from src.models.configs.policy import (
    BackboneConfig,
    FlowHeadConfig,
    PolicyConfig,
    PriorityHeadConfig,
)
from src.models.configs.search import BeamSearchConfig, RolloutConfig
from src.models.configs.training import OptimizerConfig, SchedulerConfig, TrainingConfig
from src.models.dual_flow_module import DualFlowModule
from src.models.environment import (
    CsrAdjacency,
    GraphEnvContext,
    has_super_source_layout,
)


def _build_csr(
    *, row: torch.Tensor, col: torch.Tensor, edge_ids: torch.Tensor, num_nodes: int
) -> CsrAdjacency:
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


def _build_context(*, with_super_layout: bool) -> GraphEnvContext:
    if with_super_layout:
        # local real nodes: 0,1,2 ; forward super: 3 ; backward super: 4
        # forward path: 3 -> 0 -> 1 -> 2
        # backward hook in original graph: 2 -> 4
        # (rollout on reversed topology will see 4 -> 2 for backward first hop)
        num_nodes = 5
        edge_index = torch.tensor([[3, 0, 1, 2], [0, 1, 2, 4]], dtype=torch.long)
        node_global_ids = torch.tensor([100, 101, 102, -1, -2], dtype=torch.long)
        q_local_indices = torch.tensor([0], dtype=torch.long)
        a_local_indices = torch.tensor([2], dtype=torch.long)
    else:
        num_nodes = 3
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        node_global_ids = torch.arange(num_nodes, dtype=torch.long)
        q_local_indices = torch.tensor([0], dtype=torch.long)
        a_local_indices = torch.tensor([2], dtype=torch.long)
    edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)
    adj_t_fwd = _build_csr(
        row=edge_index[0], col=edge_index[1], edge_ids=edge_ids, num_nodes=num_nodes
    )
    adj_t_bwd = _build_csr(
        row=edge_index[1], col=edge_index[0], edge_ids=edge_ids, num_nodes=num_nodes
    )
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
        q_local_indices=q_local_indices,
        a_local_indices=a_local_indices,
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=a_local_indices.clone(),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=node_global_ids,
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
    )


def _build_module(*, subtb_cfg: SubTBConfig | None = None) -> DualFlowModule:
    env_cfg = EnvironmentConfig(super_source_enabled=True)
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
    if subtb_cfg is None:
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


def test_uniform_in_degree_prior_uses_exact_in_degree_without_stop_pseudocount() -> (
    None
):
    context = _build_context(with_super_layout=False)
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
    base_context = _build_context(with_super_layout=True)
    assert has_super_source_layout(
        node_ptr=base_context.node_ptr,
        node_global_ids=base_context.node_global_ids,
        num_nodes_total=base_context.num_nodes_total,
        device=base_context.node_ptr.device,
    )
    encoded_context = (
        torch.zeros((base_context.num_nodes_total, 4), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((base_context.num_graphs, 4), dtype=torch.float32),
    )
    captured: dict[str, GraphEnvContext] = {}
    captured_direction: dict[str, str] = {}

    def _fake_sample_forward(
        env_context: GraphEnvContext,
        policy,
        *,
        flow_direction: str = "forward",
        deterministic: bool = False,
        encoded_context=None,
        collect_traces: bool = True,
    ) -> RolloutResult:
        del policy, deterministic, encoded_context, collect_traces
        captured["context"] = env_context
        captured_direction["flow_direction"] = flow_direction
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
        terminal_done_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del (
            context,
            reward_beta,
            target_local_indices,
            target_ptr,
            target_field_name,
            terminal_done_mask,
        )
        return torch.ones_like(stop_nodes_abs, dtype=torch.float32), {}

    def _fake_compute_hit_mask(
        stop_nodes_abs: torch.Tensor,
        context: GraphEnvContext,
        *,
        target_local_indices: torch.Tensor | None = None,
        target_ptr: torch.Tensor | None = None,
        target_field_name: str = "a_local_indices",
        terminal_done_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del (
            context,
            target_local_indices,
            target_ptr,
            target_field_name,
            terminal_done_mask,
        )
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
    assert captured_direction["flow_direction"] == "backward"
    assert bwd_context.adj_t_fwd is base_context.adj_t_bwd
    assert bwd_context.adj_t_bwd is base_context.adj_t_fwd
    assert float(valid_ratio.item()) == 1.0


def test_stop_gate_aux_loss_penalizes_missing_stop_at_hit_state() -> None:
    module = _build_module(
        subtb_cfg=SubTBConfig(
            backward_weight=0.0, stop_gate_weight=1.0, stop_gate_margin=0.0
        )
    )
    context = _build_context(with_super_layout=False)
    epsilon = float(module.cfg.env_cfg.stop.reward_epsilon)
    stop_logprob = torch.log(torch.tensor(0.2, dtype=torch.float32))
    rollout = RolloutResult(
        log_pf_sum=torch.zeros((1, 1), dtype=torch.float32),
        stop_nodes=torch.tensor([[1]], dtype=torch.long),
        num_moves=torch.tensor([[2]], dtype=torch.long),
        num_steps=torch.tensor([[2]], dtype=torch.long),
        stop_reason=torch.tensor([[STOP_REASON_ACTION]], dtype=torch.long),
        stop_logprob_steps=torch.tensor(
            [[[stop_logprob.item(), stop_logprob.item(), 0.0]]], dtype=torch.float32
        ),
        state_nodes_steps=torch.tensor([[[0, 2, -1]]], dtype=torch.long),
        continue_valid_steps=torch.tensor([[[True, True, False]]], dtype=torch.bool),
        stop_valid_steps=torch.tensor([[[True, True, False]]], dtype=torch.bool),
    )
    rewards_raw = torch.tensor([[epsilon]], dtype=torch.float32)

    loss, metrics = module._compute_stop_gate_aux_loss(
        rollout=rollout,
        rewards_raw=rewards_raw,
        context=context,
    )

    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0
    assert float(metrics["subtb/stop_gate_valid_ratio"].item()) > 0.0
    assert float(metrics["subtb/stop_gate_target_stop_ratio"].item()) == 1.0


def test_timeout_rollout_scores_terminal_node() -> None:
    module = _build_module(subtb_cfg=SubTBConfig(backward_weight=0.0))
    context = _build_context(with_super_layout=False)
    rollout = RolloutResult(
        log_pf_sum=torch.zeros((1, 1), dtype=torch.float32),
        stop_nodes=torch.tensor([[2]], dtype=torch.long),
        num_moves=torch.zeros((1, 1), dtype=torch.long),
        num_steps=torch.zeros((1, 1), dtype=torch.long),
        stop_reason=torch.tensor([[STOP_REASON_MAX_STEPS_REACHED]], dtype=torch.long),
    )
    _, terminal_mask = module._build_rollout_diagnostics(
        rollout=rollout,
        context=context,
        flow_direction="forward",
    )
    rewards, _ = module.compute_rewards(
        stop_nodes_abs=rollout.stop_nodes,
        context=context,
        terminal_done_mask=terminal_mask,
    )
    assert bool(terminal_mask[0, 0].item()) is True
    assert float(rewards[0, 0].item()) == float(module.cfg.env_cfg.stop.reward_base)
