from __future__ import annotations

import torch

from src.models.policy import DualFlowPolicy
from src.models.configs.policy import (
    BackboneConfig,
    FlowHeadConfig,
    PolicyConfig,
    PriorityHeadConfig,
)
from src.models.environment import CsrAdjacency, DynamicAgentState, GraphEnvContext


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


def _build_context() -> GraphEnvContext:
    num_nodes = 3
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)
    adj_t_fwd = _build_csr(
        row=edge_index[0], col=edge_index[1], edge_ids=edge_ids, num_nodes=num_nodes
    )
    adj_t_bwd = _build_csr(
        row=edge_index[1], col=edge_index[0], edge_ids=edge_ids, num_nodes=num_nodes
    )
    edge_rel = torch.tensor([0, 1], dtype=torch.long)
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
        node_embeddings=torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.10],
                [0.40, 0.20, 0.10, 0.30],
                [0.20, 0.50, 0.10, 0.10],
            ],
            dtype=torch.float32,
        ),
        node_tokens=torch.zeros((num_nodes, 4), dtype=torch.float32),
        relation_tokens=torch.tensor(
            [
                [0.10, 0.10, 0.20, 0.10],
                [0.00, 0.20, 0.10, 0.10],
            ],
            dtype=torch.float32,
        ),
        question_emb=torch.tensor([[0.20, 0.10, 0.10, 0.20]], dtype=torch.float32),
        question_ctx=torch.tensor(
            [
                [
                    [0.20, 0.10, 0.10, 0.20],
                    [0.10, 0.20, 0.10, 0.20],
                ]
            ],
            dtype=torch.float32,
        ),
        question_ctx_mask=torch.tensor([[True, True]], dtype=torch.bool),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=torch.tensor([2], dtype=torch.long),
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=torch.tensor([0, 1], dtype=torch.long),
        answer_entity_ids=torch.tensor([2], dtype=torch.long),
        answer_ptr=torch.tensor([0, 1], dtype=torch.long),
        node_global_ids=torch.arange(num_nodes, dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
    )


def _build_policy(*, doob_h_alpha: float) -> DualFlowPolicy:
    cfg = PolicyConfig(
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
        ),
        priority_head=PriorityHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        stop_bias_init=0.0,
        stop_delta_scale=1.0,
        stop_delta_temperature=1.0,
        doob_h_alpha=doob_h_alpha,
        doob_h_node_temperature=1.0,
    )
    return DualFlowPolicy(cfg, backward_prior_mode="uniform")


def _build_state() -> DynamicAgentState:
    return DynamicAgentState(
        step_t=0,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        flow_direction="forward",
        hidden_states=torch.zeros((1, 1, 4), dtype=torch.float32),
        visited_mask=torch.zeros((1, 3), dtype=torch.bool),
        cumulative_rewards=torch.zeros((1, 1), dtype=torch.float32),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
        num_moves=torch.zeros((1, 1), dtype=torch.long),
    )


def test_doob_h_shift_is_added_to_move_logits() -> None:
    context = _build_context()
    state = _build_state()
    log_h = torch.tensor([-5.0, -0.1, -2.0], dtype=torch.float32)
    alpha = 2.0

    torch.manual_seed(0)
    policy_base = _build_policy(doob_h_alpha=0.0)
    policy_base.eval()
    node_tokens, relation_tokens, question_tokens = policy_base.encode_context(context)
    cache_base = policy_base.build_action_cache(
        env_context=context,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
    )
    cache_base["node_log_h"] = log_h.to(dtype=node_tokens.dtype)
    out_base = policy_base.compute_action_scores(
        env_context=context,
        agent_state=state,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        relation_tokens=relation_tokens,
        action_cache=cache_base,
    )

    torch.manual_seed(0)
    policy_shift = _build_policy(doob_h_alpha=alpha)
    policy_shift.load_state_dict(policy_base.state_dict(), strict=True)
    policy_shift.eval()
    cache_shift = policy_shift.build_action_cache(
        env_context=context,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
    )
    cache_shift["node_log_h"] = log_h.to(dtype=node_tokens.dtype)
    out_shift = policy_shift.compute_action_scores(
        env_context=context,
        agent_state=state,
        node_tokens=node_tokens,
        question_tokens=question_tokens,
        relation_tokens=relation_tokens,
        action_cache=cache_shift,
    )

    base_logits = out_base["edge_logits"].to(dtype=torch.float32)
    shift_logits = out_shift["edge_logits"].to(dtype=torch.float32)
    target_nodes = out_base["target_nodes"]
    expected_delta = alpha * log_h.index_select(0, target_nodes).to(dtype=torch.float32)
    actual_delta = shift_logits - base_logits
    assert torch.allclose(actual_delta, expected_delta, atol=1.0e-6, rtol=1.0e-6)
