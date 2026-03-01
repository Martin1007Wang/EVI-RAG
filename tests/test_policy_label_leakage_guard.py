from __future__ import annotations

import torch

from src.models.components.policy import DualFlowPolicy
from src.models.configs.policy import BackboneConfig, FlowHeadConfig, PolicyConfig, PriorityHeadConfig
from src.models.environment.contracts import CsrAdjacency, DynamicAgentState, GraphEnvContext


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


def _build_context(*, a_local_indices: torch.Tensor) -> GraphEnvContext:
    num_nodes = 3
    edge_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )
    edge_ids = torch.arange(edge_index.size(1), dtype=torch.long)
    row_fwd = edge_index[0]
    col_fwd = edge_index[1]
    row_bwd = edge_index[1]
    col_bwd = edge_index[0]
    adj_t_fwd = _build_csr(row=row_fwd, col=col_fwd, edge_ids=edge_ids, num_nodes=num_nodes)
    adj_t_bwd = _build_csr(row=row_bwd, col=col_bwd, edge_ids=edge_ids, num_nodes=num_nodes)
    edge_rel = torch.tensor([0, 1, 0], dtype=torch.long)
    a_ptr = torch.tensor([0, int(a_local_indices.numel())], dtype=torch.long)
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
                [0.10, 0.20, 0.30, 0.40],
                [0.30, 0.10, 0.40, 0.20],
                [0.60, 0.20, 0.10, 0.10],
            ],
            dtype=torch.float32,
        ),
        node_tokens=torch.zeros((num_nodes, 4), dtype=torch.float32),
        relation_tokens=torch.tensor(
            [
                [0.10, 0.20, 0.00, 0.10],
                [0.00, 0.10, 0.20, 0.10],
            ],
            dtype=torch.float32,
        ),
        question_emb=torch.tensor([[0.20, 0.10, 0.10, 0.40]], dtype=torch.float32),
        q_local_indices=torch.tensor([0], dtype=torch.long),
        a_local_indices=a_local_indices,
        q_ptr=torch.tensor([0, 1], dtype=torch.long),
        a_ptr=a_ptr,
        answer_entity_ids=a_local_indices.clone(),
        answer_ptr=a_ptr.clone(),
        node_global_ids=torch.arange(num_nodes, dtype=torch.long),
        dummy_mask=torch.tensor([False]),
        sample_ids=["sample_0"],
    )


def _make_agent_state(*, hidden_dim: int, num_nodes: int) -> DynamicAgentState:
    return DynamicAgentState(
        step_t=0,
        current_nodes=torch.tensor([[0]], dtype=torch.long),
        hidden_states=torch.zeros((1, 1, hidden_dim), dtype=torch.float32),
        visited_mask=torch.zeros((1, num_nodes), dtype=torch.bool),
        cumulative_rewards=torch.zeros((1, 1), dtype=torch.float32),
        done_mask=torch.zeros((1, 1), dtype=torch.bool),
    )


def _build_policy() -> DualFlowPolicy:
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
            flow_projection_eps=1.0e-8,
        ),
        priority_head=PriorityHeadConfig(hidden_dim=8, num_layers=2, dropout=0.0),
        stop_bias_init=-8.0,
    )
    policy = DualFlowPolicy(cfg, backward_prior_mode="uniform")
    policy.eval()
    return policy


def test_policy_logits_do_not_depend_on_answer_labels() -> None:
    torch.manual_seed(7)
    policy = _build_policy()
    context_a = _build_context(a_local_indices=torch.tensor([2], dtype=torch.long))
    context_b = _build_context(a_local_indices=torch.tensor([1], dtype=torch.long))
    state = _make_agent_state(hidden_dim=4, num_nodes=3)

    node_tokens_a, relation_tokens_a, question_tokens_a = policy.encode_context(context_a)
    out_a = policy.compute_action_scores(
        env_context=context_a,
        agent_state=state,
        node_tokens=node_tokens_a,
        question_tokens=question_tokens_a,
        relation_tokens=relation_tokens_a,
    )

    node_tokens_b, relation_tokens_b, question_tokens_b = policy.encode_context(context_b)
    out_b = policy.compute_action_scores(
        env_context=context_b,
        agent_state=state,
        node_tokens=node_tokens_b,
        question_tokens=question_tokens_b,
        relation_tokens=relation_tokens_b,
    )

    assert torch.equal(out_a["edge_ids"], out_b["edge_ids"])
    assert torch.equal(out_a["target_nodes"], out_b["target_nodes"])
    assert torch.allclose(out_a["edge_logits"], out_b["edge_logits"], atol=1.0e-7)
    assert torch.allclose(out_a["stop_logits"], out_b["stop_logits"], atol=1.0e-7)
    assert bool(torch.isfinite(out_a["edge_logits"]).all().item())
