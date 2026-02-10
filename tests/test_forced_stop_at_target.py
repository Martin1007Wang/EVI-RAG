from __future__ import annotations

import types

import torch

from src.models.dual_flow_constants import _STOP_ACTION_ID, _TERMINAL_HIT
from src.models.dual_flow_module import DualFlowModule


def _tiny_dual_flow_module() -> DualFlowModule:
    return DualFlowModule(
        hidden_dim=8,
        emb_dim=8,
        max_steps=2,
        training_cfg={
            "db_cfg": {"sampling_temperature_start": 1.0, "sampling_temperature_end": 1.0},
        },
        evaluation_cfg={},
        runtime_cfg={},
    )


def test_rollout_forces_stop_on_target_node() -> None:
    module = _tiny_dual_flow_module()
    prepared = types.SimpleNamespace(
        num_graphs=1,
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        relation_tokens=torch.zeros((1, module.hidden_dim), dtype=torch.float32),
    )

    calls: list[torch.Tensor] = []

    def fake_sample_edges(
        self: DualFlowModule,
        *,
        prepared: object,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        parent_nodes: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        collect_policy_metrics: bool = False,
        prev_rel_emb: torch.Tensor | None = None,
        force_stop_mask: torch.Tensor | None = None,
        prior_weight_override: float | None = None,
        node_is_target: torch.Tensor | None = None,
        lookahead_cfg: dict[str, float | bool] | None = None,
    ):
        _ = (prepared, edge_batch, steps, temperature, context_tokens, collect_policy_metrics, prev_rel_emb, prior_weight_override, node_is_target, lookahead_cfg)
        device = parent_nodes.device
        if force_stop_mask is None:
            force_stop_mask = torch.zeros((num_graphs,), device=device, dtype=torch.bool)
        force_stop_mask = force_stop_mask.to(device=device, dtype=torch.bool).view(-1)
        calls.append(force_stop_mask.detach().clone())
        chosen_edge = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        chosen_edge = torch.where(force_stop_mask, torch.full_like(chosen_edge, _STOP_ACTION_ID), chosen_edge)
        zeros = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        return chosen_edge, zeros, zeros, None

    module._sample_edges = types.MethodType(fake_sample_edges, module)  # type: ignore[assignment]

    graph_mask = torch.tensor([True])
    start_nodes = torch.tensor([0], dtype=torch.long)
    node_is_target = torch.tensor([True, False], dtype=torch.bool)
    edge_ids_by_head = torch.tensor([0], dtype=torch.long)
    edge_ptr_by_head = torch.tensor([0, 1, 1], dtype=torch.long)

    result = module._rollout_policy(
        prepared=prepared,
        graph_mask=graph_mask,
        start_nodes=start_nodes,
        node_is_target=node_is_target,
        edge_ids_by_head=edge_ids_by_head,
        edge_ptr_by_head=edge_ptr_by_head,
        record_actions=True,
        record_log_pf=False,
        temperature=1.0,
        context_tokens=torch.zeros((1, module.hidden_dim), dtype=torch.float32),
        collect_policy_metrics=False,
        exploration_cfg=None,
        edge_mask=None,
        prior_weight_override=None,
        lookahead_cfg=None,
    )

    assert calls and bool(calls[0].item()) is True
    assert result.stop_reason.tolist() == [_TERMINAL_HIT]
    assert result.num_moves.tolist() == [0]
    assert result.stop_nodes.tolist() == [0]

