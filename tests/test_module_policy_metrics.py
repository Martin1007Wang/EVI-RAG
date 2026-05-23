from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.weaver.context import GraphContext, TargetContext, build_directed_adjacency_index
from src.weaver.module import WeaverModule, policy_diagnostic_metrics
from src.weaver.nn.feature_encoder import EncodedFeatures
from src.weaver.objectives import SubTBLoss
from src.weaver.policy.forward import ForwardPolicy
from src.weaver.policy.output import ForwardPolicyOutput
from src.weaver.rollout.runner import RolloutRunner
from src.weaver.state import State
from src.weaver.transition import ExpansionBatch, SampleMeta, TerminalBatch, TrainingBatch


def make_policy_output(
    *,
    terminal_log_flow: torch.Tensor,
    continue_log_flow: torch.Tensor,
    frontier_row_ids: torch.Tensor,
    frontier_edge_ids: torch.Tensor,
    edge_logit: torch.Tensor,
    edge_log_prob: torch.Tensor,
    num_rows: int,
    num_edges: int,
) -> ForwardPolicyOutput:
    edge_log_flow = continue_log_flow.index_select(0, frontier_row_ids) + edge_log_prob
    state_log_flow = torch.logaddexp(terminal_log_flow, continue_log_flow)
    return ForwardPolicyOutput(
        frontier_row_ids=frontier_row_ids,
        frontier_edge_ids=frontier_edge_ids,
        terminal_log_flow=terminal_log_flow,
        continue_log_flow=continue_log_flow,
        state_log_flow=state_log_flow,
        edge_logit=edge_logit,
        edge_log_prob=edge_log_prob,
        edge_log_flow=edge_log_flow,
        stop_log_prob=terminal_log_flow - state_log_flow,
        expand_log_prob=continue_log_flow - state_log_flow,
        edge_action_log_prob=edge_log_flow - state_log_flow.index_select(0, frontier_row_ids),
        num_rows=num_rows,
        num_edges=num_edges,
    )


def test_policy_diagnostic_metrics_accepts_stop_continue_outputs() -> None:
    expansion_out = make_policy_output(
        terminal_log_flow=torch.tensor([0.2, 0.7], dtype=torch.float32),
        continue_log_flow=torch.tensor([-0.2, 1.0], dtype=torch.float32),
        frontier_row_ids=torch.tensor([0, 1], dtype=torch.long),
        frontier_edge_ids=torch.tensor([2, 3], dtype=torch.long),
        edge_logit=torch.tensor([0.0, 0.1], dtype=torch.float32),
        edge_log_prob=torch.tensor([0.0, 0.0], dtype=torch.float32),
        num_rows=2,
        num_edges=5,
    )
    terminal_out = make_policy_output(
        terminal_log_flow=torch.tensor([0.5], dtype=torch.float32),
        continue_log_flow=torch.tensor([float("-inf")], dtype=torch.float32),
        frontier_row_ids=torch.empty(0, dtype=torch.long),
        frontier_edge_ids=torch.empty(0, dtype=torch.long),
        edge_logit=torch.empty(0, dtype=torch.float32),
        edge_log_prob=torch.empty(0, dtype=torch.float32),
        num_rows=1,
        num_edges=5,
    )

    metrics = policy_diagnostic_metrics(
        expansion_out=expansion_out,
        expansion_depth=torch.tensor([0, 1], dtype=torch.long),
        terminal_out=terminal_out,
        terminal_depth=torch.tensor([2], dtype=torch.long),
    )

    assert metrics["policy_stop_vs_continue_log_ratio_depth0_mean"].item() == pytest.approx(0.4)
    assert metrics["policy_stop_vs_continue_log_ratio_depth1_mean"].item() == pytest.approx(-0.3)
    assert metrics["policy_stop_vs_continue_log_ratio_depth2_mean"].item() == float("inf")
    assert torch.isfinite(metrics["policy_frontier_size_mean"])
    assert metrics["policy_stop_prob_depth0_mean"].item() == pytest.approx(torch.sigmoid(torch.tensor(0.4)).item())


class StubStateEncoder(torch.nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_encoder = SimpleNamespace(output_dim=edge_dim)

    def forward(self, *, features, state, context):
        del context
        rows = state.num_rows
        device = state.device
        return SimpleNamespace(
            query_h=features.query_model.index_select(0, state.graph_ids).to(device=device),
            row_state_h=torch.zeros(rows, self.hidden_dim, dtype=torch.float32, device=device),
        )

    def encode_edge_tokens(
        self,
        *,
        features,
        src_node_ids,
        edge_ids,
        dst_node_ids,
        query_h=None,
    ) -> torch.Tensor:
        del src_node_ids, dst_node_ids, query_h
        return features.edge_token_model.index_select(0, edge_ids.to(dtype=torch.long))


def tiny_graph_context() -> GraphContext:
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    return GraphContext(
        edge_index=edge_index,
        node_to_graph=torch.tensor([0, 0], dtype=torch.long),
        edge_to_graph=torch.tensor([0], dtype=torch.long),
        anchor_mask=torch.tensor([True, False]),
        adjacency=build_directed_adjacency_index(edge_index=edge_index, num_nodes=2),
        num_nodes=2,
        num_edges=1,
        num_graphs=1,
    )


def tiny_target_context(context: GraphContext) -> TargetContext:
    return TargetContext(
        target_mask=torch.tensor([False, True], dtype=torch.bool),
        reachable_target_node_ids=torch.tensor([1], dtype=torch.long),
        reachable_target_node_ids_ptr=torch.tensor([0, 1], dtype=torch.long),
        target_count_by_graph=torch.tensor([1], dtype=torch.long),
        node_target_shortest_path_edge_mask_flat=torch.zeros(1, dtype=torch.bool),
    )


def tiny_features() -> EncodedFeatures:
    query_model = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    node_model = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
    edge_relation_model = torch.tensor([[0.2, -0.2]], dtype=torch.float32)
    edge_token_model = torch.tensor([[0.1, 0.0, 0.2, -0.2, 0.0, 0.1]], dtype=torch.float32)
    return EncodedFeatures(
        node_text_semantic=torch.zeros(2, 2, dtype=torch.float32),
        node_has_text=torch.zeros(2, dtype=torch.bool),
        edge_relation_semantic=torch.zeros(1, 2, dtype=torch.float32),
        query_semantic=torch.zeros(1, 2, dtype=torch.float32),
        node_model=node_model,
        edge_relation_model=edge_relation_model,
        query_model=query_model,
        edge_token_model=edge_token_model,
    )


def test_policy_step_output_backprops_without_terminal_samples() -> None:
    context = tiny_graph_context()
    target = tiny_target_context(context)
    features = tiny_features()

    parent = State.initial(
        graph=context,
        graph_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    )
    child = parent.expand(
        graph=context,
        rows=torch.tensor([0], dtype=torch.long),
        edge_ids=torch.tensor([0], dtype=torch.long),
        expand_budget=2,
    )
    expansions = ExpansionBatch(
        parent=parent,
        child=child,
        edge_ids=torch.tensor([0], dtype=torch.long),
        meta=SampleMeta(
            trajectory_ids=torch.tensor([0], dtype=torch.long),
            step_ids=torch.tensor([0], dtype=torch.long),
            source_ids=torch.tensor([0], dtype=torch.long),
        ),
    )
    terminals = TerminalBatch.empty_like(graph_like=parent)
    training = TrainingBatch(
        expansions=expansions,
        terminals=terminals,
    )

    policy = ForwardPolicy(
        state_encoder=StubStateEncoder(hidden_dim=2, edge_dim=6),
        max_expand_budget=2,
    )
    module = WeaverModule(
        policy_feature_encoder=torch.nn.Identity(),
        policy=policy,
        reward_model=torch.nn.Identity(),
        policy_objective=SubTBLoss(),
        runner=SimpleNamespace(engine=SimpleNamespace(expand_budget=2), progress_fn=None),
        optimization=SimpleNamespace(),
        evaluation=SimpleNamespace(
            k_windows=(1,),
            exclude_anchors_from_retrieved=False,
            use_reachable_targets=False,
            enable_calibration_metrics=False,
            enable_terminal_diagnostics=False,
        ),
    )

    output = module.policy_step_output(
        graph=context,
        target=target,
        policy_features=features,
        training=training,
    )

    assert output.loss.requires_grad

    policy.zero_grad(set_to_none=True)
    output.loss.backward()

    assert any(param.grad is not None for param in policy.parameters())
