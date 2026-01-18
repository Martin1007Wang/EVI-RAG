import pytest
import torch
import torch.nn as nn

pytest.importorskip("torch_scatter")

from src.gfn.engine import (
    FlowFeatureSpec,
    GFlowNetEngine,
    SinkSelectionSpec,
    StartSelectionSpec,
)
from src.gfn.ops import (
    GFlowNetBatchProcessor,
    GFlowNetInputValidator,
    RolloutInputs,
    compute_policy_log_probs,
)
from src.models.components import RewardOutput, TrajectoryAgent
from src.models.components.gflownet_actor import GFlowNetActor, RolloutResult
from src.models.components.gflownet_env import GraphEnv


class DummyBackbone(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, batch: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _ = batch
        zeros = torch.zeros((1, self.hidden_dim))
        return zeros, zeros, zeros


class DummyLogF(nn.Module):
    def forward(
        self,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        graph_features: torch.Tensor,
        state_vec: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        graph_feat = graph_features.index_select(0, node_batch)
        questions = question_tokens.index_select(0, node_batch)
        return (node_tokens + state_vec + graph_feat + questions).sum(dim=1)


class DummyReward(nn.Module):
    def forward(self, **kwargs: object) -> RewardOutput:
        _ = kwargs
        log_reward = torch.zeros(1)
        success = torch.ones_like(log_reward, dtype=torch.bool)
        return RewardOutput(log_reward=log_reward, success=success)


def build_rollout_inputs(*, device: torch.device, hidden_dim: int) -> RolloutInputs:
    node_tokens = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.3, 0.1, -0.2, 0.5],
            [0.2, -0.1, 0.4, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2],
            [1, 0, 2, 1],
        ],
        device=device,
        dtype=torch.long,
    )
    edge_relations = torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long)
    edge_batch = torch.zeros(edge_index.size(1), device=device, dtype=torch.long)
    edge_ptr = torch.tensor([0, edge_index.size(1)], device=device, dtype=torch.long)
    relation_tokens = torch.tensor(
        [
            [0.1, 0.0, 0.2, -0.1],
            [-0.2, 0.3, 0.0, 0.1],
            [0.05, 0.05, 0.05, 0.05],
            [-0.1, 0.2, -0.2, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    question_tokens = torch.zeros((1, hidden_dim), device=device, dtype=torch.float32)
    node_ptr = torch.tensor([0, node_tokens.size(0)], device=device, dtype=torch.long)
    start_node_locals = torch.tensor([0, 1], device=device, dtype=torch.long)
    start_ptr = torch.tensor([0, start_node_locals.numel()], device=device, dtype=torch.long)
    answer_node_locals = torch.tensor([1, 2], device=device, dtype=torch.long)
    answer_ptr = torch.tensor([0, answer_node_locals.numel()], device=device, dtype=torch.long)
    node_embedding_ids = torch.zeros(node_tokens.size(0), device=device, dtype=torch.long)
    dummy_mask = torch.zeros(1, device=device, dtype=torch.bool)
    return RolloutInputs(
        edge_batch=edge_batch,
        edge_ptr=edge_ptr,
        node_ptr=node_ptr,
        edge_index=edge_index,
        edge_relations=edge_relations,
        node_tokens=node_tokens,
        relation_tokens=relation_tokens,
        question_tokens=question_tokens,
        node_embedding_ids=node_embedding_ids,
        start_node_locals=start_node_locals,
        answer_node_locals=answer_node_locals,
        start_ptr=start_ptr,
        answer_ptr=answer_ptr,
        dummy_mask=dummy_mask,
    )


def build_engine_inputs(device: torch.device):
    hidden_dim = 4
    env = GraphEnv(max_steps=3)
    agent = TrajectoryAgent(
        token_dim=hidden_dim,
        hidden_dim=hidden_dim,
        dropout=0.0,
        force_fp32=True,
    )
    actor = GFlowNetActor(
        env=env,
        agent=agent,
        max_steps=3,
        policy_temperature=1.0,
    )
    flow_spec = FlowFeatureSpec(graph_stats_log1p=False, stats_dim=2)
    processor = GFlowNetBatchProcessor(
        backbone=DummyBackbone(hidden_dim),
        cvt_init=nn.Identity(),
        require_precomputed_edge_batch=False,
    )
    validator = GFlowNetInputValidator(validate_edge_batch=False, validate_rollout_batch=False)
    engine = GFlowNetEngine(
        actor=actor,
        reward_fn=DummyReward(),
        env=env,
        log_f=DummyLogF(),
        log_f_backward=None,
        start_selector=None,
        sink_selector=None,
        flow_spec=flow_spec,
        relation_inverse_map=torch.tensor([1, 0], device=device, dtype=torch.long),
        batch_processor=processor,
        input_validator=validator,
        composite_score_cfg=None,
        dual_stream_cfg=None,
        subtb_cfg=None,
        start_selection_cfg=None,
        sink_selection_cfg=None,
        z_align_cfg=None,
    )
    inputs = build_rollout_inputs(device=device, hidden_dim=hidden_dim)
    graph_cache = processor.build_graph_cache(inputs, device=device)
    return engine, inputs, graph_cache


def compute_log_pb_reference(
    engine: GFlowNetEngine,
    rollout: RolloutResult,
    inputs: RolloutInputs,
    graph_cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    actions = rollout.actions_seq
    num_graphs, num_steps = actions.shape
    state_nodes = engine._build_state_nodes(actions=actions, inputs=inputs)
    state_nodes_after = state_nodes.clone()
    if num_steps > 1:
        state_nodes_after[:, :-1] = state_nodes[:, 1:]
        state_nodes_after[:, -1] = state_nodes[:, -1]
    move_mask = actions >= 0
    lengths = move_mask.to(dtype=torch.long).sum(dim=1)
    step_ids = torch.arange(num_steps, device=actions.device, dtype=torch.long).view(1, -1)
    step_counts_after = (lengths.view(-1, 1) - 1 - step_ids).clamp(min=0)
    num_nodes_total = int(inputs.node_ptr[-1].detach().tolist())
    inverse_edge_ids = engine._build_inverse_edge_ids(
        edge_index=graph_cache["edge_index"],
        edge_relations=graph_cache["edge_relations"],
        num_nodes=num_nodes_total,
    )
    action_ids = actions.clamp(min=0)
    inv_edge_ids = inverse_edge_ids.index_select(0, action_ids.view(-1)).view(num_graphs, num_steps)
    edge_index = graph_cache["edge_index"]
    edge_batch = graph_cache["edge_batch"]
    edge_tail_tokens = graph_cache["node_tokens"].index_select(0, edge_index[1])
    _, state_vec_after, _, _ = engine._compute_state_sequence(
        actions=actions,
        inputs=inputs,
        actor=engine.actor,
    )
    log_pb = torch.zeros_like(actions, dtype=torch.float32)
    temp, _ = engine.actor._resolve_temperature(None)
    for t in range(num_steps):
        if not bool(move_mask[:, t].any().detach().tolist()):
            continue
        curr_nodes = state_nodes_after[:, t]
        step_counts_t = step_counts_after[:, t]
        curr_nodes_per_edge = curr_nodes.index_select(0, edge_batch)
        step_counts_per_edge = step_counts_t.index_select(0, edge_batch)
        horizon_exhausted = step_counts_per_edge >= int(engine.env.max_steps)
        head_match = edge_index[0] == curr_nodes_per_edge
        valid_edges = head_match & (~horizon_exhausted) & (curr_nodes_per_edge >= 0)
        edge_scores = engine.actor.agent.score(
            hidden=state_vec_after[:, t, :],
            relation_tokens=graph_cache["relation_tokens"],
            node_tokens=edge_tail_tokens,
            edge_batch=edge_batch,
        )
        log_prob_edge, _, _, _ = compute_policy_log_probs(
            edge_logits=edge_scores,
            stop_logits=None,
            edge_batch=edge_batch,
            valid_edges=valid_edges,
            num_graphs=num_graphs,
            temperature=temp,
            allow_stop=None,
        )
        selected_edges = inv_edge_ids[:, t]
        selected_log_prob = log_prob_edge.index_select(0, selected_edges)
        valid_selected = valid_edges.index_select(0, selected_edges)
        if not bool(valid_selected[move_mask[:, t]].all().detach().tolist()):
            raise ValueError("Reference log_pb encountered invalid inverse edge.")
        log_pb[:, t] = torch.where(move_mask[:, t], selected_log_prob, torch.zeros_like(selected_log_prob))
    return log_pb


def test_backward_log_pb_matches_reference():
    device = torch.device("cpu")
    engine, inputs, graph_cache = build_engine_inputs(device)
    actions = torch.tensor([[0, 2]], device=device, dtype=torch.long)
    rollout = RolloutResult(
        actions_seq=actions,
        log_pf=torch.zeros(1, device=device),
        log_pf_steps=torch.zeros(1, actions.size(1), device=device),
        log_pb_steps=torch.zeros(1, actions.size(1), device=device),
        reach_success=torch.zeros(1, device=device),
        length=torch.tensor([2.0], device=device),
        stop_node_locals=torch.zeros(1, device=device, dtype=torch.long),
    )
    log_pb_ref = compute_log_pb_reference(engine, rollout, inputs, graph_cache)
    engine._log_pb_max_repeated_edges = 1_000_000
    log_pb_full = engine._compute_backward_log_pb_steps(
        rollout=rollout,
        inputs=inputs,
        graph_cache=graph_cache,
        temperature=None,
        flow_modules=engine._resolve_flow_modules(direction="forward"),
    )
    engine._log_pb_max_repeated_edges = 2
    log_pb_chunked = engine._compute_backward_log_pb_steps(
        rollout=rollout,
        inputs=inputs,
        graph_cache=graph_cache,
        temperature=None,
        flow_modules=engine._resolve_flow_modules(direction="forward"),
    )
    assert torch.allclose(log_pb_full, log_pb_ref, atol=1e-6)
    assert torch.allclose(log_pb_chunked, log_pb_ref, atol=1e-6)


def test_flow_selection_aligns_with_flow_values():
    device = torch.device("cpu")
    engine, inputs, graph_cache = build_engine_inputs(device)
    flow_features = engine._compute_flow_features(inputs)
    graph_cache["flow_features"] = flow_features
    log_f_start = engine._compute_log_f_start(
        inputs=inputs,
        graph_cache=graph_cache,
        flow_features=flow_features,
        log_f_module=engine.log_f,
    )
    start_spec = StartSelectionSpec(
        mode="flow",
        temperature=1.0,
        sample_gumbel=False,
        epsilon=0.0,
        force_uniform=False,
    )
    start_dist = engine._prepare_start_selection_distribution(
        inputs=inputs,
        graph_cache=graph_cache,
        spec=start_spec,
    )
    start_choice = engine._sample_start_node_indices(
        scores=start_dist.scores,
        start_batch=start_dist.candidate_batch,
        num_graphs=1,
        sample_gumbel=False,
    )
    log_pf_start = start_dist.log_probs.index_select(0, start_choice)
    log_f_selected = start_dist.scores.index_select(0, start_choice)
    assert torch.allclose(log_pf_start + log_f_start, log_f_selected, atol=1e-6)

    sink_spec = SinkSelectionSpec(
        mode="flow",
        temperature=1.0,
        sample_gumbel=False,
        epsilon=0.0,
        force_uniform=False,
    )
    sink_dist = engine._prepare_sink_selection_distribution(
        inputs=inputs,
        graph_cache=graph_cache,
        candidate_nodes=inputs.answer_node_locals,
        candidate_ptr=inputs.answer_ptr,
        spec=sink_spec,
    )
    sink_choice = engine._sample_start_node_indices(
        scores=sink_dist.scores,
        start_batch=sink_dist.candidate_batch,
        num_graphs=1,
        sample_gumbel=False,
    )
    log_pf_sink = sink_dist.log_probs.index_select(0, sink_choice)
    log_f_sink = engine._compute_log_f_start(
        inputs=inputs,
        graph_cache=graph_cache,
        flow_features=flow_features,
        log_f_module=engine.log_f_backward,
        start_node_locals=inputs.answer_node_locals,
        start_ptr=inputs.answer_ptr,
    )
    log_f_sink_selected = sink_dist.scores.index_select(0, sink_choice)
    assert torch.allclose(log_pf_sink + log_f_sink, log_f_sink_selected, atol=1e-6)
