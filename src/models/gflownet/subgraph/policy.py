from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components import EmbeddingBackbone, NodeFlowHead, TransitionPolicyHead
from src.models.configs import GFlowNetTrainingConfig, PolicyConfig
from src.models.configs.policy import SUBGRAPH_STATE_MODE
from src.utils.nn_init import init_linear_xavier
from src.utils.precision_utils import align_float_input_dtype
from src.utils.segment_ops import segment_logsumexp_1d

from .mdp import SubgraphEnv
from .prepared_batch import SubgraphPreparedBatch
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch


def _build_mlp(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(max(int(num_layers) - 1, 0)):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        layers.append(nn.GELU())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _bit_count(bits: int) -> int:
    return int(int(bits).bit_count())


def _selected_node_pool(
    *,
    node_tokens: torch.Tensor,
    selected_nodes: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not selected_nodes:
        return node_tokens.new_zeros(
            (int(node_tokens.size(-1)),), dtype=node_tokens.dtype
        )
    indices = torch.tensor(selected_nodes, device=device, dtype=torch.long)
    return node_tokens.index_select(0, indices).mean(dim=0)


def _selected_relation_pool(
    *,
    prepared_batch: SubgraphPreparedBatch,
    edge_ids: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if not edge_ids:
        return prepared_batch.relation_tokens.new_zeros(
            (int(prepared_batch.relation_tokens.size(-1)),),
            dtype=prepared_batch.relation_tokens.dtype,
        )
    edge_indices = torch.tensor(edge_ids, device=device, dtype=torch.long)
    relation_ids = prepared_batch.topology.edge_type.index_select(0, edge_indices)
    return prepared_batch.relation_tokens.index_select(0, relation_ids).mean(dim=0)


def _coverage_features(
    *,
    graph_idx: int,
    analysis: SubgraphAnalysis,
    full_mask: int,
) -> torch.Tensor:
    max_anchor_count = max(_bit_count(full_mask), 1)
    if analysis.selected_node_ids:
        coverages = [
            float(_bit_count(int(analysis.reachability_bits.get(int(node_id), 0))))
            / float(max_anchor_count)
            for node_id in analysis.selected_node_ids
        ]
        mean_coverage = float(sum(coverages)) / float(len(coverages))
        max_coverage = float(max(coverages))
    else:
        mean_coverage = 0.0
        max_coverage = 0.0
    return torch.tensor(
        [
            float(graph_idx),
            float(len(analysis.selected_node_ids)),
            float(analysis.num_selected_edges),
            float(analysis.anchor_component_count),
            float(mean_coverage),
            float(max_coverage),
        ],
        dtype=torch.float32,
    )


@dataclass(frozen=True)
class SubgraphActionDistribution:
    flat_state_indices: torch.Tensor
    actions: tuple[SubgraphAction, ...]
    edge_ids: torch.Tensor
    is_stop_action: torch.Tensor
    logits: torch.Tensor
    segment_ids: torch.Tensor
    current_component_counts: torch.Tensor
    next_component_counts: torch.Tensor
    state_features: torch.Tensor
    current_best_answer_distance: torch.Tensor
    target_answer_distance: torch.Tensor
    question_similarity: torch.Tensor
    merge_bonus: torch.Tensor
    action_new_bit_gain: torch.Tensor
    candidate_answer_counts: torch.Tensor
    target_nodes: torch.Tensor


class SubgraphPolicy(nn.Module):
    def __init__(
        self,
        *,
        policy_cfg: PolicyConfig,
        training_cfg: GFlowNetTrainingConfig,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.state_mode = SUBGRAPH_STATE_MODE
        self.config = policy_cfg
        self.training_cfg = training_cfg
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 for SubgraphPolicy.")
        self.backbone = EmbeddingBackbone(policy_cfg.backbone)
        self.env = SubgraphEnv(
            max_steps=self.max_steps,
            reward_cfg=training_cfg.subgraph_reward,
        )
        hidden_dim = int(policy_cfg.backbone.hidden_dim)
        self.hidden_dim = hidden_dim
        state_struct_dim = 6
        candidate_struct_dim = 6
        self.state_encoder_norm = nn.LayerNorm((3 * hidden_dim) + state_struct_dim)
        self.state_encoder = _build_mlp(
            input_dim=(3 * hidden_dim) + state_struct_dim,
            output_dim=hidden_dim,
            hidden_dim=int(policy_cfg.subgraph_state_encoder.hidden_dim),
            num_layers=int(policy_cfg.subgraph_state_encoder.num_layers),
            dropout=float(policy_cfg.subgraph_state_encoder.dropout),
        )
        self.state_flow_head = NodeFlowHead(
            node_dim=hidden_dim,
            question_dim=hidden_dim,
            hidden_dim=int(policy_cfg.state_score_head.hidden_dim),
            num_layers=int(policy_cfg.state_score_head.num_layers),
            dropout=float(policy_cfg.state_score_head.dropout),
            conditioning=str(policy_cfg.state_score_head.conditioning),
        )
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.candidate_encoder_norm = nn.LayerNorm(
            (3 * hidden_dim) + candidate_struct_dim
        )
        self.candidate_encoder = _build_mlp(
            input_dim=(3 * hidden_dim) + candidate_struct_dim,
            output_dim=hidden_dim,
            hidden_dim=int(policy_cfg.subgraph_action_head.hidden_dim),
            num_layers=int(policy_cfg.subgraph_action_head.num_layers),
            dropout=float(policy_cfg.subgraph_action_head.dropout),
        )
        self.action_policy_head = TransitionPolicyHead(
            state_dim=hidden_dim,
            relation_dim=hidden_dim,
            hidden_dim=int(policy_cfg.subgraph_action_head.hidden_dim),
            num_layers=int(policy_cfg.subgraph_action_head.num_layers),
            dropout=float(policy_cfg.subgraph_action_head.dropout),
            detach_input_features=False,
        )
        init_linear_xavier(self.stop_head)

    @property
    def prefix_memory_size(self) -> int:
        return 0

    @property
    def prefix_memory_ready(self) -> bool:
        return False

    def record_sampled_prefix_experience(self, *args: Any, **kwargs: Any) -> int:
        del args, kwargs
        return 0

    def prepare_batch(self, batch: Any) -> SubgraphPreparedBatch:
        batch.require_raw_features()
        return self.env.prepare_batch(batch=batch, backbone=self.backbone)

    def initialize_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        num_rollouts: int,
    ) -> SubgraphRolloutBatch:
        return self.env.initialize_rollout_batch(
            prepared_batch=prepared_batch,
            num_rollouts=num_rollouts,
        )

    def analyze_rollout_batch(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
    ) -> tuple[SubgraphAnalysis, ...]:
        return self.env.analyze_rollout_batch(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
        )

    def _encode_state_rows(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...],
    ) -> torch.Tensor:
        features: list[torch.Tensor] = []
        device = prepared_batch.device
        for state_idx, state in enumerate(rollout_batch.states):
            graph_idx = int(rollout_batch.graph_ids[state_idx].item())
            analysis = analyses[state_idx]
            node_pool = _selected_node_pool(
                node_tokens=prepared_batch.node_tokens,
                selected_nodes=analysis.selected_node_ids,
                device=device,
            )
            relation_pool = _selected_relation_pool(
                prepared_batch=prepared_batch,
                edge_ids=state.edge_ids,
                device=device,
            )
            question_pool = prepared_batch.question_tokens[graph_idx]
            struct = _coverage_features(
                graph_idx=graph_idx,
                analysis=analysis,
                full_mask=int(prepared_batch.graph_anchor_full_mask[graph_idx]),
            ).to(device=device)
            features.append(
                torch.cat((node_pool, relation_pool, question_pool, struct), dim=0)
            )
        if not features:
            return prepared_batch.node_tokens.new_empty((0, self.hidden_dim))
        stacked = torch.stack(features, dim=0)
        stacked = align_float_input_dtype(stacked, module=self.state_encoder_norm)
        stacked = self.state_encoder_norm(stacked)
        stacked = align_float_input_dtype(stacked, module=self.state_encoder[0])
        return self.state_encoder(stacked)

    def compute_log_flows(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None = None,
    ) -> torch.Tensor:
        if analyses is None:
            analyses = self.analyze_rollout_batch(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
            )
        state_features = self._encode_state_rows(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        question_features = prepared_batch.question_tokens.index_select(
            0, rollout_batch.graph_ids
        )
        log_flows = self.state_flow_head(state_features, question_features).to(
            dtype=torch.float32
        )
        return torch.where(
            rollout_batch.done_mask,
            torch.zeros_like(log_flows, dtype=torch.float32),
            log_flows,
        )

    def compute_action_distribution(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...] | None = None,
    ) -> SubgraphActionDistribution:
        if analyses is None:
            analyses = self.analyze_rollout_batch(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
            )
        device = prepared_batch.device
        state_features = self._encode_state_rows(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            analyses=analyses,
        )
        active_state_indices = rollout_batch.active_state_indices()
        if not active_state_indices:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            empty_bool = torch.empty((0,), device=device, dtype=torch.bool)
            empty_float = torch.empty((0,), device=device, dtype=torch.float32)
            empty_state = state_features.new_empty((0, self.hidden_dim))
            return SubgraphActionDistribution(
                flat_state_indices=empty_long,
                actions=(),
                edge_ids=empty_long,
                is_stop_action=empty_bool,
                logits=empty_float,
                segment_ids=empty_long,
                current_component_counts=empty_long,
                next_component_counts=empty_long,
                state_features=empty_state,
                current_best_answer_distance=empty_float,
                target_answer_distance=empty_float,
                question_similarity=empty_float,
                merge_bonus=empty_float,
                action_new_bit_gain=empty_float,
                candidate_answer_counts=empty_long,
                target_nodes=empty_long,
            )
        actions: list[SubgraphAction] = []
        edge_ids: list[int] = []
        stop_flags: list[bool] = []
        segment_ids: list[int] = []
        current_component_counts: list[int] = []
        next_component_counts: list[int] = []
        question_similarity: list[float] = []
        merge_bonus: list[float] = []
        action_new_bit_gain: list[float] = []
        candidate_answer_counts: list[int] = []
        current_best_answer_distance: list[float] = []
        target_answer_distance: list[float] = []
        target_nodes: list[int] = []
        candidate_rows: list[torch.Tensor] = []
        relation_rows: list[torch.Tensor] = []
        stop_logits_by_state: list[float] = []
        active_state_tensor = torch.tensor(
            active_state_indices, device=device, dtype=torch.long
        )
        active_state_features = state_features.index_select(0, active_state_tensor)
        for local_state_idx, flat_state_idx in enumerate(active_state_indices):
            graph_idx = int(rollout_batch.graph_ids[flat_state_idx].item())
            state = rollout_batch.states[flat_state_idx]
            analysis = analyses[flat_state_idx]
            selected_edge_set = set(int(edge_id) for edge_id in state.edge_ids)
            selected_node_set = set(
                int(node_id) for node_id in analysis.selected_node_ids
            )
            current_components = int(analysis.anchor_component_count)
            oracle_distance_map = prepared_batch.graph_oracle_answer_distance[graph_idx]
            current_answer_count, _ = self.env.count_gold_answers(
                prepared_batch=prepared_batch,
                graph_idx=graph_idx,
                analysis=analysis,
            )
            current_oracle_distance = self.env.oracle_distance(
                prepared_batch=prepared_batch,
                graph_idx=graph_idx,
                analysis=analysis,
            )
            outgoing_map = prepared_batch.graph_outgoing_edge_ids[graph_idx]
            if int(state.num_edges) < self.max_steps:
                candidate_edge_ids: list[int] = []
                for node_id in analysis.selected_node_ids:
                    for edge_id in outgoing_map.get(int(node_id), ()):
                        edge_id = int(edge_id)
                        if edge_id in selected_edge_set:
                            continue
                        candidate_edge_ids.append(edge_id)
                for edge_id in dict.fromkeys(candidate_edge_ids):
                    edge_idx = int(edge_id)
                    src = int(prepared_batch.topology.edge_index[0, edge_idx].item())
                    dst = int(prepared_batch.topology.edge_index[1, edge_idx].item())
                    relation_id = int(
                        prepared_batch.topology.edge_type[edge_idx].item()
                    )
                    is_new_node = 1.0 if int(dst) not in selected_node_set else 0.0
                    merge_flag = 0.0
                    next_components = current_components
                    if int(dst) in selected_node_set:
                        src_component = int(analysis.component_labels.get(src, -1))
                        dst_component = int(analysis.component_labels.get(dst, -1))
                        if (
                            src_component >= 0
                            and dst_component >= 0
                            and src_component != dst_component
                        ):
                            merge_flag = 1.0
                            next_components = max(current_components - 1, 1)
                    new_bit_gain = _bit_count(
                        int(analysis.reachability_bits.get(src, 0))
                        & ~int(analysis.reachability_bits.get(dst, 0))
                    )
                    full_mask = int(prepared_batch.graph_anchor_full_mask[graph_idx])
                    dst_bits_after = int(analysis.reachability_bits.get(dst, 0)) | int(
                        analysis.reachability_bits.get(src, 0)
                    )
                    candidate_full_answers = (
                        1 if full_mask > 0 and dst_bits_after == full_mask else 0
                    )
                    relation_features = prepared_batch.relation_tokens[relation_id]
                    dst_features = prepared_batch.node_tokens[dst]
                    src_features = prepared_batch.node_tokens[src]
                    question_features = prepared_batch.question_tokens[graph_idx]
                    action_struct = torch.tensor(
                        [
                            float(is_new_node),
                            float(merge_flag),
                            float(new_bit_gain),
                            float(current_components),
                            float(next_components),
                            float(candidate_full_answers),
                        ],
                        device=device,
                        dtype=torch.float32,
                    )
                    candidate_rows.append(
                        torch.cat(
                            (
                                src_features,
                                relation_features,
                                dst_features,
                                action_struct,
                            ),
                            dim=0,
                        )
                    )
                    relation_rows.append(relation_features)
                    actions.append(SubgraphAction.add_edge(edge_idx))
                    edge_ids.append(edge_idx)
                    target_nodes.append(dst)
                    stop_flags.append(False)
                    segment_ids.append(local_state_idx)
                    current_component_counts.append(current_components)
                    next_component_counts.append(next_components)
                    merge_bonus.append(float(merge_flag))
                    action_new_bit_gain.append(float(new_bit_gain))
                    candidate_answer_counts.append(int(candidate_full_answers))
                    current_best_answer_distance.append(float(current_oracle_distance))
                    target_answer_distance.append(
                        float(oracle_distance_map.get(dst, -1))
                    )
                    similarity = F.cosine_similarity(
                        (relation_features + dst_features)
                        .unsqueeze(0)
                        .to(dtype=torch.float32),
                        question_features.unsqueeze(0).to(dtype=torch.float32),
                        dim=-1,
                    )
                    question_similarity.append(float(similarity.item()))
            state_feature = active_state_features[local_state_idx]
            stop_logit = (
                self.stop_head(state_feature).squeeze(-1).to(dtype=torch.float32)
            )
            stop_logits_by_state.append(float(stop_logit.item()))
            candidate_rows.append(
                torch.cat(
                    (
                        state_feature,
                        state_feature,
                        state_feature,
                        torch.zeros((6,), device=device, dtype=torch.float32),
                    ),
                    dim=0,
                )
            )
            relation_rows.append(prepared_batch.question_tokens[graph_idx])
            actions.append(SubgraphAction.stop())
            edge_ids.append(-1)
            target_nodes.append(-1)
            stop_flags.append(True)
            segment_ids.append(local_state_idx)
            current_component_counts.append(current_components)
            next_component_counts.append(current_components)
            merge_bonus.append(0.0)
            action_new_bit_gain.append(0.0)
            candidate_answer_counts.append(int(current_answer_count))
            current_best_answer_distance.append(float(current_oracle_distance))
            target_answer_distance.append(float(current_oracle_distance))
            question_similarity.append(0.0)
        candidate_inputs = torch.stack(candidate_rows, dim=0)
        candidate_inputs = align_float_input_dtype(
            candidate_inputs, module=self.candidate_encoder_norm
        )
        candidate_inputs = self.candidate_encoder_norm(candidate_inputs)
        candidate_inputs = align_float_input_dtype(
            candidate_inputs, module=self.candidate_encoder[0]
        )
        candidate_features = self.candidate_encoder(candidate_inputs)
        relation_feature_tensor = torch.stack(relation_rows, dim=0)
        state_feature_tensor = active_state_features.index_select(
            0,
            torch.tensor(segment_ids, device=device, dtype=torch.long),
        )
        logits = self.action_policy_head(
            current_state_features=state_feature_tensor,
            candidate_state_features=candidate_features,
            relation_features=relation_feature_tensor,
        ).to(dtype=torch.float32)
        stop_mask = torch.tensor(stop_flags, device=device, dtype=torch.bool)
        if bool(stop_mask.any().item()):
            stop_positions = torch.nonzero(stop_mask, as_tuple=False).view(-1)
            logits[stop_positions] = torch.tensor(
                [
                    float(stop_logits_by_state[segment_ids[int(pos)]])
                    for pos in stop_positions.detach().cpu().tolist()
                ],
                device=device,
                dtype=torch.float32,
            )
        return SubgraphActionDistribution(
            flat_state_indices=active_state_tensor,
            actions=tuple(actions),
            edge_ids=torch.tensor(edge_ids, device=device, dtype=torch.long),
            is_stop_action=stop_mask,
            logits=logits,
            segment_ids=torch.tensor(segment_ids, device=device, dtype=torch.long),
            current_component_counts=torch.tensor(
                current_component_counts, device=device, dtype=torch.long
            ),
            next_component_counts=torch.tensor(
                next_component_counts, device=device, dtype=torch.long
            ),
            state_features=active_state_features,
            current_best_answer_distance=torch.tensor(
                current_best_answer_distance,
                device=device,
                dtype=torch.float32,
            ),
            target_answer_distance=torch.tensor(
                target_answer_distance,
                device=device,
                dtype=torch.float32,
            ),
            question_similarity=torch.tensor(
                question_similarity, device=device, dtype=torch.float32
            ),
            merge_bonus=torch.tensor(merge_bonus, device=device, dtype=torch.float32),
            action_new_bit_gain=torch.tensor(
                action_new_bit_gain, device=device, dtype=torch.float32
            ),
            candidate_answer_counts=torch.tensor(
                candidate_answer_counts, device=device, dtype=torch.long
            ),
            target_nodes=torch.tensor(target_nodes, device=device, dtype=torch.long),
        )

    def compute_target_log_probs(
        self,
        distribution: SubgraphActionDistribution,
    ) -> torch.Tensor:
        if int(distribution.logits.numel()) == 0:
            return distribution.logits
        lse, _ = segment_logsumexp_1d(
            values=distribution.logits,
            segment_ids=distribution.segment_ids,
            num_segments=int(distribution.flat_state_indices.numel()),
            dtype=torch.float32,
            ignore_non_finite=True,
            empty_value=float("-inf"),
        )
        return distribution.logits - lse.index_select(0, distribution.segment_ids)

    def compute_proposal_bias(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        distribution: SubgraphActionDistribution,
        action_prior_scale: float,
    ) -> torch.Tensor:
        del prepared_batch
        if int(distribution.logits.numel()) == 0 or float(action_prior_scale) <= 0.0:
            return torch.zeros_like(distribution.logits, dtype=torch.float32)
        cfg = self.training_cfg.subgraph_proposal
        bias = torch.zeros_like(distribution.logits, dtype=torch.float32)
        edge_mask = ~distribution.is_stop_action
        if bool(edge_mask.any().item()):
            if float(cfg.prior_component_merge_weight) > 0.0:
                bias[edge_mask] = bias[edge_mask] + (
                    float(cfg.prior_component_merge_weight)
                    * distribution.merge_bonus[edge_mask]
                )
            if float(cfg.prior_question_similarity_weight) > 0.0:
                bias[edge_mask] = bias[edge_mask] + (
                    float(cfg.prior_question_similarity_weight)
                    * distribution.question_similarity[edge_mask]
                )
            if float(cfg.oracle_answer_distance_weight) > 0.0:
                current_distance = distribution.current_best_answer_distance[edge_mask]
                target_distance = distribution.target_answer_distance[edge_mask]
                valid = (current_distance >= 0.0) & (target_distance >= 0.0)
                progress = torch.zeros_like(current_distance)
                if bool(valid.any().item()):
                    progress[valid] = (
                        current_distance[valid] - target_distance[valid]
                    ).clamp_min(0.0)
                bias[edge_mask] = bias[edge_mask] + (
                    float(cfg.oracle_answer_distance_weight) * progress
                )
        if (
            bool(distribution.is_stop_action.any().item())
            and float(cfg.stop_hit_bias) > 0.0
        ):
            stop_mask = distribution.is_stop_action
            stop_hits = distribution.candidate_answer_counts[stop_mask] > 0
            bias[stop_mask] = bias[stop_mask] + stop_hits.to(
                dtype=torch.float32
            ) * float(cfg.stop_hit_bias)
        return bias * float(action_prior_scale)


__all__ = [
    "SubgraphActionDistribution",
    "SubgraphPolicy",
]
