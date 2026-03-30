from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.models.components import ActionScoringHead
from src.utils.precision_utils import align_float_input_dtype

from .prepared_batch import SubgraphPreparedBatch
from .reward import resolve_subgraph_answer_entities
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


def _oracle_distance(
    *,
    prepared_batch: SubgraphPreparedBatch,
    graph_idx: int,
    analysis: SubgraphAnalysis,
) -> int:
    oracle_distance_map = prepared_batch.graph_oracle_answer_distance[int(graph_idx)]
    return min(
        (
            int(oracle_distance_map[node_id])
            for node_id in analysis.selected_node_ids
            if int(node_id) in oracle_distance_map
        ),
        default=-1,
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


class SubgraphActor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        actor: dict[str, Any],
        subgraph_proposal: dict[str, Any],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.subgraph_proposal = dict(subgraph_proposal)
        candidate_struct_dim = 6
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        self.candidate_encoder_norm = nn.LayerNorm(
            (3 * self.hidden_dim) + candidate_struct_dim
        )
        self.candidate_encoder = _build_mlp(
            input_dim=(3 * self.hidden_dim) + candidate_struct_dim,
            output_dim=self.hidden_dim,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.action_head = ActionScoringHead(
            state_dim=self.hidden_dim,
            relation_dim=self.hidden_dim,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
            detach_input_features=False,
        )

    def build_action_distribution(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...],
        state_features: torch.Tensor,
    ) -> SubgraphActionDistribution:
        device = prepared_batch.device
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
            current_answer_count = sum(
                1
                for entity_id in resolve_subgraph_answer_entities(
                    prepared_batch=prepared_batch,
                    graph_idx=graph_idx,
                    analysis=analysis,
                )
                if entity_id in prepared_batch.graph_answer_entities[graph_idx]
            )
            current_oracle_distance = _oracle_distance(
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
        logits = self.action_head(
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

    def compute_proposal_bias(
        self,
        *,
        distribution: SubgraphActionDistribution,
        proposal_bias_scale: float,
    ) -> torch.Tensor:
        if int(distribution.logits.numel()) == 0 or float(proposal_bias_scale) <= 0.0:
            return torch.zeros_like(distribution.logits, dtype=torch.float32)
        proposal_cfg = self.subgraph_proposal
        merge_weight = float(proposal_cfg.get("prior_component_merge_weight", 0.0))
        question_similarity_weight = float(
            proposal_cfg.get("prior_question_similarity_weight", 0.0)
        )
        oracle_distance_weight = float(
            proposal_cfg.get("oracle_answer_distance_weight", 0.0)
        )
        stop_hit_bias = float(proposal_cfg.get("stop_hit_bias", 0.0))
        bias = torch.zeros_like(distribution.logits, dtype=torch.float32)
        edge_mask = ~distribution.is_stop_action
        if bool(edge_mask.any().item()):
            if merge_weight > 0.0:
                bias[edge_mask] = bias[edge_mask] + (
                    merge_weight * distribution.merge_bonus[edge_mask]
                )
            if question_similarity_weight > 0.0:
                bias[edge_mask] = bias[edge_mask] + (
                    question_similarity_weight
                    * distribution.question_similarity[edge_mask]
                )
            if oracle_distance_weight > 0.0:
                current_distance = distribution.current_best_answer_distance[edge_mask]
                target_distance = distribution.target_answer_distance[edge_mask]
                valid = (current_distance >= 0.0) & (target_distance >= 0.0)
                progress = torch.zeros_like(current_distance)
                if bool(valid.any().item()):
                    progress[valid] = (
                        current_distance[valid] - target_distance[valid]
                    ).clamp_min(0.0)
                bias[edge_mask] = bias[edge_mask] + (oracle_distance_weight * progress)
        if bool(distribution.is_stop_action.any().item()) and stop_hit_bias > 0.0:
            stop_mask = distribution.is_stop_action
            stop_hits = distribution.candidate_answer_counts[stop_mask] > 0
            bias[stop_mask] = (
                bias[stop_mask] + stop_hits.to(dtype=torch.float32) * stop_hit_bias
            )
        return bias * float(proposal_bias_scale)


__all__ = ["SubgraphActionDistribution", "SubgraphActor"]
