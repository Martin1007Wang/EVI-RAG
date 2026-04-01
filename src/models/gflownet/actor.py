from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from src.models.components import ActionScoringHead
from src.utils.cuda_memory import cuda_memory_profiling_enabled
from src.utils.logging_utils import get_logger
from src.utils.precision_utils import align_float_input_dtype

from .prepared_batch import SubgraphPreparedBatch
from .reward import resolve_admissible_answer_commit_set
from .state import SubgraphAction, SubgraphAnalysis, SubgraphRolloutBatch


logger = get_logger(__name__)


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


def _answer_ready_entity_pool(
    *,
    prepared_batch: SubgraphPreparedBatch,
    analysis: SubgraphAnalysis,
    entity_id: int,
    full_mask: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    supporting_node_ids = [
        int(node_id)
        for node_id in analysis.selected_node_ids
        if int(prepared_batch.node_entity_ids[int(node_id)].item()) == int(entity_id)
        and int(analysis.reachability_bits.get(int(node_id), 0)) == int(full_mask)
    ]
    if not supporting_node_ids:
        raise RuntimeError(
            "Answer-ready entities must have at least one supporting state node. "
            f"entity_id={entity_id}"
        )
    node_indices = torch.tensor(supporting_node_ids, device=device, dtype=torch.long)
    pooled = prepared_batch.node_tokens.index_select(0, node_indices).mean(dim=0)
    return pooled, int(len(supporting_node_ids))


@dataclass(frozen=True)
class HierarchicalEdgeChoice:
    action: SubgraphAction
    edge_id: int
    source_graph_node: int
    relation_id: int
    target_graph_node: int
    logit: torch.Tensor
    current_component_count: int
    next_component_count: int
    question_similarity: float
    semantic_overlap: float
    action_new_bit_gain: int
    candidate_commit_count: int
    current_best_answer_distance: float
    target_answer_distance: float


@dataclass(frozen=True)
class HierarchicalRelationChoice:
    relation_id: int
    logit: torch.Tensor
    edges: tuple[HierarchicalEdgeChoice, ...]


@dataclass(frozen=True)
class HierarchicalNodeChoice:
    graph_node_id: int
    logit: torch.Tensor
    relations: tuple[HierarchicalRelationChoice, ...]


@dataclass(frozen=True)
class AnswerStopChoice:
    action: SubgraphAction
    answer_entity_id: int | None
    logit: torch.Tensor
    support_node_count: int


@dataclass(frozen=True)
class HierarchicalStateActionDistribution:
    flat_state_index: int
    stop_logit: torch.Tensor
    continue_logit: torch.Tensor
    stop_choice_logits: torch.Tensor
    stop_choice_answer_entity_ids: torch.Tensor
    stop_choice_support_node_counts: torch.Tensor
    node_choice_graph_node_ids: torch.Tensor
    node_choice_logits: torch.Tensor
    relation_choice_relation_ids: torch.Tensor
    relation_choice_logits: torch.Tensor
    relation_choice_node_choice_indices: torch.Tensor
    edge_choice_edge_ids: torch.Tensor
    edge_choice_source_graph_nodes: torch.Tensor
    edge_choice_relation_ids: torch.Tensor
    edge_choice_target_graph_nodes: torch.Tensor
    edge_choice_logits: torch.Tensor
    edge_choice_next_component_counts: torch.Tensor
    edge_choice_question_similarity: torch.Tensor
    edge_choice_semantic_overlap: torch.Tensor
    edge_choice_action_new_bit_gain: torch.Tensor
    edge_choice_candidate_commit_counts: torch.Tensor
    edge_choice_target_answer_distance: torch.Tensor
    edge_choice_relation_choice_indices: torch.Tensor
    node_relation_ptr: torch.Tensor
    relation_edge_ptr: torch.Tensor
    current_component_count: int
    current_commit_candidate_count: int
    current_oracle_distance: float

    def _stop_answer_entity_id(self, stop_choice_idx: int) -> int | None:
        answer_entity_id = int(
            self.stop_choice_answer_entity_ids[int(stop_choice_idx)].item()
        )
        return None if answer_entity_id < 0 else int(answer_entity_id)

    def relation_slice(self, node_choice_idx: int) -> slice:
        start = int(self.node_relation_ptr[int(node_choice_idx)].item())
        end = int(self.node_relation_ptr[int(node_choice_idx) + 1].item())
        return slice(start, end)

    def edge_slice(self, relation_choice_idx: int) -> slice:
        start = int(self.relation_edge_ptr[int(relation_choice_idx)].item())
        end = int(self.relation_edge_ptr[int(relation_choice_idx) + 1].item())
        return slice(start, end)

    def build_stop_action(self, stop_choice_idx: int) -> SubgraphAction:
        return SubgraphAction.stop(self._stop_answer_entity_id(int(stop_choice_idx)))

    def build_edge_action(self, edge_choice_idx: int) -> SubgraphAction:
        edge_choice_idx = int(edge_choice_idx)
        return SubgraphAction.add_edge(
            int(self.edge_choice_edge_ids[edge_choice_idx].item()),
            source_graph_node=int(
                self.edge_choice_source_graph_nodes[edge_choice_idx].item()
            ),
            relation_id=int(self.edge_choice_relation_ids[edge_choice_idx].item()),
            target_graph_node=int(
                self.edge_choice_target_graph_nodes[edge_choice_idx].item()
            ),
        )

    def edge_next_component_count(self, edge_choice_idx: int) -> int:
        return int(self.edge_choice_next_component_counts[int(edge_choice_idx)].item())

    @property
    def stop_choices(self) -> tuple[AnswerStopChoice, ...]:
        return tuple(
            AnswerStopChoice(
                action=self.build_stop_action(stop_choice_idx),
                answer_entity_id=self._stop_answer_entity_id(stop_choice_idx),
                logit=self.stop_choice_logits[int(stop_choice_idx)],
                support_node_count=int(
                    self.stop_choice_support_node_counts[int(stop_choice_idx)].item()
                ),
            )
            for stop_choice_idx in range(int(self.stop_choice_logits.numel()))
        )

    @property
    def node_choices(self) -> tuple[HierarchicalNodeChoice, ...]:
        node_choices: list[HierarchicalNodeChoice] = []
        for node_choice_idx in range(int(self.node_choice_logits.numel())):
            relation_choices: list[HierarchicalRelationChoice] = []
            relation_slice = self.relation_slice(node_choice_idx)
            for relation_choice_idx in range(relation_slice.start, relation_slice.stop):
                edge_choices: list[HierarchicalEdgeChoice] = []
                edge_slice = self.edge_slice(relation_choice_idx)
                for edge_choice_idx in range(edge_slice.start, edge_slice.stop):
                    edge_choices.append(
                        HierarchicalEdgeChoice(
                            action=self.build_edge_action(edge_choice_idx),
                            edge_id=int(
                                self.edge_choice_edge_ids[edge_choice_idx].item()
                            ),
                            source_graph_node=int(
                                self.edge_choice_source_graph_nodes[
                                    edge_choice_idx
                                ].item()
                            ),
                            relation_id=int(
                                self.edge_choice_relation_ids[edge_choice_idx].item()
                            ),
                            target_graph_node=int(
                                self.edge_choice_target_graph_nodes[
                                    edge_choice_idx
                                ].item()
                            ),
                            logit=self.edge_choice_logits[edge_choice_idx],
                            current_component_count=int(self.current_component_count),
                            next_component_count=self.edge_next_component_count(
                                edge_choice_idx
                            ),
                            question_similarity=float(
                                self.edge_choice_question_similarity[
                                    edge_choice_idx
                                ].item()
                            ),
                            semantic_overlap=float(
                                self.edge_choice_semantic_overlap[
                                    edge_choice_idx
                                ].item()
                            ),
                            action_new_bit_gain=int(
                                self.edge_choice_action_new_bit_gain[
                                    edge_choice_idx
                                ].item()
                            ),
                            candidate_commit_count=int(
                                self.edge_choice_candidate_commit_counts[
                                    edge_choice_idx
                                ].item()
                            ),
                            current_best_answer_distance=float(
                                self.current_oracle_distance
                            ),
                            target_answer_distance=float(
                                self.edge_choice_target_answer_distance[
                                    edge_choice_idx
                                ].item()
                            ),
                        )
                    )
                relation_choices.append(
                    HierarchicalRelationChoice(
                        relation_id=int(
                            self.relation_choice_relation_ids[
                                relation_choice_idx
                            ].item()
                        ),
                        logit=self.relation_choice_logits[relation_choice_idx],
                        edges=tuple(edge_choices),
                    )
                )
            node_choices.append(
                HierarchicalNodeChoice(
                    graph_node_id=int(
                        self.node_choice_graph_node_ids[node_choice_idx].item()
                    ),
                    logit=self.node_choice_logits[node_choice_idx],
                    relations=tuple(relation_choices),
                )
            )
        return tuple(node_choices)


@dataclass(frozen=True)
class SubgraphActionDistribution:
    flat_state_indices: torch.Tensor
    state_features: torch.Tensor
    state_distributions: tuple[HierarchicalStateActionDistribution, ...]


@dataclass(frozen=True)
class _ActionDistributionBuildStats:
    raw_candidate_count: int
    stop_choice_count: int


@dataclass(frozen=True)
class _LinearizedStateChoices:
    edge_choice_edge_ids: torch.Tensor
    edge_choice_source_graph_nodes: torch.Tensor
    edge_choice_relation_ids: torch.Tensor
    edge_choice_target_graph_nodes: torch.Tensor
    edge_choice_next_component_counts: torch.Tensor
    edge_choice_question_similarity: torch.Tensor
    edge_choice_semantic_overlap: torch.Tensor
    edge_choice_action_new_bit_gain: torch.Tensor
    edge_choice_candidate_commit_counts: torch.Tensor
    edge_choice_target_answer_distance: torch.Tensor
    edge_choice_relation_choice_indices: torch.Tensor
    node_choice_graph_node_ids: torch.Tensor
    node_choice_candidate_counts: torch.Tensor
    node_choice_reachability_bits: torch.Tensor
    node_choice_is_anchor: torch.Tensor
    relation_choice_relation_ids: torch.Tensor
    relation_choice_node_choice_indices: torch.Tensor
    relation_choice_num_edges: torch.Tensor
    relation_choice_max_new_bit_gain: torch.Tensor
    relation_choice_max_candidate_commit_counts: torch.Tensor
    relation_choice_max_semantic_overlap: torch.Tensor
    node_relation_ptr: torch.Tensor
    relation_edge_ptr: torch.Tensor

    @classmethod
    def empty(cls, *, device: torch.device) -> "_LinearizedStateChoices":
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        empty_float = torch.empty((0,), device=device, dtype=torch.float32)
        zero_ptr = torch.zeros((1,), device=device, dtype=torch.long)
        empty_bool = torch.empty((0,), device=device, dtype=torch.bool)
        return cls(
            edge_choice_edge_ids=empty_long,
            edge_choice_source_graph_nodes=empty_long,
            edge_choice_relation_ids=empty_long,
            edge_choice_target_graph_nodes=empty_long,
            edge_choice_next_component_counts=empty_long,
            edge_choice_question_similarity=empty_float,
            edge_choice_semantic_overlap=empty_float,
            edge_choice_action_new_bit_gain=empty_long,
            edge_choice_candidate_commit_counts=empty_long,
            edge_choice_target_answer_distance=empty_float,
            edge_choice_relation_choice_indices=empty_long,
            node_choice_graph_node_ids=empty_long,
            node_choice_candidate_counts=empty_long,
            node_choice_reachability_bits=empty_long,
            node_choice_is_anchor=empty_bool,
            relation_choice_relation_ids=empty_long,
            relation_choice_node_choice_indices=empty_long,
            relation_choice_num_edges=empty_long,
            relation_choice_max_new_bit_gain=empty_long,
            relation_choice_max_candidate_commit_counts=empty_long,
            relation_choice_max_semantic_overlap=empty_float,
            node_relation_ptr=zero_ptr,
            relation_edge_ptr=zero_ptr,
        )


@dataclass(frozen=True)
class _StateDistributionContext:
    flat_state_index: int
    current_components: int
    current_commit_candidate_count: int
    current_oracle_distance: float
    max_anchor_count: int
    full_mask: int
    state_num_edges: int
    linearized: _LinearizedStateChoices
    answer_choice_answer_entity_ids: torch.Tensor
    answer_choice_support_node_counts: torch.Tensor
    answer_choice_features: torch.Tensor


class SubgraphActor(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        max_steps: int,
        actor: dict[str, Any],
        proposal_prior: dict[str, Any],
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.proposal_prior = dict(proposal_prior)
        node_struct_dim = 4
        relation_struct_dim = 4
        candidate_struct_dim = 6
        self.node_focus_norm = nn.LayerNorm((2 * self.hidden_dim) + node_struct_dim)
        self.node_focus_head = _build_mlp(
            input_dim=(2 * self.hidden_dim) + node_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        self.continue_head = nn.Linear(self.hidden_dim, 1)
        stop_struct_dim = 3
        self.stop_choice_norm = nn.LayerNorm((2 * self.hidden_dim) + stop_struct_dim)
        self.stop_choice_head = _build_mlp(
            input_dim=(2 * self.hidden_dim) + stop_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.failure_stop_norm = nn.LayerNorm(self.hidden_dim + stop_struct_dim)
        self.failure_stop_head = _build_mlp(
            input_dim=self.hidden_dim + stop_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
        self.relation_norm = nn.LayerNorm((3 * self.hidden_dim) + relation_struct_dim)
        self.relation_head = _build_mlp(
            input_dim=(3 * self.hidden_dim) + relation_struct_dim,
            output_dim=1,
            hidden_dim=int(actor["hidden_dim"]),
            num_layers=int(actor["num_layers"]),
            dropout=float(actor["dropout"]),
        )
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
        # A moderate edge batch keeps the deeper-step actor kernels large enough
        # to amortize launch overhead without recreating the giant full-support
        # candidate surface in one allocation.
        self.edge_logit_chunk_size = 4096

    @staticmethod
    def _resolve_action_pruning_limits(
        action_pruning: Mapping[str, Any] | None,
    ) -> tuple[int | None, int | None]:
        if action_pruning is None:
            return None, None
        per_node_top_k = int(action_pruning.get("per_node_top_k", 0)) or None
        per_state_top_k = int(action_pruning.get("per_state_top_k", 0)) or None
        return per_node_top_k, per_state_top_k

    @staticmethod
    def _log_action_distribution_stats(
        *,
        active_state_count: int,
        total_raw_candidates: int,
        max_raw_candidates: int,
        total_stop_choices: int,
        max_stop_choices: int,
    ) -> None:
        if not cuda_memory_profiling_enabled():
            return
        mean_raw_candidates = (
            float(total_raw_candidates) / float(active_state_count)
            if active_state_count > 0
            else 0.0
        )
        logger.info(
            "actor_action_distribution_stats active_states=%d total_raw_candidates=%d mean_raw_candidates=%.1f max_raw_candidates=%d total_stop_choices=%d max_stop_choices=%d",
            active_state_count,
            total_raw_candidates,
            mean_raw_candidates,
            max_raw_candidates,
            total_stop_choices,
            max_stop_choices,
        )

    @staticmethod
    def _estimate_next_components(
        *,
        analysis: SubgraphAnalysis,
        source_graph_node: int,
        target_graph_node: int,
    ) -> int:
        current_components = int(analysis.anchor_component_count)
        source_component = int(
            analysis.component_labels.get(int(source_graph_node), -1)
        )
        target_component = int(
            analysis.component_labels.get(int(target_graph_node), -1)
        )
        if (
            source_component >= 0
            and target_component >= 0
            and int(source_component) != int(target_component)
        ):
            return max(current_components - 1, 1)
        return current_components

    def _build_node_logits(
        self,
        *,
        state_features: torch.Tensor,
        node_features: torch.Tensor,
        node_is_anchor: torch.Tensor,
        node_reachability_bits: torch.Tensor,
        max_anchor_counts: torch.Tensor,
        full_masks: torch.Tensor,
        node_candidate_counts: torch.Tensor,
    ) -> torch.Tensor:
        if int(node_features.size(0)) <= 0:
            return state_features.new_empty((0,), dtype=torch.float32)
        device = node_features.device
        max_anchor_counts_float = max_anchor_counts.to(
            device=device, dtype=torch.float32
        ).clamp(min=1.0)
        coverage = (
            torch.tensor(
                [
                    float(_bit_count(int(bits)))
                    for bits in node_reachability_bits.detach().cpu().tolist()
                ],
                device=device,
                dtype=torch.float32,
            )
            / max_anchor_counts_float
        )
        full_coverage = ((full_masks > 0) & (node_reachability_bits == full_masks)).to(
            device=device, dtype=torch.float32
        )
        struct = torch.stack(
            (
                node_is_anchor.to(device=device, dtype=torch.float32),
                coverage,
                full_coverage,
                torch.log1p(
                    node_candidate_counts.to(device=device, dtype=torch.float32)
                ),
            ),
            dim=-1,
        )
        focus_inputs = torch.cat((state_features, node_features, struct), dim=-1)
        focus_inputs = align_float_input_dtype(
            focus_inputs, module=self.node_focus_norm
        )
        focus_inputs = self.node_focus_norm(focus_inputs)
        focus_inputs = align_float_input_dtype(
            focus_inputs, module=self.node_focus_head[0]
        )
        return self.node_focus_head(focus_inputs).squeeze(-1).to(dtype=torch.float32)

    def _build_node_logits_batch(
        self,
        *,
        state_feature: torch.Tensor,
        node_features: torch.Tensor,
        node_is_anchor: torch.Tensor,
        node_reachability_bits: torch.Tensor,
        max_anchor_count: int,
        full_mask: int,
        node_candidate_counts: torch.Tensor,
    ) -> torch.Tensor:
        if int(node_features.size(0)) <= 0:
            return state_feature.new_empty((0,), dtype=torch.float32)
        count = int(node_features.size(0))
        device = node_features.device
        return self._build_node_logits(
            state_features=state_feature.unsqueeze(0).expand(count, -1),
            node_features=node_features,
            node_is_anchor=node_is_anchor,
            node_reachability_bits=node_reachability_bits,
            max_anchor_counts=torch.full(
                (count,),
                fill_value=max(int(max_anchor_count), 1),
                device=device,
                dtype=torch.long,
            ),
            full_masks=torch.full(
                (count,),
                fill_value=int(full_mask),
                device=device,
                dtype=torch.long,
            ),
            node_candidate_counts=node_candidate_counts,
        )

    def _build_relation_logits(
        self,
        *,
        state_features: torch.Tensor,
        src_features: torch.Tensor,
        relation_features: torch.Tensor,
        relation_num_edges: torch.Tensor,
        relation_max_new_bit_gain: torch.Tensor,
        relation_max_candidate_commit_counts: torch.Tensor,
        relation_max_semantic_overlap: torch.Tensor,
    ) -> torch.Tensor:
        if int(relation_features.size(0)) <= 0:
            return state_features.new_empty((0,), dtype=torch.float32)
        device = relation_features.device
        relation_inputs = torch.cat(
            (
                state_features,
                src_features,
                relation_features,
                torch.stack(
                    (
                        torch.log1p(
                            relation_num_edges.to(device=device, dtype=torch.float32)
                        ),
                        relation_max_new_bit_gain.to(
                            device=device, dtype=torch.float32
                        ),
                        relation_max_candidate_commit_counts.to(
                            device=device, dtype=torch.float32
                        ),
                        relation_max_semantic_overlap.to(
                            device=device, dtype=torch.float32
                        ),
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        relation_inputs = align_float_input_dtype(
            relation_inputs, module=self.relation_norm
        )
        relation_inputs = self.relation_norm(relation_inputs)
        relation_inputs = align_float_input_dtype(
            relation_inputs, module=self.relation_head[0]
        )
        return self.relation_head(relation_inputs).squeeze(-1).to(dtype=torch.float32)

    def _build_relation_logits_batch(
        self,
        *,
        state_feature: torch.Tensor,
        src_features: torch.Tensor,
        relation_features: torch.Tensor,
        relation_num_edges: torch.Tensor,
        relation_max_new_bit_gain: torch.Tensor,
        relation_max_candidate_commit_counts: torch.Tensor,
        relation_max_semantic_overlap: torch.Tensor,
    ) -> torch.Tensor:
        if int(relation_features.size(0)) <= 0:
            return state_feature.new_empty((0,), dtype=torch.float32)
        return self._build_relation_logits(
            state_features=state_feature.unsqueeze(0).expand(
                int(relation_features.size(0)), -1
            ),
            src_features=src_features,
            relation_features=relation_features,
            relation_num_edges=relation_num_edges,
            relation_max_new_bit_gain=relation_max_new_bit_gain,
            relation_max_candidate_commit_counts=relation_max_candidate_commit_counts,
            relation_max_semantic_overlap=relation_max_semantic_overlap,
        )

    def _build_edge_logits(
        self,
        *,
        state_features: torch.Tensor,
        src_features: torch.Tensor,
        relation_features: torch.Tensor,
        dst_features: torch.Tensor,
        current_components: torch.Tensor,
        next_component_counts: torch.Tensor,
        semantic_overlap: torch.Tensor,
        action_new_bit_gain: torch.Tensor,
        candidate_commit_counts: torch.Tensor,
    ) -> torch.Tensor:
        if int(src_features.size(0)) <= 0:
            return state_features.new_empty((0,), dtype=torch.float32)
        logits_chunks: list[torch.Tensor] = []
        total = int(src_features.size(0))
        for start in range(0, total, int(self.edge_logit_chunk_size)):
            stop = min(start + int(self.edge_logit_chunk_size), total)
            chunk = slice(start, stop)
            chunk_src = src_features[chunk]
            device = chunk_src.device
            candidate_inputs = torch.cat(
                (
                    chunk_src,
                    relation_features[chunk],
                    dst_features[chunk],
                    torch.stack(
                        (
                            torch.ones(
                                (int(chunk_src.size(0)),),
                                device=device,
                                dtype=torch.float32,
                            ),
                            semantic_overlap[chunk].to(
                                device=device, dtype=torch.float32
                            ),
                            action_new_bit_gain[chunk].to(
                                device=device, dtype=torch.float32
                            ),
                            current_components[chunk].to(
                                device=device, dtype=torch.float32
                            ),
                            next_component_counts[chunk].to(
                                device=device, dtype=torch.float32
                            ),
                            candidate_commit_counts[chunk].to(
                                device=device, dtype=torch.float32
                            ),
                        ),
                        dim=-1,
                    ),
                ),
                dim=-1,
            )
            candidate_inputs = align_float_input_dtype(
                candidate_inputs, module=self.candidate_encoder_norm
            )
            candidate_inputs = self.candidate_encoder_norm(candidate_inputs)
            candidate_inputs = align_float_input_dtype(
                candidate_inputs, module=self.candidate_encoder[0]
            )
            candidate_features = self.candidate_encoder(candidate_inputs)
            actor_query = self.action_head.encode_query(state_features[chunk])
            edge_key = self.action_head.encode_edge_keys(
                candidate_state_features=candidate_features,
                relation_features=relation_features[chunk],
            )
            logits_chunks.append(
                self.action_head.score_from_encoded(
                    actor_query=actor_query,
                    edge_key=edge_key,
                ).to(dtype=torch.float32)
            )
        return torch.cat(logits_chunks, dim=0)

    def _build_edge_logits_batch(
        self,
        *,
        state_feature: torch.Tensor,
        src_features: torch.Tensor,
        relation_features: torch.Tensor,
        dst_features: torch.Tensor,
        current_components: int,
        next_component_counts: torch.Tensor,
        semantic_overlap: torch.Tensor,
        action_new_bit_gain: torch.Tensor,
        candidate_commit_counts: torch.Tensor,
    ) -> torch.Tensor:
        if int(src_features.size(0)) <= 0:
            return state_feature.new_empty((0,), dtype=torch.float32)
        count = int(src_features.size(0))
        device = src_features.device
        return self._build_edge_logits(
            state_features=state_feature.unsqueeze(0).expand(count, -1),
            src_features=src_features,
            relation_features=relation_features,
            dst_features=dst_features,
            current_components=torch.full(
                (count,),
                fill_value=int(current_components),
                device=device,
                dtype=torch.long,
            ),
            next_component_counts=next_component_counts,
            semantic_overlap=semantic_overlap,
            action_new_bit_gain=action_new_bit_gain,
            candidate_commit_counts=candidate_commit_counts,
        )

    def _build_stop_choice_logits(
        self,
        *,
        state_features: torch.Tensor,
        answer_features: torch.Tensor,
        support_node_counts: torch.Tensor,
        current_components: torch.Tensor,
        current_edges: torch.Tensor,
    ) -> torch.Tensor:
        if int(answer_features.size(0)) <= 0:
            return state_features.new_empty((0,), dtype=torch.float32)
        device = answer_features.device
        support_node_counts_float = support_node_counts.to(
            device=device, dtype=torch.float32
        )
        stop_inputs = torch.cat(
            (
                state_features,
                answer_features,
                torch.stack(
                    (
                        torch.log1p(support_node_counts_float),
                        current_components.to(device=device, dtype=torch.float32),
                        torch.log1p(
                            current_edges.to(device=device, dtype=torch.float32)
                        ),
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        stop_inputs = align_float_input_dtype(stop_inputs, module=self.stop_choice_norm)
        stop_inputs = self.stop_choice_norm(stop_inputs)
        stop_inputs = align_float_input_dtype(
            stop_inputs, module=self.stop_choice_head[0]
        )
        return self.stop_choice_head(stop_inputs).squeeze(-1).to(dtype=torch.float32)

    def _build_stop_choice_logits_batch(
        self,
        *,
        state_feature: torch.Tensor,
        answer_features: torch.Tensor,
        support_node_counts: torch.Tensor,
        current_components: int,
        current_edges: int,
    ) -> torch.Tensor:
        if int(answer_features.size(0)) <= 0:
            return state_feature.new_empty((0,), dtype=torch.float32)
        count = int(answer_features.size(0))
        device = answer_features.device
        return self._build_stop_choice_logits(
            state_features=state_feature.unsqueeze(0).expand(count, -1),
            answer_features=answer_features,
            support_node_counts=support_node_counts,
            current_components=torch.full(
                (count,),
                fill_value=int(current_components),
                device=device,
                dtype=torch.long,
            ),
            current_edges=torch.full(
                (count,),
                fill_value=int(current_edges),
                device=device,
                dtype=torch.long,
            ),
        )

    def _build_failure_stop_logits(
        self,
        *,
        state_features: torch.Tensor,
        num_answer_ready: torch.Tensor,
        current_components: torch.Tensor,
        current_edges: torch.Tensor,
    ) -> torch.Tensor:
        if int(state_features.size(0)) <= 0:
            return state_features.new_empty((0,), dtype=torch.float32)
        device = state_features.device
        failure_inputs = torch.cat(
            (
                state_features,
                torch.stack(
                    (
                        num_answer_ready.to(device=device, dtype=torch.float32),
                        current_components.to(device=device, dtype=torch.float32),
                        torch.log1p(
                            current_edges.to(device=device, dtype=torch.float32)
                        ),
                    ),
                    dim=-1,
                ),
            ),
            dim=-1,
        )
        failure_inputs = align_float_input_dtype(
            failure_inputs, module=self.failure_stop_norm
        )
        failure_inputs = self.failure_stop_norm(failure_inputs)
        failure_inputs = align_float_input_dtype(
            failure_inputs, module=self.failure_stop_head[0]
        )
        return (
            self.failure_stop_head(failure_inputs).squeeze(-1).to(dtype=torch.float32)
        )

    def _build_failure_stop_logit(
        self,
        *,
        state_feature: torch.Tensor,
        num_answer_ready: int,
        current_components: int,
        current_edges: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._build_failure_stop_logits(
            state_features=state_feature.unsqueeze(0),
            num_answer_ready=torch.tensor(
                [int(num_answer_ready)], device=device, dtype=torch.long
            ),
            current_components=torch.tensor(
                [int(current_components)], device=device, dtype=torch.long
            ),
            current_edges=torch.tensor(
                [int(current_edges)], device=device, dtype=torch.long
            ),
        ).squeeze(0)

    def _collect_state_candidates(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        graph_idx: int,
        state: Any,
        analysis: SubgraphAnalysis,
        current_components: int,
        current_oracle_distance: float,
        full_mask: int,
        anchor_nodes: set[int],
        reachable_entity_ids: set[int],
        per_node_top_k: int | None,
        per_state_top_k: int | None,
    ) -> _LinearizedStateChoices:
        device = prepared_batch.device
        if int(state.num_edges) >= self.max_steps:
            return _LinearizedStateChoices.empty(device=device)

        selected_edge_ids = {int(edge_id) for edge_id in state.edge_ids}
        candidate_edge_ids_list: list[int] = []
        candidate_source_graph_nodes_list: list[int] = []
        graph_outgoing = prepared_batch.graph_outgoing_edge_ids[int(graph_idx)]
        for graph_node_id in analysis.selected_node_ids:
            outgoing_edge_ids = graph_outgoing.get(int(graph_node_id), ())
            if per_node_top_k is not None:
                outgoing_edge_ids = outgoing_edge_ids[: int(per_node_top_k)]
            for edge_id in outgoing_edge_ids:
                if int(edge_id) in selected_edge_ids:
                    continue
                candidate_edge_ids_list.append(int(edge_id))
                candidate_source_graph_nodes_list.append(int(graph_node_id))
        if not candidate_edge_ids_list:
            return _LinearizedStateChoices.empty(device=device)

        candidate_edge_ids = torch.tensor(
            candidate_edge_ids_list,
            device=device,
            dtype=torch.long,
        )
        candidate_source_graph_nodes = torch.tensor(
            candidate_source_graph_nodes_list,
            device=device,
            dtype=torch.long,
        )
        candidate_question_similarity = (
            prepared_batch.edge_question_similarity.index_select(
                0, candidate_edge_ids
            ).to(dtype=torch.float32)
        )

        if per_state_top_k is not None and len(candidate_edge_ids_list) > int(
            per_state_top_k
        ):
            question_similarity_values = [
                float(value)
                for value in candidate_question_similarity.detach().cpu().tolist()
            ]
            keep_indices = sorted(
                range(len(candidate_edge_ids_list)),
                key=lambda idx: (
                    -float(question_similarity_values[idx]),
                    int(candidate_edge_ids_list[idx]),
                ),
            )[: int(per_state_top_k)]
            keep_tensor = torch.tensor(keep_indices, device=device, dtype=torch.long)
            candidate_edge_ids_list = [
                int(candidate_edge_ids_list[idx]) for idx in keep_indices
            ]
            candidate_source_graph_nodes_list = [
                int(candidate_source_graph_nodes_list[idx]) for idx in keep_indices
            ]
            candidate_edge_ids = candidate_edge_ids.index_select(0, keep_tensor)
            candidate_source_graph_nodes = candidate_source_graph_nodes.index_select(
                0, keep_tensor
            )
            candidate_question_similarity = candidate_question_similarity.index_select(
                0, keep_tensor
            )

        candidate_target_graph_nodes = prepared_batch.topology.edge_index[
            1
        ].index_select(0, candidate_edge_ids)
        candidate_relation_ids = prepared_batch.topology.edge_type.index_select(
            0, candidate_edge_ids
        )
        candidate_target_graph_nodes_list = [
            int(value) for value in candidate_target_graph_nodes.detach().cpu().tolist()
        ]
        candidate_relation_ids_list = [
            int(value) for value in candidate_relation_ids.detach().cpu().tolist()
        ]
        candidate_question_similarity_values = [
            float(value)
            for value in candidate_question_similarity.detach().cpu().tolist()
        ]
        candidate_target_entity_ids_list = [
            int(value)
            for value in prepared_batch.node_entity_ids.index_select(
                0, candidate_target_graph_nodes
            )
            .detach()
            .cpu()
            .tolist()
        ]
        oracle_distance_map = prepared_batch.graph_oracle_answer_distance[
            int(graph_idx)
        ]
        candidate_source_bits_list = [
            int(analysis.reachability_bits.get(int(graph_node_id), 0))
            for graph_node_id in candidate_source_graph_nodes_list
        ]
        candidate_current_entity_bits_list = [
            int(analysis.entity_reachability_bits.get(int(entity_id), 0))
            for entity_id in candidate_target_entity_ids_list
        ]
        candidate_next_component_counts_list = [
            self._estimate_next_components(
                analysis=analysis,
                source_graph_node=int(source_graph_node),
                target_graph_node=int(target_graph_node),
            )
            for source_graph_node, target_graph_node in zip(
                candidate_source_graph_nodes_list,
                candidate_target_graph_nodes_list,
            )
        ]
        candidate_semantic_overlap_list = [
            1.0 if int(entity_id) in reachable_entity_ids else 0.0
            for entity_id in candidate_target_entity_ids_list
        ]
        candidate_commit_counts_list = [
            1
            if int(full_mask) > 0
            and (int(current_entity_bits) | int(source_bits)) == int(full_mask)
            else 0
            for source_bits, current_entity_bits in zip(
                candidate_source_bits_list,
                candidate_current_entity_bits_list,
            )
        ]
        candidate_new_bit_gain_list = [
            int(_bit_count(int(source_bits) & ~int(current_entity_bits)))
            for source_bits, current_entity_bits in zip(
                candidate_source_bits_list,
                candidate_current_entity_bits_list,
            )
        ]
        candidate_target_answer_distance_list = [
            float(oracle_distance_map.get(int(graph_node_id), -1))
            for graph_node_id in candidate_target_graph_nodes_list
        ]

        candidates_by_node: dict[int, list[int]] = {}
        for candidate_idx, graph_node_id in enumerate(
            candidate_source_graph_nodes_list
        ):
            candidates_by_node.setdefault(int(graph_node_id), []).append(
                int(candidate_idx)
            )

        ordered_candidate_indices: list[int] = []
        node_choice_graph_node_ids_list: list[int] = []
        node_choice_candidate_counts_list: list[int] = []
        node_choice_reachability_bits_list: list[int] = []
        node_choice_is_anchor_list: list[bool] = []
        relation_choice_relation_ids_list: list[int] = []
        relation_choice_node_choice_indices_list: list[int] = []
        relation_choice_num_edges_list: list[int] = []
        relation_choice_max_new_bit_gain_list: list[int] = []
        relation_choice_max_candidate_commit_counts_list: list[int] = []
        relation_choice_max_semantic_overlap_list: list[float] = []
        edge_choice_relation_choice_indices_list: list[int] = []
        node_relation_ptr = [0]
        relation_edge_ptr = [0]
        relation_choice_count = 0

        for node_choice_idx, graph_node_id in enumerate(sorted(candidates_by_node)):
            node_candidate_indices = candidates_by_node[int(graph_node_id)]
            node_choice_graph_node_ids_list.append(int(graph_node_id))
            node_choice_candidate_counts_list.append(int(len(node_candidate_indices)))
            node_choice_reachability_bits_list.append(
                int(analysis.reachability_bits.get(int(graph_node_id), 0))
            )
            node_choice_is_anchor_list.append(int(graph_node_id) in anchor_nodes)
            relation_groups: dict[int, list[int]] = {}
            for candidate_idx in node_candidate_indices:
                relation_groups.setdefault(
                    int(candidate_relation_ids_list[int(candidate_idx)]), []
                ).append(int(candidate_idx))
            for relation_id in sorted(relation_groups):
                relation_candidate_indices = relation_groups[int(relation_id)]
                relation_choice_relation_ids_list.append(int(relation_id))
                relation_choice_node_choice_indices_list.append(int(node_choice_idx))
                relation_choice_num_edges_list.append(
                    int(len(relation_candidate_indices))
                )
                relation_choice_max_new_bit_gain_list.append(
                    max(
                        int(candidate_new_bit_gain_list[int(candidate_idx)])
                        for candidate_idx in relation_candidate_indices
                    )
                )
                relation_choice_max_candidate_commit_counts_list.append(
                    max(
                        int(candidate_commit_counts_list[int(candidate_idx)])
                        for candidate_idx in relation_candidate_indices
                    )
                )
                relation_choice_max_semantic_overlap_list.append(
                    max(
                        float(candidate_semantic_overlap_list[int(candidate_idx)])
                        for candidate_idx in relation_candidate_indices
                    )
                )
                ordered_candidate_indices.extend(relation_candidate_indices)
                edge_choice_relation_choice_indices_list.extend(
                    [int(relation_choice_count)] * int(len(relation_candidate_indices))
                )
                relation_choice_count += 1
                relation_edge_ptr.append(int(len(ordered_candidate_indices)))
            node_relation_ptr.append(int(relation_choice_count))

        order_tensor = torch.tensor(
            ordered_candidate_indices,
            device=device,
            dtype=torch.long,
        )
        return _LinearizedStateChoices(
            edge_choice_edge_ids=candidate_edge_ids.index_select(0, order_tensor),
            edge_choice_source_graph_nodes=candidate_source_graph_nodes.index_select(
                0, order_tensor
            ),
            edge_choice_relation_ids=candidate_relation_ids.index_select(
                0, order_tensor
            ),
            edge_choice_target_graph_nodes=candidate_target_graph_nodes.index_select(
                0, order_tensor
            ),
            edge_choice_next_component_counts=torch.tensor(
                candidate_next_component_counts_list,
                device=device,
                dtype=torch.long,
            ).index_select(0, order_tensor),
            edge_choice_question_similarity=candidate_question_similarity.index_select(
                0, order_tensor
            ),
            edge_choice_semantic_overlap=torch.tensor(
                candidate_semantic_overlap_list,
                device=device,
                dtype=torch.float32,
            ).index_select(0, order_tensor),
            edge_choice_action_new_bit_gain=torch.tensor(
                candidate_new_bit_gain_list,
                device=device,
                dtype=torch.long,
            ).index_select(0, order_tensor),
            edge_choice_candidate_commit_counts=torch.tensor(
                candidate_commit_counts_list,
                device=device,
                dtype=torch.long,
            ).index_select(0, order_tensor),
            edge_choice_target_answer_distance=torch.tensor(
                candidate_target_answer_distance_list,
                device=device,
                dtype=torch.float32,
            ).index_select(0, order_tensor),
            edge_choice_relation_choice_indices=torch.tensor(
                edge_choice_relation_choice_indices_list,
                device=device,
                dtype=torch.long,
            ),
            node_choice_graph_node_ids=torch.tensor(
                node_choice_graph_node_ids_list,
                device=device,
                dtype=torch.long,
            ),
            node_choice_candidate_counts=torch.tensor(
                node_choice_candidate_counts_list,
                device=device,
                dtype=torch.long,
            ),
            node_choice_reachability_bits=torch.tensor(
                node_choice_reachability_bits_list,
                device=device,
                dtype=torch.long,
            ),
            node_choice_is_anchor=torch.tensor(
                node_choice_is_anchor_list,
                device=device,
                dtype=torch.bool,
            ),
            relation_choice_relation_ids=torch.tensor(
                relation_choice_relation_ids_list,
                device=device,
                dtype=torch.long,
            ),
            relation_choice_node_choice_indices=torch.tensor(
                relation_choice_node_choice_indices_list,
                device=device,
                dtype=torch.long,
            ),
            relation_choice_num_edges=torch.tensor(
                relation_choice_num_edges_list,
                device=device,
                dtype=torch.long,
            ),
            relation_choice_max_new_bit_gain=torch.tensor(
                relation_choice_max_new_bit_gain_list,
                device=device,
                dtype=torch.long,
            ),
            relation_choice_max_candidate_commit_counts=torch.tensor(
                relation_choice_max_candidate_commit_counts_list,
                device=device,
                dtype=torch.long,
            ),
            relation_choice_max_semantic_overlap=torch.tensor(
                relation_choice_max_semantic_overlap_list,
                device=device,
                dtype=torch.float32,
            ),
            node_relation_ptr=torch.tensor(
                node_relation_ptr,
                device=device,
                dtype=torch.long,
            ),
            relation_edge_ptr=torch.tensor(
                relation_edge_ptr,
                device=device,
                dtype=torch.long,
            ),
        )

    def _collect_state_distribution_context(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        flat_state_index: int,
        analysis: SubgraphAnalysis,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> _StateDistributionContext:
        device = prepared_batch.device
        flat_state_idx = int(flat_state_index)
        graph_idx = int(rollout_batch.graph_ids[flat_state_idx].item())
        state = rollout_batch.states[flat_state_idx]
        current_components = int(analysis.anchor_component_count)
        commit_set = resolve_admissible_answer_commit_set(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            analysis=analysis,
        )
        current_oracle_distance = float(
            _oracle_distance(
                prepared_batch=prepared_batch,
                graph_idx=graph_idx,
                analysis=analysis,
            )
        )
        full_mask = int(prepared_batch.graph_anchor_full_mask[graph_idx])
        max_anchor_count = max(_bit_count(full_mask), 1)
        anchor_nodes = {
            int(node_id)
            for node_id in prepared_batch.graph_anchor_abs_node_ids[int(graph_idx)]
        }
        reachable_entity_ids = {
            int(entity_id) for entity_id in analysis.entity_reachability_bits
        }
        per_node_top_k, per_state_top_k = self._resolve_action_pruning_limits(
            action_pruning
        )
        linearized = self._collect_state_candidates(
            prepared_batch=prepared_batch,
            graph_idx=graph_idx,
            state=state,
            analysis=analysis,
            current_components=current_components,
            current_oracle_distance=current_oracle_distance,
            full_mask=full_mask,
            anchor_nodes=anchor_nodes,
            reachable_entity_ids=reachable_entity_ids,
            per_node_top_k=per_node_top_k,
            per_state_top_k=per_state_top_k,
        )
        answer_ready_entities = tuple(
            int(entity_id) for entity_id in commit_set.entities
        )
        if answer_ready_entities:
            answer_features: list[torch.Tensor] = []
            support_node_counts: list[int] = []
            for answer_entity_id in answer_ready_entities:
                answer_feature, support_node_count = _answer_ready_entity_pool(
                    prepared_batch=prepared_batch,
                    analysis=analysis,
                    entity_id=int(answer_entity_id),
                    full_mask=full_mask,
                    device=device,
                )
                answer_features.append(answer_feature)
                support_node_counts.append(int(support_node_count))
            answer_choice_answer_entity_ids = torch.tensor(
                answer_ready_entities,
                device=device,
                dtype=torch.long,
            )
            answer_choice_support_node_counts = torch.tensor(
                support_node_counts,
                device=device,
                dtype=torch.long,
            )
            answer_choice_features = torch.stack(answer_features, dim=0)
        else:
            answer_choice_answer_entity_ids = torch.empty(
                (0,), device=device, dtype=torch.long
            )
            answer_choice_support_node_counts = torch.empty(
                (0,), device=device, dtype=torch.long
            )
            answer_choice_features = prepared_batch.node_tokens.new_empty(
                (0, int(prepared_batch.node_tokens.size(-1)))
            )
        return _StateDistributionContext(
            flat_state_index=flat_state_idx,
            current_components=current_components,
            current_commit_candidate_count=int(commit_set.count),
            current_oracle_distance=float(current_oracle_distance),
            max_anchor_count=max_anchor_count,
            full_mask=full_mask,
            state_num_edges=int(state.num_edges),
            linearized=linearized,
            answer_choice_answer_entity_ids=answer_choice_answer_entity_ids,
            answer_choice_support_node_counts=answer_choice_support_node_counts,
            answer_choice_features=answer_choice_features,
        )

    def build_state_distribution(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        flat_state_index: int,
        analysis: SubgraphAnalysis,
        state_feature: torch.Tensor,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> tuple[HierarchicalStateActionDistribution, _ActionDistributionBuildStats]:
        device = prepared_batch.device
        context = self._collect_state_distribution_context(
            prepared_batch=prepared_batch,
            rollout_batch=rollout_batch,
            flat_state_index=int(flat_state_index),
            analysis=analysis,
            action_pruning=action_pruning,
        )
        flat_state_idx = int(context.flat_state_index)
        linearized = context.linearized

        if int(linearized.node_choice_graph_node_ids.numel()) > 0:
            node_features = prepared_batch.node_tokens.index_select(
                0, linearized.node_choice_graph_node_ids
            )
            node_choice_logits = self._build_node_logits_batch(
                state_feature=state_feature,
                node_features=node_features,
                node_is_anchor=linearized.node_choice_is_anchor,
                node_reachability_bits=linearized.node_choice_reachability_bits,
                max_anchor_count=context.max_anchor_count,
                full_mask=context.full_mask,
                node_candidate_counts=linearized.node_choice_candidate_counts,
            )
        else:
            node_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

        if int(linearized.relation_choice_relation_ids.numel()) > 0:
            relation_source_graph_nodes = (
                linearized.node_choice_graph_node_ids.index_select(
                    0, linearized.relation_choice_node_choice_indices
                )
            )
            relation_choice_logits = self._build_relation_logits_batch(
                state_feature=state_feature,
                src_features=prepared_batch.node_tokens.index_select(
                    0, relation_source_graph_nodes
                ),
                relation_features=prepared_batch.relation_tokens.index_select(
                    0, linearized.relation_choice_relation_ids
                ),
                relation_num_edges=linearized.relation_choice_num_edges,
                relation_max_new_bit_gain=linearized.relation_choice_max_new_bit_gain,
                relation_max_candidate_commit_counts=(
                    linearized.relation_choice_max_candidate_commit_counts
                ),
                relation_max_semantic_overlap=(
                    linearized.relation_choice_max_semantic_overlap
                ),
            )
        else:
            relation_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

        if int(linearized.edge_choice_edge_ids.numel()) > 0:
            edge_choice_logits = self._build_edge_logits_batch(
                state_feature=state_feature,
                src_features=prepared_batch.node_tokens.index_select(
                    0, linearized.edge_choice_source_graph_nodes
                ),
                relation_features=prepared_batch.relation_tokens.index_select(
                    0, linearized.edge_choice_relation_ids
                ),
                dst_features=prepared_batch.node_tokens.index_select(
                    0, linearized.edge_choice_target_graph_nodes
                ),
                current_components=context.current_components,
                next_component_counts=linearized.edge_choice_next_component_counts,
                semantic_overlap=linearized.edge_choice_semantic_overlap,
                action_new_bit_gain=linearized.edge_choice_action_new_bit_gain,
                candidate_commit_counts=(
                    linearized.edge_choice_candidate_commit_counts
                ),
            )
        else:
            edge_choice_logits = state_feature.new_empty((0,), dtype=torch.float32)

        failure_stop_logit = self._build_failure_stop_logit(
            state_feature=state_feature,
            num_answer_ready=int(context.answer_choice_answer_entity_ids.numel()),
            current_components=context.current_components,
            current_edges=context.state_num_edges,
            device=device,
        )
        stop_choice_answer_entity_ids = torch.full(
            (1,), fill_value=-1, device=device, dtype=torch.long
        )
        stop_choice_support_node_counts = torch.zeros(
            (1,), device=device, dtype=torch.long
        )
        stop_choice_logits = failure_stop_logit.unsqueeze(0)
        if int(context.answer_choice_answer_entity_ids.numel()) > 0:
            answer_stop_logits = self._build_stop_choice_logits_batch(
                state_feature=state_feature,
                answer_features=context.answer_choice_features,
                support_node_counts=context.answer_choice_support_node_counts,
                current_components=context.current_components,
                current_edges=context.state_num_edges,
            )
            stop_choice_answer_entity_ids = torch.cat(
                (
                    stop_choice_answer_entity_ids,
                    context.answer_choice_answer_entity_ids,
                ),
                dim=0,
            )
            stop_choice_support_node_counts = torch.cat(
                (
                    stop_choice_support_node_counts,
                    context.answer_choice_support_node_counts,
                ),
                dim=0,
            )
            stop_choice_logits = torch.cat(
                (stop_choice_logits, answer_stop_logits),
                dim=0,
            )

        stop_logit = self.stop_head(state_feature.unsqueeze(0)).squeeze(0).squeeze(-1)
        stop_logit = stop_logit.to(dtype=torch.float32)
        continue_logit = (
            self.continue_head(state_feature.unsqueeze(0)).squeeze(0).squeeze(-1)
        )
        continue_logit = continue_logit.to(dtype=torch.float32)
        if (
            int(linearized.node_choice_graph_node_ids.numel()) <= 0
            or int(context.state_num_edges) >= self.max_steps
        ):
            continue_logit = torch.full_like(continue_logit, float("-inf"))

        distribution = HierarchicalStateActionDistribution(
            flat_state_index=flat_state_idx,
            stop_logit=stop_logit,
            continue_logit=continue_logit,
            stop_choice_logits=stop_choice_logits.to(dtype=torch.float32),
            stop_choice_answer_entity_ids=stop_choice_answer_entity_ids,
            stop_choice_support_node_counts=stop_choice_support_node_counts,
            node_choice_graph_node_ids=linearized.node_choice_graph_node_ids,
            node_choice_logits=node_choice_logits,
            relation_choice_relation_ids=linearized.relation_choice_relation_ids,
            relation_choice_logits=relation_choice_logits,
            relation_choice_node_choice_indices=(
                linearized.relation_choice_node_choice_indices
            ),
            edge_choice_edge_ids=linearized.edge_choice_edge_ids,
            edge_choice_source_graph_nodes=linearized.edge_choice_source_graph_nodes,
            edge_choice_relation_ids=linearized.edge_choice_relation_ids,
            edge_choice_target_graph_nodes=linearized.edge_choice_target_graph_nodes,
            edge_choice_logits=edge_choice_logits,
            edge_choice_next_component_counts=(
                linearized.edge_choice_next_component_counts
            ),
            edge_choice_question_similarity=(
                linearized.edge_choice_question_similarity
            ),
            edge_choice_semantic_overlap=linearized.edge_choice_semantic_overlap,
            edge_choice_action_new_bit_gain=(
                linearized.edge_choice_action_new_bit_gain
            ),
            edge_choice_candidate_commit_counts=(
                linearized.edge_choice_candidate_commit_counts
            ),
            edge_choice_target_answer_distance=(
                linearized.edge_choice_target_answer_distance
            ),
            edge_choice_relation_choice_indices=(
                linearized.edge_choice_relation_choice_indices
            ),
            node_relation_ptr=linearized.node_relation_ptr,
            relation_edge_ptr=linearized.relation_edge_ptr,
            current_component_count=int(context.current_components),
            current_commit_candidate_count=int(context.current_commit_candidate_count),
            current_oracle_distance=float(context.current_oracle_distance),
        )
        return (
            distribution,
            _ActionDistributionBuildStats(
                raw_candidate_count=int(linearized.edge_choice_edge_ids.numel()),
                stop_choice_count=int(distribution.stop_choice_logits.numel()),
            ),
        )

    def build_action_distribution(
        self,
        *,
        prepared_batch: SubgraphPreparedBatch,
        rollout_batch: SubgraphRolloutBatch,
        analyses: tuple[SubgraphAnalysis, ...],
        state_features: torch.Tensor,
        action_pruning: Mapping[str, Any] | None = None,
    ) -> SubgraphActionDistribution:
        device = prepared_batch.device
        active_state_indices = rollout_batch.active_state_indices()
        if not active_state_indices:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            empty_state = state_features.new_empty((0, self.hidden_dim))
            return SubgraphActionDistribution(
                flat_state_indices=empty,
                state_features=empty_state,
                state_distributions=(),
            )
        active_state_tensor = torch.tensor(
            active_state_indices,
            device=device,
            dtype=torch.long,
        )
        active_state_features = state_features.index_select(0, active_state_tensor)
        contexts = [
            self._collect_state_distribution_context(
                prepared_batch=prepared_batch,
                rollout_batch=rollout_batch,
                flat_state_index=int(flat_state_idx),
                analysis=analyses[int(flat_state_idx)],
                action_pruning=action_pruning,
            )
            for flat_state_idx in active_state_indices
        ]
        num_states = int(len(contexts))
        current_components = torch.tensor(
            [int(context.current_components) for context in contexts],
            device=device,
            dtype=torch.long,
        )
        state_num_edges = torch.tensor(
            [int(context.state_num_edges) for context in contexts],
            device=device,
            dtype=torch.long,
        )
        answer_choice_counts = [
            int(context.answer_choice_answer_entity_ids.numel()) for context in contexts
        ]
        answer_choice_count_tensor = torch.tensor(
            answer_choice_counts,
            device=device,
            dtype=torch.long,
        )
        stop_logits = (
            self.stop_head(active_state_features).squeeze(-1).to(dtype=torch.float32)
        )
        continue_logits = (
            self.continue_head(active_state_features)
            .squeeze(-1)
            .to(dtype=torch.float32)
        )
        failure_stop_logits = self._build_failure_stop_logits(
            state_features=active_state_features,
            num_answer_ready=answer_choice_count_tensor,
            current_components=current_components,
            current_edges=state_num_edges,
        )

        def _offsets(counts: list[int]) -> list[int]:
            offsets = [0]
            for count in counts:
                offsets.append(int(offsets[-1] + int(count)))
            return offsets

        node_counts = [
            int(context.linearized.node_choice_graph_node_ids.numel())
            for context in contexts
        ]
        node_offsets = _offsets(node_counts)
        if node_offsets[-1] > 0:
            node_state_indices = torch.repeat_interleave(
                torch.arange(num_states, device=device, dtype=torch.long),
                torch.tensor(node_counts, device=device, dtype=torch.long),
            )
            node_graph_node_ids = torch.cat(
                [
                    context.linearized.node_choice_graph_node_ids
                    for context in contexts
                    if int(context.linearized.node_choice_graph_node_ids.numel()) > 0
                ],
                dim=0,
            )
            node_choice_logits = self._build_node_logits(
                state_features=active_state_features.index_select(
                    0, node_state_indices
                ),
                node_features=prepared_batch.node_tokens.index_select(
                    0, node_graph_node_ids
                ),
                node_is_anchor=torch.cat(
                    [
                        context.linearized.node_choice_is_anchor
                        for context in contexts
                        if int(context.linearized.node_choice_is_anchor.numel()) > 0
                    ],
                    dim=0,
                ),
                node_reachability_bits=torch.cat(
                    [
                        context.linearized.node_choice_reachability_bits
                        for context in contexts
                        if int(context.linearized.node_choice_reachability_bits.numel())
                        > 0
                    ],
                    dim=0,
                ),
                max_anchor_counts=torch.tensor(
                    [int(context.max_anchor_count) for context in contexts],
                    device=device,
                    dtype=torch.long,
                ).index_select(0, node_state_indices),
                full_masks=torch.tensor(
                    [int(context.full_mask) for context in contexts],
                    device=device,
                    dtype=torch.long,
                ).index_select(0, node_state_indices),
                node_candidate_counts=torch.cat(
                    [
                        context.linearized.node_choice_candidate_counts
                        for context in contexts
                        if int(context.linearized.node_choice_candidate_counts.numel())
                        > 0
                    ],
                    dim=0,
                ),
            )
        else:
            node_choice_logits = active_state_features.new_empty(
                (0,), dtype=torch.float32
            )

        relation_counts = [
            int(context.linearized.relation_choice_relation_ids.numel())
            for context in contexts
        ]
        relation_offsets = _offsets(relation_counts)
        if relation_offsets[-1] > 0:
            relation_state_indices = torch.repeat_interleave(
                torch.arange(num_states, device=device, dtype=torch.long),
                torch.tensor(relation_counts, device=device, dtype=torch.long),
            )
            relation_choice_relation_ids = torch.cat(
                [
                    context.linearized.relation_choice_relation_ids
                    for context in contexts
                    if int(context.linearized.relation_choice_relation_ids.numel()) > 0
                ],
                dim=0,
            )
            relation_source_graph_nodes = torch.cat(
                [
                    context.linearized.node_choice_graph_node_ids.index_select(
                        0, context.linearized.relation_choice_node_choice_indices
                    )
                    for context in contexts
                    if int(context.linearized.relation_choice_relation_ids.numel()) > 0
                ],
                dim=0,
            )
            relation_choice_logits = self._build_relation_logits(
                state_features=active_state_features.index_select(
                    0, relation_state_indices
                ),
                src_features=prepared_batch.node_tokens.index_select(
                    0, relation_source_graph_nodes
                ),
                relation_features=prepared_batch.relation_tokens.index_select(
                    0, relation_choice_relation_ids
                ),
                relation_num_edges=torch.cat(
                    [
                        context.linearized.relation_choice_num_edges
                        for context in contexts
                        if int(context.linearized.relation_choice_num_edges.numel()) > 0
                    ],
                    dim=0,
                ),
                relation_max_new_bit_gain=torch.cat(
                    [
                        context.linearized.relation_choice_max_new_bit_gain
                        for context in contexts
                        if int(
                            context.linearized.relation_choice_max_new_bit_gain.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
                relation_max_candidate_commit_counts=torch.cat(
                    [
                        context.linearized.relation_choice_max_candidate_commit_counts
                        for context in contexts
                        if int(
                            context.linearized.relation_choice_max_candidate_commit_counts.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
                relation_max_semantic_overlap=torch.cat(
                    [
                        context.linearized.relation_choice_max_semantic_overlap
                        for context in contexts
                        if int(
                            context.linearized.relation_choice_max_semantic_overlap.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
            )
        else:
            relation_choice_logits = active_state_features.new_empty(
                (0,), dtype=torch.float32
            )

        edge_counts = [
            int(context.linearized.edge_choice_edge_ids.numel()) for context in contexts
        ]
        edge_offsets = _offsets(edge_counts)
        if edge_offsets[-1] > 0:
            edge_state_indices = torch.repeat_interleave(
                torch.arange(num_states, device=device, dtype=torch.long),
                torch.tensor(edge_counts, device=device, dtype=torch.long),
            )
            edge_source_graph_nodes = torch.cat(
                [
                    context.linearized.edge_choice_source_graph_nodes
                    for context in contexts
                    if int(context.linearized.edge_choice_source_graph_nodes.numel())
                    > 0
                ],
                dim=0,
            )
            edge_relation_ids = torch.cat(
                [
                    context.linearized.edge_choice_relation_ids
                    for context in contexts
                    if int(context.linearized.edge_choice_relation_ids.numel()) > 0
                ],
                dim=0,
            )
            edge_target_graph_nodes = torch.cat(
                [
                    context.linearized.edge_choice_target_graph_nodes
                    for context in contexts
                    if int(context.linearized.edge_choice_target_graph_nodes.numel())
                    > 0
                ],
                dim=0,
            )
            edge_choice_logits = self._build_edge_logits(
                state_features=active_state_features.index_select(
                    0, edge_state_indices
                ),
                src_features=prepared_batch.node_tokens.index_select(
                    0, edge_source_graph_nodes
                ),
                relation_features=prepared_batch.relation_tokens.index_select(
                    0, edge_relation_ids
                ),
                dst_features=prepared_batch.node_tokens.index_select(
                    0, edge_target_graph_nodes
                ),
                current_components=current_components.index_select(
                    0, edge_state_indices
                ),
                next_component_counts=torch.cat(
                    [
                        context.linearized.edge_choice_next_component_counts
                        for context in contexts
                        if int(
                            context.linearized.edge_choice_next_component_counts.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
                semantic_overlap=torch.cat(
                    [
                        context.linearized.edge_choice_semantic_overlap
                        for context in contexts
                        if int(context.linearized.edge_choice_semantic_overlap.numel())
                        > 0
                    ],
                    dim=0,
                ),
                action_new_bit_gain=torch.cat(
                    [
                        context.linearized.edge_choice_action_new_bit_gain
                        for context in contexts
                        if int(
                            context.linearized.edge_choice_action_new_bit_gain.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
                candidate_commit_counts=torch.cat(
                    [
                        context.linearized.edge_choice_candidate_commit_counts
                        for context in contexts
                        if int(
                            context.linearized.edge_choice_candidate_commit_counts.numel()
                        )
                        > 0
                    ],
                    dim=0,
                ),
            )
        else:
            edge_choice_logits = active_state_features.new_empty(
                (0,), dtype=torch.float32
            )

        answer_offsets = _offsets(answer_choice_counts)
        if answer_offsets[-1] > 0:
            answer_state_indices = torch.repeat_interleave(
                torch.arange(num_states, device=device, dtype=torch.long),
                answer_choice_count_tensor,
            )
            answer_choice_logits = self._build_stop_choice_logits(
                state_features=active_state_features.index_select(
                    0, answer_state_indices
                ),
                answer_features=torch.cat(
                    [
                        context.answer_choice_features
                        for context in contexts
                        if int(context.answer_choice_features.size(0)) > 0
                    ],
                    dim=0,
                ),
                support_node_counts=torch.cat(
                    [
                        context.answer_choice_support_node_counts
                        for context in contexts
                        if int(context.answer_choice_support_node_counts.numel()) > 0
                    ],
                    dim=0,
                ),
                current_components=current_components.index_select(
                    0, answer_state_indices
                ),
                current_edges=state_num_edges.index_select(0, answer_state_indices),
            )
        else:
            answer_choice_logits = active_state_features.new_empty(
                (0,), dtype=torch.float32
            )

        state_distributions: list[HierarchicalStateActionDistribution] = []
        total_raw_candidates = 0
        max_raw_candidates = 0
        total_stop_choices = 0
        max_stop_choices = 0
        for local_state_idx, context in enumerate(contexts):
            node_choice_slice = slice(
                int(node_offsets[local_state_idx]),
                int(node_offsets[local_state_idx + 1]),
            )
            relation_choice_slice = slice(
                int(relation_offsets[local_state_idx]),
                int(relation_offsets[local_state_idx + 1]),
            )
            edge_choice_slice = slice(
                int(edge_offsets[local_state_idx]),
                int(edge_offsets[local_state_idx + 1]),
            )
            answer_choice_slice = slice(
                int(answer_offsets[local_state_idx]),
                int(answer_offsets[local_state_idx + 1]),
            )
            stop_choice_answer_entity_ids = torch.full(
                (1,), fill_value=-1, device=device, dtype=torch.long
            )
            stop_choice_support_node_counts = torch.zeros(
                (1,), device=device, dtype=torch.long
            )
            stop_choice_logits = failure_stop_logits[
                local_state_idx : local_state_idx + 1
            ]
            if int(context.answer_choice_answer_entity_ids.numel()) > 0:
                stop_choice_answer_entity_ids = torch.cat(
                    (
                        stop_choice_answer_entity_ids,
                        context.answer_choice_answer_entity_ids,
                    ),
                    dim=0,
                )
                stop_choice_support_node_counts = torch.cat(
                    (
                        stop_choice_support_node_counts,
                        context.answer_choice_support_node_counts,
                    ),
                    dim=0,
                )
                stop_choice_logits = torch.cat(
                    (
                        stop_choice_logits,
                        answer_choice_logits[answer_choice_slice],
                    ),
                    dim=0,
                )
            continue_logit = continue_logits[local_state_idx]
            if (
                int(context.linearized.node_choice_graph_node_ids.numel()) <= 0
                or int(context.state_num_edges) >= self.max_steps
            ):
                continue_logit = torch.full_like(continue_logit, float("-inf"))
            distribution = HierarchicalStateActionDistribution(
                flat_state_index=int(context.flat_state_index),
                stop_logit=stop_logits[local_state_idx],
                continue_logit=continue_logit,
                stop_choice_logits=stop_choice_logits,
                stop_choice_answer_entity_ids=stop_choice_answer_entity_ids,
                stop_choice_support_node_counts=stop_choice_support_node_counts,
                node_choice_graph_node_ids=context.linearized.node_choice_graph_node_ids,
                node_choice_logits=node_choice_logits[node_choice_slice],
                relation_choice_relation_ids=(
                    context.linearized.relation_choice_relation_ids
                ),
                relation_choice_logits=relation_choice_logits[relation_choice_slice],
                relation_choice_node_choice_indices=(
                    context.linearized.relation_choice_node_choice_indices
                ),
                edge_choice_edge_ids=context.linearized.edge_choice_edge_ids,
                edge_choice_source_graph_nodes=(
                    context.linearized.edge_choice_source_graph_nodes
                ),
                edge_choice_relation_ids=context.linearized.edge_choice_relation_ids,
                edge_choice_target_graph_nodes=(
                    context.linearized.edge_choice_target_graph_nodes
                ),
                edge_choice_logits=edge_choice_logits[edge_choice_slice],
                edge_choice_next_component_counts=(
                    context.linearized.edge_choice_next_component_counts
                ),
                edge_choice_question_similarity=(
                    context.linearized.edge_choice_question_similarity
                ),
                edge_choice_semantic_overlap=(
                    context.linearized.edge_choice_semantic_overlap
                ),
                edge_choice_action_new_bit_gain=(
                    context.linearized.edge_choice_action_new_bit_gain
                ),
                edge_choice_candidate_commit_counts=(
                    context.linearized.edge_choice_candidate_commit_counts
                ),
                edge_choice_target_answer_distance=(
                    context.linearized.edge_choice_target_answer_distance
                ),
                edge_choice_relation_choice_indices=(
                    context.linearized.edge_choice_relation_choice_indices
                ),
                node_relation_ptr=context.linearized.node_relation_ptr,
                relation_edge_ptr=context.linearized.relation_edge_ptr,
                current_component_count=int(context.current_components),
                current_commit_candidate_count=int(
                    context.current_commit_candidate_count
                ),
                current_oracle_distance=float(context.current_oracle_distance),
            )
            total_raw_candidates += int(context.linearized.edge_choice_edge_ids.numel())
            max_raw_candidates = max(
                max_raw_candidates,
                int(context.linearized.edge_choice_edge_ids.numel()),
            )
            total_stop_choices += int(stop_choice_logits.numel())
            max_stop_choices = max(max_stop_choices, int(stop_choice_logits.numel()))
            state_distributions.append(distribution)
        self._log_action_distribution_stats(
            active_state_count=int(len(active_state_indices)),
            total_raw_candidates=total_raw_candidates,
            max_raw_candidates=max_raw_candidates,
            total_stop_choices=total_stop_choices,
            max_stop_choices=max_stop_choices,
        )
        return SubgraphActionDistribution(
            flat_state_indices=active_state_tensor,
            state_features=active_state_features,
            state_distributions=tuple(state_distributions),
        )

    def compute_proposal_bias(
        self,
        *,
        distribution: SubgraphActionDistribution,
        proposal_bias_scale: float,
    ) -> torch.Tensor:
        del distribution, proposal_bias_scale
        return torch.zeros((0,), dtype=torch.float32)


__all__ = [
    "AnswerStopChoice",
    "HierarchicalEdgeChoice",
    "HierarchicalNodeChoice",
    "HierarchicalRelationChoice",
    "HierarchicalStateActionDistribution",
    "SubgraphActionDistribution",
    "SubgraphActor",
]
