from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

from src.models.components.dtypes import align_float_input_dtype
from src.utils.logging_utils import get_logger

from .cuda_memory import cuda_memory_profiling_enabled

if TYPE_CHECKING:
    from .actor import SubgraphActor


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


def _build_node_logits(
    self: SubgraphActor,
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
            torch.log1p(node_candidate_counts.to(device=device, dtype=torch.float32)),
        ),
        dim=-1,
    )
    focus_inputs = torch.cat((state_features, node_features, struct), dim=-1)
    focus_inputs = align_float_input_dtype(focus_inputs, module=self.node_focus_norm)
    focus_inputs = self.node_focus_norm(focus_inputs)
    focus_inputs = align_float_input_dtype(focus_inputs, module=self.node_focus_head[0])
    return self.node_focus_head(focus_inputs).squeeze(-1).to(dtype=torch.float32)


def _build_node_logits_batch(
    self: SubgraphActor,
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
    self: SubgraphActor,
    *,
    state_features: torch.Tensor,
    src_features: torch.Tensor,
    relation_features: torch.Tensor,
    relation_num_edges: torch.Tensor,
    relation_max_new_bit_gain: torch.Tensor,
    relation_max_answer_candidate_counts: torch.Tensor,
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
                    relation_max_new_bit_gain.to(device=device, dtype=torch.float32),
                    relation_max_answer_candidate_counts.to(
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
    self: SubgraphActor,
    *,
    state_feature: torch.Tensor,
    src_features: torch.Tensor,
    relation_features: torch.Tensor,
    relation_num_edges: torch.Tensor,
    relation_max_new_bit_gain: torch.Tensor,
    relation_max_answer_candidate_counts: torch.Tensor,
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
        relation_max_answer_candidate_counts=relation_max_answer_candidate_counts,
        relation_max_semantic_overlap=relation_max_semantic_overlap,
    )


def _build_edge_logits(
    self: SubgraphActor,
    *,
    state_features: torch.Tensor,
    src_features: torch.Tensor,
    relation_features: torch.Tensor,
    dst_features: torch.Tensor,
    current_components: torch.Tensor,
    next_component_counts: torch.Tensor,
    semantic_overlap: torch.Tensor,
    action_new_bit_gain: torch.Tensor,
    answer_candidate_counts: torch.Tensor,
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
                        semantic_overlap[chunk].to(device=device, dtype=torch.float32),
                        action_new_bit_gain[chunk].to(
                            device=device, dtype=torch.float32
                        ),
                        current_components[chunk].to(
                            device=device, dtype=torch.float32
                        ),
                        next_component_counts[chunk].to(
                            device=device, dtype=torch.float32
                        ),
                        answer_candidate_counts[chunk].to(
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
    self: SubgraphActor,
    *,
    state_feature: torch.Tensor,
    src_features: torch.Tensor,
    relation_features: torch.Tensor,
    dst_features: torch.Tensor,
    current_components: int,
    next_component_counts: torch.Tensor,
    semantic_overlap: torch.Tensor,
    action_new_bit_gain: torch.Tensor,
    answer_candidate_counts: torch.Tensor,
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
        answer_candidate_counts=answer_candidate_counts,
    )


def _build_stop_choice_logits(
    self: SubgraphActor,
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
                    torch.log1p(current_edges.to(device=device, dtype=torch.float32)),
                ),
                dim=-1,
            ),
        ),
        dim=-1,
    )
    stop_inputs = align_float_input_dtype(stop_inputs, module=self.stop_choice_norm)
    stop_inputs = self.stop_choice_norm(stop_inputs)
    stop_inputs = align_float_input_dtype(stop_inputs, module=self.stop_choice_head[0])
    return self.stop_choice_head(stop_inputs).squeeze(-1).to(dtype=torch.float32)


def _build_stop_choice_logits_batch(
    self: SubgraphActor,
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
    self: SubgraphActor,
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
                    torch.log1p(current_edges.to(device=device, dtype=torch.float32)),
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
    return self.failure_stop_head(failure_inputs).squeeze(-1).to(dtype=torch.float32)


def _build_failure_stop_logit(
    self: SubgraphActor,
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


__all__ = [
    "_bit_count",
    "_build_edge_logits",
    "_build_edge_logits_batch",
    "_build_failure_stop_logit",
    "_build_failure_stop_logits",
    "_build_mlp",
    "_build_node_logits",
    "_build_node_logits_batch",
    "_build_relation_logits",
    "_build_relation_logits_batch",
    "_build_stop_choice_logits",
    "_build_stop_choice_logits_batch",
    "_log_action_distribution_stats",
]
