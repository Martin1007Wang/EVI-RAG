from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

_GUMBEL_EPS = 1e-10
_UNIT_TEMPERATURE = 1.0
_ZERO = 0
_ONE = 1
_TWO = 2
_EDGE_HEAD_INDEX = 0

EDGE_POLICY_MASK_KEY = "edge_policy_mask"
STOP_NODE_MASK_KEY = "stop_node_mask"


def neg_inf_value(tensor: torch.Tensor) -> float:
    return float(torch.finfo(tensor.dtype).min)


def _coerce_temperature(temperature: Union[float, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temp = temperature
    else:
        temp = torch.tensor(float(temperature), device=ref.device, dtype=ref.dtype)
    return temp.to(device=ref.device, dtype=ref.dtype).view(())


def segment_max(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment-wise max for 1D tensors with argmax indices.

    Returns:
      - max_per: [num_segments]
      - argmax: [num_segments] indices into `src` (0 if empty segment)
    """

    if src.numel() == 0:
        max_per = torch.full((num_segments,), neg_inf_value(src), device=src.device, dtype=src.dtype)
        argmax = torch.zeros((num_segments,), device=src.device, dtype=torch.long)
        return max_per, argmax

    segment_ids = segment_ids.to(device=src.device, dtype=torch.long).view(-1)
    neg_inf_src = neg_inf_value(src)
    max_per = torch.full((num_segments,), neg_inf_src, device=src.device, dtype=src.dtype)
    max_per.scatter_reduce_(0, segment_ids, src, reduce="amax", include_self=True)

    positions = torch.arange(src.numel(), device=src.device, dtype=torch.long)
    is_max = src == max_per.index_select(0, segment_ids)
    sentinel = src.numel()
    candidate = torch.where(is_max, positions, torch.full_like(positions, sentinel))
    argmin = torch.full((num_segments,), sentinel, device=src.device, dtype=torch.long)
    argmin.scatter_reduce_(0, segment_ids, candidate, reduce="amin", include_self=True)
    argmax = torch.where(argmin == sentinel, torch.zeros_like(argmin), argmin)
    return max_per, argmax


def segment_softmax_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Stable segment-wise softmax for 1D logits."""

    if logits.numel() == 0:
        return logits
    log_denom = segment_logsumexp_1d(logits, segment_ids.to(device=logits.device), num_segments)
    return torch.exp(logits - log_denom.index_select(0, segment_ids.to(device=logits.device, dtype=torch.long)))


def segment_logsumexp_1d(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    if logits.numel() == 0:
        return torch.full((num_segments,), neg_inf_value(logits), device=logits.device, dtype=logits.dtype)

    device = logits.device
    calc_dtype = logits.dtype
    neg_inf = torch.finfo(calc_dtype).min
    max_per = torch.full((num_segments,), neg_inf, device=device, dtype=calc_dtype)
    max_per.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_per[segment_ids]
    exp = torch.exp(shifted)
    sum_per = torch.zeros((num_segments,), device=device, dtype=calc_dtype)
    sum_per.index_add_(0, segment_ids, exp)
    eps = torch.finfo(calc_dtype).eps
    return torch.log(sum_per.clamp(min=eps)) + max_per


def scatter_logsumexp(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    return segment_logsumexp_1d(logits, segment_ids, num_segments)


def compute_policy_log_probs(
    *,
    edge_logits: torch.Tensor,
    stop_logits: Optional[torch.Tensor],
    edge_batch: torch.Tensor,
    valid_edges: torch.Tensor,
    num_graphs: int,
    temperature: Union[float, torch.Tensor],
    allow_stop: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    if num_graphs <= 0:
        device = edge_logits.device
        zeros = torch.zeros(0, device=device)
        has_edge = torch.zeros(0, device=device, dtype=torch.bool)
        if stop_logits is None:
            return zeros, None, zeros, has_edge
        return zeros, zeros, zeros, has_edge
    temperature = _coerce_temperature(temperature, edge_logits)

    stop_scaled = None
    if stop_logits is not None:
        stop_scaled = stop_logits / temperature
    if edge_logits.numel() == 0:
        log_denom = edge_logits.new_full((num_graphs,), neg_inf_value(edge_logits))
        has_edge = torch.zeros(num_graphs, device=edge_logits.device, dtype=torch.bool)
        if stop_scaled is None:
            return edge_logits, None, log_denom, has_edge
        allow_stop_mask = (
            torch.ones(num_graphs, device=stop_scaled.device, dtype=torch.bool)
            if allow_stop is None
            else allow_stop.to(device=stop_scaled.device, dtype=torch.bool).view(-1)
        )
        log_denom = torch.where(allow_stop_mask, stop_scaled, log_denom)
        neg_inf = neg_inf_value(stop_scaled)
        log_prob_stop = torch.where(
            allow_stop_mask,
            stop_scaled - log_denom,
            torch.full_like(stop_scaled, neg_inf),
        )
        return edge_logits, log_prob_stop, log_denom, has_edge

    edge_scaled = edge_logits / temperature
    valid_edges = valid_edges.to(device=edge_scaled.device, dtype=torch.bool)
    logsumexp_edges = segment_logsumexp_1d(edge_scaled[valid_edges], edge_batch[valid_edges], num_graphs)
    has_edge = logsumexp_edges > neg_inf_value(logsumexp_edges)
    if stop_scaled is None:
        log_denom = logsumexp_edges
        log_prob_edge = edge_scaled - log_denom[edge_batch]
        log_prob_edge = torch.where(
            valid_edges,
            log_prob_edge,
            torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
        )
        return log_prob_edge, None, log_denom, has_edge
    allow_stop_mask = (
        torch.ones(num_graphs, device=stop_logits.device, dtype=torch.bool)
        if allow_stop is None
        else allow_stop.to(device=stop_logits.device, dtype=torch.bool).view(-1)
    )
    log_denom = torch.where(
        allow_stop_mask,
        torch.logaddexp(logsumexp_edges, stop_scaled),
        logsumexp_edges,
    )
    log_prob_edge = edge_scaled - log_denom[edge_batch]
    log_prob_edge = torch.where(
        valid_edges,
        log_prob_edge,
        torch.full_like(log_prob_edge, neg_inf_value(log_prob_edge)),
    )
    log_prob_stop = stop_scaled - log_denom
    log_prob_stop = torch.where(
        allow_stop_mask,
        log_prob_stop,
        torch.full_like(log_prob_stop, neg_inf_value(log_prob_stop)),
    )
    return log_prob_edge, log_prob_stop, log_denom, has_edge


@dataclass(frozen=True)
class PolicyLogProbs:
    edge: torch.Tensor
    stop: torch.Tensor
    not_stop: torch.Tensor
    has_edge: torch.Tensor


def compute_forward_log_probs(
    *,
    edge_scores: torch.Tensor,
    stop_logits: torch.Tensor,
    allow_stop: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    temperature: Union[float, torch.Tensor],
    edge_valid_mask: torch.Tensor,
) -> PolicyLogProbs:
    if num_graphs <= _ZERO:
        empty_edge = edge_scores.new_empty((_ZERO,))
        empty_stop = stop_logits.new_empty((_ZERO,))
        empty_bool = stop_logits.new_empty((_ZERO,), dtype=torch.bool)
        return PolicyLogProbs(edge=empty_edge, stop=empty_stop, not_stop=empty_stop, has_edge=empty_bool)
    temp = _coerce_temperature(temperature, edge_scores)
    edge_logits = edge_scores / temp
    stop_logits = stop_logits / temp
    valid_edges = edge_valid_mask.view(-1)
    edge_log_prob, log_prob_stop, log_denom, has_edge = compute_policy_log_probs(
        edge_logits=edge_logits,
        stop_logits=stop_logits,
        edge_batch=edge_batch,
        valid_edges=valid_edges,
        num_graphs=num_graphs,
        temperature=_UNIT_TEMPERATURE,
        allow_stop=allow_stop,
    )
    logsumexp_edges = segment_logsumexp_1d(edge_logits[valid_edges], edge_batch[valid_edges], num_graphs)
    log_prob_not_stop = logsumexp_edges - log_denom
    neg_inf = neg_inf_value(log_prob_not_stop)
    log_prob_not_stop = torch.where(has_edge, log_prob_not_stop, torch.full_like(log_prob_not_stop, neg_inf))
    return PolicyLogProbs(edge=edge_log_prob, stop=log_prob_stop, not_stop=log_prob_not_stop, has_edge=has_edge)


def sample_stop_mask(
    *,
    log_prob_stop: torch.Tensor,
    log_prob_not_stop: torch.Tensor,
    has_edge: torch.Tensor,
    is_greedy: bool,
) -> torch.Tensor:
    score_stop = log_prob_stop
    score_move = log_prob_not_stop
    if not is_greedy:
        score_stop = score_stop + gumbel_noise_like(score_stop)
        score_move = score_move + gumbel_noise_like(score_move)
    return (score_stop > score_move) | (~has_edge)


@dataclass(frozen=True)
class OutgoingEdges:
    edge_ids: torch.Tensor
    edge_batch: torch.Tensor
    edge_counts: torch.Tensor
    has_edge: torch.Tensor


def apply_edge_policy_mask(
    *,
    outgoing: OutgoingEdges,
    edge_policy_mask: torch.Tensor,
    num_graphs: int,
) -> OutgoingEdges:
    if outgoing.edge_ids.numel() == _ZERO:
        return outgoing
    edge_policy_mask = edge_policy_mask.to(device=outgoing.edge_ids.device, dtype=torch.bool).view(-1)
    keep = edge_policy_mask.index_select(0, outgoing.edge_ids)
    edge_ids = outgoing.edge_ids[keep]
    edge_batch = outgoing.edge_batch[keep]
    if edge_ids.numel() == _ZERO:
        empty = edge_ids.new_empty((_ZERO,))
        edge_counts = edge_ids.new_zeros((num_graphs,))
        has_edge = edge_ids.new_zeros((num_graphs,), dtype=torch.bool)
        return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=edge_counts, has_edge=has_edge)
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    has_edge = edge_counts > _ZERO
    return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=edge_counts, has_edge=has_edge)


def gather_outgoing_edges(
    *,
    curr_nodes: torch.Tensor,
    edge_ids_by_head: torch.Tensor,
    edge_ptr_by_head: torch.Tensor,
    active_mask: torch.Tensor,
) -> OutgoingEdges:
    curr_nodes = curr_nodes.to(device=edge_ptr_by_head.device, dtype=torch.long).view(-1)
    active_mask = active_mask.to(device=curr_nodes.device, dtype=torch.bool).view(-1)
    num_graphs = int(curr_nodes.numel())
    if num_graphs == _ZERO:
        empty = torch.zeros((_ZERO,), device=curr_nodes.device, dtype=torch.long)
        return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=empty, has_edge=empty.to(dtype=torch.bool))
    active_mask = active_mask & (curr_nodes >= _ZERO)
    safe_nodes = torch.where(active_mask, curr_nodes, torch.zeros_like(curr_nodes))
    offsets = edge_ptr_by_head.index_select(0, safe_nodes)
    next_offsets = edge_ptr_by_head.index_select(0, safe_nodes + _ONE)
    counts = (next_offsets - offsets).to(dtype=torch.long)
    counts = torch.where(active_mask, counts, torch.zeros_like(counts))
    has_edge = counts > _ZERO
    total = int(counts.sum().detach().tolist())
    if total == _ZERO:
        empty = torch.zeros((_ZERO,), device=curr_nodes.device, dtype=torch.long)
        return OutgoingEdges(edge_ids=empty, edge_batch=empty, edge_counts=counts, has_edge=has_edge)
    graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=curr_nodes.device), counts)
    offset_rep = torch.repeat_interleave(offsets, counts)
    starts = counts.cumsum(0) - counts
    start_rep = torch.repeat_interleave(starts, counts)
    local_pos = torch.arange(total, device=curr_nodes.device, dtype=offset_rep.dtype) - start_rep
    edge_pos = offset_rep + local_pos
    edge_ids = edge_ids_by_head.index_select(0, edge_pos)
    return OutgoingEdges(edge_ids=edge_ids, edge_batch=graph_ids, edge_counts=counts, has_edge=has_edge)


@dataclass(frozen=True)
class RelationGroups:
    pair_graph: torch.Tensor
    pair_rel: torch.Tensor
    pair_inverse: torch.Tensor


def build_relation_groups(*, edge_batch: torch.Tensor, edge_relations: torch.Tensor) -> RelationGroups:
    edge_batch = edge_batch.to(device=edge_relations.device, dtype=torch.long).view(-1)
    edge_relations = edge_relations.to(device=edge_batch.device, dtype=torch.long).view(-1)
    if edge_batch.numel() == _ZERO:
        empty = torch.zeros((_ZERO,), device=edge_batch.device, dtype=torch.long)
        return RelationGroups(pair_graph=empty, pair_rel=empty, pair_inverse=empty)
    pairs = torch.stack((edge_batch, edge_relations), dim=_ONE)
    unique_pairs, pair_inverse = torch.unique(pairs, dim=_ZERO, return_inverse=True)
    pair_graph = unique_pairs[:, _ZERO]
    pair_rel = unique_pairs[:, _ONE]
    return RelationGroups(pair_graph=pair_graph, pair_rel=pair_rel, pair_inverse=pair_inverse)


def compute_relation_log_probs(
    *,
    edge_logits: torch.Tensor,
    relation_groups: RelationGroups,
    num_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_logits = edge_logits.view(-1)
    if edge_logits.numel() == _ZERO:
        empty = torch.zeros((_ZERO,), device=edge_logits.device, dtype=edge_logits.dtype)
        return empty, empty
    num_pairs = int(relation_groups.pair_graph.numel())
    if num_pairs == _ZERO:
        empty = torch.zeros((_ZERO,), device=edge_logits.device, dtype=edge_logits.dtype)
        return empty, empty
    rel_logsumexp_pair = segment_logsumexp_1d(edge_logits, relation_groups.pair_inverse, num_pairs)
    rel_logsumexp_graph = segment_logsumexp_1d(rel_logsumexp_pair, relation_groups.pair_graph, num_graphs)
    log_prob_relation_pair = rel_logsumexp_pair - rel_logsumexp_graph.index_select(0, relation_groups.pair_graph)
    log_prob_tail_edge = edge_logits - rel_logsumexp_pair.index_select(0, relation_groups.pair_inverse)
    return log_prob_relation_pair, log_prob_tail_edge


def gumbel_noise_like(tensor: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(tensor)
    return -torch.log(-torch.log(u.clamp(min=_GUMBEL_EPS, max=1.0 - _GUMBEL_EPS)))


def sample_relation_pairs(
    *,
    log_prob_edge_cond: torch.Tensor,
    edge_relations: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    is_greedy: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    relation_groups = build_relation_groups(edge_batch=edge_batch, edge_relations=edge_relations)
    log_prob_relation_pair, log_prob_tail_edge = compute_relation_log_probs(
        edge_logits=log_prob_edge_cond,
        relation_groups=relation_groups,
        num_graphs=num_graphs,
    )
    score_relation = log_prob_relation_pair
    if not is_greedy:
        score_relation = score_relation + gumbel_noise_like(score_relation)
    score_relation_max, chosen_pair = segment_max(score_relation, relation_groups.pair_graph, num_graphs)
    neg_inf = neg_inf_value(score_relation_max)
    chosen_pair = torch.where(score_relation_max > neg_inf, chosen_pair, torch.zeros_like(chosen_pair))
    return log_prob_tail_edge, relation_groups.pair_inverse, chosen_pair


def sample_tail_edges(
    *,
    log_prob_tail_edge: torch.Tensor,
    pair_inverse: torch.Tensor,
    chosen_pair: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    is_greedy: bool,
) -> torch.Tensor:
    chosen_pair_per_edge = chosen_pair.index_select(0, edge_batch)
    relation_match = pair_inverse == chosen_pair_per_edge
    score_tail = log_prob_tail_edge
    if not is_greedy:
        score_tail = score_tail + gumbel_noise_like(score_tail)
    score_tail = torch.where(relation_match, score_tail, torch.full_like(score_tail, neg_inf_value(score_tail)))
    _, edge_argmax = segment_max(score_tail, edge_batch, num_graphs)
    return edge_argmax


def sample_actions(
    *,
    outgoing: OutgoingEdges,
    policy: PolicyLogProbs,
    edge_relations: torch.Tensor,
    is_greedy: bool,
    stop_action: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if policy.edge.numel() == _ZERO:
        actions = torch.full_like(policy.has_edge, stop_action, dtype=torch.long)
        return actions, policy.stop
    edge_relations = (
        edge_relations.index_select(0, outgoing.edge_ids)
        if outgoing.edge_ids.numel() > _ZERO
        else outgoing.edge_ids
    )
    choose_stop = sample_stop_mask(
        log_prob_stop=policy.stop,
        log_prob_not_stop=policy.not_stop,
        has_edge=policy.has_edge,
        is_greedy=is_greedy,
    )
    log_prob_edge_cond = policy.edge - policy.not_stop.index_select(0, outgoing.edge_batch)
    log_prob_tail_edge, pair_inverse, chosen_pair = sample_relation_pairs(
        log_prob_edge_cond=log_prob_edge_cond,
        edge_relations=edge_relations,
        edge_batch=outgoing.edge_batch,
        num_graphs=policy.has_edge.numel(),
        is_greedy=is_greedy,
    )
    edge_argmax = sample_tail_edges(
        log_prob_tail_edge=log_prob_tail_edge,
        pair_inverse=pair_inverse,
        chosen_pair=chosen_pair,
        edge_batch=outgoing.edge_batch,
        num_graphs=policy.has_edge.numel(),
        is_greedy=is_greedy,
    )
    edge_argmax = torch.where(policy.has_edge, edge_argmax, torch.zeros_like(edge_argmax))
    chosen_edges = outgoing.edge_ids.index_select(0, edge_argmax)
    log_prob_edge_sel = policy.edge.index_select(0, edge_argmax)
    actions = torch.where(choose_stop, torch.full_like(chosen_edges, stop_action), chosen_edges)
    log_pf = torch.where(choose_stop, policy.stop, log_prob_edge_sel)
    return actions, log_pf


def build_edge_head_csr(
    *,
    edge_index: torch.Tensor,
    num_nodes_total: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index.numel() == _ZERO:
        edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
        edge_ids = torch.zeros((_ZERO,), device=device, dtype=torch.long)
        return edge_ids, edge_ptr
    heads = edge_index[_EDGE_HEAD_INDEX].to(device=device, dtype=torch.long).view(-1)
    counts = torch.bincount(heads, minlength=int(num_nodes_total)).to(device=device, dtype=torch.long)
    edge_ptr = torch.zeros((num_nodes_total + _ONE), device=device, dtype=torch.long)
    edge_ptr[_ONE:] = counts.cumsum(0)
    edge_ids = torch.argsort(heads)
    return edge_ids, edge_ptr


__all__ = [
    "EDGE_POLICY_MASK_KEY",
    "STOP_NODE_MASK_KEY",
    "OutgoingEdges",
    "PolicyLogProbs",
    "build_edge_head_csr",
    "compute_forward_log_probs",
    "compute_policy_log_probs",
    "gather_outgoing_edges",
    "gumbel_noise_like",
    "neg_inf_value",
    "segment_max",
    "segment_softmax_1d",
    "sample_actions",
    "scatter_logsumexp",
    "segment_logsumexp_1d",
]
