from __future__ import annotations

import torch
from torch_scatter import scatter_sum

from .context import RewardBatchContext
from .rows import TerminalRows
from .schema import AnswerStats, SupportStats


def supported_answer_utility(
    *,
    rows: TerminalRows,
    context: RewardBatchContext,
    beta: float,
    max_connectivity_steps: int | None = None,
) -> SupportStats:
    """
    Anchor-supported answer utility.

    This is an undirected graph-connectivity proxy for evidence support, not
    semantic entailment.
    """
    return _batched_anchor_answer_support(
        edge_index=context.edge_index,
        active_nodes=rows.active_nodes,
        active_edges=rows.active_edges,
        anchor_mask=context.anchor_mask,
        target_mask=context.target_mask,
        node_batch=context.node_batch,
        edge_batch=context.edge_batch,
        row_to_graph=rows.graph_ids,
        num_graphs=context.num_graphs,
        dtype=context.dtype,
        beta=float(beta),
        max_connectivity_steps=max_connectivity_steps,
        inputs_are_graph_scoped=True,
    )


def compute_answer_support_mask(
    *,
    rows: TerminalRows,
    context: RewardBatchContext,
    max_connectivity_steps: int | None = None,
) -> torch.Tensor:
    """
    Per-answer support mask used by both U_ans and deficit process reward.

    A target is supported iff it is active and anchor-connected in the current
    active subgraph. Rows are graph-scoped, so targets outside a row's graph are
    always false.
    """
    reached = anchor_connected_nodes_for_rows(
        edge_index=context.edge_index,
        active_nodes=rows.active_nodes,
        active_edges=rows.active_edges,
        anchor_mask=context.anchor_mask,
        node_batch=context.node_batch,
        edge_batch=context.edge_batch,
        row_to_graph=rows.graph_ids,
        num_nodes=context.num_nodes,
        num_edges=context.num_edges,
        max_connectivity_steps=max_connectivity_steps,
        inputs_are_graph_scoped=True,
    )
    return reached & context.target_mask.view(1, -1)


def anchor_answer_support(
    *,
    edge_index: torch.Tensor,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
    beta: float = 2.0,
    max_connectivity_steps: int | None = None,
) -> SupportStats:
    """
    Anchor-supported answer precision/recall.

    Connectivity is undirected. That is intentional: this is evidence support,
    not directed logical entailment.
    """
    device = active_nodes.device
    num_graphs = int(num_graphs)
    row_to_graph = torch.arange(num_graphs, dtype=torch.long, device=device)
    node_batch = node_batch.to(device=device, dtype=torch.long)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    node_belongs = node_batch.view(1, -1).eq(row_to_graph.view(-1, 1))
    edge_belongs = edge_batch.view(1, -1).eq(row_to_graph.view(-1, 1))

    return _batched_anchor_answer_support(
        edge_index=edge_index,
        active_nodes=active_nodes.view(1, -1).expand(num_graphs, -1) & node_belongs,
        active_edges=active_edges.view(1, -1).expand(num_graphs, -1) & edge_belongs,
        anchor_mask=anchor_mask,
        target_mask=target_mask,
        node_batch=node_batch,
        edge_batch=edge_batch,
        row_to_graph=row_to_graph,
        num_graphs=num_graphs,
        dtype=dtype,
        beta=float(beta),
        max_connectivity_steps=max_connectivity_steps,
        inputs_are_graph_scoped=True,
    )


def connectivity_step_bound(
    state: object | None,
    *,
    num_nodes: int,
    num_edges: int,
) -> int:
    expand_budget = getattr(state, "expand_budget", None)
    if expand_budget is None:
        return _resolve_connectivity_steps(None, num_nodes=num_nodes, num_edges=num_edges)
    return _resolve_connectivity_steps(
        int(expand_budget),
        num_nodes=num_nodes,
        num_edges=num_edges,
    )


def _batched_anchor_answer_support(
    *,
    edge_index: torch.Tensor,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    anchor_mask: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    row_to_graph: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
    beta: float,
    max_connectivity_steps: int | None,
    inputs_are_graph_scoped: bool,
) -> SupportStats:
    device = active_nodes.device
    num_rows = int(active_nodes.size(0))

    edge_index = edge_index.to(device=device, dtype=torch.long)
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    target_mask = target_mask.to(device=device, dtype=torch.bool)
    node_batch = node_batch.to(device=device, dtype=torch.long)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    row_to_graph = row_to_graph.to(device=device, dtype=torch.long).view(-1)

    reached = anchor_connected_nodes_for_rows(
        edge_index=edge_index,
        active_nodes=active_nodes,
        active_edges=active_edges,
        anchor_mask=anchor_mask,
        node_batch=node_batch,
        edge_batch=edge_batch,
        row_to_graph=row_to_graph,
        num_nodes=int(node_batch.numel()),
        num_edges=int(edge_index.size(1)),
        max_connectivity_steps=max_connectivity_steps,
        inputs_are_graph_scoped=inputs_are_graph_scoped,
    )

    reward_answer_count = count_by_graph(
        target_mask,
        node_batch,
        int(num_graphs),
        dtype=dtype,
    ).index_select(0, row_to_graph)
    supported_answer_count = (
        reached & target_mask.view(1, -1)
    ).sum(dim=1).to(dtype=dtype)
    supported_retrieved_count = (
        reached & (~anchor_mask).view(1, -1)
    ).sum(dim=1).to(dtype=dtype)

    supported_answer_recall = torch.zeros(num_rows, dtype=dtype, device=device)
    supported_answer_precision = torch.zeros(num_rows, dtype=dtype, device=device)
    has_targets = reward_answer_count > 0.0
    supported_answer_recall[has_targets] = (
        supported_answer_count[has_targets] / reward_answer_count[has_targets]
    )
    supported_answer_precision[has_targets] = (
        supported_answer_count[has_targets]
        / supported_retrieved_count[has_targets].clamp_min(1.0)
    )
    supported_answer_f_beta = f_beta_tensor(
        precision=supported_answer_precision,
        recall=supported_answer_recall,
        beta=float(beta),
    )

    return SupportStats(
        supported_answer_recall=supported_answer_recall,
        supported_answer_precision=supported_answer_precision,
        supported_answer_f_beta=supported_answer_f_beta,
        supported_answer_count=supported_answer_count,
        reward_answer_count=reward_answer_count,
        supported_retrieved_count=supported_retrieved_count,
        answer_support_mask=reached & target_mask.view(1, -1),
    )


def anchor_connected_nodes_for_rows(
    *,
    edge_index: torch.Tensor,
    active_nodes: torch.Tensor,
    active_edges: torch.Tensor,
    anchor_mask: torch.Tensor,
    node_batch: torch.Tensor,
    edge_batch: torch.Tensor,
    row_to_graph: torch.Tensor,
    num_nodes: int,
    num_edges: int,
    max_connectivity_steps: int | None,
    inputs_are_graph_scoped: bool,
) -> torch.Tensor:
    device = active_nodes.device
    num_rows = int(active_nodes.size(0))
    edge_index = edge_index.to(device=device, dtype=torch.long)
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    node_batch = node_batch.to(device=device, dtype=torch.long)
    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    row_to_graph = row_to_graph.to(device=device, dtype=torch.long).view(-1)

    if row_to_graph.shape != (num_rows,):
        raise ValueError(
            f"row_to_graph must have shape [{num_rows}], got {tuple(row_to_graph.shape)}."
        )

    node_belongs = None
    if bool(inputs_are_graph_scoped):
        reached = active_nodes & anchor_mask.view(1, -1)
    else:
        node_belongs = node_batch.view(1, -1).eq(row_to_graph.view(-1, 1))
        edge_belongs = edge_batch.view(1, -1).eq(row_to_graph.view(-1, 1))
        active_edges = active_edges & edge_belongs
        reached = active_nodes & anchor_mask.view(1, -1) & node_belongs

    max_steps = _resolve_connectivity_steps(
        max_connectivity_steps,
        num_nodes=int(num_nodes),
        num_edges=int(num_edges),
    )
    src_all, dst_all = edge_index
    for _ in range(max_steps):
        edge_reached = active_edges & (
            reached.index_select(1, src_all) | reached.index_select(1, dst_all)
        )
        row_ids, edge_ids = edge_reached.nonzero(as_tuple=True)
        next_reached = reached.clone()
        next_reached[
            row_ids,
            src_all.index_select(0, edge_ids),
        ] = True
        next_reached[
            row_ids,
            dst_all.index_select(0, edge_ids),
        ] = True
        reached = next_reached if node_belongs is None else next_reached & node_belongs

    return reached


def sparse_rollout_active_node_trace(
    *,
    anchor_node_trace: torch.Tensor,
    anchor_node_lengths: torch.Tensor,
    expanded_edge_trace: torch.Tensor,
    expanded_edge_lengths: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return V_s as sparse node traces for canonical rollout states.

    V_s = anchors union endpoints(expanded edges). Root edges are anchor-induced,
    so they do not add non-anchor nodes.
    """
    device = edge_index.device
    anchor_node_trace = anchor_node_trace.to(device=device, dtype=torch.long)
    anchor_node_lengths = anchor_node_lengths.to(device=device, dtype=torch.long).view(-1)
    expanded_edge_trace = expanded_edge_trace.to(device=device, dtype=torch.long)
    expanded_edge_lengths = expanded_edge_lengths.to(device=device, dtype=torch.long).view(-1)
    edge_index = edge_index.to(device=device, dtype=torch.long)

    if anchor_node_trace.size(0) != expanded_edge_trace.size(0):
        raise ValueError(
            "anchor_node_trace and expanded_edge_trace must have the same row count: "
            f"{anchor_node_trace.size(0)} != {expanded_edge_trace.size(0)}."
        )
    num_rows = int(anchor_node_trace.size(0))
    if anchor_node_lengths.shape != (num_rows,):
        raise ValueError(
            f"anchor_node_lengths must have shape [{num_rows}], "
            f"got {tuple(anchor_node_lengths.shape)}."
        )
    if expanded_edge_lengths.shape != (num_rows,):
        raise ValueError(
            f"expanded_edge_lengths must have shape [{num_rows}], "
            f"got {tuple(expanded_edge_lengths.shape)}."
        )

    return _sparse_trace_active_node_candidates(
        expanded_edge_trace=expanded_edge_trace,
        expanded_edge_lengths=expanded_edge_lengths,
        anchor_node_trace=anchor_node_trace,
        anchor_node_lengths=anchor_node_lengths,
        edge_index=edge_index,
    )


def sparse_rollout_answer_stats_from_active_nodes(
    *,
    active_node_trace: torch.Tensor,
    active_node_valid: torch.Tensor,
    rollout_to_graph: torch.Tensor,
    target_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> AnswerStats:
    device = rollout_to_graph.device
    rollout_to_graph = rollout_to_graph.to(device=device, dtype=torch.long).view(-1)
    active_node_trace = active_node_trace.to(device=device, dtype=torch.long)
    active_node_valid = active_node_valid.to(device=device, dtype=torch.bool)
    target_mask = target_mask.to(device=device, dtype=torch.bool)
    anchor_mask = anchor_mask.to(device=device, dtype=torch.bool)
    node_batch = node_batch.to(device=device, dtype=torch.long)

    num_rows = int(rollout_to_graph.numel())
    if active_node_trace.size(0) != num_rows:
        raise ValueError(
            "active_node_trace row count must match rollout_to_graph: "
            f"{active_node_trace.size(0)} != {num_rows}."
        )
    if active_node_valid.shape != active_node_trace.shape:
        raise ValueError(
            "active_node_valid must have the same shape as active_node_trace: "
            f"{tuple(active_node_valid.shape)} != {tuple(active_node_trace.shape)}."
        )

    unique_active = _first_valid_node_occurrence(
        node_ids=active_node_trace,
        valid=active_node_valid,
    )
    target_nodes = _safe_node_mask_select(target_mask, active_node_trace)
    anchor_nodes = _safe_node_mask_select(anchor_mask, active_node_trace)

    hits = (unique_active & target_nodes).sum(dim=1).to(dtype=dtype)
    retrieved_count = (
        unique_active & ((~anchor_nodes) | target_nodes)
    ).sum(dim=1).to(dtype=dtype)
    gold = count_by_graph(
        target_mask,
        node_batch,
        int(num_graphs),
        dtype=dtype,
    ).index_select(0, rollout_to_graph)

    precision = torch.zeros_like(hits)
    recall = torch.zeros_like(hits)
    has_retrieved = retrieved_count > 0.0
    has_gold = gold > 0.0
    precision[has_retrieved] = hits[has_retrieved] / retrieved_count[has_retrieved]
    recall[has_gold] = hits[has_gold] / gold[has_gold]
    denom = precision + recall
    f1 = torch.zeros_like(hits)
    valid = denom > 0.0
    f1[valid] = 2.0 * precision[valid] * recall[valid] / denom[valid]

    return AnswerStats(
        hits=hits,
        gold=gold,
        retrieved=retrieved_count,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def support_stats_from_answer_stats(
    answer: AnswerStats,
    *,
    beta: float,
    answer_support_mask: torch.Tensor,
) -> SupportStats:
    return SupportStats(
        supported_answer_recall=answer.recall,
        supported_answer_precision=answer.precision,
        supported_answer_f_beta=f_beta_tensor(
            precision=answer.precision,
            recall=answer.recall,
            beta=float(beta),
        ),
        supported_answer_count=answer.hits,
        reward_answer_count=answer.gold,
        supported_retrieved_count=answer.retrieved,
        answer_support_mask=answer_support_mask,
    )


def sparse_rollout_answer_degree_excess_from_traces(
    *,
    active_edge_trace: torch.Tensor,
    active_edge_lengths: torch.Tensor,
    rollout_to_graph: torch.Tensor,
    edge_index: torch.Tensor,
    target_mask: torch.Tensor,
    node_batch: torch.Tensor,
    num_graphs: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    device = rollout_to_graph.device
    rollout_to_graph = rollout_to_graph.to(device=device, dtype=torch.long).view(-1)
    active_edge_trace = active_edge_trace.to(device=device, dtype=torch.long)
    active_edge_lengths = active_edge_lengths.to(device=device, dtype=torch.long).view(-1)
    edge_index = edge_index.to(device=device, dtype=torch.long)
    target_mask = target_mask.to(device=device, dtype=torch.bool)
    node_batch = node_batch.to(device=device, dtype=torch.long)

    edge_valid = _trace_valid_mask(active_edge_trace, active_edge_lengths)
    if active_edge_trace.size(1) == 0:
        values = torch.zeros_like(rollout_to_graph, dtype=dtype)
    else:
        edge_ids = active_edge_trace.clamp_min(0)
        src = edge_index[0].index_select(0, edge_ids.view(-1)).view_as(edge_ids)
        dst = edge_index[1].index_select(0, edge_ids.view(-1)).view_as(edge_ids)
        endpoints = torch.cat([src, dst], dim=1)
        endpoint_valid = torch.cat([edge_valid, edge_valid], dim=1)
        endpoint_is_target = endpoint_valid & _safe_node_mask_select(
            target_mask,
            endpoints,
        )
        first_target = _first_valid_node_occurrence(
            node_ids=endpoints,
            valid=endpoint_is_target,
        )
        if endpoints.size(1) == 0:
            excess = torch.zeros_like(rollout_to_graph, dtype=dtype)
        else:
            same = endpoints.unsqueeze(2).eq(endpoints.unsqueeze(1))
            counts = (
                same & endpoint_is_target.unsqueeze(1)
            ).sum(dim=2).to(dtype=dtype)
            excess = ((counts - 1.0).clamp_min(0.0) * first_target.to(dtype=dtype)).sum(
                dim=1,
            )
        target_count = count_by_graph(
            target_mask,
            node_batch,
            int(num_graphs),
            dtype=dtype,
        ).index_select(0, rollout_to_graph).clamp_min(1.0)
        values = excess / target_count
    return values


def _sparse_trace_active_node_candidates(
    *,
    expanded_edge_trace: torch.Tensor,
    expanded_edge_lengths: torch.Tensor,
    anchor_node_trace: torch.Tensor,
    anchor_node_lengths: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_valid = _trace_valid_mask(expanded_edge_trace, expanded_edge_lengths)
    anchor_valid = _trace_valid_mask(anchor_node_trace, anchor_node_lengths)
    if expanded_edge_trace.size(1) == 0:
        return anchor_node_trace.clamp_min(0), anchor_valid

    edge_ids = expanded_edge_trace.clamp_min(0)
    src = edge_index[0].index_select(0, edge_ids.view(-1)).view_as(edge_ids)
    dst = edge_index[1].index_select(0, edge_ids.view(-1)).view_as(edge_ids)
    nodes = torch.cat([anchor_node_trace.clamp_min(0), src, dst], dim=1)
    valid = torch.cat([anchor_valid, edge_valid, edge_valid], dim=1)
    return nodes, valid


def _trace_valid_mask(trace: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(device=trace.device, dtype=torch.long).view(-1)
    if trace.size(1) == 0:
        return torch.zeros(trace.shape, dtype=torch.bool, device=trace.device)
    return torch.arange(
        trace.size(1),
        dtype=torch.long,
        device=trace.device,
    ).view(1, -1) < lengths.view(-1, 1)


def _safe_node_mask_select(mask: torch.Tensor, node_ids: torch.Tensor) -> torch.Tensor:
    if node_ids.numel() == 0:
        return torch.zeros(node_ids.shape, dtype=torch.bool, device=node_ids.device)
    return mask.index_select(0, node_ids.clamp_min(0).view(-1)).view_as(node_ids)


def _first_valid_node_occurrence(
    *,
    node_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if node_ids.size(1) == 0:
        return torch.zeros_like(valid)
    width = int(node_ids.size(1))
    previous = torch.ones(
        (width, width),
        dtype=torch.bool,
        device=node_ids.device,
    ).tril(diagonal=-1)
    duplicate_previous = (
        node_ids.unsqueeze(2).eq(node_ids.unsqueeze(1))
        & valid.unsqueeze(1)
        & previous.view(1, width, width)
    ).any(dim=2)
    return valid & ~duplicate_previous


def _resolve_connectivity_steps(
    max_connectivity_steps: int | None,
    *,
    num_nodes: int,
    num_edges: int,
) -> int:
    if max_connectivity_steps is None:
        return max(0, min(int(num_nodes) - 1, int(num_edges)))
    return max(0, min(int(max_connectivity_steps), int(num_nodes) - 1, int(num_edges)))


def f_beta_tensor(
    *,
    precision: torch.Tensor,
    recall: torch.Tensor,
    beta: float,
    delta: float = 1.0e-8,
) -> torch.Tensor:
    beta_sq = float(beta) ** 2
    denom = beta_sq * precision + recall + float(delta)
    return (1.0 + beta_sq) * precision * recall / denom.clamp_min(
        torch.finfo(precision.dtype).tiny
    )


def count_by_graph(
    mask: torch.Tensor,
    batch_index: torch.Tensor,
    num_graphs: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    return scatter_sum(
        mask.to(dtype=dtype),
        batch_index.to(device=mask.device, dtype=torch.long),
        dim=0,
        dim_size=int(num_graphs),
    )


__all__ = [
    "anchor_answer_support",
    "anchor_connected_nodes_for_rows",
    "compute_answer_support_mask",
    "connectivity_step_bound",
    "count_by_graph",
    "f_beta_tensor",
    "sparse_rollout_active_node_trace",
    "sparse_rollout_answer_degree_excess_from_traces",
    "sparse_rollout_answer_stats_from_active_nodes",
    "support_stats_from_answer_stats",
    "supported_answer_utility",
]
