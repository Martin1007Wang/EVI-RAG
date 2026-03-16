from __future__ import annotations

import torch

from src.graph_runtime import GraphObservation, GraphTopology


def compute_topology_log_heuristic(
    *,
    topology: GraphTopology,
    observation: GraphObservation,
    restart_prob: float,
    num_iters: int,
    eps: float,
) -> torch.Tensor:
    device = topology.edge_index.device
    log_heuristic = torch.full(
        (topology.num_nodes,), fill_value=float("-inf"), device=device
    )
    q_abs, q_graph_ids = topology.resolve_local_node_indices(
        observation.q_local_indices,
        field_name="q_local_indices",
    )
    graph_offsets = topology.graph_node_offsets.to(device=device)
    edge_ids = torch.arange(
        int(topology.edge_index.size(1)), device=device, dtype=torch.long
    )
    edge_graph_ids = topology.graph_index_from_edges(edge_ids)
    for graph_idx in range(int(topology.num_graphs)):
        node_start = int(graph_offsets[graph_idx].item())
        node_end = int(graph_offsets[graph_idx + 1].item())
        num_nodes = node_end - node_start
        if num_nodes <= 0:
            continue
        seed_nodes = q_abs[q_graph_ids == graph_idx] - node_start
        if int(seed_nodes.numel()) == 0:
            continue
        seeds = torch.zeros((num_nodes,), device=device, dtype=torch.float32)
        seeds.scatter_(0, seed_nodes, 1.0 / float(seed_nodes.numel()))
        mass = seeds.clone()
        edge_mask = edge_graph_ids == graph_idx
        local_edge_index = topology.edge_index[:, edge_mask] - node_start
        if int(local_edge_index.numel()) == 0:
            log_heuristic[node_start:node_end] = (seeds + float(eps)).log()
            continue
        src = local_edge_index[0]
        dst = local_edge_index[1]
        out_degree = torch.bincount(src, minlength=num_nodes).to(dtype=torch.float32)
        norm = out_degree.index_select(0, src).clamp_min(1.0)
        for _ in range(int(num_iters)):
            propagated = torch.zeros_like(mass)
            propagated.scatter_add_(0, dst, mass.index_select(0, src) / norm)
            mass = (
                float(restart_prob) * seeds + (1.0 - float(restart_prob)) * propagated
            )
        log_heuristic[node_start:node_end] = (mass + float(eps)).log()
    return log_heuristic


def compute_embedding_log_heuristic(
    *,
    topology: GraphTopology,
    node_tokens: torch.Tensor,
    question_tokens: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    graph_ids = topology.all_node_graph_index(device=node_tokens.device)
    question = question_tokens.index_select(0, graph_ids)
    cosine = torch.nn.functional.cosine_similarity(node_tokens, question, dim=-1)
    return cosine.to(dtype=torch.float32) / float(temperature)


__all__ = ["compute_embedding_log_heuristic", "compute_topology_log_heuristic"]
