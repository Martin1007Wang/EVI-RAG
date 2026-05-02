from __future__ import annotations

import torch

from src.graph.ops import compute_uniform_nonroot_backward_removals
from src.weaver.state import RolloutState


def compute_candidate_uniform_backward_log_probs(
    *,
    state: RolloutState,
    edge_index: torch.Tensor,
    edge_batch: torch.Tensor,
    candidate_edge_ids: torch.Tensor,
    candidate_batch_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """
    Candidate-level log P_B(s | s + e) for uniform removable-edge backward policy.

    candidate_batch_ids are rollout row ids, not static graph ids. For every
    candidate (row, edge), this constructs the child edge mask obtained by
    adding edge to that row and counts valid removable non-root edges in that
    child. The just-added edge must be removable; otherwise the forward action
    would not have a valid backward transition.
    """
    device = state.active_edges.device
    edge_ids = candidate_edge_ids.to(device=device, dtype=torch.long).view(-1)
    row_ids = candidate_batch_ids.to(device=device, dtype=torch.long).view(-1)
    if edge_ids.shape != row_ids.shape:
        raise ValueError(
            "candidate_edge_ids and candidate_batch_ids must have matching shape: "
            f"{tuple(edge_ids.shape)} != {tuple(row_ids.shape)}."
        )
    if edge_ids.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)

    num_edges = int(edge_batch.numel())
    num_rollouts = int(state.num_rollouts)
    if bool((edge_ids < 0).any()) or bool((edge_ids >= num_edges).any()):
        raise ValueError("candidate_edge_ids must contain physical edge ids.")
    if bool((row_ids < 0).any()) or bool((row_ids >= num_rollouts).any()):
        raise ValueError("candidate_batch_ids must contain rollout row ids.")

    edge_batch = edge_batch.to(device=device, dtype=torch.long)
    rollout_to_graph = state.rollout_to_graph.to(device=device, dtype=torch.long)
    selected_edge_graph = edge_batch.index_select(0, edge_ids)
    expected_graph = rollout_to_graph.index_select(0, row_ids)
    if not torch.equal(selected_edge_graph, expected_graph):
        raise RuntimeError(
            "Each candidate edge must belong to its rollout row's static graph."
        )

    active_at_candidate = state.active_edges[row_ids, edge_ids]
    if bool(active_at_candidate.any()):
        bad = torch.nonzero(active_at_candidate, as_tuple=False).view(-1)
        raise RuntimeError(
            "Candidate expansion edges must be inactive in the parent state, "
            f"bad candidate positions={bad.tolist()}."
        )

    values: list[torch.Tensor] = []
    edge_index = edge_index.to(device=device, dtype=torch.long)
    for row_tensor, edge_tensor in zip(row_ids, edge_ids):
        row_id = int(row_tensor.item())
        edge_id = int(edge_tensor.item())
        static_graph_id = int(rollout_to_graph[row_id].item())

        child_edges = state.active_edges[row_id].clone()
        child_edges[edge_id] = True

        removable_mask, counts = compute_uniform_nonroot_backward_removals(
            active_edges=child_edges,
            edge_index=edge_index,
            anchor_mask=state.anchor_nodes[row_id],
            edge_batch=edge_batch,
            num_graphs=int(num_graphs),
            root_edges=state.root_edges[row_id],
            validate=False,
        )
        if not bool(removable_mask[edge_id]):
            raise RuntimeError(
                "Candidate edge is not removable from its child state under the "
                f"uniform backward policy: rollout_id={row_id}, edge_id={edge_id}."
            )

        count = counts[static_graph_id].to(dtype=torch.float32)
        if bool(count.lt(1.0)):
            raise RuntimeError(
                "No valid backward removals for candidate child state: "
                f"rollout_id={row_id}, static_graph_id={static_graph_id}."
            )
        values.append(-torch.log(count))

    return torch.stack(values, dim=0).to(device=device, dtype=torch.float32)


__all__ = ["compute_candidate_uniform_backward_log_probs"]
