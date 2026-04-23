from __future__ import annotations

import torch

from src.data.schema import RetrievalBatch
from src.utils.reward_utils import build_anchor_induced_edge_mask

from .types import RolloutBatch


class StateReconstructor:
    """Reconstruct rollout terminal subgraphs from the rollout trace tensors.

    Single source of truth:
        - retrieval_batch
        - rollout.stats
        - rollout.traces

    The rollout engine no longer stores duplicated terminal-state snapshots.
    Evaluation and analysis should derive graph masks on demand through this helper.
    """

    @staticmethod
    def terminal_active_edges(
        retrieval_batch: RetrievalBatch,
        rollout: RolloutBatch,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        resolved_device = StateReconstructor._resolve_device(
            rollout=rollout,
            device=device,
        )
        root_active_edges = build_anchor_induced_edge_mask(
            retrieval_batch.edge_index.to(resolved_device),
            retrieval_batch.is_anchor_mask.to(resolved_device),
        )
        selected_edge_ids = rollout.traces.selected_edge_ids.to(resolved_device)
        lengths = rollout.stats.traj_len.to(device=resolved_device, dtype=torch.long)
        if selected_edge_ids.ndim != 2:
            raise ValueError(
                "rollout.traces.selected_edge_ids must be 2-D, "
                f"got {tuple(selected_edge_ids.shape)}."
            )
        if lengths.shape != selected_edge_ids.shape[:1]:
            raise ValueError(
                "rollout.stats.traj_len batch shape must match selected_edge_ids. "
                f"Got traj_len={tuple(lengths.shape)}, selected_edge_ids={tuple(selected_edge_ids.shape)}."
            )

        step_ids = torch.arange(selected_edge_ids.shape[1], device=resolved_device).unsqueeze(0)
        valid_steps = step_ids < lengths.unsqueeze(1)
        chosen_mask = selected_edge_ids.ge(0) & valid_steps

        active_edges = root_active_edges.clone()
        if bool(chosen_mask.any().item()):
            active_edges[selected_edge_ids[chosen_mask].long()] = True
        return active_edges

    @staticmethod
    def terminal_active_nodes(
        retrieval_batch: RetrievalBatch,
        rollout: RolloutBatch,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        active_edges = StateReconstructor.terminal_active_edges(
            retrieval_batch,
            rollout,
            device=device,
        )
        return StateReconstructor._active_nodes_from_edges(retrieval_batch, active_edges)

    @staticmethod
    def terminal_subgraph_masks(
        retrieval_batch: RetrievalBatch,
        rollout: RolloutBatch,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active_edges = StateReconstructor.terminal_active_edges(
            retrieval_batch,
            rollout,
            device=device,
        )
        active_nodes = StateReconstructor._active_nodes_from_edges(retrieval_batch, active_edges)
        return active_nodes, active_edges

    @staticmethod
    def _active_nodes_from_edges(
        retrieval_batch: RetrievalBatch,
        active_edges: torch.Tensor,
    ) -> torch.Tensor:
        device = active_edges.device
        active_nodes = retrieval_batch.is_anchor_mask.to(device).clone()
        if bool(active_edges.any().item()):
            edge_index = retrieval_batch.edge_index.to(device)
            src = edge_index[0]
            dst = edge_index[1]
            active_nodes[src[active_edges]] = True
            active_nodes[dst[active_edges]] = True
        return active_nodes

    @staticmethod
    def _resolve_device(
        *,
        rollout: RolloutBatch,
        device: torch.device | None,
    ) -> torch.device:
        if device is not None:
            return torch.device(device)
        return rollout.traces.selected_edge_ids.device


__all__ = ["StateReconstructor"]
