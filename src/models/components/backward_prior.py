from __future__ import annotations

import torch

from src.models.environment.contracts import GraphEnvContext


class StructuralBackwardPrior:
    """Structured backward prior P_B(s_t | s_{t+1}) defined by graph topology."""

    def __init__(self, *, mode: str = "uniform_in_degree") -> None:
        normalized = str(mode).strip().lower()
        if normalized not in {"uniform_in_degree", "uniform"}:
            raise ValueError(
                "backward prior mode must be one of {'uniform_in_degree', 'uniform'}, "
                f"got {mode!r}."
            )
        self.mode = "uniform_in_degree" if normalized == "uniform" else normalized

    def log_prob(
        self,
        *,
        env_context: GraphEnvContext,
        source_nodes: torch.Tensor,
        chosen_target_nodes: torch.Tensor,
        active_flat: torch.Tensor,
        is_stop_flat: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mode != "uniform_in_degree":
            raise ValueError(f"Unsupported backward prior mode: {self.mode!r}")
        return self._uniform_in_degree_log_prob_transition(
            env_context=env_context,
            source_nodes=source_nodes,
            chosen_target_nodes=chosen_target_nodes,
            active_flat=active_flat,
            is_stop_flat=is_stop_flat,
            dtype=dtype,
        )

    def log_prob_edges(
        self,
        *,
        env_context: GraphEnvContext,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_graph_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mode != "uniform_in_degree":
            raise ValueError(f"Unsupported backward prior mode: {self.mode!r}")
        return self._uniform_in_degree_log_prob_edges(
            env_context=env_context,
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_graph_ids=edge_graph_ids,
            dtype=dtype,
        )

    @staticmethod
    def _flat_graph_ids(*, num_items: int, num_graphs: int, device: torch.device) -> torch.Tensor:
        if num_graphs <= 0:
            raise ValueError("num_graphs must be positive when constructing graph ids.")
        if num_items % num_graphs != 0:
            raise ValueError(
                "source_nodes shape mismatch with env_context.num_graphs in structural backward prior."
            )
        num_agents = max(num_items // num_graphs, 1)
        return torch.arange(num_graphs, device=device, dtype=torch.long).repeat_interleave(num_agents)

    @staticmethod
    def _uniform_in_degree_log_prob_edges(
        *,
        env_context: GraphEnvContext,
        source_nodes: torch.Tensor,
        target_nodes: torch.Tensor,
        edge_graph_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        crow = env_context.adj_t_bwd.crow_indices()
        target = target_nodes.clamp(min=0)
        in_deg = (crow[target + 1] - crow[target]).to(dtype=dtype)
        if bool((in_deg <= 0).any().item()):
            raise ValueError("uniform_in_degree backward prior encountered non-positive in-degree on sampled moves.")
        log_pb = -torch.log(in_deg)

        if env_context.start_local_indices is None:
            return log_pb
        super_abs = env_context.start_local_indices + env_context.node_ptr[:-1]
        super_nodes = super_abs.index_select(0, edge_graph_ids)
        source_is_super = source_nodes == super_nodes
        if bool(source_is_super.any().item()):
            log_pb = torch.where(source_is_super, torch.zeros_like(log_pb), log_pb)
        return log_pb

    @classmethod
    def _uniform_in_degree_log_prob_transition(
        cls,
        *,
        env_context: GraphEnvContext,
        source_nodes: torch.Tensor,
        chosen_target_nodes: torch.Tensor,
        active_flat: torch.Tensor,
        is_stop_flat: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        log_pb = torch.zeros_like(chosen_target_nodes, dtype=dtype)
        active_stop = active_flat & is_stop_flat
        if bool(active_stop.any().item()):
            log_pb[active_stop] = 0
        active_move = active_flat & (~is_stop_flat)
        if not bool(active_move.any().item()):
            return log_pb

        graph_ids = cls._flat_graph_ids(
            num_items=int(source_nodes.numel()),
            num_graphs=int(env_context.num_graphs),
            device=source_nodes.device,
        )
        move_rows = torch.where(active_move)[0]
        move_log_pb = cls._uniform_in_degree_log_prob_edges(
            env_context=env_context,
            source_nodes=source_nodes.index_select(0, move_rows),
            target_nodes=chosen_target_nodes.index_select(0, move_rows),
            edge_graph_ids=graph_ids.index_select(0, move_rows),
            dtype=dtype,
        )
        log_pb.index_copy_(0, move_rows, move_log_pb)
        return log_pb


__all__ = ["StructuralBackwardPrior"]
