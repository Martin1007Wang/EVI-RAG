from __future__ import annotations

import torch
from torch import nn

from src.models.components import GraphLogZHead
from src.models.configs import HeuristicConfig

from .heuristic import TrajectoryHeuristic
from .trajectory_policy import (
    build_start_distribution_from_logits,
    resolve_start_candidates,
    TrajectoryPolicy,
)
from .state import SearchState
from .types import (
    ForwardActionDistribution,
    PreparedGFlowNetBatch,
    StartDistribution,
)


def _segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    output = values.new_zeros((num_segments, int(values.size(1))))
    counts = values.new_zeros((num_segments, 1))
    if int(values.numel()) == 0:
        return output
    output.scatter_add_(0, segment_ids.unsqueeze(1).expand_as(values), values)
    counts.scatter_add_(
        0,
        segment_ids.unsqueeze(1),
        values.new_ones((int(values.size(0)), 1)),
    )
    return output / counts.clamp_min(1.0)


def _mask_nonfinite_scores(values: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(values, float("-inf"))
    return torch.where(torch.isfinite(values), values, neg_inf)


class GFlowNetPolicy(nn.Module):
    def __init__(
        self,
        *,
        base_policy: TrajectoryPolicy,
        heuristic_cfg: HeuristicConfig,
        graph_log_z_head: GraphLogZHead,
        trajectory_heuristic: TrajectoryHeuristic,
    ) -> None:
        super().__init__()
        self.base_policy = base_policy
        self._heuristic_cfg = heuristic_cfg
        self.graph_log_z_head = graph_log_z_head
        self.trajectory_heuristic = trajectory_heuristic

    @property
    def heuristic_cfg(self) -> HeuristicConfig:
        return self._heuristic_cfg

    def prepare_batch(self, batch) -> PreparedGFlowNetBatch:
        prepared_batch = self.base_policy.prepare_batch(batch)
        heuristic_cache = self.trajectory_heuristic.build_cache(prepared_batch)
        return PreparedGFlowNetBatch(
            topology=prepared_batch.topology,
            observation=prepared_batch.observation,
            node_tokens=prepared_batch.node_tokens,
            question_tokens=prepared_batch.question_tokens,
            heuristic_cache=heuristic_cache,
        )

    def encode(self, batch) -> PreparedGFlowNetBatch:
        return self.prepare_batch(batch)

    def compute_graph_log_z(
        self, prepared_batch: PreparedGFlowNetBatch
    ) -> torch.Tensor:
        q_local_indices = prepared_batch.observation.q_local_indices
        candidate_nodes_abs, candidate_graph_ids = (
            prepared_batch.topology.resolve_local_node_indices(
                q_local_indices,
                field_name="q_local_indices",
            )
        )
        if int(candidate_nodes_abs.numel()) == 0:
            return torch.zeros(
                (int(prepared_batch.topology.num_graphs),),
                device=prepared_batch.node_tokens.device,
                dtype=torch.float32,
            )
        start_summary = _segment_mean(
            prepared_batch.node_tokens.index_select(0, candidate_nodes_abs),
            candidate_graph_ids,
            num_segments=int(prepared_batch.topology.num_graphs),
        )
        return self.graph_log_z_head(
            question_features=prepared_batch.question_tokens,
            start_summary=start_summary,
        ).to(dtype=torch.float32)

    def compute_start_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
    ) -> StartDistribution:
        candidate_nodes_abs, candidate_graph_ids = resolve_start_candidates(
            prepared_batch
        )
        node_features = prepared_batch.node_tokens.index_select(0, candidate_nodes_abs)
        question_features = prepared_batch.question_tokens.index_select(
            0, candidate_graph_ids
        )
        logits = self.base_policy.start_head(
            node_features=node_features,
            question_features=question_features,
        ).to(dtype=torch.float32)
        start_bias = self.trajectory_heuristic.compute_start_bias(
            prepared_batch=prepared_batch,
            heuristic_cache=prepared_batch.heuristic_cache,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            build_state_features=self.base_policy.build_state_features,
        )
        return build_start_distribution_from_logits(
            prepared_batch=prepared_batch,
            candidate_nodes_abs=candidate_nodes_abs,
            candidate_graph_ids=candidate_graph_ids,
            logits=logits + float(self.heuristic_cfg.beta) * start_bias,
        )

    @staticmethod
    def sample_start_nodes(
        distribution: StartDistribution,
        *,
        num_rollouts: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(distribution.candidate_graph_ids.max().item()) + 1
        selected_nodes: list[torch.Tensor] = []
        selected_log_probs: list[torch.Tensor] = []
        for graph_idx in range(num_graphs):
            mask = distribution.candidate_graph_ids == graph_idx
            graph_nodes = distribution.candidate_nodes_abs[mask]
            graph_log_probs = distribution.log_probs[mask]
            if int(graph_nodes.numel()) == 0:
                raise ValueError("Each graph must expose at least one start candidate.")
            if deterministic:
                order = torch.argsort(graph_log_probs, descending=True)
                graph_nodes = graph_nodes.index_select(0, order)
                graph_log_probs = graph_log_probs.index_select(0, order)
                if int(graph_nodes.numel()) < num_rollouts:
                    repeat_idx = torch.remainder(
                        torch.arange(num_rollouts, device=graph_nodes.device),
                        int(graph_nodes.numel()),
                    )
                    graph_nodes = graph_nodes.index_select(0, repeat_idx)
                    graph_log_probs = graph_log_probs.index_select(0, repeat_idx)
                else:
                    graph_nodes = graph_nodes[:num_rollouts]
                    graph_log_probs = graph_log_probs[:num_rollouts]
            else:
                probs = torch.softmax(graph_log_probs, dim=0)
                sample_idx = torch.multinomial(
                    probs,
                    num_samples=num_rollouts,
                    replacement=True,
                )
                graph_nodes = graph_nodes.index_select(0, sample_idx)
                graph_log_probs = graph_log_probs.index_select(0, sample_idx)
            selected_nodes.append(graph_nodes)
            selected_log_probs.append(graph_log_probs)
        return torch.stack(selected_nodes, dim=0), torch.stack(
            selected_log_probs, dim=0
        )

    def compute_log_state_scores(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> torch.Tensor:
        return self.base_policy.compute_log_state_scores(prepared_batch, state)

    def compute_forward_distribution(
        self,
        prepared_batch: PreparedGFlowNetBatch,
        state: SearchState,
    ) -> ForwardActionDistribution:
        distribution = self.base_policy.compute_forward_distribution(
            prepared_batch,
            state,
        )
        if int(distribution.edge_logits.numel()) == 0:
            return distribution
        transition_bias = self.trajectory_heuristic.compute_transition_bias(
            prepared_batch=prepared_batch,
            heuristic_cache=prepared_batch.heuristic_cache,
            distribution=distribution,
            state=state,
            build_state_features=self.base_policy.build_state_features,
        )
        return ForwardActionDistribution(
            edge_logits=_mask_nonfinite_scores(
                distribution.edge_logits.to(dtype=torch.float32)
                + float(self.heuristic_cfg.beta) * transition_bias
            ),
            edge_agent_batch=distribution.edge_agent_batch,
            edge_ids=distribution.edge_ids,
            target_nodes=distribution.target_nodes,
            out_degrees=distribution.out_degrees,
        )

    @staticmethod
    def compute_move_log_probs(
        distribution: ForwardActionDistribution,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return TrajectoryPolicy.compute_move_log_probs(distribution)


__all__ = [
    "GFlowNetPolicy",
    "PreparedGFlowNetBatch",
]
