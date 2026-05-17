from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext, RewardContext
from src.weaver.state import State, derive_node_mask


@dataclass(frozen=True, slots=True)
class RewardOutput:
    log_reward: torch.Tensor
    answer_gain: torch.Tensor
    edge_penalty: torch.Tensor
    fail_penalty: torch.Tensor

    @property
    def answer_recall(self) -> torch.Tensor:
        return self.answer_gain

    @property
    def selected_edge_count(self) -> torch.Tensor:
        return self.edge_penalty

    @property
    def no_answer(self) -> torch.Tensor:
        return self.fail_penalty


class EvidenceLogReward(nn.Module):
    """
    Deterministic answer-support log reward used only in training objectives.
    """

    def __init__(
        self,
        *,
        alpha: float = 4.0,
        lambda_: float = 0.02,
        eta: float = 6.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.eta = float(eta)
        if self.lambda_ < 0.0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}.")
        if self.eta < 0.0:
            raise ValueError(f"eta must be non-negative, got {self.eta}.")

    @torch.no_grad()
    def prepare_context(
        self,
        batch: RetrievalBatch,
        *,
        expand_budget: int,
    ) -> RewardContext:
        device = batch.edge_index.device
        node_to_graph = batch.batch.to(device=device, dtype=torch.long).view(-1)
        edge_index = batch.edge_index.to(device=device, dtype=torch.long)

        target_nodes = torch.unique(
            batch.reachable_target_node_ids.to(device=device, dtype=torch.long).view(-1),
            sorted=False,
        )
        target_mask = torch.zeros(
            int(batch.num_nodes_total),
            dtype=torch.bool,
            device=device,
        )
        if target_nodes.numel() > 0:
            target_mask[target_nodes] = True

        anchor_mask = torch.zeros(
            int(batch.num_nodes_total),
            dtype=torch.bool,
            device=device,
        )
        anchors = batch.anchor_node_ids.to(device=device, dtype=torch.long).view(-1)
        if anchors.numel() > 0:
            anchor_mask[anchors] = True

        target_count_by_graph = _count_targets_by_graph(
            target_nodes=target_nodes,
            node_to_graph=node_to_graph,
            num_graphs=int(batch.num_graphs_total),
        )
        return RewardContext(
            target_mask=target_mask,
            target_count_by_graph=target_count_by_graph,
            edge_index=edge_index,
            node_to_graph=node_to_graph,
            anchor_mask=anchor_mask,
            expand_budget=int(expand_budget),
        )

    def forward(
        self,
        *,
        state: State,
        context: RewardContext,
    ) -> RewardOutput:
        with torch.no_grad():
            context = context.to(device=state.device)
            row_to_graph = state.row_to_graph.to(device=state.device, dtype=torch.long)
            target_count = context.target_count_by_graph.index_select(0, row_to_graph)
            node_mask = derive_node_mask(
                state=state,
                graph_context=GraphContext(
                    edge_index=context.edge_index,
                    node_to_graph=context.node_to_graph,
                    edge_to_graph=context.node_to_graph.index_select(
                        0,
                        context.edge_index[0],
                    ),
                    anchor_mask=context.anchor_mask,
                    num_nodes=int(context.node_to_graph.numel()),
                    num_edges=int(context.edge_index.size(1)),
                    num_graphs=int(context.target_count_by_graph.numel()),
                    device=context.edge_index.device,
                ),
            )
            active_targets = node_mask & context.target_mask.view(1, -1)
            hit_count = active_targets.sum(dim=1).to(dtype=torch.float32)
            answer_gain = hit_count / target_count.clamp_min(1).to(dtype=torch.float32)
            fail_penalty = hit_count.eq(0.0)
            edge_penalty = state.edge_mask.sum(dim=1).to(dtype=torch.float32)
            log_reward = (
                self.alpha * answer_gain
                - self.lambda_ * edge_penalty
                - self.eta * fail_penalty.to(dtype=torch.float32)
            )
        return RewardOutput(
            log_reward=log_reward,
            answer_gain=answer_gain,
            edge_penalty=edge_penalty,
            fail_penalty=fail_penalty,
        )


def _count_targets_by_graph(
    *,
    target_nodes: torch.Tensor,
    node_to_graph: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    if target_nodes.numel() == 0:
        return torch.zeros(int(num_graphs), dtype=torch.long, device=node_to_graph.device)
    return torch.bincount(
        node_to_graph.index_select(0, target_nodes),
        minlength=int(num_graphs),
    ).to(dtype=torch.long)


__all__ = [
    "EvidenceLogReward",
    "RewardOutput",
]
