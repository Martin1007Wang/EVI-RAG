from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.models.policy import Policy
from src.models.rollout import RolloutState, _scatter_log_softmax
from src.utils.path_utils import compute_shortest_path_labels


@dataclass(frozen=True)
class TeacherWarmstartOutput:
    loss: torch.Tensor
    type_loss: torch.Tensor
    edge_loss: torch.Tensor
    supervised_states: torch.Tensor
    expand_states: torch.Tensor
    active_graphs: torch.Tensor


def _zero(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def _choose_teacher_edge(
    *,
    gold_edge_ids: torch.Tensor,
    active_nodes: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    if gold_edge_ids.numel() == 0:
        raise ValueError("gold_edge_ids must be non-empty.")

    src = edge_index[0].index_select(0, gold_edge_ids)
    dst = edge_index[1].index_select(0, gold_edge_ids)
    src_active = active_nodes.index_select(0, src)
    dst_active = active_nodes.index_select(0, dst)
    activates_new_node = (src_active & ~dst_active) | (dst_active & ~src_active)
    preferred = gold_edge_ids[activates_new_node]
    if preferred.numel() == 0:
        preferred = gold_edge_ids
    return preferred[:1]


class ShortestPathTeacherWarmup(nn.Module):
    def __init__(
        self,
        *,
        max_steps: int,
        initial_weight: float = 1.0,
        final_weight: float = 0.0,
        total_steps: int = 5000,
        path_mode: str = "undirected",
        stop_on_first_hit: bool = True,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.initial_weight = float(initial_weight)
        self.final_weight = float(final_weight)
        self.total_steps = int(total_steps)
        self.path_mode = str(path_mode)
        self.stop_on_first_hit = bool(stop_on_first_hit)

    def weight(self, global_step: int) -> float:
        if self.total_steps <= 0:
            return self.final_weight
        progress = min(max(float(global_step), 0.0) / float(self.total_steps), 1.0)
        return self.initial_weight + progress * (
            self.final_weight - self.initial_weight
        )

    def _build_positive_edge_mask(
        self,
        batch: RetrievalBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch.edge_index.device
        num_graphs = batch.num_graphs
        positive_edge_mask = torch.zeros(
            batch.edge_index.size(1), dtype=torch.bool, device=device
        )
        reachable_graph_mask = torch.zeros(num_graphs, dtype=torch.bool, device=device)

        for graph_idx in range(num_graphs):
            node_start = int(batch.ptr[graph_idx].item())
            node_end = int(batch.ptr[graph_idx + 1].item())
            edge_start = int(batch.edge_ptr[graph_idx].item())
            edge_end = int(batch.edge_ptr[graph_idx + 1].item())

            labels = compute_shortest_path_labels(
                edge_index=(
                    batch.edge_index[:, edge_start:edge_end] - node_start
                ).cpu(),
                is_anchor_mask=batch.is_anchor_mask[node_start:node_end].cpu(),
                is_target_mask=batch.is_target_mask[node_start:node_end].cpu(),
                num_nodes=node_end - node_start,
                path_mode=self.path_mode,
            )
            if (
                labels.positive_edge_ids.numel() == 0
                or labels.reachable_target_node_ids.numel() == 0
            ):
                continue

            reachable_graph_mask[graph_idx] = True
            positive_edge_mask[
                edge_start + labels.positive_edge_ids.long().to(device=device)
            ] = True

        return positive_edge_mask, reachable_graph_mask

    def forward(
        self,
        *,
        policy: Policy,
        batch: RetrievalBatch,
    ) -> TeacherWarmstartOutput:
        device = batch.node_tokens.device
        positive_edge_mask, reachable_graph_mask = self._build_positive_edge_mask(batch)
        if not bool(reachable_graph_mask.any().item()):
            zero = _zero(device)
            return TeacherWarmstartOutput(
                loss=zero,
                type_loss=zero,
                edge_loss=zero,
                supervised_states=zero,
                expand_states=zero,
                active_graphs=zero,
            )

        active_teacher_graphs = reachable_graph_mask.clone()
        rollout_state = RolloutState.initialize(batch)
        type_losses: list[torch.Tensor] = []
        edge_losses: list[torch.Tensor] = []
        src = batch.edge_index[0]
        dst = batch.edge_index[1]
        edge_batch_idx = batch.edge_batch
        num_graphs = batch.num_graphs
        supervised_states = 0
        expand_states = 0

        for _ in range(self.max_steps + 1):
            if not bool(active_teacher_graphs.any().item()):
                break

            step_output = policy(batch, rollout_state.snapshot())
            type_log_probs = torch.log_softmax(
                step_output.action_logits["type_logits"], dim=-1
            )
            valid_edges_mask = (
                rollout_state.active_nodes[src] | rollout_state.active_nodes[dst]
            ) & ~rollout_state.active_edges
            teacher_candidate_mask = valid_edges_mask & positive_edge_mask
            teacher_candidate_counts = scatter_sum(
                teacher_candidate_mask.int(), edge_batch_idx, dim=0, dim_size=num_graphs
            )
            target_active_counts = scatter_sum(
                (rollout_state.active_nodes & batch.is_target_mask).int(),
                batch.batch,
                dim=0,
                dim_size=num_graphs,
            )
            target_active = target_active_counts.gt(0)

            if self.stop_on_first_hit:
                stop_target = active_teacher_graphs & target_active
            else:
                stop_target = (
                    active_teacher_graphs
                    & target_active
                    & teacher_candidate_counts.eq(0)
                )
            expand_target = (
                active_teacher_graphs & ~stop_target & teacher_candidate_counts.gt(0)
            )
            stalled_graphs = active_teacher_graphs & ~stop_target & ~expand_target

            if bool(stop_target.any().item()):
                type_losses.append(-type_log_probs[stop_target, 1])
                supervised_states += int(stop_target.sum().item())

            if bool(expand_target.any().item()):
                type_losses.append(-type_log_probs[expand_target, 0])
                supervised_states += int(expand_target.sum().item())
                expand_states += int(expand_target.sum().item())

                expand_graph_ids = torch.nonzero(expand_target, as_tuple=False).view(-1)
                candidate_edge_ids = torch.nonzero(
                    valid_edges_mask & expand_target[edge_batch_idx], as_tuple=False
                ).view(-1)
                graph_remap = torch.full(
                    (num_graphs,), -1, dtype=torch.long, device=device
                )
                graph_remap[expand_graph_ids] = torch.arange(
                    expand_graph_ids.numel(), dtype=torch.long, device=device
                )
                candidate_batch_idx = graph_remap[edge_batch_idx[candidate_edge_ids]]
                candidate_log_probs = _scatter_log_softmax(
                    step_output.action_logits["expand_edge_logits"].index_select(
                        0, candidate_edge_ids
                    ),
                    candidate_batch_idx,
                    num_segments=expand_graph_ids.numel(),
                )
                candidate_is_gold = positive_edge_mask.index_select(
                    0, candidate_edge_ids
                )

                chosen_edges: list[torch.Tensor] = []
                for local_graph_idx, graph_id in enumerate(expand_graph_ids.tolist()):
                    local_mask = candidate_batch_idx == local_graph_idx
                    local_gold_mask = candidate_is_gold[local_mask]
                    if not bool(local_gold_mask.any().item()):
                        raise RuntimeError(
                            "Teacher expand state has no positive candidate edge for graph "
                            f"{graph_id}."
                        )

                    local_edge_ids = candidate_edge_ids[local_mask]
                    local_log_probs = candidate_log_probs[local_mask]
                    chosen_teacher_edge = _choose_teacher_edge(
                        gold_edge_ids=local_edge_ids[local_gold_mask],
                        active_nodes=rollout_state.active_nodes,
                        edge_index=batch.edge_index,
                    )
                    chosen_local_idx = torch.nonzero(
                        local_edge_ids == chosen_teacher_edge.item(), as_tuple=False
                    ).view(-1)
                    if chosen_local_idx.numel() != 1:
                        raise RuntimeError(
                            "Expected exactly one chosen teacher edge in local candidate set for "
                            f"graph {graph_id}, got {chosen_local_idx.numel()}."
                        )
                    edge_losses.append(-local_log_probs[chosen_local_idx.item()])
                    chosen_edges.append(chosen_teacher_edge)

                rollout_state.apply_expansion(
                    chosen_edges=torch.cat(chosen_edges, dim=0),
                    src=src,
                    dst=dst,
                )

            active_teacher_graphs = active_teacher_graphs & ~(
                stop_target | stalled_graphs
            )

        type_loss = (
            torch.cat(type_losses, dim=0).mean() if type_losses else _zero(device)
        )
        edge_loss = torch.stack(edge_losses).mean() if edge_losses else _zero(device)
        loss = type_loss + edge_loss
        return TeacherWarmstartOutput(
            loss=loss,
            type_loss=type_loss,
            edge_loss=edge_loss,
            supervised_states=torch.tensor(float(supervised_states), device=device),
            expand_states=torch.tensor(float(expand_states), device=device),
            active_graphs=torch.tensor(
                float(reachable_graph_mask.sum().item()), device=device
            ),
        )


__all__ = ["ShortestPathTeacherWarmup", "TeacherWarmstartOutput"]
