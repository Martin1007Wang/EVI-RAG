from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.models.reward import RewardModel
from src.models.rollout import RolloutState
from src.utils.path_utils import compute_shortest_path_labels


@dataclass(frozen=True)
class HitGraphRewardResult:
    status: str
    log_reward: float | None
    recall: float | None
    added_edges: int | None


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


def build_teacher_hit_graph(
    batch: RetrievalBatch,
    *,
    path_mode: str = "qa_directed",
    stop_on_first_hit: bool = True,
) -> tuple[str, RolloutState]:
    sp_labels = compute_shortest_path_labels(
        edge_index=batch.edge_index.cpu(),
        is_anchor_mask=batch.is_anchor_mask.cpu(),
        is_target_mask=batch.is_target_mask.cpu(),
        num_nodes=batch.num_nodes,
        path_mode=path_mode,
    )

    rollout_state = RolloutState.initialize(batch)
    target_active = rollout_state.active_nodes & batch.is_target_mask
    if bool(target_active.any().item()):
        return "root_hit", rollout_state

    if (
        sp_labels.positive_edge_ids.numel() == 0
        or sp_labels.reachable_target_node_ids.numel() == 0
    ):
        return "skipped_no_path", rollout_state

    positive_edge_mask = torch.zeros(
        batch.edge_index.size(1), dtype=torch.bool, device=batch.edge_index.device
    )
    positive_edge_mask[
        sp_labels.positive_edge_ids.long().to(batch.edge_index.device)
    ] = True

    src = batch.edge_index[0]
    dst = batch.edge_index[1]
    for _ in range(int(batch.edge_index.size(1)) + 1):
        target_active = rollout_state.active_nodes & batch.is_target_mask
        if bool(target_active.any().item()):
            return "ok", rollout_state

        valid_edges = (
            rollout_state.active_nodes[src] | rollout_state.active_nodes[dst]
        ) & ~rollout_state.active_edges
        candidate_edge_ids = torch.nonzero(valid_edges, as_tuple=False).view(-1)
        if candidate_edge_ids.numel() == 0:
            break

        gold_mask_in_candidates = positive_edge_mask.index_select(0, candidate_edge_ids)
        if not bool(gold_mask_in_candidates.any().item()):
            break

        teacher_gold_edges = candidate_edge_ids[gold_mask_in_candidates]
        chosen_teacher_edge = _choose_teacher_edge(
            gold_edge_ids=teacher_gold_edges,
            active_nodes=rollout_state.active_nodes,
            edge_index=batch.edge_index,
        )
        rollout_state.apply_expansion(
            chosen_edges=chosen_teacher_edge,
            src=src,
            dst=dst,
        )
        if stop_on_first_hit:
            target_active = rollout_state.active_nodes & batch.is_target_mask
            if bool(target_active.any().item()):
                return "ok", rollout_state

    target_active = rollout_state.active_nodes & batch.is_target_mask
    if bool(target_active.any().item()):
        return "ok", rollout_state
    return "stalled_before_hit", rollout_state


@torch.no_grad()
def evaluate_hit_graph_reward(
    batch: RetrievalBatch,
    *,
    reward_model: RewardModel,
    path_mode: str = "qa_directed",
    stop_on_first_hit: bool = True,
) -> HitGraphRewardResult:
    if batch.num_graphs != 1:
        raise ValueError(
            f"evaluate_hit_graph_reward expects a single-graph batch, got {batch.num_graphs}."
        )

    status, rollout_state = build_teacher_hit_graph(
        batch,
        path_mode=path_mode,
        stop_on_first_hit=stop_on_first_hit,
    )
    if status not in {"ok", "root_hit"}:
        return HitGraphRewardResult(
            status=status,
            log_reward=None,
            recall=None,
            added_edges=None,
        )

    log_reward = reward_model(
        base_graph=batch,
        active_nodes=rollout_state.active_nodes,
        active_edges=rollout_state.active_edges,
        root_active_edges=rollout_state.root_active_edges,
    )
    active_gold = rollout_state.active_nodes & batch.is_target_mask
    recall = active_gold.sum().float() / batch.is_target_mask.sum().clamp(min=1).float()
    added_edges = int(
        (rollout_state.active_edges & ~rollout_state.root_active_edges).sum().item()
    )
    return HitGraphRewardResult(
        status=status,
        log_reward=float(log_reward.item()),
        recall=float(recall.item()),
        added_edges=added_edges,
    )


def summarize_hit_graph_rewards(results: list[HitGraphRewardResult]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    successful = [result for result in results if result.log_reward is not None]
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1

    if not successful:
        return {
            "num_graphs": len(results),
            "graphs_with_hit_graph": 0,
            "hit_graph_rate": 0.0,
            "avg_hit_graph_log_reward": 0.0,
            "avg_hit_graph_recall": 0.0,
            "avg_hit_graph_added_edges": 0.0,
            "status_counts": status_counts,
        }

    return {
        "num_graphs": len(results),
        "graphs_with_hit_graph": len(successful),
        "hit_graph_rate": len(successful) / len(results),
        "avg_hit_graph_log_reward": sum(
            r.log_reward for r in successful if r.log_reward is not None
        )
        / len(successful),
        "avg_hit_graph_recall": sum(
            r.recall for r in successful if r.recall is not None
        )
        / len(successful),
        "avg_hit_graph_added_edges": sum(
            r.added_edges for r in successful if r.added_edges is not None
        )
        / len(successful),
        "status_counts": status_counts,
    }


__all__ = [
    "HitGraphRewardResult",
    "build_teacher_hit_graph",
    "evaluate_hit_graph_reward",
    "summarize_hit_graph_rewards",
]
