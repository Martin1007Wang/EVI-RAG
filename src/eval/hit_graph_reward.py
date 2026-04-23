from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from src.data.schema import RetrievalBatch
from src.models.guidance import TeacherGuidance
from src.models.policy import CandidateEdges
from src.models.reward import RewardModel
from src.models.state import State


@dataclass(frozen=True)
class HitGraphRewardResult:
    status: str
    log_reward: float | None
    recall: float | None
    added_edges: int | None


def _choose_teacher_edge(
    *,
    gold_edge_ids: torch.Tensor,
    teacher_scores: torch.Tensor,
) -> torch.Tensor:
    if gold_edge_ids.numel() == 0:
        raise ValueError("gold_edge_ids must be non-empty.")
    if teacher_scores.numel() != gold_edge_ids.numel():
        raise ValueError(
            "teacher_scores must align with gold_edge_ids: "
            f"{teacher_scores.numel()} != {gold_edge_ids.numel()}."
        )
    chosen_pos = int(torch.argmax(teacher_scores).item())
    return gold_edge_ids[chosen_pos : chosen_pos + 1]


def _build_frontier_candidates(batch: RetrievalBatch, state: State) -> CandidateEdges:
    src = batch.edge_index[0]
    dst = batch.edge_index[1]
    valid_edges = (
        state.active_nodes[src] & ~state.active_nodes[dst] & ~state.active_edges
    )
    candidate_edge_ids = torch.nonzero(valid_edges, as_tuple=False).view(-1)
    return CandidateEdges(
        edge_ids=candidate_edge_ids,
        expand_logits=torch.zeros(
            candidate_edge_ids.numel(),
            dtype=torch.float32,
            device=batch.edge_index.device,
        ),
        batch_index=torch.zeros(
            candidate_edge_ids.numel(), dtype=torch.long, device=batch.edge_index.device
        ),
    )


def build_teacher_hit_graph(
    batch: RetrievalBatch,
    *,
    teacher_guidance: TeacherGuidance | None = None,
    path_mode: str = "qa_directed",
    expand_budget: int | None = None,
) -> tuple[str, State]:
    if str(path_mode or "qa_directed").strip().lower() != "qa_directed":
        raise ValueError(
            "build_teacher_hit_graph only supports path_mode='qa_directed'."
        )
    if batch.num_graphs != 1:
        raise ValueError(
            f"build_teacher_hit_graph expects a single-graph batch, got {batch.num_graphs}."
        )

    guidance = teacher_guidance or TeacherGuidance(score_exponent=1.0)
    resolved_expand_budget = (
        int(batch.edge_index.size(1)) if expand_budget is None else int(expand_budget)
    )
    rollout_state = State.create_initial(batch, expand_budget=resolved_expand_budget)
    train_target_mask = _get_train_target_mask(batch)
    train_target_node_ids = _get_train_target_node_ids(batch)
    if bool((rollout_state.active_nodes & train_target_mask).any().item()):
        return "root_hit", rollout_state

    if not hasattr(batch, "target_node_distance_flat"):
        return "missing_teacher_labels", rollout_state

    src = batch.edge_index[0]
    dst = batch.edge_index[1]

    for num_expands in range(resolved_expand_budget + 1):
        rollout_state.num_expands = num_expands
        candidates = _build_frontier_candidates(batch, rollout_state)
        should_stop = guidance.graph_should_stop(
            retrieval_batch=batch,
            state=rollout_state,
            candidates=candidates,
            remaining_expand_budget=resolved_expand_budget - (num_expands + 1),
            num_graphs=1,
        )
        if bool(should_stop[0].item()) or num_expands >= resolved_expand_budget:
            active_gold = rollout_state.active_nodes & train_target_mask
            if bool(active_gold.any().item()):
                return "ok", rollout_state
            if train_target_node_ids.numel() == 0:
                return "skipped_no_path", rollout_state
            return "stalled_before_hit", rollout_state

        valid_mask, teacher_scores = guidance.candidate_scores(
            retrieval_batch=batch,
            state=rollout_state,
            candidates=candidates,
            remaining_expand_budget=resolved_expand_budget - (num_expands + 1),
        )
        teacher_gold_edges = candidates.edge_ids[valid_mask]
        if teacher_gold_edges.numel() == 0:
            break

        chosen_teacher_edge = _choose_teacher_edge(
            gold_edge_ids=teacher_gold_edges,
            teacher_scores=teacher_scores[valid_mask],
        )
        rollout_state.apply_expansion(
            chosen_edges=chosen_teacher_edge,
            src=src,
            dst=dst,
        )

    active_gold = rollout_state.active_nodes & train_target_mask
    if bool(active_gold.any().item()):
        return "ok", rollout_state
    if train_target_node_ids.numel() == 0:
        return "skipped_no_path", rollout_state
    return "stalled_before_hit", rollout_state


@torch.no_grad()
def evaluate_hit_graph_reward(
    batch: RetrievalBatch,
    *,
    reward_model: RewardModel,
    teacher_guidance: TeacherGuidance | None = None,
    path_mode: str = "qa_directed",
    expand_budget: int | None = None,
) -> HitGraphRewardResult:
    if batch.num_graphs != 1:
        raise ValueError(
            f"evaluate_hit_graph_reward expects a single-graph batch, got {batch.num_graphs}."
        )

    status, rollout_state = build_teacher_hit_graph(
        batch,
        teacher_guidance=teacher_guidance,
        path_mode=path_mode,
        expand_budget=expand_budget,
    )
    if status == "missing_teacher_labels":
        return HitGraphRewardResult(
            status=status,
            log_reward=None,
            recall=None,
            added_edges=None,
        )

    log_reward = reward_model(
        retrieval_batch=batch,
        active_nodes=rollout_state.active_nodes,
        active_edges=rollout_state.active_edges,
        state=rollout_state,
    )
    train_target_mask = _get_train_target_mask(batch)
    active_gold = rollout_state.active_nodes & train_target_mask
    recall = active_gold.sum().float() / train_target_mask.sum().clamp(min=1).float()
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


def _get_train_target_mask(batch: RetrievalBatch) -> torch.Tensor:
    return batch.train_target_mask


def _get_train_target_node_ids(batch: RetrievalBatch) -> torch.Tensor:
    return batch.train_target_node_ids.long()
