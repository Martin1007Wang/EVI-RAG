from __future__ import annotations

from collections.abc import Sequence

import torch

from src.data.schema import RetrievalBatch
from src.models.replay import TrajectoryTrace

from .buffers import RolloutAccumulators


def resolve_edge_ptr(base_graph: RetrievalBatch) -> torch.Tensor:
    edge_ptr = getattr(base_graph, "edge_ptr", None)
    if isinstance(edge_ptr, torch.Tensor):
        return edge_ptr.long().to(base_graph.edge_index.device)
    edge_batch = getattr(base_graph, "edge_batch", None)
    if not isinstance(edge_batch, torch.Tensor):
        raise ValueError(
            "RetrievalBatch.edge_ptr or edge_batch is required for replay traces."
        )
    num_graphs = int(base_graph.ptr.numel()) - 1
    edge_counts = torch.bincount(edge_batch.long(), minlength=num_graphs)
    resolved = torch.zeros(num_graphs + 1, dtype=torch.long, device=edge_batch.device)
    torch.cumsum(edge_counts, dim=0, out=resolved[1:])
    return resolved


def resolve_batch_sample_ids(base_graph: RetrievalBatch) -> list[str] | None:
    raw = getattr(base_graph, "sample_id", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        sample_ids = [raw]
    elif isinstance(raw, (list, tuple)):
        sample_ids = [str(value) for value in raw]
    else:
        raise TypeError(
            "RetrievalBatch.sample_id must be a string or sequence of strings, "
            f"got {type(raw)!r}."
        )
    num_graphs = int(base_graph.ptr.numel()) - 1
    if len(sample_ids) != num_graphs:
        raise ValueError(
            f"RetrievalBatch.sample_id length {len(sample_ids)} != num_graphs {num_graphs}."
        )
    return sample_ids


def validate_traces(
    traces: Sequence[TrajectoryTrace],
    *,
    batch_size: int,
) -> tuple[TrajectoryTrace, ...]:
    if len(traces) != batch_size:
        raise ValueError(
            f"Forced trace count {len(traces)} != batch size {batch_size}."
        )
    return tuple(traces)


def build_trajectory_traces(
    *,
    base_graph: RetrievalBatch,
    batch_size: int,
    acc: RolloutAccumulators,
    validated_traces: tuple[TrajectoryTrace, ...] | None,
    recorded_edge_traces: list[list[int]] | None,
    sample_ids: list[str] | None,
    teacher_action_counts: torch.Tensor | None,
) -> tuple[TrajectoryTrace, ...] | None:
    if validated_traces is not None:
        return tuple(
            TrajectoryTrace(
                sample_id=trace.sample_id,
                edge_trace_local=trace.edge_trace_local,
                traj_len=trace.traj_len,
                terminal_log_reward=float(acc.terminal_log_rewards[i].detach().cpu()),
                priority=trace.priority,
                insert_step=trace.insert_step,
                source=trace.source,
                positive_edge_hit_count=trace.positive_edge_hit_count,
                positive_prefix_hit_len=trace.positive_prefix_hit_len,
                relation_only_score_mean=trace.relation_only_score_mean,
                relation_only_score_max=trace.relation_only_score_max,
                final_score_mean=trace.final_score_mean,
                teacher_forced_action_count=trace.teacher_forced_action_count,
            )
            for i, trace in enumerate(validated_traces)
        )
    if (
        recorded_edge_traces is not None
        and sample_ids is not None
        and teacher_action_counts is not None
    ):
        positive_edge_mask = base_graph.positive_edge_mask
        return tuple(
            TrajectoryTrace(
                sample_id=sample_ids[i],
                edge_trace_local=tuple(recorded_edge_traces[i]),
                traj_len=int(acc.traj_len[i].item()),
                terminal_log_reward=float(acc.terminal_log_rewards[i].detach().cpu()),
                priority=1.0,
                insert_step=0,
                source=_trajectory_source(
                    teacher_forced_action_count=int(teacher_action_counts[i].item()),
                    traj_len=int(acc.traj_len[i].item()),
                ),
                positive_edge_hit_count=_positive_edge_hit_count(
                    positive_edge_mask=positive_edge_mask,
                    selected_edge_ids=acc.selected_edge_ids[i],
                ),
                positive_prefix_hit_len=_positive_prefix_hit_len(
                    positive_edge_mask=positive_edge_mask,
                    selected_edge_ids=acc.selected_edge_ids[i],
                ),
                relation_only_score_mean=_selected_score_mean(
                    selected_edge_ids=acc.selected_edge_ids[i],
                    selected_scores=acc.selected_relation_only_logits[i],
                ),
                relation_only_score_max=_selected_score_max(
                    selected_edge_ids=acc.selected_edge_ids[i],
                    selected_scores=acc.selected_relation_only_logits[i],
                ),
                final_score_mean=_selected_score_mean(
                    selected_edge_ids=acc.selected_edge_ids[i],
                    selected_scores=acc.selected_final_logits[i],
                ),
                teacher_forced_action_count=int(teacher_action_counts[i].item()),
            )
            for i in range(batch_size)
        )
    return None


def _trajectory_source(*, teacher_forced_action_count: int, traj_len: int) -> str:
    if teacher_forced_action_count <= 0:
        return "online"
    if teacher_forced_action_count >= traj_len:
        return "teacher"
    return "mixed"


def _selected_mask(selected_edge_ids: torch.Tensor) -> torch.Tensor:
    return selected_edge_ids.ge(0)


def _selected_score_mean(
    *,
    selected_edge_ids: torch.Tensor,
    selected_scores: torch.Tensor,
) -> float:
    selected_mask = _selected_mask(selected_edge_ids)
    if not bool(selected_mask.any().item()):
        return 0.0
    return float(selected_scores[selected_mask].float().mean().item())


def _selected_score_max(
    *,
    selected_edge_ids: torch.Tensor,
    selected_scores: torch.Tensor,
) -> float:
    selected_mask = _selected_mask(selected_edge_ids)
    if not bool(selected_mask.any().item()):
        return 0.0
    return float(selected_scores[selected_mask].float().max().item())


def _positive_edge_hit_count(
    *,
    positive_edge_mask: torch.Tensor,
    selected_edge_ids: torch.Tensor,
) -> int:
    selected_mask = _selected_mask(selected_edge_ids)
    if not bool(selected_mask.any().item()):
        return 0
    selected_edges = selected_edge_ids[selected_mask].long()
    return int(positive_edge_mask.index_select(0, selected_edges).sum().item())


def _positive_prefix_hit_len(
    *,
    positive_edge_mask: torch.Tensor,
    selected_edge_ids: torch.Tensor,
) -> int:
    prefix = 0
    for edge_id in selected_edge_ids.tolist():
        if edge_id < 0:
            continue
        if not bool(positive_edge_mask[int(edge_id)].item()):
            break
        prefix += 1
    return prefix


__all__ = [
    "build_trajectory_traces",
    "resolve_batch_sample_ids",
    "resolve_edge_ptr",
    "validate_traces",
]
