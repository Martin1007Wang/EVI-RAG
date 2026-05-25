from __future__ import annotations

from dataclasses import dataclass

import torch

from src.data.schema import RetrievalBatch
from src.weaver.context import GraphContext, TargetContext
from src.weaver.rollout.trajectory import (
    BUDGET,
    POLICY_STOP,
    SRC_REPLAY,
    TrajectoryBatch,
)

Tensor = torch.Tensor


@dataclass(frozen=True, slots=True)
class ReplaySource:
    """
    Off-policy replay source.

    Replay candidates are precomputed in graph-local edge coordinates.
    This class converts them to batch-physical edge coordinates before they
    enter rollout/loss code.

    Required batch fields:
    - replay_trajectory_lengths: [M]
    - replay_trajectory_lengths_batch: [M], graph id per replay trajectory
    - replay_trajectory_edge_ids: [sum(lengths)], graph-local edge ids

    Output contract:
    - TrajectoryBatch.edge_ids contains batch-physical KG edge ids.
    """

    budget: int

    @torch.no_grad()
    def sample(
        self,
        *,
        batch: RetrievalBatch,
        graph: GraphContext,
        target: TargetContext,
        trajectories_per_graph: int,
    ) -> TrajectoryBatch:
        return precomputed_replay_trajectory_batch(
            batch=batch,
            graph=graph,
            target=target,
            budget=int(self.budget),
            trajectories_per_graph=int(trajectories_per_graph),
        )


@dataclass(frozen=True, slots=True)
class PrecomputedReplayFields:
    """
    Raw replay fields from RetrievalBatch.

    Coordinates:
    - graph_ids are batch-local graph ids.
    - lengths are replay trajectory lengths.
    - flat_local_edge_ids are graph-local edge ids, concatenated by trajectory.
    """

    lengths: Tensor  # [M]
    graph_ids: Tensor  # [M]
    flat_local_edge_ids: Tensor  # [sum(lengths)]


def precomputed_replay_trajectory_batch(
    *,
    batch: RetrievalBatch,
    graph: GraphContext,
    target: TargetContext,
    budget: int,
    trajectories_per_graph: int,
) -> TrajectoryBatch:
    """
    Convert precomputed replay candidates into TrajectoryBatch.

    The only coordinate conversion performed here is:

        graph-local replay edge id
        -> batch-physical edge id

    using GraphContext.edge_ptr.

    Replay selection rule:
    - ignore graphs with no reachable targets;
    - ignore trajectories longer than budget;
    - keep at most trajectories_per_graph candidates per graph;
    - preserve candidate order within each graph.
    """

    budget = int(budget)
    trajectories_per_graph = int(trajectories_per_graph)

    if budget < 0:
        raise ValueError("budget must be nonnegative.")

    if trajectories_per_graph <= 0:
        return TrajectoryBatch.empty(
            device=graph.device,
            budget=budget,
        )

    fields = read_precomputed_replay_fields(
        batch=batch,
        device=graph.device,
    )

    if fields is None:
        return TrajectoryBatch.empty(
            device=graph.device,
            budget=budget,
        )

    validate_precomputed_replay_fields(
        fields=fields,
        graph=graph,
    )

    candidate_ids = select_replay_candidates(
        fields=fields,
        target=target,
        num_graphs=int(graph.num_graphs),
        budget=budget,
        trajectories_per_graph=trajectories_per_graph,
    )

    if int(candidate_ids.numel()) == 0:
        return TrajectoryBatch.empty(
            device=graph.device,
            budget=budget,
        )

    return build_replay_trajectory_batch(
        fields=fields,
        candidate_ids=candidate_ids,
        graph=graph,
        budget=budget,
    )


def read_precomputed_replay_fields(
    *,
    batch: RetrievalBatch,
    device: torch.device,
) -> PrecomputedReplayFields | None:
    lengths = getattr(batch, "replay_trajectory_lengths", None)
    graph_ids = getattr(batch, "replay_trajectory_lengths_batch", None)
    edge_ids = getattr(batch, "replay_trajectory_edge_ids", None)

    if lengths is None and graph_ids is None and edge_ids is None:
        return None

    if lengths is None or graph_ids is None or edge_ids is None:
        raise ValueError(
            "Incomplete precomputed replay fields. Expected all of: "
            "replay_trajectory_lengths, replay_trajectory_lengths_batch, "
            "replay_trajectory_edge_ids."
        )

    return PrecomputedReplayFields(
        lengths=lengths.to(device=device, dtype=torch.long).view(-1),
        graph_ids=graph_ids.to(device=device, dtype=torch.long).view(-1),
        flat_local_edge_ids=edge_ids.to(device=device, dtype=torch.long).view(-1),
    )


def validate_precomputed_replay_fields(
    *,
    fields: PrecomputedReplayFields,
    graph: GraphContext,
) -> None:
    if fields.lengths.ndim != 1:
        raise ValueError(f"replay_trajectory_lengths must have shape [M], " f"got {tuple(fields.lengths.shape)}.")

    if fields.graph_ids.ndim != 1:
        raise ValueError(f"replay_trajectory_lengths_batch must have shape [M], " f"got {tuple(fields.graph_ids.shape)}.")

    if int(fields.lengths.numel()) != int(fields.graph_ids.numel()):
        raise ValueError("replay_trajectory_lengths and replay_trajectory_lengths_batch " "must have the same length.")

    if bool(fields.lengths.lt(0).any()):
        raise ValueError("replay_trajectory_lengths must be nonnegative.")

    _check_id_range(
        ids=fields.graph_ids,
        upper=int(graph.num_graphs),
        name="replay trajectory graph ids",
    )

    expected_edge_count = int(fields.lengths.sum().item())
    actual_edge_count = int(fields.flat_local_edge_ids.numel())

    if expected_edge_count != actual_edge_count:
        raise ValueError(
            "replay_trajectory_edge_ids length does not match " "sum(replay_trajectory_lengths): " f"{actual_edge_count} vs {expected_edge_count}."
        )


def select_replay_candidates(
    *,
    fields: PrecomputedReplayFields,
    target: TargetContext,
    num_graphs: int,
    budget: int,
    trajectories_per_graph: int,
) -> Tensor:
    """
    Select at most K valid replay trajectories per graph.

    Valid:
    - graph has at least one reachable target;
    - trajectory length <= budget.
    """

    num_candidates = int(fields.lengths.numel())

    if num_candidates == 0:
        return _empty_long(fields.lengths.device)

    valid_graph = target.target_count_by_graph.to(
        device=fields.lengths.device,
        dtype=torch.long,
    ).gt(0)

    eligible = fields.lengths.le(int(budget)) & valid_graph.index_select(0, fields.graph_ids)

    return first_k_per_graph(
        graph_ids=fields.graph_ids,
        eligible=eligible,
        num_graphs=int(num_graphs),
        k=int(trajectories_per_graph),
    )


def build_replay_trajectory_batch(
    *,
    fields: PrecomputedReplayFields,
    candidate_ids: Tensor,
    graph: GraphContext,
    budget: int,
) -> TrajectoryBatch:
    """
    Build a TrajectoryBatch from selected precomputed replay candidates.

    Output edge_ids are batch-physical edge ids.
    """

    budget = int(budget)
    device = graph.device

    candidate_ids = candidate_ids.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    num_trajectories = int(candidate_ids.numel())

    if num_trajectories == 0:
        return TrajectoryBatch.empty(
            device=device,
            budget=budget,
        )

    graph_ids = fields.graph_ids.index_select(0, candidate_ids)
    edge_count = fields.lengths.index_select(0, candidate_ids)

    edge_ids = torch.full(
        (num_trajectories, budget),
        -1,
        dtype=torch.long,
        device=device,
    )

    if budget > 0 and bool(edge_count.gt(0).any()):
        trajectory_ptr = build_ptr_from_lengths(
            lengths=fields.lengths,
            device=device,
        )

        row_ids = _repeat_interleave(
            torch.arange(
                num_trajectories,
                dtype=torch.long,
                device=device,
            ),
            edge_count,
        )

        local_pos = _segment_arange(edge_count)

        source_starts = trajectory_ptr.index_select(0, candidate_ids)
        flat_pos = (
            _repeat_interleave(
                source_starts,
                edge_count,
            )
            + local_pos
        )

        local_edge_ids = fields.flat_local_edge_ids.index_select(0, flat_pos)

        edge_graph_ids = graph_ids.index_select(0, row_ids)

        physical_edge_ids = local_edge_ids_to_physical(
            graph_ids=edge_graph_ids,
            local_edge_ids=local_edge_ids,
            graph=graph,
        )

        edge_ids[row_ids, local_pos] = physical_edge_ids

    edge_logp = torch.zeros(
        (num_trajectories, budget),
        dtype=torch.float32,
        device=device,
    )

    stop_logp = torch.zeros(
        num_trajectories,
        dtype=torch.float32,
        device=device,
    )

    stop_reason = torch.where(
        edge_count.ge(budget),
        torch.full_like(edge_count, int(BUDGET)),
        torch.full_like(edge_count, int(POLICY_STOP)),
    )

    source = torch.full(
        (num_trajectories,),
        int(SRC_REPLAY),
        dtype=torch.long,
        device=device,
    )

    return TrajectoryBatch(
        graph_ids=graph_ids,
        edge_ids=edge_ids,
        edge_logp=edge_logp,
        edge_count=edge_count,
        stop_reason=stop_reason,
        stop_logp=stop_logp,
        source=source,
    )


def local_edge_ids_to_physical(
    *,
    graph_ids: Tensor,
    local_edge_ids: Tensor,
    graph: GraphContext,
) -> Tensor:
    """
    Convert graph-local edge ids to batch-physical edge ids.

    Contract:
    - graph_ids[k] is the graph id for local_edge_ids[k].
    - local_edge_ids[k] is local inside that graph's edge range.
    - output[k] is a physical edge id, i.e. edge_index column id.

    Requires:
    - graph.edge_ptr exists and maps graph id -> contiguous physical edge range.
    - graph.edge_to_graph validates the converted ids.
    """

    graph_ids = graph_ids.to(
        device=graph.device,
        dtype=torch.long,
    ).view(-1)

    local_edge_ids = local_edge_ids.to(
        device=graph.device,
        dtype=torch.long,
    ).view(-1)

    if int(graph_ids.numel()) != int(local_edge_ids.numel()):
        raise ValueError("graph_ids and local_edge_ids must have the same length: " f"{graph_ids.numel()} vs {local_edge_ids.numel()}.")

    if int(local_edge_ids.numel()) == 0:
        return _empty_long(graph.device)

    _check_id_range(
        ids=graph_ids,
        upper=int(graph.num_graphs),
        name="graph_ids",
    )

    starts = graph.edge_ptr.index_select(0, graph_ids)
    ends = graph.edge_ptr.index_select(0, graph_ids + 1)
    edge_count = ends - starts

    outside = local_edge_ids.lt(0) | local_edge_ids.ge(edge_count)

    if bool(outside.any()):
        first = int(outside.nonzero(as_tuple=False).flatten()[0].item())
        raise ValueError(
            "Replay local edge id is outside graph edge range: "
            f"graph_id={int(graph_ids[first].item())}, "
            f"local_edge_id={int(local_edge_ids[first].item())}, "
            f"graph_edge_count={int(edge_count[first].item())}."
        )

    physical_edge_ids = starts + local_edge_ids

    _check_id_range(
        ids=physical_edge_ids,
        upper=int(graph.num_edges),
        name="physical replay edge ids",
    )

    actual_graph_ids = graph.edge_to_graph.index_select(0, physical_edge_ids)
    mismatch = actual_graph_ids.ne(graph_ids)

    if bool(mismatch.any()):
        first = int(mismatch.nonzero(as_tuple=False).flatten()[0].item())
        raise ValueError(
            "Converted replay edge id belongs to the wrong graph. "
            "This means graph.edge_ptr is inconsistent with graph.edge_to_graph "
            "or replay edge ids are not graph-local. "
            f"graph_id={int(graph_ids[first].item())}, "
            f"local_edge_id={int(local_edge_ids[first].item())}, "
            f"physical_edge_id={int(physical_edge_ids[first].item())}, "
            f"actual_graph_id={int(actual_graph_ids[first].item())}."
        )

    return physical_edge_ids


def first_k_per_graph(
    *,
    graph_ids: Tensor,
    eligible: Tensor,
    num_graphs: int,
    k: int,
) -> Tensor:
    """
    Return candidate ids for the first k eligible rows per graph.

    Preserves original candidate ordering in the returned ids.
    """

    if k <= 0:
        return _empty_long(graph_ids.device)

    graph_ids = graph_ids.to(dtype=torch.long).view(-1)
    eligible = eligible.to(dtype=torch.bool).view(-1)

    if int(graph_ids.numel()) != int(eligible.numel()):
        raise ValueError("graph_ids and eligible must have the same length: " f"{graph_ids.numel()} vs {eligible.numel()}.")

    candidate_ids = eligible.nonzero(as_tuple=False).flatten()

    if int(candidate_ids.numel()) == 0:
        return candidate_ids

    candidate_graph_ids = graph_ids.index_select(0, candidate_ids)

    _check_id_range(
        ids=candidate_graph_ids,
        upper=int(num_graphs),
        name="eligible replay graph ids",
    )

    num_rows = int(graph_ids.numel())

    sort_key = candidate_graph_ids * num_rows + candidate_ids
    order = torch.argsort(sort_key)

    sorted_candidate_ids = candidate_ids.index_select(0, order)
    sorted_graph_ids = candidate_graph_ids.index_select(0, order)

    counts = torch.bincount(
        sorted_graph_ids,
        minlength=int(num_graphs),
    ).to(dtype=torch.long)

    group_starts = torch.cumsum(counts, dim=0) - counts

    rank = torch.arange(
        int(sorted_candidate_ids.numel()),
        dtype=torch.long,
        device=graph_ids.device,
    ) - _repeat_interleave(
        group_starts,
        counts,
    )

    selected = sorted_candidate_ids[rank.lt(int(k))]

    return torch.sort(selected).values


def build_ptr_from_lengths(
    *,
    lengths: Tensor,
    device: torch.device,
) -> Tensor:
    lengths = lengths.to(
        device=device,
        dtype=torch.long,
    ).view(-1)

    ptr = torch.empty(
        int(lengths.numel()) + 1,
        dtype=torch.long,
        device=device,
    )
    ptr[0] = 0
    ptr[1:] = torch.cumsum(lengths, dim=0)

    return ptr


def has_precomputed_replay_candidates(batch: RetrievalBatch) -> bool:
    return (
        read_precomputed_replay_fields(
            batch=batch,
            device=batch.edge_index.device,
        )
        is not None
    )


def _segment_arange(lengths: Tensor) -> Tensor:
    """
    For lengths [l0, l1, ...], return:

        [0, ..., l0 - 1, 0, ..., l1 - 1, ...]
    """

    lengths = lengths.to(dtype=torch.long).view(-1)

    if int(lengths.numel()) == 0:
        return _empty_long(lengths.device)

    total = int(lengths.sum().item())

    if total == 0:
        return _empty_long(lengths.device)

    starts = torch.cumsum(lengths, dim=0) - lengths

    return torch.arange(
        total,
        dtype=torch.long,
        device=lengths.device,
    ) - _repeat_interleave(
        starts,
        lengths,
    )


def _repeat_interleave(
    values: Tensor,
    repeats: Tensor,
) -> Tensor:
    return torch.repeat_interleave(
        values,
        repeats.to(dtype=torch.long),
    )


def _check_id_range(
    *,
    ids: Tensor,
    upper: int,
    name: str,
) -> None:
    ids = ids.to(dtype=torch.long)

    if int(ids.numel()) == 0:
        return

    min_id = int(ids.min().item())
    max_id = int(ids.max().item())

    if min_id < 0 or max_id >= int(upper):
        raise ValueError(f"{name} contains ids outside [0, {upper}): " f"min={min_id}, max={max_id}.")


def _empty_long(device: torch.device) -> Tensor:
    return torch.empty(
        0,
        dtype=torch.long,
        device=device,
    )


__all__ = [
    "PrecomputedReplayFields",
    "ReplaySource",
    "build_replay_trajectory_batch",
    "first_k_per_graph",
    "has_precomputed_replay_candidates",
    "local_edge_ids_to_physical",
    "precomputed_replay_trajectory_batch",
    "read_precomputed_replay_fields",
    "select_replay_candidates",
]
