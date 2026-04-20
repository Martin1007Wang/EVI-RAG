from __future__ import annotations

import torch
from torch_scatter import scatter_max, scatter_sum

from src.data.schema import RetrievalBatch
from src.models.policy import CandidateEdges
from src.models.replay import TrajectoryTrace
from src.models.state import State
from src.models.teacher_guidance import TeacherGuidance


def scatter_log_softmax(
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    max_logits, _ = scatter_max(logits, batch_idx, dim=0, dim_size=num_segments)
    shifted = logits - max_logits[batch_idx]
    sum_exp = scatter_sum(shifted.exp(), batch_idx, dim=0, dim_size=num_segments)
    eps = torch.finfo(logits.dtype).eps
    log_z = max_logits + sum_exp.clamp_min(eps).log()
    return shifted - (log_z - max_logits)[batch_idx]


def segmented_gumbel_sample(
    logits: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    eps = torch.finfo(logits.dtype).tiny
    u = torch.rand_like(logits.detach()).clamp(min=eps)
    noisy = logits.detach() / temperature - torch.log(-torch.log(u))
    _, sampled_idx = scatter_max(noisy, batch_idx, dim=0, dim_size=num_segments)
    log_probs = scatter_log_softmax(logits, batch_idx, num_segments)
    return sampled_idx, log_probs[sampled_idx]


def resolve_forced_expand_choices(
    *,
    traces: tuple[TrajectoryTrace, ...],
    step: int,
    expand_graph_ids: torch.Tensor,
    edge_ptr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_ptr.device
    local_ids = torch.tensor(
        [traces[int(graph_id)].edge_trace_local[step] for graph_id in expand_graph_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    starts = edge_ptr.index_select(0, expand_graph_ids)
    ends = edge_ptr.index_select(0, expand_graph_ids + 1)
    chosen = starts + local_ids
    invalid = local_ids.lt(0) | chosen.ge(ends)
    if invalid.any():
        bad = expand_graph_ids.index_select(
            0, torch.nonzero(invalid, as_tuple=False).view(-1)
        )
        raise RuntimeError(
            f"Forced replay out-of-range edge ids for graph ids {bad.tolist()}."
        )
    return chosen, local_ids


def lookup_forced_edge_log_probs(
    *,
    candidates: CandidateEdges,
    expand_graph_ids: torch.Tensor,
    chosen_edges: torch.Tensor,
    compact_batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del compact_batch_idx
    graph_mask = torch.zeros(
        (
            int(candidates.batch_index.max().item()) + 1 if len(candidates) > 0 else 0,
        ),
        dtype=torch.bool,
        device=candidates.batch_index.device,
    )
    if graph_mask.numel() > 0:
        graph_mask[expand_graph_ids] = True
    candidate_mask = graph_mask[candidates.batch_index]
    forced_candidates = candidates.filter(candidate_mask)
    remap = torch.empty(
        graph_mask.numel(), dtype=torch.long, device=graph_mask.device
    )
    remap[expand_graph_ids] = torch.arange(
        expand_graph_ids.numel(), dtype=torch.long, device=graph_mask.device
    )
    compact_idx = remap[forced_candidates.batch_index]
    log_probs = scatter_log_softmax(
        forced_candidates.expand_logits,
        compact_idx,
        num_segments=expand_graph_ids.numel(),
    )
    positions: list[int] = []
    for graph_id, edge_id in zip(expand_graph_ids.tolist(), chosen_edges.tolist()):
        mask = forced_candidates.batch_index.eq(int(graph_id)) & forced_candidates.edge_ids.eq(
            int(edge_id)
        )
        matched = torch.nonzero(mask, as_tuple=False).view(-1)
        if matched.numel() != 1:
            raise RuntimeError(
                "Forced replay selected invalid candidate: "
                f"graph_id={graph_id}, edge_id={edge_id}."
            )
        positions.append(int(matched.item()))
    pos_tensor = torch.tensor(
        positions, dtype=torch.long, device=chosen_edges.device
    )
    original_positions = torch.nonzero(candidate_mask, as_tuple=False).view(-1).index_select(
        0, pos_tensor
    )
    return log_probs.index_select(0, pos_tensor), original_positions


class ActionSampler:
    """Switch among online sampling, teacher-guided mixing, and replay."""

    def __init__(
        self,
        *,
        forced_traces: tuple[TrajectoryTrace, ...] | None,
        teacher_guidance: TeacherGuidance | None,
        teacher_force_prob: float,
        edge_ptr: torch.Tensor,
        batch_size: int,
        device: torch.device,
        max_steps: int,
    ) -> None:
        self._traces = forced_traces
        self._teacher_guidance = teacher_guidance
        self._teacher_force_prob = float(teacher_force_prob)
        self._edge_ptr = edge_ptr
        self._batch_size = batch_size
        self._device = device
        self._max_steps = int(max_steps)
        self._teacher_action_counts = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
        self._current_teacher_expand_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )

    @property
    def is_replay(self) -> bool:
        return self._traces is not None

    def teacher_action_counts(self) -> torch.Tensor:
        return self._teacher_action_counts.detach().clone()

    def sample_action_types(
        self,
        behavior_logits: torch.Tensor,
        target_logits: torch.Tensor,
        step_mask: torch.Tensor,
        t: int,
        *,
        base_graph: RetrievalBatch,
        state: State,
        candidates: CandidateEdges,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._current_teacher_expand_mask.zero_()
        if self.is_replay:
            action_type = torch.ones(
                self._batch_size, dtype=torch.long, device=self._device
            )
            assert self._traces is not None
            for graph_id in torch.nonzero(step_mask, as_tuple=False).view(-1).tolist():
                expand_count = len(self._traces[graph_id].edge_trace_local)
                if t < expand_count:
                    action_type[graph_id] = 0
                elif t == expand_count:
                    action_type[graph_id] = 1
                else:
                    raise RuntimeError(
                        "Forced replay exceeded stored trajectory for "
                        f"graph_id={graph_id}."
                    )
            type_log_prob = torch.distributions.Categorical(logits=target_logits).log_prob(
                action_type
            )
            return action_type, type_log_prob

        action_type = torch.distributions.Categorical(logits=behavior_logits.detach()).sample()
        teacher_stop_mask = torch.zeros(
            self._batch_size, dtype=torch.bool, device=self._device
        )

        if self._teacher_guidance is not None and self._teacher_force_prob > 0.0:
            remaining_expand_budget = self._max_steps - (int(t) + 1)
            teacher_expandable = self._teacher_guidance.graph_has_teacher_expand(
                base_graph=base_graph,
                state=state,
                candidates=candidates,
                remaining_expand_budget=remaining_expand_budget,
                num_graphs=self._batch_size,
            )
            teacher_stoppable = self._teacher_guidance.graph_has_terminal_target(
                base_graph=base_graph,
                state=state,
                num_graphs=self._batch_size,
            )

            active_graph_ids = torch.nonzero(step_mask, as_tuple=False).view(-1)
            if active_graph_ids.numel() > 0:
                drawn_mask = torch.zeros(
                    self._batch_size, dtype=torch.bool, device=self._device
                )
                draws = torch.rand(active_graph_ids.numel(), device=self._device).lt(
                    self._teacher_force_prob
                )
                if bool(draws.any().item()):
                    drawn_graph_ids = active_graph_ids.index_select(
                        0, torch.nonzero(draws, as_tuple=False).view(-1)
                    )
                    drawn_mask[drawn_graph_ids] = True
                    teacher_stop_mask = step_mask & drawn_mask & teacher_stoppable
                    self._current_teacher_expand_mask = (
                        step_mask & drawn_mask & ~teacher_stop_mask & teacher_expandable
                    )
                    unresolved = drawn_mask & ~self._current_teacher_expand_mask & ~teacher_stop_mask
                    if unresolved.any() and not self._teacher_guidance.fallback_to_policy:
                        bad = torch.nonzero(unresolved, as_tuple=False).view(-1)
                        raise RuntimeError(
                            "Teacher guidance could not provide an action for graph ids "
                            f"{bad.tolist()}."
                        )
                    action_type[self._current_teacher_expand_mask] = 0
                    action_type[teacher_stop_mask] = 1
                    teacher_controlled = self._current_teacher_expand_mask | teacher_stop_mask
                    self._teacher_action_counts[teacher_controlled] += 1

        type_log_prob = torch.distributions.Categorical(logits=target_logits).log_prob(
            action_type
        )
        invalid = step_mask & ~torch.isfinite(type_log_prob)
        if invalid.any():
            bad = torch.nonzero(invalid, as_tuple=False).view(-1)
            raise RuntimeError(
                f"Invalid action type log_prob for graph ids {bad.tolist()}."
            )
        return action_type, type_log_prob

    def sample_expand_edge(
        self,
        candidates: CandidateEdges,
        expand_graph_ids: torch.Tensor,
        compact_batch_idx: torch.Tensor,
        temperature: float,
        t: int,
        *,
        base_graph: RetrievalBatch,
        state: State,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_replay:
            assert self._traces is not None
            chosen_edges, chosen_local_ids = resolve_forced_expand_choices(
                traces=self._traces,
                step=t,
                expand_graph_ids=expand_graph_ids,
                edge_ptr=self._edge_ptr,
            )
            edge_log_prob, chosen_positions = lookup_forced_edge_log_probs(
                candidates=candidates,
                expand_graph_ids=expand_graph_ids,
                chosen_edges=chosen_edges,
                compact_batch_idx=compact_batch_idx,
            )
            return chosen_edges, chosen_local_ids, edge_log_prob, chosen_positions

        num_expand = int(expand_graph_ids.numel())
        chosen_edges = torch.empty(num_expand, dtype=torch.long, device=self._device)
        chosen_local_ids = torch.empty(num_expand, dtype=torch.long, device=self._device)
        edge_log_prob = torch.empty(
            num_expand, dtype=candidates.expand_logits.dtype, device=self._device
        )
        chosen_positions = torch.empty(num_expand, dtype=torch.long, device=self._device)

        teacher_positions: list[int] = []
        online_positions: list[int] = []
        for pos, graph_id in enumerate(expand_graph_ids.tolist()):
            if bool(self._current_teacher_expand_mask[int(graph_id)].item()):
                teacher_positions.append(pos)
            else:
                online_positions.append(pos)

        if online_positions:
            online_graph_ids = expand_graph_ids.index_select(
                0,
                torch.tensor(online_positions, dtype=torch.long, device=self._device),
            )
            online_graph_mask = torch.zeros(
                (self._batch_size,), dtype=torch.bool, device=self._device
            )
            online_graph_mask[online_graph_ids] = True
            online_candidate_mask = online_graph_mask[candidates.batch_index]
            online_candidates = candidates.filter(online_candidate_mask)
            online_remap = torch.empty(
                self._batch_size, dtype=torch.long, device=self._device
            )
            online_remap[online_graph_ids] = torch.arange(
                online_graph_ids.numel(), dtype=torch.long, device=self._device
            )
            online_compact_batch_idx = online_remap[online_candidates.batch_index]
            sampled_local_idx, online_edge_log_prob = segmented_gumbel_sample(
                logits=online_candidates.expand_logits,
                batch_idx=online_compact_batch_idx,
                num_segments=online_graph_ids.numel(),
                temperature=temperature,
            )
            sampled_edges = online_candidates.edge_ids[sampled_local_idx]
            sampled_local_ids = sampled_edges - self._edge_ptr.index_select(
                0, online_graph_ids
            )
            sampled_positions = torch.nonzero(
                online_candidate_mask, as_tuple=False
            ).view(-1)[sampled_local_idx]
            index_tensor = torch.tensor(
                online_positions, dtype=torch.long, device=self._device
            )
            chosen_edges[index_tensor] = sampled_edges
            chosen_local_ids[index_tensor] = sampled_local_ids
            edge_log_prob[index_tensor] = online_edge_log_prob
            chosen_positions[index_tensor] = sampled_positions

        if teacher_positions:
            if self._teacher_guidance is None:
                raise RuntimeError("Teacher positions requested without teacher guidance.")
            teacher_graph_ids = expand_graph_ids.index_select(
                0,
                torch.tensor(teacher_positions, dtype=torch.long, device=self._device),
            )
            remaining_expand_budget = self._max_steps - (int(t) + 1)
            valid_mask, teacher_scores = self._teacher_guidance.candidate_scores(
                base_graph=base_graph,
                state=state,
                candidates=candidates,
                remaining_expand_budget=remaining_expand_budget,
            )
            teacher_edge_positions: list[int] = []
            for graph_id in teacher_graph_ids.tolist():
                graph_mask = candidates.batch_index.eq(int(graph_id)) & valid_mask
                matched = torch.nonzero(graph_mask, as_tuple=False).view(-1)
                if matched.numel() == 0:
                    raise RuntimeError(
                        "Teacher guidance selected expand but found no legal teacher "
                        f"candidate for graph_id={graph_id}."
                    )
                graph_scores = teacher_scores.index_select(0, matched).clamp_min(
                    torch.finfo(teacher_scores.dtype).eps
                )
                sampled_pos = matched[
                    torch.multinomial(graph_scores, num_samples=1, replacement=True)[0]
                ]
                teacher_edge_positions.append(int(sampled_pos.item()))

            teacher_pos_tensor = torch.tensor(
                teacher_edge_positions, dtype=torch.long, device=self._device
            )
            teacher_edges = candidates.edge_ids.index_select(0, teacher_pos_tensor)
            teacher_local_ids = teacher_edges - self._edge_ptr.index_select(
                0, teacher_graph_ids
            )
            teacher_edge_log_prob, teacher_positions_in_candidates = lookup_forced_edge_log_probs(
                candidates=candidates,
                expand_graph_ids=teacher_graph_ids,
                chosen_edges=teacher_edges,
                compact_batch_idx=compact_batch_idx,
            )
            index_tensor = torch.tensor(
                teacher_positions, dtype=torch.long, device=self._device
            )
            chosen_edges[index_tensor] = teacher_edges
            chosen_local_ids[index_tensor] = teacher_local_ids
            edge_log_prob[index_tensor] = teacher_edge_log_prob
            chosen_positions[index_tensor] = teacher_positions_in_candidates

        return chosen_edges, chosen_local_ids, edge_log_prob, chosen_positions


__all__ = [
    "ActionSampler",
    "lookup_forced_edge_log_probs",
    "resolve_forced_expand_choices",
    "scatter_log_softmax",
    "segmented_gumbel_sample",
]
