from __future__ import annotations

from typing import Any, Optional

import torch

from src.models.components.graph_ops import gather_outgoing_edges, segment_max

from .constants import _NEG_ONE
from .types import _BeamCandidateMatrix, _BeamCandidates, _BeamState, _PreparedBatch


class DualFlowBeamMixin:
    @staticmethod
    def _index_candidates(candidates: _BeamCandidates, index: torch.Tensor) -> _BeamCandidates:
        return _BeamCandidates(
            cand_scores=candidates.cand_scores.index_select(0, index),
            cand_nodes=candidates.cand_nodes.index_select(0, index),
            cand_graph=candidates.cand_graph.index_select(0, index),
            cand_src_beam=candidates.cand_src_beam.index_select(0, index),
            cand_edge_id=candidates.cand_edge_id.index_select(0, index),
            cand_is_edge=candidates.cand_is_edge.index_select(0, index),
            cand_done=candidates.cand_done.index_select(0, index),
        )

    @staticmethod
    def _merge_indices_by_graph(
        *,
        cand_graph_edge: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        counts_edge = torch.bincount(cand_graph_edge, minlength=num_graphs)
        counts_stay = torch.bincount(cand_graph_stay, minlength=num_graphs)
        total_counts = counts_edge + counts_stay
        total = int(total_counts.sum().item())
        offsets = total_counts.cumsum(0) - total_counts

        total_edges = cand_graph_edge.numel()
        start_edge = (counts_edge.cumsum(0) - counts_edge).index_select(0, cand_graph_edge)
        pos_edge = torch.arange(total_edges, device=cand_graph_edge.device) - start_edge
        idx_edge = offsets.index_select(0, cand_graph_edge) + pos_edge

        total_stay = cand_graph_stay.numel()
        start_stay = (counts_stay.cumsum(0) - counts_stay).index_select(0, cand_graph_stay)
        pos_stay = torch.arange(total_stay, device=cand_graph_stay.device) - start_stay
        idx_stay = offsets.index_select(0, cand_graph_stay) + counts_edge.index_select(0, cand_graph_stay) + pos_stay
        return idx_edge, idx_stay, total

    @staticmethod
    def _scatter_merged_candidates(
        *,
        idx_edge: torch.Tensor,
        idx_stay: torch.Tensor,
        total: int,
        cand_scores_edge: torch.Tensor,
        cand_nodes_edge: torch.Tensor,
        cand_graph_edge: torch.Tensor,
        cand_src_beam_edge: torch.Tensor,
        cand_edge_id_edge: torch.Tensor,
        cand_is_edge_edge: torch.Tensor,
        cand_done_edge: torch.Tensor,
        cand_scores_stay: torch.Tensor,
        cand_nodes_stay: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        cand_src_beam_stay: torch.Tensor,
        cand_edge_id_stay: torch.Tensor,
        cand_is_edge_stay: torch.Tensor,
        cand_done_stay: torch.Tensor,
    ) -> _BeamCandidates:
        device = cand_scores_edge.device
        out_scores = torch.empty((total,), device=device, dtype=cand_scores_edge.dtype)
        out_nodes = torch.empty((total,), device=device, dtype=cand_nodes_edge.dtype)
        out_graph = torch.empty((total,), device=device, dtype=torch.long)
        out_src = torch.empty((total,), device=device, dtype=torch.long)
        out_edge_id = torch.empty((total,), device=device, dtype=cand_edge_id_edge.dtype)
        out_is_edge = torch.empty((total,), device=device, dtype=torch.bool)
        out_done = torch.empty((total,), device=device, dtype=torch.bool)

        out_scores.index_copy_(0, idx_edge, cand_scores_edge)
        out_nodes.index_copy_(0, idx_edge, cand_nodes_edge)
        out_graph.index_copy_(0, idx_edge, cand_graph_edge)
        out_src.index_copy_(0, idx_edge, cand_src_beam_edge)
        out_edge_id.index_copy_(0, idx_edge, cand_edge_id_edge)
        out_is_edge.index_copy_(0, idx_edge, cand_is_edge_edge)
        out_done.index_copy_(0, idx_edge, cand_done_edge)

        out_scores.index_copy_(0, idx_stay, cand_scores_stay)
        out_nodes.index_copy_(0, idx_stay, cand_nodes_stay)
        out_graph.index_copy_(0, idx_stay, cand_graph_stay)
        out_src.index_copy_(0, idx_stay, cand_src_beam_stay)
        out_edge_id.index_copy_(0, idx_stay, cand_edge_id_stay)
        out_is_edge.index_copy_(0, idx_stay, cand_is_edge_stay)
        out_done.index_copy_(0, idx_stay, cand_done_stay)

        return _BeamCandidates(
            cand_scores=out_scores,
            cand_nodes=out_nodes,
            cand_graph=out_graph,
            cand_src_beam=out_src,
            cand_edge_id=out_edge_id,
            cand_is_edge=out_is_edge,
            cand_done=out_done,
        )

    @staticmethod
    def _coerce_candidate_graph(cand_graph: torch.Tensor) -> torch.Tensor:
        if cand_graph.dtype != torch.long:
            return cand_graph.to(dtype=torch.long)
        return cand_graph

    @staticmethod
    def _coerce_candidates_graph(candidates: _BeamCandidates) -> _BeamCandidates:
        if candidates.cand_graph.dtype == torch.long:
            return candidates
        return _BeamCandidates(
            cand_scores=candidates.cand_scores,
            cand_nodes=candidates.cand_nodes,
            cand_graph=candidates.cand_graph.to(dtype=torch.long),
            cand_src_beam=candidates.cand_src_beam,
            cand_edge_id=candidates.cand_edge_id,
            cand_is_edge=candidates.cand_is_edge,
            cand_done=candidates.cand_done,
        )

    @staticmethod
    def _maybe_sort_candidates_by_graph(
        candidates: _BeamCandidates,
        cand_graph: torch.Tensor,
    ) -> tuple[_BeamCandidates, torch.Tensor]:
        if cand_graph.numel() > 1 and not (cand_graph[:-1] <= cand_graph[1:]).all().item():
            order = torch.argsort(cand_graph)
            candidates = DualFlowBeamMixin._index_candidates(candidates, order)
            cand_graph = candidates.cand_graph
        return candidates, cand_graph

    @staticmethod
    def _truncate_candidates_by_score(
        candidates: _BeamCandidates,
        *,
        cand_graph: torch.Tensor,
        num_graphs: int,
        cap: Optional[int],
    ) -> tuple[Optional[_BeamCandidates], torch.Tensor, torch.Tensor, int, bool]:
        counts = torch.bincount(cand_graph, minlength=num_graphs)
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        if cap is None or cap <= 0 or max_count <= cap:
            return candidates, cand_graph, counts, max_count, False

        order_score = torch.argsort(candidates.cand_scores, descending=True)
        graph_sorted = cand_graph.index_select(0, order_score)
        order_graph = torch.argsort(graph_sorted, stable=True)
        order = order_score.index_select(0, order_graph)
        candidates = DualFlowBeamMixin._index_candidates(candidates, order)
        cand_graph = candidates.cand_graph

        counts = torch.bincount(cand_graph, minlength=num_graphs)
        start = (counts.cumsum(0) - counts).index_select(0, cand_graph)
        pos = torch.arange(cand_graph.numel(), device=cand_graph.device) - start
        keep = pos < cap
        if not keep.any():
            return None, cand_graph, counts, 0, True
        keep_idx = torch.nonzero(keep, as_tuple=False).view(-1)
        candidates = DualFlowBeamMixin._index_candidates(candidates, keep_idx)
        cand_graph = candidates.cand_graph
        counts = torch.bincount(cand_graph, minlength=num_graphs)
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        return candidates, cand_graph, counts, max_count, True

    @staticmethod
    def _merge_candidates_by_graph(
        *,
        cand_scores_edge: torch.Tensor,
        cand_nodes_edge: torch.Tensor,
        cand_graph_edge: torch.Tensor,
        cand_src_beam_edge: torch.Tensor,
        cand_edge_id_edge: torch.Tensor,
        cand_is_edge_edge: torch.Tensor,
        cand_done_edge: torch.Tensor,
        cand_scores_stay: torch.Tensor,
        cand_nodes_stay: torch.Tensor,
        cand_graph_stay: torch.Tensor,
        cand_src_beam_stay: torch.Tensor,
        cand_edge_id_stay: torch.Tensor,
        cand_is_edge_stay: torch.Tensor,
        cand_done_stay: torch.Tensor,
        num_graphs: int,
    ) -> Optional[_BeamCandidates]:
        total_edges = cand_scores_edge.numel()
        total_stay = cand_scores_stay.numel()
        total = total_edges + total_stay
        if total == 0:
            return None
        if total_edges == 0:
            return _BeamCandidates(
                cand_scores=cand_scores_stay,
                cand_nodes=cand_nodes_stay,
                cand_graph=cand_graph_stay,
                cand_src_beam=cand_src_beam_stay,
                cand_edge_id=cand_edge_id_stay,
                cand_is_edge=cand_is_edge_stay,
                cand_done=cand_done_stay,
            )
        if total_stay == 0:
            return _BeamCandidates(
                cand_scores=cand_scores_edge,
                cand_nodes=cand_nodes_edge,
                cand_graph=cand_graph_edge,
                cand_src_beam=cand_src_beam_edge,
                cand_edge_id=cand_edge_id_edge,
                cand_is_edge=cand_is_edge_edge,
                cand_done=cand_done_edge,
            )
        idx_edge, idx_stay, total = DualFlowBeamMixin._merge_indices_by_graph(
            cand_graph_edge=cand_graph_edge,
            cand_graph_stay=cand_graph_stay,
            num_graphs=num_graphs,
        )
        return DualFlowBeamMixin._scatter_merged_candidates(
            idx_edge=idx_edge,
            idx_stay=idx_stay,
            total=total,
            cand_scores_edge=cand_scores_edge,
            cand_nodes_edge=cand_nodes_edge,
            cand_graph_edge=cand_graph_edge,
            cand_src_beam_edge=cand_src_beam_edge,
            cand_edge_id_edge=cand_edge_id_edge,
            cand_is_edge_edge=cand_is_edge_edge,
            cand_done_edge=cand_done_edge,
            cand_scores_stay=cand_scores_stay,
            cand_nodes_stay=cand_nodes_stay,
            cand_graph_stay=cand_graph_stay,
            cand_src_beam_stay=cand_src_beam_stay,
            cand_edge_id_stay=cand_edge_id_stay,
            cand_is_edge_stay=cand_is_edge_stay,
            cand_done_stay=cand_done_stay,
        )

    def _beam_search_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> _BeamState:
        num_graphs = int(prepared.num_graphs)
        if num_graphs <= 0:
            return _BeamState(
                beam_nodes=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_scores=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.float32),
                beam_paths=torch.zeros((0, 0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_lengths=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_done=torch.zeros((0, 0), device=prepared.node_ptr.device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                beam_context=torch.zeros((0, prepared.context_tokens.size(-1)), device=prepared.node_ptr.device, dtype=prepared.context_tokens.dtype),
                num_graphs=0,
                beam_size=0,
                max_steps=int(self.max_steps),
                neg_inf=float("-inf"),
            )
        if beam_size <= 0:
            return _BeamState(
                beam_nodes=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_scores=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.float32),
                beam_paths=torch.zeros((num_graphs, 0, int(self.max_steps)), device=prepared.node_ptr.device, dtype=torch.long),
                beam_lengths=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.long),
                beam_done=torch.zeros((num_graphs, 0), device=prepared.node_ptr.device, dtype=torch.bool),
                flat_graph_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                flat_beam_ids=torch.zeros((0,), device=prepared.node_ptr.device, dtype=torch.long),
                beam_context=prepared.context_tokens[:0],
                num_graphs=num_graphs,
                beam_size=0,
                max_steps=int(self.max_steps),
                neg_inf=float("-inf"),
            )
        state = self._init_beam_state(prepared=prepared, beam_size=beam_size, node_is_target=node_is_target)
        diverse_cfg = self._resolve_diverse_beam_cfg()
        for step in range(state.max_steps):
            candidates = self._beam_expand_candidates(
                prepared=prepared,
                state=state,
                step=step,
                node_is_target=node_is_target,
            )
            if candidates is None:
                break
            state = self._beam_update_from_candidates(
                state=state,
                candidates=candidates,
                step=step,
                diverse_cfg=diverse_cfg,
            )
        return state

    def _beam_search(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> list[list[tuple[int, float, list[int]]]]:
        state = self._beam_search_state(prepared=prepared, beam_size=beam_size, node_is_target=node_is_target)
        if state.beam_nodes.numel() == 0:
            return []
        return self._beam_finalize(state)

    def _init_beam_state(
        self,
        *,
        prepared: _PreparedBatch,
        beam_size: int,
        node_is_target: torch.Tensor,
    ) -> _BeamState:
        num_graphs = int(prepared.num_graphs)
        device = prepared.node_ptr.device
        max_steps = int(self.max_steps)
        neg_inf = float("-inf")
        start_nodes = prepared.start_nodes_fwd.to(device=device, dtype=torch.long)
        beam_nodes = torch.full((num_graphs, beam_size), _NEG_ONE, device=device, dtype=torch.long)
        beam_scores = torch.full((num_graphs, beam_size), neg_inf, device=device, dtype=torch.float32)
        beam_paths = torch.full((num_graphs, beam_size, max_steps), _NEG_ONE, device=device, dtype=torch.long)
        beam_lengths = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.long)
        valid_start = start_nodes >= 0
        beam_nodes[:, 0] = start_nodes
        beam_scores[:, 0] = torch.where(valid_start, torch.zeros_like(beam_scores[:, 0]), beam_scores[:, 0])
        start_target = node_is_target.index_select(0, start_nodes.clamp(min=0))
        beam_done = torch.zeros((num_graphs, beam_size), device=device, dtype=torch.bool)
        beam_done[:, 0] = valid_start & start_target
        flat_graph_ids = torch.arange(num_graphs, device=device).repeat_interleave(beam_size)
        flat_beam_ids = torch.arange(beam_size, device=device).repeat(num_graphs)
        beam_context = prepared.context_tokens.index_select(0, flat_graph_ids)
        return _BeamState(
            beam_nodes=beam_nodes,
            beam_scores=beam_scores,
            beam_paths=beam_paths,
            beam_lengths=beam_lengths,
            beam_done=beam_done,
            flat_graph_ids=flat_graph_ids,
            flat_beam_ids=flat_beam_ids,
            beam_context=beam_context,
            num_graphs=num_graphs,
            beam_size=beam_size,
            max_steps=max_steps,
            neg_inf=neg_inf,
        )

    def _beam_expand_candidates(
        self,
        *,
        prepared: _PreparedBatch,
        state: _BeamState,
        step: int,
        node_is_target: torch.Tensor,
    ) -> Optional[_BeamCandidates]:
        flat_nodes = state.beam_nodes.view(-1)
        flat_scores = state.beam_scores.view(-1)
        flat_done = state.beam_done.view(-1)
        flat_valid = flat_nodes >= 0
        expand_mask = flat_valid & ~flat_done
        outgoing = gather_outgoing_edges(
            curr_nodes=flat_nodes,
            edge_ids_by_head=prepared.edge_ids_by_head_fwd,
            edge_ptr_by_head=prepared.edge_ptr_by_head_fwd,
            active_mask=expand_mask,
        )
        empty_long = torch.zeros((0,), device=flat_nodes.device, dtype=torch.long)
        empty_bool = torch.zeros((0,), device=flat_nodes.device, dtype=torch.bool)
        empty_float = torch.zeros((0,), device=flat_nodes.device, dtype=torch.float32)

        if outgoing.edge_ids.numel() > 0:
            step_ids = torch.full((state.num_graphs * state.beam_size,), step, device=flat_nodes.device, dtype=torch.long)
            logits = self._compute_edge_logits(
                policy=self.policy_fwd,
                prepared=prepared,
                edge_ids=outgoing.edge_ids,
                edge_batch=outgoing.edge_batch,
                steps=step_ids + 1,
                temperature=1.0,
                context_tokens=state.beam_context,
            )
            log_denom = self._compute_log_denom(
                logits=logits, edge_batch=outgoing.edge_batch, num_graphs=state.num_graphs * state.beam_size
            )
            log_probs = logits - log_denom.index_select(0, outgoing.edge_batch)
            cand_scores_edge = flat_scores.index_select(0, outgoing.edge_batch) + log_probs
            cand_nodes_edge = prepared.edge_index[1].index_select(0, outgoing.edge_ids)
            cand_graph_edge = state.flat_graph_ids.index_select(0, outgoing.edge_batch)
            cand_src_beam_edge = state.flat_beam_ids.index_select(0, outgoing.edge_batch)
            cand_edge_id_edge = outgoing.edge_ids
            cand_is_edge_edge = torch.ones_like(cand_scores_edge, dtype=torch.bool)
            cand_done_edge = node_is_target.index_select(0, cand_nodes_edge)
        else:
            cand_scores_edge = empty_float
            cand_nodes_edge = empty_long
            cand_graph_edge = empty_long
            cand_src_beam_edge = empty_long
            cand_edge_id_edge = empty_long
            cand_is_edge_edge = empty_bool
            cand_done_edge = empty_bool

        stay_mask = flat_valid & (flat_done | ~outgoing.has_edge)
        cand_scores_stay = flat_scores[stay_mask]
        cand_nodes_stay = flat_nodes[stay_mask]
        cand_graph_stay = state.flat_graph_ids[stay_mask]
        cand_src_beam_stay = state.flat_beam_ids[stay_mask]
        cand_edge_id_stay = torch.full_like(cand_nodes_stay, _NEG_ONE)
        cand_is_edge_stay = torch.zeros_like(cand_scores_stay, dtype=torch.bool)
        cand_done_stay = torch.ones_like(cand_scores_stay, dtype=torch.bool)

        if cand_scores_edge.numel() + cand_scores_stay.numel() == 0:
            return None
        return self._merge_candidates_by_graph(
            cand_scores_edge=cand_scores_edge,
            cand_nodes_edge=cand_nodes_edge,
            cand_graph_edge=cand_graph_edge,
            cand_src_beam_edge=cand_src_beam_edge,
            cand_edge_id_edge=cand_edge_id_edge,
            cand_is_edge_edge=cand_is_edge_edge,
            cand_done_edge=cand_done_edge,
            cand_scores_stay=cand_scores_stay,
            cand_nodes_stay=cand_nodes_stay,
            cand_graph_stay=cand_graph_stay,
            cand_src_beam_stay=cand_src_beam_stay,
            cand_edge_id_stay=cand_edge_id_stay,
            cand_is_edge_stay=cand_is_edge_stay,
            cand_done_stay=cand_done_stay,
            num_graphs=state.num_graphs,
        )

    @staticmethod
    def _build_candidate_matrix(
        candidates: _BeamCandidates,
        *,
        num_graphs: int,
        neg_inf: float,
        max_candidates_per_graph: Optional[int] = None,
    ) -> Optional[_BeamCandidateMatrix]:
        cand_graph = candidates.cand_graph
        if cand_graph.numel() == 0:
            return None
        candidates = DualFlowBeamMixin._coerce_candidates_graph(candidates)
        cand_graph = candidates.cand_graph
        cap = int(max_candidates_per_graph) if max_candidates_per_graph is not None else None
        candidates, cand_graph, counts, max_count, truncated = DualFlowBeamMixin._truncate_candidates_by_score(
            candidates,
            cand_graph=cand_graph,
            num_graphs=num_graphs,
            cap=cap,
        )
        if candidates is None:
            return None
        if not truncated:
            candidates, cand_graph = DualFlowBeamMixin._maybe_sort_candidates_by_graph(candidates, cand_graph)
            cand_graph = DualFlowBeamMixin._coerce_candidate_graph(cand_graph)
            counts = torch.bincount(cand_graph, minlength=num_graphs)
            max_count = int(counts.max().item()) if counts.numel() > 0 else 0

        device = cand_graph.device
        start = (counts.cumsum(0) - counts).index_select(0, cand_graph)
        pos = torch.arange(cand_graph.numel(), device=device) - start
        scores = torch.full((num_graphs, max_count), neg_inf, device=device, dtype=torch.float32)
        nodes = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        src_beam = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        edge_id = torch.full((num_graphs, max_count), _NEG_ONE, device=device, dtype=torch.long)
        is_edge = torch.zeros((num_graphs, max_count), device=device, dtype=torch.bool)
        done = torch.zeros((num_graphs, max_count), device=device, dtype=torch.bool)
        scores[cand_graph, pos] = candidates.cand_scores
        nodes[cand_graph, pos] = candidates.cand_nodes
        src_beam[cand_graph, pos] = candidates.cand_src_beam
        edge_id[cand_graph, pos] = candidates.cand_edge_id
        is_edge[cand_graph, pos] = candidates.cand_is_edge
        done[cand_graph, pos] = candidates.cand_done
        return _BeamCandidateMatrix(
            scores=scores,
            nodes=nodes,
            src_beam=src_beam,
            edge_id=edge_id,
            is_edge=is_edge,
            done=done,
            counts=counts,
        )

    @staticmethod
    def _build_diverse_keys(
        *,
        similarity: str,
        nodes: torch.Tensor,
        edge_id: torch.Tensor,
        src_beam: torch.Tensor,
        is_edge: torch.Tensor,
    ) -> torch.Tensor:
        if similarity == "tail":
            return nodes.to(dtype=torch.long)
        if similarity == "edge":
            stay_keys = -src_beam.to(dtype=torch.long) - 2
            edge_keys = edge_id.to(dtype=torch.long)
            return torch.where(is_edge, edge_keys, stay_keys)
        if similarity == "source":
            return src_beam.to(dtype=torch.long)
        raise ValueError(f"Unsupported diverse beam similarity: {similarity!r}.")

    @staticmethod
    def _compute_group_sizes(k_per_graph: torch.Tensor, groups: int, group_idx: int) -> torch.Tensor:
        base = k_per_graph // groups
        remainder = k_per_graph % groups
        return base + (remainder > group_idx).to(dtype=base.dtype)

    @staticmethod
    def _apply_diverse_penalty(
        scores: torch.Tensor,
        keys: torch.Tensor,
        *,
        used_keys: torch.Tensor,
        penalty: str,
        penalty_lambda: float,
        neg_inf: float,
    ) -> torch.Tensor:
        scores_adj = scores
        used_valid = used_keys != _NEG_ONE
        used_mask = (keys.unsqueeze(-1) == used_keys.unsqueeze(1)) & used_valid.unsqueeze(1)
        used_mask = used_mask.any(dim=2)
        if penalty == "hard":
            return scores_adj.masked_fill(used_mask, neg_inf)
        return scores_adj - penalty_lambda * used_mask.to(dtype=scores_adj.dtype)

    @staticmethod
    def _insert_group_selection(
        *,
        selected_pos: torch.Tensor,
        used_keys: torch.Tensor,
        selected_mask: torch.Tensor,
        selected_count: torch.Tensor,
        top_pos: torch.Tensor,
        top_scores: torch.Tensor,
        group_size: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        range_pos = torch.arange(top_pos.size(1), device=top_pos.device)
        take_mask = range_pos.unsqueeze(0) < group_size.unsqueeze(1)
        take_mask = take_mask & torch.isfinite(top_scores)
        rank = torch.cumsum(take_mask, dim=1) - 1
        flat_rows, flat_cols = torch.nonzero(take_mask, as_tuple=True)
        flat_rank = rank[flat_rows, flat_cols]
        insert_pos = selected_count[flat_rows] + flat_rank
        k = int(selected_pos.size(1))
        linear = flat_rows * k + insert_pos
        flat_pos = top_pos[flat_rows, flat_cols]
        selected_pos.view(-1)[linear] = flat_pos
        used_keys.view(-1)[linear] = keys[flat_rows, flat_pos]
        selected_mask[flat_rows, flat_pos] = True
        selected_count = selected_count + take_mask.sum(dim=1)
        return selected_count

    def _select_beam_positions(
        self,
        *,
        scores: torch.Tensor,
        keys: torch.Tensor,
        counts: torch.Tensor,
        beam_size: int,
        diverse_cfg: dict[str, Any],
        neg_inf: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs, max_count = scores.size()
        if beam_size <= 0 or max_count <= 0 or num_graphs <= 0:
            empty_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            empty_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
            return empty_pos, empty_scores
        k_per_graph = counts.clamp(max=beam_size)
        range_beam = torch.arange(beam_size, device=scores.device).unsqueeze(0)
        if not diverse_cfg["enabled"] or beam_size <= 1 or diverse_cfg["groups"] <= 1:
            k_top = min(int(beam_size), int(max_count))
            if k_top <= 0:
                empty_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
                empty_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
                return empty_pos, empty_scores
            top_scores, top_pos = torch.topk(scores, k_top, dim=1)
            sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            sel_scores = torch.full((num_graphs, beam_size), neg_inf, device=scores.device, dtype=torch.float32)
            sel_pos[:, :k_top] = top_pos
            sel_scores[:, :k_top] = top_scores
            valid = range_beam < k_per_graph.unsqueeze(1)
            sel_pos = torch.where(valid, sel_pos, torch.full_like(sel_pos, _NEG_ONE))
            sel_scores = torch.where(valid, sel_scores, torch.full_like(sel_scores, neg_inf))
            return sel_pos, sel_scores
        sel_pos = self._diverse_select_positions(
            scores=scores,
            keys=keys,
            counts=counts,
            beam_size=beam_size,
            groups=int(diverse_cfg["groups"]),
            penalty=str(diverse_cfg["penalty"]),
            penalty_lambda=float(diverse_cfg["lambda"]),
            neg_inf=neg_inf,
        )
        pos_safe = sel_pos.clamp(min=0)
        sel_scores = torch.gather(scores, 1, pos_safe)
        valid = range_beam < k_per_graph.unsqueeze(1)
        valid = valid & (sel_pos >= 0)
        sel_scores = torch.where(valid, sel_scores, torch.full_like(sel_scores, neg_inf))
        sel_pos = torch.where(valid, sel_pos, torch.full_like(sel_pos, _NEG_ONE))
        return sel_pos, sel_scores

    def _diverse_select_positions(
        self,
        *,
        scores: torch.Tensor,
        keys: torch.Tensor,
        counts: torch.Tensor,
        beam_size: int,
        groups: int,
        penalty: str,
        penalty_lambda: float,
        neg_inf: float,
    ) -> torch.Tensor:
        num_graphs, max_count = scores.size()
        if max_count <= 0 or beam_size <= 0 or num_graphs <= 0:
            return torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        range_count = torch.arange(max_count, device=scores.device).unsqueeze(0)
        valid_mask = range_count < counts.unsqueeze(1)
        valid_mask = valid_mask & torch.isfinite(scores)
        graph_ids = torch.arange(num_graphs, device=scores.device).unsqueeze(1).expand_as(scores)
        pos_ids = range_count.expand_as(scores)
        flat_scores = scores[valid_mask]
        if flat_scores.numel() == 0:
            return torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        flat_keys = keys[valid_mask].to(dtype=torch.long)
        flat_graph = graph_ids[valid_mask].to(dtype=torch.long)
        flat_pos = pos_ids[valid_mask]
        key_min = flat_keys.min().to(dtype=torch.long)
        key_max = flat_keys.max().to(dtype=torch.long)
        key_stride = (key_max - key_min + 1).clamp(min=1).to(dtype=torch.long)
        comp = flat_graph * key_stride + (flat_keys - key_min)
        order = torch.argsort(comp)
        comp_sorted = comp.index_select(0, order)
        scores_sorted = flat_scores.index_select(0, order)
        pos_sorted = flat_pos.index_select(0, order)
        graph_sorted = flat_graph.index_select(0, order)
        change = comp_sorted[1:] != comp_sorted[:-1]
        group_ids = torch.zeros_like(comp_sorted, dtype=torch.long)
        group_ids[1:] = torch.cumsum(change.to(dtype=torch.long), dim=0)
        num_groups = group_ids[-1] + 1
        if penalty == "soft":
            group_counts = torch.bincount(group_ids).to(dtype=scores_sorted.dtype)
            penalty_counts = group_counts.index_select(0, group_ids)
            scores_sorted = scores_sorted - penalty_lambda * penalty_counts
            inv_order = torch.empty_like(order)
            inv_order[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
            scores_adj = scores.clone().masked_fill(~valid_mask, neg_inf)
            scores_adj_flat = scores_sorted.index_select(0, inv_order)
            scores_adj[valid_mask] = scores_adj_flat
            k_top = min(int(beam_size), int(max_count))
            top_scores, top_pos = torch.topk(scores_adj, k_top, dim=1)
            sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
            sel_pos[:, :k_top] = top_pos
            return sel_pos
        _, argmax = segment_max(scores_sorted, group_ids, num_groups)
        best_scores = scores_sorted.index_select(0, argmax)
        best_pos = pos_sorted.index_select(0, argmax)
        best_graph = graph_sorted.index_select(0, argmax)
        counts_unique = torch.bincount(best_graph, minlength=num_graphs)
        if counts_unique.numel() > 0:
            max_unique = counts_unique.max().clamp(min=0)
        else:
            max_unique = torch.zeros((), device=scores.device, dtype=torch.long)
        unique_scores = torch.full((num_graphs, max_unique), neg_inf, device=scores.device, dtype=scores.dtype)
        unique_pos = torch.full((num_graphs, max_unique), _NEG_ONE, device=scores.device, dtype=torch.long)
        start = (counts_unique.cumsum(0) - counts_unique).index_select(0, best_graph)
        pos_unique = torch.arange(best_graph.numel(), device=scores.device) - start
        unique_scores[best_graph, pos_unique] = best_scores
        unique_pos[best_graph, pos_unique] = best_pos
        max_unique_val = unique_scores.size(1)
        k_top = min(int(beam_size), int(max_unique_val))
        sel_pos = torch.full((num_graphs, beam_size), _NEG_ONE, device=scores.device, dtype=torch.long)
        if k_top > 0:
            top_scores, top_idx = torch.topk(unique_scores, k_top, dim=1)
            _ = top_scores
            sel_pos[:, :k_top] = torch.gather(unique_pos, 1, top_idx)
        valid_sel = sel_pos >= 0
        selected_count = valid_sel.sum(dim=1)
        selected_mask = torch.zeros((num_graphs, max_count), device=scores.device, dtype=torch.bool)
        if torch.any(valid_sel):
            batch_idx = torch.arange(num_graphs, device=scores.device).unsqueeze(1).expand_as(sel_pos)
            selected_mask[batch_idx[valid_sel], sel_pos[valid_sel]] = True
        remaining = (beam_size - selected_count).clamp(min=0)
        scores_remain = scores.masked_fill(~valid_mask, neg_inf)
        scores_remain = scores_remain.masked_fill(selected_mask, neg_inf)
        k_remain = min(int(beam_size), int(max_count))
        top_scores_r, top_pos_r = torch.topk(scores_remain, k_remain, dim=1)
        range_k = torch.arange(beam_size, device=scores.device)
        take_mask = range_k.unsqueeze(0) < remaining.unsqueeze(1)
        rank = torch.cumsum(take_mask, dim=1) - 1
        flat_rows, flat_cols = torch.nonzero(take_mask, as_tuple=True)
        if flat_rows.numel() > 0:
            insert_pos = selected_count[flat_rows] + rank[flat_rows, flat_cols]
            linear = flat_rows * beam_size + insert_pos
            sel_pos.view(-1)[linear] = top_pos_r[flat_rows, flat_cols]
        return sel_pos

    def _beam_update_from_candidates(
        self,
        *,
        state: _BeamState,
        candidates: _BeamCandidates,
        step: int,
        diverse_cfg: dict[str, Any],
    ) -> _BeamState:
        max_candidates = diverse_cfg.get("max_candidates_per_graph")
        if max_candidates is None:
            groups = int(diverse_cfg.get("groups", 1))
            if not diverse_cfg.get("enabled", False):
                groups = 1
            max_candidates = state.beam_size * max(1, groups)
        else:
            max_candidates = int(max_candidates)
            if max_candidates <= 0:
                max_candidates = None
        matrix = self._build_candidate_matrix(
            candidates,
            num_graphs=state.num_graphs,
            neg_inf=state.neg_inf,
            max_candidates_per_graph=max_candidates,
        )
        if matrix is None:
            return _BeamState(
                beam_nodes=torch.full_like(state.beam_nodes, _NEG_ONE),
                beam_scores=torch.full_like(state.beam_scores, state.neg_inf),
                beam_paths=torch.full_like(state.beam_paths, _NEG_ONE),
                beam_lengths=torch.zeros_like(state.beam_lengths),
                beam_done=torch.zeros_like(state.beam_done),
                flat_graph_ids=state.flat_graph_ids,
                flat_beam_ids=state.flat_beam_ids,
                beam_context=state.beam_context,
                num_graphs=state.num_graphs,
                beam_size=state.beam_size,
                max_steps=state.max_steps,
                neg_inf=state.neg_inf,
            )
        keys = self._build_diverse_keys(
            similarity=str(diverse_cfg["similarity"]),
            nodes=matrix.nodes,
            edge_id=matrix.edge_id,
            src_beam=matrix.src_beam,
            is_edge=matrix.is_edge,
        )
        sel_pos, sel_scores = self._select_beam_positions(
            scores=matrix.scores,
            keys=keys,
            counts=matrix.counts,
            beam_size=state.beam_size,
            diverse_cfg=diverse_cfg,
            neg_inf=state.neg_inf,
        )
        pos_safe = sel_pos.clamp(min=0)
        sel_nodes = torch.gather(matrix.nodes, 1, pos_safe)
        sel_src = torch.gather(matrix.src_beam, 1, pos_safe)
        sel_edge_id = torch.gather(matrix.edge_id, 1, pos_safe)
        sel_is_edge = torch.gather(matrix.is_edge, 1, pos_safe)
        sel_done = torch.gather(matrix.done, 1, pos_safe)
        valid_sel = sel_pos >= 0
        sel_nodes = torch.where(valid_sel, sel_nodes, torch.full_like(sel_nodes, _NEG_ONE))
        sel_src = torch.where(valid_sel, sel_src, torch.full_like(sel_src, _NEG_ONE))
        sel_edge_id = torch.where(valid_sel, sel_edge_id, torch.full_like(sel_edge_id, _NEG_ONE))
        sel_is_edge = torch.where(valid_sel, sel_is_edge, torch.zeros_like(sel_is_edge))
        sel_done = torch.where(valid_sel, sel_done, torch.zeros_like(sel_done))
        sel_scores = torch.where(valid_sel, sel_scores, torch.full_like(sel_scores, state.neg_inf))
        batch_idx = torch.arange(state.num_graphs, device=state.beam_nodes.device).unsqueeze(1).expand_as(sel_src)
        sel_src_safe = sel_src.clamp(min=0)
        sel_paths = state.beam_paths[batch_idx, sel_src_safe]
        sel_lengths = state.beam_lengths[batch_idx, sel_src_safe]
        sel_paths = sel_paths.clone()
        sel_paths[:, :, step] = torch.where(sel_is_edge, sel_edge_id, sel_paths[:, :, step])
        sel_lengths = sel_lengths + sel_is_edge.to(dtype=sel_lengths.dtype)
        sel_paths = torch.where(valid_sel.unsqueeze(-1), sel_paths, torch.full_like(sel_paths, _NEG_ONE))
        sel_lengths = torch.where(valid_sel, sel_lengths, torch.zeros_like(sel_lengths))
        sel_done = torch.where(valid_sel, sel_done, torch.zeros_like(sel_done))
        return _BeamState(
            beam_nodes=sel_nodes,
            beam_scores=sel_scores,
            beam_paths=sel_paths,
            beam_lengths=sel_lengths,
            beam_done=sel_done,
            flat_graph_ids=state.flat_graph_ids,
            flat_beam_ids=state.flat_beam_ids,
            beam_context=state.beam_context,
            num_graphs=state.num_graphs,
            beam_size=state.beam_size,
            max_steps=state.max_steps,
            neg_inf=state.neg_inf,
        )

    @staticmethod
    def _beam_finalize(state: _BeamState) -> list[list[tuple[int, float, list[int]]]]:
        beam_nodes_np = state.beam_nodes.detach().cpu().numpy()
        beam_scores_np = state.beam_scores.detach().cpu().numpy()
        beam_paths_np = state.beam_paths.detach().cpu().numpy()
        beam_lengths_np = state.beam_lengths.detach().cpu().numpy()
        beams: list[list[tuple[int, float, list[int]]]] = []
        for graph_idx in range(state.num_graphs):
            graph_beams: list[tuple[int, float, list[int]]] = []
            for beam_idx in range(state.beam_size):
                node_id = int(beam_nodes_np[graph_idx, beam_idx])
                if node_id < 0:
                    continue
                score = float(beam_scores_np[graph_idx, beam_idx])
                length = int(beam_lengths_np[graph_idx, beam_idx])
                if length <= 0:
                    path = []
                else:
                    path = beam_paths_np[graph_idx, beam_idx, :length].tolist()
                graph_beams.append((node_id, score, path))
            beams.append(graph_beams)
        return beams


__all__ = ["DualFlowBeamMixin"]
