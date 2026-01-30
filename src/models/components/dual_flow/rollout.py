from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional

import torch

from src.models.components import QCBiANetwork
from src.models.components.graph_ops import (
    OutgoingEdges,
    gather_outgoing_edges,
    gumbel_noise_like,
    segment_logsumexp_1d,
    segment_max,
)

from .constants import (
    _NEG_ONE,
    _ONE,
    _PB_MODE_TOPO_SEMANTIC,
    _PB_MODE_UNIFORM,
    _TERMINAL_DEAD_END,
    _TERMINAL_HIT,
    _TERMINAL_INVALID_START,
    _TERMINAL_MAX_STEPS,
    _TERMINAL_NONE,
    _TWO,
    _ZERO,
)
from .types import _PreparedBatch, _RolloutResult


class DualFlowRolloutMixin:
    def _compute_log_z_for_nodes(
        self,
        *,
        node_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        steps: torch.Tensor,
        node_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        context_tokens = self._resolve_context_tokens(context_tokens)
        if node_ids is None:
            node_tokens_sel = node_tokens
            node_batch_sel = node_batch
        else:
            node_ids = node_ids.to(device=node_tokens.device, dtype=torch.long).view(-1)
            node_tokens_sel = node_tokens.index_select(0, node_ids)
            node_batch_sel = node_batch.index_select(0, node_ids)
        steps = steps.to(device=node_tokens_sel.device, dtype=torch.long).view(-1)
        max_batch = node_batch_sel.max()
        steps_num = torch.tensor(steps.numel(), device=max_batch.device)
        torch._assert(steps_num > max_batch, "steps length must cover max node batch index.")
        time_emb = self.z_time_encoder(steps).index_select(0, node_batch_sel)
        node_tokens_sel = node_tokens_sel + time_emb
        return self.z_predictor(
            node_tokens=node_tokens_sel,
            question_tokens=context_tokens,
            node_batch=node_batch_sel,
        )

    def _compute_edge_logits(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if edge_ids.numel() == _ZERO:
            return torch.zeros((_ZERO,), device=edge_ids.device, dtype=torch.float32)
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        head_tokens = prepared.node_tokens.index_select(0, heads)
        tail_tokens = prepared.node_tokens.index_select(0, tails)
        relation_tokens = prepared.relation_tokens.index_select(0, edge_ids)
        steps = steps.to(device=head_tokens.device, dtype=torch.long).view(-1)
        time_emb = self.z_time_encoder(steps).index_select(0, edge_batch)
        head_tokens = head_tokens + time_emb
        context_tokens = self._resolve_context_tokens(context_tokens)
        context_edge = context_tokens.index_select(0, edge_batch)
        logits = policy(context_edge, head_tokens, relation_tokens, tail_tokens, None)
        if temperature != float(_ONE):
            logits = logits / float(temperature)
        return logits

    @staticmethod
    def _cosine_similarity(x: torch.Tensor, y: torch.Tensor, *, eps: float) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        x_norm = x / x.norm(dim=-1, keepdim=True).clamp(min=eps)
        y_norm = y / y.norm(dim=-1, keepdim=True).clamp(min=eps)
        return (x_norm * y_norm).sum(dim=-1)

    def _compute_distance_to_starts(
        self,
        *,
        prepared: _PreparedBatch,
        max_hops: int,
    ) -> torch.Tensor:
        num_nodes_total = int(prepared.num_nodes_total)
        max_hops = int(max_hops)
        if num_nodes_total <= _ZERO:
            return torch.zeros((_ZERO,), device=prepared.edge_index.device, dtype=torch.long)
        distance_inf = max_hops + _ONE
        dist = torch.full((num_nodes_total,), distance_inf, device=prepared.edge_index.device, dtype=torch.long)
        start_nodes = prepared.q_local_indices.to(device=dist.device, dtype=torch.long).view(-1)
        if start_nodes.numel() == _ZERO:
            return dist
        valid = start_nodes >= _ZERO
        start_nodes = start_nodes[valid]
        dist.index_fill_(0, start_nodes, int(_ZERO))
        frontier = torch.zeros((num_nodes_total,), device=dist.device, dtype=torch.bool)
        frontier.index_fill_(0, start_nodes, True)
        edge_ids = prepared.edge_ids_by_head_fwd
        if edge_ids.numel() == _ZERO or max_hops <= _ZERO:
            return dist
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        for step in range(max_hops):
            active = frontier.index_select(0, heads)
            candidate_tails = tails[active]
            unseen = dist.index_select(0, candidate_tails) == distance_inf
            new_nodes = candidate_tails[unseen]
            dist.index_fill_(0, new_nodes, int(step + _ONE))
            frontier = torch.zeros_like(frontier)
            frontier.index_fill_(0, new_nodes, True)
        return dist

    def _compute_pb_logits(
        self,
        *,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        dist_to_start: Optional[torch.Tensor],
        pb_cfg: dict[str, float | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=edge_ids.device, dtype=torch.long).view(-1)
        mode = str(pb_cfg["mode"])
        if mode == _PB_MODE_UNIFORM:
            logits = torch.zeros((edge_ids.numel(),), device=edge_ids.device, dtype=torch.float32)
            allowed = torch.ones_like(edge_ids, dtype=torch.bool)
            return logits, allowed
        if mode != _PB_MODE_TOPO_SEMANTIC:
            raise ValueError(f"Unsupported static pb mode: {mode!r}.")
        if dist_to_start is None:
            raise ValueError("dist_to_start is required for topo_semantic pb.")
        heads = prepared.edge_index[_ZERO].index_select(0, edge_ids)
        tails = prepared.edge_index[_ONE].index_select(0, edge_ids)
        dist_to_start = dist_to_start.to(device=edge_ids.device, dtype=torch.long)
        dist_heads = dist_to_start.index_select(0, heads)
        dist_tails = dist_to_start.index_select(0, tails)
        allowed = dist_tails < dist_heads
        topo_penalty = float(pb_cfg["topo_penalty"])
        topo_logits = torch.where(
            allowed,
            torch.zeros_like(dist_heads, dtype=torch.float32),
            torch.full_like(dist_heads, topo_penalty, dtype=torch.float32),
        )
        question_emb = self._resolve_context_tokens(prepared.question_emb_raw)
        query = question_emb.index_select(0, edge_batch)
        rel_emb = prepared.edge_embeddings_raw.index_select(0, edge_ids)
        cosine_eps = float(pb_cfg["cosine_eps"])
        sem = self._cosine_similarity(query, rel_emb, eps=cosine_eps)
        semantic_weight = float(pb_cfg["semantic_weight"])
        logits = topo_logits + sem.mul(semantic_weight)
        return logits, allowed

    def _compute_pb_log_prob(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        pb_cfg: dict[str, float | int | str],
        visited_nodes: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
        return_no_allowed: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for pb log prob.")
        outgoing = gather_outgoing_edges(
            curr_nodes=parent_nodes,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
            active_mask=move_mask,
        )
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=move_mask.numel(),
            edge_index=prepared.edge_index,
            edge_mask=edge_mask,
            visited_nodes=visited_nodes,
        )
        if outgoing.edge_ids.numel() == _ZERO:
            zeros = torch.zeros_like(move_mask, dtype=torch.float32)
            if return_no_allowed:
                return zeros, move_mask.to(dtype=torch.bool)
            return zeros
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        logits, allowed = self._compute_pb_logits(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            dist_to_start=dist_to_start,
            pb_cfg=pb_cfg,
        )
        num_graphs = move_mask.numel()
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, num_graphs)
        log_pb_edge = chosen_logits - log_denom
        allowed_batch = edge_batch[allowed]
        allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        no_allowed = allowed_counts == _ZERO
        topo_penalty = float(pb_cfg["topo_penalty"])
        log_pb_edge = torch.where(no_allowed, torch.full_like(log_pb_edge, topo_penalty), log_pb_edge)
        log_pb_step = torch.where(move_mask, log_pb_edge, torch.zeros_like(log_pb_edge))
        if return_no_allowed:
            return log_pb_step, no_allowed
        return log_pb_step

    def _sample_pb_edges(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        pb_cfg: dict[str, float | int | str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return torch.full((num_graphs,), _NEG_ONE, device=prepared.edge_index.device, dtype=torch.long), zeros, zeros
        logits, allowed = self._compute_pb_logits(
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            dist_to_start=dist_to_start,
            pb_cfg=pb_cfg,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        allowed_batch = edge_batch[allowed]
        allowed_counts = torch.bincount(allowed_batch, minlength=num_graphs)
        has_allowed = allowed_counts > _ZERO
        return chosen_edge, log_prob_chosen, has_allowed

    def _rollout_pb(
        self,
        *,
        prepared: _PreparedBatch,
        dist_to_start: Optional[torch.Tensor],
        graph_mask: torch.Tensor,
        start_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        record_actions: bool,
        record_log_pf: bool,
        pb_cfg: dict[str, float | int | str],
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.num_graphs)
        device = prepared.edge_index.device
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for pb rollout.")
        log_pf_sum = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        num_moves = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        curr_nodes = start_nodes.clone()
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        stop_reason = torch.full((num_graphs,), _TERMINAL_NONE, device=device, dtype=torch.long)
        invalid_start = graph_mask & (curr_nodes < _ZERO)
        stop_reason = torch.where(
            invalid_start, torch.full_like(stop_reason, _TERMINAL_INVALID_START), stop_reason
        )
        active = graph_mask & (curr_nodes >= _ZERO)
        visited_nodes = None
        if self._avoid_revisit:
            visited_nodes = torch.zeros((prepared.node_batch.numel(),), device=device, dtype=torch.bool)
            visited_nodes.index_fill_(0, curr_nodes[active], True)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        if record_actions:
            actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=device, dtype=torch.long)
        if record_log_pf:
            log_pf_steps = torch.zeros((num_graphs, self.max_steps), device=device, dtype=torch.float32)
        for step in range(int(self.max_steps)):
            at_target = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & active
            stop_nodes = torch.where(at_target, curr_nodes, stop_nodes)
            stop_reason = torch.where(at_target, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
            active = active & ~at_target
            outgoing = gather_outgoing_edges(
                curr_nodes=curr_nodes,
                edge_ids_by_head=edge_ids_by_head,
                edge_ptr_by_head=edge_ptr_by_head,
                active_mask=active,
            )
            outgoing = self._apply_action_constraints_to_outgoing(
                outgoing,
                num_graphs=num_graphs,
                edge_index=prepared.edge_index,
                edge_mask=edge_mask,
                visited_nodes=visited_nodes,
            )
            move_mask = active & outgoing.has_edge
            if outgoing.edge_ids.numel() > _ZERO:
                chosen_edge, log_pf_step, has_allowed = self._sample_pb_edges(
                    prepared=prepared,
                    dist_to_start=dist_to_start,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    pb_cfg=pb_cfg,
                )
                move_mask = move_mask & has_allowed
                chosen_edge = torch.where(move_mask, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                if visited_nodes is not None:
                    visited_nodes.index_fill_(0, chosen_tail[move_mask], True)
                log_pf_step = torch.where(move_mask, log_pf_step, torch.zeros_like(log_pf_step))
                log_pf_sum = log_pf_sum + log_pf_step
                num_moves = num_moves + move_mask.to(dtype=torch.long)
                if record_actions and actions is not None:
                    actions[:, step] = torch.where(move_mask, chosen_edge, actions[:, step])
                if record_log_pf and log_pf_steps is not None:
                    log_pf_steps[:, step] = log_pf_step
            no_edge = active & ~move_mask
            stop_nodes = torch.where(no_edge, curr_nodes, stop_nodes)
            stop_reason = torch.where(no_edge, torch.full_like(stop_reason, _TERMINAL_DEAD_END), stop_reason)
            active = active & move_mask
        stop_nodes = torch.where(
            stop_nodes >= _ZERO,
            stop_nodes,
            torch.where(active, curr_nodes, torch.full_like(curr_nodes, _NEG_ONE)),
        )
        stop_reason = torch.where(active, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        return _RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            stop_reason=stop_reason,
            actions=actions,
            log_pf_steps=log_pf_steps,
        )

    def _sample_pb_edge_dropout_mask(self, *, edge_index: torch.Tensor) -> Optional[torch.Tensor]:
        drop_prob = float(self._resolve_db_cfg()["pb_edge_dropout"])
        if drop_prob <= float(_ZERO):
            return None
        num_edges = int(edge_index.size(1))
        if num_edges <= _ZERO:
            return None
        keep = torch.rand((num_edges,), device=edge_index.device) >= drop_prob
        return keep

    def _sample_edges(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        edge_ids: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        steps: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        if edge_ids.numel() == _ZERO:
            zeros = torch.zeros((num_graphs,), device=prepared.edge_index.device, dtype=torch.float32)
            return torch.full((num_graphs,), _NEG_ONE, device=prepared.edge_index.device, dtype=torch.long), zeros, zeros
        logits = self._compute_edge_logits(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps + _ONE,
            temperature=temperature,
            context_tokens=context_tokens,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=num_graphs)
        log_probs = logits - log_denom.index_select(0, edge_batch)
        scores = log_probs + gumbel_noise_like(log_probs)
        _, argmax = segment_max(scores, edge_batch, num_graphs)
        chosen_edge = edge_ids.index_select(0, argmax)
        log_prob_chosen = log_probs.index_select(0, argmax)
        return chosen_edge, log_prob_chosen, log_denom

    def _compute_forward_log_prob(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        chosen_edge: torch.Tensor,
        parent_nodes: torch.Tensor,
        move_mask: torch.Tensor,
        steps: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        temperature: float,
        context_tokens: torch.Tensor,
        visited_nodes: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for forward log prob.")
        outgoing = gather_outgoing_edges(
            curr_nodes=parent_nodes,
            edge_ids_by_head=edge_ids_by_head,
            edge_ptr_by_head=edge_ptr_by_head,
            active_mask=move_mask,
        )
        outgoing = self._apply_action_constraints_to_outgoing(
            outgoing,
            num_graphs=move_mask.numel(),
            edge_index=prepared.edge_index,
            edge_mask=edge_mask,
            visited_nodes=visited_nodes,
        )
        if outgoing.edge_ids.numel() == _ZERO:
            return torch.zeros_like(move_mask, dtype=torch.float32)
        edge_ids = outgoing.edge_ids.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        edge_batch = outgoing.edge_batch.to(device=prepared.edge_index.device, dtype=torch.long).view(-1)
        logits = self._compute_edge_logits(
            policy=policy,
            prepared=prepared,
            edge_ids=edge_ids,
            edge_batch=edge_batch,
            steps=steps + _ONE,
            temperature=temperature,
            context_tokens=context_tokens,
        )
        log_denom = self._compute_log_denom(logits=logits, edge_batch=edge_batch, num_graphs=move_mask.numel())
        chosen_edge_safe = chosen_edge.clamp(min=_ZERO)
        chosen_for_edge = chosen_edge_safe.index_select(0, edge_batch)
        match = edge_ids == chosen_for_edge
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(match, logits, torch.full_like(logits, neg_inf))
        chosen_logits, _ = segment_max(masked, edge_batch, move_mask.numel())
        log_pf_edge = chosen_logits - log_denom
        has_edge = outgoing.has_edge.to(device=log_pf_edge.device, dtype=torch.bool)
        log_pf_edge = torch.where(has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        log_pf_step = torch.where(move_mask & has_edge, log_pf_edge, torch.zeros_like(log_pf_edge))
        return log_pf_step

    def _rollout_policy(
        self,
        *,
        policy: QCBiANetwork,
        prepared: _PreparedBatch,
        graph_mask: torch.Tensor,
        start_nodes: torch.Tensor,
        node_is_target: torch.Tensor,
        edge_ids_by_head: torch.Tensor,
        edge_ptr_by_head: torch.Tensor,
        record_actions: bool,
        record_log_pf: bool,
        temperature: float,
        context_tokens: torch.Tensor,
        edge_mask: Optional[torch.Tensor] = None,
    ) -> _RolloutResult:
        num_graphs = int(prepared.num_graphs)
        device = prepared.edge_index.device
        if edge_mask is not None and edge_mask.numel() != prepared.edge_index.size(1):
            raise ValueError("edge_mask length must match edge_index for rollout policy.")
        log_pf_sum = torch.zeros((num_graphs,), device=device, dtype=torch.float32)
        num_moves = torch.zeros((num_graphs,), device=device, dtype=torch.long)
        curr_nodes = start_nodes.clone()
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        stop_reason = torch.full((num_graphs,), _TERMINAL_NONE, device=device, dtype=torch.long)
        invalid_start = graph_mask & (curr_nodes < _ZERO)
        stop_reason = torch.where(
            invalid_start, torch.full_like(stop_reason, _TERMINAL_INVALID_START), stop_reason
        )
        active = graph_mask & (curr_nodes >= _ZERO)
        visited_nodes = None
        if self._avoid_revisit:
            visited_nodes = torch.zeros((prepared.node_batch.numel(),), device=device, dtype=torch.bool)
            visited_nodes.index_fill_(0, curr_nodes[active], True)
        stop_nodes = torch.full((num_graphs,), _NEG_ONE, device=device, dtype=torch.long)
        actions = None
        log_pf_steps = None
        if record_actions:
            actions = torch.full((num_graphs, self.max_steps), _NEG_ONE, device=device, dtype=torch.long)
        if record_log_pf:
            log_pf_steps = torch.zeros((num_graphs, self.max_steps), device=device, dtype=torch.float32)
        for step in range(int(self.max_steps)):
            at_target = node_is_target.index_select(0, curr_nodes.clamp(min=_ZERO)) & active
            stop_nodes = torch.where(at_target, curr_nodes, stop_nodes)
            stop_reason = torch.where(at_target, torch.full_like(stop_reason, _TERMINAL_HIT), stop_reason)
            active = active & ~at_target
            outgoing = gather_outgoing_edges(
                curr_nodes=curr_nodes,
                edge_ids_by_head=edge_ids_by_head,
                edge_ptr_by_head=edge_ptr_by_head,
                active_mask=active,
            )
            outgoing = self._apply_action_constraints_to_outgoing(
                outgoing,
                num_graphs=num_graphs,
                edge_index=prepared.edge_index,
                edge_mask=edge_mask,
                visited_nodes=visited_nodes,
            )
            move_mask = active & outgoing.has_edge
            if outgoing.edge_ids.numel() > _ZERO:
                step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
                chosen_edge, log_pf_step, _ = self._sample_edges(
                    policy=policy,
                    prepared=prepared,
                    edge_ids=outgoing.edge_ids,
                    edge_batch=outgoing.edge_batch,
                    num_graphs=num_graphs,
                    steps=step_ids,
                    temperature=temperature,
                    context_tokens=context_tokens,
                )
                chosen_edge = torch.where(outgoing.has_edge, chosen_edge, torch.full_like(chosen_edge, _NEG_ONE))
                chosen_tail = prepared.edge_index[_ONE].index_select(0, chosen_edge.clamp(min=_ZERO))
                curr_nodes = torch.where(move_mask, chosen_tail, curr_nodes)
                if visited_nodes is not None:
                    visited_nodes.index_fill_(0, chosen_tail[move_mask], True)
                log_pf_step = torch.where(move_mask, log_pf_step, torch.zeros_like(log_pf_step))
                log_pf_sum = log_pf_sum + log_pf_step
                num_moves = num_moves + move_mask.to(dtype=torch.long)
                if record_actions and actions is not None:
                    actions[:, step] = torch.where(move_mask, chosen_edge, actions[:, step])
                if record_log_pf and log_pf_steps is not None:
                    log_pf_steps[:, step] = log_pf_step
            no_edge = active & ~outgoing.has_edge
            stop_nodes = torch.where(no_edge, curr_nodes, stop_nodes)
            stop_reason = torch.where(no_edge, torch.full_like(stop_reason, _TERMINAL_DEAD_END), stop_reason)
            active = active & outgoing.has_edge
        stop_nodes = torch.where(
            stop_nodes >= _ZERO,
            stop_nodes,
            torch.where(active, curr_nodes, torch.full_like(curr_nodes, _NEG_ONE)),
        )
        stop_reason = torch.where(active, torch.full_like(stop_reason, _TERMINAL_MAX_STEPS), stop_reason)
        return _RolloutResult(
            log_pf_sum=log_pf_sum,
            stop_nodes=stop_nodes,
            num_moves=num_moves,
            stop_reason=stop_reason,
            actions=actions,
            log_pf_steps=log_pf_steps,
        )

    def _apply_action_constraints_to_outgoing(
        self,
        outgoing: OutgoingEdges,
        *,
        num_graphs: int,
        edge_index: torch.Tensor,
        edge_mask: Optional[torch.Tensor],
        visited_nodes: Optional[torch.Tensor],
    ) -> OutgoingEdges:
        if edge_mask is not None:
            outgoing = self._apply_edge_mask_to_outgoing(outgoing, edge_mask=edge_mask, num_graphs=num_graphs)
        if visited_nodes is not None:
            outgoing = self._apply_no_revisit_to_outgoing(
                outgoing,
                visited_nodes=visited_nodes,
                edge_index=edge_index,
                num_graphs=num_graphs,
            )
        return outgoing

    @staticmethod
    def _apply_edge_mask_to_outgoing(
        outgoing: OutgoingEdges,
        *,
        edge_mask: torch.Tensor,
        num_graphs: int,
    ) -> OutgoingEdges:
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        if edge_ids.numel() == _ZERO:
            return outgoing
        edge_mask = edge_mask.to(device=edge_ids.device, dtype=torch.bool).view(-1)
        if edge_mask.numel() == _ZERO:
            return outgoing
        keep = edge_mask.index_select(0, edge_ids)
        edge_ids = edge_ids[keep]
        edge_batch = edge_batch[keep]
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=edge_ids.device, dtype=torch.long)
        has_edge = counts > _ZERO
        return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=counts, has_edge=has_edge)

    @staticmethod
    def _apply_no_revisit_to_outgoing(
        outgoing: OutgoingEdges,
        *,
        visited_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        num_graphs: int,
    ) -> OutgoingEdges:
        edge_ids = outgoing.edge_ids
        edge_batch = outgoing.edge_batch
        if edge_ids.numel() == _ZERO:
            return outgoing
        visited_nodes = visited_nodes.to(device=edge_ids.device, dtype=torch.bool).view(-1)
        tails = edge_index[_ONE].index_select(0, edge_ids)
        keep = ~visited_nodes.index_select(0, tails)
        edge_ids = edge_ids[keep]
        edge_batch = edge_batch[keep]
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(device=edge_ids.device, dtype=torch.long)
        has_edge = counts > _ZERO
        return OutgoingEdges(edge_ids=edge_ids, edge_batch=edge_batch, edge_counts=counts, has_edge=has_edge)

    def _compute_db_loss(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: Optional[_PreparedBatch],
        actions: torch.Tensor,
        graph_mask: torch.Tensor,
        traj_lengths: torch.Tensor,
        stop_reason: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        edge_mask_bwd: Optional[torch.Tensor] = None,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = prepared_fwd.node_ptr.device
        graph_mask = graph_mask.to(device=device, dtype=torch.bool)
        num_graphs, max_steps = actions.shape
        if max_steps == _ZERO:
            zero = torch.zeros((), device=device, dtype=torch.float32)
            return self._ensure_loss_requires_grad(zero), {"db_loss": zero.detach()}

        db_cfg = self._resolve_db_cfg()
        dead_end_log_reward = float(db_cfg["dead_end_log_reward"])
        dead_end_weight = float(db_cfg["dead_end_weight"])
        edge_mask = actions >= _ZERO
        failure_mask = (stop_reason != _TERMINAL_HIT) & graph_mask
        weight = torch.ones((num_graphs,), device=device, dtype=torch.float32)
        if dead_end_weight != float(_ONE):
            weight = torch.where(failure_mask, weight * dead_end_weight, weight)
        dist_to_start = None
        if pb_distances is not None:
            dist_to_start = pb_distances.to(device=device, dtype=torch.long)

        accum = self._init_db_accumulators(device=device)
        total = accum["total"]
        denom = accum["denom"]
        valid_count = accum["valid_count"]
        move_count = accum["move_count"]
        log_pb_sum = accum["log_pb_sum"]
        log_pb_min = accum["log_pb_min"]
        log_z_u_sum = accum["log_z_u_sum"]
        log_z_v_sum = accum["log_z_v_sum"]
        inv_invalid_count = accum["inv_invalid_count"]
        topo_violation_count = accum["topo_violation_count"]
        no_allowed_count = accum["no_allowed_count"]
        finite_pf_count = accum["finite_pf_count"]
        finite_pb_count = accum["finite_pb_count"]
        finite_z_u_count = accum["finite_z_u_count"]
        finite_z_v_count = accum["finite_z_v_count"]
        visited_fwd = None
        visited_bwd = None
        if self._avoid_revisit:
            num_nodes_total = int(prepared_fwd.node_batch.numel())
            visited_fwd = torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
            visited_bwd = torch.zeros((num_nodes_total,), device=device, dtype=torch.bool)
            mask_all = edge_mask & graph_mask.view(-1, _ONE)
            edges_taken = actions[mask_all].to(device=device, dtype=torch.long).clamp(min=_ZERO)
            tails_taken = prepared_fwd.edge_index[_ONE].index_select(0, edges_taken)
            visited_bwd.index_fill_(0, tails_taken, True)
        for step in range(max_steps):
            edge_ids = actions[:, step]
            move_mask = edge_mask[:, step] & graph_mask
            move_count = move_count + move_mask.to(dtype=torch.float32).sum()
            safe_edges = edge_ids.clamp(min=_ZERO)
            heads = prepared_fwd.edge_index[_ZERO].index_select(0, safe_edges)
            tails = prepared_fwd.edge_index[_ONE].index_select(0, safe_edges)
            if visited_fwd is not None:
                visited_fwd.index_fill_(0, heads[move_mask], True)
            step_ids = self._build_step_ids(num_graphs=num_graphs, step=step, device=device)
            next_step_ids = step_ids + _ONE
            log_z_u = self._compute_log_z_for_nodes(
                node_tokens=prepared_fwd.node_tokens,
                context_tokens=prepared_fwd.context_tokens,
                node_batch=prepared_fwd.node_batch,
                steps=step_ids,
                node_ids=heads,
            )
            log_z_v = self._compute_log_z_for_nodes(
                node_tokens=prepared_fwd.node_tokens,
                context_tokens=prepared_fwd.context_tokens,
                node_batch=prepared_fwd.node_batch,
                steps=next_step_ids,
                node_ids=tails,
            )
            log_pf = self._compute_forward_log_prob(
                policy=self.policy_fwd,
                prepared=prepared_fwd,
                chosen_edge=edge_ids,
                parent_nodes=heads,
                move_mask=move_mask,
                steps=step_ids,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
                temperature=sampling_temperature,
                context_tokens=prepared_fwd.context_tokens,
                visited_nodes=visited_fwd,
            )
            inv_edge = prepared_fwd.edge_inverse_map.index_select(0, safe_edges)
            inv_valid = inv_edge >= _ZERO
            inv_edge = torch.where(inv_valid, inv_edge, torch.full_like(inv_edge, _NEG_ONE))
            active_bwd = move_mask & inv_valid
            if self._is_static_pb():
                if pb_cfg is None:
                    pb_cfg = self._resolve_pb_cfg()
                if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC and pb_distances is None:
                    raise ValueError("pb_distances required for topo_semantic pb DB loss.")
                log_pb, no_allowed = self._compute_pb_log_prob(
                    prepared=prepared_fwd,
                    dist_to_start=pb_distances,
                    chosen_edge=inv_edge,
                    parent_nodes=tails,
                    move_mask=active_bwd,
                    edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                    pb_cfg=pb_cfg,
                    visited_nodes=visited_bwd,
                    edge_mask=edge_mask_bwd,
                    return_no_allowed=True,
                )
                no_allowed_count = no_allowed_count + (no_allowed & active_bwd).to(dtype=torch.float32).sum()
            else:
                if prepared_bwd is None:
                    raise RuntimeError("prepared_bwd required for learned backward policy.")
                log_pb = self._compute_forward_log_prob(
                    policy=self.policy_bwd,
                    prepared=prepared_bwd,
                    chosen_edge=inv_edge,
                    parent_nodes=tails,
                    move_mask=active_bwd,
                    steps=next_step_ids,
                    edge_ids_by_head=prepared_bwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_bwd.edge_ptr_by_head_bwd,
                    temperature=float(_ONE),
                    context_tokens=prepared_bwd.context_tokens,
                    visited_nodes=visited_bwd,
                    edge_mask=edge_mask_bwd,
                )
            is_target = node_is_target.index_select(0, tails.clamp(min=_ZERO)) & move_mask
            log_z_v = torch.where(is_target, torch.zeros_like(log_z_v), log_z_v)
            is_terminal = traj_lengths == (step + _ONE)
            dead_end = is_terminal & failure_mask
            log_z_v = torch.where(
                dead_end,
                torch.full_like(log_z_v, dead_end_log_reward),
                log_z_v,
            )
            inv_invalid_count = inv_invalid_count + (move_mask & ~inv_valid).to(dtype=torch.float32).sum()
            finite_pf = torch.isfinite(log_pf) & move_mask
            finite_pb = torch.isfinite(log_pb) & move_mask
            finite_z_u = torch.isfinite(log_z_u) & move_mask
            finite_z_v = torch.isfinite(log_z_v) & move_mask
            finite_pf_count = finite_pf_count + finite_pf.to(dtype=torch.float32).sum()
            finite_pb_count = finite_pb_count + finite_pb.to(dtype=torch.float32).sum()
            finite_z_u_count = finite_z_u_count + finite_z_u.to(dtype=torch.float32).sum()
            finite_z_v_count = finite_z_v_count + finite_z_v.to(dtype=torch.float32).sum()
            finite_all = finite_pf & finite_pb & finite_z_u & finite_z_v
            valid = move_mask & inv_valid & finite_all
            valid_f = valid.to(dtype=torch.float32)
            valid_count = valid_count + valid_f.sum()
            log_pb_sum = log_pb_sum + (log_pb * valid_f).sum()
            log_z_u_sum = log_z_u_sum + (log_z_u * valid_f).sum()
            log_z_v_sum = log_z_v_sum + (log_z_v * valid_f).sum()
            pb_for_min = torch.where(valid, log_pb, torch.full_like(log_pb, float("inf")))
            log_pb_min = torch.minimum(log_pb_min, pb_for_min.min())
            if dist_to_start is not None:
                inv_edge_safe = inv_edge.clamp(min=_ZERO)
                inv_heads = prepared_fwd.edge_index[_ZERO].index_select(0, inv_edge_safe)
                inv_tails = prepared_fwd.edge_index[_ONE].index_select(0, inv_edge_safe)
                dist_heads = dist_to_start.index_select(0, inv_heads)
                dist_tails = dist_to_start.index_select(0, inv_tails)
                allowed_inv = dist_tails < dist_heads
                topo_violation = valid & ~allowed_inv
                topo_violation_count = topo_violation_count + topo_violation.to(dtype=torch.float32).sum()
            delta = (log_z_u + log_pf) - (log_z_v + log_pb)
            delta = torch.where(valid, delta, torch.zeros_like(delta))
            step_weight = weight * valid.to(dtype=weight.dtype)
            total = total + (delta.pow(_TWO) * step_weight).sum()
            denom = denom + step_weight.sum()
            if visited_fwd is not None and visited_bwd is not None:
                visited_fwd.index_fill_(0, tails[move_mask], True)
                visited_bwd.index_fill_(0, tails[move_mask], False)
        loss, metrics = self._finalize_db_metrics(
            total=total,
            denom=denom,
            valid_count=valid_count,
            move_count=move_count,
            log_pb_sum=log_pb_sum,
            log_pb_min=log_pb_min,
            log_z_u_sum=log_z_u_sum,
            log_z_v_sum=log_z_v_sum,
            inv_invalid_count=inv_invalid_count,
            topo_violation_count=topo_violation_count,
            no_allowed_count=no_allowed_count,
            finite_pf_count=finite_pf_count,
            finite_pb_count=finite_pb_count,
            finite_z_u_count=finite_z_u_count,
            finite_z_v_count=finite_z_v_count,
            device=device,
        )
        return self._ensure_loss_requires_grad(loss), metrics

    @staticmethod
    def _init_db_accumulators(*, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "total": torch.zeros((), device=device, dtype=torch.float32),
            "denom": torch.zeros((), device=device, dtype=torch.float32),
            "valid_count": torch.zeros((), device=device, dtype=torch.float32),
            "move_count": torch.zeros((), device=device, dtype=torch.float32),
            "log_pb_sum": torch.zeros((), device=device, dtype=torch.float32),
            "log_pb_min": torch.full((), float("inf"), device=device, dtype=torch.float32),
            "log_z_u_sum": torch.zeros((), device=device, dtype=torch.float32),
            "log_z_v_sum": torch.zeros((), device=device, dtype=torch.float32),
            "inv_invalid_count": torch.zeros((), device=device, dtype=torch.float32),
            "topo_violation_count": torch.zeros((), device=device, dtype=torch.float32),
            "no_allowed_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_pf_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_pb_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_z_u_count": torch.zeros((), device=device, dtype=torch.float32),
            "finite_z_v_count": torch.zeros((), device=device, dtype=torch.float32),
        }

    @staticmethod
    def _finalize_db_metrics(
        *,
        total: torch.Tensor,
        denom: torch.Tensor,
        valid_count: torch.Tensor,
        move_count: torch.Tensor,
        log_pb_sum: torch.Tensor,
        log_pb_min: torch.Tensor,
        log_z_u_sum: torch.Tensor,
        log_z_v_sum: torch.Tensor,
        inv_invalid_count: torch.Tensor,
        topo_violation_count: torch.Tensor,
        no_allowed_count: torch.Tensor,
        finite_pf_count: torch.Tensor,
        finite_pb_count: torch.Tensor,
        finite_z_u_count: torch.Tensor,
        finite_z_v_count: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        has_denom = denom > float(_ZERO)
        denom_safe = torch.where(has_denom, denom, torch.ones_like(denom))
        loss = total / denom_safe
        loss = torch.where(has_denom, loss, zero)
        valid_any = valid_count > _ZERO
        move_any = move_count > _ZERO
        valid_count_safe = torch.where(valid_any, valid_count, torch.ones_like(valid_count))
        move_count_safe = torch.where(move_any, move_count, torch.ones_like(move_count))
        log_pb_mean = torch.where(valid_any, log_pb_sum / valid_count_safe, zero)
        log_z_u_mean = torch.where(valid_any, log_z_u_sum / valid_count_safe, zero)
        log_z_v_mean = torch.where(valid_any, log_z_v_sum / valid_count_safe, zero)
        log_pb_min = torch.where(valid_any, log_pb_min, zero)
        inv_edge_invalid_rate = torch.where(move_any, inv_invalid_count / move_count_safe, zero)
        no_allowed_rate = torch.where(move_any, no_allowed_count / move_count_safe, zero)
        topo_violation_rate = torch.where(valid_any, topo_violation_count / valid_count_safe, zero)
        valid_step_rate = torch.where(move_any, valid_count / move_count_safe, zero)
        finite_pf_rate = torch.where(move_any, finite_pf_count / move_count_safe, zero)
        finite_pb_rate = torch.where(move_any, finite_pb_count / move_count_safe, zero)
        finite_z_u_rate = torch.where(move_any, finite_z_u_count / move_count_safe, zero)
        finite_z_v_rate = torch.where(move_any, finite_z_v_count / move_count_safe, zero)
        metrics = {
            "db_loss": loss.detach(),
            "db_log_pb_mean": log_pb_mean.detach(),
            "db_log_pb_min": log_pb_min.detach(),
            "db_log_z_u_mean": log_z_u_mean.detach(),
            "db_log_z_v_mean": log_z_v_mean.detach(),
            "db_inv_edge_invalid_rate": inv_edge_invalid_rate.detach(),
            "db_no_allowed_rate": no_allowed_rate.detach(),
            "db_topo_violation_rate": topo_violation_rate.detach(),
            "db_valid_step_rate": valid_step_rate.detach(),
            "db_finite_pf_rate": finite_pf_rate.detach(),
            "db_finite_pb_rate": finite_pb_rate.detach(),
            "db_finite_z_u_rate": finite_z_u_rate.detach(),
            "db_finite_z_v_rate": finite_z_v_rate.detach(),
        }
        return loss, metrics

    @staticmethod
    def _build_terminal_metrics(
        *,
        stop_reason: torch.Tensor,
        graph_mask: torch.Tensor,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        stop_reason = stop_reason.to(device=graph_mask.device, dtype=torch.long)
        graph_mask = graph_mask.to(device=stop_reason.device, dtype=torch.bool)
        denom = graph_mask.to(dtype=torch.float32).sum().clamp(min=_ONE)
        hit = ((stop_reason == _TERMINAL_HIT) & graph_mask).to(dtype=torch.float32).sum() / denom
        dead = ((stop_reason == _TERMINAL_DEAD_END) & graph_mask).to(dtype=torch.float32).sum() / denom
        max_steps = ((stop_reason == _TERMINAL_MAX_STEPS) & graph_mask).to(dtype=torch.float32).sum() / denom
        invalid = ((stop_reason == _TERMINAL_INVALID_START) & graph_mask).to(dtype=torch.float32).sum() / denom
        other = ((stop_reason == _TERMINAL_NONE) & graph_mask).to(dtype=torch.float32).sum() / denom
        return {
            f"{prefix}_terminal_hit_rate": hit,
            f"{prefix}_terminal_dead_end_rate": dead,
            f"{prefix}_terminal_max_steps_rate": max_steps,
            f"{prefix}_terminal_invalid_start_rate": invalid,
            f"{prefix}_terminal_other_rate": other,
        }

    @staticmethod
    def _validate_training_batch(prepared_fwd: _PreparedBatch) -> torch.Tensor:
        num_graphs = int(prepared_fwd.num_graphs)
        if num_graphs <= _ZERO:
            raise ValueError("Empty batch.")
        graph_mask = ~prepared_fwd.dummy_mask
        torch._assert(graph_mask.any(), "Training batch contains no valid graphs.")
        return graph_mask

    def _run_training_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: Optional[_PreparedBatch],
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        sampling_temperature: float,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.no_grad():
            rollout_fwd = self._rollout_policy(
                policy=self.policy_fwd,
                prepared=prepared_fwd,
                graph_mask=graph_mask,
                start_nodes=prepared_fwd.start_nodes_fwd,
                node_is_target=node_is_target,
                edge_ids_by_head=prepared_fwd.edge_ids_by_head_fwd,
                edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_fwd,
                record_actions=True,
                record_log_pf=False,
                temperature=sampling_temperature,
                context_tokens=prepared_fwd.context_tokens,
            )
        if rollout_fwd.actions is None:
            raise RuntimeError("Rollout actions are required for detailed balance training.")
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=rollout_fwd.actions,
            graph_mask=graph_mask,
            traj_lengths=rollout_fwd.num_moves,
            stop_reason=rollout_fwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_fwd.stop_reason == _TERMINAL_HIT) & graph_mask
        lengths = rollout_fwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        length_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        metrics = {
            **db_metrics,
            "rollout_success_rate": success.to(dtype=torch.float32).mean(),
            "rollout_length_mean": length_mean,
        }
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_fwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout",
            )
        )
        return db_loss, metrics

    def _run_backward_rollout(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: Optional[_PreparedBatch],
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        edge_index_for_dropout = prepared_fwd.edge_index if prepared_bwd is None else prepared_bwd.edge_index
        edge_mask_bwd = self._sample_pb_edge_dropout_mask(edge_index=edge_index_for_dropout)
        with torch.no_grad():
            if self._is_static_pb():
                if pb_cfg is None:
                    pb_cfg = self._resolve_pb_cfg()
                if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC and pb_distances is None:
                    raise ValueError("pb_distances required for topo_semantic pb rollout.")
                rollout_bwd = self._rollout_pb(
                    prepared=prepared_fwd,
                    dist_to_start=pb_distances,
                    graph_mask=graph_mask,
                    start_nodes=start_nodes_bwd,
                    node_is_target=node_is_start,
                    edge_ids_by_head=prepared_fwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_fwd.edge_ptr_by_head_bwd,
                    record_actions=True,
                    record_log_pf=False,
                    pb_cfg=pb_cfg,
                    edge_mask=edge_mask_bwd,
                )
            else:
                if prepared_bwd is None:
                    raise RuntimeError("prepared_bwd required for learned backward policy.")
                rollout_bwd = self._rollout_policy(
                    policy=self.policy_bwd,
                    prepared=prepared_bwd,
                    graph_mask=graph_mask,
                    start_nodes=start_nodes_bwd,
                    node_is_target=node_is_start,
                    edge_ids_by_head=prepared_bwd.edge_ids_by_head_bwd,
                    edge_ptr_by_head=prepared_bwd.edge_ptr_by_head_bwd,
                    record_actions=True,
                    record_log_pf=False,
                    temperature=float(_ONE),
                    context_tokens=prepared_bwd.context_tokens,
                    edge_mask=edge_mask_bwd,
                )
        if rollout_bwd.actions is None:
            raise RuntimeError("Backward rollout actions are required for detailed balance training.")
        actions_fwd = self._map_inverse_actions(
            actions=rollout_bwd.actions,
            edge_inverse_map=prepared_fwd.edge_inverse_map,
        )
        actions_fwd = self._reverse_actions_by_length(actions=actions_fwd, lengths=rollout_bwd.num_moves)
        db_loss, db_metrics = self._compute_db_loss(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            actions=actions_fwd,
            graph_mask=graph_mask,
            traj_lengths=rollout_bwd.num_moves,
            stop_reason=rollout_bwd.stop_reason,
            node_is_target=node_is_target,
            sampling_temperature=sampling_temperature,
            edge_mask_bwd=edge_mask_bwd,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )
        success = (rollout_bwd.stop_reason == _TERMINAL_HIT) & graph_mask
        lengths = rollout_bwd.num_moves.to(dtype=torch.float32)
        denom = graph_mask.to(dtype=lengths.dtype).sum().clamp(min=_ONE)
        length_mean = (lengths * graph_mask.to(dtype=lengths.dtype)).sum() / denom
        metrics = {
            **db_metrics,
            "rollout_bwd_success_rate": success.to(dtype=torch.float32).mean(),
            "rollout_bwd_length_mean": length_mean,
        }
        metrics.update(
            self._build_terminal_metrics(
                stop_reason=rollout_bwd.stop_reason,
                graph_mask=graph_mask,
                prefix="rollout_bwd",
            )
        )
        return db_loss, metrics

    def _aggregate_training_rollouts(
        self,
        *,
        prepared_fwd: _PreparedBatch,
        prepared_bwd: Optional[_PreparedBatch],
        graph_mask: torch.Tensor,
        node_is_target: torch.Tensor,
        node_is_start: torch.Tensor,
        start_nodes_bwd: torch.Tensor,
        sampling_temperature: float,
        num_rollouts: int,
        pb_distances: Optional[torch.Tensor] = None,
        pb_cfg: Optional[dict[str, float | int | str]] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if num_rollouts <= _ZERO:
            raise ValueError("num_rollouts must be > 0.")
        losses: list[torch.Tensor] = []
        metric_series: dict[str, list[torch.Tensor]] = {}
        for _ in range(num_rollouts):
            db_loss_fwd, metrics_fwd = self._run_training_rollout(
                prepared_fwd=prepared_fwd,
                prepared_bwd=prepared_bwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                sampling_temperature=sampling_temperature,
                pb_distances=pb_distances,
                pb_cfg=pb_cfg,
            )
            db_loss_bwd, metrics_bwd = self._run_backward_rollout(
                prepared_fwd=prepared_fwd,
                prepared_bwd=prepared_bwd,
                graph_mask=graph_mask,
                node_is_target=node_is_target,
                node_is_start=node_is_start,
                start_nodes_bwd=start_nodes_bwd,
                sampling_temperature=sampling_temperature,
                pb_distances=pb_distances,
                pb_cfg=pb_cfg,
            )
            db_loss = (db_loss_fwd + db_loss_bwd) / float(_TWO)
            metrics = self._merge_rollout_metrics(
                metrics_fwd=metrics_fwd,
                metrics_bwd=metrics_bwd,
                db_loss_fwd=db_loss_fwd,
                db_loss_bwd=db_loss_bwd,
                db_loss=db_loss,
            )
            losses.append(db_loss)
            for name, value in metrics.items():
                metric_series.setdefault(name, []).append(value)
        loss = torch.stack(losses).mean()
        averaged = {name: torch.stack(values).mean() for name, values in metric_series.items()}
        averaged["loss_total"] = loss.detach()
        return loss, averaged

    @staticmethod
    def _map_inverse_actions(*, actions: torch.Tensor, edge_inverse_map: torch.Tensor) -> torch.Tensor:
        if actions.numel() == _ZERO:
            return actions
        edge_inverse_map = edge_inverse_map.to(device=actions.device, dtype=torch.long)
        actions = actions.to(device=edge_inverse_map.device, dtype=torch.long)
        safe = actions.clamp(min=_ZERO).view(-1)
        mapped = edge_inverse_map.index_select(0, safe).view_as(actions)
        invalid = (actions >= _ZERO) & (mapped < _ZERO)
        torch._assert(~invalid.any(), "Backward rollout sampled edges without forward inverse.")
        return torch.where(actions >= _ZERO, mapped, actions)

    @staticmethod
    def _reverse_actions_by_length(*, actions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if actions.numel() == _ZERO:
            return actions
        lengths = lengths.to(device=actions.device, dtype=torch.long).view(-1)
        num_graphs, max_steps = actions.shape
        if lengths.numel() != num_graphs:
            raise ValueError("lengths length mismatch with actions batch dimension.")
        steps = torch.arange(max_steps, device=actions.device, dtype=torch.long).view(_ONE, -1)
        lengths = lengths.clamp(min=_ZERO, max=max_steps).view(-1, _ONE)
        idx = torch.where(steps < lengths, lengths - _ONE - steps, steps).expand(num_graphs, -1)
        return actions.gather(1, idx)

    @staticmethod
    def _merge_rollout_metrics(
        *,
        metrics_fwd: dict[str, torch.Tensor],
        metrics_bwd: dict[str, torch.Tensor],
        db_loss_fwd: torch.Tensor,
        db_loss_bwd: torch.Tensor,
        db_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        merged = dict(metrics_fwd)
        for name, value in metrics_bwd.items():
            if name in merged and name.startswith("db_"):
                merged[name] = (merged[name] + value) / float(_TWO)
            else:
                merged[name] = value
        merged.pop("db_loss", None)
        merged["db_loss_fwd"] = db_loss_fwd.detach()
        merged["db_loss_bwd"] = db_loss_bwd.detach()
        merged["db_loss"] = db_loss.detach()
        return merged

    def _apply_target_roulette(
        self,
        *,
        batch: Any,
        prepared: _PreparedBatch,
        target_nodes: torch.Tensor,
    ) -> _PreparedBatch:
        num_graphs = int(prepared.num_graphs)
        if target_nodes.numel() != num_graphs:
            raise ValueError("target_nodes length mismatch with batch graph count.")
        question_emb = getattr(batch, "question_emb", None)
        if not torch.is_tensor(question_emb):
            raise AttributeError("Batch missing question_emb required for target roulette.")
        question_emb = self._ensure_tensor(
            question_emb, device=prepared.node_tokens.device, non_blocking=True
        )
        question_tokens = self._resolve_context_tokens(
            self.backbone_bwd.project_question_embeddings(question_emb)
        )
        target_tokens = self._build_anchor_tokens(node_tokens=prepared.node_tokens, node_ids=target_nodes)
        context_tokens = self._build_backward_context(
            question_tokens=question_tokens,
            start_tokens=prepared.start_tokens_bwd,
            answer_tokens=target_tokens,
        )
        return replace(prepared, context_tokens=context_tokens)

    def _compute_training_loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        build_bwd = not self._is_static_pb()
        prepared_fwd, prepared_bwd = self._prepare_batch(batch, build_bwd=build_bwd)
        graph_mask = self._validate_training_batch(prepared_fwd)
        num_nodes_total = int(prepared_fwd.num_nodes_total)
        start_nodes_bwd = self._sample_nodes_uniform(
            local_indices=prepared_fwd.a_local_indices,
            ptr=prepared_fwd.a_ptr,
            allow_empty=True,
            name="a_local_indices",
        )
        node_is_target = self._build_node_mask(num_nodes_total, prepared_fwd.a_local_indices)
        node_is_start = self._build_node_mask(num_nodes_total, prepared_fwd.q_local_indices)
        pb_cfg = None
        pb_distances = None
        if self._is_static_pb():
            pb_cfg = self._resolve_pb_cfg()
            if pb_cfg["mode"] == _PB_MODE_TOPO_SEMANTIC:
                precomputed = getattr(batch, "dist_to_start", None)
                if torch.is_tensor(precomputed):
                    precomputed = precomputed.to(device=prepared_fwd.edge_index.device, dtype=torch.long).view(-1)
                    if precomputed.numel() != num_nodes_total:
                        raise ValueError("dist_to_start length mismatch with num_nodes_total.")
                    pb_distances = precomputed
                else:
                    with torch.no_grad():
                        pb_distances = self._compute_distance_to_starts(
                            prepared=prepared_fwd,
                            max_hops=int(pb_cfg["max_hops"]),
                        )
        else:
            if prepared_bwd is None:
                raise RuntimeError("prepared_bwd required for learned backward policy.")
            prepared_bwd = self._apply_target_roulette(
                batch=batch,
                prepared=prepared_bwd,
                target_nodes=start_nodes_bwd,
            )
        sampling_temperature = self._resolve_sampling_temperature()
        num_rollouts = self._resolve_num_rollouts()
        return self._aggregate_training_rollouts(
            prepared_fwd=prepared_fwd,
            prepared_bwd=prepared_bwd,
            graph_mask=graph_mask,
            node_is_target=node_is_target,
            node_is_start=node_is_start,
            start_nodes_bwd=start_nodes_bwd,
            sampling_temperature=sampling_temperature,
            num_rollouts=num_rollouts,
            pb_distances=pb_distances,
            pb_cfg=pb_cfg,
        )

    @staticmethod
    def _ensure_loss_requires_grad(loss: torch.Tensor) -> torch.Tensor:
        if loss.requires_grad:
            return loss
        return loss + torch.zeros((), device=loss.device, dtype=loss.dtype, requires_grad=True)

    def _collect_logit_scale_metrics(self) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        for prefix, policy in (("fwd", self.policy_fwd), ("bwd", self.policy_bwd)):
            logit_scale = getattr(policy, "logit_scale", None)
            if logit_scale is None:
                continue
            scale = logit_scale.exp()
            scale = scale.clamp(min=policy.logit_scale_min, max=policy.logit_scale_max)
            metrics[f"logit_scale_{prefix}"] = scale.detach()
        if metrics:
            metrics["logit_scale_max"] = torch.stack(list(metrics.values())).max()
        return metrics


__all__ = ["DualFlowRolloutMixin"]
