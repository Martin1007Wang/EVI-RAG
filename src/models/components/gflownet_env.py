from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import logging

import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max


@dataclass
class GraphBatch:
    edge_index: torch.Tensor          # [2, E_total], local node idx already offset by PyG Batch
    edge_batch: torch.Tensor          # [E_total], graph id per edge
    node_global_ids: torch.Tensor     # [N_total]
    heads_global: torch.Tensor        # [E_total], cached global head ids
    tails_global: torch.Tensor        # [E_total], cached global tail ids
    edge_scores_z: torch.Tensor       # [E_total], per-graph z-scored edge scores
    node_ptr: torch.Tensor            # [B+1], prefix sum of nodes per graph
    edge_ptr: torch.Tensor            # [B+1], prefix sum of edges per graph
    start_node_locals: torch.Tensor   # [S_total] local node idx (offset per graph)
    start_ptr: torch.Tensor           # [B+1], prefix sum of start nodes
    answer_node_locals: torch.Tensor  # [A_total] local node idx
    answer_ptr: torch.Tensor          # [B+1], prefix sum of answer nodes
    node_is_start: torch.Tensor       # [N_total] bool
    node_is_answer: torch.Tensor      # [N_total] bool
    edge_starts_mask: torch.Tensor        # [E_total] bool, edges touching any start node
    directed: bool                   # whether traversal is directed along edge_index


@dataclass
class GraphState:
    graph: GraphBatch
    selected_mask: torch.Tensor       # [E_total] bool
    visited_nodes: torch.Tensor       # [N_total] bool
    selection_order: torch.Tensor     # [E_total] long
    current_tail: torch.Tensor        # [B] global entity id
    prev_tail: torch.Tensor           # [B] global entity id
    current_tail_local: torch.Tensor  # [B] batch-global node index (edge_index space)
    prev_tail_local: torch.Tensor     # [B] batch-global node index (edge_index space)
    done: torch.Tensor                # [B] bool
    step_counts: torch.Tensor         # [B] long
    actions: torch.Tensor             # [B, max_steps + 1] long (edge idx within slice or stop idx)
    answer_hits: torch.Tensor         # [B] bool
    debug_logged: torch.Tensor        # [B] bool, used only when debug=True to throttle logs
    debug_enabled: bool               # whether this state should emit debug logs


class GraphEnv(nn.Module):
    """Graph environment on PyG flat batch (no dense padding)."""

    def __init__(
        self,
        max_steps: int,
        mode: str = "path",
        forbid_backtrack: bool = True,
        forbid_revisit: bool = True,
        directed: bool = True,
        debug: bool = False,
        debug_max_resets: int = 0,
        debug_max_graphs: int = 1,
        debug_max_hits: int = 1,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        mode = str(mode).lower()
        if mode != "path":
            raise ValueError(f"GraphEnv supports only mode='path' in the final model, got {mode!r}.")
        self.mode = "path"
        self.forbid_backtrack = bool(forbid_backtrack)
        self.forbid_revisit = bool(forbid_revisit)
        self.directed = bool(directed)
        self.debug = bool(debug)
        self.debug_max_resets = max(int(debug_max_resets), 0)
        self.debug_max_graphs = max(int(debug_max_graphs), 1)
        self.debug_max_hits = max(int(debug_max_hits), 1)
        self._debug_resets_logged = 0
        self._debug_hits_logged = 0

    def reset(self, batch: Dict[str, torch.Tensor], *, device: torch.device) -> GraphState:
        edge_index = batch["edge_index"].to(device)
        edge_batch = batch["edge_batch"].to(device)
        node_global_ids = batch["node_global_ids"].to(device)
        node_ptr = batch["node_ptr"].to(device)
        edge_ptr = batch["edge_ptr"].to(device)
        if "edge_scores" not in batch:
            raise ValueError("edge_scores missing in graph batch; g_agent cache must provide per-edge scores.")
        edge_scores = batch["edge_scores"].to(device=device, dtype=torch.float32).view(-1)
        start_node_locals = batch["start_node_locals"].to(device)
        start_ptr = batch["start_ptr"].to(device)
        answer_node_locals = batch["answer_node_locals"].to(device)
        answer_ptr = batch["answer_ptr"].to(device)

        num_edges = edge_index.size(1)
        num_graphs = int(node_ptr.numel() - 1)

        start_counts = start_ptr[1:] - start_ptr[:-1]
        if start_counts.numel() != num_graphs:
            raise ValueError("start_ptr has inconsistent length; ensure g_agent cache uses PyG slicing.")
        if (start_counts < 0).any():
            raise ValueError("start_ptr must be non-decreasing.")
        missing_start = start_counts == 0

        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]
        if edge_scores.numel() != num_edges:
            raise ValueError(f"edge_scores length {edge_scores.numel()} != num_edges {num_edges}")
        edge_scores_z = self._zscore_edges(edge_scores=edge_scores, edge_batch=edge_batch, num_graphs=num_graphs)
        selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        selection_order = torch.full((num_edges,), -1, dtype=torch.long, device=device)
        current_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        prev_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        current_tail_local = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        prev_tail_local = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        done = missing_start.to(device=device)
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        actions = torch.full((num_graphs, self.max_steps + 1), -1, dtype=torch.long, device=device)
        # Debug 控制：仅首图打印一次 ENV_HIT_DEBUG
        debug_logged = torch.zeros(num_graphs, dtype=torch.bool, device=device)

        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= 0:
            raise ValueError("GraphEnv.reset received empty node_ptr; g_agent cache may be corrupt.")
        if start_node_locals.numel() > 0 and (start_node_locals < 0).any():
            raise ValueError("start_node_locals contains negative indices; g_agent cache must store valid node indices.")
        if start_node_locals.numel() > 0 and int(start_node_locals.max().item()) >= num_nodes_total:
            raise ValueError("start_node_locals contains out-of-range indices relative to node_ptr.")
        if answer_node_locals.numel() > 0 and (answer_node_locals < 0).any():
            raise ValueError("answer_node_locals contains negative indices; g_agent cache must store valid node indices.")
        if answer_node_locals.numel() > 0 and int(answer_node_locals.max().item()) >= num_nodes_total:
            raise ValueError("answer_node_locals contains out-of-range indices relative to node_ptr.")

        self._validate_locals_per_graph(
            locals_tensor=start_node_locals,
            ptr=start_ptr,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            name="start_node_locals",
        )
        self._validate_locals_per_graph(
            locals_tensor=answer_node_locals,
            ptr=answer_ptr,
            node_ptr=node_ptr,
            num_graphs=num_graphs,
            name="answer_node_locals",
        )

        node_is_start = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals.long()] = True
        node_is_answer = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals.long()] = True
        answer_hits = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        if start_node_locals.numel() > 0:
            start_batch = torch.repeat_interleave(torch.arange(num_graphs, device=device), start_counts.clamp(min=0))
            start_is_answer = node_is_answer[start_node_locals.long()].float()
            answer_hits = torch.bincount(start_batch, weights=start_is_answer, minlength=num_graphs) > 0
        # Directed traversal: step0 only allows outgoing edges from start nodes.
        if self.directed:
            edge_starts_mask = node_is_start[edge_index[0]]
        else:
            # Undirected traversal: step0 allows any incident edge.
            edge_starts_mask = node_is_start[edge_index[0]] | node_is_start[edge_index[1]]

        should_debug = bool(self.debug) and (self._debug_resets_logged < self.debug_max_resets)
        if should_debug:
            self._debug_resets_logged += 1
            try:
                logging.getLogger("gflownet.debug").info(
                    "[ENV_DEBUG] debug=True num_graphs=%d num_edges=%d", num_graphs, num_edges
                )
            except Exception:
                pass

        graph = GraphBatch(
            edge_index=edge_index,
            edge_batch=edge_batch,
            node_global_ids=node_global_ids,
            heads_global=heads_global,
            tails_global=tails_global,
            edge_scores_z=edge_scores_z,
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
            answer_node_locals=answer_node_locals,
            answer_ptr=answer_ptr,
            node_is_start=node_is_start,
            node_is_answer=node_is_answer,
            edge_starts_mask=edge_starts_mask,
            directed=self.directed,
        )
        return GraphState(
            graph=graph,
            selected_mask=selected_mask,
            visited_nodes=node_is_start.clone(),
            selection_order=selection_order,
            current_tail=current_tail,
            prev_tail=prev_tail,
            current_tail_local=current_tail_local,
            prev_tail_local=prev_tail_local,
            done=done,
            step_counts=step_counts,
            actions=actions,
            answer_hits=answer_hits,
            debug_logged=debug_logged,
            debug_enabled=should_debug,
        )

    @staticmethod
    def _zscore_edges(
        *,
        edge_scores: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        edge_scores = edge_scores.to(dtype=torch.float32)
        counts = torch.bincount(edge_batch, minlength=num_graphs).to(dtype=edge_scores.dtype).clamp(min=1.0)
        sum_scores = torch.zeros(num_graphs, device=edge_scores.device, dtype=edge_scores.dtype)
        sum_scores.index_add_(0, edge_batch, edge_scores)
        sum_sq = torch.zeros(num_graphs, device=edge_scores.device, dtype=edge_scores.dtype)
        sum_sq.index_add_(0, edge_batch, edge_scores * edge_scores)
        mean = sum_scores / counts
        var = (sum_sq / counts) - mean * mean
        std = torch.sqrt(var.clamp(min=0.0))
        eps = torch.finfo(edge_scores.dtype).eps
        std = torch.where(std > eps, std, torch.ones_like(std))
        return (edge_scores - mean[edge_batch]) / std[edge_batch]

    def potential(self, state: GraphState, *, valid_edges_override: torch.Tensor | None = None) -> torch.Tensor:
        """Potential based on global (per-graph) z-scored edge scores over legal actions.

        Args:
            valid_edges_override: Optional extra mask over edges (shape [E_total]) to further restrict
                which actions are considered legal for the potential. This is useful when the actor
                prunes the action set (e.g., Top-K) and we want φ(s) to reflect that same action set.
        """
        edge_scores_z = state.graph.edge_scores_z
        edge_batch = state.graph.edge_batch
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        if num_graphs <= 0:
            return torch.zeros(0, device=edge_scores_z.device, dtype=edge_scores_z.dtype)

        valid = self.action_mask_edges(state)
        if valid_edges_override is not None:
            valid_edges_override = valid_edges_override.to(device=edge_scores_z.device, dtype=torch.bool).view(-1)
            if valid_edges_override.shape != valid.shape:
                raise ValueError(
                    f"valid_edges_override shape {tuple(valid_edges_override.shape)} != action_mask_edges {tuple(valid.shape)}"
                )
            valid = valid & valid_edges_override
        valid = valid & (~state.selected_mask)
        if state.done.any():
            valid = valid & (~state.done[edge_batch])
        if not bool(valid.any().item()):
            return torch.zeros(num_graphs, device=edge_scores_z.device, dtype=edge_scores_z.dtype)

        valid_counts = torch.bincount(edge_batch[valid], minlength=num_graphs).to(dtype=edge_scores_z.dtype)
        scores = edge_scores_z.to(dtype=torch.float32)
        neg_inf = torch.finfo(scores.dtype).min
        masked = scores.masked_fill(~valid, neg_inf)
        max_per_graph, _ = scatter_max(masked, edge_batch, dim=0, dim_size=num_graphs)
        max_per_graph = torch.where(valid_counts > 0, max_per_graph, torch.zeros_like(max_per_graph))
        exp_sum = scatter_add(
            torch.exp(masked - max_per_graph[edge_batch]) * valid.to(dtype=scores.dtype),
            edge_batch,
            dim=0,
            dim_size=num_graphs,
        )
        eps = torch.finfo(scores.dtype).tiny
        logsumexp = max_per_graph + torch.log(exp_sum.clamp(min=eps))
        logmeanexp = logsumexp - torch.log(valid_counts.clamp(min=1.0))
        phi = torch.where(valid_counts > 0, logmeanexp, torch.zeros_like(logmeanexp))
        return phi.to(dtype=edge_scores_z.dtype)

    @staticmethod
    def _validate_locals_per_graph(
        *,
        locals_tensor: torch.Tensor,
        ptr: torch.Tensor,
        node_ptr: torch.Tensor,
        num_graphs: int,
        name: str,
    ) -> None:
        counts = ptr.view(-1)[1:] - ptr.view(-1)[:-1]
        total = int(counts.sum().item()) if counts.numel() > 0 else 0
        if total != int(locals_tensor.numel()):
            raise ValueError(
                f"{name} count mismatch: ptr_sum={total} vs locals_numel={int(locals_tensor.numel())}"
            )
        if total == 0 or num_graphs <= 0:
            return
        batch_ids = torch.repeat_interleave(torch.arange(num_graphs, device=ptr.device), counts.clamp(min=0))
        lower = node_ptr[batch_ids]
        upper = node_ptr[batch_ids + 1]
        invalid = (locals_tensor < lower) | (locals_tensor >= upper)
        if not bool(invalid.any().item()):
            return
        bad_idx = torch.nonzero(invalid, as_tuple=False).view(-1)
        preview = []
        for i in bad_idx[:5].tolist():
            preview.append(
                f"(i={i} val={int(locals_tensor[i].item())} range=[{int(lower[i].item())},{int(upper[i].item())}))"
            )
        raise ValueError(f"{name} contains cross-graph indices: {', '.join(preview)}")

    def action_mask_edges(self, state: GraphState) -> torch.Tensor:
        edge_batch = state.graph.edge_batch
        if not hasattr(state.graph, "edge_starts_mask"):
            raise ValueError("edge_starts_mask missing in GraphBatch; ensure GraphEnv.reset precomputes it.")

        # Hard horizon: once a graph has selected >= max_steps edges, only stop is legal.
        horizon_exhausted = state.step_counts[edge_batch] >= self.max_steps

        heads_global = state.graph.heads_global
        tails_global = state.graph.tails_global
        is_step0 = state.step_counts[edge_batch] == 0
        current_tail = state.current_tail[edge_batch]

        if state.graph.directed:
            # Directed traversal: only follow head -> tail.
            head_match = heads_global == current_tail
            valid_next = head_match
            next_local = state.graph.edge_index[1]
        else:
            # Undirected traversal: current node can match head or tail.
            head_match = heads_global == current_tail
            tail_match = tails_global == current_tail
            valid_next = head_match | tail_match

            # 下一节点（局部索引）由匹配端点确定：head_match -> tail，否则 tail_match -> head。
            # 对非 incident 边该值无意义（会被 valid_next 屏蔽）。
            next_local = torch.where(head_match, state.graph.edge_index[1], state.graph.edge_index[0])

        if self.forbid_revisit:
            visited = state.visited_nodes[next_local]
            valid_next = valid_next & (~visited)

        if self.forbid_backtrack:
            prev_tail = state.prev_tail[edge_batch]
            next_global = tails_global if state.graph.directed else torch.where(head_match, tails_global, heads_global)
            is_backtrack = next_global == prev_tail
            valid_next = valid_next & (~is_backtrack)

        edge_mask = torch.where(is_step0, state.graph.edge_starts_mask, valid_next)
        return edge_mask & (~horizon_exhausted)

    def frontier_mask_edges(self, state: GraphState) -> torch.Tensor:
        """返回策略前沿边掩码（不含重复边），用于边级 bonus。"""
        edge_batch = state.graph.edge_batch
        horizon_exhausted = state.step_counts[edge_batch] >= self.max_steps
        is_step0 = state.step_counts[edge_batch] == 0
        heads = state.graph.heads_global
        tails = state.graph.tails_global
        current_tail = state.current_tail[edge_batch]
        if state.graph.directed:
            frontier = heads == current_tail
        else:
            frontier = (heads == current_tail) | (tails == current_tail)
        frontier = torch.where(is_step0, state.graph.edge_starts_mask, frontier)
        frontier = frontier & (~state.selected_mask)
        return frontier & (~horizon_exhausted)

    def step(self, state: GraphState, actions: torch.Tensor, *, step_index: int) -> GraphState:
        device = state.graph.edge_index.device
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        edge_ptr = state.graph.edge_ptr
        es = edge_ptr[:-1]
        ee = edge_ptr[1:]
        stop_idx = ee  # virtual stop at end of slice

        if actions.numel() != num_graphs:
            raise ValueError(f"Actions length {actions.numel()} != num_graphs {num_graphs}")

        invalid_low = actions < es
        invalid_high = actions > ee
        if invalid_low.any() or invalid_high.any():
            bad = torch.nonzero(invalid_low | invalid_high, as_tuple=False).view(-1)
            # Provide the first few offending graphs for deterministic debugging.
            preview = []
            for idx in bad[:5].tolist():
                preview.append(
                    f"(g={idx} action={int(actions[idx].item())} es={int(es[idx].item())} ee={int(ee[idx].item())})"
                )
            raise ValueError(
                "Actions contain out-of-range indices for per-graph edge slices; "
                f"min_diff={int((actions - es).min().item())} "
                f"max_diff={int((actions - ee).max().item())} "
                f"bad_examples={', '.join(preview)}"
            )

        is_stop = (actions == stop_idx) | state.done
        edge_mask_valid = self.action_mask_edges(state)
        edge_actions_mask = ~is_stop
        edge_indices = actions[edge_actions_mask]

        if edge_indices.numel() > 0:
            if (edge_indices < 0).any() or (edge_indices >= state.graph.edge_index.size(1)).any():
                raise ValueError("Edge actions contain out-of-range indices for current batch.")
            if not edge_mask_valid[edge_indices].all():
                raise ValueError("Edge actions violate action mask; check actor sampling or environment constraints.")

        # State tensors are non-differentiable (bool/long); mutate in-place to avoid per-step clones.
        selected_mask = state.selected_mask
        selection_order = state.selection_order
        current_tail = state.current_tail
        prev_tail = state.prev_tail
        current_tail_local = state.current_tail_local
        prev_tail_local = state.prev_tail_local
        actions_record = state.actions
        answer_hits = state.answer_hits
        visited_nodes = state.visited_nodes

        if edge_indices.numel() > 0:
            selected_mask[edge_indices] = True
            selection_order[edge_indices] = step_index

            graph_ids = state.graph.edge_batch[edge_indices]
            heads_idx = state.graph.edge_index[0, edge_indices]
            tails_idx = state.graph.edge_index[1, edge_indices]

            is_step0 = state.step_counts[graph_ids] == 0

            # Step0：从 start 节点出发（有向时仅允许 head==start）；非 step0：从 current_tail 出发。
            head_is_start = state.graph.node_is_start[heads_idx]
            tail_is_start = state.graph.node_is_start[tails_idx]
            if state.graph.directed:
                src_local_step0 = heads_idx
                dst_local_step0 = tails_idx
            else:
                src_local_step0 = torch.where(head_is_start, heads_idx, tails_idx)
                src_is_head_step0 = src_local_step0 == heads_idx
                dst_local_step0 = torch.where(src_is_head_step0, tails_idx, heads_idx)

            src_local = torch.where(is_step0, src_local_step0, current_tail_local[graph_ids])
            src_is_head = heads_idx == src_local
            src_is_tail = tails_idx == src_local
            if state.graph.directed:
                if bool((~is_step0 & ~src_is_head).any().item()):
                    raise ValueError("Directed action is not incident to current head; action_mask_edges must prevent this.")
            else:
                if bool((~is_step0 & ~(src_is_head | src_is_tail)).any().item()):
                    raise ValueError("Edge action is not incident to current tail; action_mask_edges must prevent this.")
            dst_local_non0 = torch.where(src_is_head, tails_idx, heads_idx)
            dst_local = torch.where(is_step0, dst_local_step0, dst_local_non0)

            src_global = state.graph.node_global_ids[src_local]
            dst_global = state.graph.node_global_ids[dst_local]

            prev_tail[graph_ids] = src_global
            current_tail[graph_ids] = dst_global
            prev_tail_local[graph_ids] = src_local
            current_tail_local[graph_ids] = dst_local
            visited_nodes[dst_local] = True
            hit = state.graph.node_is_answer[dst_local]

            write_pos = state.step_counts[graph_ids].clamp(max=actions_record.size(1) - 1)
            actions_record[graph_ids, write_pos] = edge_indices

            answer_hits[graph_ids] = answer_hits[graph_ids] | hit

            if state.debug_enabled and getattr(self, "debug", False):
                graphs_with_selection = graph_ids.unique()
                for gid in graphs_with_selection.tolist():
                    if gid >= self.debug_max_graphs or state.debug_logged[gid]:
                        continue
                    if self._debug_hits_logged >= self.debug_max_hits:
                        continue
                    mask_gid = graph_ids == gid
                    sel_next = dst_local[mask_gid]
                    sel_next_global = state.graph.node_global_ids[sel_next]
                    answers_start = int(state.graph.answer_ptr[gid].item())
                    answers_end = int(state.graph.answer_ptr[gid + 1].item())
                    answers_local = state.graph.answer_node_locals[answers_start:answers_end]
                    answers = state.graph.node_global_ids[answers_local]
                    try:
                        logging.getLogger("gflownet.debug").info(
                            "[ENV_HIT_DEBUG] graph=%d hit=%s answers=%s selected_nodes=%s",
                            gid,
                            bool(hit[mask_gid].any().item()),
                            answers.detach().cpu().tolist(),
                            sel_next_global.detach().cpu().tolist(),
                        )
                        state.debug_logged[gid] = True
                        self._debug_hits_logged += 1
                    except Exception:
                        pass

        done = state.done | is_stop
        state.done = done
        state.step_counts.add_((~done).to(dtype=state.step_counts.dtype))
        return state


__all__ = ["GraphEnv", "GraphBatch", "GraphState"]
