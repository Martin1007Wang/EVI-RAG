from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import logging

import torch
from torch import nn


@dataclass
class GraphBatch:
    edge_index: torch.Tensor          # [2, E_total], local node idx already offset by PyG Batch
    edge_batch: torch.Tensor          # [E_total], graph id per edge
    node_global_ids: torch.Tensor     # [N_total]
    heads_global: torch.Tensor        # [E_total], cached global head ids
    tails_global: torch.Tensor        # [E_total], cached global tail ids
    node_ptr: torch.Tensor            # [B+1], prefix sum of nodes per graph
    edge_ptr: torch.Tensor            # [B+1], prefix sum of edges per graph
    start_node_locals: torch.Tensor   # [S_total] local node idx (offset per graph)
    start_ptr: torch.Tensor           # [B+1], prefix sum of start nodes
    start_entity_ids: torch.Tensor    # [S_total] global ids
    start_entity_ptr: torch.Tensor    # [B+1]
    answer_node_locals: torch.Tensor  # [A_total] local node idx
    answer_ptr: torch.Tensor          # [B+1], prefix sum of answer nodes
    answer_entity_ids: torch.Tensor   # [A_total] global ids (aligned with answer_ptr)
    node_is_start: torch.Tensor       # [N_total] bool
    node_is_answer: torch.Tensor      # [N_total] bool
    edge_relations: torch.Tensor      # [E_total]
    edge_labels: torch.Tensor         # [E_total]
    is_answer_reachable: torch.Tensor     # [B] bool
    edge_starts_mask: torch.Tensor        # [E_total] bool, edges touching any start node


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
        start_node_locals = batch["start_node_locals"].to(device)
        start_ptr = batch["start_ptr"].to(device)
        start_entity_ids = batch.get("start_entity_ids", torch.empty(0, dtype=torch.long, device=device)).to(device)
        start_entity_ptr = batch.get("start_entity_ptr", torch.zeros_like(start_ptr)).to(device)
        answer_node_locals = batch["answer_node_locals"].to(device)
        answer_ptr = batch["answer_ptr"].to(device)
        answer_entity_ids = batch["answer_entity_ids"].to(device)
        edge_relations = batch["edge_relations"].to(device)
        edge_labels = batch["edge_labels"].to(device)
        is_answer_reachable = batch["is_answer_reachable"].to(device).bool()

        num_edges = edge_index.size(1)
        num_graphs = int(node_ptr.numel() - 1)

        start_counts = start_ptr[1:] - start_ptr[:-1]
        if start_counts.numel() != num_graphs:
            raise ValueError("start_ptr has inconsistent length; ensure g_agent cache uses PyG slicing.")
        if (start_counts <= 0).any():
            missing = torch.nonzero(start_counts <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(
                f"GraphEnv.reset received graphs without start_node_locals at indices {missing}; "
                "g_agent cache must materialize non-empty start anchors per graph instead of relying on runtime inference."
            )

        heads_global = node_global_ids[edge_index[0]]
        tails_global = node_global_ids[edge_index[1]]
        selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        selection_order = torch.full((num_edges,), -1, dtype=torch.long, device=device)
        current_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        prev_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        current_tail_local = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        prev_tail_local = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        done = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        actions = torch.full((num_graphs, self.max_steps + 1), -1, dtype=torch.long, device=device)
        answer_hits = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        # Debug 控制：仅首图打印一次 ENV_HIT_DEBUG
        debug_logged = torch.zeros(num_graphs, dtype=torch.bool, device=device)

        num_nodes_total = int(node_ptr[-1].item())
        if num_nodes_total <= 0:
            raise ValueError("GraphEnv.reset received empty node_ptr; g_agent cache may be corrupt.")
        if start_node_locals.numel() > 0 and (start_node_locals < 0).any():
            raise ValueError("start_node_locals contains negative indices; g_agent cache must store valid node indices.")
        if start_node_locals.numel() > 0 and int(start_node_locals.max().item()) >= num_nodes_total:
            raise ValueError("start_node_locals contains out-of-range indices relative to node_ptr.")

        node_is_start = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if start_node_locals.numel() > 0:
            node_is_start[start_node_locals.long()] = True
        node_is_answer = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
        if answer_node_locals.numel() > 0:
            node_is_answer[answer_node_locals.long()] = True
        # 有向模式：仅允许从 start 节点出发的正向边
        edge_starts_mask = node_is_start[edge_index[0]]

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
            node_ptr=node_ptr,
            edge_ptr=edge_ptr,
            start_node_locals=start_node_locals,
            start_ptr=start_ptr,
            start_entity_ids=start_entity_ids,
            start_entity_ptr=start_entity_ptr,
            answer_node_locals=answer_node_locals,
            answer_ptr=answer_ptr,
            answer_entity_ids=answer_entity_ids,
            node_is_start=node_is_start,
            node_is_answer=node_is_answer,
            edge_relations=edge_relations,
            edge_labels=edge_labels,
            is_answer_reachable=is_answer_reachable,
            edge_starts_mask=edge_starts_mask,
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
        # 有向：仅允许从当前尾节点作为 head 出发
        valid_next = heads_global == current_tail
        next_local = state.graph.edge_index[1]

        if self.forbid_revisit:
            visited = state.visited_nodes[next_local]
            valid_next = valid_next & (~visited)

        if self.forbid_backtrack:
            prev_tail = state.prev_tail[edge_batch]
            is_backtrack = tails_global == prev_tail
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
        frontier = heads == current_tail
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
            heads = state.graph.heads_global[edge_indices]
            tails = state.graph.tails_global[edge_indices]
            next_local = tails_idx
            next_global = state.graph.node_global_ids[next_local]

            is_step0 = state.step_counts[graph_ids] == 0
            prev_tail[graph_ids] = torch.where(is_step0, heads, state.current_tail[graph_ids])
            current_tail[graph_ids] = next_global
            prev_tail_local[graph_ids] = torch.where(is_step0, heads_idx, current_tail_local[graph_ids])
            current_tail_local[graph_ids] = next_local
            visited_nodes[next_local] = True
            hit = state.graph.node_is_answer[next_local]

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
                    sel_next = next_local[mask_gid]
                    sel_next_global = state.graph.node_global_ids[sel_next]
                    answers_start = int(state.graph.answer_ptr[gid].item())
                    answers_end = int(state.graph.answer_ptr[gid + 1].item())
                    answers = state.graph.answer_entity_ids[answers_start:answers_end]
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
