from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import logging

import torch
from torch import nn


@dataclass
class GraphBatch:
    edge_index: torch.Tensor          # [2, E_total], local node idx already offset by PyG Batch
    edge_batch: torch.Tensor          # [E_total], graph id per edge
    node_global_ids: torch.Tensor     # [N_total]
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
    top_edge_mask: torch.Tensor       # [E_total] bool
    gt_path_edge_local_ids: torch.Tensor  # [P_total] local edge idx (offset per graph)
    gt_edge_ptr: torch.Tensor             # [B+1]
    gt_path_exists: torch.Tensor          # [B] bool
    is_answer_reachable: torch.Tensor     # [B] bool
    edge_starts_mask: torch.Tensor        # [E_total] bool, edges touching any start node
    bypass_action_mask: torch.Tensor      # [B] bool, allow GT replay to skip mask checks


@dataclass
class GraphState:
    graph: GraphBatch
    selected_mask: torch.Tensor       # [E_total] bool
    visited_nodes: torch.Tensor       # [N_total] bool
    selection_order: torch.Tensor     # [E_total] long
    current_tail: torch.Tensor        # [B] global entity id
    prev_tail: torch.Tensor           # [B] global entity id
    done: torch.Tensor                # [B] bool
    step_counts: torch.Tensor         # [B] long
    actions: torch.Tensor             # [B, max_steps + 1] long (edge idx within slice or stop idx)
    direction: Optional[torch.Tensor] # unused in current policies
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
        bidir_token: bool = False,
        debug: bool = False,
        debug_max_resets: int = 0,
        debug_max_graphs: int = 1,
        debug_max_hits: int = 1,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        mode = str(mode).lower()
        if mode not in ("path", "subgraph"):
            raise ValueError(f"GraphEnv.mode must be 'path' or 'subgraph', got {mode}")
        self.mode = mode
        self.forbid_backtrack = bool(forbid_backtrack)
        self.forbid_revisit = bool(forbid_revisit)
        self.bidir_token = bool(bidir_token)
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
        top_edge_mask = batch["top_edge_mask"].to(device).bool()
        gt_path_edge_local_ids = batch["gt_path_edge_local_ids"].to(device)
        gt_edge_ptr = batch["gt_edge_ptr"].to(device)
        gt_path_exists = batch["gt_path_exists"].to(device).bool()
        is_answer_reachable = batch["is_answer_reachable"].to(device).bool()
        bypass_action_mask = batch.get("bypass_action_mask", torch.zeros_like(gt_path_exists)).to(device).bool()

        num_edges = edge_index.size(1)
        num_graphs = int(node_ptr.numel() - 1)

        if getattr(self, "debug", False):
            try:
                logging.getLogger("gflownet.debug").info(
                    "[ENV_DEBUG] debug=True num_graphs=%d num_edges=%d", num_graphs, num_edges
                )
            except Exception:
                pass

        start_counts = start_ptr[1:] - start_ptr[:-1]
        if start_counts.numel() != num_graphs:
            raise ValueError("start_ptr has inconsistent length; ensure g_agent cache uses PyG slicing.")
        if (start_counts <= 0).any():
            missing = torch.nonzero(start_counts <= 0, as_tuple=False).view(-1).cpu().tolist()
            raise ValueError(
                f"GraphEnv.reset received graphs without start_node_locals at indices {missing}; "
                "g_agent cache must materialize non-empty start anchors per graph instead of relying on runtime inference."
            )

        selected_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
        selection_order = torch.full((num_edges,), -1, dtype=torch.long, device=device)
        current_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        prev_tail = torch.full((num_graphs,), -1, dtype=torch.long, device=device)
        done = torch.zeros(num_graphs, dtype=torch.bool, device=device)
        step_counts = torch.zeros(num_graphs, dtype=torch.long, device=device)
        actions = torch.full((num_graphs, self.max_steps + 1), -1, dtype=torch.long, device=device)
        direction = torch.zeros(num_edges, 1, device=device) if self.bidir_token else None
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
            top_edge_mask=top_edge_mask,
            gt_path_edge_local_ids=gt_path_edge_local_ids,
            gt_edge_ptr=gt_edge_ptr,
            gt_path_exists=gt_path_exists,
            is_answer_reachable=is_answer_reachable,
            edge_starts_mask=edge_starts_mask,
            bypass_action_mask=bypass_action_mask,
        )
        return GraphState(
            graph=graph,
            selected_mask=selected_mask,
            visited_nodes=node_is_start.clone(),
            selection_order=selection_order,
            current_tail=current_tail,
            prev_tail=prev_tail,
            done=done,
            step_counts=step_counts,
            actions=actions,
            direction=direction,
            answer_hits=answer_hits,
            debug_logged=debug_logged,
            debug_enabled=should_debug,
        )

    def action_mask_edges(self, state: GraphState) -> torch.Tensor:
        edge_batch = state.graph.edge_batch
        if not hasattr(state.graph, "edge_starts_mask"):
            raise ValueError("edge_starts_mask missing in GraphBatch; ensure GraphEnv.reset precomputes it.")

        heads_global = state.graph.node_global_ids[state.graph.edge_index[0]]
        tails_global = state.graph.node_global_ids[state.graph.edge_index[1]]
        is_step0 = state.step_counts[edge_batch] == 0

        if self.mode == "path":
            current_tail = state.current_tail[edge_batch]
            valid_next = (heads_global == current_tail) | (tails_global == current_tail)

            next_local = torch.where(
                heads_global == current_tail,
                state.graph.edge_index[1],
                state.graph.edge_index[0],
            )

            if self.forbid_revisit:
                visited = state.visited_nodes[next_local]
                valid_next = valid_next & (~visited)

            if self.forbid_backtrack:
                prev_tail = state.prev_tail[edge_batch]
                is_backtrack = (
                    ((heads_global == current_tail) & (tails_global == prev_tail))
                    | ((tails_global == current_tail) & (heads_global == prev_tail))
                )
                valid_next = valid_next & (~is_backtrack)

            edge_mask = torch.where(is_step0, state.graph.edge_starts_mask, valid_next)
            return edge_mask

        # subgraph frontier：允许从已访问节点集的任意边扩张
        frontier_nodes = state.visited_nodes
        heads_frontier = frontier_nodes[state.graph.edge_index[0]]
        tails_frontier = frontier_nodes[state.graph.edge_index[1]]
        candidate = heads_frontier | tails_frontier
        # 不允许重复选择同一条边
        candidate = candidate & (~state.selected_mask)
        if self.forbid_revisit:
            # 至少有一个新节点被引入，避免无意义的封闭环
            heads_visited = state.visited_nodes[state.graph.edge_index[0]]
            tails_visited = state.visited_nodes[state.graph.edge_index[1]]
            introduces_new = ~(heads_visited & tails_visited)
            candidate = candidate & introduces_new
        return candidate

    def frontier_mask_edges(self, state: GraphState) -> torch.Tensor:
        """返回策略前沿边掩码（不含重复边），用于边级 bonus。"""
        edge_batch = state.graph.edge_batch
        is_step0 = state.step_counts[edge_batch] == 0
        if self.mode == "path":
            heads = state.graph.node_global_ids[state.graph.edge_index[0]]
            tails = state.graph.node_global_ids[state.graph.edge_index[1]]
            current_tail = state.current_tail[edge_batch]
            frontier = (heads == current_tail) | (tails == current_tail)
            frontier = torch.where(is_step0, state.graph.edge_starts_mask, frontier)
            frontier = frontier & (~state.selected_mask)
            return frontier

        frontier_nodes = state.visited_nodes
        heads_frontier = frontier_nodes[state.graph.edge_index[0]]
        tails_frontier = frontier_nodes[state.graph.edge_index[1]]
        frontier = heads_frontier | tails_frontier
        frontier = frontier & (~state.selected_mask)
        return frontier

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
            raise ValueError(
                f"Actions contain out-of-range indices; min_diff={int((actions - es).min().item())} "
                f"max_diff={int((actions - ee).max().item())}"
            )

        is_stop = (actions == stop_idx) | state.done
        edge_mask_valid = self.action_mask_edges(state)
        edge_actions_mask = ~is_stop
        edge_indices = actions[edge_actions_mask]

        if edge_indices.numel() > 0:
            if (edge_indices < 0).any() or (edge_indices >= state.graph.edge_index.size(1)).any():
                raise ValueError("Edge actions contain out-of-range indices for current batch.")
            # 允许指定图跳过 mask 校验（用于 GT replay 的对拍）
            if not state.graph.bypass_action_mask.any():
                if not edge_mask_valid[edge_indices].all():
                    raise ValueError("Edge actions violate action mask; check actor sampling or environment constraints.")
            else:
                violating = ~edge_mask_valid[edge_indices]
                if violating.any():
                    viol_graphs = state.graph.edge_batch[edge_indices[violating]].unique().tolist()
                    logging.getLogger("gflownet.debug").info(
                        "[ENV_BYPASS] step=%d graphs=%s bypassed action mask for GT replay.",
                        step_index,
                        viol_graphs,
                    )

        selected_mask = state.selected_mask.clone()
        selection_order = state.selection_order.clone()
        current_tail = state.current_tail.clone()
        prev_tail = state.prev_tail.clone()
        actions_record = state.actions.clone()
        answer_hits = state.answer_hits.clone()
        visited_nodes = state.visited_nodes.clone()

        if edge_indices.numel() > 0:
            selected_mask[edge_indices] = True
            selection_order[edge_indices] = step_index

            graph_ids = state.graph.edge_batch[edge_indices]
            heads_idx = state.graph.edge_index[0, edge_indices]
            tails_idx = state.graph.edge_index[1, edge_indices]
            heads = state.graph.node_global_ids[heads_idx]
            tails = state.graph.node_global_ids[tails_idx]

            if self.mode == "path":
                is_step0 = state.step_counts[graph_ids] == 0
                source_is_head = torch.where(
                    is_step0,
                    state.graph.node_is_start[heads_idx],
                    heads == state.current_tail[graph_ids],
                )
                next_local = torch.where(source_is_head, tails_idx, heads_idx)
                next_global = state.graph.node_global_ids[next_local]

                prev_tail[graph_ids] = state.current_tail[graph_ids]
                current_tail[graph_ids] = next_global
                visited_nodes[next_local] = True
                hit = state.graph.node_is_answer[next_local]
            else:
                # 子图模式：将边两端节点都纳入已访问集合
                prev_tail[graph_ids] = state.current_tail[graph_ids]
                visited_nodes[heads_idx] = True
                visited_nodes[tails_idx] = True
                # 维持 current_tail 的形状以兼容旧策略；此处不再驱动 mask
                hit = state.graph.node_is_answer[heads_idx] | state.graph.node_is_answer[tails_idx]

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
                    if self.mode == "path":
                        sel_next = next_local[mask_gid]
                        sel_next_global = state.graph.node_global_ids[sel_next]
                    else:
                        sel_heads = heads_idx[mask_gid]
                        sel_tails = tails_idx[mask_gid]
                        sel_nodes = torch.unique(torch.cat([sel_heads, sel_tails], dim=0))
                        sel_next_global = state.graph.node_global_ids[sel_nodes]
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
        step_counts = torch.where(done, state.step_counts, state.step_counts + 1)

        return GraphState(
            graph=state.graph,
            selected_mask=selected_mask,
            visited_nodes=visited_nodes,
            selection_order=selection_order,
            current_tail=current_tail,
            prev_tail=prev_tail,
            done=done,
            step_counts=step_counts,
            actions=actions_record,
            direction=state.direction,
            answer_hits=answer_hits,
            debug_logged=state.debug_logged,
            debug_enabled=state.debug_enabled,
        )


__all__ = ["GraphEnv", "GraphBatch", "GraphState"]
