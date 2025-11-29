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
    edge_relations: torch.Tensor      # [E_total]
    edge_scores: torch.Tensor         # [E_total]
    edge_labels: torch.Tensor         # [E_total]
    top_edge_mask: torch.Tensor       # [E_total] bool
    gt_path_edge_local_ids: torch.Tensor  # [P_total] local edge idx (offset per graph)
    gt_edge_ptr: torch.Tensor             # [B+1]
    gt_path_exists: torch.Tensor          # [B] bool
    is_answer_reachable: torch.Tensor     # [B] bool
    edge_starts_mask: torch.Tensor        # [E_total] bool, edges touching any start node


@dataclass
class GraphState:
    graph: GraphBatch
    selected_mask: torch.Tensor       # [E_total] bool
    selection_order: torch.Tensor     # [E_total] long
    current_tail: torch.Tensor        # [B] global entity id
    prev_tail: torch.Tensor           # [B] global entity id
    done: torch.Tensor                # [B] bool
    step_counts: torch.Tensor         # [B] long
    actions: torch.Tensor             # [B, max_steps + 1] long (edge idx within slice or stop idx)
    direction: Optional[torch.Tensor] # unused in current policies
    answer_hits: torch.Tensor         # [B] bool
    debug_logged: torch.Tensor        # [B] bool, used only when debug=True to throttle logs


class GraphEnv(nn.Module):
    """Graph environment on PyG flat batch (no dense padding)."""

    def __init__(
        self,
        max_steps: int,
        forbid_backtrack: bool = True,
        bidir_token: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.max_steps = int(max_steps)
        self.forbid_backtrack = bool(forbid_backtrack)
        self.bidir_token = bool(bidir_token)
        self.debug = bool(debug)

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
        edge_scores = batch["edge_scores"].to(device)
        edge_labels = batch["edge_labels"].to(device)
        top_edge_mask = batch["top_edge_mask"].to(device).bool()
        gt_path_edge_local_ids = batch["gt_path_edge_local_ids"].to(device)
        gt_edge_ptr = batch["gt_edge_ptr"].to(device)
        gt_path_exists = batch["gt_path_exists"].to(device).bool()
        is_answer_reachable = batch["is_answer_reachable"].to(device).bool()

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
        edge_starts_mask = node_is_start[edge_index[0]] | node_is_start[edge_index[1]]

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
            edge_relations=edge_relations,
            edge_scores=edge_scores,
            edge_labels=edge_labels,
            top_edge_mask=top_edge_mask,
            gt_path_edge_local_ids=gt_path_edge_local_ids,
            gt_edge_ptr=gt_edge_ptr,
            gt_path_exists=gt_path_exists,
            is_answer_reachable=is_answer_reachable,
            edge_starts_mask=edge_starts_mask,
        )
        return GraphState(
            graph=graph,
            selected_mask=selected_mask,
            selection_order=selection_order,
            current_tail=current_tail,
            prev_tail=prev_tail,
            done=done,
            step_counts=step_counts,
            actions=actions,
            direction=direction,
            answer_hits=answer_hits,
            debug_logged=debug_logged,
        )

    def action_mask_edges(self, state: GraphState) -> torch.Tensor:
        edge_batch = state.graph.edge_batch
        if not hasattr(state.graph, "edge_starts_mask"):
            raise ValueError("edge_starts_mask missing in GraphBatch; ensure GraphEnv.reset precomputes it.")

        heads_global = state.graph.node_global_ids[state.graph.edge_index[0]]
        tails_global = state.graph.node_global_ids[state.graph.edge_index[1]]
        current_tail = state.current_tail[edge_batch]
        valid_next = (heads_global == current_tail) | (tails_global == current_tail)

        if self.forbid_backtrack:
            prev_tail = state.prev_tail[edge_batch]
            is_backtrack = (
                ((heads_global == current_tail) & (tails_global == prev_tail))
                | ((tails_global == current_tail) & (heads_global == prev_tail))
            )
            valid_next = valid_next & (~is_backtrack)

        is_step0 = state.step_counts[edge_batch] == 0
        edge_mask = torch.where(is_step0, state.graph.edge_starts_mask, valid_next)
        return edge_mask

    def step(self, state: GraphState, actions: torch.Tensor, *, step_index: int) -> GraphState:
        device = state.graph.edge_index.device
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        edge_mask = state.graph.edge_batch.new_zeros(state.graph.edge_batch.size(0), dtype=torch.bool)
        edge_mask[:] = self.action_mask_edges(state)

        stop_actions = []
        edge_actions = []
        for g in range(num_graphs):
            es, ee = int(state.graph.edge_ptr[g].item()), int(state.graph.edge_ptr[g + 1].item())
            stop_idx = ee  # virtual stop at end of slice
            act = int(actions[g].item())
            if act == stop_idx or state.done[g]:
                stop_actions.append(g)
                continue
            if not (es <= act < ee):
                raise ValueError(f"Action {act} out of slice [{es}, {ee}) for graph {g}")
            if not edge_mask[act]:
                raise ValueError(f"Action {act} not valid for graph {g} at step {step_index}")
            edge_actions.append((g, act))

        selected_mask = state.selected_mask.clone()
        selection_order = state.selection_order.clone()
        current_tail = state.current_tail.clone()
        prev_tail = state.prev_tail.clone()
        actions_record = state.actions.clone()
        answer_hits = state.answer_hits.clone()
        if edge_actions:
            edge_indices = torch.tensor([a for _, a in edge_actions], device=device, dtype=torch.long)
            graph_ids = torch.tensor([g for g, _ in edge_actions], device=device, dtype=torch.long)
            selected_mask[edge_indices] = True
            selection_order[edge_indices] = step_index

            heads = state.graph.node_global_ids[state.graph.edge_index[0, edge_indices]]
            tails = state.graph.node_global_ids[state.graph.edge_index[1, edge_indices]]

            use_start = state.step_counts[graph_ids] == 0
            next_from_start = tails
            if use_start.any():
                start_head = torch.zeros_like(heads, dtype=torch.bool)
                start_tail = torch.zeros_like(tails, dtype=torch.bool)
                unique_graphs = graph_ids.unique()
                for gid in unique_graphs.tolist():
                    mask = graph_ids == gid
                    if not mask.any():
                        continue
                    # 起点实体既可能来自原始 start_entity_ids，也可能由 builder 兜底用 start_node_locals 选取的端点
                    starts: torch.Tensor
                    s, e = int(state.graph.start_entity_ptr[gid].item()), int(state.graph.start_entity_ptr[gid + 1].item())
                    if s < e:
                        starts = state.graph.start_entity_ids[s:e]
                    else:
                        starts = torch.empty(0, dtype=torch.long, device=device)
                    ns, ne = int(state.graph.start_ptr[gid].item()), int(state.graph.start_ptr[gid + 1].item())
                    if ns < ne:
                        start_locals = state.graph.start_node_locals[ns:ne]
                        if start_locals.numel() > 0:
                            starts_from_locals = state.graph.node_global_ids[start_locals]
                            starts = torch.unique(torch.cat([starts, starts_from_locals])) if starts.numel() > 0 else starts_from_locals
                    if starts.numel() == 0:
                        continue
                    start_head[mask] = torch.isin(heads[mask], starts)
                    start_tail[mask] = torch.isin(tails[mask], starts)
                next_from_start = torch.where(start_head, tails, heads)
            if use_start.any():
                current_tail[graph_ids] = torch.where(use_start, next_from_start, tails)
            else:
                current_tail[graph_ids] = heads  # arbitrary choice; we only care about connectivity

            # if forbid_backtrack, set prev_tail for next step
            prev_tail[graph_ids] = torch.where(use_start, current_tail[graph_ids], tails)

            actions_record[graph_ids, state.step_counts[graph_ids].clamp(max=actions_record.size(1) - 1)] = edge_indices

        if stop_actions:
            done_idx = torch.tensor(stop_actions, device=device, dtype=torch.long)
            done = state.done.clone()
            done[done_idx] = True
        else:
            done = state.done

        step_counts = state.step_counts.clone()
        step_counts = torch.where(done, step_counts, step_counts + 1)

        # 如果某图的任何选中边命中了答案实体，则标记成功。
        if selected_mask.any():
            heads = state.graph.node_global_ids[state.graph.edge_index[0]]
            tails = state.graph.node_global_ids[state.graph.edge_index[1]]
            selected_heads = torch.where(selected_mask, heads, torch.full_like(heads, -1))
            selected_tails = torch.where(selected_mask, tails, torch.full_like(tails, -1))
            for g in range(num_graphs):
                mask = state.graph.edge_batch == g
                if not mask.any():
                    continue
                selected_g = selected_mask[mask]
                answers_start, answers_end = int(state.graph.answer_ptr[g].item()), int(state.graph.answer_ptr[g + 1].item())
                answers = state.graph.answer_entity_ids[answers_start:answers_end]
                if answers.numel() == 0:
                    continue
                h_g = selected_heads[mask]
                t_g = selected_tails[mask]
                hit = ((h_g.unsqueeze(1) == answers) | (t_g.unsqueeze(1) == answers)).any()
                # Debug: 对 rank0 / debug 模式下的首图打印一次命中细节，便于对拍 success vs reach_fraction。
                # 仅输出被选中的边端点，避免日志爆炸。
                if getattr(self, "debug", False) and g == 0 and not state.debug_logged[g]:
                    try:
                        answers_list = answers.detach().cpu().tolist()
                        sel_pairs = torch.stack([h_g[selected_g], t_g[selected_g]], dim=1).detach().cpu().tolist()
                        logging.getLogger("gflownet.debug").info(
                            "[ENV_HIT_DEBUG] graph=%d hit=%s answers=%s selected_pairs=%s",
                            g,
                            bool(hit.item()),
                            answers_list,
                            sel_pairs,
                        )
                        state.debug_logged[g] = True
                    except Exception:
                        pass
                if hit:
                    answer_hits[g] = True

        return GraphState(
            graph=state.graph,
            selected_mask=selected_mask,
            selection_order=selection_order,
            current_tail=current_tail,
            prev_tail=prev_tail,
            done=done,
            step_counts=step_counts,
            actions=actions_record,
            direction=state.direction,
            answer_hits=answer_hits,
            debug_logged=state.debug_logged,
        )


__all__ = ["GraphEnv", "GraphBatch", "GraphState"]
