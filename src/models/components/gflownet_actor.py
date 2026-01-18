from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import nn
from torch_scatter import scatter_max

from src.gfn.ops import compute_policy_log_probs, gumbel_noise_like, neg_inf_value
from src.models.components.gflownet_env import GraphEnv, GraphState, STOP_RELATION
from src.models.components.gflownet_layers import TrajectoryAgent

MIN_TEMPERATURE = 1e-5
_STOP_STEP_OFFSET = 1
_STOP_NODE_NONE = -1
_STOP_LOGIT_DIM = 1
_EDGE_HEAD_INDEX = 0
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_CONTEXT_QUESTION = "question"
_CONTEXT_MODES = {_CONTEXT_QUESTION}
_DEFAULT_CONTEXT_MODE = _CONTEXT_QUESTION


@dataclass(frozen=True)
class RolloutResult:
    actions_seq: Optional[torch.Tensor]
    log_pf: torch.Tensor
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    reach_success: torch.Tensor
    length: torch.Tensor
    stop_node_locals: torch.Tensor


class GFlowNetActor(nn.Module):
    """Edge-level GFlowNet actor with trajectory-aware state."""

    def __init__(
        self,
        *,
        env: GraphEnv,
        agent: TrajectoryAgent,
        max_steps: int,
        policy_temperature: float,
        stop_bias_init: Optional[float] = None,
        context_mode: str = _DEFAULT_CONTEXT_MODE,
    ) -> None:
        super().__init__()
        self.env = env
        self.agent = agent
        mode = str(context_mode or _DEFAULT_CONTEXT_MODE).strip().lower()
        if mode not in _CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {sorted(_CONTEXT_MODES)}, got {mode!r}.")
        self.context_mode = mode
        self.stop_head = nn.Linear(int(agent.hidden_dim), _STOP_LOGIT_DIM)
        if stop_bias_init is not None:
            if self.stop_head.bias is None:
                raise RuntimeError("stop_head bias must be enabled for stop bias init.")
            with torch.no_grad():
                self.stop_head.bias.fill_(float(stop_bias_init))
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)

    def set_temperatures(self, *, policy_temperature: float) -> None:
        self.policy_temperature = max(float(policy_temperature), MIN_TEMPERATURE)

    def rollout(
        self,
        *,
        graph: dict[str, torch.Tensor],
    ) -> RolloutResult:
        device = graph["edge_index"].device
        temp, is_greedy = self._resolve_temperature(None)
        state = self.env.reset(graph, device=device, mode="forward", max_steps_override=None)
        num_graphs = int(state.batch["node_ptr"].numel() - 1)
        num_steps = self._resolve_rollout_max_steps(None) + _STOP_STEP_OFFSET
        log_pf_steps: list[torch.Tensor] = []
        actions_steps: list[torch.Tensor] = []
        stop_node_locals = torch.full((num_graphs,), _STOP_NODE_NONE, device=device, dtype=torch.long)
        tail_tokens = self._edge_tail_tokens(graph)
        head_tokens = self._edge_head_tokens(graph)
        start_nodes = self._start_node_tokens(graph=graph, state=state)
        hidden = self._initialize_hidden(graph=graph, state=state, start_nodes=start_nodes)

        for step in range(num_steps):
            state_vec = hidden
            valid_edges = self.env.forward_edge_mask(state)
            allow_stop = ~state.done
            edge_scores = self._compute_edge_scores(
                state_vec=state_vec,
                relation_tokens=graph["relation_tokens"],
                tail_tokens=tail_tokens,
                head_tokens=head_tokens,
                edge_batch=graph["edge_batch"],
                edge_ptr=graph["edge_ptr"],
            )
            stop_logits = self._compute_stop_logits(state_vec=state_vec)
            log_prob_edge, log_prob_stop, has_edge = self._compute_forward_log_probs(
                edge_scores=edge_scores,
                stop_logits=stop_logits,
                valid_edges=valid_edges,
                allow_stop=allow_stop,
                edge_batch=graph["edge_batch"],
                num_graphs=num_graphs,
                temp=temp,
                edge_guidance=None,
            )
            actions, log_pf = self._sample_actions(
                log_prob_edge=log_prob_edge,
                log_prob_stop=log_prob_stop,
                valid_edges=valid_edges,
                edge_batch=graph["edge_batch"],
                has_edge=has_edge,
                is_greedy=is_greedy,
            )
            actions, log_pf = self._apply_done_mask(state.done, actions, log_pf)
            stop_node_locals = self._update_stop_nodes(
                stop_node_locals=stop_node_locals,
                actions=actions,
                done_mask=state.done,
                curr_nodes=state.curr_nodes,
                node_ptr=graph["node_ptr"],
            )
            log_pf_steps.append(log_pf)
            actions_steps.append(actions)
            state.traj_actions.append(actions)
            state.traj_log_pf.append(log_pf)
            state.traj_log_pb.append(torch.zeros_like(log_pf))
            hidden = self._update_hidden(
                hidden=hidden,
                graph=graph,
                actions=actions,
                done_mask=state.done,
            )
            state = self.env.step(state, actions, step_index=step)

            stop_node_locals = self._finalize_stop_nodes(
                stop_node_locals=stop_node_locals,
                state=state,
                force_stop_at_end=False,
            )
        log_pf_steps_tensor = self._stack_steps(log_pf_steps, num_graphs=num_graphs, num_steps=num_steps, device=device)
        actions_seq = self._stack_steps(actions_steps, num_graphs=num_graphs, num_steps=num_steps, device=device)
        log_pf_total = log_pf_steps_tensor.sum(dim=1)
        log_pb_steps = torch.zeros_like(log_pf_steps_tensor)
        reach_success = self._compute_reach_success(stop_node_locals=stop_node_locals, graph=graph)
        length = state.step_counts.to(dtype=torch.float32)
        return RolloutResult(
            actions_seq=actions_seq,
            log_pf=log_pf_total,
            log_pf_steps=log_pf_steps_tensor,
            log_pb_steps=log_pb_steps,
            reach_success=reach_success.to(dtype=torch.float32),
            length=length,
            stop_node_locals=stop_node_locals,
        )

    def _resolve_rollout_max_steps(self, max_steps_override: Optional[int]) -> int:
        if max_steps_override is None:
            return int(self.max_steps)
        override = int(max_steps_override)
        if override <= _ZERO:
            raise ValueError(f"max_steps_override must be > 0, got {override}.")
        return min(int(self.max_steps), override)

    def _resolve_temperature(self, temperature: Optional[float]) -> tuple[float, bool]:
        base = self.policy_temperature if temperature is None else float(temperature)
        is_greedy = base < MIN_TEMPERATURE
        return max(base, MIN_TEMPERATURE), is_greedy

    def _edge_tail_tokens(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = graph["edge_index"]
        if edge_index.numel() == _ZERO:
            return torch.zeros(
                (0, graph["node_tokens"].size(-1)),
                device=edge_index.device,
                dtype=graph["node_tokens"].dtype,
            )
        tail_nodes = edge_index[_EDGE_TAIL_INDEX]
        return graph["node_tokens"].index_select(0, tail_nodes)

    def _edge_head_tokens(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        edge_index = graph["edge_index"]
        if edge_index.numel() == _ZERO:
            return torch.zeros(
                (0, graph["node_tokens"].size(-1)),
                device=edge_index.device,
                dtype=graph["node_tokens"].dtype,
            )
        head_nodes = edge_index[_EDGE_HEAD_INDEX]
        return graph["node_tokens"].index_select(0, head_nodes)

    def _start_node_tokens(self, *, graph: dict[str, torch.Tensor], state: GraphState) -> torch.Tensor:
        curr_nodes = state.curr_nodes.to(device=graph["node_tokens"].device, dtype=torch.long).view(-1)
        valid = curr_nodes >= _ZERO
        if not bool(valid.any().detach().tolist()):
            return torch.zeros_like(graph["question_tokens"])
        safe_nodes = curr_nodes.clamp(min=_ZERO)
        node_tokens = graph["node_tokens"].index_select(0, safe_nodes)
        return torch.where(valid.unsqueeze(-1), node_tokens, torch.zeros_like(node_tokens))

    def _initialize_hidden(
        self,
        *,
        graph: dict[str, torch.Tensor],
        state: GraphState,
        start_nodes: torch.Tensor,
    ) -> torch.Tensor:
        context_tokens = graph["question_tokens"]
        return self.agent.initialize_state(context_tokens, start_nodes=start_nodes)

    def _compute_edge_scores(
        self,
        *,
        state_vec: torch.Tensor,
        relation_tokens: torch.Tensor,
        tail_tokens: torch.Tensor,
        head_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_ptr: torch.Tensor,
    ) -> torch.Tensor:
        return self.agent.score(
            hidden=state_vec,
            relation_tokens=relation_tokens,
            node_tokens=tail_tokens,
            head_tokens=head_tokens,
            edge_batch=edge_batch,
            edge_ptr=edge_ptr,
        )

    def _compute_stop_logits(self, *, state_vec: torch.Tensor) -> torch.Tensor:
        return self.stop_head(state_vec).squeeze(-1)

    def _update_hidden(
        self,
        *,
        hidden: torch.Tensor,
        graph: dict[str, torch.Tensor],
        actions: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        move_mask = (actions >= _ZERO) & (~done_mask.to(device=actions.device, dtype=torch.bool))
        if not bool(move_mask.any().detach().tolist()):
            return hidden
        idx = move_mask.nonzero(as_tuple=False).view(-1)
        edge_ids = actions[move_mask].to(device=graph["edge_index"].device, dtype=torch.long)
        rel_tokens = graph["relation_tokens"].index_select(0, edge_ids)
        tail_nodes = graph["edge_index"][_EDGE_TAIL_INDEX].index_select(0, edge_ids)
        node_tokens = graph["node_tokens"].index_select(0, tail_nodes)
        hidden_updates = self.agent.step(
            hidden=hidden.index_select(0, idx),
            relation_tokens=rel_tokens,
            node_tokens=node_tokens,
        )
        updated = hidden.clone()
        updated.index_copy_(0, idx, hidden_updates)
        return updated

    @staticmethod
    def _compute_forward_log_probs(
        *,
        edge_scores: torch.Tensor,
        stop_logits: torch.Tensor,
        valid_edges: torch.Tensor,
        allow_stop: torch.Tensor,
        edge_batch: torch.Tensor,
        num_graphs: int,
        temp: float,
        edge_guidance: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if edge_guidance is not None:
            edge_guidance = edge_guidance.to(device=edge_scores.device, dtype=edge_scores.dtype).view(-1)
            if edge_guidance.numel() != edge_scores.numel():
                raise ValueError("edge_guidance length mismatch with edge_scores.")
            edge_scores = edge_scores + edge_guidance
        edge_logits = edge_scores / float(temp)
        stop_logits = stop_logits / float(temp)
        neg_inf = neg_inf_value(edge_logits)
        edge_logits = torch.where(valid_edges, edge_logits, torch.full_like(edge_logits, neg_inf))
        edge_log_prob, _, edge_log_denom, has_edge = compute_policy_log_probs(
            edge_logits=edge_logits,
            stop_logits=torch.full_like(stop_logits, neg_inf),  # placeholder; stop handled separately
            edge_batch=edge_batch,
            valid_edges=valid_edges,
            num_graphs=num_graphs,
            temperature=1.0,  # already applied
            allow_stop=allow_stop,
        )
        stop_logit = stop_logits
        stop_logit = torch.where(allow_stop, stop_logit, torch.full_like(stop_logit, neg_inf))
        # log p_stop = log_sigmoid
        log_prob_stop = stop_logit - torch.logaddexp(stop_logit, torch.zeros_like(stop_logit))
        log_prob_not_stop = -torch.logaddexp(stop_logit, torch.zeros_like(stop_logit))
        log_prob_edge = edge_log_prob + log_prob_not_stop.index_select(0, edge_batch)
        return log_prob_edge, log_prob_stop, has_edge

    def _sample_actions(
        self,
        *,
        log_prob_edge: torch.Tensor,
        log_prob_stop: torch.Tensor,
        valid_edges: torch.Tensor,
        edge_batch: torch.Tensor,
        has_edge: torch.Tensor,
        is_greedy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if log_prob_edge.numel() == 0:
            actions = torch.full_like(has_edge, STOP_RELATION, dtype=torch.long)
            return actions, log_prob_stop.to(dtype=torch.float32)
        neg_inf = neg_inf_value(log_prob_edge)
        score_edge = torch.where(valid_edges, log_prob_edge, torch.full_like(log_prob_edge, neg_inf))
        if not is_greedy:
            score_edge = score_edge + gumbel_noise_like(score_edge)
        score_edge_max, edge_argmax = scatter_max(score_edge, edge_batch, dim=0, dim_size=has_edge.numel())
        score_edge_max = torch.where(has_edge, score_edge_max, torch.full_like(score_edge_max, neg_inf))
        edge_argmax = torch.where(has_edge, edge_argmax, torch.zeros_like(edge_argmax))
        score_stop = log_prob_stop
        if not is_greedy:
            score_stop = score_stop + gumbel_noise_like(score_stop)
        choose_stop = (score_stop > score_edge_max) | (~has_edge)
        actions = torch.where(choose_stop, torch.full_like(edge_argmax, STOP_RELATION), edge_argmax)
        log_pf = torch.where(choose_stop, log_prob_stop, log_prob_edge[edge_argmax])
        return actions, log_pf

    @staticmethod
    def _apply_done_mask(
        done_mask: torch.Tensor,
        actions: torch.Tensor,
        log_pf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = torch.where(done_mask, torch.full_like(actions, STOP_RELATION), actions)
        log_pf = torch.where(done_mask, torch.zeros_like(log_pf), log_pf)
        return actions, log_pf

    @staticmethod
    def _resolve_current_stop_locals(
        *,
        curr_nodes: torch.Tensor,
        node_ptr: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].detach().item()) if node_ptr.numel() > 0 else 0
        sentinel = num_nodes_total + _ONE
        out = torch.full((num_graphs,), sentinel, device=node_ptr.device, dtype=dtype)
        if num_graphs <= _ZERO:
            return out, out != sentinel
        curr_nodes = curr_nodes.to(device=node_ptr.device, dtype=torch.long).view(-1)
        if curr_nodes.numel() != num_graphs:
            raise ValueError("curr_nodes length mismatch with batch size.")
        valid = curr_nodes >= _ZERO
        if not bool(valid.any().detach().tolist()):
            return out, valid
        safe_nodes = curr_nodes.clamp(min=_ZERO)
        graph_ids = torch.arange(num_graphs, device=node_ptr.device)
        local_idx = safe_nodes - node_ptr.index_select(0, graph_ids)
        out = torch.where(valid, local_idx.to(dtype=dtype), out)
        return out, valid

    @staticmethod
    def _update_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
        curr_nodes: torch.Tensor,
        node_ptr: torch.Tensor,
    ) -> torch.Tensor:
        done_mask = done_mask.to(device=stop_node_locals.device, dtype=torch.bool)
        needs_update = (~done_mask) & (stop_node_locals == _STOP_NODE_NONE)
        if not needs_update.any():
            return stop_node_locals
        updated = stop_node_locals.clone()
        update_stop = needs_update & (actions == STOP_RELATION)
        if update_stop.any():
            stop_locals, has_active = GFlowNetActor._resolve_current_stop_locals(
                curr_nodes=curr_nodes,
                node_ptr=node_ptr,
                dtype=stop_node_locals.dtype,
            )
            update_stop = update_stop & has_active
            updated = torch.where(update_stop, stop_locals, updated)
        return updated

    @staticmethod
    def _fill_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        curr_nodes: torch.Tensor,
        node_ptr: torch.Tensor,
        require_done: bool,
        done_mask: torch.Tensor,
    ) -> torch.Tensor:
        needs_backfill = stop_node_locals == _STOP_NODE_NONE
        if require_done:
            needs_backfill = needs_backfill & done_mask
        if not needs_backfill.any():
            return stop_node_locals
        locals_now, has_active = GFlowNetActor._resolve_current_stop_locals(
            curr_nodes=curr_nodes,
            node_ptr=node_ptr,
            dtype=stop_node_locals.dtype,
        )
        backfill_mask = needs_backfill & has_active
        return torch.where(backfill_mask, locals_now, stop_node_locals)

    def _finalize_stop_nodes(
        self,
        *,
        stop_node_locals: torch.Tensor,
        state: GraphState,
        force_stop_at_end: bool,
    ) -> torch.Tensor:
        if force_stop_at_end:
            return self._fill_stop_nodes(
                stop_node_locals=stop_node_locals,
                curr_nodes=state.curr_nodes,
                node_ptr=state.batch["node_ptr"],
                require_done=False,
                done_mask=state.done,
            )
        return self._fill_stop_nodes(
            stop_node_locals=stop_node_locals,
            curr_nodes=state.curr_nodes,
            node_ptr=state.batch["node_ptr"],
            require_done=True,
            done_mask=state.done,
        )

    @staticmethod
    def _stack_steps(
        steps: list[torch.Tensor],
        *,
        num_graphs: int,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not steps:
            return torch.zeros((num_graphs, num_steps), device=device, dtype=torch.float32)
        return torch.stack(steps, dim=1)

    @staticmethod
    def _compute_reach_success(
        *,
        stop_node_locals: torch.Tensor,
        graph: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        node_ptr = graph["node_ptr"]
        num_graphs = int(node_ptr.numel() - 1)
        stop_locals = stop_node_locals.to(device=node_ptr.device, dtype=torch.long).view(-1)
        valid_stop = stop_locals >= _ZERO
        stop_globals = node_ptr[:-1] + stop_locals.clamp(min=_ZERO)
        node_is_target = graph.get("node_is_target") or graph.get("node_is_answer")
        if node_is_target is None:
            num_nodes_total = int(node_ptr[-1].detach().item()) if node_ptr.numel() > 0 else 0
            node_is_target = torch.zeros(num_nodes_total, device=node_ptr.device, dtype=torch.bool)
            target_node_locals = graph["answer_node_locals"].to(device=node_ptr.device, dtype=torch.long).view(-1)
            target_ptr = graph["answer_ptr"].to(device=node_ptr.device, dtype=torch.long).view(-1)
            if target_node_locals.numel() > _ZERO:
                counts = target_ptr[1:] - target_ptr[:-1]
                graph_ids = torch.repeat_interleave(torch.arange(num_graphs, device=node_ptr.device), counts)
                target_globals = node_ptr.index_select(0, graph_ids) + target_node_locals
                node_is_target[target_globals] = True
        hits = node_is_target.index_select(0, stop_globals.clamp(min=_ZERO))
        return hits & valid_stop


__all__ = ["GFlowNetActor", "RolloutResult"]
