from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import contextlib

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_max

from src.data.schema.constants import _NON_TEXT_EMBEDDING_ID as _CVT_EMBEDDING_ID
from src.models.components.gflownet_env import GraphEnv, STOP_EDGE_RELATION, STOP_RELATION
from src.gfn.ops import compute_factorized_log_probs, gumbel_noise_like, neg_inf_value, segment_logsumexp_1d

MIN_TEMPERATURE = 1e-5
_STOP_STEP_OFFSET = 1
_STOP_NODE_NONE = -1
_DEFAULT_CHECK_FINITE = True
_DEFAULT_COSINE_BIAS_ALPHA = 0.0
_DEFAULT_COSINE_RELATION_BIAS_ALPHA = 0.0
_COSINE_SIM_EPS = 1.0e-8
_STOP_LOGIT_DIM = 1
_EDGE_HEAD_INDEX = 0
_EDGE_TAIL_INDEX = 1
_ZERO = 0
_ONE = 1
_NAN = float("nan")


@dataclass(frozen=True)
class RolloutResult:
    actions_seq: Optional[torch.Tensor]
    directions_seq: Optional[torch.Tensor]
    log_pf: torch.Tensor
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor
    selected_mask: torch.Tensor
    reach_success: torch.Tensor
    length: torch.Tensor
    stop_node_locals: torch.Tensor
    answer_node_hit: torch.Tensor
    start_node_hit: torch.Tensor
    visited_nodes: Optional[torch.Tensor]
    prior_loss: Optional[torch.Tensor]


@dataclass
class RolloutBuffers:
    actions_seq: Optional[torch.Tensor]
    log_pf_steps: torch.Tensor
    log_pb_steps: torch.Tensor

    @classmethod
    def create(
        cls,
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        *,
        record_actions: bool,
    ) -> "RolloutBuffers":
        actions_seq = None
        if record_actions:
            actions_seq = torch.full(
                (num_graphs, num_steps),
                STOP_RELATION,
                dtype=torch.long,
                device=device,
            )
        log_pf_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        log_pb_steps = torch.zeros(num_graphs, num_steps, dtype=torch.float32, device=device)
        return cls(actions_seq=actions_seq, log_pf_steps=log_pf_steps, log_pb_steps=log_pb_steps)


@dataclass
class DistancePriorState:
    loss_sum: torch.Tensor
    steps: torch.Tensor
    edge_base: torch.Tensor
    valid_dist: torch.Tensor


@dataclass
class RolloutEdgeCache:
    edge_head_tokens: torch.Tensor
    edge_tail_tokens: torch.Tensor
    tail_cosine: Optional[torch.Tensor]
    tail_bias_mask: Optional[torch.Tensor]
    edge_relation_cosine: Optional[torch.Tensor]
    relation_bias: Optional[torch.Tensor]


@dataclass
class RolloutContext:
    graph: dict[str, torch.Tensor]
    state: Any
    num_graphs: int
    num_steps: int
    buffers: RolloutBuffers
    log_pf_total: torch.Tensor
    stop_node_locals: torch.Tensor
    state_vec: torch.Tensor
    autocast_ctx: contextlib.AbstractContextManager
    edge_cache: RolloutEdgeCache
    prior: Optional[DistancePriorState]


class GFlowNetActor(nn.Module):
    """Edge-level GFlowNet actor."""

    def __init__(
        self,
        *,
        policy: nn.Module,
        env: GraphEnv,
        forward_head: nn.Module,
        backward_head: nn.Module,
        edge_forward_head: nn.Module,
        edge_backward_head: nn.Module,
        state_encoder: nn.Module,
        state_input_dim: int,
        hidden_dim: int,
        max_steps: int,
        policy_temperature: float,
        backward_temperature: Optional[float] = None,
        stop_bias_init: Optional[float] = None,
        cosine_bias_alpha: float = _DEFAULT_COSINE_BIAS_ALPHA,
        cosine_relation_bias_alpha: float = _DEFAULT_COSINE_RELATION_BIAS_ALPHA,
        relation_use_active_nodes: bool = True,
        check_finite: bool = _DEFAULT_CHECK_FINITE,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.env = env
        self.forward_head = forward_head
        self.backward_head = backward_head
        self.edge_forward_head = edge_forward_head
        self.edge_backward_head = edge_backward_head
        self.state_encoder = state_encoder
        self.stop_head = nn.Linear(int(state_input_dim), _STOP_LOGIT_DIM)
        if stop_bias_init is not None:
            if self.stop_head.bias is None:
                raise RuntimeError("stop_head bias must be enabled for stop bias init.")
            with torch.no_grad():
                self.stop_head.bias.fill_(float(stop_bias_init))
        self.hidden_dim = int(hidden_dim)
        self.max_steps = int(max_steps)
        self.policy_temperature = float(policy_temperature)
        self.backward_temperature = self._resolve_backward_temperature(backward_temperature)
        self.cosine_bias_alpha = float(cosine_bias_alpha)
        self.cosine_relation_bias_alpha = float(cosine_relation_bias_alpha)
        self.relation_use_active_nodes = bool(relation_use_active_nodes)
        self.check_finite = bool(check_finite)

    def set_temperatures(
        self,
        *,
        policy_temperature: float,
        backward_temperature: Optional[float] = None,
    ) -> None:
        policy_temp = max(float(policy_temperature), MIN_TEMPERATURE)
        if backward_temperature is None:
            backward_temp = policy_temp
        else:
            backward_temp = max(float(backward_temperature), MIN_TEMPERATURE)
        self.policy_temperature = policy_temp
        self.backward_temperature = backward_temp

    def _assert_finite_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        *,
        step: Optional[int] = None,
    ) -> None:
        if not self.check_finite:
            return
        if torch.isfinite(tensor).all():
            return
        summary = self._summarize_non_finite(tensor)
        step_part = f" step={step}" if step is not None else ""
        raise RuntimeError(f"Non-finite detected in {name}{step_part}: {summary}")

    @staticmethod
    def _summarize_non_finite(tensor: torch.Tensor) -> str:
        finite = torch.isfinite(tensor)
        non_finite = int((~finite).sum().item())
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_vals = tensor[finite]
        if finite_vals.numel() > _ZERO:
            calc = finite_vals.to(dtype=torch.float32)
            min_val = float(calc.min().item())
            max_val = float(calc.max().item())
            mean_val = float(calc.mean().item())
            abs_max = float(calc.abs().max().item())
        else:
            min_val = _NAN
            max_val = _NAN
            mean_val = _NAN
            abs_max = _NAN
        return (
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"non_finite={non_finite} (nan={nan_count}, inf={inf_count}), "
            f"finite_min={min_val}, finite_max={max_val}, finite_mean={mean_val}, abs_max={abs_max}"
        )

    def rollout(
        self,
        *,
        graph: dict[str, torch.Tensor],
        temperature: Optional[float] = None,
        record_actions: bool = True,
        record_visited: bool = True,
        max_steps_override: Optional[int] = None,
        force_stop_at_end: bool = False,
        distance_prior_beta: Optional[float] = None,
    ) -> RolloutResult:
        temp, is_greedy = self._resolve_temperature(temperature)
        ctx = self._init_rollout_context(
            graph=graph,
            record_actions=record_actions,
            max_steps_override=max_steps_override,
            distance_prior_beta=distance_prior_beta,
        )
        for step in range(ctx.num_steps):
            self._rollout_step(ctx, step, temp=temp, is_greedy=is_greedy)
        return self._finalize_rollout(
            ctx,
            force_stop_at_end=force_stop_at_end,
            record_actions=record_actions,
            record_visited=record_visited,
        )

    def _init_rollout_context(
        self,
        *,
        graph: dict[str, torch.Tensor],
        record_actions: bool,
        max_steps_override: Optional[int],
        distance_prior_beta: Optional[float],
    ) -> RolloutContext:
        edge_index = graph["edge_index"]
        device = edge_index.device
        state = self.env.reset(graph, device=device)
        num_graphs = int(state.graph.node_ptr.numel() - 1)
        max_steps = self._resolve_rollout_max_steps(max_steps_override)
        num_steps = max_steps + _STOP_STEP_OFFSET
        buffers = self._init_buffers(num_graphs, num_steps, device, record_actions=record_actions)
        log_pf_total = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        stop_node_locals = torch.full((num_graphs,), _STOP_NODE_NONE, device=device, dtype=torch.long)
        autocast_ctx = self._autocast_context(device)
        state_vec = self.state_encoder.init_state(num_graphs, device=device, dtype=graph["node_tokens"].dtype)
        edge_cache = self._build_edge_cache(graph, autocast_ctx)
        prior = self._init_distance_prior(
            graph=graph,
            edge_index=edge_index,
            num_graphs=num_graphs,
            distance_prior_beta=distance_prior_beta,
        )
        return RolloutContext(
            graph=graph,
            state=state,
            num_graphs=num_graphs,
            num_steps=num_steps,
            buffers=buffers,
            log_pf_total=log_pf_total,
            stop_node_locals=stop_node_locals,
            state_vec=state_vec,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
            prior=prior,
        )

    def _init_distance_prior(
        self,
        *,
        graph: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        num_graphs: int,
        distance_prior_beta: Optional[float],
    ) -> Optional[DistancePriorState]:
        if distance_prior_beta is None or float(distance_prior_beta) <= float(_ZERO):
            return None
        node_min_dists = graph["node_min_dists"].to(device=edge_index.device, dtype=torch.long)
        heads = edge_index[_EDGE_HEAD_INDEX]
        tails = edge_index[_EDGE_TAIL_INDEX]
        dist_head = node_min_dists.index_select(0, heads)
        dist_tail = node_min_dists.index_select(0, tails)
        valid_dist = (dist_head >= _ZERO) & (dist_tail >= _ZERO)
        edge_base = (dist_head - dist_tail).to(dtype=torch.float32) * float(distance_prior_beta)
        loss_sum = torch.zeros(num_graphs, dtype=torch.float32, device=edge_index.device)
        steps = torch.zeros(num_graphs, dtype=torch.float32, device=edge_index.device)
        return DistancePriorState(loss_sum=loss_sum, steps=steps, edge_base=edge_base, valid_dist=valid_dist)

    def _build_edge_cache(
        self,
        graph: dict[str, torch.Tensor],
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> RolloutEdgeCache:
        edge_index = graph["edge_index"]
        node_tokens = graph["node_tokens"]
        edge_heads = edge_index[_EDGE_HEAD_INDEX]
        edge_tails = edge_index[_EDGE_TAIL_INDEX]
        edge_head_tokens = node_tokens.index_select(0, edge_heads)
        edge_tail_tokens = node_tokens.index_select(0, edge_tails)
        tail_cosine = None
        tail_bias_mask = None
        edge_relation_cosine = None
        if self.cosine_bias_alpha != float(_ZERO) and edge_tails.numel() > _ZERO:
            edge_batch = graph["edge_batch"]
            with autocast_ctx:
                q_tokens = graph["question_tokens"].index_select(0, edge_batch)
                tail_cosine = F.cosine_similarity(
                    q_tokens,
                    edge_tail_tokens,
                    dim=-1,
                    eps=_COSINE_SIM_EPS,
                )
            tail_bias_mask = self._build_node_bias_mask(
                node_embedding_ids=graph.get("node_embedding_ids"),
                edge_nodes=edge_tails,
                device=edge_index.device,
                dtype=tail_cosine.dtype,
            )
        if self.cosine_relation_bias_alpha != float(_ZERO) and edge_tails.numel() > _ZERO:
            edge_batch = graph["edge_batch"]
            with autocast_ctx:
                q_tokens = graph["question_tokens"].index_select(0, edge_batch)
                edge_relation_cosine = F.cosine_similarity(
                    q_tokens,
                    graph["relation_tokens"],
                    dim=-1,
                    eps=_COSINE_SIM_EPS,
                )
        return RolloutEdgeCache(
            edge_head_tokens=edge_head_tokens,
            edge_tail_tokens=edge_tail_tokens,
            tail_cosine=tail_cosine,
            tail_bias_mask=tail_bias_mask,
            edge_relation_cosine=edge_relation_cosine,
            relation_bias=None,
        )

    def _rollout_step(
        self,
        ctx: RolloutContext,
        step: int,
        *,
        temp: float,
        is_greedy: bool,
    ) -> None:
        pre_done = ctx.state.done
        valid_edges = self._valid_edges(ctx.state)
        allow_stop = self._allow_stop(ctx.state)
        log_prob_edge, log_prob_stop, has_edge = self._compute_forward_log_probs(
            ctx,
            valid_edges=valid_edges,
            allow_stop=allow_stop,
            temp=temp,
            step=step,
        )
        self._update_distance_prior(ctx, log_prob_edge, valid_edges)
        actions = self._sample_actions_step(
            ctx,
            log_prob_edge=log_prob_edge,
            log_prob_stop=log_prob_stop,
            valid_edges=valid_edges,
            has_edge=has_edge,
            pre_done=pre_done,
            step=step,
            is_greedy=is_greedy,
        )
        self._advance_env_and_log_pb(ctx, actions, pre_done=pre_done, step=step)

    def _update_distance_prior(
        self,
        ctx: RolloutContext,
        log_prob_edge: torch.Tensor,
        valid_edges: torch.Tensor,
    ) -> None:
        if ctx.prior is None:
            return
        prior_step, has_prior = self._compute_distance_prior_step(
            edge_prior_base=ctx.prior.edge_base,
            valid_dist=ctx.prior.valid_dist,
            valid_edges=valid_edges,
            edge_batch=ctx.graph["edge_batch"],
            log_prob_edge=log_prob_edge,
            num_graphs=ctx.num_graphs,
        )
        ctx.prior.loss_sum = ctx.prior.loss_sum + prior_step
        ctx.prior.steps = ctx.prior.steps + has_prior.to(dtype=ctx.prior.steps.dtype)

    def _sample_actions_step(
        self,
        ctx: RolloutContext,
        *,
        log_prob_edge: torch.Tensor,
        log_prob_stop: torch.Tensor | None,
        valid_edges: torch.Tensor,
        has_edge: torch.Tensor,
        pre_done: torch.Tensor,
        step: int,
        is_greedy: bool,
    ) -> torch.Tensor:
        actions, log_pf = self._sample_actions(
            log_prob_edge=log_prob_edge,
            log_prob_stop=log_prob_stop,
            valid_edges=valid_edges,
            edge_batch=ctx.graph["edge_batch"],
            has_edge=has_edge,
            is_greedy=is_greedy,
        )
        self._assert_finite_tensor(log_pf, "log_pf", step=step)
        actions, log_pf = self._apply_done_mask(pre_done, actions, log_pf)
        ctx.stop_node_locals = self._update_stop_nodes(
            stop_node_locals=ctx.stop_node_locals,
            actions=actions,
            done_mask=pre_done,
            edge_index=ctx.graph["edge_index"],
            edge_batch=ctx.graph["edge_batch"],
            edge_relations=ctx.graph["edge_relations"],
            node_ptr=ctx.state.graph.node_ptr,
            active_nodes=ctx.state.active_nodes,
            node_batch=ctx.state.graph.node_batch,
            node_is_answer=ctx.state.graph.node_is_answer,
        )
        if ctx.buffers.actions_seq is not None:
            ctx.buffers.actions_seq[:, step] = actions
        ctx.buffers.log_pf_steps[:, step] = log_pf
        ctx.log_pf_total = ctx.log_pf_total + log_pf
        self._update_state_vec(ctx, actions, pre_done=pre_done, step=step)
        return actions

    def _update_state_vec(
        self,
        ctx: RolloutContext,
        actions: torch.Tensor,
        *,
        pre_done: torch.Tensor,
        step: int,
    ) -> None:
        relation_tokens = self._gather_relation_tokens(
            relation_tokens=ctx.graph["relation_tokens"],
            actions=actions,
            num_graphs=ctx.num_graphs,
        )
        update_mask = (actions >= _ZERO) & (~pre_done)
        ctx.state_vec = self.state_encoder.update_state(
            ctx.state_vec,
            relation_tokens=relation_tokens,
            update_mask=update_mask,
        )
        self._assert_finite_tensor(ctx.state_vec, "state_vec", step=step)

    def _advance_env_and_log_pb(
        self,
        ctx: RolloutContext,
        actions: torch.Tensor,
        *,
        pre_done: torch.Tensor,
        step: int,
    ) -> None:
        ctx.state = self.env.step(ctx.state, actions, step_index=step)
        log_pb = self._compute_log_pb(
            state=ctx.state,
            graph=ctx.graph,
            state_vec=ctx.state_vec,
            actions=actions,
            autocast_ctx=ctx.autocast_ctx,
            num_graphs=ctx.num_graphs,
            edge_cache=ctx.edge_cache,
        )
        self._assert_finite_tensor(log_pb, "log_pb", step=step)
        log_pb = torch.where(pre_done, torch.zeros_like(log_pb), log_pb)
        ctx.buffers.log_pb_steps[:, step] = log_pb

    def _compute_forward_log_probs(
        self,
        ctx: RolloutContext,
        *,
        valid_edges: torch.Tensor,
        allow_stop: torch.Tensor,
        temp: float,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        relation_logits, edge_scores, stop_logits = self._score_forward_with_stop(
            ctx.graph,
            ctx.state_vec,
            active_nodes=ctx.state.active_nodes,
            autocast_ctx=ctx.autocast_ctx,
            edge_cache=ctx.edge_cache,
        )
        self._assert_finite_tensor(relation_logits, "relation_logits", step=step)
        self._assert_finite_tensor(edge_scores, "edge_scores", step=step)
        if stop_logits is not None:
            self._assert_finite_tensor(stop_logits, "stop_logits", step=step)
        edge_logits = self._policy_logits(
            valid_edges=valid_edges,
            edge_logits=edge_scores,
            autocast_ctx=ctx.autocast_ctx,
        )
        self._assert_finite_tensor(edge_logits, "edge_logits", step=step)
        log_prob_edge, log_prob_stop, _, has_edge = compute_factorized_log_probs(
            relation_logits=relation_logits,
            edge_logits=edge_logits,
            edge_relations=ctx.graph["edge_relations"],
            edge_batch=ctx.graph["edge_batch"],
            valid_edges=valid_edges,
            num_graphs=ctx.num_graphs,
            temperature=temp,
            stop_logits=stop_logits,
            allow_stop=allow_stop,
        )
        self._assert_finite_tensor(log_prob_edge, "log_prob_edge", step=step)
        if log_prob_stop is not None:
            self._assert_finite_tensor(log_prob_stop, "log_prob_stop", step=step)
        return log_prob_edge, log_prob_stop, has_edge

    def _finalize_rollout(
        self,
        ctx: RolloutContext,
        *,
        force_stop_at_end: bool,
        record_actions: bool,
        record_visited: bool,
    ) -> RolloutResult:
        stop_node_locals = self._finalize_stop_nodes(
            stop_node_locals=ctx.stop_node_locals,
            state=ctx.state,
            force_stop_at_end=force_stop_at_end,
        )
        reach_success = ctx.state.answer_hits.float()
        length = ctx.state.step_counts.to(dtype=torch.float32)
        prior_loss = None
        if ctx.prior is not None:
            denom = ctx.prior.steps.clamp(min=float(_ONE))
            prior_loss = ctx.prior.loss_sum / denom
        return RolloutResult(
            actions_seq=ctx.buffers.actions_seq,
            directions_seq=ctx.state.directions if record_actions else None,
            log_pf=ctx.log_pf_total,
            log_pf_steps=ctx.buffers.log_pf_steps,
            log_pb_steps=ctx.buffers.log_pb_steps,
            selected_mask=ctx.state.used_edge_mask,
            reach_success=reach_success,
            length=length,
            stop_node_locals=stop_node_locals,
            answer_node_hit=ctx.state.answer_node_hit,
            start_node_hit=ctx.state.start_node_hit,
            visited_nodes=ctx.state.visited_nodes if record_visited else None,
            prior_loss=prior_loss,
        )

    @staticmethod
    def _compute_distance_prior_step(
        *,
        edge_prior_base: torch.Tensor,
        valid_dist: torch.Tensor,
        valid_edges: torch.Tensor,
        edge_batch: torch.Tensor,
        log_prob_edge: torch.Tensor,
        num_graphs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_prior = valid_edges & valid_dist
        if not torch.any(valid_prior):
            device = edge_prior_base.device
            dtype = edge_prior_base.dtype
            return (
                torch.zeros((num_graphs,), device=device, dtype=dtype),
                torch.zeros((num_graphs,), device=device, dtype=torch.bool),
            )
        logits = edge_prior_base[valid_prior]
        seg_ids = edge_batch[valid_prior]
        log_denom = segment_logsumexp_1d(logits, seg_ids, num_graphs)
        log_prior = logits - log_denom.index_select(0, seg_ids)
        log_prob = log_prob_edge[valid_prior]
        edge_loss = torch.exp(log_prior) * (log_prior - log_prob)
        prior_loss = torch.zeros((num_graphs,), device=edge_prior_base.device, dtype=edge_prior_base.dtype)
        prior_loss.index_add_(0, seg_ids, edge_loss)
        has_prior = log_denom > neg_inf_value(log_denom)
        return prior_loss, has_prior

    @staticmethod
    def _autocast_context(device: torch.device):
        if device.type == "cpu":
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, enabled=torch.is_autocast_enabled())

    @staticmethod
    def _init_buffers(
        num_graphs: int,
        num_steps: int,
        device: torch.device,
        *,
        record_actions: bool,
    ) -> RolloutBuffers:
        return RolloutBuffers.create(
            num_graphs=num_graphs,
            num_steps=num_steps,
            device=device,
            record_actions=record_actions,
        )

    def _resolve_temperature(self, temperature: Optional[float]) -> tuple[float, bool]:
        base = self.policy_temperature if temperature is None else float(temperature)
        is_greedy = base < MIN_TEMPERATURE
        return max(base, MIN_TEMPERATURE), is_greedy

    def _resolve_backward_temperature(self, temperature: Optional[float]) -> float:
        base = self.policy_temperature if temperature is None else float(temperature)
        return max(base, MIN_TEMPERATURE)

    def _policy_logits(
        self,
        *,
        valid_edges: torch.Tensor,
        edge_logits: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
    ) -> torch.Tensor:
        with autocast_ctx:
            edge_logits = self.policy(
                edge_scores=edge_logits,
                valid_edges_mask=valid_edges,
            )
        return edge_logits

    def _valid_edges(self, state: Any) -> torch.Tensor:
        unused_edges = ~state.used_edge_mask
        forward_mask = self.env.forward_edge_mask(state)
        return forward_mask & unused_edges

    def _allow_stop(self, state: Any) -> torch.Tensor:
        step_counts = state.step_counts
        allow_stop = ~state.done
        if step_counts.numel() > _ZERO:
            horizon_exhausted = step_counts >= self.max_steps
            allow_stop = allow_stop & (~horizon_exhausted)
        min_stop_steps = int(getattr(self.env, "min_stop_steps", 0))
        if min_stop_steps > _ZERO:
            allow_stop = allow_stop & (step_counts >= min_stop_steps)
        return allow_stop

    def _sample_actions(
        self,
        *,
        log_prob_edge: torch.Tensor,
        log_prob_stop: torch.Tensor | None,
        valid_edges: torch.Tensor,
        edge_batch: torch.Tensor,
        has_edge: torch.Tensor,
        is_greedy: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if log_prob_stop is None:
            raise ValueError("log_prob_stop must be provided for stop-aware sampling.")
        if log_prob_edge.numel() == 0:
            actions = torch.full_like(has_edge, STOP_RELATION, dtype=torch.long)
            log_pf = log_prob_stop.to(dtype=torch.float32)
            return actions, log_pf

        if is_greedy:
            score_edge = log_prob_edge
            score_stop = log_prob_stop
        else:
            score_edge = log_prob_edge + gumbel_noise_like(log_prob_edge)
            score_stop = log_prob_stop + gumbel_noise_like(log_prob_stop)

        neg_inf = neg_inf_value(score_edge)
        score_edge = torch.where(valid_edges, score_edge, torch.full_like(score_edge, neg_inf))
        score_edge_max, edge_argmax = scatter_max(score_edge, edge_batch, dim=0, dim_size=has_edge.numel())
        score_edge_max = torch.where(has_edge, score_edge_max, torch.full_like(score_edge_max, neg_inf))
        edge_argmax = torch.where(has_edge, edge_argmax, torch.zeros_like(edge_argmax))
        choose_stop = score_stop > score_edge_max
        choose_stop = choose_stop | (~has_edge)
        actions = torch.where(choose_stop, torch.full_like(edge_argmax, STOP_RELATION), edge_argmax)
        log_pf = torch.where(choose_stop, log_prob_stop, log_prob_edge[edge_argmax])
        return actions, log_pf

    def _score_edges_forward(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache] = None,
    ) -> torch.Tensor:
        edge_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=True,
        )
        edge_node_tokens = edge_cache.edge_tail_tokens if edge_cache is not None else None
        return self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_forward_head,
            use_tail=True,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
            edge_node_tokens=edge_node_tokens,
        )

    def _score_forward_with_stop(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=self.relation_use_active_nodes,
        )
        edge_state_inputs = relation_state_inputs
        if not self.relation_use_active_nodes:
            edge_state_inputs = self._build_state_inputs(
                node_tokens=graph["node_tokens"],
                question_tokens=graph["question_tokens"],
                state_vec=state_vec,
                active_nodes=active_nodes,
                node_batch=graph["node_batch"],
                include_active_nodes=True,
            )
        with autocast_ctx:
            relation_logits = self.forward_head(relation_state_inputs)
            stop_logits = self.stop_head(relation_state_inputs).squeeze(-1)
        relation_logits = self._apply_relation_cosine_bias(
            relation_logits=relation_logits,
            graph=graph,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
        )
        edge_node_tokens = edge_cache.edge_tail_tokens if edge_cache is not None else None
        edge_scores = self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_forward_head,
            use_tail=True,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
            edge_node_tokens=edge_node_tokens,
        )
        return relation_logits, edge_scores, stop_logits

    def _score_edges_backward(
        self,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        *,
        active_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_state_inputs = self._build_state_inputs(
            node_tokens=graph["node_tokens"],
            question_tokens=graph["question_tokens"],
            state_vec=state_vec,
            active_nodes=active_nodes,
            node_batch=graph["node_batch"],
            include_active_nodes=self.relation_use_active_nodes,
        )
        edge_state_inputs = relation_state_inputs
        if not self.relation_use_active_nodes:
            edge_state_inputs = self._build_state_inputs(
                node_tokens=graph["node_tokens"],
                question_tokens=graph["question_tokens"],
                state_vec=state_vec,
                active_nodes=active_nodes,
                node_batch=graph["node_batch"],
                include_active_nodes=True,
            )
        with autocast_ctx:
            relation_logits = self.backward_head(relation_state_inputs)
            stop_logits = self.stop_head(relation_state_inputs).squeeze(-1)
        edge_node_tokens = edge_cache.edge_head_tokens if edge_cache is not None else None
        edge_scores = self._score_edge_entities(
            graph=graph,
            state_inputs=edge_state_inputs,
            edge_head=self.edge_backward_head,
            use_tail=False,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
            edge_node_tokens=edge_node_tokens,
            apply_cosine_bias=False,
        )
        return relation_logits, edge_scores, stop_logits

    def _apply_relation_cosine_bias(
        self,
        *,
        relation_logits: torch.Tensor,
        graph: dict[str, torch.Tensor],
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache] = None,
    ) -> torch.Tensor:
        if self.cosine_relation_bias_alpha == float(_ZERO):
            return relation_logits
        edge_relations = graph["edge_relations"]
        if edge_relations.numel() == _ZERO:
            return relation_logits
        bias = self._resolve_relation_bias(
            relation_logits=relation_logits,
            graph=graph,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
        )
        return relation_logits + (self.cosine_relation_bias_alpha * bias)

    def _resolve_relation_bias(
        self,
        *,
        relation_logits: torch.Tensor,
        graph: dict[str, torch.Tensor],
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache],
    ) -> torch.Tensor:
        edge_batch = graph["edge_batch"]
        edge_relations = graph["edge_relations"]
        relation_tokens = graph["relation_tokens"]
        question_tokens = graph["question_tokens"]
        num_graphs = int(relation_logits.size(0))
        num_relations = int(relation_logits.size(1))
        if edge_cache is not None and edge_cache.edge_relation_cosine is not None:
            bias = edge_cache.relation_bias
            if bias is None:
                bias = self._aggregate_relation_cosine_bias(
                    edge_cosine=edge_cache.edge_relation_cosine,
                    edge_batch=edge_batch,
                    edge_relations=edge_relations,
                    num_graphs=num_graphs,
                    num_relations=num_relations,
                    device=relation_logits.device,
                    dtype=relation_logits.dtype,
                )
                edge_cache.relation_bias = bias
        else:
            with autocast_ctx:
                q_tokens = question_tokens.index_select(0, edge_batch)
                cosine = F.cosine_similarity(
                    q_tokens,
                    relation_tokens,
                    dim=-1,
                    eps=_COSINE_SIM_EPS,
                )
            bias = self._aggregate_relation_cosine_bias(
                edge_cosine=cosine,
                edge_batch=edge_batch,
                edge_relations=edge_relations,
                num_graphs=num_graphs,
                num_relations=num_relations,
                device=relation_logits.device,
                dtype=relation_logits.dtype,
            )
        if bias.dtype != relation_logits.dtype:
            bias = bias.to(dtype=relation_logits.dtype)
        return bias

    @staticmethod
    def _aggregate_relation_cosine_bias(
        *,
        edge_cosine: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        num_graphs: int,
        num_relations: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_graphs <= _ZERO or num_relations <= _ZERO:
            return torch.zeros((num_graphs, num_relations), device=device, dtype=dtype)
        flat_idx = edge_batch * num_relations + edge_relations
        bias = torch.zeros(num_graphs * num_relations, device=device, dtype=dtype)
        counts = torch.zeros_like(bias)
        bias.index_add_(0, flat_idx, edge_cosine.to(dtype=dtype))
        counts.index_add_(0, flat_idx, torch.ones_like(edge_cosine, dtype=dtype))
        bias = bias / counts.clamp(min=float(_ONE))
        return bias.view(num_graphs, num_relations)

    def _score_edge_entities(
        self,
        *,
        graph: dict[str, torch.Tensor],
        state_inputs: torch.Tensor,
        edge_head: nn.Module,
        use_tail: bool,
        autocast_ctx: contextlib.AbstractContextManager,
        edge_cache: Optional[RolloutEdgeCache] = None,
        edge_node_tokens: Optional[torch.Tensor] = None,
        apply_cosine_bias: bool = True,
    ) -> torch.Tensor:
        edge_index = graph["edge_index"]
        node_tokens = graph["node_tokens"]
        relation_tokens = graph["relation_tokens"]
        edge_batch = graph["edge_batch"]
        edge_nodes = edge_index[_EDGE_TAIL_INDEX if use_tail else _EDGE_HEAD_INDEX]
        if edge_node_tokens is None:
            edge_node_tokens = node_tokens.index_select(0, edge_nodes)
        with autocast_ctx:
            edge_scores = edge_head(
                state_inputs=state_inputs,
                relation_tokens=relation_tokens,
                node_tokens=edge_node_tokens,
                edge_batch=edge_batch,
            )
        edge_scores = self._apply_cosine_bias(
            edge_scores=edge_scores,
            edge_node_tokens=edge_node_tokens,
            edge_batch=edge_batch,
            question_tokens=graph["question_tokens"],
            node_embedding_ids=graph.get("node_embedding_ids"),
            edge_nodes=edge_nodes,
            autocast_ctx=autocast_ctx,
            apply_bias=apply_cosine_bias,
            edge_cache=edge_cache,
            use_tail=use_tail,
        )
        return self._apply_edge_distance_bias(
            edge_scores=edge_scores,
            edge_distance_bias=graph.get("edge_distance_bias"),
        )

    def _apply_cosine_bias(
        self,
        *,
        edge_scores: torch.Tensor,
        edge_node_tokens: torch.Tensor,
        edge_batch: torch.Tensor,
        question_tokens: torch.Tensor,
        node_embedding_ids: Optional[torch.Tensor],
        edge_nodes: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        apply_bias: bool,
        edge_cache: Optional[RolloutEdgeCache],
        use_tail: bool,
    ) -> torch.Tensor:
        if not apply_bias or self.cosine_bias_alpha == float(_ZERO):
            return edge_scores
        if edge_cache is not None and use_tail:
            if edge_cache.tail_cosine is not None and edge_cache.tail_bias_mask is not None:
                cosine = edge_cache.tail_cosine.to(dtype=edge_scores.dtype)
                mask = edge_cache.tail_bias_mask.to(dtype=edge_scores.dtype)
                return edge_scores + (self.cosine_bias_alpha * cosine * mask)
        with autocast_ctx:
            q_tokens = question_tokens.index_select(0, edge_batch)
            cosine = F.cosine_similarity(
                q_tokens,
                edge_node_tokens,
                dim=-1,
                eps=_COSINE_SIM_EPS,
            )
        mask = self._build_node_bias_mask(
            node_embedding_ids=node_embedding_ids,
            edge_nodes=edge_nodes,
            device=edge_scores.device,
            dtype=edge_scores.dtype,
        )
        return edge_scores + (self.cosine_bias_alpha * cosine * mask)

    @staticmethod
    def _apply_edge_distance_bias(
        *,
        edge_scores: torch.Tensor,
        edge_distance_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if edge_distance_bias is None or not torch.is_tensor(edge_distance_bias):
            return edge_scores
        if edge_distance_bias.numel() != edge_scores.numel():
            raise ValueError("edge_distance_bias length mismatch with edge_scores.")
        return edge_scores + edge_distance_bias.to(device=edge_scores.device, dtype=edge_scores.dtype)

    @staticmethod
    def _build_node_bias_mask(
        *,
        node_embedding_ids: Optional[torch.Tensor],
        edge_nodes: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if node_embedding_ids is None:
            return torch.ones(edge_nodes.size(0), device=device, dtype=dtype)
        edge_ids = node_embedding_ids.index_select(0, edge_nodes)
        is_cvt = edge_ids == int(_CVT_EMBEDDING_ID)
        return (~is_cvt).to(device=device, dtype=dtype)

    @staticmethod
    def _pool_active_nodes(
        *,
        node_tokens: torch.Tensor,
        node_batch: torch.Tensor,
        active_nodes: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        device = node_tokens.device
        dtype = node_tokens.dtype
        out = torch.zeros((num_graphs, node_tokens.size(-1)), device=device, dtype=dtype)
        active_mask = active_nodes.to(device=device, dtype=torch.bool)
        if not torch.any(active_mask):
            return out
        active_tokens = node_tokens[active_mask]
        active_batch = node_batch[active_mask]
        out.index_add_(0, active_batch, active_tokens)
        counts = torch.bincount(active_batch, minlength=num_graphs).to(device=device, dtype=dtype).clamp(min=_ONE)
        return out / counts.unsqueeze(-1)

    @classmethod
    def _build_state_inputs(
        cls,
        *,
        node_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        state_vec: torch.Tensor,
        active_nodes: torch.Tensor,
        node_batch: torch.Tensor,
        include_active_nodes: bool,
    ) -> torch.Tensor:
        if include_active_nodes:
            num_graphs = int(state_vec.size(0))
            current_entities = cls._pool_active_nodes(
                node_tokens=node_tokens,
                node_batch=node_batch,
                active_nodes=active_nodes,
                num_graphs=num_graphs,
            )
            return torch.cat([question_tokens, current_entities, state_vec], dim=-1)
        return torch.cat([question_tokens, state_vec], dim=-1)

    @staticmethod
    def _gather_relation_tokens(
        *,
        relation_tokens: torch.Tensor,
        actions: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        device = relation_tokens.device
        dtype = relation_tokens.dtype
        out = torch.zeros((num_graphs, relation_tokens.size(-1)), device=device, dtype=dtype)
        valid_action = actions >= _ZERO
        if not torch.any(valid_action):
            return out
        action_ids = actions[valid_action]
        out[valid_action] = relation_tokens.index_select(0, action_ids)
        return out

    def _backward_valid_edges(self, state: Any) -> torch.Tensor:
        return self.env.backward_edge_mask(state)

    def _compute_log_pb(
        self,
        *,
        state: Any,
        graph: dict[str, torch.Tensor],
        state_vec: torch.Tensor,
        actions: torch.Tensor,
        autocast_ctx: contextlib.AbstractContextManager,
        num_graphs: int,
        edge_cache: Optional[RolloutEdgeCache] = None,
    ) -> torch.Tensor:
        valid_edges = self._backward_valid_edges(state)
        valid_action = actions >= _ZERO
        if valid_action.any():
            action_ids = actions[valid_action]
            if action_ids.numel() > 0 and int(action_ids.max().item()) >= int(valid_edges.numel()):
                raise ValueError("Action id out of range for backward valid_edges.")
            action_mask = torch.zeros_like(valid_edges, dtype=torch.bool)
            action_mask[action_ids] = True
            valid_edges = valid_edges | action_mask
        relation_logits, edge_scores, stop_logits = self._score_edges_backward(
            graph,
            state_vec,
            active_nodes=state.active_nodes,
            autocast_ctx=autocast_ctx,
            edge_cache=edge_cache,
        )
        edge_logits = self._policy_logits(
            valid_edges=valid_edges,
            edge_logits=edge_scores,
            autocast_ctx=autocast_ctx,
        )
        allow_stop = torch.ones(num_graphs, device=edge_logits.device, dtype=torch.bool)
        log_prob_edge, log_prob_stop, _, _ = compute_factorized_log_probs(
            relation_logits=relation_logits,
            edge_logits=edge_logits,
            edge_relations=graph["edge_relations"],
            edge_batch=graph["edge_batch"],
            valid_edges=valid_edges,
            num_graphs=num_graphs,
            temperature=self.backward_temperature,
            stop_logits=stop_logits,
            allow_stop=allow_stop,
        )
        log_pb = self._select_log_prob(log_prob_edge, actions)
        if log_pb.numel() == 0:
            return log_pb
        if log_prob_stop is None:
            return log_pb
        edge_relations = graph["edge_relations"]
        if edge_relations.numel() == 0:
            stop_mask = actions == STOP_RELATION
            return torch.where(stop_mask, log_prob_stop, log_pb)
        stop_mask = actions == STOP_RELATION
        if valid_action.any():
            action_ids = actions[valid_action]
            if action_ids.numel() > 0 and int(action_ids.max().item()) >= int(edge_relations.numel()):
                raise ValueError("Action id out of range for edge_relations in log_pb computation.")
            rel_ids = edge_relations.index_select(0, action_ids)
            stop_edge = rel_ids == STOP_EDGE_RELATION
            stop_mask = stop_mask.clone()
            stop_mask[valid_action] = stop_mask[valid_action] | stop_edge
        log_pb = torch.where(stop_mask, log_prob_stop, log_pb)
        return log_pb

    @staticmethod
    def _select_log_prob(log_prob_edge: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(actions.size(0), device=log_prob_edge.device, dtype=log_prob_edge.dtype)
        valid_action = actions >= _ZERO
        if valid_action.any():
            out[valid_action] = log_prob_edge.index_select(0, actions[valid_action])
        return out

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
    def _resolve_active_stop_locals(
        *,
        active_nodes: torch.Tensor,
        node_is_answer: torch.Tensor,
        node_ptr: torch.Tensor,
        node_batch: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_graphs = int(node_ptr.numel() - 1)
        num_nodes_total = int(node_ptr[-1].item()) if node_ptr.numel() > 0 else 0
        sentinel = num_nodes_total + _ONE
        out = torch.full((num_graphs,), sentinel, device=node_ptr.device, dtype=dtype)
        active_idx = torch.nonzero(active_nodes, as_tuple=False).view(-1)
        if active_idx.numel() == 0:
            return out, out != sentinel
        active_batch = node_batch[active_idx]
        local_idx = active_idx - node_ptr[active_batch]
        out.scatter_reduce_(0, active_batch, local_idx, reduce="amin", include_self=True)
        has_active = out != sentinel
        answer_mask = node_is_answer[active_idx]
        if not torch.any(answer_mask):
            return out, has_active
        answer_idx = active_idx[answer_mask]
        answer_batch = node_batch[answer_idx]
        answer_local = answer_idx - node_ptr[answer_batch]
        answer_out = torch.full((num_graphs,), sentinel, device=node_ptr.device, dtype=dtype)
        answer_out.scatter_reduce_(0, answer_batch, answer_local, reduce="amin", include_self=True)
        has_answer = answer_out != sentinel
        return torch.where(has_answer, answer_out, out), has_active

    @staticmethod
    def _update_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        actions: torch.Tensor,
        done_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        edge_relations: torch.Tensor,
        node_ptr: torch.Tensor,
        active_nodes: torch.Tensor,
        node_batch: torch.Tensor,
        node_is_answer: torch.Tensor,
    ) -> torch.Tensor:
        done_mask = done_mask.to(device=stop_node_locals.device, dtype=torch.bool)
        needs_update = (~done_mask) & (stop_node_locals == _STOP_NODE_NONE)
        if not needs_update.any():
            return stop_node_locals
        updated = stop_node_locals.clone()

        stop_action = actions == STOP_RELATION
        update_stop = needs_update & stop_action
        if update_stop.any():
            stop_locals, has_active = GFlowNetActor._resolve_active_stop_locals(
                active_nodes=active_nodes,
                node_is_answer=node_is_answer,
                node_ptr=node_ptr,
                node_batch=node_batch,
                dtype=stop_node_locals.dtype,
            )
            update_stop = update_stop & has_active
            updated = torch.where(update_stop, stop_locals, updated)

        valid_action = actions >= _ZERO
        stop_edge = torch.zeros_like(actions, dtype=torch.bool)
        if valid_action.any():
            edge_ids = actions[valid_action]
            rel_ids = edge_relations[edge_ids]
            stop_edge[valid_action] = rel_ids == STOP_EDGE_RELATION
        update_edge = needs_update & stop_edge
        if update_edge.any():
            edge_ids = actions[update_edge]
            graph_ids = edge_batch[edge_ids]
            head_nodes = edge_index[0, edge_ids]
            local_idx = head_nodes - node_ptr[graph_ids]
            updated[update_edge] = local_idx
        return updated

    @staticmethod
    def _backfill_stop_nodes(*, stop_node_locals: torch.Tensor, state: Any) -> torch.Tensor:
        return GFlowNetActor._fill_stop_nodes(stop_node_locals=stop_node_locals, state=state, require_done=True)

    @staticmethod
    def _fill_stop_nodes(
        *,
        stop_node_locals: torch.Tensor,
        state: Any,
        require_done: bool,
    ) -> torch.Tensor:
        needs_backfill = stop_node_locals == _STOP_NODE_NONE
        if require_done:
            needs_backfill = needs_backfill & state.done
        if not needs_backfill.any():
            return stop_node_locals
        node_ptr = state.graph.node_ptr
        min_local, has_active = GFlowNetActor._resolve_active_stop_locals(
            active_nodes=state.active_nodes,
            node_is_answer=state.graph.node_is_answer,
            node_ptr=node_ptr,
            node_batch=state.graph.node_batch,
            dtype=stop_node_locals.dtype,
        )
        backfill_mask = needs_backfill & has_active
        return torch.where(backfill_mask, min_local, stop_node_locals)

    def _resolve_rollout_max_steps(self, max_steps_override: Optional[int]) -> int:
        if max_steps_override is None:
            return int(self.max_steps)
        override = int(max_steps_override)
        if override <= _ZERO:
            raise ValueError(f"max_steps_override must be > 0, got {override}.")
        return min(int(self.max_steps), override)

    def _finalize_stop_nodes(
        self,
        *,
        stop_node_locals: torch.Tensor,
        state: Any,
        force_stop_at_end: bool,
    ) -> torch.Tensor:
        if force_stop_at_end:
            return self._fill_stop_nodes(stop_node_locals=stop_node_locals, state=state, require_done=False)
        return self._backfill_stop_nodes(stop_node_locals=stop_node_locals, state=state)

__all__ = ["GFlowNetActor", "RolloutResult"]
