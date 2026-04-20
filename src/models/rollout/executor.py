from __future__ import annotations

import os
from typing import Any, cast

import torch
from torch_scatter import scatter_sum

from src.data.schema import RetrievalBatch
from src.models.policy import Policy
from src.models.state import State
from src.utils.graph_utils import compute_valid_backward_removals
from src.utils.logging_utils import get_logger

from .sampling import ActionSampler
from .types import ROLLOUT_DTYPE, StepResult


log = get_logger(__name__)


class StepExecutor:
    def __init__(self, *, max_steps: int, terminal_backward_mode: str) -> None:
        self.max_steps = int(max_steps)
        self.terminal_backward_mode = terminal_backward_mode
        self._prior_probe_enabled = os.environ.get("EVI_PRIOR_PROBE", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._prior_probe_logged = False

    def execute_step(
        self,
        *,
        t: int,
        step_out: Any,
        state: State,
        active: torch.Tensor,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        backbone_static_context: Any | None,
        sampler: ActionSampler,
        temperature: float,
        collect_terminal_state: bool,
        pre_stop_nodes: torch.Tensor | None,
        pre_stop_edges: torch.Tensor | None,
        recorded_edge_traces: list[list[int]] | None,
    ) -> StepResult:
        num_graphs = int(active.shape[0])
        device = active.device
        src, dst = base_graph.edge_index[0], base_graph.edge_index[1]
        edge_batch_idx = base_graph.edge_batch

        valid_edges_mask = (
            state.active_nodes[src] | state.active_nodes[dst]
        ) & ~state.active_edges
        has_valid_edges = scatter_sum(
            valid_edges_mask.int(), edge_batch_idx, dim=0, dim_size=num_graphs
        ).bool()

        horizon_stop = active & (t >= self.max_steps)
        behavior_type_logits, target_type_logits = self._compute_type_logits(
            raw_logits=step_out.type_logits,
            active=active,
            has_valid_edges=has_valid_edges,
            horizon_stop=horizon_stop,
            temperature=temperature,
            device=device,
        )

        self._log_prior_probe_once(
            t=t,
            active=active,
            base_graph=base_graph,
            candidates=step_out.candidates,
        )

        action_type, type_log_prob = sampler.sample_action_types(
            behavior_logits=behavior_type_logits,
            target_logits=target_type_logits,
            step_mask=active,
            t=t,
            base_graph=base_graph,
            state=state,
            candidates=step_out.candidates,
        )

        log_pf = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)
        log_pb = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)
        log_shaping = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)
        terminal_rewards = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)
        terminal_stop_lpb = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)
        chosen_edge_ids = torch.full(
            (num_graphs,), -1, dtype=torch.long, device=device
        )
        chosen_relation_only_logits = torch.zeros(
            num_graphs, dtype=ROLLOUT_DTYPE, device=device
        )
        chosen_final_logits = torch.zeros(num_graphs, dtype=ROLLOUT_DTYPE, device=device)

        expand_mask = (action_type == 0) & active
        if expand_mask.any():
            (
                log_pf,
                log_pb,
                log_shaping,
                chosen_edge_ids,
                chosen_relation_only_logits,
                chosen_final_logits,
            ) = self._execute_expand(
                t=t,
                expand_mask=expand_mask,
                step_out=step_out,
                state=state,
                base_graph=base_graph,
                sampler=sampler,
                reward_model=reward_model,
                backbone_static_context=backbone_static_context,
                type_log_prob=type_log_prob,
                temperature=temperature,
                num_graphs=num_graphs,
                log_pf=log_pf,
                log_pb=log_pb,
                log_shaping=log_shaping,
                chosen_edge_ids=chosen_edge_ids,
                chosen_relation_only_logits=chosen_relation_only_logits,
                chosen_final_logits=chosen_final_logits,
                recorded_edge_traces=recorded_edge_traces,
            )

        stop_mask = (action_type == 1) & active
        if stop_mask.any():
            log_pf, log_pb, terminal_rewards, terminal_stop_lpb = self._execute_stop(
                stop_mask=stop_mask,
                step_out=step_out,
                state=state,
                policy=policy,
                base_graph=base_graph,
                reward_model=reward_model,
                type_log_prob=type_log_prob,
                collect_terminal_state=collect_terminal_state,
                pre_stop_nodes=pre_stop_nodes,
                pre_stop_edges=pre_stop_edges,
                log_pf=log_pf,
                log_pb=log_pb,
                terminal_rewards=terminal_rewards,
                terminal_stop_lpb=terminal_stop_lpb,
            )

        return StepResult(
            log_pf=log_pf,
            log_pb=log_pb,
            log_shaping=log_shaping,
            stop_mask=stop_mask,
            terminal_log_rewards=terminal_rewards,
            terminal_stop_log_pb=terminal_stop_lpb,
            chosen_edge_ids=chosen_edge_ids,
            chosen_relation_only_logits=chosen_relation_only_logits,
            chosen_final_logits=chosen_final_logits,
        )

    def _log_prior_probe_once(
        self,
        *,
        t: int,
        active: torch.Tensor,
        base_graph: RetrievalBatch,
        candidates: Any,
    ) -> None:
        if not self._prior_probe_enabled or self._prior_probe_logged or t != 0:
            return
        if active.numel() == 0 or not bool(active[0].item()):
            return

        graph_mask = candidates.batch_index == 0
        if not bool(graph_mask.any().item()):
            log.warning("Prior probe step0 graph=0 has no candidate edges.")
            self._prior_probe_logged = True
            return

        graph_logits = candidates.expand_logits[graph_mask].detach()
        graph_edge_ids = candidates.edge_ids[graph_mask]
        graph_positive_mask = base_graph.positive_edge_mask.index_select(
            0, graph_edge_ids
        ).detach()
        sorted_idx = graph_logits.argsort(descending=True)
        positive_ranks = torch.nonzero(
            graph_positive_mask[sorted_idx], as_tuple=False
        ).view(-1)
        positive_logits = graph_logits[graph_positive_mask]
        topk = min(5, int(sorted_idx.numel()))
        top_idx = sorted_idx[:topk]

        sample_id = None
        sample_id_value = getattr(base_graph, "sample_id", None)
        if isinstance(sample_id_value, (list, tuple)) and len(sample_id_value) > 0:
            sample_id = str(sample_id_value[0])
        elif isinstance(sample_id_value, str) and int(base_graph.num_graphs) == 1:
            sample_id = sample_id_value

        log.warning(
            "Prior probe step0 graph=0 sample_id=%s total_candidates=%d positive_edges=%d positive_ranks=%s",
            sample_id,
            int(graph_mask.sum().item()),
            int(graph_positive_mask.sum().item()),
            positive_ranks.cpu().tolist(),
        )
        log.warning(
            "Prior probe logits graph=0 mean=%.6f std=%.6f positive_logits=%s top5_logits=%s top5_edge_ids=%s",
            float(graph_logits.mean().item()),
            float(graph_logits.std(unbiased=False).item()),
            positive_logits.cpu().tolist(),
            graph_logits[top_idx].cpu().tolist(),
            graph_edge_ids[top_idx].cpu().tolist(),
        )
        self._prior_probe_logged = True

    def _execute_expand(
        self,
        *,
        t: int,
        expand_mask: torch.Tensor,
        step_out: Any,
        state: State,
        base_graph: RetrievalBatch,
        sampler: ActionSampler,
        reward_model: Any,
        backbone_static_context: Any | None,
        type_log_prob: torch.Tensor,
        temperature: float,
        num_graphs: int,
        log_pf: torch.Tensor,
        log_pb: torch.Tensor,
        log_shaping: torch.Tensor,
        chosen_edge_ids: torch.Tensor,
        chosen_relation_only_logits: torch.Tensor,
        chosen_final_logits: torch.Tensor,
        recorded_edge_traces: list[list[int]] | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        edge_batch_idx = base_graph.edge_batch
        active_candidates = step_out.candidates.filter(
            expand_mask[step_out.candidates.batch_index]
        )
        expand_graph_ids = torch.nonzero(expand_mask, as_tuple=False).view(-1)

        candidate_counts = scatter_sum(
            torch.ones_like(active_candidates.batch_index, dtype=torch.int),
            active_candidates.batch_index,
            dim=0,
            dim_size=num_graphs,
        )
        invalid_expand = expand_mask & candidate_counts.eq(0)
        if invalid_expand.any():
            bad = torch.nonzero(invalid_expand, as_tuple=False).view(-1)
            raise RuntimeError(
                f"Expand without valid candidates for graph ids {bad.tolist()}."
            )

        expand_remap = torch.empty(
            num_graphs, dtype=torch.long, device=expand_graph_ids.device
        )
        expand_remap[expand_graph_ids] = torch.arange(
            expand_graph_ids.numel(),
            dtype=torch.long,
            device=expand_graph_ids.device,
        )
        compact_batch_idx = expand_remap[active_candidates.batch_index]

        if not bool(torch.isfinite(active_candidates.expand_logits).all().item()):
            bad_local = torch.nonzero(
                ~torch.isfinite(active_candidates.expand_logits), as_tuple=False
            ).view(-1)
            bad_edges = active_candidates.edge_ids.index_select(0, bad_local)
            bad_graphs = active_candidates.batch_index.index_select(
                0, bad_local
            ).unique(sorted=True)
            raise ValueError(
                f"Non-finite expand-edge logits at step={t}, "
                f"bad_graph_ids={bad_graphs.detach().cpu().tolist()}, "
                f"bad_edge_ids={bad_edges.detach().cpu().tolist()}"
            )

        (
            chosen_edges,
            chosen_local_ids,
            edge_log_prob,
            chosen_candidate_positions,
        ) = sampler.sample_expand_edge(
            candidates=active_candidates,
            expand_graph_ids=expand_graph_ids,
            compact_batch_idx=compact_batch_idx,
            temperature=temperature,
            t=t,
            base_graph=base_graph,
            state=state,
        )

        if active_candidates.relation_only_logits is None:
            raise RuntimeError(
                "CandidateEdges.relation_only_logits is required for replay-quality metrics."
            )
        chosen_edge_ids[expand_mask] = chosen_edges.to(dtype=torch.long)
        chosen_relation_only_logits[expand_mask] = active_candidates.relation_only_logits.index_select(
            0, chosen_candidate_positions
        ).to(dtype=ROLLOUT_DTYPE)
        chosen_final_logits[expand_mask] = active_candidates.expand_logits.index_select(
            0, chosen_candidate_positions
        ).to(dtype=ROLLOUT_DTYPE)

        if recorded_edge_traces is not None:
            for graph_id, local_id in zip(
                expand_graph_ids.tolist(), chosen_local_ids.tolist()
            ):
                recorded_edge_traces[int(graph_id)].append(int(local_id))

        edges_before = state.active_edges.clone()
        state.apply_expansion(
            chosen_edges=chosen_edges,
            src=base_graph.edge_index[0],
            dst=base_graph.edge_index[1],
        )
        shaping_fn = getattr(reward_model, "step_shaping", None)
        if callable(shaping_fn):
            if backbone_static_context is None:
                raise RuntimeError(
                    "Relation potential shaping requires backbone_static_context."
                )
            shaping_vals = cast(
                torch.Tensor,
                shaping_fn(
                    base_graph,
                    edges_before,
                    state.active_edges,
                    query_h=backbone_static_context.query_h,
                    rel_h=backbone_static_context.rel_h,
                ),
            )
            log_shaping[expand_mask] = shaping_vals[expand_mask].to(dtype=ROLLOUT_DTYPE)

        log_pf[expand_mask] = (type_log_prob[expand_mask] + edge_log_prob).to(
            dtype=ROLLOUT_DTYPE
        )

        _, removable_counts = compute_valid_backward_removals(
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
            edge_index=base_graph.edge_index,
            is_anchor_mask=base_graph.is_anchor_mask,
            node_batch=base_graph.batch,
            edge_batch=edge_batch_idx,
            num_graphs=num_graphs,
        )
        removable = removable_counts[expand_mask]
        if (removable < 1).any():
            bad = expand_graph_ids[
                torch.nonzero(removable < 1, as_tuple=False).view(-1)
            ]
            raise RuntimeError(
                f"removable_counts < 1 after expansion for graph_ids={bad.tolist()}."
            )
        log_pb[expand_mask] = (-torch.log(removable.float())).to(dtype=ROLLOUT_DTYPE)
        return (
            log_pf,
            log_pb,
            log_shaping,
            chosen_edge_ids,
            chosen_relation_only_logits,
            chosen_final_logits,
        )

    def _execute_stop(
        self,
        *,
        stop_mask: torch.Tensor,
        step_out: Any,
        state: State,
        policy: Policy,
        base_graph: RetrievalBatch,
        reward_model: Any,
        type_log_prob: torch.Tensor,
        collect_terminal_state: bool,
        pre_stop_nodes: torch.Tensor | None,
        pre_stop_edges: torch.Tensor | None,
        log_pf: torch.Tensor,
        log_pb: torch.Tensor,
        terminal_rewards: torch.Tensor,
        terminal_stop_lpb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_batch_idx = base_graph.edge_batch
        if (
            collect_terminal_state
            and pre_stop_nodes is not None
            and pre_stop_edges is not None
        ):
            node_stop = stop_mask[base_graph.batch]
            edge_stop = stop_mask[edge_batch_idx]
            pre_stop_nodes[node_stop] = state.active_nodes[node_stop].detach()
            pre_stop_edges[edge_stop] = state.active_edges[edge_stop].detach()

        stop_log_pb_vals = self._terminal_backward_log_prob(
            policy=policy,
            base_graph=base_graph,
            state=state,
            step_output=step_out,
        )

        log_pf[stop_mask] = type_log_prob[stop_mask].to(dtype=ROLLOUT_DTYPE)
        log_pb[stop_mask] = stop_log_pb_vals[stop_mask].to(dtype=ROLLOUT_DTYPE)
        terminal_stop_lpb[stop_mask] = log_pb[stop_mask]

        reward_vals = reward_model(
            base_graph=base_graph,
            active_nodes=state.active_nodes,
            active_edges=state.active_edges,
        )
        terminal_rewards[stop_mask] = reward_vals[stop_mask].to(dtype=ROLLOUT_DTYPE)
        return log_pf, log_pb, terminal_rewards, terminal_stop_lpb

    @staticmethod
    def _compute_type_logits(
        *,
        raw_logits: torch.Tensor,
        active: torch.Tensor,
        has_valid_edges: torch.Tensor,
        horizon_stop: torch.Tensor,
        temperature: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expand_slot = torch.tensor([[True, False]], dtype=torch.bool, device=device)
        stop_slot = torch.tensor([[False, True]], dtype=torch.bool, device=device)

        sampling_expand_forbidden = ~has_valid_edges | ~active | horizon_stop
        scoring_expand_forbidden = ~has_valid_edges | ~active
        stop_forbidden = ~active

        sampling_mask = (sampling_expand_forbidden.unsqueeze(1) & expand_slot) | (
            stop_forbidden.unsqueeze(1) & stop_slot
        )
        scoring_mask = (scoring_expand_forbidden.unsqueeze(1) & expand_slot) | (
            stop_forbidden.unsqueeze(1) & stop_slot
        )

        behavior_logits = (raw_logits / temperature).masked_fill(
            sampling_mask, float("-inf")
        )
        target_logits = raw_logits.masked_fill(scoring_mask, float("-inf"))

        inactive_fallback = torch.stack(
            [
                torch.full(
                    (active.shape[0],),
                    float("-inf"),
                    device=device,
                    dtype=behavior_logits.dtype,
                ),
                torch.zeros(
                    active.shape[0], device=device, dtype=behavior_logits.dtype
                ),
            ],
            dim=1,
        )
        behavior_logits = torch.where(
            (~active).unsqueeze(1), inactive_fallback, behavior_logits
        )
        target_logits = torch.where(
            (~active).unsqueeze(1),
            inactive_fallback.to(dtype=target_logits.dtype),
            target_logits,
        )
        return behavior_logits, target_logits

    def _terminal_backward_log_prob(
        self,
        *,
        policy: Policy,
        base_graph: RetrievalBatch,
        state: State,
        step_output: Any,
    ) -> torch.Tensor:
        batch_size = int(base_graph.ptr.numel()) - 1
        device = base_graph.node_tokens.device
        if self.terminal_backward_mode == "deterministic":
            return torch.zeros(batch_size, device=device, dtype=ROLLOUT_DTYPE)
        log_prob_fn = getattr(policy, "terminal_backward_log_prob", None)
        if not callable(log_prob_fn):
            raise ValueError(
                "terminal_backward_mode='policy' requires "
                "policy.terminal_backward_log_prob(...)."
            )
        stop_log_pb = log_prob_fn(
            base_graph=base_graph,
            state=state.as_policy_input(),
            step_output=step_output,
        )
        if not isinstance(stop_log_pb, torch.Tensor):
            raise TypeError(
                "policy.terminal_backward_log_prob(...) must return a torch.Tensor."
            )
        if stop_log_pb.shape != (batch_size,):
            raise ValueError(
                "policy.terminal_backward_log_prob(...) must return shape "
                f"({batch_size},), got {tuple(stop_log_pb.shape)}."
            )
        if not torch.isfinite(stop_log_pb).all():
            bad = torch.nonzero(~torch.isfinite(stop_log_pb), as_tuple=False).view(-1)
            raise ValueError(
                "policy.terminal_backward_log_prob(...) returned non-finite "
                f"values for graph ids {bad.tolist()}."
            )
        return stop_log_pb.to(device=device, dtype=ROLLOUT_DTYPE)


__all__ = ["StepExecutor"]
