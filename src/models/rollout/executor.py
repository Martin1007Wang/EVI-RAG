from __future__ import annotations

import torch
from src.data.schema import RetrievalBatch
from src.models.policy import PolicyStepOutput
from src.models.reward import RewardModel
from src.models.state import State
from src.utils.graph_utils import compute_valid_backward_removals
from .sampling import ActionSampler, EdgeSampleResult
from .types import StepResult


class Executor:
    """Rollout executor: samples one environment step for a batch of graphs.

    Action type encoding
    --------------------
    0 → Expand  (add an edge to the active sub-graph)
    1 → Stop    (terminate the trajectory, collect reward)
    """

    def __init__(
        self,
        *,
        expand_budget: int,
        retrieval_batch: RetrievalBatch,
        reward_model: RewardModel,
        sampler: ActionSampler,
    ) -> None:
        self.expand_budget = int(expand_budget)
        self.batch = retrieval_batch
        self.reward_model = reward_model
        self.sampler = sampler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_step(
        self,
        *,
        num_expands: int,
        step_out: PolicyStepOutput,
        state: State,
        active: torch.Tensor,      # (B,) bool — graphs still running
        temperature: float,
    ) -> StepResult:
        num_graphs = int(active.shape[0])
        device = active.device

        # ── 1. Which graphs have at least one candidate edge? ──────────
        has_valid_edges = (
            torch.bincount(step_out.candidates.batch_index, minlength=num_graphs) > 0
        )

        # ── 2. Graphs that have reached the horizon ────────────────────
        #   For these, Stop is the ONLY legal action; P_F(Stop|horizon) = 1.
        horizon_stop = active & (num_expands >= self.expand_budget)

        # ── 3. Build behavior & target logits ─────────────────────────
        #   behavior_logits : used for sampling (temperature-scaled)
        #   target_logits   : used for computing log P_F (NOT temperature-scaled)
        #   The key distinction: horizon_stop is an *environment* constraint and
        #   must be reflected in BOTH distributions identically so that the
        #   sampler always stops at the horizon AND the logged log P_F = 0
        #   (i.e., Stop has probability 1 at the horizon).
        behavior_type_logits, target_type_logits = _compute_type_logits(
            raw_logits=step_out.type_logits,
            active=active,
            has_valid_edges=has_valid_edges,
            horizon_stop=horizon_stop,
            temperature=temperature,
            device=device,
        )

        # ── 4. Sample action type ──────────────────────────────────────
        action_type, type_log_prob = self.sampler.sample_action_types(
            behavior_logits=behavior_type_logits,
            target_logits=target_type_logits,
            step_mask=active,
            num_expands=num_expands,
            retrieval_batch=self.batch,
            state=state,
            candidates=step_out.candidates,
        )
        # type_log_prob[i] = log P_F^target(action_type[i] | s_t, graph i)

        expand_mask = (action_type == 0) & active
        stop_mask   = (action_type == 1) & active

        # ── 5. Allocate output buffers ─────────────────────────────────
        log_pf           = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        log_pb           = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        terminal_rewards = torch.zeros(num_graphs, dtype=torch.float32, device=device)
        selected_edge_ids = torch.full(
            (num_graphs,), -1, dtype=torch.long, device=device
        )

        # ── 6. Expand branch ──────────────────────────────────────────
        if expand_mask.any():
            expand_graph_ids = torch.nonzero(expand_mask, as_tuple=False).view(-1)
            er: EdgeSampleResult = self.sampler.sample_expand_edge(
                candidates=step_out.candidates,
                expand_graph_ids=expand_graph_ids,
                temperature=temperature,
                num_expands=num_expands,
                retrieval_batch=self.batch,
                state=state,
            )
            selected_edge_ids[expand_mask] = er.chosen_edges.to(torch.long)

            # log P_F(expand, edge | s_t) = log P_F(type=expand) + log P_F(edge | expand)
            log_pf[expand_mask] = (
                type_log_prob[expand_mask] + er.edge_log_prob
            ).to(torch.float32)

            # Mutate state *after* recording log P_F
            state.apply_expansion(
                chosen_edges=er.chosen_edges,
                src=self.batch.edge_index[0],
                dst=self.batch.edge_index[1],
            )

            # Uniform backward policy: log P_B = -log(|removable edges after expansion|)
            log_pb[expand_mask] = _compute_expand_log_pb(
                state=state,
                retrieval_batch=self.batch,
                expand_mask=expand_mask,
                expand_graph_ids=expand_graph_ids,
                num_graphs=num_graphs,
            )

        # ── 7. Stop branch ────────────────────────────────────────────
        if stop_mask.any():
            # General case: log P_F = log P_F^target(Stop | s_t)
            log_pf[stop_mask] = type_log_prob[stop_mask].to(torch.float32)

            # Horizon override: environment *forces* Stop → P_F(Stop) = 1 → log = 0.
            # Apply AFTER the general assignment so the override wins.
            #
            # NOTE: Because _compute_type_logits sets Expand to -inf for
            # horizon_stop graphs in BOTH behavior and target logits, the
            # softmax already yields log P_F(Stop) = 0.  The explicit override
            # below is a defensive guard against any numerical imprecision in
            # the sampler (e.g., log(softmax([-inf, 0])) returning -1e-7).
            horizon_forced = stop_mask & horizon_stop
            if horizon_forced.any():
                log_pf[horizon_forced] = 0.0

            # Collect rewards for all stopping graphs in one model call
            reward_vals = self.reward_model(
                retrieval_batch=self.batch,
                active_nodes=state.active_nodes,
                active_edges=state.active_edges,
                state=state,
            )
            terminal_rewards[stop_mask] = reward_vals[stop_mask].to(torch.float32)

        return StepResult(
            log_pf=log_pf,
            log_pb=log_pb,
            stop_mask=stop_mask,
            terminal_log_rewards=terminal_rewards,
            selected_edge_ids=selected_edge_ids,
        )


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------


def _compute_type_logits(
    *,
    raw_logits: torch.Tensor,      # (B, 2)  — policy output, unscaled
    active: torch.Tensor,          # (B,)    bool
    has_valid_edges: torch.Tensor, # (B,)    bool
    horizon_stop: torch.Tensor,    # (B,)    bool
    temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (behavior_logits, target_logits), both shape (B, 2).

    Slot 0 = Expand, slot 1 = Stop.

    Masking rules
    -------------
    Expand is *forbidden* when:
      - the graph has no candidate edges, OR
      - the graph is inactive, OR
      - the horizon has been reached (environment constraint).

    Stop is *forbidden* only when the graph is inactive.

    Critically, horizon_stop is applied to BOTH behavior and target logits.
    This ensures:
      (a) The sampler always picks Stop at the horizon.
      (b) The target distribution also assigns probability 1 to Stop, so
          log P_F(Stop|horizon) = log(softmax([-inf, raw_stop])[1]) = 0.
    """
    # One-hot masks for the two action slots
    expand_slot = torch.tensor([[True, False]], dtype=torch.bool, device=device)
    stop_slot   = torch.tensor([[False, True]], dtype=torch.bool, device=device)

    expand_forbidden = ~has_valid_edges | ~active | horizon_stop
    stop_forbidden   = ~active

    # Combined mask: True iff that (graph, slot) combination is illegal
    mask = (
        (expand_forbidden.unsqueeze(1) & expand_slot)
        | (stop_forbidden.unsqueeze(1) & stop_slot)
    )  # (B, 2)

    # behavior_logits: temperature-scaled (for sampling)
    behavior_logits = (raw_logits / temperature).masked_fill(mask, float("-inf"))
    # target_logits:  unscaled (for computing the *true* policy log-prob)
    target_logits   = raw_logits.masked_fill(mask, float("-inf"))

    # Inactive graphs: force a degenerate distribution so downstream code
    # can safely call softmax/log_softmax without NaN.
    #   Expand slot → -inf  (never sampled)
    #   Stop  slot  →  0    (log_softmax = 0, prob = 1)
    inactive_fallback = torch.stack(
        [
            torch.full((active.shape[0],), float("-inf"), device=device, dtype=behavior_logits.dtype),
            torch.zeros(active.shape[0], device=device, dtype=behavior_logits.dtype),
        ],
        dim=1,
    )  # (B, 2)

    inactive = (~active).unsqueeze(1)
    behavior_logits = torch.where(inactive, inactive_fallback, behavior_logits)
    target_logits   = torch.where(
        inactive,
        inactive_fallback.to(dtype=target_logits.dtype),
        target_logits,
    )

    return behavior_logits, target_logits


def _compute_expand_log_pb(
    *,
    state: State,
    retrieval_batch: RetrievalBatch,
    expand_mask: torch.Tensor,     # (B,) bool
    expand_graph_ids: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Uniform backward policy: log P_B(expand | s_{t+1}) = -log(removable_count).

    Called *after* state.apply_expansion so removable_counts reflects the
    post-expansion graph.
    """
    _, removable_counts = compute_valid_backward_removals(
        active_nodes=state.active_nodes,
        active_edges=state.active_edges,
        edge_index=retrieval_batch.edge_index,
        is_anchor_mask=retrieval_batch.is_anchor_mask,
        node_batch=retrieval_batch.batch,
        edge_batch=retrieval_batch.edge_batch,
        num_graphs=num_graphs,
    )
    removable = removable_counts[expand_mask]
    if (removable < 1).any():
        bad = expand_graph_ids[
            torch.nonzero(removable < 1, as_tuple=False).view(-1)
        ]
        raise RuntimeError(
            f"removable_counts < 1 after expansion for graph_ids={bad.tolist()}. "
            "The expanded edge cannot be removed; the backward policy is undefined."
        )
    return (-torch.log(removable.float())).to(torch.float32)


__all__ = ["Executor"]
