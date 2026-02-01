Audit Notes: Methodology vs Code Alignment
==========================================

Scope
-----
This note records the current alignment gaps between the theoretical claims
in the draft methodology and the implemented code (as of the latest changes).

Key Alignment Gaps / Clarifications Needed
------------------------------------------
1) Reference process vs time-reversal prior
   - Draft defines P0 as out-degree uniform random walk.
   - Code uses indegree-based prior (time-reversal) as p0_cfg.mode=indegree.
   - Action: clarify in paper that the implemented prior is the time-reversal
     of P0, not P0 itself.

2) PB prior consistency under dynamic constraints
   - Code PB is uniform over backward edges.
   - With avoid_revisit or pb_edge_dropout enabled, the backward candidate set
     is dynamically masked, while p0 prior remains static indegree.
   - Action: either disable these constraints for strict DB alignment, or
     explicitly state this as an approximation.

3) Finite-horizon grounding uses finite negative value (not -inf)
   - Code uses dead_end_log_reward (default -10) at terminal failure.
   - Draft uses -infinity for time-out grounding.
   - Action: either update paper to "finite negative constant" or set
     dead_end_log_reward to a large negative value (e.g., -100).

4) Out-degree prior ablation is a constant offset per decision
   - p0_cfg.mode=degree uses -log(out_degree(s_t)) which is constant across
     candidate edges at a given decision and cancels in softmax.
   - It does not induce a per-edge rich-get-richer bias in one step; any
     hub-seeking behavior is a path-level effect.
   - Action: add a preferential-attachment prior (+log indegree of next node)
     for a true rich-get-richer baseline.

Current Recommended Experimental Variants
-----------------------------------------
- train_dual_flow_p0_indegree: learned residual + indegree prior (main claim)
- train_dual_flow_p0_none: no prior
- train_dual_flow_p0_outdegree: outdegree constant prior (ablation with caveat)
- train_dual_flow_p0_preferential: preferential attachment (+log indegree)
- train_dual_flow_p0_semantic: semantic prior
