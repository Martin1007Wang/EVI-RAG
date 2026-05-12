# Budgeted Successor Planner

## Main-Path Change

The previous Weaver policy path used a learned terminal head and a final
RelationResidualEdgeScorer-derived edge policy, with BDB traces as the active
training objective. That setup compared stop and expand through separate
mechanisms and did not directly train the edge policy to represent
budget-conditioned successor value.

The new main path is:

1. FeatureEncoder builds static graph/query features.
2. FrontierBuilder enumerates legal directed boundary expansions.
3. EvidenceStateEncoder encodes the parent state `s` and each successor state
   `s+e`.
4. BudgetedSuccessorPolicy emits `T_theta(s)`, `G_theta(s,e,b)`, and
   `V_theta(s,b)=logsumexp(T_theta(s), G_theta(s,e,b))`.
5. BudgetedLexicographicOracle computes exact budgeted teacher targets for the
   current training state.
6. BudgetedFlowDistillLoss trains policy KL plus terminal/value Huber losses.

RelationResidualEdgeScorer remains available as an ablation and diagnostic
component, but it is no longer the configured final edge policy. Stop is no
longer an independent sigmoid hazard gate; it is one action in the same
softmax/log-value scale as frontier expansion.

## Mathematical Target

For state `s=(V_s,E_s,boundary)` and remaining budget
`b=B-|E_s\E0|`, terminal utility is:

```text
C(s)=sum_{a in A*} 1[
    a in active_nodes(s)
    and exists directed path from any anchor in A0 to a using active_edges(s)
]

k(s)=|E_s\E0|
J(s)=eta*C(s)-lambda*k(s)-zeta*1[C(s)==0]
```

The reward config enforces the lexicographic condition through
`eta > lambda * B`.

Budgeted flow is:

```text
F*(s,0)=exp(J(s))

F*(s,b)=exp(J(s)) + sum_{e in frontier(s)}
    F*(s+e,b-1) * P_B(s|s+e)

V*(s,b)=log F*(s,b)
```

Action targets are:

```text
pi*(stop|s,b)=exp(J(s)-V*(s,b))

pi*(e|s,b)=exp(V*(s+e,b-1)+logP_B(s|s+e)-V*(s,b))
```

The model mirrors this normalization:

```text
pi_theta = softmax([T_theta(s), G_theta(s,e,b) for e in frontier(s)])
V_theta(s,b)=logsumexp(T_theta(s), G_theta(s,e,b))
```

## Training Objective

The active configured loss is:

```text
L_policy   = KL(pi_star || pi_theta)
L_terminal = Huber(T_theta(s), J(s))
L_value    = Huber(V_theta(s,b), V*(s,b))

L = L_policy + alpha*L_terminal + beta*L_value
```

Defaults:

```text
alpha = 1.0
beta  = 0.5
```

The rollout trace now stores budgeted-flow losses and oracle diagnostics instead
of making rollout-first BDB the default objective. BDB code remains present for
compatibility and ablations.

Training states include root/model-rollout states from the normal rollout
driver. When `planner.include_oracle_prefix_states=true`, each supervised state
also adds one oracle-prefix successor selected by the oracle's highest-probability
edge when budget remains. Those prefix losses are averaged with the current
state losses so the middle of the oracle path receives direct supervision.

## Label Leakage Boundary

Training:

Gold answers are used only inside `BudgetedLexicographicOracle` through the
existing lexicographic reward evaluator. This computes `J(s)`, `V*(s,b)`, and
`pi*` as teacher targets.

Inference:

The policy input path does not consume gold answers, `C(s)`, `J(s)`, `V*(s,b)`,
path labels, or target labels. BudgetedSuccessorPolicy receives only question
and graph features, current state encoding, successor state encoding, frontier
structure, raw semantic cosine scalars, DDE/static feature summaries, and
remaining budget.

## Oracle Exactness And Pruning

`BudgetedLexicographicOracle` memoizes states by:

```text
(static graph id, sorted active non-root edge ids, remaining budget)
```

Budget `B<=2` is exact over the full frontier. For budget 3 and larger, the
oracle can prune each state to `top_m_for_budget3` frontier edges using semantic
scores, and it reports:

```text
oracle/topm_coverage
oracle/topm_pruned_row_rate
```

## Diagnostics Added

Oracle:

```text
oracle/V_star_mean
oracle/terminal_J_mean
oracle/oracle_stop_prob_mean
oracle/oracle_edge_entropy
oracle/oracle_policy_kl
oracle/topm_coverage
oracle/topm_pruned_row_rate
```

Policy:

```text
policy/model_stop_prob
policy/budgeted_oracle_good_edge_policy_mass
policy/sampled_oracle_good_edge_rate
```

The validation metrics requested in the design remain owned by the existing
rollout/evaluation metric layer. The new policy exposes the required stop and
oracle-good-edge trace quantities so those aggregate metrics can be computed
without passing labels into inference.

## Result Status

Implemented and statically checked:

```text
src/weaver/planner/lexicographic_oracle.py
src/weaver/nn/budgeted_successor_policy.py
src/weaver/loss/budgeted_flow_distill.py
configs/model/weaver.yaml
```

Verification run:

```text
python -m py_compile ...
pytest -q tests/weaver/loss/test_budgeted_flow_distill_loss.py
```

The loss unit test passes. Full checkpoint comparison has not been run in this
change, so acceptance items involving old-full-checkpoint topK/mass,
best_of_k@8, and long-run KL descent still need an actual training/eval job.
