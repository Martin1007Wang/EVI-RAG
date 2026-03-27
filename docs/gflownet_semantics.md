# RankFlow / GFlowNet Semantics

This document describes the current algorithmic contract.

It intentionally avoids copying descriptive numeric values from experiment
bundles. If you need the canonical runnable settings, read these files
directly:

- `configs/model/gflownet.yaml`
- `configs/experiment/train_rankflow.yaml`
- `configs/experiment/rankflow.yaml`

## 1. Mainline in one sentence

The current RankFlow stack trains a strict successor-flow GFlowNet with
MS-SubTB, samples trajectories from a proposal policy that may differ from the
target policy, and interprets online rollouts plus replay as a coverage-mixture
for residual fitting rather than as an importance-corrected off-policy target.

## 2. State space

The search space is a prefix tree over graph-walk trajectories.

- The abstract root boundary is not materialized as an ordinary `SearchState`.
- Root actions choose a start node and are represented by
  `RootActionDistribution`.
- Non-root states are explicit prefix states stored in `SearchState`.
- `STOP` is an explicit action and absorbing prefixes store it directly.

This split matters because root actions are query-conditioned start selections,
while non-root actions are graph transitions plus `STOP`.

## 3. Target policy and target measure

### Root boundary

For each candidate start node, the target branch mass is

`rho_start([e_0]) + log F([e_0])`

and the root flow is the segment-wise `logsumexp` over those branch masses:

`log F(s_root) = logsumexp_{[e_0]} (rho_start([e_0]) + log F([e_0]))`

The target root action distribution is the normalized decomposition of that root
mass.

### Non-root move actions

For a graph move `a_t: s_t -> s_{t+1}`, the target branch log-mass is

`rho_move(s_t, a_t) + log F(s_{t+1})`

where `rho_move` contains the configured step-level reward terms.

### STOP action

For `STOP`, the target branch mass is anchored by the terminal reward semantics.
Depending on the answer-quotient settings, this can use the direct terminal
reward or an answer-sink reward together with an in-sink allocation factor.

### Reward decomposition

The target trajectory measure is always described by two pieces:

- `terminal_log_rewards`: terminal anchor
- `log_reward_steps`: per-step reward increments

The public knobs that change the target measure live under
`model.training_cfg`.

## 4. Target-side shaping

The current target-side shaping mechanism is answer-distance potential shaping.

- The start action gets a root reward equal to the chosen start state's
  potential.
- Each move step gets the potential difference
  `Phi(s_{t+1}) - Phi(s_t)`.

When answer-distance shaping is active, successful answer-reaching trajectories
keep the same total reward, while failed trajectories are penalized according to
remaining answer distance.

This is a target-measure change. It is part of the algebra seen by SubTB.

## 5. Proposal policy

The sampler may use a proposal policy `Q` that differs from the target policy
`P_F`.

### Root proposals

Root proposals start from the target root branch masses and may add proposal
priors from `model.action_prior_cfg`.

### Edge proposals

For non-root graph actions, the live proposal logits are built from three terms:

`logits_Q = logits_F + b_transition + alpha_t * b_prior`

where:

- `logits_F` are the strict successor-flow target branch masses
- `b_transition` is the learned proposal-only transition head bias from
  `model.policy_cfg.transition_head`
- `b_prior` is the heuristic proposal bias from `model.action_prior_cfg`
- `alpha_t` is the training-time proposal-prior scale from
  `model.training_cfg.action_prior_schedule`

The important boundary is:

- `b_transition` changes proposal sampling only
- `b_prior` changes proposal sampling only
- neither term enters the target-policy residual algebra

Evaluation and prediction stay on target-policy search.

## 6. Proposal-side guidance terms

`model.action_prior_cfg` controls proposal-only guidance components.

The config is now component-based rather than family-labeled. In practice there
are three groups of proposal terms:

- root / edge / stop strength knobs (`root_beta`, `edge_beta`, `stop_beta`)
- static feature weights (topology, node embedding, relation embedding,
  target-node, progress)
- optional dynamic guidance terms (intent alignment, shortest-path bonus,
  answer-distance progress)

Two answer-distance mechanisms now exist and must not be conflated:

- `model.action_prior_cfg.answer_distance_weight` is proposal-only guidance
- `model.training_cfg.potential_reward.answer_distance_weight` changes the target
  measure

If you change the first one, you change which constraints are visited. If you
change the second one, you change the target residuals themselves.

## 7. Coverage-measure interpretation

The implementation does not importance-reweight proposal-sampled trajectories
back to a target-policy expectation.

Instead, it minimizes SubTB residuals under the sampled coverage measure:

`L_nu(theta) = E_{tau ~ nu} [sum_{c in C(tau)} w(c) * Delta_c(tau; theta)^2]`

So proposal engineering changes coverage, not the definition of each residual.

## 8. Replay

Replay is treated as another coverage source.

- Online rollouts come from the current proposal sampler.
- Replay stores trajectory definitions, not frozen `log P_F` or `log F` tensors.
- Replayed trajectories are rebuilt under the current parameters before SubTB is
  evaluated.

The resulting optimization view is a coverage mixture:

`nu_mix = (1 - alpha_replay) * nu_online + alpha_replay * nu_replay`

This is not a second objective and not an importance-correction layer.

## 9. MS-SubTB objective

The main objective is a multi-scale SubTrajectory Balance loss over three
families of residuals.

### Root residual

`r_root = log F(s_root) + log P_F([e_0] | s_root) - rho_start([e_0]) - log F([e_0])`

### Pairwise residual

`r_pair(i, j) = f_i + sum_{t=i}^{j-1} p_t - sum_{t=i}^{j-1} rho_t - f_j`

### Terminal residual

`r_term(i) = f_i + sum_{t=i}^{T-1} p_t - (ell_term(tau) + sum_{t=i}^{T-1} rho_t)`

The component weights and length weighting are controlled only by
`model.training_cfg.subtb.*`.

## 10. Config surface by role

### Target measure

These knobs change the target distribution fitted by SubTB:

- `model.training_cfg.terminal_failure_log_reward`
- `model.training_cfg.step_log_penalty`
- `model.training_cfg.answer_stop_log_reward_bonus`
- `model.training_cfg.potential_reward.*`
- `model.training_cfg.answer_quotient.*`

### Proposal sampler

These knobs change sampling coverage without changing the target residual
definitions:

- `model.policy_cfg.transition_head.*`
- `model.action_prior_cfg.*`
- `model.training_cfg.action_prior_schedule.*`
- `model.training_cfg.sampling_temperature`
- `model.training_cfg.sampling_temperature_schedule.*`

### Coverage supply

These knobs change how much online and replay coverage is gathered:

- `model.training_cfg.rollouts_per_graph`
- `model.training_cfg.success_replay.*`

### Evaluation

These knobs change validation and final-eval search/reporting behavior:

- canonical answer-ranking validation and final eval both use
  `answer_posterior_backend: flow_frontier`
- `monte_carlo` remains for edge retrieval and optional diagnostics, not for the
  canonical answer-ranking checkpoint selector

- `model.eval_cfg.report_profile`
- `model.eval_cfg.answer_posterior_backend`
- `model.eval_cfg.monte_carlo.*`
- `model.eval_cfg.flow_frontier.*`

## 11. Canonical source of truth

To keep docs stable, this document explains mechanisms only.

- `configs/model/gflownet.yaml` defines the minimal public schema surface.
- `configs/experiment/train_rankflow.yaml` defines the canonical training bundle.
- `configs/experiment/rankflow.yaml` defines the canonical final-eval bundle.

If any numeric value in an older note disagrees with those files, treat the
config files as the source of truth.

## 12. Reading order in code

If you want to audit the implementation from config to loss, read files in this
order:

1. `src/models/configs/gflownet_training.py`
2. `src/models/configs/policy.py`
3. `src/models/configs/gflownet_eval.py`
4. `src/models/gflownet/types.py`
5. `src/models/gflownet/heuristics.py`
6. `src/models/gflownet/policy.py`
7. `src/models/gflownet/sampler.py`
8. `src/models/gflownet/losses.py`
9. `src/models/gflownet_module.py`
