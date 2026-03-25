# GFlowNet Algorithm Overview

This document is the reader-first description of the current answer-reachability
GFlowNet in this repo.

At a high level, the model defines a probability flow over graph-search
prefixes. A trajectory starts from an abstract root boundary, chooses a start
node, repeatedly takes either a graph-edge move or an explicit `STOP` action,
and then receives terminal and step-level rewards. Training minimizes an
MS-SubTB objective built from root, pairwise, and terminal consistency
residuals.

If you only want the main idea, remember these three points:

1. The state space is a prefix tree, not just a set of current nodes.
2. The base reward is terminal correctness plus a fixed per-move log cost.
3. The training objective is just SubTB on that reward measure; there are no
   extra oracle-imitation or success-classification losses in the current path.

## 1. One rollout from start to finish

One training rollout looks like this:

1. Build an abstract root boundary state from the question and graph context.
2. Predict the root boundary flow `log F(s_root)`.
3. Predict a start-node action distribution `P_F([e_0] | s_root)`.
4. Sample a start node.
5. From the resulting prefix state, repeatedly choose either:
   - a graph-edge expansion action, or
   - the explicit `STOP` action.
6. Once the rollout stops or can no longer expand, assign terminal reward and
   accumulate any step-level rewards.
7. Fit the resulting trajectory measure with MS-SubTB.

Here is the smallest useful mental picture:

```text
root -> [e0] -> [e0, r1, e1] -> [e0, r1, e1, r2, e2] -> [e0, r1, e1, r2, e2, STOP]
```

- `root` is an abstract boundary, not a graph node.
- `[e0]` is the start state chosen from the root action distribution.
- Each graph move appends `relation, node` to the prefix.
- `STOP` is a real action and is explicitly appended for absorbing states.

In this example:

- `terminal_num_steps = 2` because the rollout took two graph moves.
- `termination_action_steps = 3` because the third action was `STOP`.

Those two counters are intentionally different.

## 2. Search space and runtime objects

The runtime uses a small set of objects repeatedly. Their roles are:

| Object | Meaning |
| --- | --- |
| `RootState` | Explicit abstract root boundary state |
| `RootActionDistribution` | Outgoing start-node actions from the root |
| `SearchState` | Exact graph-prefix state for non-root search |
| `ForwardActionDistribution` | Legal graph moves plus one `STOP` action per active state |
| `TrajectoryGFNSampleBatch` | Rollout tensors consumed by training |

### Root boundary

The root boundary is modeled separately from ordinary prefix states.

- `graph_log_z` stores `log F(s_root)`.
- `log_probs` stores the root action probabilities
  `log P_F([e_0] | s_root)`.
- `log_flows` stores the child start-state values `log F([e_0])`.

This matters because the model learns root consistency explicitly instead of
deriving root flow from start states.

### Prefix state

For non-root search, the semantic state is the exact trajectory prefix. The
important `SearchState` fields are:

- `path_token_ids`: exact prefix encoded as
  `node, relation, node, relation, ...`
- `current_nodes`: current terminal node of each prefix
- `num_steps`: number of graph moves already taken
- `done_mask`: rows that are inactive in the current batched call
- `absorbing_mask`: rows whose prefix already contains explicit `STOP`
- `control_state`: cached recurrent controller state used for fast rescoring

The key distinction is:

- `done_mask` means "do not expand this row right now"
- `absorbing_mask` means "this row is a true STOP-terminated state"

So a row can be inactive because it is padded or at a dead end without being a
true absorbing state.

### STOP semantics

`STOP` is not a bookkeeping trick. It is a real public action.

- `ForwardActionDistribution.is_stop_action` marks which logits correspond to
  `STOP`.
- `TrajectoryGFNSampleBatch.trace_stop_mask` records where `STOP` was sampled.
- `TrajectoryGFNSampleBatch.termination_action_steps` stores the 1-based action
  index where termination happened.

Absorbing prefixes literally end with a `STOP` token. Active prefixes do not.

### Forward legality

The sampler applies forward constraints before sampling actions.

- Graph moves that revisit previously seen entities on the same prefix are
  masked.
- `STOP` remains legal even when graph moves are masked.
- If no legal non-forced action remains, the rollout becomes inactive.

Rollout behavior can also optionally force immediate `STOP` on answer hit via
`training.force_stop_on_answer_hit`.

This keeps the search space tree-structured without allowing arbitrary revisit
loops.

## 3. Target policy and behavior policy

The code distinguishes the policy used for training equations from the policy
used for rollout sampling.

### Target policy

The target policy defines the quantities that enter MS-SubTB:

- root action probabilities
- forward action probabilities
- state flow values

Whenever training stores `log P_F`, it is using the target policy.

### Behavior policy

The sampler goes through behavior-policy hooks when it chooses starts and graph
actions.

- Start-node sampling uses `compute_behavior_start_distribution(...)`.
- Move sampling uses `compute_behavior_edge_logits(...)`.

Behavior and target match when heuristic bias is off. This is the default public
setup in the base YAML:

- `heuristic.kind: none`
- `heuristic.beta: 0.0`

When heuristic bias is enabled, behavior sampling is intentionally tilted for
exploration, but the stored training log-probabilities still come from the
target policy.

### Sampling temperature

The same training-time sampling temperature is applied to:

- the root start distribution, and
- the rollout action distributions.

The temperature itself can be constant or annealed through
`training.sampling_temperature_schedule`.

## 4. Policy parameterization

The model uses one parameterization for the root boundary and another for
ordinary prefix states.

### Root boundary scoring

At the root, the model predicts:

- `log F(s_root)` from question features, pooled graph summaries, pooled
  start-node summaries, and graph-size scalars
- start-node action logits with a dedicated root-action head

So the root boundary flow and the start-node action distribution are explicit
and decoupled.

### Non-root state scoring

For a non-root active prefix at hop `t`, the implementation uses a state
representation of the form:

`MLP([node(e_t) || step(t) || remain(T_max - t) || h_t])`

where the recurrent controller state `h_t` is updated from the previous prefix
history, relation token, next node, and question context.

The important point is not the exact block choice, but the contract:

- the state is still the exact prefix
- the scoring network also carries a recurrent summary to make rescoring fast

## 5. Reward and target measure

The training target is a trajectory measure. In log space, it is the terminal
log reward plus any step-level log rewards.

The core relation is:

`log R_target(tau) = ell_term(tau) + sum_t rho_t`

where:

- `ell_term(tau)` is the terminal log reward
- `rho_t` is the step-level log reward stored in `log_reward_steps`

### Terminal reward

Terminal reward is defined at the entity level.

- Gold answer entities get `log R_terminal = 0.0`.
- Non-gold terminal entities get
  `log R_terminal = training.terminal_failure_log_reward`.

By default that failure value is `-3.0`, but the field is configurable.

Alias answer entities count as gold too, because terminal supervision is matched
against entity ids rather than only one local answer node index.

### Base step reward

Each graph move contributes a fixed log cost.

- Public config: `training.step_log_penalty`

In the default public config, each graph move contributes the same constant
negative log reward, so longer trajectories are less preferred unless they are
needed to reach a correct answer.

### Backward scores

The current SubTB hot path does not reconstruct move-step backward scores.

- `log_pb_steps` is still present in the sample batch for interface stability.
- The terminal `STOP` backward factor is fixed to `log P_B = 0`.

So the objective is driven by forward prefixes and reward prefixes, not by a
full backward-policy reconstruction at every move step.

## 6. MS-SubTB objective

The main objective is a multi-scale SubTrajectory Balance loss specialized to
this prefix-tree state space.

Let:

- `f_t = log F(s_t)`
- `p_t = log P_F(a_t | s_t)`
- `rho_t = log_reward_steps[t]`
- `ell_term(tau) = terminal_log_rewards(tau)`

The implementation fits three residual families.

### Root residual

`r_root = log F(s_root) + log P_F([e_0] | s_root) - log F([e_0])`

This is the explicit root-boundary consistency equation.

### Pairwise residual

For prefix states `s_i -> s_j` on the same rollout:

`r_pair(i, j) = f_i + sum_{t=i}^{j-1} p_t - sum_{t=i}^{j-1} rho_t - f_j`

This forces consistency between intermediate prefixes under the shaped
trajectory measure.

### Terminal residual

For a prefix state `s_i` and the rollout terminal anchor:

`r_term(i) = f_i + sum_{t=i}^{T-1} p_t - (ell_term(tau) + sum_{t=i}^{T-1} rho_t)`

This anchors every valid prefix against the final reward-bearing suffix.

### Final SubTB loss

The code squares each residual family, aggregates them separately, and then
combines them with component weights:

- `training.subtb.root_loss_weight`
- `training.subtb.pairwise_loss_weight`
- `training.subtb.terminal_loss_weight`

Subtrajectory-length weighting is controlled by `training.subtb.lambda_weight`.
This keeps long-horizon pairwise terms from dominating the root and terminal
anchors.

## 7. No auxiliary imitation or guidance loss

The current training path does not add extra behavior-cloning, oracle-action,
or success-classification losses on top of MS-SubTB.

That is intentional. The optimization target is the shaped trajectory measure
defined by terminal reward plus step-level rewards, without a second objective
trying to pull the policy toward oracle actions or a separate classifier trying
to predict eventual success.

## 8. Important config knobs

If you are trying to understand or tune the algorithm, these are the main knobs
that matter:

| Config | Role |
| --- | --- |
| `training.terminal_failure_log_reward` | Terminal penalty for non-gold entities |
| `training.step_log_penalty` | Fixed log cost per graph move |
| `training.force_stop_on_answer_hit` | Forces immediate `STOP` on answer hit during rollout |
| `training.sampling_temperature` | Sampling temperature for starts and moves |
| `training.sampling_temperature_schedule.*` | Optional temperature annealing |
| `training.subtb.lambda_weight` | Subtrajectory-length weighting |
| `training.subtb.root_loss_weight` | Root residual contribution |
| `training.subtb.pairwise_loss_weight` | Pairwise residual contribution |
| `training.subtb.terminal_loss_weight` | Terminal residual contribution |

The base YAML in `configs/model/gflownet.yaml` uses the simple reward path:

- fixed `step_log_penalty`
- fixed `terminal_failure_log_reward`

## 9. Symbol-to-code map

If you want to line the math up with the code quickly, use this map.

| Math object | Tensor or helper | Main location |
| --- | --- | --- |
| `log F(s_root)` | `TrajectoryGFNSampleBatch.graph_log_z` | `src/models/gflownet/sampler.py`, `src/models/gflownet/losses.py` |
| `log P_F([e_0] | s_root)` | `TrajectoryGFNSampleBatch.start_log_probs` | `src/models/gflownet/sampler.py` |
| `log F([e_0])` | `TrajectoryGFNSampleBatch.start_state_log_f` | `src/models/gflownet/sampler.py` |
| `p_t` | `TrajectoryGFNSampleBatch.log_pf_steps` | `src/models/gflownet/sampler.py` |
| `rho_t` | `TrajectoryGFNSampleBatch.log_reward_steps` | `src/models/gflownet/sampler.py`, `src/models/gflownet_module.py` |
| `ell_term(tau)` | `TrajectoryGFNSampleBatch.terminal_log_rewards` | `src/models/gflownet/sampler.py` |
| SubTB residual assembly | `SubTrajectoryBalanceLoss.compute(...)` | `src/models/gflownet/losses.py` |

## 10. Recommended reading order in code

If you are onboarding to the implementation, read files in this order:

1. `src/models/configs/gflownet.py`
2. `src/models/gflownet/types.py`
3. `src/models/gflownet/sampler.py`
4. `src/models/gflownet/losses.py`
5. `src/models/gflownet_module.py`

That order mirrors the algorithm: config -> state/action types -> rollout
generation -> SubTB math -> training glue.
