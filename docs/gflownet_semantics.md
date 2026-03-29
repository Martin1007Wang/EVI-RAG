# RankFlow / GFlowNet Semantics

This document describes the current subgraph-growth RankFlow contract.

If you need the runnable defaults, read these files directly:

- `configs/model/gflownet.yaml`
- `configs/experiment/train_rankflow.yaml`
- `configs/experiment/rankflow.yaml`

## 1. Mainline in one sentence

RankFlow now trains a subgraph-growth GFlowNet: each state is a partial subgraph,
each action either adds one frontier edge or emits an explicit `STOP`, and the
loss is a subgraph-space SubTB objective.

## 2. State space

The canonical state is `SubgraphState(edge_ids)`.

- The stored state identity is the tuple of selected edge ids.
- The selected node set is derived from anchors plus endpoints of selected edges.
- The initial state contains no selected edges.
- Question anchors are treated as already-present nodes in that empty state.

The environment materializes derived semantics through `SubgraphAnalysis`:

- `selected_node_ids`
- `reachability_bits`
- `component_labels`
- `anchor_component_count`
- `num_selected_edges`

These derived fields drive legality, rewards, and search ranking, but the state
identity itself is still the explicit selected-edge set.

## 3. Forward actions

Each active state exposes two action families:

- `add_edge(edge_id)` for legal frontier edges
- `stop()` for explicit termination

An edge is legal only when it extends the currently selected subgraph under the
subgraph environment rules. Evaluation and training both use this same explicit
action space; there is no longer a separate legacy prefix/path state machine.

## 4. Rewards

The environment defines two reward pieces.

- expand reward: shaped per added edge via `compute_expand_log_reward`
- stop reward: computed by `compute_stop_log_reward`

The stop reward is driven by the current subgraph semantics, especially:

- how many answer entities are covered
- whether the anchor components have merged
- the configured subgraph reward weights in `training.subgraph_reward`

## 5. Policy and proposal bias

`SubgraphPolicy` produces target logits for the legal action set and also a
proposal-only bias used during rollout sampling.

The target distribution comes from `compute_target_log_probs(...)`.

The rollout sampler may add proposal bias from `training.subgraph_proposal`:

- answer-distance progress
- question similarity
- component-merge bonus
- stop-hit bias

These terms change coverage during sampling but do not replace the target-policy
SubTB algebra.

## 6. Training objective

Training uses `SubgraphSubTrajectoryBalanceLoss` over explicit action sequences.

For each rollout, the sampler records:

- state log flows
- chosen action log-probabilities
- per-action log rewards
- explicit stop step
- terminal answer counts and component counts

The loss then evaluates SubTB residuals over sub-trajectories in this subgraph
action sequence.

## 7. Evaluation

Evaluation is subgraph-only.

- runtime factory builds `SubgraphAnswerSearchRuntime`
- runtime uses `beam_search_subgraphs(...)`
- terminal subgraphs are converted to answer entities through
  `resolve_subgraph_answer_entities(...)`

The canonical evaluation task is `answer_search`.

## 8. Unsupported legacy features

The old prefix-tree stack has been removed. Current code intentionally rejects:

- `policy.state_mode != subgraph`
- success replay
- answer quotient
- direct entity ranking
- legacy potential-reward shaping
- non-`answer_search` GFlowNet evaluation runtimes

## 9. Reading order in code

If you want to audit the implementation from config to runtime, read in this
order:

1. `configs/experiment/train_rankflow.yaml`
2. `src/models/configs/policy.py`
3. `src/models/configs/gflownet_training.py`
4. `src/models/gflownet/subgraph/state.py`
5. `src/models/gflownet/subgraph/prepared_batch.py`
6. `src/models/gflownet/subgraph/mdp.py`
7. `src/models/gflownet/subgraph/policy.py`
8. `src/models/gflownet/subgraph/sampler.py`
9. `src/models/gflownet/subgraph/search.py`
10. `src/models/gflownet/subgraph/losses.py`
11. `src/metrics/subgraph_answer_search_runtime.py`
12. `src/models/gflownet_module.py`
