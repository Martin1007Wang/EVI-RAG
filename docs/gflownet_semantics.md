# GFlowNet State And Action Semantics

This note defines the current answer-reachability GFlowNet runtime in the same
terms used by the code.

## Core objects

- `RootActionDistribution`: explicit outgoing action distribution from the
  abstract root boundary state. The root is not materialized as a regular
  `SearchState` because its outgoing actions are query-specific start-node
  selections rather than graph edges.
- `RootState`: explicit runtime object for the abstract root boundary state.
- `SearchState`: recurrent prefix state for graph search. It stores the exact
  discrete graph prefix together with cached runtime fields that make repeated
  scoring cheap.
- `ForwardActionDistribution`: per-state action distribution over graph-edge
  moves plus one explicit STOP action for every active agent.
- `TrajectoryGFNSampleBatch`: sampled trajectories with graph-prefix traces and
  separate STOP-action traces.

## Root boundary

Root selection is handled in two steps:

1. Build `RootState` from the prepared batch context.
2. Predict the root boundary flow `log F(s_root)` from the concatenation of the
   per-graph question token, pooled all-node and start-node graph embeddings,
   plus explicit graph-size scalars.
3. Build `RootActionDistribution` from the candidate start nodes resolved from
   `q_local_indices` using a dedicated root-action head.
3. Sample start nodes from that root action distribution.

This keeps the abstract root explicit in the probability model without forcing a
pseudo-node into the graph topology or into the prefix tensor layout.

The root boundary and the start-state flow are now deliberately decoupled:

- `graph_log_z` is the explicit `log F(s_root)` boundary value
- `log_probs` parameterize `P_F([e_0] | s_root)`
- `log_flows` still store the child-state values `log F([e_0])`

SubTB then learns the root consistency equation explicitly through the residual

`log F(s_root) + log P_F([e_0] | s_root) - log F([e_0])`.

## Search state

`SearchState` keeps the following fields:

- `path_token_ids`: exact graph prefix encoded as
  `node, relation, node, relation, ...`
- `current_nodes`: cached terminal node of each active prefix
- `num_steps`: number of graph moves already taken
- `done_mask`: marks inactive rows inside a batched search call
- `absorbing_mask`: marks the rows whose prefix is a true STOP-terminated
  absorbing state
- `control_state`: cached recurrent controller state used for fast rescoring

The semantic state is still the graph prefix plus termination status. The extra
fields are cached views used to avoid reconstructing the same information on
every policy call.

`done_mask` and `absorbing_mask` are intentionally not identical anymore:

- STOP-terminated prefixes are both inactive and absorbing
- dead-end or padded rows can be inactive without pretending to carry an
  explicit STOP token

## Feature mapping

For a non-root active prefix state at hop `t`, the current implementation uses

`MLP([node(e_t) || step(t) || remain(T_max - t) || h_t])`

where `h_t` is updated by a GRU over

`[Attn(Phi(q), h_{t-1}) || rel(r_t) || node(e_t)]`.

This matches the paper-facing parameterization directly: node identity, current
hop index, remaining budget, and recurrent question-conditioned prefix history
are all explicit inputs to the state feature network.

## STOP action

The runtime now treats STOP as the primary public term.

- `ForwardActionDistribution.is_stop_action` marks which logits correspond to
  STOP rather than graph moves.
- `TrajectoryGFNSampleBatch.trace_stop_mask` records when STOP was chosen.
- `TrajectoryGFNSampleBatch.termination_action_steps` stores the 1-based action
  position where STOP occurred.

The exact state sequence now does contain a dedicated STOP token for absorbing
states.

- active states end with a node token
- absorbing states end with the explicit STOP token
- trace reconstruction appends STOP immediately after a sampled STOP action so
  later inactive rows keep the exact absorbing prefix

This matches the paper-facing sequence definition directly while keeping
`trace_stop_mask` and `termination_action_steps` as convenient rollout-side
derived traces.

## Backward semantics

Backward transitions on graph prefixes remain tree-structured:

- active non-root states recover their unique parent edge from `path_token_ids`
- active start states have the abstract root as their unique predecessor
- absorbing states have the corresponding active prefix as their unique
  predecessor through STOP

This keeps the prefix-tree interpretation while avoiding unnecessary backward
reconstruction work on move steps that are not consumed by the current SubTB hot
path.

## Terminal reward realization

The runtime now uses a single fixed terminal-energy scheme.

- gold terminal entities get `log R_terminal = 0.0`, so `R_terminal = 1.0`
- non-gold terminal entities get `log R_terminal = -3.0`, so
  `R_terminal = exp(-3.0)`
- sampled trajectories then apply a length discount
  `R(τ) = R_terminal * gamma^|τ|`, where `gamma` is
  `training.trajectory_length_discount`
- revisit counts and other rollout bookkeeping are still not part of the reward
  definition
- full no-repeat is now enforced as a forward legality rule: graph moves that
  revisit any previously seen entity on the current prefix are masked, while
  STOP remains legal

This keeps the supervision contract simple: SubTB always anchors against a
stable terminal log-reward table instead of switching between multiple reward
parameterizations.

`success_mask`, termination steps, and revisit bookkeeping are still tracked for
rollout reporting, but they are not part of the terminal reward definition.

The terminal STOP backward factor is also fixed: the absorbing STOP edge uses
`log P_B = 0`, matching the unique-prefix-parent interpretation used elsewhere
in the runtime.

## Forced STOP behavior

Training-time forced STOP is now explicit and configurable through
`training.force_stop_on_answer_hit`.

- `false` means STOP is treated as a normal action, even if the current node is
  already a gold answer
- `true` means the sampler immediately emits STOP when an active rollout lands
  on a gold answer

This is a behavior-policy choice, not an environment axiom. Keep that
distinction in mind when comparing theory notes with rollout behavior.

With the default training config, start-node sampling and rollout action
sampling are on-policy: the sampler draws from the target root/action
distributions rather than a separate behavior policy.
