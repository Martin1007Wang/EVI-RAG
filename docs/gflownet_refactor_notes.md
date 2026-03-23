# GFlowNet Refactor Notes

This repo now uses a clearer boundary-vs-prefix split.

The latest refactor also makes the root boundary explicit in the training
objective instead of deriving it from start-state flows.

## Naming changes

- `StartDistribution` is still available as a compatibility alias, but the
  preferred name is `RootActionDistribution`.
- `RootState` is now the explicit runtime object for the abstract virtual root.
- `is_submit` is still available as a compatibility alias, but the preferred
  name is `is_stop_action`.
- `trace_submit_mask` is still available as a compatibility alias, but the
  preferred name is `trace_stop_mask`.
- `terminal_action_counts` is still available as a compatibility alias, but the
  preferred name is `termination_action_steps`.

## Why the prefix tensor now includes STOP for absorbing states

The code now encodes STOP explicitly so the published math and runtime state
space agree exactly:

- active prefixes end with an entity token
- absorbing prefixes end with the dedicated STOP token

The runtime also separates two masks that used to be conflated:

- `SearchState.done_mask` now means "inactive in this batched policy call"
- `SearchState.absorbing_mask` means "this prefix already carries explicit
  STOP"

That separation lets Monte Carlo and guidance traces keep padded or dead-end
rows inactive without incorrectly forcing them to masquerade as true absorbing
states.

The implementation still keeps separate rollout-side traces because they are
useful for bookkeeping and metrics:

- parent recovery on the prefix tree
- full no-repeat masking over previously seen entities
- recurrent control-state reconstruction

## Root and reward boundaries

- `RootActionDistribution.graph_log_z` is now an explicit root-flow boundary
  `log F(s_root)`, predicted from question tokens, pooled graph summaries, and
  explicit graph-size features instead of `logsumexp` over start states.
- `RootActionDistribution.log_probs` now come from a dedicated root-action head.
- `RootActionDistribution.log_flows` continue to store the child start-state
  values `log F([e_0])`.
- SubTB now trains the root consistency residual
  `log F(s_root) + log P_F([e_0] | s_root) - log F([e_0])` directly.
- Terminal rewards now use a single fixed terminal-energy table:
  gold entities map to `log R = 0.0`, non-gold entities map to `log R = -3.0`.
- Sampled rollouts additionally apply the configured path-length discount
  `gamma^|tau|` before SubTB consumes the terminal anchor.
- STOP termination keeps `log P_B = 0`, so there is no separate terminal
  backward heuristic layered on top of the prefix-tree backward semantics.

## Recommended terminology

Use these terms in future code and docs:

- abstract root boundary
- root state
- root action distribution
- graph prefix state
- STOP action
- termination step

Avoid mixing `submit`, `stop`, and `terminal action` in the same explanation
unless you explicitly say they refer to the same event.
