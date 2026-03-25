# GFlowNet Refactor Notes

This file is background and compatibility guidance.

For the current algorithm itself, read `docs/gflownet_semantics.md` first.

## What the refactor stabilized

The current codebase has converged on four important contracts:

1. The root boundary is explicit.
2. `STOP` is an explicit action and absorbing states store it directly.
3. Step-level rewards live in `log_reward_steps` instead of being hidden inside
   terminal-only bookkeeping.
4. Training now uses only the main MS-SubTB objective instead of layering on
   extra oracle-imitation or success-classification losses.

These decisions make the training logic easier to reason about and easier to map
from math to tensors.

## Public contract vs compatibility shims

The repo still carries a few compatibility aliases so older call sites do not
break immediately.

| Preferred name | Compatibility alias or legacy field |
| --- | --- |
| `RootActionDistribution` | `StartDistribution` |
| `is_stop_action` | `is_submit` |
| `trace_stop_mask` | `trace_submit_mask` |
| `termination_action_steps` | `terminal_action_counts` |

Treat the left column as the real contract for new code and new docs.

## Root boundary changes

The root boundary used to be easier to blur together with start states. The
current implementation keeps it explicit.

- `RootActionDistribution.graph_log_z` is the boundary flow `log F(s_root)`.
- `RootActionDistribution.log_probs` is the start-node action distribution.
- `RootActionDistribution.log_flows` stores the child start-state values.

This lets training use the root consistency residual directly:

`log F(s_root) + log P_F([e_0] | s_root) - log F([e_0])`

That is simpler to inspect than recovering root behavior indirectly from start
states.

## STOP and prefix representation

The refactor made the exact prefix representation match the conceptual state
space.

- Active prefixes end with a node token.
- Absorbing prefixes end with an explicit `STOP` token.

This also made it useful to separate:

- `done_mask`: inactive in the current batched computation
- `absorbing_mask`: truly STOP-terminated prefix

That distinction matters because padded rows, dead ends, and STOP-terminated
rows are all inactive for different reasons.

## Reward design after the refactor

The public training contract is now easier to explain:

- terminal correctness determines `terminal_log_rewards`
- step-level costs live in `log_reward_steps`
- MS-SubTB consumes both explicitly

In the base public config, this means:

- gold terminal entity -> `0.0`
- non-gold terminal entity -> `training.terminal_failure_log_reward`
- every graph move -> `training.step_log_penalty`

## Auxiliary losses removed

The old auxiliary losses were removed from the training path:

- no learned success-classification head
- no oracle action imitation loss

The model is now trained only through MS-SubTB on the configured reward
measure. This keeps the optimization target single-purpose and avoids mixing
flow matching with separate supervised objectives.

## Terminology to standardize

Use these terms in future docs and code comments:

- abstract root boundary
- root state
- root action distribution
- graph prefix state
- STOP action
- termination action step
- terminal number of graph moves
- terminal reward
- step log reward

Avoid mixing `submit`, `stop`, and `terminal action` unless you immediately say
they refer to the same event.

## Practical note for future edits

If a future change touches reward design, keep these three questions separate:

1. What defines the trajectory measure?
2. What only changes sampling behavior?
3. What is merely auxiliary supervision?

Keeping those layers separate is the easiest way to prevent the docs and the
implementation from drifting apart again.
