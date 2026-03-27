# GFlowNet Refactor Notes

This file records the cleanup decisions behind the current RankFlow config and
documentation surface.

For the algorithm itself, read `docs/gflownet_semantics.md` first.

## What changed

The current cleanup pass removed several sources of drift and redundancy.

### Removed compatibility and alias surface

- legacy start/root naming shims were removed earlier from the code surface
- legacy `answer_reachability` train/eval experiment aliases were removed
- legacy `answer_reachability` run aliases were removed
- eval answer-task aliases were removed from the public config schema

### Removed redundant config knobs

- `ActionPriorConfig.beta` was removed; proposal strengths are now explicit via
  `root_beta`, `edge_beta`, and `stop_beta`
- `ActionPriorConfig.kind` was removed; node-prior construction is now driven
  directly by component weights
- `GFlowNetTrainingConfig.rollout_batch_size` was removed; only
  `rollouts_per_graph` remains
- `PotentialRewardConfig.kind` was removed; answer-distance shaping is active
  exactly when `answer_distance_weight > 0`
- model-level validation/final-eval profile contract metadata was removed;
  training and eval bundles are now the only sources of truth for those choices

## Config design rules

The config surface now follows four rules.

1. One mechanism gets one public knob.
2. Base dataclass defaults and shipped Hydra defaults should agree.
3. Base model YAML should only spell out structure or true overrides.
4. Canonical numeric choices belong in experiment bundles, not in prose docs.

## Documentation rules

To avoid future drift, the docs now separate mechanism from numbers.

- `docs/gflownet_semantics.md` explains the algorithm and config roles.
- `configs/model/gflownet.yaml` defines the minimal public schema surface.
- `configs/experiment/train_rankflow.yaml` carries the canonical training values.
- `configs/experiment/rankflow.yaml` carries the canonical final-eval values.

If you need to reproduce the live setup, read the config files directly instead
of relying on a copied list of numbers in Markdown.

## Proposal vs target boundary

The most important conceptual boundary is unchanged but now documented more
explicitly:

- target-side quantities define the SubTB residual algebra
- proposal-side quantities only change coverage

Concretely, that means:

- `model.policy_cfg.transition_head.*` is proposal-only
- `model.action_prior_cfg.*` is proposal-only
- `model.training_cfg.potential_reward.*` changes the target measure
- `model.training_cfg.success_replay.*` changes coverage mixture, not target
  equations

## Migration guide

If you are updating local configs or scripts, use these replacements.

| Removed surface | Use instead |
| --- | --- |
| `action_prior_cfg.beta` | set `root_beta` and `edge_beta` explicitly |
| `action_prior_cfg.kind` | set component weights directly |
| `training.rollout_batch_size` | `training.rollouts_per_graph` |
| `potential_reward.kind: answer_distance` | set `potential_reward.answer_distance_weight > 0` |
| `experiment=train_answer_reachability` | `experiment=train_rankflow` |
| `experiment=answer_reachability` | `experiment=rankflow` |
| `run=train_answer_reachability` | `run=train_rankflow` |
| `run=answer_reachability` | `run=rankflow` |

## What to inspect first

If something looks inconsistent again, audit in this order:

1. `src/models/configs/gflownet_training.py`
2. `configs/model/gflownet.yaml`
3. `configs/experiment/train_rankflow.yaml`
4. `src/models/gflownet/policy.py`
5. `src/models/gflownet/heuristics.py`
6. `docs/gflownet_semantics.md`
