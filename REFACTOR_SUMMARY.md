# Refactor Summary

## Files changed

- `configs/model/weaver.yaml`
- `configs/experiment/train/webqsp_baseline.yaml`
- `src/weaver/config.py`
- `src/weaver/module.py`
- `src/weaver/policy.py`
- `src/weaver/state.py`
- `src/weaver/nn/frontier_builder.py`
- `src/weaver/rollout/sampling.py`
- `src/weaver/reward/utility.py`
- `src/weaver/loss/__init__.py`
- `src/weaver/loss/bdb.py`
- `src/weaver/loss/subtb.py`
- `src/weaver/loss/te_bfm.py`
- `src/weaver/__init__.py`
- `scripts/diagnose_weaver_rollout.py`
- `tests/test_state_semantics.py`
- `tests/test_gflownet_rollout_split.py`
- `tests/test_semantic_model_space_features.py`
- `tests/test_vectorized_online_rollouts.py`
- `tests/weaver/loss/test_bdb_loss.py`
- `tests/weaver/loss/test_removed_objectives.py`

## Files deleted

- `tests/test_weaver_losses.py`

No data loading, graph construction, encoder architecture, evaluation metric, or logging infrastructure files were intentionally deleted. Alternative objective implementations were stubbed rather than removed outright where import compatibility was needed.

## New invariants added

- State transitions assert `b_z == B - |E_z \ E_0|` after every expansion.
- Directed boundary frontier construction asserts candidates satisfy `u in partial_z`, `v not in V_z`, and `e not in E_z`.
- Policy sampling asserts `sum(P_theta(a | z, q)) == 1` for every non-forced state after the softmax over `[stop, frontier edges]`.
- BDB loss asserts `L_BDB >= 0`.
- Reward utility now includes `delta > 0` in the `U_beta` denominator.
- Child target flow follows hard terminal reward anchors and detached current-network continuation values.

## Ambiguities in `methodology.md`

- The document requires `sg(r_Y(z))` everywhere but does not specify whether reward tensors should be computed outside autograd, detached at the call site, or both. The implementation uses no-grad reward evaluation where reward is produced and detach semantics where BDB targets are formed.
- `|Par(z_i)|` is defined mathematically by the DAG but not operationally for duplicate relation edges or multigraph edge ids. The implementation keeps the existing parent-count machinery and applies the `-log|Par(z_i)|` correction to those counts.
- The training-state distribution is not uniquely specified beyond allowing rollout, oracle, and counterfactual sources. The implementation keeps rollout-driven training and removes non-BDB objectives.
- The methodology names learned MLP logits but does not require a particular pointer/head parameterization. The implementation keeps the existing learned pointer policy head and removes reward-derived or backup-derived inference logits.
- The document defines `Ev(z) = (V_z, E_z)` as the inference output but does not prescribe external serialization. Existing evaluation/logging surfaces are preserved while the inference loop's semantic output remains the evidence subgraph.
