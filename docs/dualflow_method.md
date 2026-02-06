# DualFlow (Code-Exact): Off-Policy Detailed Balance DualFlow for KGQA

This document is the single, canonical specification of the DualFlow algorithm **as implemented** in
`src/models/dual_flow_module.py`. When in doubt, the code wins.

---

## 0. SSOT Semantics (QA vs Flow)

- **Data SSOT**: `q_local_indices` / `a_local_indices` (+ ptr) are the only truth for question/answer entity sets.
- **No swapping**: backward flow is **not** implemented by swapping q/a. Direction comes from edge masks + start/target
  selection.
- **Mask semantics**:
  - `dummy_mask` is derived only from `answer_entity_ids_ptr` (data-level answer availability).
  - `node_is_target` (forward hit condition) is derived from `a_local_indices` (flow target set inside the subgraph).

---

## 1. Graph, Inverses, and Action Spaces

Each sample provides a directed subgraph `g_retrieval` with:

- `edge_index: [2, E]` local directed edges (no self-loops allowed; code hard-fails if `head == tail` exists).
- `edge_attr: [E]` relation ids. The preprocessing injects inverse relations (suffix `__inv` by default) and the runtime
  builds an `edge_inverse_map` (strictly symmetric by default).

Action spaces are **relation-type masked**:

- Forward edges `E_f`: edges whose relation is **not** marked as inverse in the relation vocab.
- Backward edges `E_b`: edges whose relation **is** marked as inverse in the relation vocab.

Code refs: `src/models/dual_flow_module.py` (`_build_edge_inverse_mask`, `_build_edge_direction_mask`,
`_build_edge_inverse_map`, `_validate_edge_inverse_map`).

---

## 2. Parameterization

### 2.1 Single backbone

A single `EmbeddingBackbone` produces node tokens for all computations:

- node tokens: `prepared.node_tokens`

CVT initialization is always applied before the GNN.

### 2.2 Start selector (learnable)

A learnable MLP scores nodes in `q_local_indices` and samples a single start node with Gumbel-Max + straight-through
weights. The last linear layer is zero-initialized, so the initial distribution is uniform.

Code refs: `src/models/dual_flow_module.py` (`_build_start_selector`, `_select_start_nodes`).

### 2.3 Context tokens

- Forward context (always used):
  - `c_fwd = forward_ctx_proj([question_tokens ; start_tokens])`

Code refs: `src/models/dual_flow_module.py` (`_build_forward_context`).

### 2.4 Policies (Explicit Potential Difference)

Forward policy is trainable but parameterized via a **state potential**:

\[
\logit(u\to v)= -\log d_{in}(v) + \alpha\,(\log F(v) - \log F(u))
\]

where $\log F(\cdot)$ is produced by `z_predictor` and $\alpha=\exp(\text{logit_scale})$ is a learnable scale.

Backward policy is **static uniform** over inverse outgoing edges; it does not use a network.

### 2.5 LogZ predictor

A single `LogZPredictor` is used for all trajectories, and **never** conditions on answers (it uses the forward context
only). It is time-conditioned via `SinusoidalPositionalEncoding`.

Code refs: `src/models/dual_flow_module.py` (`_compute_log_z_for_nodes`).

---

## 3. Rollouts (Two Directions)

Training samples two types of off-policy trajectories (both sampled under `torch.no_grad()`):

### 3.1 Forward rollout (exploration)

- start: `start_nodes_fwd` sampled from `q_local_indices` by the start selector
- transitions: sample edges from `E_f` using `policy_fwd` with scheduled temperature
- terminal conditions: hit any node in `a_local_indices`, dead-end (no outgoing), max_steps, invalid start

### 3.2 Backward rollout (demonstration)

- start: `a_seed ~ Uniform(a_local_indices)` (allow-empty; empty implies invalid start)
- target set: `q_local_indices` (hit condition uses `node_is_start`)
- transitions:
  - static PB: sample from uniform `P_B`

Backward actions are mapped back to forward-edge ids via `edge_inverse_map` before computing the DB loss.

Code refs: `src/models/dual_flow_module.py` (`_rollout_policy`, `_rollout_pb`, `_map_inverse_actions`).

### 3.3 How training mixes them (code-exact)

For each rollout iteration (repeat `training_cfg.num_rollouts` times):

1. sample one forward trajectory `tau_f ~ P_F`
2. sample one backward trajectory `tau_b ~ P_B`, map to forward-edge ids
3. compute DB loss on each, then average: `loss = (loss_fwd + loss_bwd) / 2`

Finally, average the loss over rollout iterations.

Code ref: `src/models/dual_flow_module.py` (`_aggregate_training_rollouts`).

---

## 4. Detailed Balance Objective (What We Optimize)

For a forward edge at step `t`:

- edge `e_t: u_t -> v_{t+1}`
- inverse edge id `e_t^{-1}` is looked up by `edge_inverse_map[e_t]`

Per-step residual (implemented in `_compute_db_loss`):

$$
\\Delta_t =
(\\log Z(u_t,t) + \\log P_F(e_t \\mid u_t,t))
-
(\\log Z(v_{t+1},t+1) + \\log P_B(e_t^{-1} \\mid v_{t+1},t+1))
$$

Boundary overrides (hard-coded):

- **target clamp**: if `v_{t+1} in a_local_indices`, set `log Z(v_{t+1},t+1) = 0`
- **terminal failure clamp**: if the trajectory terminates without hit, set `log Z(v_{t+1},t+1) = dead_end_log_reward`
  (and optionally weight the whole trajectory by `dead_end_weight`)

Only steps with a valid inverse edge (`edge_inverse_map[e_t] >= 0`) contribute.

Final loss is the weighted mean squared residual over valid steps.

Code ref: `src/models/dual_flow_module.py` (`_compute_db_loss`).

---

## 5. Static $P_B$ (Uniform)

The backward policy is fixed **uniform** over inverse outgoing edges:

- `logit_B(e) = 0` for all `e in Out_b(v)`
- `P_B` is uniform over `Out_b(v)`; equivalently `log P_B = -log |Out_b(v)|`

Implementation: `_compute_pb_logits`, `_compute_pb_log_prob`, `_rollout_pb`.

---

## 6. Hyperparameters (What Actually Exists in Code)

Key knobs under `model.training_cfg.db_cfg`:

- `sampling_temperature_start`, `sampling_temperature_end`
- `dead_end_log_reward`, `dead_end_weight`

Training:

- `training_cfg.num_rollouts`: number of (forward+backward) rollout pairs per batch.

---

## 7. Minimal Code Map (Where To Read)

- Batch build (tokens, masks, inverse map): `src/models/dual_flow_module.py` (`_prepare_batch`)
- Forward rollout: `src/models/dual_flow_module.py` (`_rollout_policy`)
- Static PB rollout: `src/models/dual_flow_module.py` (`_rollout_pb`)
- DB loss: `src/models/dual_flow_module.py` (`_compute_db_loss`)
- PB static logic: `src/models/dual_flow_module.py` (`_compute_pb_logits`, `_compute_pb_log_prob`)
