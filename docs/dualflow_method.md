# DualFlow (Code-Exact): Off-Policy Detailed Balance GFlowNet for KGQA

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

### 2.1 Dual backbones

Two independent `EmbeddingBackbone`s (no parameter sharing) produce node tokens:

- forward tokens: `prepared_fwd.node_tokens`
- backward tokens: `prepared_bwd.node_tokens`

Optional CVT initialization is applied before the GNN (`cvt_init_cfg.enabled`).

### 2.2 Start selector (learnable)

A learnable MLP scores nodes in `q_local_indices` and samples a single start node with Gumbel-Max + straight-through
weights. The last linear layer is zero-initialized, so the initial distribution is uniform.

Code refs: `src/models/dual_flow_module.py` (`_build_start_selector`, `_select_start_nodes`).

### 2.3 Context tokens

- Forward context (always used):
  - `c_fwd = forward_ctx_proj([question_tokens_fwd ; start_tokens_fwd])`
- Backward context (only used when `pb_mode=learned`):
  - built as `backward_ctx_proj([question_tokens_bwd ; start_tokens_bwd ; answer_tokens])`
  - during training, `answer_tokens` is replaced by a **target roulette** anchor token (see Sec. 5.3).

Code refs: `src/models/dual_flow_module.py` (`_build_forward_context`, `_build_backward_context`, `_apply_target_roulette`).

### 2.4 Policies (QC-BiA)

Forward policy is always trainable and uses `QCBiANetwork` over edges:

- `policy_fwd(context, head, relation, tail) -> logits`

Backward policy depends on `pb_mode`:

- static (`uniform`, `topo_semantic`): no network; log-prob comes from a hand-crafted `P_B` (Sec. 5.1/5.2).
- learned (`learned`): trainable `policy_bwd` with backward backbone + backward context (Sec. 5.3).

Code ref: `src/models/components/qc_bia_network.py`.

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
  - static PB: sample from the chosen static `P_B` (Sec. 5.1/5.2)
  - learned PB: sample from `policy_bwd` (temperature fixed to 1.0)
- edge dropout: a per-batch Bernoulli mask `pb_edge_dropout` is applied to backward outgoing edges for both rollout and
  DB evaluation

Backward actions are mapped back to forward-edge ids via `edge_inverse_map` before computing the DB loss.

Code refs: `src/models/dual_flow_module.py` (`_rollout_policy`, `_rollout_pb`, `_map_inverse_actions`,
`_sample_pb_edge_dropout_mask`).

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

## 5. Our Three $P_B$ Strategies (Paper Focus)

All PB settings live under `model.training_cfg.db_cfg.*` and are composed via:

- `configs/model/db/{uniform,topo_semantic,learned}.yaml`
- `configs/experiment/train_gflownet_pb_{uniform,topo_semantic,learned}.yaml`

At a glance:

| `pb_mode` | Trainable? | Core idea | Key knobs |
| --- | --- | --- | --- |
| `uniform` | no | uniform over inverse outgoing edges | `pb_edge_dropout` |
| `topo_semantic` | no | topo monotone (`dist` decreases) + semantic cosine bias | `pb_max_hops`, `pb_semantic_weight`, `pb_topo_penalty`, `pb_cosine_eps`, `pb_edge_dropout` |
| `learned` | yes | QC-BiA over inverse edges with target roulette context | `pb_edge_dropout` |

### 5.1 PB-Uniform (static)

Config: `configs/model/db/uniform.yaml`

Definition (code): for a backward state `v`, assign equal logit to every outgoing inverse edge:

- `logit_B(e) = 0` for all `e in Out_b(v)`
- `P_B` is uniform over `Out_b(v)`; equivalently `log P_B = -log |Out_b(v)|`

Implementation: `_compute_pb_logits(mode="uniform")`, `_compute_pb_log_prob`, `_rollout_pb`.

### 5.2 PB-Topo-Semantic (static)

Config: `configs/model/db/topo_semantic.yaml`

This PB is a hand-crafted distribution over inverse edges combining:

1. **Topology constraint** (distance-to-start monotonicity)
2. **Semantic bias** (question-vs-relation cosine similarity in raw embedding space)

Distance-to-start:

- compute `dist_to_start[node]` as the BFS distance from any node in `q_local_indices` along **forward** edges `E_f`
  up to `pb_max_hops` (defaults to `max_steps`)

Allowed inverse edge:

- for an inverse edge `e: v -> u`, it is **allowed** iff `dist_to_start[u] < dist_to_start[v]`

PB logits (code):

$$
\\text{logit}_B(e)=
\\underbrace{\\mathbf{1}[\\text{allowed}]\\cdot 0 + \\mathbf{1}[\\neg\\text{allowed}]\\cdot \\text{pb\\_topo\\_penalty}}_{\\text{topo term}}
\\;+
\\underbrace{\\text{pb\\_semantic\\_weight}\\cdot \\cos(q_{emb}, r_{emb})}_{\\text{semantic term}}
$$

Special handling (code):

- if a state has **no allowed** outgoing edges, the step log-prob is set to `pb_topo_penalty` (tracked by
  `db_no_allowed_rate`)
- choosing a disallowed inverse edge is tracked as a topology violation (`db_topo_violation_rate`)

Implementation: `_compute_distance_to_starts`, `_compute_pb_logits(mode="topo_semantic")`, `_compute_pb_log_prob`,
`_rollout_pb`.

### 5.3 PB-Learned (trainable)

Config: `configs/model/db/learned.yaml`

PB is parameterized by `policy_bwd` (QC-BiA) and trained jointly via DB.

Target roulette (code-exact):

- sample `a_seed ~ Uniform(a_local_indices)` (same node used as the start of the backward rollout)
- build backward context using the **anchor token** of `a_seed` (not pooled `A`)

Backward policy:

$$
P_B(e^{-1} \\mid v,t+1) = \\text{softmax}_{e'\\in Out_b(v)}(\\text{QCBiA}(c_{bwd}, h_v^B, h_{r^{-1}}^B, h_u^B))
$$

Regularization:

- `pb_edge_dropout` randomly drops backward edges (mask shared between rollout and DB evaluation)

Implementation: `_apply_target_roulette`, `_rollout_policy(policy_bwd, temp=1.0)`, `_compute_forward_log_prob` on
inverse edges inside `_compute_db_loss`.

---

## 6. Hyperparameters (What Actually Exists in Code)

Key knobs under `model.training_cfg.db_cfg` (see `configs/model/db/base.yaml`):

- `pb_mode`: `uniform | topo_semantic | learned`
- `pb_edge_dropout`
- `pb_semantic_weight` (topo_semantic only)
- `pb_topo_penalty` (topo_semantic only; also used as fallback log-prob when no allowed edges)
- `pb_cosine_eps`
- `pb_max_hops` (<=0 means use `max_steps`)
- `sampling_temperature_schedule`: `constant | cosine` (forward rollout temperature)
- `sampling_temperature_start`, `sampling_temperature_end`, `sampling_temperature`
- `dead_end_log_reward`, `dead_end_weight`

Training:

- `training_cfg.num_rollouts`: number of (forward+backward) rollout pairs per batch.

---

## 7. Minimal Code Map (Where To Read)

- Batch build (tokens, masks, inverse map): `src/models/dual_flow_module.py` (`_prepare_batch`)
- Forward rollout: `src/models/dual_flow_module.py` (`_rollout_policy`)
- Static PB rollout: `src/models/dual_flow_module.py` (`_rollout_pb`)
- DB loss: `src/models/dual_flow_module.py` (`_compute_db_loss`)
- PB static logic: `src/models/dual_flow_module.py` (`_compute_distance_to_starts`, `_compute_pb_logits`, `_compute_pb_log_prob`)
