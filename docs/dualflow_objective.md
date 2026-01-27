# DualFlow Training Objective (Formal Derivation + Pseudocode)

This document specifies the DualFlow training objective exactly as implemented in `src/models/dual_flow_module.py`. The notation is aligned to code variables but uses a standard mathematical form.

## 1) Notation

For a graph sample g in a batch:

- Directed graph: G_g = (V_g, E_g), with edges e = (u -> v, r).
- Local node indices come from the batch; global entity ids are only for output.
- Start set: S_g (from `q_local_indices`).
- Target set: A_g (from `a_local_indices`).
- Max rollout length: T (from `env.max_steps`).
- Inverse relation mapping: inv(r) exists for non-inverse relations; self-loop uses relation id = -1.
- Forward edge set E_f: non-inverse edges + self-loops.
- Backward edge set E_b: inverse edges + self-loops.

For each graph g, a single start node s_g is sampled uniformly from S_g.

## 2) Model Parameterization

### 2.1 Node and edge representations

- Node embeddings and relation embeddings come from precomputed tables (data pipeline).
- Two independent backbones are used:
  - forward backbone produces node tokens h^f_v and relation tokens h^f_e,
  - backward backbone produces h^b_v and h^b_e.

Both run a relational GNN over the graph (`EmbeddingBackbone` in `src/models/components/gflownet_layers.py`).

### 2.2 Context vectors

Let q_g be the question embedding, and s_g the sampled start node. Let Pool(A_g) be the attention-pooled answer token (from node tokens + question token).

- Forward context:

  c^f_g = MLP([q_g; h^f_{s_g}])

- Backward context:

  c^b_g = MLP([q_g; h^b_{s_g}; Pool(A_g)])

### 2.3 LogZ predictor

For node v at step t:

  log Z^f(v,t) = LogZPredictor(h^f_v + TimeEmb(t), c^f_g)

The code uses the same Z predictor for all steps with a sinusoidal time embedding.

### 2.4 Edge logits (QC-BiA)

For an edge e = (u -> v, r) at step t:

  logit^f_t(e) = QCBiA(c^f_g, h^f_u + TimeEmb(t), h^f_r, h^f_v)

Backward logits are computed similarly but with backward tokens and c^b_g.

## 3) Forward Policy and Rollout

At step t, for current node u_t, define the set of outgoing forward edges:

  Out_f(u_t) = { e in E_f | head(e) = u_t }

Forward policy (temperature T_samp):

  P_F(e | u_t, t) = softmax_e( logit^f_t(e) / T_samp )

### MC subsampling (high-degree correction)
When degree is large, the code subsamples Out_f(u_t). Let Out_full be the full set and Out_sample be the subsample. The log normalizer is corrected as:

  log denom = logsumexp_{e in Out_sample}(logit^f_t(e)/T_samp)
              + log |Out_full| - log |Out_sample|

Then for a chosen edge e:

  log P_F(e | u_t, t) = logit^f_t(e)/T_samp - log denom

Note: the chosen edge logit is computed even if e is not in Out_sample; log denom still uses Out_sample. This matches `_compute_forward_log_prob`.

### Rollout
A rollout proceeds until one of:

- u_t in A_g (hit),
- Out_f(u_t) is empty (dead end),
- t reaches T (max steps),
- invalid start (missing start node).

The forward rollout is sampled with Gumbel-max using the forward policy.

## 4) Backward Policy and Inverse Edges

For a forward edge e = (u -> v, r), define its inverse edge e^{-1} = (v -> u, inv(r)).

The backward policy is parameterized by its own network and uses temperature 1.0:

  P_B(e^{-1} | v, t+1) = softmax_{e' in Out_b(v)}( logit^b_{t+1}(e') )

`Out_b(v)` is built from inverse edges (plus self-loops).

## 5) Detailed Balance Objective

For each step t in a rollout and its chosen edge e_t = (u_t -> v_{t+1}), define the detailed balance residual:

  Delta_t = log Z^f(u_t, t)
            + log P_F(e_t | u_t, t)
            - log Z^f(v_{t+1}, t+1)
            - log P_B(e_t^{-1} | v_{t+1}, t+1)

### Terminal overrides
- If v_{t+1} in A_g (target hit), set log Z^f(v_{t+1}, t+1) = 0.
- If the trajectory terminates at step t+1 without hit (dead end or max step), set log Z^f(v_{t+1}, t+1) = dead_end_log_reward.

### Per-trajectory weighting
Let failure(g) = (stop_reason != HIT). For all steps in a failure trajectory, a weight `dead_end_weight` may be applied to Delta_t^2.

### Final loss
Let V be the set of valid steps where the inverse edge exists. Then:

  L = ( sum_{(g,t) in V} w_g * Delta_t^2 ) / ( sum_{(g,t) in V} w_g )

where w_g = dead_end_weight if failure(g), else 1.

This is exactly `_compute_db_loss` in `src/models/dual_flow_module.py`.

## 6) Pseudocode (Training Step)

```text
function DualFlowTrainingStep(batch):
    prepared_fwd, prepared_bwd = prepare_batch(batch)
    graph_mask = not prepared_fwd.dummy_mask
    node_is_target = build_node_mask(prepared_fwd.a_local_indices)

    # Rollout with forward policy
    rollout = rollout_forward(
        policy=policy_fwd,
        prepared=prepared_fwd,
        graph_mask=graph_mask,
        node_is_target=node_is_target,
        temperature=sampling_temperature
    )

    actions = rollout.actions  # shape [B, T]
    stop_reason = rollout.stop_reason
    traj_lengths = rollout.num_moves

    # Detailed balance loss
    total = 0
    denom = 0
    weight = ones(B)
    failure = (stop_reason != HIT) and graph_mask
    weight[failure] *= dead_end_weight

    for t in 0..T-1:
        edge_ids = actions[:, t]
        move_mask = (edge_ids >= 0) and graph_mask
        if not any(move_mask): continue

        heads = edge_index[0, edge_ids]
        tails = edge_index[1, edge_ids]
        log_z_u = logZ(heads, t)
        log_z_v = logZ(tails, t+1)

        # target override
        if tails in target: log_z_v = 0

        # terminal failure override
        terminal = (traj_lengths == t+1) and failure
        if terminal: log_z_v = dead_end_log_reward

        log_pf = forward_log_prob(edge_ids, heads, t, temp=sampling_temperature)

        inv_edge = edge_inverse_map[edge_ids]
        inv_valid = inv_edge >= 0
        log_pb = backward_log_prob(inv_edge, tails, t+1, temp=1.0)

        valid = move_mask and inv_valid
        delta = log_z_u + log_pf - log_z_v - log_pb

        total += sum(weight[valid] * delta[valid]^2)
        denom += sum(weight[valid])

    loss = total / denom
    return loss
```

## 7) Implementation References

- Batch preparation, edge masks, inverse map: `src/models/dual_flow_module.py`.
- LogZ predictor and time encoding: `src/models/components/gflownet_layers.py`, `src/models/components/gflownet_actor.py`.
- Policy logits: `src/models/components/qc_bia_network.py`.
- Edge CSR and sampling ops: `src/models/components/gflownet_ops.py`.
