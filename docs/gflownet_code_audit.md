# GFlowNet 代码审计与形式化复盘（指定范围）

本文件是对以下范围的“事无巨细”复盘与形式化描述：

- `src/models/gflownet_module.py`
- `src/models/components/*`

目标是从逻辑分支、状态/张量语义、形状约束、异常条件与数学定义上逐一给出精确说明。

---

## 符号约定

- 批大小：$B = |\text{node\_ptr}| - 1$
- 全局节点数：$N = \text{node\_ptr}[-1]$
- 边数：$E = \text{edge\_index}.shape[1]$
- 边索引：$\text{edge\_index} \in \mathbb{Z}^{2 \times E}$，其中 `edge_index[0]=head`，`edge_index[1]=tail`
- STOP 动作常量：$\text{STOP} = -1$
- 动作序列：$\text{actions\_seq} \in (\{0,\dots,E-1\} \cup \{-1\})^{B \times T}$
- `q_local_indices` / `a_local_indices`：按 `node_ptr` 偏移后的全局节点索引（批内全局索引）
- `q_ptr` / `a_ptr`：对应的 CSR 指针（长度 `B+1`）
- `edge_batch ∈ [0,B-1]^E`：每条边所属图 ID
- `edge_ptr`：按 `edge_batch` 的 CSR 指针
- `edge_ids_by_head` / `edge_ptr_by_head`：按 head 节点构建的 CSR

---

## src/models/gflownet_module.py

### 数据结构与常量

- `_TBDiagSpec`: `log_prob_min`, `delta_max` 两个阈值，用于 TB 诊断裁剪。
- `_PreparedBatch`: 训练/评估所需的全部批量张量打包结果。
- `_TBViewTerms`: TB 视图输出，含 `loss_per_graph`、`weight_mask`、`success_mask`、`diag`。
- `_TBBackwardInputs` / `_TBForwardInputs`: TB 反向/正向路径中间量封装（含 log_pf/log_pb、reward、diag 等）。
- 常量：`_ZERO/_ONE/_TWO`、`STOP`、`_FLOW_STATS_DIM=2`、调度器默认配置、诊断分位数等。

### 初始化流程

`__init__`:

- 若 `training_cfg` 或 `evaluation_cfg` 为 None，抛出异常。
- `automatic_optimization=False`（手动优化）。
- 初始化 `backbone`、`cvt_init`、正反向 `agent`、正反向 `actor`、`log_f`。
- `_validate_cfg_contract` 强制白名单配置。
- `_save_serializable_hparams` 仅保存可序列化超参。
- 运行时缓存 `relation_inverse_map`/`relation_is_inverse` 置为 None。

`_init_backbone`:

- 构造 `EmbeddingBackbone(emb_dim, hidden_dim, finetune)`；无分支。

`_init_cvt_init`:

- 若 `cvt_init_cfg.enabled=False` → 抛错（强制启用 CVT 初始化）。
- 否则构造 `CvtNodeInitializer()`。

`_build_agent` / `_resolve_agent_hidden_dim`:

- `state_dim=None` → 直接用 `hidden_dim`。
- `state_dim != hidden_dim` → 抛错（强制一致）。
- 构造 `TrajectoryAgent(token_dim=hidden_dim, hidden_dim, dropout)`。

`_init_agents`:

- 若 `backward_cfg.share_state_encoder` 与 `share_edge_scorer` 不一致 → 抛错。
- 若两者为 True → 正反共享同一 agent；否则单独构建 `agent_backward`。

`_init_actors`:

- 若 `actor_cfg.score_mode` 存在 → 抛错（只允许 H-transform）。
- 解析 `policy_temperature / stop_bias_init / h_transform_bias / h_transform_clip`。
- 构造 `GFlowNetActor`（forward）与 `GFlowNetActor`（backward）。

`_init_flow_predictors`:

- 读取 `flow_cfg.graph_stats_log1p`；构造 `FlowPredictor(hidden_dim, 2)`。

`_validate_cfg_contract`:

- `training_cfg` 仅允许 {`num_train_rollouts`, `num_miner_rollouts`, `num_forward_rollouts`,
  `accumulate_grad_batches`, `allow_zero_hop`, `shaping`, `tb`}。
- `evaluation_cfg` 仅允许 {`num_eval_rollouts`, `rollout_temperature`}。
- 多余键立即抛错。

### 运行时资源初始化

`setup` / `_ensure_runtime_initialized`:

- 若缓存已存在直接返回。
- 必须有 `trainer.datamodule.shared_resources`，否则抛错。
- 从 datamodule 取 `cvt_mask`、`relation_inverse_map`、`relation_is_inverse`。
- 作为 buffer 注册（非 persistent），并缓存到成员变量。

### 优化器与调度器

`configure_optimizers`:

- 使用 `setup_optimizer` 构造 optimizer。
- `_build_scheduler` 有效时返回含 `lr_scheduler` 的 dict，否则仅返回 optimizer。

`_build_scheduler`:

- `scheduler_cfg.type` 为空 → `None`。
- `interval` 必须为 `epoch` 或 `step`，否则抛错。
- `type=cosine` → `CosineAnnealingLR`。
- `type=cosine_warm_restarts`（含若干别名）→ `CosineAnnealingWarmRestarts`。
- 其他 type → `None`。

`_step_scheduler`:

- 兼容 list/单个调度器，逐一 `lr_scheduler_step`。

`on_train_epoch_end`:

- `interval=epoch` 时 step scheduler。

### Lightning Hooks

`forward`:

- 明确不支持，直接抛 `NotImplementedError`。

`training_step`:

- `self._ensure_runtime_initialized()`。
- 手动 zero_grad（按 accumulate 规则）。
- 计算 loss/metrics，loss 非有限 → 抛错。
- `manual_backward(loss/accum)`；满足条件时 `optimizer.step()`。
- `interval=step` 时 step scheduler。
- 记录 `train/*` 指标与 `train/loss`。

`validation_step`/`test_step`:

- 计算 eval metrics。
- 按 dataset scope 记录 `val/<scope>/*` 或 `test/<scope>/*`。
- 对 `terminal_hit@*` 额外记录 `val/terminal_hit@*` 与 `test/terminal_hit@*`。

### 梯度累积控制

- `_accumulate_grad_batches`：优先 `training_cfg.accumulate_grad_batches`，否则用 `trainer.accumulate_grad_batches`，最小值为 1。
- `_should_zero_grad`：`accum<=1` 或 `batch_idx % accum == 0`。
- `_should_step_optimizer`：`accum<=1` 或最后一批，或 `(batch_idx+1) % accum == 0`。

### Batch 准备与一致性校验

`_validate_packed_node_locals`:

- 校验 ptr 长度、起点为 0、终点匹配 node_locals 数量、指针非递减。
- 对每个 packed 元素 i：若 `node_locals[i] ∉ [node_ptr[g], node_ptr[g+1])` 则抛错。

`_compute_node_batch`:

- `node_batch = repeat_interleave(graph_ids, node_counts)`。

`_build_dummy_mask`:

- `dummy_mask = (a_ptr[1:] - a_ptr[:-1] == 0)`。

`_sample_one_start_nodes`:

- 每图从 `node_locals` 中均匀随机取 1 个起点。
- 若任一图计数为 0 → 抛错。

`_resolve_node_is_cvt`:

- 无 `cvt_mask` → 全 False。
- 要求 `node_global_ids` 存在且长度一致；否则抛错。
- 返回 `cvt_mask[node_global_ids]`。

`_build_inverse_edge_ids`:

- 通过 `(head, tail, relation)` 构造 key，匹配 `(tail, head, inverse_relation)`。
- 若任一 inverse 不存在 → 抛错。

`_prepare_batch`:

- 必须提供 `ptr/edge_index/edge_attr` 与 `edge_batch/edge_ptr`；缺失即抛错。
- 必须提供 `q_local_indices/a_local_indices` 与 `_slice_dict` 中的 `q_ptr/a_ptr`；缺失即抛错。
- 校验 packed indices，采样起点 `start_nodes_q`。
- 必须提供 `question_emb/node_embeddings/edge_embeddings`；缺失即抛错。
- CVT 初始化后投影为 tokens。
- 构建 `flow_features`。
- 构建 `edge_is_inverse/inverse_edge_ids/edge_ids_by_head/edge_ptr_by_head`。

### 核心图与轨迹数学

`_build_state_nodes`:

- `state_nodes[:,0]=start_nodes`。
- 若 `actions[t]` 有效（>=0），则 `state_nodes[:,t+1]=tail(action_t)`；无效则保持最后有效状态。

`_compute_stop_node_locals`:

- `stop_nodes = state_nodes[stop_idx]`，本地索引 `stop_nodes - node_ptr[g]`。
- 无效 stop（<0）置为 -1。

`_build_node_is_target`:

- 构建长度 N 的布尔 mask，将 `target_locals` 置 True。

`_compute_log_f_nodes`:

- `log_f_nodes = FlowPredictor(node_tokens, question_tokens, flow_features, node_batch)`。

`_gather_log_f_start`:

- `log_f_start = log_f_nodes[start_nodes]`（无效起点置 0）。

### Potential Shaping（势能塑形）

`_compute_cosine_potential`:

- $\phi_{b,t} = \frac{\langle q_b, s_{b,t} \rangle}{\|q_b\|\cdot\|s_{b,t}\|}$，形状不匹配直接抛错。

`_compute_potential_shaping_sum`:

- $\Delta\phi_{b,t} = \phi_{b,t+1} - \phi_{b,t}$，末尾补 0。
- $S_b = \sum_t \Delta\phi_{b,t}\cdot \mathbb{1}[\text{move\_mask}]$。

`_apply_potential_shaping`:

- `shaping_weight<=0` → 返回原 reward。
- 否则 `log_reward' = log_reward + shaping_weight · S`。

### TB Loss 计算与诊断

`_compute_tb_loss_per_graph`:

- $\sum \log P_F = \sum_{t\le t^*} \log p_F(a_t)$
- $\sum \log P_B = \sum_{t< t^*} \log p_B(a_t)$
- $L = ( \log F(s_0) + \sum \log P_F - \sum \log P_B - \log R )^2$。

`_summarize_tb_stats`:

- 裁剪 log-prob 与差值，统计 clip 率与均值/分位数。
- `zero_hop_rate` 仅基于有效图（graph_mask）。

`_summarize_stop_diagnostics`:

- `has_edge_rate`、`allow_stop_rate`、`stop_margin` 分位数、`stop_logit_mean`、`stop_at_*` 事件统计。

### Teacher Forcing Log-Prob

`_compute_teacher_log_pf_steps`:

- 基于给定动作序列计算 teacher policy log-prob。
- 计算状态序列，收集 outgoing edges。
- 使用 `compute_forward_log_probs` 得到 log-prob。
- 对动作匹配选取 log-prob；STOP 则取 `log_prob_stop`。
- 返回 `log_pf_steps` 与 `RolloutDiagnostics`。

### 训练主流程

`_validate_training_batch`:

- `num_graphs<=0` → 抛错。
- 若包含 `dummy_mask=True` → 抛错（必须 sub 数据集）。

`_resolve_training_specs`:

- `num_miner_rollouts>0`、`num_forward_rollouts>0`，否则抛错。

`_build_training_graphs`:

- 计算 `log_f_nodes`。
- 构建 forward graph cache（start=q, target=a, policy=非逆边）。
- 构建 backward graph cache（start=a, target=q, policy=逆边）。

`_compute_tb_training_loss`:

- 分别计算 backward/forward 视图损失与指标。
- `tb_loss = (num_bwd + num_fwd) / (den_bwd + den_fwd)`。

`_compute_training_loss`:

- 依次完成：batch 准备 → 校验 → masks → specs → graphs → TB loss → 统计补充。

### 评估流程

`_compute_eval_metrics`:

- 仅 forward rollout。
- 对每次 rollout 计算 terminal hits 与 stop 诊断。
- 指标包括 `terminal_hit@k`、`pass@1`、`length_mean`、`zero_hop_rate`、`stop_at_horizon_rate`。
- 最后按 `valid_mask` 过滤取均值。

---

## src/models/components/gflownet_actor.py

### 结构与常量

- `RolloutOutput` 含 `actions_seq/log_pf_steps/num_moves/diagnostics/state_vec_seq`。
- `STOP_RELATION=-1`，`num_steps = max_steps+1`。

### 初始化

- `context_mode` 必须为 `question_start`；`default_mode` 必须为 `forward/backward`。
- `h_transform_bias/h_transform_clip` 必须 ≥0。
- stop head 输入维度为 `hidden_dim + 2`（拼接 logsumexp 与 has_edge）。

### rollout 过程

1. `_resolve_temperature`：
   - `temperature=None` → 用 `policy_temperature`。
   - `< MIN_TEMPERATURE` → greedy。
2. `_resolve_rollout_state`：
   - 可覆盖 `init_node_locals`，但 `init_ptr` 必须提供。
3. `_run_rollout_steps`：
   - 每步计算 `edge_scores/stop_logits` → 采样 → 更新状态。
   - 若全部停止则提前退出。
4. `_finalize_rollout_outputs`：
   - `actions_seq` 必须存在，否则抛错。
   - 补齐 `log_pf_steps` 与 `actions_seq`。
   - 可选输出诊断与 state 序列。

### 每步评分与采样

`_compute_step_scores`:

- `horizon_exhausted = step_counts >= max_steps`。
- `allow_stop = active & (is_target | no_edge | last_step)`。
- 若 `horizon_exhausted` 但不允许 stop → 抛错。

`_sample_actions_from_scores`:

- 调用 `compute_forward_log_probs` 与 `sample_actions`。

### H-Transform 计算

`_compute_edge_scores`:

- `score = prior_scores + h_transform_bias * flow_scores`。

`_compute_flow_scores`:

- 取 tail 节点 `log_f_nodes`，可选 tanh clip。

### Stop 头

`_compute_stop_logits`:

- `logsumexp_edges` 与 `has_edge` 拼接到 state 向量后经 LayerNorm + Linear 输出。

---

## src/models/components/gflownet_env.py

### reset

- 校验 `start_ptr` 长度、起点为 0、非递减。
- 若任一图多个起点 → 抛错（严格单起点）。
- `stopped = missing_start | dummy_mask`。
- `max_steps` 可被 override 下调。

### step

- stop 动作立即标记 stopped。
- move 动作更新 `curr_nodes` 为 tail。
- `step_counts` 仅对未停止图累加。
- `step_counts >= max_steps` 自动停止。

---

## src/models/components/gflownet_layers.py

### EmbeddingBackbone

- 对 node/relation/question embedding 逐一 LayerNorm + Linear。
- `finetune=False` 时冻结全部参数。

### CvtNodeInitializer

- 对 CVT 节点计算入边均值：$m_v = \frac{1}{deg(v)}\sum_{u \to v} (e_u + r_{uv})$。
- 仅对 `node_is_cvt & has_incoming` 的节点替换。

### TrajectoryAgent

- `initialize_state`：融合 question 与 node context，经 MLP/LN 得到 `h0`。
- `step`：GRU 单步更新。
- `precompute_action_keys`：
  - $k_e = [W_r r_e ; W_n x_{tail(e)}] \odot \sigma(W_q q_{graph(e)})$。
- `score_cached`：
  - $s_e = \langle W_q h_{graph(e)}, k_e \rangle$，无效边置 -inf。
- `encode_state_sequence`：
  - 输出 `state_seq=[h_0, h_1, ..., h_{T-1}]` 与 GRU `output`。

### FlowPredictor

- `log_f_i = f([q_{graph(i)}, x_i, g_{graph(i)}])`，MLP 输出标量。

---

## src/models/components/gflownet_ops.py

### Policy 计算

- `compute_policy_log_probs` 处理 stop/edge 的 log-softmax，支持无边与禁止 stop 的图。
- `compute_forward_log_probs` 返回 `PolicyLogProbs(edge, stop, not_stop, has_edge)`。

### Two-Stage Sampling

1. 采样 `stop` vs `move`（Gumbel 可选）。
2. 若 move：先采样 relation，再采样 tail edge。

### CSR 与 OutgoingEdges

- `gather_outgoing_edges` 按 head 节点收集边；空集返回空结构。
- `apply_edge_policy_mask` 按 mask 过滤边，并更新 `edge_counts/has_edge`。

---

## src/models/components/gflownet_reward.py

`GraphFusionReward`:

- `hard_target_bonus >= 0`，否则抛错。
- `log_reward = log(1 + bonus)`（命中）或 `min_log_reward`（未命中）。
- `success = hard_hit`。

---

## src/models/components/trajectory_utils.py

`derive_trajectory`:

- 必须包含 STOP，否则抛错。
- STOP 后动作必须全 STOP，否则抛错。
- 负数动作（非 STOP）非法 → 抛错。

`reflect_backward_to_forward`:

- 基于 `stop_idx` 反转并用 `edge_inverse_map` 映射。
- 无效 `stop_idx` 立即抛错。

`stack_steps`:

- 不足补齐，超过预期长度直接抛错。

---

## src/models/components/__init__.py

仅做符号导出，无额外逻辑。

