# Weaver 核心训练算法 Pipeline

本文复盘当前代码中 Weaver 模型呈现出的核心训练算法。整体上，它不是普通的逐边分类或 REINFORCE，而是一个以 terminal/edge action flow 为参数化对象、用 SubTB 目标训练的 GFlowNet 风格证据子图生成器。

主要代码入口：

- 训练入口：`src/train.py`
- Lightning 模块：`src/weaver/module.py`
- 状态与 frontier：`src/weaver/state.py`
- forward policy：`src/weaver/policy/forward.py`
- rollout：`src/weaver/rollout/engine.py`
- replay：`src/weaver/rollout/replay.py`
- reward：`src/weaver/utility/reward.py`
- SubTB loss：`src/weaver/objectives/subtb.py`

## 1. 一句话算法概览

给定一个问题图，模型从问题 anchor 节点出发，逐步选择一条合法出边加入证据子图，或选择 `TERMINAL` 停止。每个 terminal 子图按是否覆盖答案目标与边数成本得到 reward。训练时，代码采样 policy rollout，并混入最短路 replay 轨迹，把轨迹拆成 expansion/terminal events，用 SubTB 子轨迹一致性损失约束 forward flow 与 terminal reward 对齐。

默认配置见 `configs/model/weaver.yaml`：

- `expand_budget: 3`
- `train_num_rollouts: 8`
- `eval_num_rollouts: 16`
- `subtb_lambda: 0.9`
- `residual_loss: huber`
- `huber_delta: 2.0`
- `alpha_replay: 1.0`
- reward 默认 `edge_cost: 0.05`、`fail_cost: 1.0`、`reward_temperature: 1.0`

## 2. Static Context 与 Feature

### GraphContext

`GraphContext.from_batch` 从 `RetrievalBatch` 构造静态图上下文，包含：

- `edge_index: [2, E]`：物理有向 KG 边。
- `node_to_graph: [N]`：节点所属图。
- `edge_to_graph: [E]`：边所属图，按 source node 所在图推断。
- `anchor_mask: [N]`：问题 anchor 节点。
- `adjacency`：按 source/destination 建好的 CSR 邻接索引。

`GraphContext` 明确不包含 target label、reward、oracle path 或 rollout 轨迹，因此 rollout 和 inference 的 frontier 构造不直接读答案。

### TargetContext

`TargetContext.from_batch` 是监督上下文，包含：

- `target_mask: [N]`：可达答案节点 mask。
- `reachable_target_node_ids` 与 ptr。
- `target_count_by_graph`。
- shortest-path 监督相关扁平张量。

它用于 reward、replay 和 evaluation，不进入 `State.frontier` 的合法动作判断。

### FeatureEncoder

`FeatureEncoder` 把节点文本、关系文本和问题 embedding 投到 Weaver model space：

- 文本节点：由 upstream PLM semantic embedding 投影到 model space。
- 非文本节点：使用可学习的 `non_text_node_model` token。
- 关系和 query：同样通过 `project_to_model + LayerNorm`。

输出 `EncodedFeatures`：

- `node_model`
- `edge_relation_model`
- `query_model`

这些 model-space feature 后续被 policy/state encoder 使用。

## 3. State

`State` 是动态证据子图状态。对 rollout row `r`：

- `S_r`：已选择的证据边集合，对应 `selected_edge_mask[r, :]`。
- `X_r`：当前 active node 集合，对应 `active_node_mask[r, :]`。
- `step_r`：已执行的 expansion 次数，也就是 depth。

代码中的不变量是：

```text
X_r = anchors(graph_ids[r]) union endpoints(S_r)
step_r = |S_r|    在合法 transition 下成立
```

初始状态由 `State.initial(graph, graph_ids)` 构造：

- `selected_edge_mask` 全 False。
- `active_node_mask` 只激活该图的 anchor 节点。
- `step` 为 0。

从已选边恢复状态时，`State.from_selected_edges` 会重新激活 anchor 与所有已选边端点，并检查边是否属于对应图。

## 4. Action

每个状态的完整动作空间为：

```text
A(z) = {TERMINAL} union Frontier(z)
```

`TERMINAL` 的 edge id 约定为 `-1`。真实 KG expansion action 使用非负物理 edge id。

### Frontier 合法性

`State.frontier(graph, expand_budget)` 返回物理有向出边 frontier，不合成 inverse edge。对 row `i`，边 `e = (u, r, v)` 合法当且仅当：

```text
u in X_i
e not in S_i
edge_to_graph[e] == graph_ids[i]
depth(i) < expand_budget    如果传入 expand_budget
```

frontier 会对 `(row_id, edge_id)` 去重。`State.expand` 在加入边前会调用 `validate_expansion_actions`，确保每个 row 最多一个 expansion action，且 action 必须在当前 frontier 中。

### Transition

选择 expansion edge 后：

```text
S' = S union {e}
X' = X union {src(e), dst(e)}
step' = step + 1
```

选择 `TERMINAL` 不调用 `State.expand`，而是停止当前 row。

## 5. Policy

`ForwardPolicy` 输出的是 action flow 分布，而不是只输出 action logits。它对每个 state row 产生：

- `terminal_log_flow`
- 每条 frontier edge 的 `edge_log_flow`
- `state_log_flow`
- 归一化后的 `terminal_log_prob` 与 `edge_log_prob`

### StateEncoder

`StateEncoder` 先为每个 row 编码：

```text
query_h      = select_query_model(features, state.graph_ids)
edge_state_h = mean_pool(EdgeEncoder(selected edges))
row_state_h  = MLP([query_h, edge_state_h])
```

其中 `EdgeEncoder` 对一条边 `e = (u, r, v)` 只做 role-preserving 拼接：

```text
h_e = concat(h_u, h_r, h_v)
```

如果当前 state 没有已选边，则 `edge_state_h` 为 0。

### Action Flow Head

terminal flow：

```text
u(z) = terminal_flow_head([query_h, row_state_h])
```

continuation flow：

```text
c(z) = continuation_flow_head([query_h, row_state_h])
```

edge policy：

```text
log π(e | z) = logsoftmax_e edge_policy_head([query_h, row_state_h, edge_h])
```

edge action-flow：

```text
log F(z, e) = c(z) + log π(e | z)
```

state flow 是 terminal 与 continuation 的二项 logaddexp：

```text
log F(z) = logaddexp(u(z), c(z))
```

action log-prob 由 action log-flow 减去 state log-flow：

```text
log P_F(TERMINAL | z) = u(z) - log F(z)
log P_F(e | z)        = c(z) - log F(z) + log π(e | z)
```

当某个 row 没有 frontier 时，`c(z) = -inf`，terminal probability 为 1。

### Sampling

`ForwardPolicyOutput.sample(rows)` 对每个 row 在 `{TERMINAL} + frontier edges` 上用 Gumbel-max 从 log-prob 采样。被 forced terminal 的 row 不走采样，policy/behavior log-prob 都记为 0。

## 6. Reward

当前 reward 是 `TrueTerminalReward`，只在 terminal state 上计算。对 terminal 子图 `z = (V_z, E_z)`：

```text
answer_count(z) = |V_z intersect Y|

raw_log_R(z)
  = answer_weight * log(1 + answer_count(z))
    - edge_cost * |E_z|
    - fail_cost * 1[answer_count(z) = 0]

log_R(z) = raw_log_R(z) / reward_temperature
```

实现细节：

- `V_z` 来自 `state.active_node_mask`。
- `E_z` 来自 `state.edge_mask`。
- `Y` 来自 `TargetContext.target_mask`。
- reward 不使用 shortest-path edge label；shortest-path 信息只用于 replay、diagnostics 或 evaluation。

这个 reward 鼓励终止子图覆盖答案节点，同时惩罚更长的证据边集合；完全没命中答案时额外扣 `fail_cost`。

## 7. Rollout 逻辑

训练和评估 rollout 共用 `RolloutEngine`。

### sample_fused_rollouts

`RolloutEngine.sample_fused_rollouts` 会把每个 graph 复制 `rollouts_per_graph` 个 row：

```text
graph_ids = arange(num_graphs).repeat_interleave(rollouts_per_graph)
state = State.initial(graph, graph_ids)
tape = RolloutTape(R, T = expand_budget + 1)
```

每个 step `t in [0, expand_budget]`：

1. 找出还未停止的 active rows。
2. 对 active rows 取 `active_state`。
3. 构造 frontier。
4. 调 policy 得到 action distribution。
5. 找 forced terminal rows：
   - 没有 frontier；或
   - `depth >= expand_budget`。
6. 对非 forced rows 采样 terminal/edge action。
7. 将 forced 与 sampled action 合并写入 `RolloutTape`。
8. 对 expansion action 调 `state.expand` 更新全局 state。

如果所有 row 都停止，提前 break。循环结束后，如果仍有 row 没记录 terminal step，则把 `terminal_step` 设为 `expand_budget`。

### RolloutResult

`RolloutResult` 记录：

- `source_graph_id`
- `selected_edge_ids: [R, T]`，terminal 或 padding 为 `-1`
- `policy_action_log_prob`
- `behavior_action_log_prob`
- `terminal_step`
- `forced_terminal`
- `expand_budget`

派生 mask：

- `valid_mask`: `step <= terminal_step`
- `terminal_mask`: `step == terminal_step`
- `expand_mask`: valid 且 edge id 非负
- `forced_terminal_mask`: terminal 且 forced

`policy_trajectory_log_prob` 是 valid step 上 policy action log-prob 的和。

## 8. Replay 逻辑

训练时 `RolloutRunner.train_rollouts` 会先按 replay schedule 分配预算：

```text
ReplaySampleBudget(policy_rollout, replay_expand)
```

默认 schedule 中 policy 与 replay 权重都是 1，因此总 `train_num_rollouts` 会按权重拆分为 policy rollout 数与 replay 轨迹数。

### ReplaySource

`ReplaySource.sample_from_rollouts` 调 `replay_trajectories_with_stats`，为有 reachable target 的图生成 replay 轨迹。核心逻辑：

1. 找出含 target 的 eligible graphs。
2. 为每个 reachable target 构建 `ReplayTargetView`，其中包含节点到 target 的距离、edge shortest-path count 等预计算监督。
3. 从该图 anchor 出发，用 `ranked_next_edges` 选择满足 `dst_dist = current_dist - 1` 且 `edge_counts > 0` 的出边，beam 式生成到 target 的短路径。
4. 按 `(路径长度, edge ids)` 排序、去重，并限制每图数量。
5. 如果传入 reward model 与 target context，则比较当前 policy rollouts 的 best reward 与 replay candidates 的 best oracle reward；若 policy 已经达到或超过 oracle，则该图 replay 会被跳过。

### ReplayBuilder

`ReplayBuilder.build` 将 replay edge sequence 转成 `TrainingBatch`：

- 每条 replay edge 形成一个 expansion event。
- 轨迹末尾补一个 terminal event。
- replay transition 后续用 `SRC_REPLAY` 标记。

## 9. TrainingBatch 与事件

训练样本统一为 `TrainingBatch`，包含：

- `ExpansionBatch`
  - `parent: State`
  - `child: State`
  - `edge_ids`
  - `meta`
- `TerminalBatch`
  - `state`
  - `meta`
  - `forced_terminal`

`training_from_rollouts` 会按 rollout tape 重放 trajectory：

- expansion step 生成 parent/child transition。
- terminal step 生成 terminal state。

`RolloutRunner.training_batch` 会把 policy rollout 产生的 training batch 标记为 `SRC_POLICY`，把 replay 产生的 training batch 标记为 `SRC_REPLAY`，再用 `concat_reindex_trajectories` 合并并重编号轨迹。

## 10. Backward Policy

训练 expansion event 时使用 `UniformValidPredecessorBackwardPolicy`。它不是 rollout policy，也不读 reward。

对 child state `S'`，可删除边 `e` 是合法 predecessor 当且仅当：

```text
S = S' \ {e}
S 可以通过当前 frontier 语义 forward-reachable
e in Frontier(S)
```

若 child state 有 `|Pred(S')|` 个合法 predecessor，则：

```text
log P_B(S | S') = -log |Pred(S')|
```

terminal event 的 backward log-prob 为 0。

## 11. Loss: SubTB

`WeaverModule.policy_step_output` 对 training batch 做三类 policy/reward 前向：

1. expansion parent state：
   - 得到 parent action flow/prob。
2. expansion child state：
   - 得到 child state flow。
3. terminal state：
   - 得到 terminal action flow/prob。
   - 调 reward model 得到 `log_R`。

然后 `build_subtb_input` 构造 `SubTBEventBatch`。

### 单步 expansion event

对 expansion action `z -> z'`：

- `parent_state_log_flow = log F(z)`
- `child_state_log_flow = log F(z')`
- `action_log_prob = log P_F(a | z)`
- `action_log_flow = log F(z, a)`
- `backward_log_prob = log P_B(z | z')`
- `terminal_log_reward = 0`
- `terminal = False`

单步 diagnostic residual：

```text
residual_expand = c(z) + log π(a | z) - log F(z') - log P_B(z | z')
```

### 单步 terminal event

对 terminal action：

- `parent_state_log_flow = log F(z)`
- `child_state_log_flow = 0`
- `action_log_prob = log P_F(TERMINAL | z)`
- `action_log_flow = log F_terminal(z)`
- `backward_log_prob = 0`
- `terminal_log_reward = log R(z)`
- `terminal = True`

单步 diagnostic residual：

```text
residual_terminal = u(z) - log R(z)
```

### SubTB 子轨迹 residual

`subtrajectory_terms` 会按 `trajectory_id` 分组，并按 `step_id` 排序。对每条连续子轨迹片段，从 `start` 到 `end` 累加 forward action log-prob 和 backward log-prob。

如果子轨迹终点不是 terminal：

```text
residual
  = log F(z_start)
    + sum log P_F(a_t | z_t)
    - log F(z_end_child)
    - sum log P_B(z_t | z_{t+1})
```

如果子轨迹终点是 terminal：

```text
residual
  = log F(z_start)
    + sum log P_F(a_t | z_t)
    - log R(z_terminal)
    - sum log P_B(z_t | z_{t+1})
```

子轨迹长度为 `L` 时，权重为：

```text
weight = subtb_lambda ** (L - 1)
```

如果配置了 `max_len`，只枚举最长不超过 `max_len` 的子轨迹；默认 `max_len: null`，即不额外截断。

### Residual unit 与 source-balanced mean

`residual_loss_units` 支持：

- `mse`: `residual^2`
- `huber`: 默认，delta 为 `huber_delta`

最终 loss 先分别对 policy/source unknown 与 replay source 做加权均值：

```text
policy_loss = weighted_mean(units, weights, source in {SRC_POLICY, SRC_UNKNOWN})
replay_loss = weighted_mean(units, weights, source == SRC_REPLAY)
```

组合方式：

```text
if no policy samples:
    loss = alpha_replay * replay_loss
elif no replay samples:
    loss = policy_loss
else:
    loss = policy_loss + alpha_replay * replay_loss
```

默认 `alpha_replay = 1.0`。

## 12. 一次 training_step 的完整流程

`WeaverModule.training_step` 使用 manual optimization。核心流程如下：

```text
training_step(batch):
    output = compute_step(batch)

    optimizer.zero_grad()
    collect terminal/expansion branch gradient diagnostics
    manual_backward(output.loss)
    collect policy gradient norm metrics
    clip gradients if configured
    optimizer.step()
    step scheduler if interval == "step"
    log train metrics
```

`compute_step(batch)`：

```text
graph = GraphContext.from_batch(batch)
target = TargetContext.from_batch(batch, graph)

policy_features = FeatureEncoder(batch)

rollout_batch = RolloutRunner.train_rollouts(
    policy,
    batch,
    context=graph,
    features=policy_features,
    reward_model,
    target_context=target,
)

training = rollout_batch.training
assert training has transitions

output = policy_step_output(
    graph,
    target,
    policy_features,
    training,
)
```

`policy_step_output(...)`：

```text
if expansions exist:
    parent_frontier = expansions.parent.frontier(...)
    child_frontier = expansions.child.frontier(...)
    parent_out = policy(parent state)
    child_out = policy(child state)
    backward_log_prob = UniformValidPredecessorBackwardPolicy(...)
else:
    use empty policy output

terminal_frontier = terminals.state.frontier(...)
terminal_out = policy(terminal state)
reward_out = TrueTerminalReward(terminal state)

subtb_input = build_subtb_input(...)
loss = SubTBLoss(subtb_input)
```

## 13. 评估与预测中的 rollout

`validation_step`、`test_step` 和 `predict_step` 不采 replay，只调用：

```text
RolloutRunner.eval_rollouts(policy, context, features)
```

评估会把多个 rollout sample 转成 terminal subgraph，并计算 retrieval、compactness、diversity、calibration 等指标。部分 selector 会用：

- trajectory probability
- terminal flow
- state flow
- reward oracle/best

但这些 evaluation selector 不改变训练 loss 本身。

## 14. 关键实现约束与注意点

- `TERMINAL` 是 action，不是 KG edge；其 id 固定为 `-1`。
- frontier 只看当前 active nodes 的出边，不读答案标签。
- reward 只在 terminal state 上计算。
- expansion 的 child flow 必须通过 child state 的显式 policy forward 得到。
- replay trajectory 会被转为和 policy rollout 相同结构的 `TrainingBatch`，再通过 source id 在 loss 中平衡。
- forced terminal 的采样 log-prob 记为 0，它表示结构或 budget 强制停止，而不是 policy 主动选择停止。
- 当前 policy 训练的是 action-flow 一致性；`action_log_prob` 用于 SubTB 子轨迹残差，`action_log_flow` 主要用于单步 branch diagnostics。
