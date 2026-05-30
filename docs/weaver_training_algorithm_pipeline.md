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

- `budget: 3`
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

它用于 reward、replay 和 evaluation，不进入 `StateBatch.action_space` 的合法动作判断。

### FeatureEncoder

`FeatureEncoder` 把节点文本、关系文本和问题 embedding 投到 Weaver model space。输入契约是这些 semantic tensor 已经处于 upstream PLM L2 space，模型投影直接信任并消费这些向量：

- 文本节点：upstream PLM L2 semantic embedding 通过节点投影 `Wn`。
- 非文本节点：使用可学习的 `non_text_node_model` token。
- 关系：upstream PLM L2 semantic embedding 通过关系投影 `Wr`。
- query：upstream PLM L2 semantic embedding 通过 query 投影 `Wq`。

输出 `EncodedFeatures`：

- `node_model`
- `edge_relation_model`
- `query_model`

这些 model-space feature 后续被 policy/state encoder 使用。

## 3. State

`StateBatch` 是 canonical evidence graph state。rollout 仍记录 ordered construction trace，
但训练/策略看到的状态是边集合等价类 `s=[tau]_pi`。对 row `r`：

- `S_r`：已选择的证据边集合，对应 `edge_ids[r, :edge_count[r]]`。
- `edge_ids` 的有效区间按物理 edge id 排序，padding 为 `-1`。
- `X_r`：由 anchors 与 `S_r` 中所有边端点派生得到。
- `step_r = |S_r|`。

代码中的不变量是：

```text
X_r = anchors(graph_ids[r]) union endpoints(S_r)
step_r = |S_r|
```

初始状态由 `StateBatch.initial(graph_ids, budget)` 构造：

- `edge_ids` 全 `-1`。
- `edge_count` 为 0。

从 trajectory 记录恢复状态时，先 canonicalize，因此 `(e1,e2)` 与 `(e2,e1)` 是同一个状态。

## 4. Action

每个状态的完整动作空间为：

```text
A(z) = {TERMINAL} union Frontier(z)
```

`TERMINAL` 的 edge id 约定为 `-1`。真实 KG expansion action 使用非负物理 edge id。

### Frontier 合法性

`StateBatch.action_space(graph)` 返回物理有向出边 frontier，不合成 inverse edge。对 row `i`，边 `e = (u, r, v)` 合法当且仅当：

```text
u in X_i
e not in S_i
edge_to_graph[e] == graph_ids[i]
edge_count(i) < budget
```

frontier 会按 state row 分组。`advance`/`branch` 写入新边后会重新排序 selected edge set。

### Transition

选择 expansion edge 后：

```text
S' = S union {e}
X' = X union {src(e), dst(e)}
step' = step + 1
```

选择 `TERMINAL` 不改变 `StateBatch`，而是停止当前 row。

## 5. Policy

`ForwardPolicy` 输出的是 action flow 分布，而不是只输出 action logits。它对每个 state row 产生：

- `terminal_log_flow`
- `continue_log_flow`
- 每条 frontier edge 的 `edge_logit`
- 每条 frontier edge 的 `edge_log_prob`
- 每条 frontier edge 的 `edge_log_flow`
- `state_log_flow`
- 归一化后的 `stop_log_prob`、`expand_log_prob` 与 `edge_action_log_prob`

### StateEncoder

`StateEncoder` 先为每个 row 编码：

```text
query_h      = select_query_model(features, state.graph_ids)
edge_h       = W_src h_src + W_rel h_rel + W_dst h_dst for each edge
row_state_h  = query attends to selected edge_h tokens + anchor tokens
edge_state_h = row_state_h
```

其中线性三路 edge encoder 对一条边 `e = (u, r, v)` 做角色保持的线性投影：

```text
h_e = W_src h_u + W_rel h_r + W_dst h_v
```

然后 state encoder 让 query 读取 selected edge_h tokens + anchor tokens。state 没有已选边时，query 读取 anchor token；如果连 anchor 也没有，则退化到 learned empty token。

### Action Flow Head

当前实现显式建模 terminal head、continue head 和条件 edge 分布。frontier 内边分布直接由 edge logits 做按-row `logsoftmax`。

budget embedding：

```text
remaining = clamp(state.remaining_budget, 0, budget)
budget_h  = Embedding(remaining)
```

stop flow：

```text
G(z, STOP) = stop_head([query_h, row_state_h, budget_h])
```

对 row `z` 的第 `e` 条 frontier edge：

```text
prior_rel(q, e) = <query_sem, rel_sem>
prior_dst(q, e) = 1[dst has text] * <query_sem, dst_text_sem>
s0(q, e)        = alpha_r * prior_rel(q, e) + alpha_v * prior_dst(q, e)
Delta(z, e)     = edge_residual_head([query_h, row_state_h, edge_h, budget_h])
G(z, e)         = s0(q, e) + Delta(z, e)
```

state flow 是 STOP 与所有 legal edge action 的统一 partition：

```text
log F(z) = logsumexp(
    G(z, STOP),
    {G(z, e) : e in Frontier(z)}
)
```

action probability：

```text
log pi(a | z) = G(z, a) - log F(z)
```

action log-prob 由 action log-flow 减去 state log-flow：

```text
log P_F(TERMINAL | z) = -softplus(g(z))
log P_F(e | z)        = log F(z, e) - log F(z)
```

当某个 row 没有 frontier 时，`g(z) = -inf`，terminal probability 为 1。

### Sampling

`ForwardPolicyOutput.sample(rows)` 对每个 row 在 `{TERMINAL} + frontier edges` 上用 Gumbel-max 从 log-prob 采样。被 forced terminal 的 row 不走采样，policy/behavior log-prob 都记为 0。

## 6. Reward

当前 reward 是 `Reward`，只在 terminal state 上计算。对 terminal 子图 `z = (V_z, E_z)`：

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
tape = RolloutTape(R, T = budget + 1)
```

每个 step `t in [0, budget]`：

1. 找出还未停止的 active rows。
2. 对 active rows 取 `active_state`。
3. 构造 frontier。
4. 调 policy 得到 action distribution。
5. 找 forced terminal rows：
   - 没有 frontier；或
   - `depth >= budget`。
6. 对非 forced rows 采样 terminal/edge action。
7. 将 forced 与 sampled action 合并写入 `RolloutTape`。
8. 对 expansion action 调 `state.expand` 更新全局 state。

如果所有 row 都停止，提前 break。循环结束后，如果仍有 row 没记录 terminal step，则把 `terminal_step` 设为 `budget`。

### RolloutResult

`RolloutResult` 记录：

- `source_graph_id`
- `selected_edge_ids: [R, T]`，terminal 或 padding 为 `-1`
- `policy_action_log_prob`
- `behavior_action_log_prob`
- `terminal_step`
- `stop_reason`
- `budget`
- `terminal_state`

派生 mask：

- `valid_mask`: `step <= terminal_step`
- `terminal_mask`: `step == terminal_step`
- `expand_mask`: valid 且 edge id 非负
- `policy_stop_mask`: terminal 且 stop reason 为 policy stop
- `no_frontier_stop_mask`: terminal 且 stop reason 为 no frontier stop
- `budget_truncated_mask`: terminal 且 stop reason 为 budget truncated
- `forced_terminal_mask`: terminal 且 stop reason 非 policy stop

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
2. 对每个 graph 构建 graph-level label view：
   - `target_node_ids`
   - `admissible_edge_mask = union_t shortest_path_edge_mask[t, :]`
3. 在当前 `State.frontier/expand` 语义下，对 `E ⊆ admissible_edge_mask` 且 `|E| <= budget` 的可达 state 做 exact 枚举；每个 visited state 都视为 terminal candidate。
4. 对所有 terminal candidate 直接调用当前 reward model 计算 `log_R(z)`，选出该图 reward 最大的 oracle terminal states。
5. 从 oracle terminal states 的 predecessor DAG 精确回溯合法 action sequences，按数量限制采样 replay trajectories。
6. 比较当前 policy rollout 的 best reward 与 oracle best reward；若 policy 已达到或超过 oracle，则该图 replay 被跳过。

### ReplayBuilder

`ReplayBuilder.build` 将 replay edge sequence 转成 `TrainingBatch`：

- 每条 replay edge 形成一个 expansion event。
- 轨迹末尾补一个 terminal event。
- replay transition 后续用 `SRC_REPLAY` 标记。

这意味着当前 replay 不是 shortest-path path-first teacher，而是：

```text
label-constrained
reward-first
exact terminal-state oracle
```

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
  - `stop_reason`

当前 `RolloutEngine.sample_fused_rollouts` 在采样时同步构造 training transitions：

- expansion step 生成 parent/child transition。
- terminal step 生成 terminal state。

`RolloutRunner.training_batch` 会把 policy rollout 产生的 training batch 标记为 `SRC_POLICY`，把 replay 产生的 training batch 标记为 `SRC_REPLAY`，再用 `concat_reindex_trajectories` 合并并重编号轨迹。

## 10. Backward Policy

训练 expansion event 时使用 `BackwardPolicy`。它不是 rollout policy，也不读 reward。

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
residual_expand = log F(z, a) - log F(z') - log P_B(z | z')
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
    backward_log_prob = BackwardPolicy(...)
else:
    use empty policy output

terminal_frontier = terminals.state.frontier(...)
terminal_out = policy(terminal state)
reward_out = Reward(terminal state)

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

## 14. 条件分支总表：用于裁剪和冗余识别

本节只列当前主算法路径中的条件分支。目标是后续可以逐项判断：保留、删除、合并、参数化，或移到 debug/eval 逻辑中。

### 14.1 Feature / Context 分支

`FeatureEncoder.encode_node_text_semantic`：

```text
if node has text:
    node_text_semantic = entity_text_semantic_table[text_row]
else:
    node_text_semantic = 0
```

`FeatureEncoder.encode_node_model`：

```text
if node_has_text:
    node_model = Linear(node_text_semantic)
else:
    node_model = learned non_text_node_model
```

裁剪判断：

- 如果数据保证所有节点都有文本，`non_text_node_model` 分支可以删除。
- 如果非文本节点很多，这个分支是必要建模能力，不是冗余。

`TargetContext`：

```text
target / shortest-path tensors exist:
    reward、replay、eval 可用
else:
    训练 reward-first replay 无法运行
```

裁剪判断：

- target 不进入 frontier 和 rollout policy，不能为了简化把 target 接进 state/action 合法性，否则会改变算法定义。

### 14.2 State / Frontier 分支

`State.initial`：

```text
if graph has anchors:
    active_node_ids = anchors(graph)
else:
    active_node_ids = empty
```

影响：

- 没 anchor 的 row 初始 frontier 必为空，rollout 会 forced terminal。

`State.frontier(graph, budget)`：

```text
if budget is not None and all depth >= budget:
    return empty_frontier

if cached frontier exists and budget is None:
    return cached frontier

if no active nodes:
    return empty_frontier

if no outgoing edges from active nodes:
    return empty_frontier

filter edges outside current graph
if no same-graph edges:
    return empty_frontier

remove already selected edges
if no remaining edges:
    return empty_frontier

deduplicate (row, edge)

if budget is not None:
    remove rows with depth >= budget
    if no remaining edges:
        return empty_frontier
```

裁剪判断：

- `budget is None` 缓存分支服务 exact predecessor / replay 等无 budget frontier 调用；若统一所有 frontier 都传 budget，需要重新确认 backward predecessor 语义。
- `same graph` 过滤是 batch 图拼接后的安全条件，不能删。
- `already selected` 过滤防止重复边，不能删，除非算法允许 multiset edge。
- `deduplicate` 防止同一边由多个 active node 路径重复出现；在 directed outgoing edge by src 设计下通常重复较少，但仍是安全条件。

`State.expand`：

```text
if rows and edge_ids length mismatch:
    error
if rows empty:
    return self
if validate:
    validate action in current frontier
else:
    validate only row/edge shape, range, one action per row

append selected edge
append src/dst to active nodes
step += 1
```

裁剪判断：

- rollout hot path默认 `validate=False`，依赖 sample 来自 frontier；这减少开销。
- 如果要更强安全性，可以打开 validate，但会增加 frontier 重算。

### 14.3 ForwardPolicy 分支

`ForwardPolicy.action_log_flows`：

```text
stop_log_flow = stop_flow_head(...)
continue_log_gain = -inf
continue_log_flow = -inf

if frontier empty:
    return stop-only output

row_frontier_size = bincount(frontier rows)
edge_log_reference = -log(row_frontier_size)
edge_log_advantage = edge_advantage_head(...)
edge_log_measure = edge_log_reference + edge_log_advantage
continue_log_gain = segment_logsumexp(edge_log_measure by row)
continue_log_flow = stop_log_flow + continue_log_gain
edge_log_flow = stop_log_flow[row] + edge_log_measure
```

`ForwardPolicy.forward`：

```text
state_log_flow = stop_log_flow + softplus(continue_log_gain)
stop_log_prob = -softplus(continue_log_gain)
edge_log_prob = edge_log_flow - state_log_flow[row]
```

裁剪判断：

- 当前主线 edge policy 已切到统一表示：`typed edge token -> state attn pool -> state-action interaction -> flow head`。
- STOP 和 EXPAND 共用同一 `state_h`，不再通过 `4-scalar state summary` 预测停止。
- 主干不再读取 `edge_log_conductance`、`marginal_coverage_gain`、`degree_correction` 或 `frontier size` 这类 handcrafted scalar。

`ForwardPolicyOutput.sample`：

```text
if sampled rows empty:
    return empty

if selected rows have no frontier edges:
    return TERMINAL for all selected rows

else:
    concatenate stop logits and edge logits
    sample by per-row Gumbel-max
```

裁剪判断：

- rollout engine 已经提前把 no-frontier row 标记为 forced terminal，所以 `sample` 内部 no-edge terminal 是防御性分支。若只保留 engine 调用路径，可考虑删除或转 assert。

### 14.4 RolloutEngine 分支

`sample_rollouts`：

```text
with no_grad:
    sample_fused_rollouts
split fused rollouts by rollout_id
```

`sample_fused_rollouts` 每步：

```text
active_rows = rows not stopped
if active_rows empty:
    break

frontier = active_state.frontier(..., budget)
policy_out = policy(active_state, frontier)

forced_local = rows where no frontier OR depth >= budget
sample_rows = all rows except forced_local

actions = []
if sample_rows non-empty:
    actions += policy sampled actions
if forced_local non-empty:
    actions += forced TERMINAL actions with stop_reason

sampled = concat and sort actions by row
write action to tape

if any expansion action:
    build ExpansionBatch parent/child
    update global state by expand

if any terminal action:
    drop budget_truncated terminals from TerminalBatch
    keep policy_stop and no_frontier_stop terminals
```

循环结束：

```text
terminal_step = tape.terminal_step
if any row never stopped:
    terminal_step = budget

if expansion_parts or terminal_parts:
    build TrainingBatch
else:
    training = None
```

stop reason：

```text
POLICY_STOP = 0
NO_FRONTIER_STOP = 1
BUDGET_TRUNCATED = 2
```

裁剪判断：

- `budget_truncated` terminal 当前不会进入 `TerminalBatch`，因此不会直接产生 terminal reward event；它主要保留在 rollout result/eval 里。这是一个可以重点审查的分支。
- forced no-frontier terminal 会进入 training terminal batch，reward 会约束死胡同状态的 stop flow。
- policy stop 与 no-frontier stop 在 `RolloutResult` 中可区分，但进入 SubTB terminal event 后都走同一个 terminal reward 公式。
- `behavior_log_prob` 目前等于 `policy_log_prob`；没有 off-policy correction 使用。若不做重要性采样或行为策略分析，它可能是冗余字段。

### 14.5 RolloutRunner / Replay Budget 分支

`policy_rollouts`：

```text
if num_rollouts <= 0:
    return empty rollouts, None training
else:
    call engine.sample_rollouts
```

`replay_trajectories`：

```text
if num_trajectories <= 0:
    return None
if replay_source is None:
    error
else:
    sample_from_rollouts
```

`training_batch`：

```text
parts = []
if policy_training exists and non-empty:
    add policy_training as SRC_POLICY

if replay exists and num_trajectories > 0:
    if replay_builder is None:
        error
    replay_training = replay_builder.build(...)
    if replay_training non-empty:
        add as SRC_REPLAY

if parts empty:
    return None
else:
    concat_reindex_trajectories(parts)
```

`sample_budget`：

```text
if replay_schedule is None:
    policy_rollout = total
    replay_expand = 0
else:
    weights = schedule.weights_at(progress)
    allocate_replay_budget(total, weights)
```

`allocate_replay_budget`：

```text
if total <= 0:
    return 0, 0
if policy_weight + replay_weight <= 0:
    error

floor proportional allocation
assign remainder to larger fractional part
```

裁剪判断：

- 如果确定不使用 replay，可以删除 replay schedule/source/builder 整条路径，loss 中 source-balanced replay 逻辑也可同步简化。
- 如果 replay 永远启用，`replay_source is None` 与 `replay_builder is None` 可变成构造期校验。
- `SRC_UNKNOWN` 主要作为 build 阶段临时 source，最终通常会被 runner 改成 `SRC_POLICY` 或 `SRC_REPLAY`。

### 14.6 ReplaySource 分支

`replay_trajectories_with_stats`：

```text
if max_trajectories_per_graph is not None and <= 0:
    return empty

targets = reachable_target_node_ids
if no targets:
    return empty

eligible_graphs = graphs containing targets
if no eligible graphs:
    return empty

if reward_model is None or target_context is None:
    error

build graph label views from shortest-path edge masks
compute best policy rollout reward per graph

for each graph view:
    if graph not eligible:
        continue

    enumerate reachable replay state DAG under admissible edges
    if no replay states:
        continue

    score every replay state by reward
    if no terminal candidates:
        continue

    best_oracle = max reward among replay states
    best_policy = best rollout reward for graph
    if best_policy exists and best_policy >= best_oracle:
        skipped_by_reward += 1
        continue

    sample oracle trajectories from best terminal states
    if none sampled:
        continue

    add sampled trajectories
```

`build_replay_graph_label_views`：

```text
for each graph:
    if graph has no targets:
        continue
    admissible_edge_mask = union shortest-path-edge masks over targets
    if no admissible edges:
        continue
    create graph view
```

`enumerate_replay_state_dag`：

```text
start from initial state
for depth in range(budget):
    for current state:
        frontier = parent.frontier(..., budget=budget)
        if frontier empty:
            continue
        keep local row 0 edges
        if no local edges:
            continue
        filter to admissible edges
        if no admissible edges:
            continue
        expand each admissible edge
        dedupe by selected-edge-set key
```

`score_replay_states`：

```text
score all enumerated states by reward
best = max reward
terminal_nodes = all states with reward == best
```

`sample_oracle_trajectories`：

```text
if no terminal nodes:
    return empty

sort terminal states by trajectory_count desc, length asc, key asc
enumerate all predecessor action sequences
dedupe sequences
sort by length asc, lexicographic
if max_trajectories is set:
    truncate
```

裁剪判断：

- replay 依赖 shortest-path edge masks，但最终按 reward 选 terminal state；不是固定 shortest path teacher。
- `best_policy >= best_oracle` 会跳过 replay，属于自适应节省分支；若想稳定 teacher signal，可以考虑删除该 skip。
- exact DAG 枚举复杂度随 budget 和 admissible frontier 增长；这是最可能的训练开销来源之一。
- 空 trajectory 是允许的：如果初始 anchor state 就是 best terminal，ReplayBuilder 会只产生 terminal event。

### 14.7 BackwardPolicy 分支

`BackwardPolicy.log_prob`：

```text
out = zeros
expand = action_edge_ids >= 0
if no expand:
    return zeros

counts = valid_predecessor_count(child_state)
if any expanded row count <= 0:
    error

for each expanded row:
    remove action edge from selected edges
    check parent is exact forward predecessor
    if invalid:
        error

out[expand] = -log(counts)
```

`valid_predecessor_count`：

```text
for each child row:
    if selected edges empty:
        count = 0
        continue
    for each selected edge:
        remove it
        if resulting parent can forward-reach child by that edge:
            count += 1
```

裁剪判断：

- 当前 backward policy 是非参数 uniform kernel；若砍掉多 predecessor 支持，就会把 GFlowNet 的 DAG credit assignment 改成 tree 假设。
- Python loop 是潜在性能热点，但逻辑上保证 exact predecessor。

### 14.8 Reward 分支

`Reward.forward`：

```text
answer_count = |active_nodes intersect target_mask|
failed = answer_count == 0

answer_gain = answer_weight * log1p(answer_count)
edge_penalty = edge_cost * selected_edge_count
fail_penalty = fail_cost * failed

raw_log_reward = answer_gain - edge_penalty - fail_penalty
log_reward = raw_log_reward / reward_temperature
```

构造期参数校验：

```text
if answer_weight <= 0: error
if edge_cost < 0: error
if fail_cost < 0: error
if reward_temperature <= 0: error
```

裁剪判断：

- reward 是 `no_grad`，不训练 reward model。
- `fail_cost` 与 `answer_weight` 都影响命中/未命中的 margin；如果只保留 `answer_count`，失败图可能只由 edge cost 区分。
- `reward_temperature` 只缩放 reward 边界条件，不改变 terminal state 排序。

### 14.9 SubTB Loss 分支

`policy_step_output`：

```text
if terminals.num_items <= 0:
    return zero loss

if expansions.num_items > 0:
    compute parent_frontier, child_frontier, terminal_frontier
    run batched policy for parent/child/terminal states
    compute backward_log_prob for expansion actions
else:
    parent_out = empty
    child_out = empty
    backward_log_prob = empty
    run policy only for terminal states

reward_out = reward(terminals.state)
subtb_input = build_subtb_input(...)
loss = SubTBLoss(...)
```

`build_subtb_input`：

```text
expansion event:
    terminal_log_reward = 0
    terminal = False

terminal event:
    child_state_log_flow = 0
    backward_log_prob = 0
    terminal_log_reward = reward.log_reward
    terminal = True
```

`subtrajectory_terms`：

```text
events = assemble_events
if no events:
    return empty terms

group by trajectory_id
sort each group by step_id

for every start position:
    running_forward = 0
    running_backward = 0
    end_limit = n or start + max_len

    for end position:
        if step is not consecutive:
            break

        running_forward += action_log_prob
        if not terminal:
            running_backward += backward_log_prob

        if terminal:
            residual = start_flow + running_forward - terminal_reward - running_backward
        else:
            residual = start_flow + running_forward - child_flow - running_backward

        if terminal:
            break

if no residuals:
    return empty terms
```

`residual_loss_units`：

```text
if residual_loss == "mse":
    unit = residual^2
else:
    unit = huber(residual, huber_delta)
```

注意：当前代码里非 `"mse"` 都走 Huber，没有显式校验 `residual_loss == "huber"`。

`weighted_source_balanced_mean`：

```text
if no values:
    return 0

policy_mask = source in {SRC_POLICY, SRC_UNKNOWN}
replay_mask = source == SRC_REPLAY

if no policy samples:
    return alpha_replay * replay_loss
if no replay samples:
    return policy_loss
else:
    return policy_loss + alpha_replay * replay_loss
```

裁剪判断：

- 如果禁用 replay，可以把 source-balanced mean 简化成单一 weighted mean。
- 如果只保留单步 DB loss，可以删除 `subtrajectory_terms` 中所有多长度枚举与 `subtb_lambda/max_len`。
- `action_log_flow` 只用于单步 branch diagnostics 和 build input 字段；真正 SubTB residual 使用 `action_log_prob` 加首状态 flow。

### 14.10 Training / Optimization 分支

`compute_step`：

```text
rollout_batch = runner.train_rollouts(...)
if not rollout_batch.has_transitions:
    raise RuntimeError
else:
    policy_step_output(...)
```

`training_step`：

```text
optimizer.zero_grad()
manual_backward(loss)
if gradient_clip_val is set and > 0:
    clip gradients
optimizer.step()
if scheduler interval == "step":
    scheduler.step()
log metrics
```

`validation/test/predict`：

```text
with no_grad:
    eval_rollouts only
    no replay
    no loss update
```

裁剪判断：

- manual optimization 是 Lightning 风格选择；若不需要 branch gradient diagnostics，可以切回自动优化，但要同步改 logging。
- eval selector 使用 trajectory probability / flow / reward 等指标，不影响训练 loss；可单独裁剪评估复杂度。

## 15. 当前可优先审查的冗余/复杂度候选

这些不是结论，只是基于分支复盘得到的优先检查点：

1. `behavior_action_log_prob`：当前等于 `policy_action_log_prob`，没有看到 off-policy correction 使用。
2. `budget_truncated` terminal：不进入 `TerminalBatch` reward event，只留在 rollout/eval 统计里。
3. `ForwardPolicyOutput.sample` 内 no-frontier fallback：engine 已提前 forced terminal，可能是防御性重复。
4. `residual_loss` 字符串：除 `"mse"` 外全部走 Huber，缺少显式非法值报错。
5. replay 的 exact DAG 枚举：算法上清晰，但可能是训练耗时最大来源。
6. `SRC_UNKNOWN`：多数情况下只是中间态 source id，最终会被 runner 改写。
7. `action_log_flow` 字段：SubTB 主 residual 不直接用它，主要服务单步 diagnostics。
8. `expand(validate=False)` 与后续 backward exact predecessor 校验并存：一个为速度，一个为训练一致性；是否都需要取决于是否保留 replay/DB diagnostics。

## 16. 关键实现约束与注意点

- `TERMINAL` 是 action，不是 KG edge；其 id 固定为 `-1`。
- frontier 只看当前 active nodes 的出边，不读答案标签。
- reward 只在 terminal state 上计算。
- expansion 的 child flow 必须通过 child state 的显式 policy forward 得到。
- replay trajectory 会被转为和 policy rollout 相同结构的 `TrainingBatch`，再通过 source id 在 loss 中平衡。
- forced terminal 的采样 log-prob 记为 0，它表示结构或 budget 强制停止，而不是 policy 主动选择停止。
- 当前 policy 训练的是 action-flow 一致性；`action_log_prob` 用于 SubTB 子轨迹残差，`action_log_flow` 主要用于单步 branch diagnostics。
