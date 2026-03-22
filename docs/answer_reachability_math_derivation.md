# Answer Reachability Math Derivation

本文只解释当前主线实现，也就是：

- 模型：`src/models/gflownet_module.py`
- 策略：`src/models/gflownet/policy.py`
- 采样：`src/models/gflownet/sampler.py`
- 主损失：`src/models/gflownet/losses.py`
- 精确分析：`src/metrics/answer_reachability/exact_analysis.py`
- replay：`src/models/gflownet/replay.py`

这份文档只描述当前代码里的数学对象，不复述已经删除的旧版
`trajectory_policy` / `guidance_cfg` / 多目标混训设计。

## 1. 任务、状态与记号

给定一个样本图 `G = (V, E)`、问题起点集合 `Q`、答案节点集合 `A`，以及最大步数
`H`，当前实现把搜索过程定义在一个有限 horizon 的离散 prefix-tree 上。

环境状态不是静态 `(node, time)`，而是精确离散前缀：

```text
x_t = (v_0, r_1, v_1, ..., r_t, v_t)
```

其中：

- `v_0 in Q`
- `t in {0, 1, ..., H}` 表示已经执行的 graph move 数
- 当前节点是 `v_t`

为了高效前向打分，策略不会在每一步重新对整段 prefix 做 self-attention，而是额外维护一
个连续控制状态：

```text
c_t in R^d
```

`c_t` 压缩了问题条件下的 prefix 历史；而 `x_t` 仍以离散 `path_token_ids` 的形式保留在
环境状态里，用于 backward、replay、trace 和离线路径重建。

下面统一使用这些记号：

- `F(x_t)`: prefix state `x_t` 的非归一化 flow。
- `f(x_t) = log F(x_t)`: 对应的 log-flow。
- `P_F^start(q)`: 从起点集合中选择起始节点 `q` 的目标分布。
- `P_F(e | x_t)`: 在 prefix `x_t` 上选择 graph move `e = (v_t -r-> u)` 的目标前向分布。
- `P_B(parent(x_t) | x_t)`: 非根 prefix 的目标后向分布。
- `R(tau)`: 轨迹 `tau` 的终止奖励。

当前实现采用共享 encoder + 两个学习头 + 一个精确 backward kernel：

- state-flow head：输出 `f(x_t)`
- forward-policy head：输出 `P_F`
- backward kernel：从离散 prefix 直接恢复唯一合法 parent

也就是说，当前主线不再使用历史 path self-attention，也不再学习一个独立 backward
head。

## 2. 状态流参数化

`prepare_batch()` 先一次性编码整张图与问题，得到：

- 节点表示 `z_v`
- 关系表示 `z_r`
- 全局问题向量 `q_root`
- 问题 token 序列 `H_Q`

### 2.1 recurrent prefix controller

控制状态以全局问题向量为根：

```text
c_root = q_root
```

对起点 `q in Q`，当前实现先用一个 learned start relation token 触发第一次 controller
更新：

```text
a_root = Attn(W_q c_root, H_Q)
c_0(q) = LN(GRU([a_root; z_q; z_start], c_root))
```

对任意后续 graph move `x_t --r_{t+1}--> x_{t+1}`，controller 按相同模式递推：

```text
a_t = Attn(W_q c_t, H_Q)
c_{t+1} = LN(GRU([a_t; z_{v_{t+1}}; z_{r_{t+1}}], c_t))
```

这里：

- `Attn` 是当前 `control_state` 对问题 token 的单头注意
- `GRU` 是轻量 prefix updater
- `LN` 表示 LayerNorm

如果 rollout 时已经显式携带 `control_state`，策略直接复用；如果只给离散
`path_token_ids`，策略会按上面的递推重新回放 prefix，重建对应的 `c_t`。

### 2.2 state feature 与 log-flow

先构造与当前节点和步数有关的静态基底：

```text
b(x_t) = z_{v_t} + step_embed(t) + remaining_embed(H - t)
```

再把这个基底与控制状态拼接，经过小 MLP 得到真正的 state feature：

```text
phi(x_t) = LN(MLP(LN([b(x_t); c_t])))
```

统一的状态流头输出：

```text
f(x_t) = log F(x_t)
```

因此当前主线学习的是：

```text
x_t -> c_t -> phi(x_t) -> f(x_t)
```

也就是说，flow 的数学状态仍是 exact prefix state，而神经表征是它的 recurrent
compression。

## 3. 起点分布与隐式虚拟源

当前实现把多起点问题写成一个隐式虚拟源 `s_root` 指向所有真实起点 prefix：

```text
x_0(q) = (q), q in Q
```

对于每个起点候选 `q in Q`，先通过上面的 start controller 得到 `c_0(q)`，再计算：

```text
f(x_0(q)) = log F(x_0(q))
```

graph 的根流量由所有起点流量归一化得到：

```text
Z = sum_{q in Q} F(x_0(q))
log Z = logsumexp_{q in Q} f(x_0(q))
```

因此目标起点分布不是额外学习的，而是直接由起点流量归一化得到：

```text
P_F^start(q)
= F(x_0(q)) / Z
= exp(f(x_0(q)) - log Z)
```

这正是 `build_start_distribution_from_log_flows()` 的数学含义。

由此立即得到一个重要恒等式：

```text
f(x_0(q)) - log P_F^start(q) = log Z
```

这个恒等式解释了为什么当前 `SubTB` 根边界和起点边界能天然对齐。

## 4. 前向 actor 与 backward kernel

### 4.1 state-flow head

state-flow head 只负责输出：

```text
f(x_t) = log F(x_t)
```

它不再直接诱导 `P_F`，而是作为独立的 flow anchor 进入 `SubTB`。

### 4.2 forward-policy head

给定当前 prefix `x_t` 的 state feature `phi(x_t)`，以及候选边
`e = (v_t -r-> u)`，forward head 直接输出未归一化 logit：

```text
ell_F(e | x_t) = g_theta(phi(x_t), z_u, z_r)
```

这里 actor 读取的是：

- 当前 prefix 的 state feature
- 静态候选节点表示 `z_u`
- 关系表示 `z_r`

它不再对整段 path 做 self-attention，也不要求先显式构造 child prefix 的完整 state
feature。

另外还有一个显式 submit 动作：

```text
ell_submit(x_t) = g_theta(phi(x_t), z_{v_t}, z_submit)
```

最终目标前向分布在所有合法 outgoing edges 与 submit 动作上归一化：

```text
P_F(a | x_t) = exp(ell(a | x_t)) / sum_{a' in A(x_t)} exp(ell(a' | x_t))
```

### 4.3 backward kernel

对非根 prefix，当前实现不学习独立 backward head，而是直接从 `path_token_ids` 恢复：

```text
parent(x_t)
```

也就是该 prefix 在离散 prefix-tree 上唯一合法的父状态。因此在 graph move 部分：

```text
P_B(parent(x_t) | x_t) = 1
P_B(x' | x_t) = 0,  for x' != parent(x_t)
```

实现上，代码会根据已编码 prefix 取出“上一个节点 + 最后一步 relation”，然后在 incoming
edges 中找到与之匹配的那条父边。

因此当前训练目标里的非终止 `log P_B` 不再来自 learned backward，也不来自 uniform
indegree，而是来自 exact parent recovery。

## 5. heuristic、behavior policy 与采样温度

当前实现里 heuristic 不进入目标分布 `P_F` / `P_B`，而只进入 behavior policy，也就是
训练期的探索分布。

如果记启发式 bias 为 `h(x_t)`，权重为 `beta`，则 behavior 分布使用：

```text
Q_F^start(q) prop exp(f(x_0(q)) + beta h(x_0(q)))
Q_F(e | x_t) prop exp(ell_F(e | x_t) + beta h(child(x_t, e)))
```

其中 `h` 的来源可以是：

- `topology`
- `embedding`
- `learned`

见 `SearchHeuristic`。

训练时真正采样边还会再经过温度 `tau`：

```text
Q_sample(a | x_t) = softmax(logits_behavior / tau)
```

因此当前训练链条是：

- 用 behavior distribution 提高探索质量；
- 用 target distribution 重新计算 `log P_F`，并记录 backward 量做诊断/兼容；
- 用前向子轨迹的一致性残差做 `SubTB`。

## 6. 终止规则与奖励

一条 rollout 会在以下任一条件下终止：

1. 当前节点已经是答案节点；
2. 当前状态没有合法 move；
3. 步数达到 `H`。

终止奖励定义为：

```text
R(tau) = 1,                      if terminal node in A
R(tau) = epsilon,                if failure_reward_mode = constant
R(tau) = epsilon / N_nonanswer,  if failure_reward_mode = graph_normalized
```

其中 `N_nonanswer` 是该 graph 的非答案节点数，至少截到 `1`。

当 `training.answer_reward.mode = binary_ranking` 时，当前实现改为：

```text
R(tau) = epsilon + exp(beta * u(y(tau)))
```

其中：

- `y(tau)` 是 terminal entity；
- `u(y)` 对 gold answer 取 `positive_utility`，对非 gold answer 取 `negative_utility`。

如果 `training.answer_reward.terminal_reward_scale = entity_alias_count`，代码还会再做一步
启发式缩放：

```text
R(tau) = [epsilon + exp(beta * u(y(tau)))] / alias_count_graph(y(tau))
```

这里的 `alias_count_graph(y)` 只是“同一实体在当前 graph 中出现了多少个节点副本”的计数。
它只能部分缓解 duplicate entity node 带来的放大效应，并不能校正 tree policy 下的
path multiplicity bias。也就是说，如果同一实体的不同 alias 节点背后可达路径数差异很大，
这个缩放仍然不是严格的 entity-level unbiased normalization，而只是 alias-level
heuristic。

当 `training.answer_reward.mode = entity_sink` 时，reward 本身保留 entity-level 形式：

```text
R_sink(y) = epsilon + exp(beta * u(y))
```

如果 `training.answer_reward.length_penalty_alpha = alpha > 0`，终止 reward 会再乘一个
随 graph move 步数衰减的长度因子：

```text
R_sink(y, t) = R_sink(y) * exp(-alpha * t)
```

其中 `t` 是命中答案前的 graph move 次数，也就是 `terminal_num_steps`；最终的
`submit -> sink(y)` 动作本身不额外计步。

不再把 alias-count 缩放直接塞进 reward，而是把最后一步 `submit -> sink(y)` 当成显式终止
转移，并给它一个单独的 backward kernel。当前默认近似是：

```text
P_B(parent | sink(y)) = 1 / alias_count_graph(y)
```

对应的 terminal backward log-prob 为：

```text
log P_B(parent | sink(y)) = -log alias_count_graph(y)
```

这样把“实体奖励”和“终止 backward 近似”在实现上拆开了，更接近 entity-sink 语义；但要强调，
`uniform_entity_alias` 仍然只是在 alias 节点层面近似 terminal parent 分布，并没有解决
path-dependent tree policy 下真正的 path multiplicity 归一化问题。

所以终止 log-reward 为：

```text
log R(tau) = log R_base(tau) - alpha * t(tau)
```

这就是 `TrajectoryGFNSampleBatch.terminal_log_rewards` 的来源。

## 7. 当前 `SubTB` 实际约束了什么

这里必须以 `src/models/gflownet/losses.py` 为准。

当前 `SubTrajectoryBalanceLoss.compute()` 读取的主要量是：

- `start_state_log_f`
- `next_state_log_f_steps`
- `log_pf_steps`
- `terminal_log_rewards`

虽然 sample batch 里也携带 `log_pb_steps` 和 `graph_log_z`，但当前 loss 实现没有把它们写进
训练残差；它们目前主要保留给日志、诊断和兼容接口。

### 7.1 一条轨迹上的前缀量

设一条 rollout 为：

```text
tau = (x_0, x_1, ..., x_T)
```

其中：

- `x_0` 是 sampled start prefix；
- `x_{k+1}` 对应第 `k` 次 graph move 之后的状态；
- `x_T` 是 rollout 的最后一个实际状态。

定义前向前缀和：

```text
G_k^F = sum_{i < k} log P_F(x_{i+1} | x_i)
```

### 7.2 状态锚点和终止锚点

对非终止状态，定义：

```text
A_k = log F(x_k) - G_k^F
```

对终止位置，不再使用 `log F(x_T)`，而是直接用 reward 锚点替换：

```text
A_term = log R(tau) - G_T^F
```

### 7.3 当前实现的 pairwise residual

对任意 `0 <= i < j < T`，代码实际构造：

```text
Delta_{i,j} = A_i - A_j
```

展开就是：

```text
Delta_{i,j}
= log F(x_i)
 + sum_{k=i}^{j-1} log P_F(x_{k+1} | x_k)
 - log F(x_j)
```

对任意中间状态到 terminal 的残差则是：

```text
Delta_{i,term} = A_i - A_term
```

也就是：

```text
Delta_{i,term}
= log F(x_i)
 + sum_{k=i}^{T-1} log P_F(x_{k+1} | x_k)
 - log R(tau)
```

### 7.4 加权 `SubTB`

如果 `lambda_weight = lambda`，当前实现对更长的子段做指数衰减：

```text
w_{i,j} = lambda^(j - i - 1)
```

单条 rollout 的 loss 可以写成：

```text
L_subtb(tau)
= WeightedMean({Delta_{i,j}^2, Delta_{i,term}^2})
```

最后 batch loss 为所有 sampled rollout 的平均。

因此当前实现更准确的一句话是：

```text
用 prefix state log-flow、forward log-prob 和 terminal log-reward
做前向子轨迹一致性约束。
```

## 8. Success replay 的数学逻辑

当前 replay buffer 不缓存旧参数下的 tensor，而只缓存成功轨迹的离散骨架：

```text
(sample_id, start_local_node, local_edge_ids)
```

这意味着 replay 的不是“旧 policy 的数值”，而是“旧成功路径在当前 policy 下的重新打分”。

### 8.1 replay 比例

设每个 graph 的 on-policy rollout 数为 `K`，希望 replay 轨迹在合并后的总轨迹里占比为
`r`，那么：

```text
r = K_rep / (K + K_rep)
```

解得：

```text
K_rep = K r / (1 - r)
```

这正是当前实现使用的 replay rollout 公式。

### 8.2 replay loss

记：

- `L_on`: on-policy sampled rollouts 的 `SubTB`
- `L_rep`: replay trajectories 的 `SubTB`
- `N_on`: on-policy 轨迹数
- `N_rep`: replay 轨迹数

当前实现按轨迹条数加权平均：

```text
L_total = (N_on L_on + N_rep L_rep) / (N_on + N_rep)
```

这样 replay 不会因为开启与否改变目标的整体量纲。

## 9. 总训练目标

当前训练目标只剩两部分：

1. on-policy sampled rollouts 的 `SubTB`
2. 可选的 success replay `SubTB`

如果 replay 关闭或当前 batch 没有可用 replay plan，则 `N_rep = 0`，退化为：

```text
L_total = L_on
```

也就是说，当前训练并不再依赖一个额外的全局 exact path objective；credit assignment
全部发生在 sampled trajectory 的前向一致性残差上。

## 10. 评估阶段的 flow-frontier analysis

当前默认 answer-reachability 评估不再用 Monte Carlo rollout 估计 posterior，而是直接沿
learned flow 做 deterministic frontier expansion。

对每张图，先由起点流得到：

```text
Z_theta(x) = sum_{q in Q(x)} F(x_0(q))
log Z_theta(x) = logsumexp_{q in Q(x)} f(x_0(q))
```

然后从所有起点 prefix 建立初始 frontier，并对每个保留下来的 state `x_t` 同时维护：

- 真实 prefix probability
- state log-flow `f(x_t)`

其中 prefix probability 由 tree policy 直接给出：

```text
P_theta(x_t | x)
= P_F^start(v_0 | x) * prod_{i=1}^t P_F(a_i | x_{i-1})
```

对任意候选 child state，当前实现使用 normalized state flow

```text
U(x_t) = F(x_t) / Z_theta(x) = exp(f(x_t) - log Z_theta(x))
```

作为 descendant terminal mass 的可剪枝上界；如果：

```text
U(x_t) < epsilon_flow
```

则该 state 会被 flow-admissible pruning 丢弃，并把对应质量累计到
`remaining_mass_upper`。

保留下来的 terminal trajectory 直接贡献 terminal / answer posterior：

```text
P_ret(u | x) = sum_{tau: term(tau)=u, retained} P_theta(tau | x)
P_ret(a | x) = sum_{u: entity(u)=a} P_ret(u | x)
```

如果 frontier 被完全穷尽且没有额外 prune/budget overflow，则这些量就是当前 learned
policy 下的精确值；否则它们表示 retained support 上的精确质量，并由
`remaining_mass_upper` 给出遗漏尾部的保守上界。

## 11. support search 的语义

当前 `FlowFrontierSupportSearch`：

- 直接复用 deterministic frontier search 得到的 discovered trajectories；
- 用 exact path probability 而不是 sampled frequency 组装 answer posterior；
- 再用 answer mass threshold 和 support mass threshold 选择要发出的 support window。

窗口结果里的 `remaining_mass_upper` 由两部分组成：

```text
remaining_mass_upper
= search_omitted_mass_upper
+ uncovered_discovered_mass
```

其中前者来自被 prune 或因 budget 未展开的 frontier 质量，后者来自已发现但未 emit 到
最终 support window 的质量。因此在 exhaustive 无剪枝时，窗口级
`remaining_mass_upper = 1 - covered_mass`。

显式切回 `support_search_method=monte_carlo` 时，旧的 Monte Carlo analysis/search
仍然可用；当前 edge retrieval 任务也固定使用这条 legacy 路径。

## 12. 一句话总结

当前主线可以概括为：

```text
用 exact discrete prefix 定义状态，
用 recurrent control state 压缩前缀历史，
用 state-flow head 估计 log F，
用 control-state actor 估计前向动作，
并用前向子轨迹一致性 loss 训练。
```

如果需要看更偏工程视角的流程图，请结合阅读：

- `docs/answer_reachability_algorithm.md`
- `docs/gflownet_architecture.md`
