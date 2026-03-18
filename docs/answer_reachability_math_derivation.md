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
`H`，当前实现把搜索过程建模为一个有限 horizon 的 node-time GFlowNet。

状态不是静态节点，而是：

```text
x_t = (v_t, t),  v_t in V,  t in {0, 1, ..., H}
```

其中 `t` 表示已经执行的 move 数。

为了和代码保持一致，下面使用这些记号：

- `F_t(v)`: 状态 `(v, t)` 的非归一化 flow。
- `f_t(v) = log F_t(v)`: 对应的 log-flow。
- `P_F^start(v)`: 从问题起点集合中选择起始节点的目标分布。
- `P_F(e | v, t)`: 在状态 `(v, t)` 选择边 `e = (v -> u)` 的目标前向分布。
- `P_B(v | u)`: 固定的 backward 参考分布。
- `R(tau)`: 轨迹 `tau` 的终止奖励。

当前实现中的 `f_t(v)` 由统一的状态流头给出，而不是单独拆一个 graph-level `log Z`
head 和一个 start-policy head。代码对应 `BaseSearchPolicy.compute_log_state_scores()`。

## 2. 状态流参数化

编码器先产生：

- 节点表示 `z_v`
- 问题表示 `z_q`

然后对任意状态 `(v, t)` 构造状态特征：

```text
phi(v, t)
= LayerNorm(z_v + step_embed(t) + remaining_embed(H - t))
```

统一的状态流头输出：

```text
f_t(v) = log F_t(v)
```

也就是说，当前主线学习的是一个 node-time flow 函数：

```text
(v, t) -> f_t(v)
```

起点状态 `(q, 0)` 与中间状态 `(v, t)` 共享同一套参数化，只是时间索引不同。

## 3. 起点分布与隐式虚拟源

当前实现把多起点问题写成一个隐式虚拟源 `s_root` 指向所有真实起点状态
`(q, 0), q in Q`。

对于每个起点候选 `q in Q`，先计算：

```text
f_0(q) = log F_0(q)
```

然后 graph 的根流量由所有起点流量归一化得到：

```text
Z = sum_{q in Q} F_0(q)
log Z = logsumexp_{q in Q} f_0(q)
```

因此目标起点分布不是额外学习的，而是直接由起点流量归一化得到：

```text
P_F^start(q)
= F_0(q) / Z
= exp(f_0(q) - log Z)
```

这正是 `build_start_distribution_from_log_flows()` 的数学含义。

由此立即得到一个重要恒等式：

```text
f_0(q) - log P_F^start(q) = log Z
```

这个恒等式解释了为什么当前 `SubTB` 根边界和起点边界能天然对齐。

## 4. 固定 backward 与目标前向分布

### 4.1 固定 backward

当前主线没有学习 backward policy，而是对目标节点使用均匀父节点分布：

```text
P_B(v | u) = 1 / indeg(u)
log P_B(v | u) = -log indeg(u)
```

代码对应：

- `src/models/gflownet/policy.py`
- `src/models/gflownet/sampler.py`
- `src/models/gflownet/replay.py`

里的 `_compute_uniform_backward_log_probs()`。

### 4.2 目标前向分布

给定当前状态 `(v, t)`，对每条候选边 `e = (v -> u)`，当前实现先看 child state
`(u, t + 1)` 的 log-flow，再加上 backward 项：

```text
ell_t(e : v -> u) = f_{t+1}(u) + log P_B(v | u)
```

再对所有 outgoing edges 做 softmax：

```text
P_F(e | v, t)
= exp(ell_t(e)) / sum_{e'=(v->u')} exp(ell_t(e'))
```

等价地写成 flow 的形式：

```text
P_F(e : v -> u | v, t)
= F_{t+1}(u) P_B(v | u)
 / sum_{u' : (v -> u') in E} F_{t+1}(u') P_B(v | u')
```

这就是 `GFlowNetPolicy.compute_forward_distribution()` 与
`BaseSearchPolicy.compute_move_log_probs()` 共同实现的目标策略。

## 5. heuristic、behavior policy 与采样温度

当前实现里 heuristic 不进入目标分布 `P_F`，而只进入 behavior policy，也就是训练期的
探索分布。

如果记启发式 bias 为 `h_t(v)`，权重为 `beta`，则 behavior 分布使用：

```text
Q_F^start(q) prop exp(f_0(q) + beta h_0(q))
Q_F(e : v -> u | v, t) prop exp(f_{t+1}(u) + log P_B(v | u) + beta h_{t+1}(u))
```

其中 `h` 的来源可以是：

- `topology`
- `embedding`
- `learned`

见 `SearchHeuristic`。

训练时真正采样边还会再经过温度 `tau`：

```text
Q_sample(e | v, t) = softmax(logits_behavior / tau)
```

因此当前训练链条是：

- 用 behavior distribution 提高探索质量；
- 用 target distribution 重新计算 `log P_F`；
- 用 target 的 flow consistency 做 `SubTB`。

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

所以终止 log-reward 为：

```text
log R(tau)
```

这就是 `TrajectoryGFNSampleBatch.terminal_log_rewards` 的来源。

## 7. SubTB 的推导

### 7.1 一条轨迹上的前缀量

设一条 rollout 为：

```text
tau = (x_0 = root, x_1, x_2, ..., x_T, x_{T+1} = terminal)
```

其中：

- `x_1` 是 sampled start state；
- `x_{k+1}` 对应第 `k` 次 move 之后的状态；
- `x_{T+1}` 是终止锚点。

定义前向和后向前缀和：

```text
G_k^F = sum_{i < k} log P_F(x_{i+1} | x_i)
G_k^B = sum_{i < k} log P_B(x_i | x_{i+1})
```

在代码里：

- `forward_prefix` 对应 `G_k^F`
- `backward_prefix` 对应 `G_k^B`

### 7.2 轨迹锚点

对非终止状态，定义：

```text
A_k = log F(x_k) - G_k^F + G_k^B
```

对终止状态，不再使用 `log F`，而是用 reward 锚点替换：

```text
A_term = log R(tau) - G_term^F + G_term^B
```

这就是代码中的 `anchored_values`。

### 7.3 为什么所有子轨迹都应该对齐

如果 flow 完全满足 trajectory balance，那么对任意一段子轨迹
`x_i -> x_{i+1} -> ... -> x_j`，都应满足：

```text
log F(x_i) + sum_{k=i}^{j-1} log P_F(x_{k+1} | x_k)
= log F(x_j) + sum_{k=i}^{j-1} log P_B(x_k | x_{k+1})
```

移项可得：

```text
A_i = A_j
```

如果 `x_j` 是终止状态，则把右侧的 `log F(x_j)` 替换成 `log R(tau)`，同样得到：

```text
A_i = A_term
```

因此理想情况下，整条轨迹上所有锚点应该彼此相等。

### 7.4 当前实现的 SubTB

代码没有只约束 root-to-terminal，而是对所有合法子轨迹做二次残差：

```text
Delta_{i,j} = A_i - A_j
```

并对每一对 `i < j` 施加权重：

```text
w_{i,j} = lambda_weight^(j - i - 1)
```

于是单条 rollout 的 `SubTB` 为：

```text
L_subtb(tau)
= WeightedMean_{i < j} [w_{i,j} (A_i - A_j)^2]
```

如果 `normalize=True`，就除以权重总和；否则直接求加权和。

最后 batch loss 为：

```text
L_on = mean_{tau in sampled rollouts} L_subtb(tau)
```

这就是 `SubTrajectoryBalanceLoss.compute()` 返回的 `loss_output.loss`。

## 8. 精确 DP：成功质量的推导

训练主损失来自 sampled rollouts，但当前主线还额外计算一个精确的 log-space DP。

### 8.1 suffix success function

定义：

```text
S_t(v) = 从状态 (v, t) 出发，在剩余步数内最终命中答案节点的总概率
```

在 success 分析里，答案节点被视为吸收成功态，所以边界条件为：

```text
S_t(v) = 1, if v in A
```

对非答案节点，递推为：

```text
S_t(v)
= sum_{e=(v->u)} P_F(e | v, t) S_{t+1}(u)
```

在 log-space 中写成：

```text
log S_t(v)
= LSE_{e=(v->u)} [log P_F(e | v, t) + log S_{t+1}(u)]
```

这对应 `log_success_by_step` 的反向 DP。

### 8.2 起点成功质量

起点分布质量记为：

```text
m_start(v) = P_F^start(v)
```

于是 graph 的总成功质量为：

```text
M_gold
= sum_{v in V} m_start(v) S_0(v)
```

对应 log-space：

```text
log M_gold = LSE_v [log m_start(v) + log S_0(v)]
```

当前 batched DP 进一步按 graph 维护：

```text
log M_gold^(g)
= LSE_{v in V_g} [log m_start(v) + log S_0(v)]
```

这就是 `log_gold_mass_by_graph`。

### 8.3 terminal mass 与 edge success mass

DP 还顺便计算：

1. `log_terminal_mass(v)`

```text
到达答案节点 v 并首次作为成功终止状态被吸收的总质量
```

2. `log_edge_success_mass(e)`

```text
所有经过边 e 且最终成功的轨迹总质量
```

第二个量可写成：

```text
M_edge_success(e : v -> u, t)
= alive_t(v) P_F(e | v, t) S_{t+1}(u)
```

再把所有时间步累加到同一条原图边上。

## 9. 精确 DP：retrieval 终止质量

除了 success analysis，当前实现还维护另一套终止质量：

```text
M_ret(v)
```

它表示在“忽略答案吸收，只在 dead-end 或 horizon 终止”的语义下，最终停在节点 `v`
的总质量。

因此：

- `success terminal mass` 是 answer-hit 视角；
- `retrieval terminal mass` 是最终落点视角。

这也是为什么 coverage auxiliary 不是直接用 `M_gold`，而是用 retrieval 终止质量在
gold entities 上的聚合。

## 10. exact auxiliary 的数学形式

当前 exact auxiliary 有两个部分。

### 10.1 success auxiliary

对每个被选中的 graph `g`，使用精确成功质量：

```text
M_gold^(g)
```

定义 success auxiliary：

```text
L_exact_success = -(1 / B') sum_g log M_gold^(g)
```

其中 `B'` 是本轮参与 exact loss 的 graph 数。

### 10.2 coverage auxiliary

设 graph `g` 的 gold entity 集合为 `Y_g`。先把 retrieval terminal mass 从节点聚合到实体：

```text
M_ret^(g)(a)
= sum_{v : entity(v) = a} M_ret(v)
```

然后定义 graph 级 coverage loss：

```text
L_cov^(g) = -(1 / |Y_g|) sum_{a in Y_g} log M_ret^(g)(a)
```

整体 coverage auxiliary 为：

```text
L_exact_coverage = (1 / B') sum_g L_cov^(g)
```

### 10.3 exact auxiliary 总和

当前 exact auxiliary 最终写成：

```text
L_exact = alpha_s L_exact_success + alpha_c L_exact_coverage
```

其中：

- `alpha_s = training.exact_aux.success_weight`
- `alpha_c = training.exact_aux.coverage_weight`

## 11. Success replay 的数学逻辑

当前 replay buffer 不缓存旧参数下的 tensor，而只缓存成功轨迹的离散骨架：

```text
(sample_id, start_local_node, local_edge_ids)
```

这意味着 replay 的不是“旧 policy 的数值”，而是“旧成功路径在当前 policy 下的重新打分”。

### 11.1 replay 比例

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

### 11.2 replay loss

记：

- `L_on`: on-policy sampled rollouts 的 `SubTB`
- `L_rep`: replay trajectories 的 `SubTB`
- `N_on`: on-policy 轨迹数
- `N_rep`: replay 轨迹数

当前实现不是简单相加，而是按轨迹条数加权平均：

```text
L_subtb_total
= (N_on L_on + N_rep L_rep) / (N_on + N_rep)
```

这样 replay 不会因为开启与否改变 `SubTB` 的整体量纲。

## 12. 总训练目标

把主损失、replay 和 exact auxiliary 合起来，当前训练目标可以写成：

```text
L_total
= (N_on L_on + N_rep L_rep) / (N_on + N_rep)
 + alpha_s L_exact_success
 + alpha_c L_exact_coverage
```

如果 replay 关闭或当前 batch 没有可用 replay plan，则 `N_rep = 0`，公式退化为：

```text
L_total = L_on + L_exact
```

## 13. 评估阶段的 best-first support search

精确 DP 还提供了一个 prefix 的上界函数。

若某个搜索前缀 `pi`：

- 当前累计 log 概率为 `log P(pi)`；
- 当前位于节点 `v`；
- 已走 `m` 步；

那么它未来所有成功补全的总概率质量上界为：

```text
U(pi) = log P(pi) + log S_m(v)
```

原因很直接：

```text
P(any successful completion extending pi)
= P(pi) * S_m(v)
```

取 log 就得到上式。

因此 `ExactSupportSearch` 使用：

- prefix 当前概率 `log P(pi)`
- 精确 suffix success `log S_m(v)`

来排序 frontier，这就是 `upper_bound_log_mass` 的数学来源。

## 14. 一句话总结

当前主线可以概括为：

```text
用 behavior-biased 的图搜索采样轨迹，
用 target policy 的 SubTB 约束学习 node-time flow，
再用一个精确 log-space DP 直接优化答案命中质量和答案实体覆盖质量，
并把同一个 DP 结果继续用作评估时的前缀质量上界。
```

如果需要看更偏工程视角的流程图，请结合阅读：

- `docs/answer_reachability_algorithm.md`
- `docs/gflownet_architecture.md`
