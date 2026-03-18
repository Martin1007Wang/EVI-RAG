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
- `P_B(v, t - 1 | u, t)`: 在状态 `(u, t)` 上的目标后向分布。
- `R(tau)`: 轨迹 `tau` 的终止奖励。

当前实现采用共享 encoder + 解耦三头：

- state-flow head：输出 `f_t(v)`
- forward-policy head：输出 `P_F`
- backward-policy head：输出 `P_B`

也就是说，当前主线不再使用 `P_F prop F * P_B` 的诱导式前向策略。

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

## 4. 解耦三头参数化

### 4.1 state-flow head

state-flow head 只负责输出：

```text
f_t(v) = log F_t(v)
```

它不再直接参与 `P_F` 或 `P_B` 的参数化。

### 4.2 forward-policy head

给定当前状态 `(v, t)` 和每条候选边 `e = (v -> u)`，forward head 直接输出未归一化
logit：

```text
ell_F(e : v -> u, t)
```

然后在所有合法 outgoing edges 上归一化：

```text
P_F(e | v, t)
= exp(ell_F(e)) / sum_{e'=(v->u')} exp(ell_F(e'))
```

当前实现里，`ell_F` 来自共享状态特征、候选 child state 特征、relation 特征和问题
特征的联合打分，而不是由 `f_{t+1}(u)` 诱导出来。

### 4.3 backward-policy head

给定当前状态 `(u, t)` 以及其每条合法入边 `e = (v -> u)`，backward head 直接输出：

```text
ell_B(e : v -> u, t)
```

再在所有合法 incoming edges 上归一化：

```text
P_B(v, t - 1 | u, t)
= exp(ell_B(e)) / sum_{e'=(v'->u)} exp(ell_B(e'))
```

因此当前训练目标里的 `log P_B` 也来自独立参数化，而不是固定均匀父分布。

## 5. heuristic、behavior policy 与采样温度

当前实现里 heuristic 不进入目标分布 `P_F` / `P_B`，而只进入 behavior policy，也就是
训练期的探索分布。

如果记启发式 bias 为 `h_t(v)`，权重为 `beta`，则 behavior 分布使用：

```text
Q_F^start(q) prop exp(f_0(q) + beta h_0(q))
Q_F(e : v -> u | v, t) prop exp(ell_F(e : v -> u, t) + beta h_{t+1}(u))
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
- 用 target distribution 重新计算 `log P_F` 与 `log P_B`；
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
tau = (x_0, x_1, ..., x_T)
```

其中：

- `x_0` 是 sampled start state；
- `x_{k+1}` 对应第 `k` 次 move 之后的状态；
- `x_T` 是 rollout 的最后一个实际状态。

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

对终止位置，不再使用 `log F(x_T)`，而是用 reward 锚点替换：

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

代码没有只约束单一 start-to-terminal，而是对所有合法子轨迹做二次残差：

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

当前实现不是简单相加，而是按轨迹条数加权平均：

```text
L_subtb_total
= (N_on L_on + N_rep L_rep) / (N_on + N_rep)
```

这样 replay 不会因为开启与否改变 `SubTB` 的整体量纲。

## 9. 总训练目标

删掉训练期 exact DP auxiliary 之后，当前主线重新回到 GFlowNet 的核心命题：

```text
基于局部流一致性的信用分配
```

因此训练目标只剩下两部分：

1. on-policy sampled rollouts 的 `SubTB`
2. 可选的 successful trajectory replay 加权 `SubTB`

最终训练目标为：

```text
L_total = (N_on L_on + N_rep L_rep) / (N_on + N_rep)
```

如果 replay 关闭或当前 batch 没有可用 replay plan，则 `N_rep = 0`，公式退化为：

```text
L_total = L_on
```

也就是说，当前训练不再通过额外的精确全局路径质量目标来干预优化；所有 credit
assignment 都回到 sampled trajectory 上的局部 flow consistency 约束。

## 10. 评估阶段仍保留 exact analysis

虽然训练已经移除了 exact DP auxiliary，但评估栈仍然保留 exact analysis，用来做：

- exact answer posterior
- gold mass 估计
- support search 的前缀上界

这部分逻辑不再参与训练目标，只服务于验证、测试和 artifact 生成。

精确分析里仍然定义：

```text
S_t(v) = 从状态 (v, t) 出发，在剩余步数内最终命中答案节点的总概率
```

其 log-space 递推为：

```text
log S_t(v)
= LSE_{e=(v->u)} [log P_F(e | v, t) + log S_{t+1}(u)]
```

并由此得到：

```text
log M_gold = LSE_v [log m_start(v) + log S_0(v)]
```

这部分现在应被理解为 evaluation-time analysis，而不是 training-time credit
assignment。

## 11. 评估阶段的 best-first support search

精确 analysis 还提供了一个 prefix 的上界函数。

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

## 12. 一句话总结

当前主线可以概括为：

```text
用 behavior-biased 的前向策略采样轨迹，
用解耦的 log F / log P_F / log P_B 做 SubTB，
让 credit assignment 回到局部流一致性本身，
并仅在评估时使用精确 DP 解释 posterior 与 support-search 上界。
```

如果需要看更偏工程视角的流程图，请结合阅读：

- `docs/answer_reachability_algorithm.md`
- `docs/gflownet_architecture.md`
