# GFlowNet 主线训练与评估的数学推导

本文只解释当前主线，也就是：

- 模型：`GFlowNetModule`
- 训练：sampled rollouts + `SubTB`
- 评估：exact answer reachability analysis

它不再讨论已经删除的 `trajectory_policy`、`guidance_cfg` 或多损失混训结构。

## 1. 状态与策略

给定问题 `x` 和样本图 `G_x = (V_x, E_x)`，模型在有限 horizon `T` 上定义 node-time 状态：

```text
s_t = (v_t, t)
```

其中：

- `v_t` 是当前节点
- `t` 是已经执行的 move 数，且 `0 <= t <= T`

问题实体集合记为 `Q(x)`，gold answer 节点集合记为 `A(x)`。

## 2. 起点分布

起点只允许从 `Q(x)` 中采样。

设起点 head 给出的打分为：

```text
u_start(v; x)
```

如果启用了 trajectory heuristic，记它在起点上的 bias 为：

```text
b_start(v; x) = beta * log h(v, x)
```

则最终起点分布为：

```text
P_theta(s_0 = v | x)
= exp(u_start(v; x) + b_start(v; x))
  / sum_{q in Q(x)} exp(u_start(q; x) + b_start(q; x))
```

这里的 `h` 可能来自：

- `topology`
- `embedding`
- `learned`

但它们都只是 bias 源，不改变训练目标定义。

## 3. 前向 move 分布

对任意非终止状态 `s_t`，策略先枚举合法 child state `c in Child(s_t)`。

基础 state score 记为：

```text
f_theta(c) = log F_theta(c)
```

若启用 trajectory heuristic，则 child state 的最终 edge logit 是：

```text
u_theta(s_t -> c)
= f_theta(c) + beta * log h(c, x)
```

于是 move-only 前向分布为：

```text
P_theta(c | s_t, x)
= exp(u_theta(s_t -> c))
  / sum_{c' in Child(s_t)} exp(u_theta(s_t -> c'))
```

代码里对应：

- `GFlowNetPolicy.compute_forward_distribution(...)`
- `TrajectoryPolicy.compute_move_log_probs(...)`

## 4. 根边界量 `log Z(x)`

主线训练还额外学习一个 graph-level 标量：

```text
log Z_theta(x)
```

它不是动作概率的一部分，而是 SubTB 根边界项的一部分。

可以把它理解为：

- rollout 轨迹在根部的归一化锚点
- 与 `log F(s_0)` 和起点概率一起进入子轨迹平衡残差

## 5. 终止与奖励

当前 sampled rollout 使用 absorbing 终止语义。

对于一条轨迹：

```text
tau = (s_0, s_1, ..., s_L)
```

终止条件是三选一：

1. 命中 gold answer
2. 没有合法 move
3. 达到 `max_steps`

reward 记为：

```text
R(tau, x) = 1,                 if terminal node in A(x)
R(tau, x) = epsilon_x,         otherwise
```

其中 `epsilon_x` 要么是常数 `epsilon`，要么是 graph-normalized failure reward。

## 6. SubTB 的当前形式

对每条 rollout，代码会构造一串 value-anchor 量：

```text
X_0 = log Z_theta(x)
X_1 = log F_theta(s_0) - log P_theta(s_0 | x)
X_2 = log F_theta(s_1) - log P_theta(s_0, s_1 | x)
...
X_k = log F_theta(s_{k-1}) - log P_theta(prefix_{k-1} | x)
```

终止位置会被 reward anchor 覆盖成：

```text
X_end = log R(tau, x) - log P_theta(prefix_end | x)
```

当前 `SubTB` 做的事情不是再单独拆 root / move / terminal 三种 loss，而是直接最小化：

```text
L_subtb(tau)
= WeightedMean_k (X_k - X_end)^2
```

其中时间权重由 `lambda_weight` 控制：

```text
w_k = lambda_weight^(distance_to_end)
```

最终 batch loss 为：

```text
L_subtb = Mean_tau L_subtb(tau)
```

代码里对应 `src/models/training/losses.py`。

## 7. 为什么说训练目标只有一个

当前主线里：

- `trajectory heuristic` 只改 logits
- `log Z` 只进入 SubTB 边界项
- reward 只提供终止锚点

但真正被优化的目标只有：

```text
min_theta L_subtb
```

没有第二个 critic objective，也没有 DB / TB / SubTB 的并联加权。

## 8. exact 评估与训练的关系

训练是 sampled SubTB，但验证和预测使用 exact analyzer。

对任意状态，exact analyzer 求解的是最终 hit gold answer 的概率质量：

```text
M_gold(x) = P_theta(eventually hit A(x) | x)
```

它通过 node-time DP 或 exact search 计算 answer posterior、gold mass 和 support window。

因此当前主线是：

- 用 `SubTB` 训练一个 GFlowNet 风格的轨迹模型
- 用 exact answer reachability 解释和评估这个模型

这也是为什么训练与评估共享同一策略参数化，但不是同一个损失函数。

## 9. 三种 trajectory heuristic 的数学差异

三种变体只影响 `log h` 的来源：

### 9.1 topology

```text
log h(v, x) = log graph_propagation_from_question_seeds(v)
```

### 9.2 embedding

```text
log h(v, x) = cosine(node(v), question(x)) / temperature
```

### 9.3 learned

```text
log h(s, x) = logsigmoid(MLP([state_feature(s), question_feature(x)]))
```

当前仓库允许切换这三种形式，但它们都服从同一个训练面：`SubTB`。

## 10. 一句话概括

当前主线可以概括成：

```text
在 question entities 上定义起点分布，
在 node-time 状态空间上定义带 trajectory heuristic bias 的前向 move policy，
对 sampled absorbing rollouts 施加单一 SubTB 约束，
再用 exact answer reachability 分析同一策略的 answer posterior 与 support coverage。
```
