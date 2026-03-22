# Answer Reachability Flow Estimation Derivation

本文只描述当前主线实现，也就是已经切到“静态图/问题编码 + recurrent control-state
controller”的版本。相关代码主线：

- `src/models/gflownet/policy.py`
- `src/models/gflownet/sampler.py`
- `src/models/gflownet/replay.py`
- `src/models/gflownet/losses.py`
- `src/models/components/scoring.py`

配合阅读：

- `docs/answer_reachability_algorithm.md`
- `docs/answer_reachability_math_derivation.md`
- `docs/answer_reachability_codepath_review.md`

## 1. 当前到底在估计什么

给定问题 `x`、图 `G_x = (V_x, E_x)`、问题实体集合 `Q(x)`、答案实体集合 `A(x)`、最大步数
`H`，当前主线估计的是四类量：

1. 起点 prefix 的状态流 `F(x_0(q))`。
2. 中间 prefix state 的状态流 `F(x_t)`。
3. 前向动作分布 `P_F(a | x_t)`。
4. 非根 prefix 的 backward delta kernel，以及 terminal submit 的 entity-level
   backward 近似。

这里的搜索状态不是 `(node, time)`，而是精确离散前缀：

```text
x_t = (v_0, r_1, v_1, ..., r_t, v_t)
```

其中 `v_0 in Q(x)`，`t <= H`。

为了避免对整段 path 反复做高成本 attention，策略再维护一个连续控制状态：

```text
c_t in R^d
```

`c_t` 是 prefix 的 recurrent summary；`x_t` 则继续通过 `path_token_ids` 精确保留在环境状态
里。这两层状态分别服务于：

- `path_token_ids`: backward / replay / trace / exact prefix dedup
- `control_state`: forward `log F` 与 `P_F` 打分

## 2. 编码器和 recurrent controller

`prepare_batch()` 每个 batch 只运行一次，得到：

- 节点表示 `z_v`
- 关系表示 `z_r`
- 全局问题向量 `q_root`
- 问题 token 序列 `H_Q`

### 2.1 start controller

根控制状态直接取全局问题向量：

```text
c_root = q_root
```

对每个起点 `q in Q(x)`，再做一次 controller 更新：

```text
a_root = Attn(W_q c_root, H_Q)
c_0(q) = LN(GRU([a_root; z_q; z_start], c_root))
```

这里 `z_start` 是 learned start relation feature。

### 2.2 transition controller

对任意 graph move `x_t --r_{t+1}--> x_{t+1}`，controller 递推为：

```text
a_t = Attn(W_q c_t, H_Q)
c_{t+1} = LN(GRU([a_t; z_{v_{t+1}}; z_{r_{t+1}}], c_t))
```

也就是说，controller 只吸收三类增量信息：

- 当前 query 对问题 token 的注意摘要
- 目标节点表示
- 本步关系表示

submit 动作不会再触发 graph move controller 更新。

### 2.3 state feature 与 log-flow

先构造与当前节点和时间有关的静态基底：

```text
b(x_t) = z_{v_t} + step_embed(t) + remaining_embed(H - t)
```

再把它与控制状态拼接，经过小 MLP 形成真正的 state feature：

```text
phi(x_t) = LN(MLP(LN([b(x_t); c_t])))
```

最后由 `NodeFlowHead` 输出：

```text
f(x_t) = log F(x_t)
```

因此当前主线在函数形式上是：

```text
x_t -> c_t -> phi(x_t) -> f(x_t)
```

如果 rollout 时已经显式携带 `control_state`，策略直接复用；如果只给离散
`path_token_ids`，策略会回放 prefix 重建 `c_t`。

## 3. 隐式虚拟源和起点分布

当前多起点问题写成一个隐式虚拟源 `s_root` 指向所有起点 prefix：

```text
x_0(q) = (q), q in Q(x)
```

对每个起点 `q`，先得到 `c_0(q)` 与 `f(x_0(q))`，然后定义：

```text
Z_theta(x) = sum_{q in Q(x)} F(x_0(q))
log Z_theta(x) = logsumexp_{q in Q(x)} f(x_0(q))
```

起点分布直接由起点 flow 归一化得到：

```text
P_F^start(q | x)
= F(x_0(q)) / Z_theta(x)
= exp(f(x_0(q)) - log Z_theta(x))
```

因此恒有：

```text
f(x_0(q)) - log P_F^start(q | x) = log Z_theta(x)
```

注意：当前 sampler 仍然会把 `graph_log_z` 记录到 batch 中，但当前 `SubTB` 实现并不把它
作为一个单独的训练残差项使用；它主要用于日志和诊断。

## 4. 前向动作分布

对当前 prefix `x_t` 的 state feature `phi(x_t)`，以及任意候选 graph move
`e = (v_t -r-> u)`，当前 actor logit 为：

```text
ell_F(e | x_t) = g_theta(phi(x_t), z_u, z_r)
```

这里 actor 只读取：

- 当前 prefix 的 state feature
- 静态候选节点表示 `z_u`
- 关系表示 `z_r`

它不再为每个 candidate 重新构造 child prefix 的完整状态表示，也不再运行 path
self-attention。

当前还存在一个 submit 动作：

```text
ell_submit(x_t) = g_theta(phi(x_t), z_{v_t}, z_submit)
```

最终目标前向分布在“合法 outgoing edges + submit”上归一化：

```text
P_F(a | x_t) = exp(ell(a | x_t)) / sum_{a' in A(x_t)} exp(ell(a' | x_t))
```

### 4.1 behavior policy

如果 heuristic 打开，它只进入 behavior sampling，不进入 target `P_F` / `F`。

记 heuristic bias 为 `h(child(x_t, a))`，权重为 `beta`，则：

```text
Q_behavior(a | x_t)
prop exp(ell(a | x_t) + beta h(child(x_t, a)))
```

实际采样还会再经过温度 `tau`：

```text
Q_sample(a | x_t) = softmax(logits_behavior / tau)
```

## 5. backward 分布

### 5.1 graph move backward

对非根 prefix，当前实现不学习 backward head，而是从 `path_token_ids` 精确恢复唯一合法
parent：

```text
parent(x_t)
```

因此在 prefix-tree 状态空间上：

```text
P_B(parent(x_t) | x_t) = 1
P_B(x' | x_t) = 0, for x' != parent(x_t)
```

实现上会取出离散 prefix 中记录的“上一个节点 + 最后一步 relation”，再在 incoming
edges 中匹配父边。

### 5.2 terminal submit backward

当 reward mode 是 `entity_sink` 时，最后一步 `submit -> sink(y)` 还会带一个单独的
terminal backward 近似，例如：

```text
P_B(parent | sink(y)) = 1 / alias_count_graph(y)
```

对应：

```text
log P_B(parent | sink(y)) = -log alias_count_graph(y)
```

它解决的是“同一实体多个 alias 节点”的一部分归一化问题，但不是严格的 path multiplicity
校正。

## 6. 当前 `SubTB` 实际约束了什么

这里必须和代码保持一致。

当前 `SubTrajectoryBalanceLoss.compute()` 读取的主要量是：

- `start_state_log_f`
- `next_state_log_f_steps`
- `log_pf_steps`
- `terminal_log_rewards`

虽然 sample batch 里也携带 `log_pb_steps`，但当前 loss 实现并没有把它真正写进残差公式。

### 6.1 前缀量

设一条 rollout 的状态序列为：

```text
tau = (x_0, x_1, ..., x_L)
```

定义前向前缀和：

```text
G_k^F = sum_{i < k} log P_F(x_{i+1} | x_i)
```

定义状态锚点：

```text
A_k = log F(x_k) - G_k^F
```

终止锚点为：

```text
A_term = log R(tau) - G_L^F
```

### 6.2 当前 loss 的 pairwise residual

对任意 `i < j < L`，代码实际构造的是：

```text
Delta_{i,j} = A_i - A_j
```

也就是：

```text
Delta_{i,j}
= log F(x_i)
 + sum_{k=i}^{j-1} log P_F(x_{k+1} | x_k)
 - log F(x_j)
```

### 6.3 terminal residual

对任意中间状态 `x_i` 到 terminal 的残差：

```text
Delta_{i,term} = A_i - A_term
```

即：

```text
Delta_{i,term}
= log F(x_i)
 + sum_{k=i}^{L-1} log P_F(x_{k+1} | x_k)
 - log R(tau)
```

### 6.4 加权 `SubTB`

如果 `lambda_weight = lambda`，当前实现对更长的子段做指数衰减：

```text
w_{i,j} = lambda^(j - i - 1)
```

然后对 state-state residual 和 state-terminal residual 的平方求和，再做归一化：

```text
L_subtb(tau)
= WeightedMean({Delta_{i,j}^2, Delta_{i,term}^2})
```

批量 loss 就是所有 sampled rollout 的平均。

因此当前实现更准确的一句话是：

```text
用 prefix state log-flow、forward log-prob 和 terminal log-reward
做前向子轨迹一致性约束。
```

## 7. replay 和评估

### 7.1 success replay

replay buffer 只缓存成功轨迹的离散骨架：

```text
(sample_id, start_node, edge_path)
```

重放时，当前参数会重新：

- 构造 exact `path_token_ids`
- 重建对应 `control_state`
- teacher-force 每一步 forward / next-state log-flow / terminal reward
- 再算同一个 `SubTB`

所以 replay 的本质是“旧成功离散路径在当前参数下重新打分”，而不是回放旧 tensor。

### 7.2 flow-frontier 评估

验证/测试阶段默认不再走 Monte Carlo posterior estimation，而是直接沿 learned flow 做
deterministic frontier expansion：

- `graph_log_z` 由起点 flow 的 `logsumexp` 给出；
- 初始 frontier 包含所有满足 `F(s_0) / Z >= flow_prune_epsilon` 的起点 state；
- child state 继续用 `exp(log F(child) - log Z)` 做 flow-admissible pruning；
- retained terminal trajectories 直接按 exact path probability 累积 answer posterior；
- 被 prune 或未展开的质量写入 `remaining_mass_upper`。

full eval 再在这批 discovered trajectories 上组装 support window；这部分只服务于评估与
artifact 生成，不参与训练期 credit assignment。显式切回
`support_search_method=monte_carlo` 时，旧的 rollout-based evaluator 仍可用；当前 edge
retrieval 也固定走这条 legacy 路径。

## 8. 一句话总结

当前主线可以概括为：

```text
用 exact discrete prefix 定义状态，
用 recurrent control state 压缩前缀历史，
用 state-flow head 估计 log F，
用 control-state actor 估计前向动作，
用 exact parent recovery 处理非根 backward，
并用前向子轨迹一致性 loss 训练。
```
