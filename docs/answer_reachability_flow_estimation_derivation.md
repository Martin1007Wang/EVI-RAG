# Answer Reachability Flow Estimation Derivation

本文复盘当前主线在 answer reachability 任务上到底是如何估计流量的，以及这些量
如何进入起点分布、转移分布、SubTB 约束和最终训练。

这份文档只讨论当前实现，也就是已经切换到“隐式虚拟源”之后的版本。相关代码主线：

- `src/models/components/scoring.py`
- `src/models/gflownet/policy.py`
- `src/models/gflownet/sampler.py`
- `src/models/gflownet/replay.py`
- `src/models/gflownet/losses.py`

相关背景文档：

- `docs/answer_reachability_algorithm.md`
- `docs/answer_reachability_math_derivation.md`
- `docs/answer_reachability_problem_diagnosis.md`

## 1. 我们到底在估计什么

给定问题 `x`、图 `G_x = (V_x, E_x)`、问题实体集合 `Q(x)`、答案实体集合 `A(x)`、
最大步数 `T`，当前主线估计的是三类量：

1. 根流量 `Z_theta(x) = F_theta(s_emptyset | x)`。
2. 起点状态流量，也就是 `Q(x)` 上每个起点状态 `(q, 0)` 的流量。
3. 非根 node-time 状态 `(v, t)` 的流量。

注意这里的状态不是单纯节点，而是：

```text
s_t = (v_t, t)
```

因此我们真正建模的是 node-time flow，而不是静态 node flow。

## 2. 编码器和单一状态流头

当前实现先用 backbone 把图和问题编码成：

- 节点表示 `z_v`
- 问题表示 `z_x`

然后统一用一个状态流头来估计 node-time state 的 bare log-flow。

### 2.1 统一状态特征

对于任意状态 `s = (v, t)`，当前先构造状态特征：

```text
phi_theta(v, t, x)
= LayerNorm(z_v + step_embed(t) + remaining_embed(T - t))
```

### 2.2 统一状态流量函数

状态流量头输出：

```text
f_theta(v, t, x)
```

它对应的是“状态 `s=(v,t)` 的 bare log-flow 估计”，实现上统一来自 `NodeFlowHead`。

因此对于起点状态和后续状态，当前都使用同一个流函数：

```text
起点状态流量      -> f_theta(q, 0, x)
中间状态流量      -> f_theta(v, t, x), t >= 1
根流量            -> 由起点状态流量做 logsumexp 得到
```

### 2.3 为什么要统一成单一状态流头

这一步重构的核心目的，是去掉过去在 `t = 0` 和 `t = 1` 之间的人为参数鸿沟。

在当前任务里：

- 起点实体 `q`
- 第一跳到达实体 `v_1`

都属于同一个知识图谱节点空间。

而当前状态已经显式带有时间 `t`，因此如果我们希望存在一个较平滑的 node-time flow
function，那么起点状态最自然的写法就是：

```text
f_theta(q, 0, x)
```

而不是额外引入一个专门的起点头。

统一之后，SubTB 在 `t=0 -> t=1` 段约束的是：

```text
同一个状态流函数在不同 node/time 截面上的连续关系
```

而不是：

```text
两个不同网络的数值对齐问题
```

### 2.4 起点状态如何落到同一流函数上

对于起点候选 `q in Q(x)`，当前不再走专门的起点头，而是直接构造标准状态特征：

```text
phi_theta(q, 0, x)
= LayerNorm(z_q + step_embed(0) + remaining_embed(T))
```

再统一定义：

```text
a_theta(q, x) := f_theta(q, 0, x)
```

因此起点特殊性仍然存在，但它来自 `t = 0` 这个状态本身，而不是来自一个独立参数孤岛。

## 3. 隐式虚拟源：根流量的严格定义

当前多起点问题不再被处理成“先学一个 graph-level `log Z`，再学一个起点 softmax”，
而是显式采用一个隐式虚拟源：

```text
s_emptyset
```

所有合法起点状态组成：

```text
S_start(x) = {(q, 0): q in Q(x)}
```

如果起点 heuristic 打开，记起点 bias 为：

```text
b_start(q, x) = beta * log h(q, x)
```

那么当前真正进入起点归一化的是：

```text
V_start(q, x) = a_theta(q, x) + b_start(q, x)
```

这里的 `V_start` 是当前实现中的“有效起点 log-flow”。

### 3.1 根流量

根流量直接定义为所有起点状态流量之和：

```text
Z_theta(x)
= F_theta(s_emptyset | x)
= sum_{q in Q(x)} exp(V_start(q, x))
```

因此：

```text
log Z_theta(x) = logsumexp_{q in Q(x)} V_start(q, x)
```

这正是当前 `build_start_distribution_from_log_flows(...)` 在分 graph 做的事情。

如果采用单一状态流头，则只需要把这里的 `V_start` 改写成：

```text
V_start(q, x) = f_theta(q, 0, x) + b_start(q, x)
```

根流量定义、起点 softmax 和隐式虚拟源框架本身都不需要改变。

### 3.2 起点分布

起点分布由同一组 `V_start` 直接归一化得到：

```text
P_F((q, 0) | s_emptyset, x)
= exp(V_start(q, x) - log Z_theta(x))
```

因此对任意被选中的起点 `q`，都有严格恒等式：

```text
V_start(q, x) - log P_F((q, 0) | s_emptyset, x) = log Z_theta(x)
```

这就是隐式虚拟源版本最关键的结构收益：

```text
根边界与起点边界是同一组量，不再是两个松耦合头。
```

## 4. 转移分布是怎么得到的

对于当前状态 `s_t = (v_t, t)`，记合法后继集合为 `Child(s_t)`。

对任意 child state `c = (u, t + 1)`，当前先定义 bare state log-flow：

```text
f_theta(c, x)
```

如果 heuristic 打开，再加上状态势函数式的 heuristic bias：

```text
b_trans(c, x) = beta * log h(c, x)
```

于是先定义有效状态值：

```text
V_theta(c, x) = f_theta(c, x) + b_trans(c, x)
```

当前 backward policy 固定为 `P_B(s_t | c, x)`，因此前向策略真正 softmax 的 edge
logit 是：

```text
u_theta(s_t -> c, x)
= V_theta(c, x) + log P_B(s_t | c, x)
```

最终 move 分布为：

```text
P_F(c | s_t, x)
= exp(u_theta(s_t -> c, x))
 / sum_{c' in Child(s_t)} exp(u_theta(s_t -> c', x))
```

### 4.1 一个必须说清楚的实现事实

当前系统中：

- 起点分布使用 `V_start(q, x)`
- move logits 使用 `V_theta(c, x) + log P_B(s_t | c, x)`
- SubTB 的状态锚点也使用同一个有效状态值 `V_theta(s, x)`

因此 heuristic 不再是游离在流匹配之外的 proposal bias，而是已经进入了有效流函数的
定义。

这意味着当前需要学习的一致性关系已经变成：

```text
V_theta(s_t, x) + log P_F(s_{t+1} | s_t, x)
= V_theta(s_{t+1}, x) + log P_B(s_t | s_{t+1}, x)
```

至少在参数化层面，前向策略、状态锚点和 backward 项现在已经是同一套方程里的量。

## 5. backward 分布如何定义

当前 backward policy 不是 learned backward，而是固定的 uniform backward：

```text
P_B(s_t | s_{t+1}, x)
= 1 / indegree(v_{t+1})
```

对应 log-prob：

```text
log P_B(s_t | s_{t+1}, x) = -log indegree(v_{t+1})
```

它只取决于 child 节点的入度，不取决于语义。

## 6. 一条轨迹上的流量锚点如何构造

考虑一条 rollout：

```text
tau = (s_0, s_1, ..., s_L)
```

其中：

- `s_0 = (q, 0)`，且 `q in Q(x)`
- `s_L` 是终止状态

先定义 forward prefix：

```text
G_0 = log P_F(s_0 | s_emptyset, x)
G_t = G_0 + sum_{i=0}^{t-1} log P_F(s_{i+1} | s_i, x),    1 <= t <= L
```

再定义 backward prefix：

```text
B_0 = 0
B_t = sum_{i=1}^{t} log P_B(s_{i-1} | s_i, x),            1 <= t <= L
```

当前实现构造四类锚点：

### 6.1 根锚点

```text
Y_root = log Z_theta(x)
```

### 6.2 起点锚点

```text
Y_start = V_start(s_0, x) - G_0
```

由于 `G_0 = log P_F(s_0 | s_emptyset, x)` 且起点分布由 `V_start` 直接归一化，因此：

```text
Y_start = log Z_theta(x) = Y_root
```

也就是说，起点锚点和根锚点在当前实现里是严格相等的。

同时，这里的 `V_start(s_0, x)` 可以写成：

```text
V_start(s_0, x) = f_theta(s_0, x) + b_start(s_0, x)
```

因此起点锚点和后续状态锚点共享同一个底层状态流函数，只是起点额外经过了起点 softmax
归一化。

### 6.3 中间状态锚点

对于 `1 <= t < L`：

```text
Y_t = V_theta(s_t, x) - G_t + B_t
```

这里使用的是与前向 move logits 一致的有效状态值 `V_theta`。

### 6.4 终止锚点

如果终止 reward 为 `R(tau, x)`，则：

```text
Y_term = log R(tau, x) - G_L + B_L
```

## 7. 为什么这些锚点应该相等

如果存在理想流函数，使得下面三类关系全部成立，那么这些锚点应该全部相等。

### 7.1 根-起点一致性

由隐式虚拟源定义直接得到：

```text
Y_root = Y_start = log Z_theta(x)
```

### 7.2 边级流量平衡

如果对任意合法边 `s_t -> s_{t+1}` 有：

```text
V_theta(s_t, x) + log P_F(s_{t+1} | s_t, x)
= V_theta(s_{t+1}, x) + log P_B(s_t | s_{t+1}, x)
```

那么：

```text
Y_{t+1} - Y_t
= V_theta(s_{t+1}, x) - V_theta(s_t, x)
 - log P_F(s_{t+1} | s_t, x)
 + log P_B(s_t | s_{t+1}, x)
= 0
```

因此所有中间状态锚点应当与起点锚点相等。

### 7.3 终止一致性

如果终止状态满足：

```text
f_theta(s_L, x) = log R(tau, x)
```

那么终止锚点也与前面的状态锚点一致。

综上，理想情况下应有：

```text
Y_root = Y_start = Y_1 = ... = Y_{L-1} = Y_term
```

## 8. 当前 SubTB 到底在最小化什么

当前实现不是只比较相邻锚点，而是比较同一条轨迹上所有有效位置的两两差值。

设有效锚点序列为：

```text
Y_0, Y_1, ..., Y_M
```

其中：

- `Y_0` 是根锚点
- `Y_1` 是起点锚点
- 后续位置对应中间状态锚点
- 最后一个有效位置被终止锚点覆盖

则当前 SubTB 近似写成：

```text
L_subtb(tau)
= [sum_{0 <= i < j <= M} w_{ij} (Y_i - Y_j)^2]
 / [sum_{0 <= i < j <= M} w_{ij}]
```

其中时间权重为：

```text
w_{ij} = lambda_weight^(j - i - 1)
```

`lambda_weight = 1` 时就是简单平均；否则更偏向短子轨迹。

### 8.1 这意味着什么

当前训练并不是在显式解一套闭式流方程，而是在做：

```text
让同一条 sampled trajectory 上所有 flow anchors 尽量一致
```

也就是说，当前流量估计是一个：

- 参数化的
- 样本驱动的
- 通过 SubTB 自洽约束校准的

amortized flow estimation 过程。

由于起点和中间状态现在已经共享同一个状态流函数，因此这个 amortized estimation
相对过去更干净：

- 所有状态锚点都来自同一个流函数
- `t=0 -> t=1` 的约束对应同一函数在不同时间截面上的延拓
- 根边界与起点边界也由同一组起点流严格导出

## 9. 当前算法的“精确部分”和“学习部分”

### 9.1 精确部分

以下量在当前实现里是结构上精确的：

1. `log Z_theta(x) = logsumexp(V_start(q, x))`
2. `P_F((q,0)|s_emptyset,x) = softmax(V_start(q, x))`
3. `u_theta(s->c) = V_theta(c, x) + log P_B(s|c,x)`
4. 因而 `Y_root = Y_start` 精确成立

### 9.2 学习部分

以下量是通过神经网络和 SubTB 近似学习得到的：

1. bare state flow `f_theta(v, t, x)`
2. 有效状态值 `V_theta(v, t, x) = f_theta(v, t, x) + b(v, t, x)`
3. 编码器中的图和问题表示

### 9.3 固定设计部分

以下量目前不是 learned：

1. backward policy `P_B`
2. 成功/失败 reward 形式
3. horizon 约束与 dead-end 终止语义

## 10. 这套流量估计算法的当前含义

现在可以把当前主线概括成：

```text
先用统一状态流头估计 bare state flow，
再把 heuristic 作为势函数并入得到有效状态值 `V_theta`，
再通过隐式虚拟源的 logsumexp 严格定义根流量 log Z，
然后用 `V_theta(child) + log P_B(parent|child)` 参数化前向策略，
最后用 SubTB 让根锚点、起点锚点、中间状态锚点和终止锚点在 sampled trajectories 上尽量一致。
```

它的优点是：

- 多起点下 root-flow 参数化现在是严格一致的
- `log Z` 不再需要单独回归一个 graph scalar

但它仍然保留了几个重要限制：

1. backward policy 仍是固定 uniform backward。
2. reward 仍然只表达“成功 vs 失败”，不表达答案排序。
3. 当前仍然是 node-time state，不是 path-aware state。

因此，隐式虚拟源修正解决的是：

```text
多起点集合下根流量参数化不严谨的问题
```

但它并没有自动解决：

```text
路径级精细信用分配
answer ranking 对齐
```

这两类问题仍然需要后续目标函数和状态表达继续推进。
