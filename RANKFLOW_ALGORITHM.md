# RankFlow 方法节草稿：Detailed-Balance Witness GFlowNet with Terminal Answer Sets for KGQA

本文档按当前主线代码重写，目标是给论文方法节提供一份与实现一致的算法说明。它严格对应现在的主线：

1. 终止对象是 witness 子图状态本身，而不是“状态-答案”联合终止对象；
2. 训练目标是单步 Detailed Balance；
3. `terminal answer set` 是终止状态导出的可答实体集合，用于评估、监控和动作建模，但不是一个单独的 stop 子动作标签。

文中的公式保持 LaTeX 友好，可直接整理进论文正文。


## 1. 方法总述

我们将多锚点知识图谱问答（multi-anchor KGQA）建模为一个条件 GFlowNet 上的证据子图构造问题。给定问题 `q`、局部候选图 `G`、锚点集合 `A` 与金答案集合 `Y^*`，模型从锚点出发逐步扩展一张 witness 子图，并在某一步执行唯一的 `stop` 动作终止 rollout。

与旧版叙述不同，当前主线并不在 stop 时再额外“选择一个答案实体”。相反，终止后的 witness 状态 `S_T` 会导出一个 `terminal answer set`：

$$
\mathcal Y_{adm}(S_T),
$$

它表示在当前结构约束下、从所有锚点都可达的实体集合。这个集合随后用于：

- 训练日志中的 answer-side 诊断；
- actor 对 stop readiness / edge usefulness 的建模；
- Monte Carlo 评估时对答案后验 surrogate 的聚合。

因此，当前方法学习的是一个对高质量终止 witness 状态赋流的策略，而答案分布是由终止状态诱导出来的边缘统计量，而不是 stop 动作里显式分类出的标签。


## 2. 问题设定

对每个样本，已知：

- 一个问题 `q`；
- 一个局部候选知识图 `G = (V, E)`；
- 一个已接地的锚点集合

$$
A = \{a_1, a_2, \dots, a_k\};
$$

- 一个金答案实体集合

$$
Y^* \subseteq V.
$$

我们的直接建模对象不是静态答案分数，而是条件于 `(q, G, A)` 的 witness 构造过程。模型输出的最终可观测量包括：

- 终止 witness 子图 `S_T`；
- 由该状态导出的 `terminal answer set`

$$
\mathcal Y_{adm}(S_T) \subseteq V;
$$

- 一个由终止 reward 与 Monte Carlo rollout 共同诱导的答案 posterior surrogate。

因此，文中的答案分布应解释为：

$$
Q_R(y \mid q, G, A),
$$

其中下标 `R` 强调它是 reward-induced / rollout-induced surrogate，而不宣称等于真实 Bayesian posterior。


## 3. 状态：语义子图的轻量句柄表示

### 3.1 动态状态

当前实现将 rollout 的动态状态记为已选边集合：

$$
S_t = E_t^{sel} \subseteq E.
$$

代码中，`SubgraphState` 只存储 `edge_ids`。这只是实现层面的轻量句柄，不是语义上的降级定义。

### 3.2 语义状态的恢复

给定静态图上下文 `(G, A)` 和动态 `edge_ids`，完整 witness 语义可由以下节点恢复：

- 所有 anchor 节点；
- 所有被选边的源节点；
- 所有被选边的目标节点。

于是，算法层面的 Markov state 仍然是一张语义证据子图，只是在实现上被分解为：

- 静态上下文：`SubgraphBatch` / `GraphSubgraphContext`；
- 动态句柄：`SubgraphState.edge_ids`。

这种分解允许我们在不复制整张图的前提下，持续恢复：

- 状态节点集合；
- anchor reachability；
- 连通分量；
- 可答实体集合；
- 终止奖励相关统计量。


## 4. Reachability 与 Terminal Answer Set

### 4.1 多锚点 bitmask 传播

为了判断一个实体是否同时由所有锚点支持，我们给第 `i` 个锚点分配一个 bit：

$$
b(a_i) = 2^{i-1}.
$$

这些 bit 沿当前 witness 中的有向边传播；若多个状态节点映射到同一实体，则在实体层面做按位或聚合。记完整 anchor mask 为：

$$
M = 2^k - 1.
$$

若某实体 `y` 的实体级 reachability bit 满足

$$
b_{entity}(y) = M,
$$

则说明该实体已经在当前 witness 中吸收了所有锚点的信息。

### 4.2 Terminal Answer Set

基于上述条件，我们定义终止状态 `S` 的可答实体集合：

$$
\mathcal Y_{adm}(S) = \{y : b_{entity}(y) = M\}.
$$

这里的 `adm` 表示 `admissible`，强调这是一种结构上可答的候选集合，而不是真值 oracle。它依赖于：

- 当前 witness 的有向结构；
- 多锚点 reachability 传播；
- 实体级 merge；
- 局部候选图本身的质量。

因此，`terminal answer set` 是算法诱导出来的结构性可答集合，不应被误写为“模型在 stop 时选择出的最终答案标签”。

### 4.3 与 Gold 命中的区别

当前主线同时区分三个互相关联但不等价的量：

1. `terminal answer set`：`\mathcal Y_{adm}(S)`；
2. `gold answer in graph`：终止状态节点中是否已经包含金答案实体；
3. `frontier hit`：当前状态尚未包含金答案，但 frontier 上一步可达金答案。

这三者的区别很重要：一个状态可以已经包含金答案实体，但仍未形成完整的 `terminal answer set`；也可以尚未命中金答案，但 frontier 已经暴露出正确扩展方向。


## 5. 动作空间：`node -> relation -> edge -> stop`

### 5.1 继续扩展分支

若模型选择继续扩展，则动作被严格分解为三层：

1. 选择当前 witness 中的源节点 `v`；
2. 选择该节点上的关系类型 `r`；
3. 选择一条具体边 `e = (v, r, u)`。

因此 continue 分支的前向策略分解为：

$$
P_F(e \mid S, q, \text{continue})
= P_F(v \mid S, q)
  P_F(r \mid S, q, v)
  P_F(e \mid S, q, v, r).
$$

这部分仍然是当前系统的主干，并由 `action_surface.py` 与 `actor.py` 共同实现。

### 5.2 唯一的 Stop 动作

当前主线的 stop 分支不再枚举多个答案 sink。它只有一个 stop 动作：

$$
a = \text{stop}.
$$

因此完整前向策略可写为：

$$
P_F(a \mid S, q) =
\begin{cases}
P_F(g = \text{stop} \mid S, q), & a = \text{stop}, \\
P_F(g = \text{continue} \mid S, q)
P_F(v \mid S, q)
P_F(r \mid S, q, v)
P_F(e \mid S, q, v, r), & a = \text{add}(e).
\end{cases}
$$

换句话说，当前系统把“何时终止”作为一个策略决策，但不把“终止后选哪个答案”建模成额外动作。

### 5.3 Answer-Set Statistics 在动作中的作用

虽然 stop 不是 answer-specific 分类，`terminal answer set` 仍然会进入动作建模：

- `current_answer_candidate_count` 用于 stop readiness 特征；
- 每条候选边还会估计一步扩展后是否可能形成 answer-ready entity，用于 edge scoring；
- 可选的 question-similarity pruning / oracle-distance 先验也可参与候选排序。

因此，答案语义并没有被移除，而是从“stop 分类标签”改成了“状态与动作价值的结构性信号”。


## 6. 状态转移与 Backward Policy

### 6.1 Forward Transition

- 若执行 `add_edge(e)`，则新状态为

$$
S' = S \cup \{e\};
$$

- 若执行 `stop`，则 rollout 在当前状态终止。

代码中的 `SubgraphAction` 也只保留这两类动作：`add_edge` 与 `stop`。

### 6.2 Backward Policy

由于同一终止状态可能由不同加边顺序到达，必须明确 backward policy 才能定义闭环 Detailed Balance。当前主线采用固定 backward policy：

- 只允许删除 `forward-valid removable edges`；
- 在这些可删边上使用均匀分布。

记状态 `S` 上所有 forward-valid removable edges 的集合为 `\mathcal R(S)`，则：

$$
P_B(S \setminus \{e\} \mid S)
= \frac{1}{|\mathcal R(S)|}, \qquad e \in \mathcal R(S).
$$

这一定义保证：

- backward 边只在语义合法的父状态上定义；
- 多条构造顺序通向同一 witness 时，Detailed Balance 仍有明确参照分布。


## 7. 终止奖励：定义在状态 `S_T` 上，而不是 `(S_T, y)` 上

这是当前主线与早期原型叙述的最大区别之一。

### 7.1 Reward 目标

当前终止奖励是一个纯状态函数：

$$
R(S_T).
$$

它不再依赖某个被显式选中的答案实体 `y`。终止后的 answer set 会被记录下来，但不是 reward 的索引参数。

### 7.2 Utility 项

设：

- `hit(S)`：当前终止状态是否已经包含至少一个金答案实体；
- `frontier(S)`：当前尚未命中金答案，但 frontier 是否一步可达金答案；
- `cov(S)`：anchor coverage ratio。

则当前代码中的 utility 部分为：

$$
U(S)
= \alpha_{hit} \cdot \mathbf 1[hit(S)]
+ \alpha_{frontier} \cdot \mathbf 1[frontier(S) \land \neg hit(S)]
+ \alpha_{cov} \cdot cov(S).
$$

默认超参数对应：

- `hit_bonus = 5.0`
- `frontier_bonus = 1.0`
- `coverage_bonus = 0.2`

### 7.3 结构惩罚

当前主线使用两个结构惩罚项：

$$
\Psi(S)
= \beta_{size} |E(S)|
+ \beta_{comp} (K(S) - 1)_+,
$$

其中：

- `|E(S)|` 是已选边数；
- `K(S)` 是 anchor 相关连通分量数。

默认超参数对应：

- `size_penalty = 0.1`
- `component_penalty = 0.5`

### 7.4 最终对数奖励

因此，当前终止对数奖励为：

$$
\log R(S_T) = U(S_T) - \Psi(S_T).
$$

### 7.5 冗余边数的角色

代码中仍会计算 `redundancy_edges`，但它当前是诊断量，不直接进入 reward。换句话说：

- `redundancy_edges` 会被记录；
- 但旧版那种单独的 redundancy penalty 已不在主线 reward 中。


## 8. 单步 Detailed Balance 训练目标

### 8.1 非终止动作

对一条非终止扩展动作 `a_t`，当前主线约束：

$$
F_\theta(S_t)
+ \log P_F(a_t \mid S_t)
\approx
F_\theta(S_{t+1})
+ \log P_B(S_t \mid S_{t+1}).
$$

### 8.2 终止动作

对 stop 动作，Detailed Balance 约束变为：

$$
F_\theta(S_t)
+ \log P_F(\text{stop} \mid S_t)
\approx
\log R(S_t).
$$

这正是当前 `losses.py` 中 terminal residual 的实现逻辑。

### 8.3 损失形式

设每个有效 action step 上的 DB residual 为 `\delta_t`，则当前主线损失就是对所有有效 step 做均方：

$$
\mathcal L_{DB}
= \frac{1}{|\mathcal T_{valid}|}
\sum_{t \in \mathcal T_{valid}} \delta_t^2.
$$

系统同时记录：

- `db_loss`
- `residual_abs`
- `residual_variance`
- `root_abs`
- `log_z_mean`
- `log_z_variance`

但这些都是诊断量；主目标本身就是单步 Detailed Balance。

### 8.4 明确不是旧的轨迹级平衡目标

当前主线已经不再使用旧的轨迹级平衡目标：

- 没有旧的 trajectory-level 主损失；
- 没有旧的 trajectory-balance 配置支线；
- 没有把旧的轨迹分段平衡目标作为论文主张的一部分。

因此，论文叙述必须避免再把当前系统描述成早期那套“联合终止对象 + 轨迹级平衡”算法。那已经不是现在的主线了。


## 9. 辅助训练信号：可选，但不改变主目标

当前代码还支持若干辅助机制，它们服务于优化稳定性和样本效率，但不改变主线数学对象：

1. `reference-sequence imitation`
   - 对 reference sequence bank 里的代表性边序列做前缀 imitation；
2. `success action supervision`
   - 在 reference success subgraph 上对候选边做 BCE 风格的“成功动作”监督；
3. `success replay / reference-path replay`
   - 将命中 rollout 或 shortest-path reference path 重新做 forced replay；
4. `expand imitation`
   - 对 replay 轨迹中的扩展动作做附加 imitation 加权。

这些辅助项只是在 DB 主损失之外叠加的 regularization / curriculum 信号。论文中若要介绍它们，应明确写成 “auxiliary supervision” 而不是重新定义主训练目标。


## 10. 预测：从终止状态到答案 posterior surrogate

### 10.1 Monte Carlo 终止状态采样

评估时，系统执行 Monte Carlo rollout，得到一组终止状态：

$$
S_T^{(1)}, S_T^{(2)}, \dots, S_T^{(N)}.
$$

对每个终止状态，提取：

- 终止 witness 子图；
- `terminal answer set = \mathcal Y_{adm}(S_T)`；
- 终止 reward；
- 轨迹前向概率 / 终止 flow 等统计量。

### 10.2 由 Answer Set 向答案边缘分配质量

若某个 rollout 的终止 answer set 为 `\mathcal Y_{adm}(S_T)`，其 support weight 为 `w(S_T)`，则系统把该 rollout 的质量平均分配给集合中的每个实体：

$$
w_y(S_T) =
\begin{cases}
\frac{w(S_T)}{|\mathcal Y_{adm}(S_T)|}, & y \in \mathcal Y_{adm}(S_T), \\
0, & \text{otherwise.}
\end{cases}
$$

然后在所有 rollout 上累加：

$$
score(y) = \sum_{i=1}^{N} w_y\bigl(S_T^{(i)}\bigr).
$$

这一步非常关键：当前答案 posterior surrogate 是由终止状态导出的 answer set 统计出来的，而不是 stop 时一次性分类出来的。

### 10.3 支持的聚合后端

当前 Monte Carlo runtime 支持五种 support weighting backend：

- `vote`
- `terminal_reward`
- `trajectory_prob`
- `terminal_flow`
- `hybrid`

其中：

- `vote`：每条终止 rollout 的 support weight 都为 1；
- `terminal_reward`：按终止 reward 指数权重；
- `trajectory_prob`：按整条轨迹的前向概率权重；
- `terminal_flow`：按终止动作所在状态流权重；
- `hybrid`：组合 trajectory probability 与 terminal reward。

### 10.4 Early Stop 与 Support Selection

当前评估还支持两个与效率/可解释性相关的机制：

1. `early_stop`
   - 只在 `vote` backend 下启用；
   - 根据 top-k 稳定性边界决定是否提前停止 rollout。
2. `support selection`
   - 从高分终止 witness 中选出 support 子图；
   - 使用 `support_mass_threshold` 与 path overlap penalty 控制输出集。

因此，推理阶段返回的不仅是答案排序，还包括一组经质量筛选的 witness supports。


## 11. 当前方法的准确 claim

为了避免再次把系统说成一个并不存在的模型，这里明确当前主线真正声称的内容。

### 11.1 我们可以声称什么

- 这是一个合法的层次化 witness-construction GFlowNet；
- 它使用单步 Detailed Balance 约束 state flow、forward policy、backward policy 与 terminal reward；
- 它通过终止状态导出的 `terminal answer set` 构造答案 posterior surrogate；
- 它把“答案可答性”作为状态语义与评估边缘，而不是 stop 分类标签。

### 11.2 我们不应声称什么

- 不应说当前系统在 stop 时显式选择一个答案实体；
- 不应说主线 reward 是定义在“状态-答案”联合对象上；
- 不应说当前训练仍然是旧的轨迹级平衡目标；
- 不应把 Monte Carlo answer ranking 说成真实 Bayesian posterior 恢复；
- 不应把 `terminal answer set` 说成语义 oracle，它只是结构约束下的 admissible set。


## 12. 数学视图与实现视图的对应

当前重构后的代码边界与上面的理论描述是一致的：

- `src/subgraph_gflownet/core/subgraph_batch.py`
  - `SubgraphBatch` 与 `GraphSubgraphContext`；
  - 静态图上下文、锚点、answer entities、reference sequence bank、可选 question similarity / oracle distance；
- `src/subgraph_gflownet/core/state.py` + `src/subgraph_gflownet/core/state_kernel.py`
  - `SubgraphState`、状态恢复、reachability、连通分量、forward-valid removable edges；
- `src/subgraph_gflownet/core/semantic_oracles.py`
  - `terminal answer set`、`gold_answer_in_graph`、frontier hit、coverage 与 terminal reward；
- `src/subgraph_gflownet/core/action_surface.py`
  - 分层 continue 动作候选面与 terminal-answer-set readiness 特征；
- `src/subgraph_gflownet/core/actor.py` + `actor_distribution.py` + `actor_scoring.py` + `actor_types.py`
  - gate / node / relation / edge / stop scoring；
  - 分层动作分布构造与 actor 侧特征打分；
- `src/subgraph_gflownet/core/rollout_engine.py` + `rollout_context.py` + `rollout_actions.py`
  - 统一 rollout、forced replay、terminal statistics 记录；
  - 唯一状态布局、分析缓存、teacher-force / sampling 动作选择；
- `src/subgraph_gflownet/core/losses.py`
  - 单步 Detailed Balance 残差与损失；
- `src/subgraph_gflownet/application/evaluation/answer_search_runtime.py`
  - Monte Carlo answer posterior surrogate 与 support selection。

与运行时数据侧对应的实现边界现在位于：

- `src/data/retrieval/dataset.py` + `datamodule.py` + `collate.py`
  - `TrajectoryBatch` 的运行时构造、Lightning DataModule 与批处理装配；
- `src/graph/batch.py` + `src/graph/batch_utils.py`
  - 运行时 `TrajectoryBatch` 及其张量校验、edge batch/ptr、relation-table 紧致化工具。

因此，论文叙述应尽量沿着这些模块边界来组织，而不要再回退到旧的“联合终止对象 + 轨迹级平衡”语言。


## 13. 一句话版本

> 我们将多锚点 KGQA 建模为一个层次化 witness-construction GFlowNet：模型从锚点出发逐步扩展证据子图，并通过唯一的 stop 动作终止到一个 witness 状态；训练使用单步 Detailed Balance，终止 reward 定义在状态本身上；评估时再从终止状态导出 `terminal answer set`，通过 Monte Carlo rollout 聚合出 reward-induced answer posterior surrogate。
