# RankFlow 方法节草稿：Answer-Committed Witness GFlowNet for KGQA

本文档按论文方法节的写法重构，用于给当前主线提供更严谨的理论叙述。核心目标有两点：

1. 说明当前算法为什么是一个合法且自洽的 GFlowNet 实例；
2. 主动收紧 claim，避免把 reward-induced posterior 误写成真实后验。

文中的公式均保持 LaTeX 友好，可直接整理进论文正文。


## 1. 方法总述

我们将多锚点知识图谱问答（multi-anchor KGQA）建模为一个带潜在证据结构的答案后验近似问题。给定问题 `q`、局部候选图 `G`、锚点集合 `A` 与金答案集合 `Y^*`，模型并不直接输出一个静态答案分数，而是通过一个条件 GFlowNet 逐步构造潜在 witness 子图，并在终止时显式提交（commit）到一个答案实体。

与标准 graph-object GFlowNet 的关键区别在于，终止对象不再是无标签图结构 `z`，而是联合对象 `(z, y)`，其中 `z` 为 witness 子图，`y` 为 stop 时提交的答案实体。因此，我们学习的不是单纯的 `P(z | q)`，而是一个终止联合分布：

$$
P_T(z, y \mid q, G, A) \propto R(z, y \mid q, G, A).
$$

这样设计的动机是：KGQA 的监督与评测都落在答案实体上，而不是落在子图本身上。answer commitment 使终止流分配直接对齐于答案决策，同时保留 GFlowNet 对多模态 witness 的探索优势。


## 2. 问题定义

对于每个样本，我们假设已知：

- 一个问题 `q`；
- 一个局部候选知识图 `G = (V, E)`；
- 一个已接地的锚点集合
  $$
  A = \{a_1, a_2, \dots, a_k\};
  $$
- 一个金答案集合
  $$
  Y^* \subseteq V.
  $$

我们的建模目标是学习一个答案分布近似：

$$
Q_R(y \mid q, G, A),
$$

其中下标 `R` 强调该分布由终止奖励所诱导，而不是天然等于真实 Bayesian posterior。更具体地，若 `z` 表示潜在 witness 子图，则：

$$
Q_R(y \mid q, G, A)
= \sum_{z \in \mathcal Z(q, G, A)} P_T(z, y \mid q, G, A)
\propto \sum_{z \in \mathcal Z(q, G, A)} R(z, y \mid q, G, A).
$$

因此，本文中的答案分布应解释为 **reward-calibrated posterior surrogate**，而不是“恢复真实答案后验”的严格统计陈述。


## 3. 为什么不能只做终止子图生成

若直接采用 graph-object GFlowNet，把终止对象定义为 witness 子图 `z`，再在推理时从 `z` 后处理得到答案，则会遇到两个问题：

1. 同一终止子图可能同时支持多个 answer-ready 实体；
2. 同一答案往往对应大量近似等价的 witness 变体。

这会把“证据多样性”与“答案概率”混在一起。特别地，若直接做后处理投票，则一个答案可能仅仅因为支持它的 witness 变体更多而获得更大质量。为此，我们引入显式 answer commitment，把原本失控的 post-hoc 映射问题，转化成对联合终止对象 `(z, y)` 的显式边缘化。

需要强调的是，这种设计**缓解了** witness multiplicity 问题，但并未从理论上彻底消除它。答案边缘仍然满足：

$$
Q_R(y \mid q, G, A) \propto \sum_z R(z, y \mid q, G, A),
$$

因此若某个答案对应特别多高分 witness，它仍可能得到更大的边缘质量。本文的贡献不在于“完全解耦 witness diversity and answer probability”，而在于把这一问题从无控制的后处理误差，转化为显式、可分析、可设计的 reward-induced marginalization。


## 4. 状态：潜在 witness 的轻量表示

### 4.1 动态状态

我们将 rollout 的动态状态定义为已选择边集合：

$$
S_t = E_t^{\mathrm{sel}} \subseteq E.
$$

这意味着实现层面只保存 `edge_ids`，而不显式保存完整子图对象。这样做并不是在削弱状态语义，而是在利用知识图谱上下文的静态性来节省内存。

### 4.2 从轻量句柄恢复语义 witness

虽然动态状态只存储边集合，但完整 witness 可由以下三部分恢复：

- 所有 anchor 节点；
- 所有被选边的源节点；
- 所有被选边的目标节点。

因此，算法层面的 Markov state 仍然是一个语义 witness，只是采用“静态图上下文 + 动态边句柄”的分解来实现。该实现同时支持实体级 reachability、连通分量分析以及终止奖励计算。


## 5. 多锚点可达性与 admissible commit set

### 5.1 多锚点 bitmask 传播

为了判断当前状态是否支持某个候选答案，我们对每个 anchor 分配一个 bit：

$$
b(a_i) = 2^{i-1}.
$$

bitmask 沿当前 witness 中的有向边传播；若多个选中节点映射到同一实体，则在实体层面做按位或聚合。定义完整 anchor mask 为：

$$
M = 2^k - 1.
$$

对任意实体 `y`，若其实体级聚合 bitmask 满足

$$
b_{entity}(y) = M,
$$

则说明该实体已经从所有锚点吸收到了信息。

### 5.2 `admissible commit set` 而不是真值答案集合

基于上述条件，我们定义状态 `S` 的可提交答案集合：

$$
\mathcal Y_{adm}(S) = \{y : b_{entity}(y) = M\}.
$$

这里我们故意使用 `adm`（admissible）而不是直接称它为“正确答案集合”。原因在于，这个集合是一个**结构上可提交的候选集合**，而不是语义上绝对正确的答案真值。它依赖于当前图、方向约束、entity-level merge 以及 witness 的结构近似。因此，`\mathcal Y_{adm}(S)` 是一个 task-specific approximation，是算法主动引入的归纳偏置，而不是语义 oracle。

这一点在论文中必须写清楚：模型在 stop 时选择的是 admissible answer sink，而不是保证语义无歧义的真值答案。


## 6. 动作空间与 stop 归一化

### 6.1 继续扩展分支

若模型选择继续扩展，则动作被严格分解为三步：

1. 选择当前 witness 中的源节点 `v`；
2. 选择该节点上的关系类型 `r`；
3. 选择一条具体边 `e = (v, r, u)`。

因此，continue 分支的前向策略可以写为：

$$
P_F(e \mid S, q, \mathrm{continue})
= P_F(v \mid S, q)
  P_F(r \mid S, q, v)
  P_F(e \mid S, q, v, r).
$$

### 6.2 终止提交分支

若模型选择 stop，则 stop 并不是单一动作，而是在如下集合上归一化：

- 一个 failure sink `\bot`；
- 对每个 `y \in \mathcal Y_{adm}(S)` 分配一个 answer-commit sink。

于是，stop 分支实际建模的是：

$$
P_F(y \mid S, q, \mathrm{stop}),
\qquad y \in \{\bot\} \cup \mathcal Y_{adm}(S).
$$

### 6.3 完整前向策略分解

因此，完整前向策略可写为：

$$
P_F(a \mid S, q) =
\begin{cases}
P_F(g=\mathrm{stop} \mid S, q)
P_F(y \mid S, q, g=\mathrm{stop}),
& a = \mathrm{commit}(y), \\
P_F(g=\mathrm{continue} \mid S, q)
P_F(v \mid S, q)
P_F(r \mid S, q, v)
P_F(e \mid S, q, v, r),
& a = \mathrm{add}(e).
\end{cases}
$$

这一定义说明：stop mass 与 continue mass 处于同一个决策体系中，而 failure sink 与 admissible answer sink 又在 stop 分支内部共享同一个 masked softmax 归一化。这样可以避免“两阶段不一致分类器”式的理论歧义。


## 7. Backward policy：多轨迹到同对象时的闭环定义

由于同一个 `(z, y)` 可能由不同加边顺序到达，必须明确 backward policy，否则轨迹概率分配不闭环。当前主线采用固定 backward policy：

- 仅允许删除 **forward-valid removable edges**；
- 在所有这类可删边上采用均匀分布。

记当前状态 `S` 上所有 forward-valid removable edges 的集合为 `\mathcal R(S)`，则：

$$
P_B(S \setminus \{e\} \mid S)
= \frac{1}{|\mathcal R(S)|},
\qquad e \in \mathcal R(S).
$$

这一定义与当前代码实现一致，也保证了在多种边添加顺序通向同一 committed witness 时，trajectory semantics 是明确的。


## 8. 终止奖励

### 8.1 奖励定义在 `(S_T, y)` 上

终止奖励不是定义在单独 witness 上，而是定义在 witness 与 committed answer 的联合对象上：

$$
R(S_T, y).
$$

这意味着：当前方法不是在回答“哪个 witness 更好”，而是在回答“哪个 witness 支持哪个答案更值得被终止流赋质量”。

### 8.2 答案效用项

若 `y` 为金答案，给予正奖励；若 `y` 为 admissible 但非金答案，则给予 wrong-answer penalty；若选择 failure sink，则给予 failure penalty。记答案效用项为：

$$
U(y, Y^*) =
\begin{cases}
+\alpha_{gold}, & y \in Y^*, \\
-\alpha_{wrong}, & y \notin Y^*, y \neq \bot, \\
-\alpha_{fail}, & y = \bot.
\end{cases}
$$

### 8.3 Witness 先验项

为了避免无约束膨胀，我们加入结构正则项：

$$
\Psi(S_T) =
\beta_{size}|E_T|
+ \beta_{red} \cdot \mathrm{red}(S_T)
+ \beta_{comp} \cdot (K_T - 1)_+,
$$

其中 `red(S_T)` 表示相对最小 anchor-connecting forest 的冗余边数，`K_T` 表示与 anchor 有关的连通分量数。

因此，最终对数奖励写为：

$$
\log R(S_T, y) = U(y, Y^*) - \Psi(S_T).
$$

### 8.4 关于 failure sink 的解释

`\alpha_{fail}` 并不是一个无关紧要的实现细节，而是当前方法中的一个 calibration knob。它直接影响：

- 模型何时倾向于 abstain；
- failure stop 与 wrong answer commit 的相对偏好；
- commit rate 的整体校准。

因此，我们建议在论文中明确把 failure sink 解释为 **selective answering / abstention-aware KGQA** 的一部分，而不是单纯的负奖励项。


## 9. 训练目标：SubTB on committed terminal objects

在训练上，我们仍然使用 GFlowNet 的平衡思想。区别不在于是否使用 GFlowNet，而在于终止监督已经从单纯的终止子图质量，变成了 committed terminal object 的质量。理想目标为：

$$
P_T(S_T, y \mid q, G, A) \propto R(S_T, y \mid q, G, A).
$$

实现中，我们采用 Subtrajectory Balance (SubTB)。对一条从初始状态到 `(S_T, y)` 的轨迹 `\tau`，最后一个 stop step 的 reward contribution 为：

$$
\log R_t = \log R(S_T, y).
$$

因此，SubTB 负责在轨迹层面约束流守恒，而终止奖励则把终止流与答案提交直接绑定起来。


## 10. 推理：Monte Carlo posterior surrogate over committed answers

推理阶段，我们从前向策略中进行 Monte Carlo rollout。每条 rollout 最多只向一个答案实体分配质量：

- 若 rollout 终止并提交 `y`，则为 `y` 增加一次计数；
- 若 rollout 终止于 failure sink，则不为任何答案分配质量。

令 `v_y` 表示提交到 `y` 的 rollout 数，`N` 为总 rollout 数，则我们返回：

$$
\mathrm{score}(y) = \log \frac{v_y}{N}.
$$

这里的 `score(y)` 应解释为 **terminal-flow-induced posterior surrogate**。它来源于 committed terminal objects 的 Monte Carlo 频率，而不是对真实贝叶斯后验的直接恢复。


## 11. 本文 claim 的边界

为了避免过度宣称，论文中建议采用如下边界清晰的说法。

### 11.1 可以强 claim 的部分

我们可以合理声称：

- 当前方法是 GFlowNet 框架下的一个合法实例；
- 终止对象 `(z, y)` 的设计比“终止子图后处理答案”更贴近 KGQA 目标；
- Monte Carlo commit frequency 为答案分布提供了更清晰的估计语义；
- 当前主线保留了 latent witness 的多模态探索能力。

### 11.2 应避免的过强表述

不建议写：

- “recover the true answer posterior”；
- “eliminate witness multiplicity bias”；
- “answer-ready entities are the semantically correct answer set”。

更稳妥的写法应为：

- reward-induced posterior surrogate；
- explicit marginalization over committed terminal objects；
- structurally admissible commit set。


## 12. 可直接放进论文的核心方法表述

下面给出一段可以直接整理进论文正文的浓缩版描述。

> We cast multi-anchor KGQA as answer posterior estimation with latent witness flows. Instead of
> terminating in an unlabeled witness subgraph, our GFlowNet terminates in a committed object
> `(z, y)`, where `z` is a latent witness and `y` is an explicitly selected answer entity. The
> policy therefore allocates terminal flow directly over answer-committed witnesses. This yields a
> reward-induced posterior surrogate over answers through marginalization over latent witnesses,
> while preserving the multi-modal exploration advantages of GFlowNets.


## 13. 可直接放进实验章节的写法

### 13.1 主模型

我们的方法 `Answer-Committed RankFlow` 是一个层次化 GFlowNet。模型从所有锚点出发逐步扩展 witness 子图，并在终止时从 failure sink 与 admissible answer sinks 中做出选择。终止奖励同时考虑答案正确性与 witness 紧凑性；训练采用 SubTB；推理通过 Monte Carlo rollout 估计 committed answers 的经验频率。

### 13.2 主实验组

当前仓库中可直接复现实验的主配置包括：

- `experiment=train_rankflow`：标准主线；
- `experiment=train_rankflow_fastiter`：快速迭代版；
- `experiment=train_rankflow_guided`：加入 replay guidance 与 imitation；
- `experiment=train_rankflow_guided_fastiter`：guided 快速版；
- `experiment=train_rankflow_ablate_answer_commit`：削弱 wrong-answer / failure penalty 的弱提交消融。

### 13.3 最关键的实验问题

我们建议将论文主问题明确成以下两个：

1. **显式 answer commitment 是否优于纯 witness 终止后处理？**
2. **teacher-guided replay 是否能进一步提升 answer commitment 的质量，尤其是在多锚点多跳样本上？**


## 14. 代码实现映射

当前实现与上述理论要素的对应关系如下：

- witness 状态与可达性分析：`src/models/gflownet/state.py`
- admissible commit set 与终止奖励：`src/models/gflownet/reward.py`
- stop/continue 分层 actor：`src/models/gflownet/actor.py`
- rollout 采样与 stop 归一化：`src/models/gflownet/sampler.py`
- backward policy 与策略封装：`src/models/gflownet/policy.py`
- SubTB 与训练日志：`src/models/gflownet/losses.py`、`src/models/gflownet_module.py`
- Monte Carlo posterior surrogate：`src/metrics/subgraph_answer_search_runtime.py`


## 15. 当前局限与后续扩展

当前方法已经构成一个较完整的主线，但仍有明显可扩展空间：

1. **Role-aware commitment**：当前 entity-level merge 仍未显式区分 query role；
2. **Answer-normalized witness prior**：未来可进一步削弱 witness multiplicity 对答案边缘的影响；
3. **Commit calibration**：failure sink 与 stop entropy 的显式校准值得单独研究；
4. **Constraint-aware replay**：可以针对 harder multi-anchor joins 做更有针对性的 replay。


## 16. 一句话版本

如果需要用一句话概括本文方法，建议写成：

> 我们将多锚点 KGQA 建模为一个由奖励诱导的答案后验近似问题，并通过在层次化 GFlowNet 中引入显式 answer-commit stop sink，把终止流分配从一般子图生成转化为对 committed answer-witness 对象的建模。
