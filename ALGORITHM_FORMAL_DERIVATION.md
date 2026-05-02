# EVI-RAG 当前算法形式化说明

本文从当前代码实现出发，说明 EVI-RAG/Weaver 模型到底在学习什么。核心结论是：

> 当前实现是一个面向 KGQA 的子图生成式检索器。给定问题和局部知识图，模型从问题实体锚点出发，逐步扩展证据子图，并用 GFlowNet 的流一致性目标让高奖励终端子图获得更高采样概率。

相关主线代码：

- 数据准备：`src/data/preprocess/source.py`、`src/data/preprocess/graph_collect.py`、`src/graph/paths.py`、`src/data/preprocess/materialize.py`
- 数据读取：`src/data/dataset.py`、`src/data/collate.py`、`src/data/datamodule.py`
- 模型：`src/weaver/nn/feature_encoder.py`、`src/weaver/nn/state_readout.py`、`src/weaver/nn/edge_scorer.py`、`src/weaver/policy.py`
- Rollout 与训练：`src/weaver/rollout/engine.py`、`src/weaver/rollout/executor.py`、`src/weaver/reward.py`、`src/weaver/loss.py`
- 评估：`src/training/rollout_eval.py`、`src/eval/retrieval.py`

## 1. 原始样本与局部图

每个样本可写成

```math
(q, \mathcal T, Q, Y)
```

其中：

- \(q\)：问题文本；
- \(\mathcal T=\{(h,r,t)\}\)：局部知识图中的原始三元组；
- \(Q\)：问题实体文本集合；
- \(Y\)：答案实体文本集合。

预处理先清洗三元组：

- 默认移除自环；
- 默认去重完全相同的 \((h,r,t)\)；
- 丢弃空图样本；
- 丢弃没有任何问题实体落在图中的样本。

然后将三元组映射为局部有向图：

```math
G=(V,E),\qquad E\subseteq V\times \mathcal R\times V .
```

问题实体在图中的落点记为 anchor 集合：

```math
A=\{v\in V:\operatorname{text}(v)\in Q\}.
```

答案实体在图中的落点记为 target 集合：

```math
Y_G=\{v\in V:\operatorname{text}(v)\in Y\}.
```

训练和验证配置默认要求：

```math
Y_G\ne\varnothing,\qquad
\exists a\in A, y\in Y_G,\ a\leadsto y .
```

测试集默认不强制答案一定在图内，也不强制 reachable。

## 2. 路径标签与可学习目标的边界

预处理会计算若干图结构标签。它们被保存到 LMDB 样本中，但当前主策略和主奖励不会直接把 target shortest-path 标签作为输入特征。

### 2.1 Anchor 距离

对任意节点 \(v\in V\)，计算：

```math
d_A^\rightarrow(v)=\min_{a\in A}d(a,v),
```

```math
d_A^\leftarrow(v)=\min_{a\in A}d(v,a).
```

不可达时距离记为 \(-1\)。

### 2.2 Reachable targets

可达答案集合为：

```math
Y^+=\{y\in Y_G:\exists a\in A,\ d(a,y)<\infty\}.
```

当前终止奖励优先使用 \(Y^+\)。如果字段存在但为空，不会退回到所有答案；这避免把图中不可达答案作为检索器训练惩罚。

### 2.3 Target-conditioned shortest path labels

对每个 \(y\in Y^+\)，预处理计算反向 BFS 距离：

```math
d_y(v)=d(v,y).
```

并计算从 \(v\) 到 \(y\) 的最短后缀条数：

```math
C_y(y)=1,
```

```math
C_y(u)=\sum_{(u,r,v)\in E,\ d_y(v)=d_y(u)-1}C_y(v).
```

还会标记边是否位于某条 anchor-to-target 最短路上。对边 \(e=(u,r,v)\)，若存在 anchor \(a\in A\) 满足

```math
d(a,u)+1+d_y(v)=d(a,y),
```

则该边被视为 \(y\) 的某条最短证据路径上的边。

这些标签当前主要用于诊断、teacher 指标和可解释性分析，而不是主 GFlowNet loss 的强监督项。

## 3. 文本编码与静态资源

预处理使用文本编码器生成：

```math
x_q\in\mathbb R^D,\quad x_r\in\mathbb R^D,\quad x_v\in\mathbb R^D .
```

其中问题文本会加 query prefix 后编码。relation 文本会把 `/` 和 `_` 替换成空格后编码。

实体分两类：

- 有文本 embedding 来源的实体：查表得到 \(x_v\)；
- 非文本实体：使用一个共享可学习向量。

运行时 `RetrievalDataModule` 加载三类模型资源：

```math
X_{\text{entity}},\quad m_{\text{entity}},\quad X_{\text{relation}} .
```

其中 \(m_{\text{entity}}\) 把全局 entity catalog id 映射到文本 embedding 表行号；非文本实体映射为 \(-1\)。

## 4. 子图状态空间

模型不是一次性选出答案，而是在局部图上逐步生成证据子图。

状态定义为：

```math
s=(V_s,E_s).
```

初始状态为：

```math
V_0=A,
```

```math
E_0=\{(u,r,v)\in E:u\in A,\ v\in A\}.
```

\(E_0\) 是 anchor-induced root edges。它们属于初始证据，会被状态读出模块读取，但不计入学习扩展边成本。

状态不额外保存历史序列语义。其 Markov 状态由当前 active 边集合和静态 batch 图共同确定：

```math
V_s=A\cup \operatorname{endpoints}(E_s).
```

代码中 `State`/`RolloutState` 为效率保存 active node mask 和 active edge mask，但语义上 active nodes 是由 anchors 与 active edges 闭包得到的。

每个图最多扩展 \(B\) 条 learned non-root edges：

```math
|E_s\setminus E_0|\le B.
```

当前默认配置中 \(B=3\)。

## 5. 动作空间与状态转移

每一步有两类动作：

```math
a\in\{\operatorname{Stop}\}\cup\{\operatorname{Expand}(e):e\in\mathcal C(s)\}.
```

合法 frontier 为：

```math
\mathcal C(s)
=
\{e=(u,r,v)\in E\setminus E_s:u\in V_s\ \lor\ v\in V_s\}.
```

选择扩展边 \(e=(u,r,v)\) 后：

```math
E_{s+e}=E_s\cup\{e\},
```

```math
V_{s+e}=V_s\cup\{u,v\}.
```

选择 Stop 后，当前状态成为终端子图。

如果没有候选边、budget 用尽或该 rollout 行已经终止，环境会强制 Stop。强制 Stop 的 log-prob 记为 0，不作为 Stop 充分性的学习证据。

## 6. 静态特征编码

模型先把语义 embedding 投影到模型空间。节点、关系、问题分别为：

```math
h_v=\operatorname{LN}(W_N x_v + b_{\text{DDE}}(v)),
```

```math
h_r=\operatorname{LN}(W_R x_r),
```

```math
h_q=\operatorname{LN}(W_Q x_q).
```

其中 \(b_{\text{DDE}}(v)\) 来自 anchor-conditioned directional diffusion encoding。

Directional DDE 的初始信号为：

```math
a_v=\mathbf 1[v\in A].
```

正向传播一轮为：

```math
f^{(k+1)}(v)
=
\frac{1}{|\operatorname{In}(v)|}
\sum_{(u,r,v)\in E}f^{(k)}(u),
```

反向传播一轮为：

```math
b^{(k+1)}(v)
=
\frac{1}{|\operatorname{Out}(v)|}
\sum_{(v,r,u)\in E}b^{(k)}(u).
```

默认包含 anchor indicator、2 轮正向传播、2 轮反向传播。

边的模型空间表示为：

```math
\phi(e=(u,r,v))=W_E[h_u,h_r,h_v].
```

## 7. Query-conditioned 状态读出

给定一组向量 \(S=\{z_i\}\)，状态读出使用 query attention pooling：

```math
\operatorname{Pool}_q(S)
=
\sum_{z_i\in S}
\frac{\exp(\langle h_q,z_i\rangle/\sqrt H)}
{\sum_{z_j\in S}\exp(\langle h_q,z_j\rangle/\sqrt H)}
z_i.
```

当前状态的节点证据、边证据、relation path memory 分别为：

```math
n_s=\operatorname{Pool}_q(\{h_v:v\in V_s\}),
```

```math
e_s=\operatorname{Pool}_q(\{\phi(e):e\in E_s\}),
```

```math
r_s=\operatorname{Pool}_q(\{h_r:(u,r,v)\in E_s\}).
```

状态表示为：

```math
h_s
=
\operatorname{LN}\left(
g_\theta([h_q,n_s,e_s,r_s])
\right).
```

状态 flow 为：

```math
\log F_\theta(s\mid q)=f_\theta(h_s).
```

根状态 flow 被视作 partition value：

```math
\log Z_\theta(q)=\log F_\theta(s_0\mid q).
```

## 8. 语义边先验

对候选边 \(e=(u,r,v)\)，当前 edge scorer 是纯语义先验，没有 residual 分支。它使用两部分分数：

```math
\rho_r(e,q)=\langle x_q,x_r\rangle,
```

以及新文本端点分数。若一端已在当前子图中，另一端是新加入的文本实体，则：

```math
\rho_n(e,s,q)=\langle x_q,x_{\text{new}}\rangle.
```

否则：

```math
\rho_n(e,s,q)=0.
```

语义分数为：

```math
\operatorname{sem}(e,s,q)
=
\rho_r(e,q)+\alpha\rho_n(e,s,q).
```

边先验 logit 为：

```math
z_0(e\mid s,q)=\tau\operatorname{sem}(e,s,q).
```

在同一状态的 frontier 内归一化：

```math
\log P_0(e\mid s,q)
=
z_0(e\mid s,q)
-
\log\sum_{e'\in\mathcal C(s)}
\exp z_0(e'\mid s,q).
```

默认配置中 `entity_weight_init=0.1`、`logit_scale_init=5.0`，并且这两个量默认冻结。

当前标准 GFlowNet target policy 不使用 \(\log P_0\)。`EdgeScorer` 只保留为语义先验诊断，后续若启用 proposal/behavior mixture，也不能把 proposal log-prob 写入 SubTB 所需的 target \(\log P_F\)。

## 9. Backward-flow GFlowNet 前向策略

当前默认 `action_parameterization` 是 `gfn_backward_flow`，反向策略为 `uniform_removable`。

对每个候选边，策略先构造 successor state \(s+e\)，再估计其 flow：

```math
\log F_\theta(s+e\mid q)
```

每条候选边还需要先计算 child state 下的反向概率：

```math
\log P_B(s\mid s+e)
=
-\log|\mathcal R(s+e)|.
```

候选扩展动作的 target logit 为：

```math
z_e(s,q)
=
\log F_\theta(s+e\mid q)
+
\log P_B(s\mid s+e).
```

Stop 动作的 target logit 是当前状态立即停止的奖励：

```math
z_{\operatorname{Stop}}(s)=\log R(s).
```

动作空间不是两阶段 Stop/Expand gate，而是单层动作集合：

```math
\mathcal A(s)
=
\{\operatorname{Stop}\}
\cup
\{\operatorname{Expand}(e):e\in\mathcal C(s)\}.
```

因此 target policy 直接在所有动作上归一化：

```math
P_F(a\mid s,q)
=
\frac{\exp z_a(s,q)}
{\exp z_{\operatorname{Stop}}(s)+
\sum_{e'\in\mathcal C(s)}\exp z_{e'}(s,q)}.
```

特别地：

```math
P_F(\operatorname{Expand}(e)\mid s,q)
=
\frac{
\exp(\log F_\theta(s+e\mid q)+\log P_B(s\mid s+e))
}{
\exp(\log R(s))+
\sum_{e'\in\mathcal C(s)}
\exp(\log F_\theta(s+e'\mid q)+\log P_B(s\mid s+e'))
}.
```

采样时 temperature 只影响 behavior distribution：

```math
P_{\text{beh}}(a\mid s)\propto \exp(z_a/T).
```

写入 loss 的 \(\log P_F\) 始终来自未加温 target policy。

## 10. 终止奖励

终端奖励由两部分构成：anchor-supported answer coverage 和边复杂度惩罚。

给定终端子图 \(s=(V_s,E_s)\)，先把 active edges 当作无向边，从 active anchors 出发求连通可达集合：

```math
\operatorname{Supp}(s)
=
\operatorname{Reach}_{\text{undir}}(A,E_s).
```

奖励答案集合默认为 \(Y^+\)。支持答案召回率为：

```math
U(s,q)
=
\frac{|Y^+\cap \operatorname{Supp}(s)|}
{|Y^+|}.
```

若某图没有 reward target，则实现中该图的 \(U\) 保持为 0。

复杂度为 learned expansion edge 数：

```math
B(s)=|E_s\setminus E_0|.
```

最终 log reward 为：

```math
\log R(s)
=
\max\left(
\log(\epsilon+U(s,q))-\lambda_E B(s),
r_{\min}
\right).
```

当前默认：

```math
\epsilon=10^{-4},\qquad
\lambda_E=0.10,\qquad
r_{\min}=-30.
```

注意：代码还计算 answer precision、answer recall、answer F1、answer degree excess 等指标，但这些是诊断项，不进入终止奖励。

## 11. 反向策略

GFlowNet loss 需要前向概率和反向概率。当前反向策略不是神经网络，而是 uniform removable-edge policy。

实现中有两个反向概率计算位置：

- target policy 构造 edge logits 前，对所有候选 \((s,e)\) 批量计算 \(\log P_B(s\mid s+e)\)；
- 采样出某条 Expand transition 后，再把同一 uniform backward policy 下的 selected \(\log P_B\) 写入 rollout trace，供 SubTB 使用。

这两者必须语义一致。若某候选扩展后“刚加的边”不在 child 的 removable set 中，代码会直接报错，因为那意味着 forward action 有非零 target 概率但 backward transition 不存在。

对扩展后的 child state \(s'=s+e\)，定义可逆移除集合：

```math
\mathcal R(s')
=
\{e'\in E_{s'}\setminus E_0:
E_{s'}\setminus\{e'\}\ \text{仍可从 anchors 构造，且}\ e'\ \text{在 parent frontier 合法}\}.
```

反向概率为：

```math
P_B(s\mid s')
=
\frac{1}{|\mathcal R(s')|}.
```

即：

```math
\log P_B(s\mid s')=-\log|\mathcal R(s')|.
```

Stop 是终止动作，没有反向移除边，因此终止步在 loss 中使用：

```math
\log P_B=0.
```

## 12. Rollout 轨迹

一条轨迹为：

```math
\tau=(s_0,a_0,s_1,a_1,\ldots,s_L),
```

其中最后一个动作必然是 Stop。由于最多扩展 \(B\) 条边，最大轨迹长度为：

```math
L\le B+1.
```

训练时每个 batch 图会采样多条独立 rollout。实现中静态图 batch 只保留一份，动态状态用 \(R=K\cdot B_{\text{batch}}\) 行表示，再按 rollout 逻辑切回多条 `RolloutBatch`。

## 13. SubTrajectory Balance 训练目标

当前主损失为 SubTrajectory Balance，加上 StopTB counterfactual。

对轨迹中的任意子段 \([i,j]\)，定义：

```math
\Delta_{i,j}
=
\log F_\theta(s_i)
+
\sum_{t=i}^{j}
\left[
\log P_F(a_t\mid s_t)
-
\log P_B(s_t\mid s_{t+1})
\right]
-
T_j.
```

如果 \(j\) 不是终止 Stop 步，则目标为下一状态 flow：

```math
T_j=\log F_\theta(s_{j+1}).
```

如果 \(j\) 是终止 Stop 步，则目标为终端 reward：

```math
T_j=\log R(s_L).
```

SubTB loss 是所有合法子段残差的加权平方和：

```math
\mathcal L_{\text{SubTB}}
=
\mathbb E_{\tau}
\left[
\frac{
\sum_{0\le i\le j<L}
w_{i,j}\Delta_{i,j}^2
}{
\sum_{0\le i\le j<L}w_{i,j}
}
\right].
```

权重随子段长度指数衰减：

```math
w_{i,j}\propto \lambda_{\text{SubTB}}^{j-i}.
```

当前默认：

```math
\lambda_{\text{SubTB}}=0.9.
```

## 14. StopTB counterfactual

因为当前策略每步都能计算“现在停下”的 reward，训练额外加入 StopTB：

```math
\mathcal L_{\text{StopTB}}
=
\mathbb E_s
\left[
\left(
\log F_\theta(s)
+
\log P_F(\operatorname{Stop}\mid s)
-
\log R(s)
\right)^2
\right].
```

默认总损失为：

```math
\mathcal L
=
\mathcal L_{\text{SubTB}}
+
\mathcal L_{\text{StopTB}}.
```

代码中还支持 StopAdv：

```math
y_{\operatorname{stop}}(s)
=
\sigma\left(
\frac{J_{\operatorname{stop}}(s)-J_{\operatorname{continue}}(s)}{\tau}
\right),
```

并用 BCE 监督 Stop 与“所有 Expand 动作之和”的边界。但默认配置 `stop_adv_coef=0.0`，因此它不是当前默认训练目标的一部分。

## 15. 训练流程

一次训练 step 的抽象流程如下：

1. 从 LMDB 读取一批 `RetrievalBatch`。
2. `FeatureEncoder` 为静态图、问题、关系构造 `FeatureBank`。
3. 对每个样本采样 \(K\) 条 rollout。
4. 每个 step：
   - 计算当前 stop-now reward；
   - 用 policy 计算 state flow、Stop logit 和候选 successor flow；
   - 对每条候选 expansion 计算 candidate-level \(\log P_B(s\mid s+e)\)；
   - 用 \(z_{\operatorname{Stop}}=\log R(s)\) 与 \(z_e=\log F_\theta(s+e)+\log P_B(s\mid s+e)\) 做单层 action softmax；
   - 按 temperature behavior policy 采样动作；
   - Expand 时更新 active edges/nodes；
   - Stop 时写入 terminal reward。
5. 将 rollout traces 拼成 `RolloutBatch`。
6. 计算 SubTB + StopTB loss。
7. Lightning manual optimization 执行反传和 optimizer step。

默认配置中：

```math
K_{\text{train}}=8,\qquad
K_{\text{eval}}=8,\qquad
B=3.
```

## 16. 推理与评估

推理时，模型采样若干终端子图。单条 rollout 的终端子图由 traces 重建：

```math
E_{\text{term}}
=
E_0\cup
\{e_t:a_t=\operatorname{Expand}(e_t)\},
```

```math
V_{\text{term}}
=
A\cup\operatorname{endpoints}(E_{\text{term}}).
```

如果需要一个合并证据子图，代码会对多条 rollout 的终端子图取并集：

```math
E_{\text{union}}=\bigcup_{k=1}^{K}E_{\text{term}}^{(k)},
```

```math
V_{\text{union}}=\bigcup_{k=1}^{K}V_{\text{term}}^{(k)}.
```

评估指标主要是节点检索质量。对某条 rollout 和某个图，retrieved nodes 为终端 active nodes；默认从 retrieved denominator 中排除非答案 anchors。target 默认使用 reachable targets。

```math
\operatorname{Precision}
=
\frac{|V_{\text{term}}\cap Y^+|}
{|V_{\text{term}}\setminus A|},
```

```math
\operatorname{Recall}
=
\frac{|V_{\text{term}}\cap Y^+|}
{|Y^+|},
```

```math
\operatorname{F1}
=
\frac{2PR}{P+R}.
```

`expected_*` 指标估计单次采样的平均质量；`best_of_k_*` 指标表示给模型 \(k\) 次采样机会时，至少一次采到好子图的能力。

## 17. 当前实现的算法定位

当前实现不是以下几类算法：

- 不是直接答案分类器；
- 不是监督式最短路边分类器；
- 不是固定 beam search；
- 不是用 target shortest-path feature 直接喂给 policy 的 teacher-forcing 系统。

它实际学习的是：

```math
P_\theta(x\mid q,G)\propto R(x,q),
```

其中 \(x\) 是从 anchor 出发生长出的有限步证据子图。当前 target policy 由 state flow、uniform removable backward policy 和 stop-now reward 共同定义；语义先验只用于 diagnostics，除非后续显式启用为 behavior proposal。终止奖励定义“答案是否被 anchor-supported evidence 覆盖且子图是否紧凑”。

可以压缩成四句话：

1. 数据准备把每个问题物化为局部有向图、anchor、reachable targets 和路径诊断标签。
2. 模型从 anchor-induced 初始子图开始，每步在 Stop 与 Expand frontier edge 之间采样。
3. 终端子图若能从 anchor 连通支持更多答案且使用更少 learned edges，就有更高 reward。
4. SubTB/StopTB 训练让前向采样分布逼近 reward-proportional 的证据子图分布。
