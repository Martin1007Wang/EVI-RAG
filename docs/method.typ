#import "/lib.typ": *
#set page(margin: auto)

= 方法 <sec:methodology>
我们提出 GFlowRAG，一种基于生成流网络（GFlowNet）的生成式推理方法。为避免在稀疏且长程依赖的图搜索空间中“从零学习”
整套策略分布，我们采用*显式势能差分学习（Explicit Potential Difference）*：固定拓扑先验作为推理的物理坐标系，
并直接用可学习势能函数 $log F(s)$ 生成策略的语义倾向（以势能差形式出现）。该设计保留 Doob-$h$ 的严格结构，
把学习容量集中到“相对于自然游走的偏离”，从而减少用参数去记忆图连通性的浪费。在线阶段，我们在固定跳数 $T$ 与固定采样/beam
预算下生成多条高概率、多样化的推理路径，作为后续阅读器的结构化证据。

*在线复杂度声明（严格口径）.* 对每个查询，我们首先对检索子图 $G_"sub"(q)$ 做一次图编码（开销与子图规模线性相关，
通常可写为 $O(|cal(E)_"sub"|)$）。随后进行固定步数 $T$ 的 rollout/beam。设每一步的候选边数上界为 $D$（由子图采样与动作约束给出），
beam 宽度为 $B$，则在线搜索与打分的复杂度为 $O(T * B * D)$（忽略常数维度）。因此，若将“大规模”定义为原始全图规模 $|cal(E)|$，
则在线推理对 $|cal(E)|$ 是 $O(1)$：只要检索模块以固定预算返回子图（例如固定的边/节点采样上限
$|cal(E)_"sub"| <= K_E$、$|cal(V)_"sub"| <= K_V$，与全图规模无关），整体在线开销就与 $|cal(E)|$ 解耦。
但对检索子图规模仍是线性的，这一点在实现上不可回避。

== 问题设置：有限视界下的生成式推理 <sec:preliminaries>

给定有向知识图谱 $cal(G) = (cal(V), cal(E), cal(R))$，其中
$cal(E) subset.eq cal(V) times cal(R) times cal(V)$ 为多关系有向边集。
对查询 $q$，实体链接器先将问题映射到种子实体集合 $cal(V)_q$（主题实体）。随后，检索模块在全图上返回一个局部子图
$G_"sub"(q)$（本文仅在该子图上训练与推理）。在本文所用的共同基线数据集中，$G_"sub"$ 通常由“$k$-hop 候选扩展 + PPR 剪枝”
构造：先对种子集合做 $k$-hop 邻域扩展得到候选集，再以 $cal(V)_q$ 为重启分布运行 Personalized PageRank，保留 Top-$N$ 节点并诱导边集。

与许多工作默认“子图必含答案”的暗含设定不同，我们显式区分两类答案集合：

+ 全局答案集合 $cal(V)_a$：标注答案在原始知识库中的实体集合（用于监督与评估）。
+ 子图内目标集合 $cal(A)_q = cal(V)_a inter cal(V)_"sub"$：答案落在子图内的部分（训练/评估时用于 hit 判断与奖励）。

当 $cal(V)_a$ 非空但 $cal(A)_q = emptyset$ 时，称为*检索失败*（retrieval failure）：此时任何仅在 $G_"sub"$ 内行动的推理器都不可能命中答案。
因此我们遵循无偏评估协议：不在测试阶段后验丢弃该类样本，而是将其保留在分母中并记为预测错误（Score=0）。同时，为了隔离“推理能力”本身，
我们也在子集 $D_"sub"$ 上报告条件指标：$D_"sub"$ 由那些满足 $cal(A)_q neq emptyset$ 且子图内至少存在从 $cal(V)_q$ 到 $cal(A)_q$ 的可达路径的样本组成；训练仅在 $D_"sub"$ 上进行。

我们在有限视界 $T$ 的 MDP 上建模推理（可等价写作在每个 episode 固定条件 $q$ 下的条件策略）：
- 状态：$s_t = (v_t, t, q)$。
- 动作：从 $v_t$ 的可行动作集合中选择一条边 $(v_t, r, v_(t+1))$。
- 轨迹：$tau = (v_0, r_1, v_1, dots, r_K, v_K)$，其中 $v_0 in cal(V)_q$，$K <= T$，且每步转移均在 $G_"sub"$ 内有效。

*关于“无环”假设.* 知识图谱一般是有环的，我们并不假设 $cal(G)$ 或 $G_"sub"$ 是 DAG。本文只假设有限视界 $T$（保证 episode 终止）。
更严格地说：由于状态已显式包含时间步 $t$，且每次转移都满足 $t -> t+1$，因此诱导的状态转移图在状态空间
$S = cal(V)_q times {0, dots, T}$ 上必然无环（无法回到同一 $(v, t)$）。底层知识图谱可以有环，但不会导致 time-augmented MDP 出现环。
若工程上额外启用 no-revisit 约束（不允许访问已访问节点），则在扩展状态
$tilde(s)_t = (v_t, t, q, V_"visited"(t))$ 上同样无环，且动作约束更强；这属于实现层面的加速/稳定化手段，不作为理论推导的必要前提。

== Doob $h$-变换与势能差分 <sec:theoretical-framework>

我们将“无语义时的自然推理”刻画为参考过程 $P_0$：在任意状态 $s_t$，参考过程执行出度均匀随机游走：
$ P_0(s_(t+1) | s_t) = frac(1, d_"out"(v_t)), quad forall (v_t, v_(t+1)) in cal(E). $

给定非负回报 $R(tau|q)$，Doob $h$-变换给出生成
$P(tau|q) prop P_0(tau) R(tau|q)$ 的最优扭曲过程：
$ pi^*(s' | s) = P_0(s' | s) dot frac(h(s'), h(s)). $ <eq:doob>

我们使用 GFlowNet 的 Detailed Balance（DB）形式将该观点落到可训练目标上。令 $Z(s_t)$ 表示状态流函数，
DB 约束为：
$ Z(s_t) P_F(s_(t+1)|s_t) = Z(s_(t+1)) P_B(s_t|s_(t+1)). $
由此可得到一个等价的 logits 形式（$P_F$ 由对 logits 的 softmax 给出；每个状态的归一化常数会被 softmax 吸收）：
$ "Logits"(s_(t+1)|s_t) = underbrace(log P_B(s_t|s_(t+1)), "Backward Prior") + underbrace([log Z(s_(t+1)) - log Z(s_t)], "Value Residual"). $ <eq:db-logit>

*实现口径（势能差分）.* 式 <eq:db-logit> 表明 Doob-$h$ 的自然参数化即“拓扑先验 + 势能差分”。
在实现中我们直接令
$ psi_theta(u,r,v,q,t) := log Z_theta(s_(t+1)) - log Z_theta(s_t) $
从而使前向策略与势能函数严格一致。

关键在于如何选择 $P_B$ 以严格对齐 $P_0$ 的时间反演，而不是“拍脑袋近似”。为此我们采用一个在共同基线数据集中成立、且实现会显式强制的假设：
对每条边 $(u, r, v)$，子图中包含其唯一逆边 $(v, r^(-1), u)$（逆关系通过固定后缀生成，并在训练前做一致性校验）。
在该“带逆边的有向多重图”上，我们将参考过程 $P_0$ 定义为*对出边（edge）均匀*的随机游走：
$ P_0(v|u) = frac(1, d_"out"(u)), quad (u, r, v) in cal(E). $

*命题（时间反演 = 入度均匀，edge-degree 口径）.* 对上述 $P_0$，平稳分布满足 $pi(u) prop d_"out"(u)$。因此其时间反演过程为
$ P_"rev"(u|v) = frac(pi(u) P_0(v|u), pi(v)) = frac(1, d_"out"(v)). $
由于我们在子图中显式加入逆边，按 edge-degree 计数有 $d_"out"(v) = d_"in"(v)$，从而得到
$ P_"rev"(u|v) = frac(1, d_"in"(v)). $
我们将后向策略锚定为该时间反演先验（实现为对逆向候选边的静态均匀分布），因此有
$ log P_B(s_t|s_(t+1)) = -log(d_"in"(s_(t+1))). $ <eq:pb-indegree>

== 势能流网络：InDegree Prior + Potential Difference <sec:architecture>

Doob-$h$ 的核心信息是：策略并非从零学习，而是对参考过程的 *multiplicative tilt*。在 log 空间中，这一结论等价为“先验项 + 势能差分”。
因此我们将 <eq:pb-indegree> 给出的拓扑先验固定为一个不可学习的偏置项，并直接用势能差分生成前向 logits。

*势能差分（策略定义）.* 对候选边 $e=(u,r,v)$、时间步 $t$，前向 step logits 定义为
$ "Logits"(u, r, v | q, t) = underbrace(-log(d_"in"(v)), "Topological Bias") + underbrace(alpha * (log Z_theta(s_(t+1)) - log Z_theta(s_t)), "Potential Difference"). $ <eq:final-logits>
其中 $alpha>0$ 是可学习尺度，用于匹配势能差与拓扑先验的数值尺度。

== 训练目标：有限视界接地的 DB 残差 <sec:finite-horizon-grounding>

由于推理必须在最大跳数 $T$ 内完成，我们显式建模时间依赖的 $log Z_theta(s, t)$，并用硬接地（Hard Grounding）
抑制超时/死路的价值幻觉。定义接地目标：
$
  log Z^dagger(s, t) = cases(
    0 & "if " s in cal(A) " (Answer Hit)",
    -infinity & "if " t = T and s in.not cal(A) " (Time-out/Dead-end)",
    log Z_theta(s, t) & "otherwise"
  )
$
实现中我们用一个足够大的负常数 $-C$ 近似 $-infinity$（默认 $C=100$），以避免数值实现中显式 $-infinity$ 带来的不稳定。
为了避免训练初期大量失败轨迹导致终止项主导梯度，我们在实现中允许对该接地强度做退火：从较“软”的 $-C_"start"$ 逐步退火到更“硬”的 $-C_"end"$，
从而在保持 $epsilon -> 0$ 极限一致性的同时提升冷启动阶段的数值稳定性。

训练时，我们对采样轨迹最小化逐步的 DB 残差平方和：
$
  cal(L)(theta) =
  bb(E)_(tau tilde P_F) [
    sum_(t=0)^(T-1)
    (log Z_theta(s_t, t) + log P_F(s_(t+1)|s_t) - log Z^dagger(s_(t+1), t+1) - log P_B(s_t|s_(t+1)))^2
  ].
$

实现上，$P_B$ 在逆边动作空间上取静态均匀分布；若启用动态约束（如去重访问或边 dropout），我们仍使用静态归一化作为高效近似，
其误差由残差模块吸收（见审计笔记）。

== 在线推理与 $P_0$ 消融 <sec:training-inference>

在线阶段，我们使用 $P_F( dot | q)$ 进行固定预算的采样/beam，得到多条推理路径，并将其线性化为文本证据供 LLM 阅读器使用。

为了验证“拓扑先验是归纳偏置的核心”，我们在实验中系统对比不同参考过程 $P_0$（实现为前向 logits 的固定偏置项 $log P_0$）：
`none`（无先验）、`outdegree`（出度常数偏置）、`indegree`（本文方法）、`preferential`（$+log d_"in"$，富者越富反例）、`semantic`（语义先验）。
注意：`outdegree` 对同一状态的所有候选边是常数项，因此 softmax 后与 `none` 等价（用于 sanity check）；真正的反例对照是 `preferential`。
