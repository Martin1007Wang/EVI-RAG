# GFlowNet 算法改进建议（Algorithm Advice）

> 视角假设：你已经有一套可运行的 GFlowNet + RAG 管线，  
> 本文不讲工程细节，只讨论“如果要把这套算法做得更极致，可以在哪些轴上优化”。

---

## 0. 当前实现的抽象视角

先把你现在的 GFlowNet 写成一个干净的数学对象，方便后续对比。

- 条件变量：  
  每个样本给定 $(q, G^{\text{agent}})$：
  - 问题 $q$；
  - 子图 $G^{\text{agent}} = (V,E)$，由 g_raw → g_retrieval → g_agent 的确定性算子构造。

- 状态与轨迹：
  - 环境状态 $s_t$ 等价于“当前 tail 节点 + 已选边集合 $\mathcal{E}_t$”；
  - 动作空间 $\mathcal{A}(s_t)$：  
    从当前 tail 出发的邻接边集合（在 simple-walk 约束下）加上 STOP。
  - 轨迹 $\tau = (a_0,\dots,a_T)$，终态用选中边集合
    $$
      x(\tau) := \mathcal{E}_\tau \subseteq E
    $$
    表示。

- 分布目标（Trajectory Balance）：
  - 希望在终态空间上学习
    $$
      \pi_\theta(x \mid q,G) \propto R(x \mid q,G),
    $$
    其中 $R$ 由 AnswerOnlyReward（命中答案 vs 未命中）给出，并可选加上答案覆盖率 shaping。
  - 前向、反向、配分函数：
    $$
      P_F(\tau) = \prod_t P_F(a_t \mid s_t,q,G),
    $$
    $$
      P_B(\tau) = \prod_t P_B(s_{t-1} \mid s_t,q,G),
    $$
    $$
      Z(q,G) = \sum_x R(x \mid q,G).
    $$
  - TB 残差：
    $$
      \delta(\tau)
      = \log Z(q,G)
        + \log P_F(\tau)
        - \log P_B(\tau)
        - \log R(x(\tau)\mid q,G),
    $$
    训练时最小化
    $$
      \mathcal{L}_{\text{TB}}
      = \mathbb{E}_{(q,G), \tau \sim P_F}
        \big[\,\delta(\tau)^2\,\big].
    $$

- 反向策略 $P_B$：
  - 均匀模式：$P_B^{\text{uniform}}(u\mid v) = 1/d_{\text{in}}(v)$；
  - 学习模式：在每个 target $v$ 上对其父边做 log-softmax，并加熵正则与 L2。

在这个基础上，下面的建议可以理解为：在“状态空间、奖励设计、先验注入、反向策略、训练目标与评估对齐”上做更优的选择。

---

## 1. 状态从单条路径扩张到“子图生成”更自然

**现状：**  
环境 `GraphEnv` 的状态接近“单条 simple path”：

- `current_tail`：单个当前节点；
- `forbid_revisit=True`、`forbid_backtrack=True`；
- 动作只能沿 `current_tail` 的邻接边前进或 STOP；
- 终态 $x(\tau)=\mathcal{E}_\tau$ 实际上是一条简单路径诱导的子图。

**问题：**

1. 很多 QA 任务需要多个 disjoint / overlapping paths 的联合证据，  
   但单条 walk 无法表达“多路径子图”的结构。
2. 单 tail 的 Markov 结构在 hub-heavy 图上过于刚性，可能限制了对高奖励区域的覆盖。

**建议：把状态升级为“子图生成过程”，而不是“单条游走”。**

- 定义状态为
  $$
    s_t = (F_t, \mathcal{E}_t),
  $$
  其中：
  - $F_t \subseteq V$ 是 frontier 节点集合；
  - $\mathcal{E}_t \subseteq E$ 是已选边集合。
- 动作：
  - 从 frontier 上选一个节点 $v\in F_t$ 和一条 incident 边 $e=(v,r,u)$；
  - 或者选 STOP。
- 状态转移：
  - $\mathcal{E}_{t+1} = \mathcal{E}_t \cup \{e\}$；
  - $F_{t+1}$ 按规则更新（例如把 $u$ 加入 frontier，或移除 $v$）。

这样，终态 $x(\tau)$ 就变成**一般子图**而不只是路径，更吻合你最终给 LLM 的“多条三元组证据”的形式。

数学上，TB 方程不变，只是 $P_F(\tau)$ 的 support 扩大到了更丰富的子图轨迹空间。

---

## 2. 奖励从“纯二元”升级为“结构可解释的 shaping”

**现状（AnswerOnlyReward）：**

- 命中答案：$R_{\text{hit}}(\tau) = r_{\text{succ}}$；
- 未命中：$R_{\text{hit}}(\tau) = r_{\text{fail}}$；
- 可选：覆盖率 shaping（命中时加一项 $\lambda_{\text{reach}}\cdot\text{reach\_frac}(\tau)$）。

本质上，所有命中答案的路径被视为“同等级的好”，内部结构差异（长度、是否绕远路、是否经过高置信边）被忽略。

**建议：在保持可解释性的前提下加入结构项。**

可以把 log-reward 写成：
$$
\log R(x(\tau) \mid q,G)
= \log R_{\text{hit}}(\tau)
  + \gamma \cdot \Phi_{\text{struct}}(x(\tau)),
$$
其中：

- $\log R_{\text{hit}}$ 与现在的命中/未命中逻辑一致；
- $\Phi_{\text{struct}}$ 是一个结构势函数，可以选择：
  - **长度惩罚**：$\Phi_{\text{len}} = -|\mathcal{E}_\tau|$，鼓励短路径/小子图；
  - **置信度加权**：$\Phi_{\text{score}} = \frac{1}{|\mathcal{E}_\tau|}\sum_{e\in\mathcal{E}_\tau}\log(\text{score}_e + \epsilon)$；
  - **GT 覆盖度**：$\Phi_{\text{GT}} = \mathbf{1}\{\mathcal{E}_\tau\supseteq \mathcal{E}_{\text{GT}}\}$ 或基于 GT F1 的连续函数。

在你已有的归一化（对齐最大可能 reward 到 1）基础上，引入 $\gamma$ 不会破坏 TB 的形式，只是改变了你想要近似的目标分布 $\pi_\theta$。

这种 shaping 是**完全可数学描述的**，不会变成 heuristics 黑箱。

---

## 3. 检索软先验：从“强依赖 retriever”到“可控的专家组合”

你现在的前向 logit 为：
$$
\ell_e^{\text{joint}}
= \ell_e^{\text{policy}}
  + \alpha(t)\cdot \log(\max(\text{score}_e, 10^{-4})),
$$
且
$$
\alpha_{\text{train}}(t)
= \alpha_{\text{start}}
  + (\alpha_{\text{end}} - \alpha_{\text{start}})
    \cdot \min\Big(1,\ \frac{t}{T_{\text{anneal}}}\Big)
$$
（配置中 $\alpha_{\text{start}}=2.0,\ \alpha_{\text{end}}=0.5,\ T_{\text{anneal}}=10^4$）。

**问题：**

1. 在大的 $\alpha$ 区间，$P_F$ 实际是在学习 “retriever 先验后的残差”，如果 retriever 有系统偏差，GFlowNet 很难纠正。
2. retriever 已经在 g_agent_builder 阶段影响了图结构（Seed→Anchor 2-hop），在 actor 中再次用强先验有“重复施压”之嫌。

**建议：把 retriever 视为“一个专家”，而不是“要绝对服从的神谕”。**

1. 显式地把联合策略写成
   $$
   P_F^{\text{joint}}(e \mid s_t)
   \propto \exp(\ell_e^{\text{GFlowNet}})\cdot \text{score}_e^{\alpha(t)},
   $$
   并意识到 **调 $\alpha(t)$ 本质上是调节“专家混合权重”**。

2. 在 tiny-setting 上做三种配置的系统对比：
   - 只用构图先验，不用 logit 先验（$\alpha\equiv 0$）；
   - 大 $\alpha$ 但退火到 0；
   - 当前设置（退火到 0.5 或更高）。

3. 关键 sanity check：构造一个 retriever 明显“故意错”的 tiny 图，看 GFlowNet 在不同 $\alpha$ 下能否纠正先验。  
   如果在合理训练步数下，GFlowNet 不能把分布从错误先验中拉回来，那说明先验注入得过重，需要减弱或只在构图层使用。

---

## 4. 反向策略 PB：让 TB 和 PB-NLL 的目标更正交

当前 PB 相关的优化目标有两部分：

1. TB 内在的 $\log P_B$：
   $$
   \delta(\tau)
   = \log Z + \log P_F(\tau) - \log P_B(\tau) - \log R(x(\tau)).
   $$
2. 额外 PB NLL + 熵正则 + L2：
   $$
   \mathcal{L}_{\text{PB}}
   = \mathcal{L}_{\text{PB-NLL}}
     + \mathcal{L}_{\text{PB-Ent}}
     + \mathcal{L}_{\text{PB-L2}}.
   $$

实现中总 loss：
$$
\mathcal{L}
= \mathcal{L}_{\text{TB}}
  + 1\cdot \mathcal{L}_{\text{PB}}
  + \mathcal{L}_{\text{GT}}.
$$

**潜在问题：**

- $\mathcal{L}_{\text{PB-NLL}}$ 本质上在做“行为克隆”：  
  让 $P_B$ 在当前轨迹访问的边上尽量大，而这些轨迹由 $P_F$ 产生。  
  当 TB 想让 $P_B$ 做某事，而 NLL 想让它拟合当前策略分布时，两者目标可能冲突。

**建议：**

1. 给 PB-loss 一个显式系数 $\lambda_{\text{PB}}$：
   $$
   \mathcal{L}
   = \mathcal{L}_{\text{TB}}
     + \lambda_{\text{PB}}\mathcal{L}_{\text{PB}}
     + \mathcal{L}_{\text{GT}},
   $$
   而不是固定为 1。

2. 训练策略可以是：
   - 早期用较大 $\lambda_{\text{PB}}$ 加速 $P_B$ 收敛到“合理分布”（避免纯均匀时的数值不稳定）；
   - 随 epochs 退火 $\lambda_{\text{PB}}\to 0$，后期主要由 TB 约束决定 $P_B$。

3. 在 toy graph 上验证：  
   关掉 PB-NLL 时，TB 是否仍能让 $\log P_F$ 与 $\log R$ 有高相关性（`logpf_logr_corr`）；  
   若可以，说明 PB-NLL 真正的角色只是“加速器”，不应该长期作为强约束。

---

## 5. 训练目标 vs 评估指标：对齐 success@K / path_hit_any@K

当前训练目标是 TB（加 PB 与 GT replay），评估指标则包括：

- 单轨迹：`rollout_reward`、`success_mean`、`answer_f1`、`path_hit_f1`；
- 多轨迹（Best-of-K / Any-of-K）：`success@K_s`、`path_hit_any@K_s`、`answer_hit_any@K_s`、`answer_recall_union@K_s`；
- 窗口指标：`path_hit_f1@K_p`（前 $K_p$ 条边上的路径 F1）；
- 多样性与对齐：`modes_found`、`unique_paths`、`logpf_logr_corr` 等。

**关键点：**  
TB 约束的是“**终态分布**”的关系，而 success@K / path_hit_any@K 是在有限采样下的函数：
$$
\text{success@}K_s
= 1 - \mathbb{E}\Big[ (1 - \text{success}(\tau))^{K_s} \Big].
$$

因此，若你主要关心的是 success@K 和 Any-of-K 指标，reward 设计与 TB 收敛应该服务于这件事，而不是反过来。

**建议：**

1. **对 reward 做“评估指标驱动”的微调**：
   - 如果发现 TB 已经收敛（`logpf_logr_corr` 高），但 success@K 仍然偏低，  
     说明 reward 没有充分 encode “多样性 + hit@K” 的需求；
   - 可以适当增加对“模式多样性”的奖励（例如鼓励命中不同答案实体、惩罚重复路径）。

2. **在 tiny-setting 上显式比较“理想分布 vs 模型分布”的 success@K 差异**：
   - 在一个小图上可以显式枚举所有终态 $x$，构造理想分布 $\pi^*(x)\propto R(x)$；
   - 分别计算：
     - 理论 success@K（从 $\pi^*$ 采样）；
     - 模型 success@K（从 $\pi_\theta$ 采样）；
   - 这样可以准确定位 TB 收敛质量对评估指标的贡献，而不会误把“reward 设计不足”归咎于“模型不够强”。

---

## 6. 实验建议：先在 toy 环境里把上述点做“闭环验证”

任何上述建议，**都应该先在玩具环境里验证**，而不是直接上 WebQSP/真实 KB。推荐一个实验模板：

1. 构造一个极小知识图：
   - $|V|\approx 20$，$|E|\approx 40$；
   - 设计 1–3 条 GT 路径，每条长度 2–4；
   - 手工指定 retriever score（包括一些故意错误的高分边）。

2. 定义理想 reward：
   - 完全命中某条 GT 路径给高 reward；
   - 偏离越远，reward 越低（可以手工设计）。

3. 对比多个变体：
   - 单路径环境 vs 子图环境；
   - 纯 AnswerOnlyReward vs 加结构 shaping；
   - 不同 $\alpha(t)$ 策略 vs 不同 $\lambda_{\text{PB}}$。

4. 观测：
   - TB 残差分布、`logpf_logr_corr`、`success@K`、`path_hit_any@K`、`modes_found` 等；
   - 看哪些设计能在 toy-setting 上**几乎完美重建理想行为**，再考虑移植到大规模任务上。

只有在 toy-setting 上把这些闭环跑通，你才能有信心说：“我的 GFlowNet 算法在数学上和数值上是干净的”，而不是靠经验堆叠 heuristic。

---

## 7. 总结：优先级排序

如果要给一个“有限时间内最值得做”的排序，我建议：

1. 在 toy graph 上验证：当前 TB + AnswerOnlyReward 是否能完美拟合一个手工设计的理想分布；
2. 改环境：从单 tail-path 升级到子图生成；
3. 在 AnswerOnlyReward 上加入可解释的结构 shaping（长度、置信、GT 覆盖）；
4. 系统地调 retriever 先验强度与退火策略，确保 GFlowNet 能纠正错误先验；
5. 给 PB-loss 显式系数，做退火或弱化，使其成为 TB 的“加速器”而非“第二优化目标”；
6. 在 tiny-setting 上对齐训练目标和评估指标 success@K / path_hit_any@K，确保 reward 设计真正服务于你关心的指标。

这些建议的共同点是：**每一条都可以被写成清晰的方程和可复现的实验，而不是模糊的“感觉上更好”。**  
如果你能按这个节奏推进，算法的“数学美感”和工程表现都会一起抬升。  

