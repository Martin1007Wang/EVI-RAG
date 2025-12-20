= 引言 <sec:introduction>
尽管检索增强生成（Retrieval-Augmented Generation, RAG）已成为大模型知识增强的主流范式之一，但在处理多跳推理（Multi-hop
Reasoning）时，系统最终都要面对一个更“工程化”的子问题：*证据生成（evidence generation）*。即，从检索得到的候选子图中，生成一小段、可被 LLM
消费的上下文 $C$（三元组集合或路径集合），使答案可达且噪声可控。

这一步同时受三类预算约束（缺一不可，且量纲不同）：
1. 上下文预算 $B$：可用 token 上限（或等价的证据边数 $K_e$），超过预算就会截断/稀释证据；
2. 本地生成预算 $(K_s, L)$：采样/生成的轨迹条数 $K_s$ 与每条轨迹的步长上限 $L$（在本仓库实现中由环境超参 `max_steps` 控制）；
3. 外部交互预算 $N_"ext"$：推理阶段对外部系统（Retriever/LLM）的调用次数，I/O 受限且难以并行。

为避免量纲混淆，本文后续讨论“交互复杂度”时专指 $N_"ext"$，而把 $(K_s, L)$ 视为可控的本地算力预算。基于这一定义，我们回顾三条主流路线及其核心代价。

#set page(margin: auto)

*线性化过程中的拓扑熵增 (Topological Entropy Increase in Linearization)*。鉴于图的非欧几里得特性与 LLM
自回归序列特性的根本性拓扑失配，常见做法是“检索—序列化”：先检索与问题相关的子图，再将其线性化为无序的三元组序列以适配推理 LLM 的输入
@liuCanWeSoft2024@huangCanLLMsEffectively2024@liSimpleEffectiveRoles2024@xuHarnessingLargeLanguage2025c
。这种路线可以把 $N_"ext"$ 压到常数级（通常一次检索 + 一次 LLM 调用），但线性化会不可逆地破坏图的拓扑依赖，迫使 LLM
在高噪上下文中隐式重构结构。在固定 $B$ 的约束下，随着候选证据规模增大，*可见证据密度*往往下降：召回率（Recall）上升不必然转化为命中率提升，反而可能因为噪声稀释与截断而退化。*这种方法为了“快”，牺牲了“结构”*。

*显式图游走的串行延迟瓶颈（Latency Bottleneck of Explicit Graph
Walking）*。为了挽回结构，图智能体范式将LLM建模为图上的游走者，将推理重构为在线序列决策过程$tau = (e_1,dots,e_L)$，通过多轮“思考-行动”循环动态搜索推理路径@sunThinkonGraphDeepResponsible2023@maThinkonGraph20Deep2024@xuGenerateonGraphTreatLLM2024
。由于每一步的边选择 $e_(t+1)$ 都依赖于 LLM 在线评估，对于长度为 $L$ 的路径，必须执行 $L$ 次不可并行的 LLM
交互，使得 $N_"ext" = O(L)$。这种硬延迟在大规模图谱推理中构成了吞吐量瓶颈@wangLLMsUserInterest2024@kimLargeLanguageModels2024。*这种方法为了“准”，牺牲了“速度”*。

*参数化对齐中的模式坍塌 (Mode Collapse in Parametric Alignment)*。为兼顾结构与速度，参数化方法试图通过 MLE
微调将图知识内化至 LLM
参数中@luoReasoningGraphsFaithful2023@luoGraphconstrainedReasoningFaithful2025a。此范式试图让模型“背诵”训练集中的最优路径，然而，同一个问题往往存在多条合理的推理路径。MLE
训练迫使概率分布坍塌至单一的高频模式，导致模型退化为特定路径的“复读机”，失去了在推理时探索解空间分布的能力。即便是引入
KG-Trie 等硬约束，也仅仅是物理屏蔽了无效路径，而未修正模型内部坍塌的概率流形
@luoGraphconstrainedReasoningFaithful2025a。*这种方法为了“效率”，牺牲了“多样性”*。

我们认为，核心低效性并非来自“图结构本身”，而来自把路径组合视为推理时的在线离散优化，并把优化算子外包给 LLM I/O（把 $N_"ext"$ 线性放大）。
为此，我们提出一种基于*分摊推理（Amortized Inference）*的新范式：将“如何在候选图上组合证据”的计算前置到训练阶段，在推理阶段用一次前向的生成采样替代在线搜索。
需要强调的是：该范式并不声称消除检索阶段的不确定性——若答案在候选图中不可达，则任何后验采样都无解；在本仓库的 `g_agent` schema 中，这一上界以 `is_answer_reachable` 的一致性校验被显式编码。

给定查询 $q$ 和大规模原始图 $G_"raw"$，第一阶段由检索器召回候选子图 $G_"sub" subset G_"raw"$。在本仓库实现中，
$G_"sub"$ 以 `g_agent`（PyG Data）作为唯一真源落盘，包含 `node_entity_ids`、`edge_head_locals/edge_tail_locals`、`edge_relations`、`edge_scores` 等字段（见 `src/data/g_agent_dataset.py`）。
核心挑战在于第二阶段：如何从含噪的 $G_"sub"$ 中构造高质量的证据路径 $tau$，以形成上下文 $C$ 供 LLM 使用。现有的主流范式与本文方法的本质差异在于*路径组合策略*的不同：

+ *线性化策略的结构有损性*。 SubgraphRAG 等方法通过映射函数 $phi$ 将子图 $G_"sub"$ 坍缩为序列上下文：
  $ C_"linear" = phi(G_"sub") = \{ (h, r, t) | (h, r, t) in G_"sub" \}, $
  该线性化会丢失多跳连接的拓扑依赖，LLM 被迫在 $P(y|q,phi(G_"sub"))$
  中隐式重构丢失的结构。由于缺乏邻接矩阵的硬约束，LLM
  极易基于文本语义的共现性产生虚假相关，从而在互无连接的实体间“幻觉”路径。

+ *显式搜索的时间复杂性*。 图智能体将推理建模为在线马尔可夫决策过程 (MDP)。路径 $tau = (e_1, dots, e_L)$
  的生成依赖于每一步的串行评估：$ e_(t+1) tilde pi_"LLM" (e_(t+1)|e_(1:t),q,G_"sub") $
  为了获得一条长度为 $L$ 的路径，系统必须执行 $L$ 次外部 LLM I/O，推理延迟为
  $O(L dot C_"LLM")$，这种线性增长的时间成本在工程上难以承受。

+ *MLE 参数化的模式坍塌*。RoG 等方法试图通过参数化微调来内化搜索。其核心目标是最大化专家路径的似然概率（MLE）：$
    theta^*="arg max"_theta sum_(tau in T) log P_theta (tau|q,G_"sub")
  $这会把概率质量挤压到单一路径模式，难以覆盖多解推理与含噪图谱下的多模态解空间。

为了打破上述“不可能三角”，我们提出训练一个生成式流网络（GFlowNet）作为概率路径采样器。不同于 MLE
试图拟合单点，我们旨在学习一个与奖励成正比的概率流形：定义策略网络$pi_theta$，目标为轨迹平衡，使其能从 $G_"sub"$
的分布中直接*采样*出完整的路径组合：
$ tau_"gfn" tilde.op pi_theta (tau | q, G_"sub") $
这里，$pi_theta$ 被训练为与路径回报 $R(tau)$ 成正比（$P(tau) prop R(tau)$）。在本仓库实现中，该对齐通过轨迹平衡族的
Sub-Trajectory Balance（SubTB）损失实现，并采用确定性的反向策略（因此 $log P_B = 0$）。
在推理阶段（Inference Phase），我们不再调用 LLM 对每一步候选边做在线打分，而是在固定候选子图上执行 $K_s$ 次、本地向量化的 rollout（每次至多 $L$ 步），得到多条逻辑链作为上下文
$C_"gfn" = \{tau_1, dots, tau_M\}$，随后一次性输入 LLM 完成最终推理。此时 $N_"ext"$ 为常数，而本地计算成本由 $(K_s, L)$ 决定。

本文的贡献主要如下：

1. 基于流网络的多模态后验匹配 (Multimodal Posterior Matching via
  GFlowNet)。传统的序列模型倾向于坍塌至单一的高频模式（Mode
  Collapse），忽略了多跳推理中答案往往通过多条不同逻辑路径（Reasoning Chains）可达的事实，我们引入 GFlowNet
  的轨迹平衡族目标函数，并在实现中采用 Sub-Trajectory Balance（SubTB）形式 $cal(L)_"SubTB"$。在数学上，该类目标对齐生成概率 $pi_theta(tau)$ 与路径回报
  $R(tau)$
  成正比（$pi_theta(tau) prop R(tau)$）。这使得模型能够通过采样覆盖所有潜在的高回报逻辑链，而非仅仅贪婪地搜索单一解，从而在根本上解决了多样性与覆盖率的问题。

2. 离散拓扑先验的可组合注入 (Composable Injection of Topological Priors)。
  我们将拓扑约束显式写入环境算子（例如起点约束、禁止回访/回退、步长上限等 hard mask），并在这些约束下训练策略网络 $pi_theta$。
  这保证了生成轨迹在结构上是“可证伪”的：任何违反约束的动作在环境层直接不可选，从而把拓扑先验从软惩罚变成硬约束。

3. 外部交互复杂度的常数化 (Constant External Interaction Complexity)。
  通过分摊机制，我们将路径组合的代价前置到训练阶段：传统 Search-based Agent 通常需要 $cal{O}(L)$ 次 LLM 交互（每步一次），而本文方法在推理时只需 $cal{O}(1)$ 次外部交互（一次检索 + 一次 LLM），其余计算均为本地 rollout。
  因此，我们并不宣称“免费获得更强推理”，而是把瓶颈从不可并行的 I/O 迁移到可控的 $(K_s, L)$ 算力预算，并通过学习到的采样分布提高单位预算下的命中与覆盖。
