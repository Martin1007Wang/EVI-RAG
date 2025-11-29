# 3. Preliminaries

**Problem Formulation**

我们将知识图谱问答（KGQA）形式化为在全图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 上的子图推理问题。给定自然语言问题 $q$，任务旨在检索并构建一个包含完整推理链的证据子图 $G_{sub} \subset \mathcal{G}$，从而最大化生成正确答案 $a$ 的后验概率 $P(a | q, G_{sub})$。通常，训练数据通过标注问题实体 $e_q$ 与答案实体 $e_a$ 之间的**最短路径（Shortest Paths）**作为弱监督信号（Weak Supervision）。

**Graph Hierarchy（三级漏斗）**

1. $G_{global}$（全图）：完整 KG（如 Freebase），静态，上帝视角。代码：`SharedDataResources` / `graph_store`。
2. $G_{retrieval}$（检索上下文）：PPR / Query-Center 召回的大子图，作为 retriever 打分的搜索空间；batch 中 scores/heads/tails 全量对应此层。
3. $G_{agent}$（Agent 视图，g_agent 缓存）：在 $G_{retrieval}$ 上按 retriever 分数截断/扩展形成的可见图，分两种模式：
   - $G_{oracle}$（train/val）：Top-K ∪ 注入 GT，GT 覆盖高，用于模仿/监督。
   - $G_{pruned}$（test/部署）：仅 Top-K + 启发式扩展，不注入 GT，GT 覆盖低，在不确定下推理。

**The Challenge: Topological Fragmentation in Pointwise Retrieval**

由于全图 $\mathcal{G}$ 的规模限制，主流范式采用“两步走”策略：首先通过 **Personalized PageRank (PPR)** 等结构化启发式算法，从 $e_q$ 的多跳邻域中提取原始子图 $\mathcal{G}_{\text{init}}$；随后利用 **Dense Retriever** 对 $\mathcal{G}_{\text{init}}$ 中的三元组进行语义评分并截断（Top-K），以获取精简候选集 $\mathcal{C}$。

然而，这一范式面临着严重的**拓扑破碎**（Topological Fragmentation）挑战，其根源在于训练与推理的机制性错位：
1.  **独立评分破坏逻辑链（Pointwise Scoring breaks Logical Chains）**：尽管 Retriever 在训练时接触过完整的路径信号，但在推理阶段，它通常采用双塔结构（Bi-Encoder）对每个三元组 $t \in \mathcal{G}_{\text{init}}$ 进行**独立评分** $s(q, t)$。这种做法强行割裂了节点间的条件依赖性。
2.  **语义与结构的失配（Semantic-Structural Mismatch）**：推理所需的关键“桥接节点”（Bridge Nodes/Hops）往往充当逻辑中转角色，其本身的文本语义可能与问题 $q$ 并无直接重叠（即 $s(q, t_{bridge})$ 较低）。导致这些关键链路在 Top-K 截断中被过滤，最终交付的候选集 $\mathcal{C}$ 退化为一堆语义相关但**结构断裂的孤立碎片（Isolated Fragments）**。

此外，$\mathcal{G}_{\text{init}}$ 中正例证据极其稀疏（Positive Ratio $< 0.1\%$），传统的静态检索无法感知不同复杂度问题所需的**信息饱和度（Information Saturation）**，导致简单问题引入噪声，而复杂多跳问题丢失关键路径。

---

# 4. Methodology

为应对上述挑战，**H-RAG (Hypothesis-Driven Reasoning-Aware Graph Retrieval)** 框架被设计为一个两阶段系统，旨在隐空间中重构逻辑连接并进行自适应推理。

## 4.1 System 1: Global Pruner (Coarse-grained)

System 1 旨在以高召回率为目标，从海量噪声中锚定潜在的推理区域。
给定查询 $q$ 和初始检索集 $\mathcal{C}_{\text{init}}$，我们利用轻量级判别器 $f_{\theta}$（如 Cross-Encoder）评估每个候选三元组 $c_i$ 的局部相关性分数 $s_i = f_{\theta}(q, c_i)$。
我们选取 Top-$K$ 个三元组构建精简候选池 $\mathcal{C} = \{c_1, \dots, c_K\}$。为了支持 System 2 的高效迭代，我们预先计算并缓存 $\mathcal{C}$ 中所有元素的文本嵌入 $\mathbf{H}_{\mathcal{C}} = \{\mathbf{h}_{c_1}, \dots, \mathbf{h}_{c_K}\} \in \mathbb{R}^{K \times d}$ 及问题嵌入 $\mathbf{h}_q$。

## 4.2 System 2: Iterative Reasoner as GFlowNet

System 2 是核心组件，被建模为一个在候选池 $\mathcal{C}$ 上构造拓扑连续证据路径的 **Generative Flow Network (GFlowNet)** Agent。它在候选三元组诱导的子图上执行受约束的路径扩展，同时借助自注意力在远距离节点之间建模语义依赖，从而自适应地组装证据链。

### 4.2.1 State Representation with Hybrid Structure

我们将推理过程建模为构建有序且**拓扑连续**的证据路径的过程。在时间步 $t$，当前状态定义为当前已构建的**连续**路径
$S_t = (c_1, c_2, \dots, c_t)$，
其中对任意 $i < t$，都满足 $c_{i+1}.\text{head} = c_i.\text{tail}$，保证路径在知识图上的连通性。
输入特征 $\mathbf{X}^{(t)}$ 由三部分组成：
1.  **Global Context**: `[CLS]` Token 和查询嵌入 $\mathbf{h}_q$。
2.  **Selected Evidence**: 已选序列 $S_t$ 中的节点嵌入。
3.  **Candidate Pool**: 剩余候选集 $\mathcal{C} \setminus S_t$ 中的节点嵌入。

为了兼顾集合的无序性和推理的时序性，我们采用**角色感知编码（Role-Aware Embedding）**：
$\mathbf{Z}_i^{(t)} = \mathbf{h}_i + \mathbf{e}_{\text{type}} + \mathbb{I}(i \in S_t) \cdot \mathbf{e}_{\text{order}}^{(t)}$

*   $\mathbf{h}_i$: 语义内容嵌入。
*   $\mathbf{e}_{\text{type}}$: 标记节点角色（Query / Context / Candidate）。
*   $\mathbf{e}_{\text{order}}^{(t)}$: 仅对已选节点 $S_t$ 添加相对次序编码，**对候选池中的节点不添加位置编码**，从而保证 Agent 对剩余候选的排列具有置换不变性。

### 4.2.2 Reasoning Policy Network

我们将 $\mathbf{Z}^{(t)}$ 输入多层 Transformer Encoder。Self-Attention 矩阵 $\mathbf{A} \in \mathbb{R}^{(K+2) \times (K+2)}$ 充当动态邻接矩阵，允许模型跨越物理断裂直接捕捉 $c_i$ 与 $c_j$ 的逻辑依赖。

**Policy Head (Action Probability):**
基于 Encoder 输出 $\mathbf{H}^{(t)}$，我们定义前向策略 $P_F(a | S_t)$。动作空间 $\mathcal{A}_t$ 包含：
1.  **Select**: 不再从整个候选池 $\mathcal{C}$ 中任意选取，而是只允许从当前路径末尾三元组 $c_t$ 的**尾实体 (Tail Entity)** 相连的候选三元组中选择 $c_j$，即
$c_j \in \{c_j \in \mathcal{C} \setminus S_t \mid c_j.\text{head} = c_t.\text{tail}\}$。
2.  **Terminate**: 选择特殊的 `Stop` 动作，结束推理。

记可行动作的邻接可行集为
$\mathcal{N}(S_t) = \{c_j \in \mathcal{C} \setminus S_t \mid c_j.\text{head} = c_t.\text{tail}\}$，则有
$P_F(c_j | S_t) \propto \exp(\mathbf{w}_{sel}^T \mathbf{h}_{c_j}^{(t)}), \quad \forall c_j \in \mathcal{N}(S_t)$
$P_F(\text{Stop} | S_t) \propto \exp(\mathbf{w}_{stop}^T \mathbf{h}_{\text{CLS}}^{(t)})$

当 $\mathcal{N}(S_t) = \varnothing$ 时，唯一合法的动作为 `Stop`，此时强制令 $P_F(\text{Stop} | S_t) = 1$。在实现中，这对应于当所有选择动作被 Mask 掉时，仅对 `Stop` 动作解 Mask 并在其上归一化。

### 4.3 Learning with Trajectory Balance GFlowNet

为了解决传统强化学习（RL）易陷入局部最优和监督学习（SL）缺乏探索性的问题，我们采用 GFlowNet 框架训练 System 2。我们的目标是学习一个条件于问题 $q$ 的策略 $P_F(\cdot \mid q)$，使得生成证据链 $S$ 的概率与其对回答当前问题的贡献（Reward）成正比：$P(S \mid q) \propto R(q, S)$。

#### 4.3.1 Theoretical Objective: Trajectory Balance (TB)

考虑到推理路径的序列依赖性，我们采用比 Detailed Balance 更适合长序列任务的 **Trajectory Balance (TB)** 损失函数。
对于条件于问题 $q$ 的一条完整推理轨迹 $\tau = (S_0 \rightarrow S_1 \rightarrow \dots \rightarrow S_T)$，TB 准则要求：
$Z_{\theta}(q) \prod_{t=0}^{T-1} P_F(S_{t+1} | S_t, q; \theta) = R(q, S_T) \prod_{t=0}^{T-1} P_B(S_t | S_{t+1}, q)$

其中：
*   $Z_{\theta}(q)$ 是关于输入问题 $q$ 的配分函数（Partition Function），由模型通过 Amortized Inference 预测，例如通过对 `[CLS]` 表示应用一个 MLP 得到标量 $\log Z_{\theta}(q)$。
*   $R(q, S_T)$ 是条件于问题 $q$ 的终态回报；为简洁起见，下文记作 $R(S_T)$。
*   $P_B(S_t | S_{t+1}, q)$ 是后向策略（Backward Policy）。

**修正后的设计方案（State / Action / Backward）：**

1.  **State 定义**：状态 $S_t$ 为当前已构建的**连续**路径 $S_t = (c_1, c_2, \dots, c_t)$，其中对任意 $i < t$，满足 $c_{i+1}.\text{head} = c_i.\text{tail}$。
2.  **Action 定义**：
    *   动作空间包含“选择下一条三元组”与“Terminate”两类；
    *   在选择动作下，仅允许从当前尾实体相邻的候选三元组中选择，即
        $Next \in \{c_j \in \mathcal{C} \mid c_j.\text{head} = c_t.\text{tail}\}$。
3.  **Backward Policy ($P_B$)**：
    *   对任意 $S_{t+1} = (c_1, \dots, c_t, c_{t+1})$，唯一合法的回退操作是移除尾部三元组 $c_{t+1}$；移除中间任意 $c_i$ 会破坏路径连续性；
    *   因此前驱状态 $S_t$ 唯一确定；
    *   于是 $P_B(S_t | S_{t+1}) = 1$ 对所有合法转移均成立。

TB 损失函数定义为：
$\mathcal{L}_{TB}(\tau) = \left( \log Z_{\theta}(q) + \sum_{t=0}^{T-1} \log P_F(S_{t+1} | S_t, q; \theta) - \log R(q, S_T) \right)^2$

#### 4.3.2 Reward Shaping & Exploration

**Reward Function:**
为了缓解稀疏奖励问题，我们设计了基于覆盖率的软奖励机制。设 Ground Truth 证据集为 $G^*$，终态 $S_T$ 的奖励定义为：
$R(S_T) = \alpha \cdot \text{Recall}(S_T, G^*) + \beta \cdot \mathbb{I}(\text{LLM Answer Correct}) + \epsilon$
其中 $\epsilon$ 为基础能量项，防止 $\log 0$ 错误。

**Local Exploration Strategy:**
在训练过程中，我们采用 On-policy 采样生成轨迹 $\tau$。为避免策略过早收敛到局部最优，我们采用简单的随机探索机制：在训练早期，以概率 $\epsilon_{exp}$ 在当前可行动作集上进行均匀采样，以概率 $1 - \epsilon_{exp}$ 按策略 $P_F(\cdot | S_t)$ 采样动作。该机制在不改变目标分布的前提下，提高了对低概率路径的覆盖度。

---

## 4.4 Handoff：Top-K 图补全

Anchor 阶段输出的 Top-$K$ 三元组在语义上已经高度相关，但结构上仍可能存在缺失。为了承上启下，把 retriever 的候选集变成 GFlowNet 可直接消费的图结构，我们在 `eval.py` 中实现了 **Handoff** 管线：

1.  **Top-K 采样**：对 retriever 的评分 $s(e)$ 按样本排序，保留 $\mathcal{E}_K = \{e_1, \dots, e_K\}$，默认 $K=50$，可通过 `g_agent.top_k` 调节。
2.  **邻域扩展**：对 $\mathcal{V}_K = \{v | v \text{ 是 } \mathcal{E}_K \text{ 的端点}\}$，添加 $h$-hop 邻居（默认 1-hop）及其 incident edges，得到扩展集合 $\hat{\mathcal{E}} = \mathcal{E}_K \cup \bigcup_{v \in \mathcal{V}_K} \mathcal{E}(v)$。
3.  **短路径挖掘**：在候选图上执行 BFS，若存在两两核心节点间的路径 $\pi_{u \rightarrow v}$ 满足 $|\pi| \leq 3$，则提取路径中所有节点/边加入 $\hat{\mathcal{E}}$，让被 Top-K 误删的“隐式”链路被自动补齐。

g_agent 阶段会输出 `g_agent_samples.pt`，包含每个样本的 Top-$K$ 边、扩展后的节点集合以及检索分数，形成 $G_{agent}$：train/val 为 $G_{oracle}$（注入 GT），test 为 $G_{pruned}$（不注入 GT）。运行方式示例：

```bash
python src/eval.py g_agent.enabled=true g_agent.top_k=75 g_agent.max_path_length=3
```

默认情况下，g_agent 会根据 `data.dataset_cfg.data_dir` 自动将 `train/validation/test` 三个 split 的缓存分别写到 `${data_dir}/g_agent/{split}_g_agent.pt`。也可以通过 `g_agent.output_dir` 或 `g_agent.output_paths.split=` 手动指定路径。

## 5. 实践指南：g_agent 数据 & GFlowNet 训练

g_agent 管线会将 `eval.py g_agent.enabled=true ...` 的输出序列化成 `g_agent_samples.pt`。为了直接用于 GFlowNet 训练，我们提供了：

*   **数据模块**：`configs/data/gflownet.yaml` 读取 `{split}_g_agent.pt`，自动 padding 出批次级别的实体/关系 ID、Top-K 掩码与图结构。
*   **策略模块**：`src/models/gflow_net.py` 实现 Role-Aware Transformer + Trajectory Balance 损失，依赖全局实体/关系嵌入 (`resources.{vocabulary_path, embeddings_dir}`) 进行 ID→语义映射。
*   **训练入口**：使用 `python src/train.py experiment=train_gflownet data.cache_paths.train=/path/to/train_g_agent.pt ...` 启动。Reward 目前默认使用正例召回 + $\epsilon$ 平滑，可替换为任意 LLM 判别信号。

**如何批量生成 train/validation/test g_agent 缓存**

```bash
python src/eval.py \
  ckpt_path=/path/to/retriever.ckpt \
  g_agent.enabled=true \
  g_agent.splits=[train,validation,test] \
  data.drop_last=false
```

命令会依次遍历 datamodule 的三种 split，并把结果写到 `${data.dataset_cfg.data_dir}/g_agent/{split}_g_agent.pt`。若想自定义路径，可通过 `g_agent.output_dir=/foo` 或 `g_agent.output_paths.train=/foo/train.pt` 这类 override 精细控制。随后将这些 `.pt` 文件填入 GFlowNet 训练命令中的 `data.cache_paths.*` 参数即可。

这条流水线把“承上启下 g_agent → 结构化候选集 → 无过程标注 GFlowNet”闭环串起来，后续只需替换 Reward 或策略超参即可快速迭代 System 2。
