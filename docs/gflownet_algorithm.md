# GFlowNet 算法：在检索子图上的条件流匹配

> 本文不是“代码速查”，而是将当前实现抽象为一个干净的数学对象：  
> 给定问题 $q$ 与其检索子图 $G$，GFlowNet 在轨迹空间上学习一个分布，使得终态采样频率与奖励 $R(x \mid q, G)$ 成正比。

---

## 动机：从三元组线性拼接到路径级生成

在很多基于知识图谱的 RAG 实现（包括本仓库的早期 SubgraphRAG 原型）中，典型做法是：

- 在检索阶段得到一个局部子图（即本仓库的 $G_{\text{retrieval}}$）；
- 将其中的所有三元组 $(h,r,t)$ 排序（通常按检索分数），线性展开为文本片段；
- 将数千条三元组以“平铺”的形式直接喂给 LLM，让 LLM 在纯文本空间里自行“找路径、做推理”。

这种做法存在两个本质问题：

1. **信噪比极低、噪声随上下文线性放大**  
   从 `docs/data_description.md` 的统计可以看到，典型局部图中：
   - 边数 $|E| \approx 4\,000$；
   - 真实最短路径的边数通常为 $1\sim 2$ 条；
   - 中位数信噪比
     $$
       \operatorname{SNR}
       = \frac{\text{最短路径边数}}{|E|}
       \approx 0.027\%.
     $$
   换句话说，**99.97% 的边对最终答案是噪声**。如果直接把所有三元组线性拼接给 LLM：
   - LLM 的上下文窗口被大量无关三元组占据；
   - LLM 必须在高噪声文本中“显式建图 + 隐式搜索”，对多跳问题尤其不友好。

2. **缺乏显式的“路径”这一中间随机变量**  
   三元组平铺方案只在文本层面提供“局部事实”的集合，没有显式表示：
   - 哪些边共同构成了一条候选推理路径；
   - 哪些路径更靠近答案，哪些只是噪声；
   - 如何在这些路径上定义一个**可采样的分布**，用于 Top-K/Best-of-K 的组合。

在这样的设置下，LLM 同时承担了：

- 从噪声极高的子图中提炼出少量有用路径；
- 在这些路径之上完成语义推理与答案生成。

这既不符合“System 1 / System 2”划分的工程美感，也浪费了我们在图层已经构建出的结构信息。

### 为什么在这里引入 GFlowNet？

本仓库的设计思路是：**把“恢复推理路径”显式建模为一个条件生成问题**，并在图层用 GFlowNet 解决它，而不是把所有负担推给 LLM。

更形式化地说：

- 已知：问题 $q$ 与一个局部检索图 $G$（对应 g_retrieval/g_agent）；
- 我们希望在“子图/路径空间”上学习一个分布
  $$
    \pi_\theta(x \mid q, G),
  $$
  其中 $x$ 是某条轨迹 $\tau$ 所诱导的终态（本实现中主要是“被选中的边集合”）；
- 奖励 $R(x \mid q, G)$ 由“答案命中率 + 路径质量”决定（AnswerOnlyReward，AnswerDiffusionReward 已退化为 AnswerOnly 的兼容别名）。

GFlowNet 的角色是：在 $(q,G)$ 条件下，学习一个满足
$$
  \pi_\theta(x \mid q,G) \propto R(x \mid q,G)
$$
的生成模型，从而在训练后：

- **高奖励路径（命中答案、覆盖率高、结构合理）被更频繁地采样**；
- **低奖励路径（纯噪声游走、陷入 hub 节点）被自然抑制**；
- 我们可以在图层执行 Best-of-K / Any-of-K 采样，只把“少量高质量路径/子图”传递给 LLM。

与 SubgraphRAG 的“直接三元组拼接”相比，这带来几个定量上的优势：

- 在同样的检索子图上，我们不再试图“解释所有边”，而是通过 GFlowNet **把边集合压缩成少量高奖励路径**；
- LLM 的输入不再是 SNR $\approx 0.03\%$ 的大包三元组，而是结构上已经对齐答案的子图/路径；
- System 1（retriever）给出边级打分；System 1.5（GFlowNet）在图上做路径级采样；System 2（LLM）仅对少量高质量证据做语义推理。

### 与 SubgraphRAG 管线的逐步对比

为避免“空谈优点”，这里把 SubgraphRAG 的 reasoning 管线与本仓库的 GFlowNet 管线做一个一一对应的对比。

1. **SubgraphRAG：基于打分的三元组平铺**

   在 SubgraphRAG 的 reasoning 代码中（`SubgraphRAG/reason/preprocess/prepare_prompts.py`），每个样本 `each_qa` 含有：

   - `graph`：原始局部图的三元组列表；
   - `scored_triplets`：带分数的三元组；
   - `good_triplets_rog`：根据 Reward-of-Graph (RoG) 策略挑出的“好三元组”子集；
   - `a_entity`：答案实体列表。

   构造 prompt 的核心函数是 `get_prompts`：

   - 根据 `mode` 选择三元组子集（`rog_*` / `scored_*` / `rand_*`）：
     - `rog_*`：优先使用 `good_triplets_rog`，再补充 `scored_triplets`；
     - `scored_*`：对 `scored_triplets` 按分数筛选、裁剪到给定数量；
     - `rand_*`：从 `graph` 中随机采样若干三元组。
   - 统一线性展平为文本块：
     ```text
     Triplets:
     (h1,r1,t1)
     (h2,r2,t2)
     ...
     ```
   - 再与问题拼接成：
     ```text
     Triplets:
     ...

     Question:
     ...
     ```

   这条管线的关键特点是：

   - **控制的是“三元组数量”和“分数阈值”**，而不是显式的路径结构；
   - 三元组之间的组合关系（是否构成从问题实体到答案实体的连通路径）完全交给 LLM 在文本层自行恢复。

2. **EVI-RAG + GFlowNet：路径级采样，再转换为三元组 prompt**

   本仓库在 retrieval 阶段与 SubgraphRAG 类似，同样构建局部图并为每条边打分；差异在于 reasoning 阶段：

   - 先通过 `GAgentBuilder` 从 g_retrieval 抽取一个结构化的 g_agent 子图：
     - 明确起点实体集合 $S$、答案实体集合 $A$；
     - 强制保留穿过 Answer 的 GT 路径；
     - 利用 Seed→Anchor 2-hop 过滤掉大量与种子/答案无关的边。
   - 在 g_agent 上运行 GFlowNet：
     - 状态是“已选中的边集合 + 当前节点”；
     - 动作为“选择一条边 / STOP”；
     - 奖励只依赖于最终访问到的答案节点及其结构特性（AnswerOnly / AnswerDiffusion）。
   - 训练收敛后，`GFlowNetActor.rollout` 把“高奖励路径”以更高频率采样出来。

   到 LLM 阶段（`src/utils/llm_prompting.py`），我们仍然使用“Triplets: ...”这种文本形式：

   ```python
   def build_triplet_prompt(question, triplets, limit):
       lines = [triplet_to_str(t) for t in triplets[:limit]]
       triplet_block = "Triplets:\n" + "\n".join(lines)
       question_block = f"Question:\n{question}"
       ...
   ```

   但此时传入的 `triplets` 不再是“任意高分三元组”的子集，而是：

   - 由 g_agent 中的边（`edge_head_locals/edge_tail_locals/edge_relations`）反查回文本；
   - 按 GFlowNet 采样到的路径/子图排序和裁剪；
   - 即，**先在图层约束出一组高奖励路径，再以三元组形式提示 LLM**。

   换句话说：

   - SubgraphRAG：**score → 子采样三元组 → LLM 自行找路径**；
   - 本仓库：**score → g_agent 子图 → GFlowNet 采样路径 → 再转三元组 → LLM 在“低噪声路径”上做语义推理**。

后文的所有章节可以理解为对这一句子的展开：  
> **GFlowNet 在 g_agent 子图上，把“答案监督”转化为一条“路径分布”，从而在图层恢复出低噪声的推理路径，再交给 LLM。**

---

## 0. 全局数据流：g_raw → g_retrieval → g_agent → GFlowNet

这一小节回答“样本是从哪里来的”，即：
从原始 KB 与问答对 $(\text{KB}, q, A)$ 出发，如何经过一系列确定性算子，得到 GFlowNet 训练所消费的 `g_agent` 图。

### 0.1 g_raw：全局图与问答对（概念层）

在概念层，我们有：

- 全局实体空间：$E = \{1, \dots, |E|\}$；
- 全局关系空间：$R = \{1, \dots, |R|\}$；
- 全局知识图：$G_{\text{global}} = (E, \mathcal{E}_{\text{global}})$，其中
  $$
    \mathcal{E}_{\text{global}} \subseteq E \times R \times E,
    \quad (h,r,t) \in \mathcal{E}_{\text{global}} \iff \text{存在三元组 } (h,r,t) .
  $$

每条样本（问题）$i$ 在 g_raw 层仅由以下对象刻画：

- 问题文本 $q_i^{\text{text}}$；
- 其问题实体集合 $S_i^{\text{raw}} \subseteq E$（seed / question entities）；
- 答案实体集合 $A_i^{\text{raw}} \subseteq E$；
- ground-truth 路径集合（在全局图上的最短路径或人工标注）
  $\Gamma_i^{\text{raw}} \subseteq E^\ast$。

这一层在代码中对应 `scripts/build_retrieval_parquet.py` 的输入 Parquet（问题 + 三元组列表 + q_entity/a_entity），不直接被训练代码读取，只作为后续所有缓存的“单一真源（SSOT）”。

### 0.2 g_retrieval：局部检索图（LMDB 缓存）

第一层物化是 `g_retrieval`，它把全局图切成“问题特定的局部子图”。  
在每个样本 $i$ 上，我们得到一个局部图
$$
  G_i^{\text{retr}} = (V_i^{\text{retr}}, E_i^{\text{retr}}),
$$
其中：

- $V_i^{\text{retr}} \subseteq E$：局部实体节点集合；
- $E_i^{\text{retr}} \subseteq V_i^{\text{retr}} \times R \times V_i^{\text{retr}}$：局部有向边；
- 每条边 $e = (u,r,v)$ 带有：
  - 全局关系 ID $r$；
  - 检索标签 $y_e \in \{0,1\}$（正例/负例）；
  - 之后 retriever 训练出的分数 $\hat{s}_e$。

在代码中，这一层由：

- `scripts/build_retrieval_dataset.py` 生成的 LMDB（每个 `sample_id` 一条记录）；
- `src/data/g_retrieval_dataset.py` 中的 `GRetrievalDataset` 读取，构造 PyG `Data`：
  - 图结构：
    - `edge_index[2, E]`：局部节点索引；
    - `edge_attr[E]`：全局关系 ID；
    - `labels[E]`：检索标签（用于 retriever 的监督）；
    - `num_nodes`，`node_global_ids[N]`，`node_embedding_ids[N]`，`topic_one_hot[N,num_topics]`。
  - 问题与锚点：
    - `question_emb[D]`，`question`；
    - `q_local_indices`（问题实体在局部图中的索引）；
    - `a_local_indices`（答案实体索引，若可达）；
    - `answer_entity_ids`（全局答案实体 ID，长度 `answer_entity_ids_len`）。
  - 监督路径：
    - `gt_paths_nodes`，`gt_paths_triples` 等，后续映射为局部路径监督。

这一层主要用于 retriever 训练，模型 `RetrieverModule` 在 $(G_i^{\text{retr}}, q_i)$ 上学习一个打分函数
$$
  f_{\text{retr}}(e \mid q_i, G_i^{\text{retr}}) \approx \log p_{\theta}^{\text{retr}}(y_e=1 \mid q_i, G_i^{\text{retr}}),
$$
输出每条边的分数 $\hat{s}_e$。

### 0.3 g_agent：从检索图到 GFlowNet 子图

第二层物化是 `g_agent`，它在 retriever 打分的基础上，为 GFlowNet 构造一个更小、更密集的子图：
$$
  G_i^{\text{agent}} = (V_i^{\text{agent}}, E_i^{\text{agent}}),
  \quad V_i^{\text{agent}} \subseteq V_i^{\text{retr}},
  \quad E_i^{\text{agent}} \subseteq E_i^{\text{retr}}.
$$

构造过程由 `src/data/components/g_agent_builder.py` 中的 `GAgentBuilder` 定义，可抽象为一个确定性算子：
$$
  \Phi_{\text{agent}}:\ (G_i^{\text{retr}}, \hat{s}_{i,\cdot}, S_i^{\text{raw}}, A_i^{\text{raw}}, \Gamma_i^{\text{raw}})
  \longmapsto \text{GAgentSample}_i,
$$
其中 `GAgentSample` 完整刻画了 $G_i^{\text{agent}}$ 以及其上的起点/答案/GT 路径。

更具体地，`GAgentBuilder.process_batch` 接收：

- 一个 retriever 的 PyG batch（来自 `GRetrievalDataset`）；
- 与之对齐的 `RetrieverOutput.scores`（每条检索边的分数 $\hat{s}_e$）；

对每个样本执行以下步骤（数学化的“Seed→Anchor 2-hop”）：

1. **构造局部图切片**  
   从 batch 中切出单样本的局部图 $G_i^{\text{retr}}$：
   $$
     \text{heads}_i,\ \text{tails}_i,\ \text{relations}_i,\ \text{labels}_i,\ \text{scores}_i,
   $$
   并保留 `node_global_ids` 与 `node_embedding_ids`。

2. **确定种子与 Anchor 结点**  
   - 种子集合 $S_i^{\text{retr}}$：来自 LMDB 的 `seed_entity_ids` 与 `q_local_indices` 映射到局部索引；
   - Anchor 集合：在所有边上按 $\hat{s}_e$ 排序，取前 $K$ 条（`anchor_top_k`，默认 50），将这些边的端点节点并为
     $$
       A_i^{\text{anchor}} := \{h_e, t_e \mid e \in \text{TopK}(\hat{s}_e)\}.
     $$

3. **两跳选边（Seed→Anchor）**  
   记 $E_i^{\text{retr}}$ 为该样本的所有局部边。
   - Hop-1：从每个种子 $s \in S_i^{\text{retr}}$ 出发，保留所有边
     $$
       (s, r, v) \in E_i^{\text{retr}}
     $$
     若 $v \in A_i^{\text{anchor}}$，直接加入候选边集合；否则将 $v$ 作为中间结点加入 frontier。
   - Hop-2：从每个 frontier 结点 $m$ 出发，保留所有边
     $$
       (m, r, t) \in E_i^{\text{retr}}
     $$
     若 $t \in A_i^{\text{anchor}}$，则将 Hop-1 中的 $(s,r,m)$ 与当前边 $(m,r,t)$ 一并纳入候选集合。  

   最终得到一个确定性的候选边集合
   $$
     \mathcal{E}_i^{\text{sel}} \subseteq E_i^{\text{retr}},
   $$
   其局部索引集合记为 `selected_indices`。

4. **重建子图拓扑（全局 ID → 子图局部 ID）**  
   记候选边的端点全局实体 ID 为
   $$
     \{h_e^{\text{global}}, t_e^{\text{global}} \mid e \in \mathcal{E}_i^{\text{sel}}\}.
   $$
   对其去重后生成局部节点集合 $V_i^{\text{agent}}$，并构建双射
   $$
     \pi_i: V_i^{\text{agent}} \to \{0,\dots, |V_i^{\text{agent}}|-1\}.
   $$
   然后将每条边映射为局部索引：
   $$
     e = (h,r,t) \mapsto (\pi_i(h), r, \pi_i(t)),
   $$
   得到
   $$
     E_i^{\text{agent}} = \{(\pi_i(h_e), r_e, \pi_i(t_e)) \mid e \in \mathcal{E}_i^{\text{sel}}\}.
   $$

5. **对齐起点/答案与 GT 路径**  
   - 起点：$S_i^{\text{raw}}$ 与 $V_i^{\text{agent}}$ 交集，映射为局部索引，形成
     $S_i^{\text{agent}} \subseteq V_i^{\text{agent}}$，即 `start_node_locals/start_entity_ids`；
   - 答案：$A_i^{\text{raw}}$ 与 $V_i^{\text{agent}}$ 交集，构成
     $A_i^{\text{agent}} \subseteq V_i^{\text{agent}}$，即 `answer_node_locals/answer_entity_ids`；
   - GT 路径：利用 LMDB 中的 `gt_path_edge_indices` 或 `gt_paths_triples`，对照 `selected_indices` 映射到
     $$
       \Gamma_i^{\text{agent}} \subseteq (E_i^{\text{agent}})^\ast,
     $$
     存入 `gt_path_edge_local_ids/gt_path_node_local_ids`。  
     若答案在子图中不可达或路径映射失败，则该样本被严格丢弃。

6. **落盘为 GAgentSample**  
   最终 `GAgentBuilder` 产生的 `GAgentSample` 精确对应“g_agent 规约”中的字段：
   - 拓扑与全局 ID：`node_entity_ids`, `edge_head_locals`, `edge_tail_locals`, `edge_relations`；
   - 检索分数与标签：`edge_scores`, `edge_labels`, `top_edge_mask`；
   - 起点/答案：`start_entity_ids/start_node_locals`, `answer_entity_ids/answer_node_locals`；
   - 监督路径：`gt_path_edge_local_ids`, `gt_path_node_local_ids`, `gt_path_exists`；
   - 可达性：`is_answer_reachable`（由构图逻辑严格保证）。

所有样本被打包到一个 `.pt` 文件中，由 `GAgentPyGDataset` 与 `GAgentDataModule` 在训练阶段读取。

### 0.4 GFlowNet 的输入视角

从 GFlowNet 的角度，**每个训练样本就是一个 g_agent 图**：
$$
  (q_i, G_i^{\text{agent}}, S_i^{\text{agent}}, A_i^{\text{agent}}, \Gamma_i^{\text{agent}}),
$$
其中：

- 问题向量 $q_i$ 来自 g_retrieval LMDB 的 `question_emb`（经 `GraphEmbedder` 投影）；
- 图 $G_i^{\text{agent}}$ 的拓扑和监督完全由 `GAgentSample` 决定；
- 后续所有章节中的 $G$、$S$、$A$、$\Gamma^{\text{GT}}$ 均指代这一层的对象。

---

## 1. 问题设定与记号

**输入：每个样本 $i$ 有**

- 问题文本及其嵌入：$q_i \in \mathbb{R}^{d_q}$（代码中为 `batch.question_emb`，经 `GraphEmbedder` 投影为 `question_tokens[i]`）。
- 检索子图（即 g_agent 图）：
  - 有限节点集合 $V_i$，每个节点为一个全局实体 ID。
  - 有向边集合 $E_i \subseteq V_i \times \mathcal{R} \times V_i$，每条边是 $(u, r, v)$。
  - 起点实体集合 $S_i \subseteq V_i$（seed / anchor，代码中 `start_node_locals/start_entity_ids`）。
  - 答案实体集合 $A_i \subseteq V_i$（`answer_node_locals/answer_entity_ids`）。
- 真实监督路径（若存在）：从一些起点 $s \in S_i$ 到一些答案 $a \in A_i$ 的路径集合 $\Gamma_i^{\text{GT}}$（`gt_path_edge_local_ids`）。

在一个样本内部，我们记

- $G = (V, E)$ 为该样本的局部图，
- $q$ 为问题嵌入，
- $S \subseteq V$ 为起点集合，
- $A \subseteq V$ 为答案集合。

**目标：** 在给定 $(q, G)$ 的条件下，学习一个在“终态”集合 $\mathcal{X}(q, G)$ 上的分布
$$
    \pi(x \mid q, G)
$$
使得
$$
    \pi(x \mid q, G) \propto R(x \mid q, G),
$$
其中 $x$ 是某条轨迹（从起点出发，有限步后选择 stop）所对应的终态结果（本实现中，终态由被选中的边集合决定）。

---

## 2. 轨迹空间与环境（GraphEnv）

### 2.1 状态与动作

在当前实现中，**环境状态**可以抽象为
$$
    s_t = (q, G, \mathcal{E}_t, v_t),
$$
其中：

- $\mathcal{E}_t \subseteq E$：截至步 $t$ 已被选中的边集合（代码中 `selected_mask`）。
- $v_t \in V$：当前位置（代码中 `current_tail[g]`，每个图一条轨迹）。
- 另外环境内部维护：
  - `visited_nodes`：已访问的节点集合，确保不重复访问（`forbid_revisit`）。
  - `prev_tail`：上一时刻节点，用于禁止回退边（`forbid_backtrack`）。

在每个状态 $s_t$，环境给出**合法动作集合**
$$
    \mathcal{A}(s_t) \subseteq E \cup \{\text{STOP}\},
$$
满足：

- 在 $t = 0$ 时（首步），$\mathcal{A}(s_0)$ 中仅包含**触及起点集合 $S$ 的边**以及 STOP：
  - 这在代码中由 `GraphEnv.action_mask_edges` 使用 `edge_starts_mask` 强制实现。
- 在 $t \ge 1$ 时：
  - 只允许连接当前节点 $v_t$ 的边；
  - 若 `forbid_revisit=True`，禁止到达已访问节点；
  - 若 `forbid_backtrack=True`，禁止立即退回 $v_{t-1}$；
  - STOP 仍然始终可选。

STOP 被实现为“每个图边段末尾的虚拟索引”，保证每个图上的动作空间是
$$
    \mathcal{A}(s_t) = \{\text{某图的边索引}\} \cup \{\text{该图的 STOP 索引}\}.
$$

### 2.2 轨迹与终态

在环境中，一条轨迹是动作序列
$$
    \tau = (a_0, a_1, \dots, a_T),
$$
其中：

- $a_t \in \mathcal{A}(s_t)$；
- 当某一步选中 STOP 或达到最大步数 $T_{\max}$ 时，轨迹终止。

令
$$
    \mathcal{E}_\tau = \{ e \in E \mid e\ \text{在}\ \tau\ \text{中被选择} \},
$$
则我们可以把**终态**定义为
$$
    x = x(\tau) := \mathcal{E}_\tau,
$$
其诱导的节点集合记为
$$
    V_\tau = \{ u \in V \mid \exists e = (u, r, v) \in \mathcal{E}_\tau \ \text{或}\ (v, r, u) \in \mathcal{E}_\tau \}.
$$

环境额外输出二值指标（代码中的）：

- `reach_success[g] = 1` 当且仅当 $V_\tau$ 与答案集合 $A$ 非空交；
- `length[g]` 为该图中轨迹采样的边数 $|\mathcal{E}_\tau|$。

---

## 3. 条件 GFlowNet：前向流、反向流与配分函数

### 3.1 前向策略 $P_F$

给定 $(q, G)$ 和环境状态 $s_t$，前向策略 $P_F$ 定义在 $\mathcal{A}(s_t)$ 上：
$$
    P_F(a_t \mid s_t, q, G; \theta).
$$

实现上：

- `GraphEmbedder` 把节点/关系/问题投影到统一维度 $H$：
  - 每条边 $e = (u, r, v)$ 对应向量 $\phi_e(e, q) \in \mathbb{R}^H$（`edge_tokens`）；
  - 问题向量 $\phi_q(q) \in \mathbb{R}^H$（`question_tokens`）；
  - 起点聚合向量 $\phi_{\text{start}}(S) \in \mathbb{R}^H$（`start_summary`）。
- 策略头（`EdgeMLPMixerPolicy` / `EdgeFrontierPolicy` / `EdgeGATPolicy`）在每一步用
  $$
    \text{inputs} = (\phi_e, \phi_q, \text{selected\_mask}, \text{current\_tail}, \dots)
  $$
  计算：
  $$
    \ell^{(t)}_e,\ \ell^{(t)}_{\text{STOP}} \quad \text{对每条候选边和 STOP 的 logits}.
  $$
- `GFlowNetActor` 再做：
  1. **检索软先验（可选）**：若配置 `retriever_prior_alpha = \alpha > 0`，则对每条边引入 retriever score 的对数偏置（默认对 `g_agent.edge_scores` 做截断，防止数值爆炸）：
     $$
       \text{prior}_e
       = \alpha \cdot \log\big(\max(\text{score}_e, 10^{-4})\big),
       \qquad
       \ell^{(t)}_e \leftarrow \ell^{(t)}_e + \text{prior}_e.
     $$
     在默认模型配置 `configs/model/gflownet_module.yaml` 与实验 `train_gflownet_default` 中，
     $$
       \alpha_{\text{train}}(t)
       = \alpha_{\text{start}}
         + (\alpha_{\text{end}} - \alpha_{\text{start}})
           \cdot \min\Big(1,\ \frac{t}{T_{\text{anneal}}}\Big),
     $$
     其中
     $$
       \alpha_{\text{start}} = 2.0,\quad
       \alpha_{\text{end}} = 0.5,\quad
       T_{\text{anneal}} = 10^4,
     $$
     即 retriever 先验在训练早期较强、随后线性退火到较弱水平；测试时使用当前时刻冻结后的 $\alpha_{\text{train}}$，
     仍然是**soft prior 而非硬约束**。
  2. **温度缩放**：$\tilde{\ell} = \ell / T$（训练用 `policy_temperature`，评估可用 `eval_policy_temperature`）。
  3. **掩码非法动作**：对 $\mathcal{A}(s_t)$ 外的边赋 $-\infty$（`LARGE_NEG`）。
  4. **混合探索**：将策略分布与均匀分布按 `random_action_prob` 线性混合。
  5. **Gumbel-Max 采样**（非贪心时）或贪心选择。

因此对每个图，$P_F$ 是在“受环境约束后的边+STOP”上的软最大分布。

轨迹的前向 log 概率为
$$
    \log P_F(\tau \mid q, G; \theta) = \sum_{t=0}^T \log P_F(a_t \mid s_t, q, G; \theta),
$$
实现中记为 `rollout["log_pf"]`。

### 3.2 反向策略 $P_B$：以终点为条件的局部流

反向策略的目标仍然是定义
$$
    P_B(s_{t-1} \mid s_t, q, G; \phi)
$$
使得 TB 方程中的反向流项由可学习网络给出。但在 hub-heavy 图上，我们不再使用 “tree / 常数 1” 的退化近似，而是显式刻画
$$
    P_B(u \mid v, q, G; \phi)
$$
作为对**同一终点 $v$ 的所有父边**的条件分布。

#### 3.2.1 记号与局部入度

对每条边 $e = (u, r, v) \in E$，定义：

- 源节点（source）$u$；
- 关系（relation）$r$；
- 目标节点（target）$v$。

记
$$
    \mathcal{I}(v) := \{ e = (u, r, v) \in E \}
$$
为所有“指向 $v$ 的边”集合，其基数
$$
    d_{\text{in}}(v) := |\mathcal{I}(v)|
$$
就是 $v$ 的入度。在 WebQSP 等数据上，$d_{\text{in}}(v)$ 呈现**长尾分布**，部分 hub 节点的入度超过 $10^3$。

在实现中：

- `edge_index[1]` 存储每条边的目标节点局部索引；
- 对某个图内的所有目标节点，$d_{\text{in}}(v)$ 由 `scatter_add` 在该轴上累加得到。

#### 3.2.2 均匀反向策略：$P_B^{\text{uniform}}$

最简单也最物理一致的基线是：对于每个目标节点 $v$，在其所有父边上取均匀分布
$$
    P_B^{\text{uniform}}(u \mid v, q, G)
    = \frac{1}{d_{\text{in}}(v)}, \qquad \forall e=(u,r,v) \in \mathcal{I}(v).
$$

在 log 域，对每条边 $e=(u,r,v)$：
$$
    \log P_B^{\text{uniform}}(u \mid v)
    = -\log d_{\text{in}}(v).
$$

实现对应于 `log_pb_mode="uniform"` 分支：

- 先用 `scatter_add` 计算每个目标节点的入度 $d_{\text{in}}(v)$；
- 再将 $\log P_B$ 写成上述 $-\log d_{\text{in}}(v)$，并在轨迹上通过 `selected_mask` 与 `edge_batch` 累加得到每个图的
  $$
      \log P_B^{\text{uniform}}(\tau) = \sum_{t=0}^T \log P_B^{\text{uniform}}(u_t \mid v_t).
  $$

这一步的关键是：相比旧的 “tree 模式 $\log P_B\equiv 0$”，它正确地反映了 hub 节点的**高入度会稀释反向概率**。  
例如，当某个节点的入度为 $1000$ 时，均匀反向给出 $\log P_B \approx -\log 1000 \approx -6.9$，而不再是错误的 $0$。

#### 3.2.3 学习式反向策略：$P_B^{\text{learned}}$

在更强的配置下，我们使用一个**以均匀为先验的学习式反向头**。  
实现中，`GFlowNetEstimator` 维护：

- 边嵌入 $\phi_e(e,q) \in \mathbb{R}^H$：由 `GraphEmbedder` 输出的 `edge_tokens`，已融合源节点和关系信息；
- 节点嵌入 $\phi_v(v,q) \in \mathbb{R}^H$：由 `GraphEmbedder` 输出的 `node_tokens`，在模型中用于 tail 节点；
- 问题嵌入 $\phi_q(q) \in \mathbb{R}^H$：`question_tokens`；
- 对每条边 $e=(u,r,v)$ 构造反向上下文：
  $$
      h_e
      = \operatorname{concat}\big(
          \phi_e(e,q),\ \phi_q(q),\ \phi_v(v,q)
        \big) \in \mathbb{R}^{3H}.
  $$

然后通过一个小型 MLP 生成边级打分：
$$
    s_e = f_{\text{B}}(h_e; \phi) \in \mathbb{R},
$$
其中 `backward_head` 的最后一层线性映射权重与偏置采用**零初始化**：
$$
    W_{\text{last}} = 0,\quad b_{\text{last}} = 0.
$$

这意味着在训练初期（参数尚未偏离 0 时），所有 $s_e \equiv 0$，从而对每个目标节点 $v$：
$$
    P_B^{\text{learned}}(u \mid v, q, G; \phi_{\text{init}})
    = \frac{\exp 0}{\sum_{e' \in \mathcal{I}(v)} \exp 0}
    = \frac{1}{d_{\text{in}}(v)},
$$
即**以均匀分布为严格先验**，后续训练只学习相对于均匀的“残差偏好”。

在实现上，我们使用 `scatter_log_softmax` 在目标节点轴上进行数值稳定的 log-softmax 计算：
$$
    \log P_B^{\text{learned}}(u \mid v, q, G; \phi)
    = \log\operatorname{softmax}\big( s_{e'} : e' \in \mathcal{I}(v) \big)_e.
$$

对一条轨迹 $\tau$，反向 log 概率为
$$
    \log P_B^{\text{learned}}(\tau \mid q, G; \phi)
    = \sum_{t=0}^T \log P_B^{\text{learned}}(u_t \mid v_t, q, G; \phi),
$$
由 `selected_mask` 与 `edge_batch` 聚合（`scatter_add`），最终得到每个图的标量 `log_pb`。

#### 3.2.4 反向头的训练目标与熵正则

当 `learn_pb=True` 时，我们对 $P_B$ 额外施加一个 NLL + 熵正则的损失：

1. **轨迹级 NLL**  
   我们期望在真实采样到的边上提高 $P_B$，因此对每个 batch 的每个图，定义
   $$
       \mathcal{L}_{\text{PB-NLL}}^{(b)}
       = - \sum_{e \in \mathcal{E}_\tau^{(b)}} \log P_B(e \mid \text{tail}(e), q^{(b)}, G^{(b)}; \phi),
   $$
   再在图维度上取平均：
   $$
       \mathcal{L}_{\text{PB-NLL}}
       = \frac{1}{B'} \sum_{b \in \mathcal{B}_{\text{has-edge}}} \mathcal{L}_{\text{PB-NLL}}^{(b)},
   $$
   其中 $\mathcal{B}_{\text{has-edge}}$ 为至少包含一条选中边的图集合。  
   实现中，这对应于 `pb_losses["pb_nll"]`。

2. **按目标节点的熵正则**  
   对每个目标节点 $v$，$P_B(\cdot \mid v)$ 是一个离散分布。我们定义其熵为
   $$
       H(v)
       = - \sum_{e \in \mathcal{I}(v)}
           P_B(e \mid v) \log P_B(e \mid v).
   $$

   在 hub-heavy 图上，如果不加约束，模型很容易在某些入度极大的节点上将概率质量坍缩到单条边（mode collapse），从而失去探索性。  
   因此我们引入一个鼓励较高熵的正则项：
   $$
       \mathcal{L}_{\text{PB-Ent}}
       = - \lambda_{\text{ent}} \cdot
         \frac{1}{|\mathcal{V}_{\text{used}}|}
         \sum_{v \in \mathcal{V}_{\text{used}}} H(v),
   $$
   其中 $\mathcal{V}_{\text{used}}$ 为本 batch 中真实出现过的目标节点集合（实现中通过 `scatter_sum` 和 `unique` 计算）。  
   代码里，这对应 `pb_entropy_coef = \lambda_{\text{ent}} > 0` 和 `pb_losses["pb_entropy"]`，并额外记录平均熵 `pb_avg_entropy` 用于监控。

   在数值实现中，熵项采用
   $$
       -P_B \log P_B
   $$
   的形式，并使用 `torch.nan_to_num` 把理论极限 $0 \cdot (-\infty)$ 安全归约为 $0$，以避免浮点 NaN。

3. **可选 L2 正则**  
   为了进一步限制反向头的复杂度，可以对其参数施加 L2 正则：
   $$
       \mathcal{L}_{\text{PB-L2}}
       = \lambda_{\text{L2}} \sum_{w \in \theta_{\text{backward}}} \|w\|_2^2,
   $$
   对应配置中的 `pb_l2_reg` 和实现中的 `pb_losses["pb_l2"]`。

因此，当 `log_pb_mode="learned", learn_pb=True` 时，反向头的总损失为
$$
    \mathcal{L}_{\text{PB}}
    = \mathcal{L}_{\text{PB-NLL}}
      + \mathcal{L}_{\text{PB-Ent}}
      + \mathcal{L}_{\text{PB-L2}}.
$$

### 3.3 配分函数 $Z(q, G)$

GFlowNet 需要一个标量 $Z(q, G) > 0$（配分函数），使得目标分布满足
$$
    \sum_{x \in \mathcal{X}(q, G)} R(x \mid q, G) = Z(q, G),
    \quad \pi(x \mid q, G) = \frac{R(x \mid q, G)}{Z(q, G)}.
$$

在实现中：

- 首先利用起点聚合和问题向量构造上下文：
  $$
    c(q, G) = \psi_{\text{ctx}}\big(\phi_{\text{start}}(S),\ \phi_q(q)\big),
  $$
  代码为 `estimator.build_context(start_summary, question_tokens)`。
- 然后通过 `log_z_head` 输出
  $$
    \log Z(q, G; \psi) = f_Z(c(q,G)),
  $$
  即每个图一个标量，代码为 `log_z`。

---

## 4. 奖励函数 $R(x \mid q, G)$

### 4.1 答案命中与覆盖率

给定一条轨迹 $\tau$ 及其终态 $x(\tau)$，我们在答案集合 $A$ 与轨迹访问到的节点集合
$$
    V_\tau \subseteq V
$$
上定义若干统计量，这些量既用于构造奖励，也用于评估指标。

1. **是否命中答案**（命中至少一个答案实体）：
   $$
       \text{hit}(\tau) =
       \begin{cases}
       1, & V_\tau \cap A \neq \varnothing, \\
       0, & \text{否则}.
       \end{cases}
   $$
   这在实现中对应环境输出的 `reach_success`。

2. **答案覆盖率**（命中的答案实体占全部答案实体的比例）：
   $$
       \text{reach\_frac}(\tau)
       = \frac{|V_\tau \cap A|}{|A|} \in [0,1],
   $$
   在代码中由 `GFlowNetModule._compute_answer_reach_fraction` 计算，用作 reward shaping 和日志统计。

3. **答案级 Precision / Recall（评估指标）**  
   令
   $$
       H(\tau) := V_\tau \cap A, \quad
       N_{\text{visit}}(\tau) := |V_\tau|, \quad
       N_{\text{ans}} := |A|.
   $$
   则 AnswerOnly/AnswerDiffusion 奖励模块内部计算的“答案精度/召回”为
   $$
       \text{answer\_precision}(\tau)
       = \frac{|H(\tau)|}{\max(1, N_{\text{visit}}(\tau))},
       \qquad
       \text{answer\_recall}(\tau)
       = \frac{|H(\tau)|}{\max(1, N_{\text{ans}})}.
   $$
   这反映了“访问到的节点中有多大比例是答案”（密度）和“答案集合被覆盖得多完整”，对应 `RewardOutput.answer_precision/answer_recall/answer_f1`，仅用于监控，不参与 TB 损失。

此外，为了监控 GFlowNet 对 GT 路径的覆盖情况，`g_agent` 中还存有

- GT 边掩码 `path_mask` 与存在标志 `path_exists`，
- 从而可以定义“预测路径 vs GT 路径”的 Precision/Recall/F1 与 full-hit 指标（代码中的 `gt_path_precision/gt_path_recall/gt_path_f1/gt_path_full_hit`）。

### 4.2 AnswerOnlyReward（默认配置）：二元奖励 + 可选覆盖率 shaping

默认实验 `train_gflownet_default` 使用 `AnswerOnlyReward` 作为奖励模块，配置：

- `success_reward = 10.0`；
- `failure_reward = 0.01`；
- `lambda_reach = 0.0`（即当前默认**关闭覆盖率 shaping**，只看是否命中答案）。

`AnswerOnlyReward` 在 log 域定义奖励，并对最大可能奖励做归一化。

1. **基础二元奖励（当前实际生效部分）**  
   对每条轨迹 $\tau$：
   $$
       \log R_{\text{base}}(\tau) =
       \begin{cases}
       \log r_{\text{succ}}, & \text{hit}(\tau) = 1, \\
       \log r_{\text{fail}}, & \text{hit}(\tau) = 0,
       \end{cases}
   $$
   其中 `success_reward = r_{\text{succ}} > 0`，`failure_reward = r_{\text{fail}} > 0`，通常 $r_{\text{succ}} \gg r_{\text{fail}}$。

2. **覆盖率 shaping（默认关闭，需 `lambda_reach>0` 才启用）**  
   若配置 `lambda_reach = \lambda_{\text{reach}} \ge 0`，额外加入一项
   $$
       \log R_{\text{reach}}(\tau)
       = \lambda_{\text{reach}} \cdot \text{reach\_frac}(\tau) \cdot \mathbf{1}\{\text{hit}(\tau)=1\}.
   $$
   直观上，这奖励那些命中了答案且覆盖率更高的轨迹。

3. **组合并在 log 域归一化**  
   先组合未归一化的 log 奖励：
   $$
       \log R_{\text{raw}}(\tau)
       = \log R_{\text{base}}(\tau)
         + \lambda_{\text{reach}} \cdot \text{reach\_frac}(\tau) \cdot \mathbf{1}\{\text{hit}(\tau)=1\}.
   $$
   然后减去一个全局常数，使“理论上最大奖励”在 log 域为 0：
   $$
       \log R_{\max}^{\text{AO}}
       = \log r_{\text{succ}} + \max(0, \lambda_{\text{reach}}),
   $$
   $$
       \log \tilde{R}(\tau)
       = \log R_{\text{raw}}(\tau) - \log R_{\max}^{\text{AO}}.
   $$
   实现中再通过
   $$
       R(x(\tau)\mid q,G) = \exp(\log \tilde{R}(\tau))
   $$
   得到数值稳定的正奖励（并在代码里用 `clamp(min=1e{-8})` 防止数值下溢）。

最终传入 TB 的 log 奖励为
$$
    \log R(x(\tau) \mid q, G)
    := \log\tilde{R}(\tau),
$$
对应 `reward_out.log_reward`；$R(x(\tau)\mid q,G)$ 对应 `reward_out.reward`。

### 4.3 AnswerDiffusionReward（已与 AnswerOnly 合并）

早期曾在 AnswerOnly 奖励上尝试引入 answer_gravity（答案条件扩散势能），但在实际实验中未观察到收益，默认配置也一直把 `lambda_gravity` 置为 0。  
为减少无效分支与超参，本实现已移除 answer_gravity，`AnswerDiffusionReward` 仅作为 `AnswerOnlyReward` 的兼容别名保留；奖励 shaping 仅由 `success_reward`、`failure_reward` 与 `lambda_reach` 控制。

---

## 5. Trajectory Balance（TB）损失

### 5.1 TB 方程

对任意一条终止于 $x(\tau)$ 的轨迹 $\tau$，Trajectory Balance 要求
$$
    \log Z(q, G) + \sum_{t=0}^T \log P_F(a_t \mid s_t, q, G)
    = \log R(x(\tau) \mid q, G) + \sum_{t=0}^T \log P_B(s_{t-1} \mid s_t, q, G).
$$

令
$$
    \delta(\tau; q, G)
    := \log Z(q, G)
       + \sum_{t=0}^T \log P_F(a_t \mid s_t, q, G)
       - \sum_{t=0}^T \log P_B(s_{t-1} \mid s_t, q, G)
       - \log R(x(\tau) \mid q, G),
$$
则理想情况下对所有轨迹 $\delta(\tau; q, G) = 0$。

### 5.2 训练目标

在本实现中，主损失为
$$
    \mathcal{L}_{\text{TB}}
    = \mathbb{E}_{(q,G),\ \tau \sim P_F(\cdot \mid q, G)}
      \big[\, \delta(\tau; q, G)^2 \,\big],
$$
代码中对应：

```python
tb_loss = torch.mean((log_z + rollout["log_pf"] - log_pb - log_reward) ** 2)
```

其中：

- `log_z` 是 $\log Z(q, G)$；
- `rollout["log_pf"]` 是 $\sum_t \log P_F(a_t \mid s_t)$；
- `log_pb` 是 $\sum_t \log P_B(s_{t-1} \mid s_t)$；
- `log_reward` 是 $\log R(x(\tau)\mid q,G)$。

当 `learn_pb=True` 时，还叠加反向头的 NLL：
$$
    \mathcal{L}_{\text{PB}}
    = - \mathbb{E}_{(q,G),\tau}
        \big[\, \tfrac{1}{|\mathcal{E}_\tau|} \sum_{t} \log P_B(s_{t-1} \mid s_t) \,\big],
$$
即实现中的 `pb_nll`。

总体 TB+PB 损失为
$$
    \mathcal{L}_{\text{main}} = \mathcal{L}_{\text{TB}} + \mathcal{L}_{\text{PB}}.
$$

---

## 6. GT Teacher Forcing（轨迹监督）

当 g_agent 样本中存在 GT 路径（`path_exists=True` 且 `gt_path_edge_local_ids` 非空）时，本实现可以进行**部分 teacher forcing**：

- 对部分图（按 `gt_replay_ratio` 采样），在 roll-out 时强制动作序列沿着某条 GT 边序列行走；
- 这为这些图提供了额外的对数概率监督。

形式上，对这类图的 GT 轨迹 $\tau^{\text{GT}}$，增加一项：
$$
    \mathcal{L}_{\text{GT}}
    = - \mathbb{E}_{(q,G),\ \tau^{\text{GT}}}
        \big[ \log P_F(\tau^{\text{GT}} \mid q, G) \big],
$$
实现中对应于 `gt_loss_value = -log_pf_gt[mask_gt].mean()`。

最终总损失为
$$
    \mathcal{L}
    = \mathcal{L}_{\text{TB}} + \mathcal{L}_{\text{PB}} + \mathcal{L}_{\text{GT}}.
$$

---

## 7. 算法级伪代码

用数学视角给出每个 batch 的训练过程：

1. **批量输入**
   - 从 `GAgentDataModule` 取一个扁平 PyG batch，包含 $(q^{(b)}, G^{(b)}, A^{(b)}, S^{(b)}, \Gamma_{\text{GT}}^{(b)})$，$b=1,\dots,B$。

2. **图与问题嵌入（GraphEmbedder）**
   - 利用全局实体/关系向量与 retriever projector，得到：
     - 边嵌入 $\phi_e^{(b)}(e)$（`edge_tokens`）；
     - 问题嵌入 $\phi_q^{(b)}$（`question_tokens`）；
     - 起点聚合 $\phi_{\text{start}}^{(b)}$（`start_summary`）。

3. **配分函数估计（Estimator.log_z）**
   - 计算
     $$
        c^{(b)} = \psi_{\text{ctx}}\big(\phi_{\text{start}}^{(b)}, \phi_q^{(b)}\big), \quad
        \log Z^{(b)} = f_Z(c^{(b)}).
     $$

4. **多次 roll-out（Actor + Env）**
   - 对每个 batch 执行 $K$ 次（训练时 $K=1$，评估时 $K>1$）：
     1. 根据 `gt_replay_ratio` 和 `path_exists` 决定哪些图使用 GT teacher forcing；
     2. 在环境中从起点状态 $s_0^{(b)}$ 出发，根据 $P_F$ 采样轨迹 $\tau_k^{(b)}$，得到：
        - 前向 log 概率 $\log P_F(\tau_k^{(b)})$；
        - 选中边集合 $\mathcal{E}_{\tau_k^{(b)}}$（`selected_mask`）；
        - 成功标志与长度等统计。

5. **奖励计算（Reward）**
   - 对每条轨迹求 $R\big(x(\tau_k^{(b)}) \mid q^{(b)}, G^{(b)}\big)$，得到：
     - $\log R_k^{(b)}$（`log_reward`）；
     - 各种评估指标（成功率、答案 F1、GT 路径 F1 等）。

6. **反向概率与 TB 残差（Estimator.log_pb + TB）**
   - 用 `GFlowNetEstimator.log_pb` 在选中边上聚合，得到 $\log P_B(\tau_k^{(b)})$ 及 NLL；
   - 对每个图、每次 roll-out 计算 TB 残差
     $$
       \delta_k^{(b)}
       = \log Z^{(b)}
         + \log P_F(\tau_k^{(b)})
         - \log P_B(\tau_k^{(b)})
         - \log R_k^{(b)}.
     $$

7. **损失聚合与反向传播**
   - 聚合得到 $\mathcal{L}_{\text{TB}}$、$\mathcal{L}_{\text{PB}}$、$\mathcal{L}_{\text{GT}}$，
   - 求和为总损失 $\mathcal{L}$，对 $(\theta,\phi,\psi)$ 做一次梯度更新。

---

## 8. 数学对象与实现字段对齐表

这一节把上文的数学对象与代码字段一一对应，方便在阅读源码时保持“方程视角”：

- 条件变量：
  - $q$：`batch.question_emb` → `GraphEmbedder._prepare_question_tokens` → `question_tokens`.
  - $G=(V,E)$：
    - $V$：`batch.node_global_ids`，PyG 扁平节点索引由 `batch.ptr` 给出每图切片；
    - $E$：`batch.edge_index`，关系 ID 为 `batch.edge_attr`，标签为 `batch.edge_labels`。
- 起点与答案：
  - $S$：`batch.start_node_locals` / `start_entity_ids`；
  - $A$：`batch.answer_node_locals` / `answer_entity_ids`；
  - GT 路径集合 $\Gamma^{\text{GT}}$：`batch.gt_path_edge_local_ids`。
- 轨迹：
  - $\mathcal{E}_\tau$：`rollout["selected_mask"]`；
  - 当前位置 $v_t$：`GraphState.current_tail`；
  - 是否命中答案：`rollout["reach_success"]`。
- 概率与配分函数：
  - $\log P_F(\tau)$：`rollout["log_pf"]`；
  - $\log P_B(\tau)$：`log_pb`（`GFlowNetEstimator.log_pb` 输出）；
  - $\log Z(q,G)$：`estimator.log_z(estimator.build_context(...))`。
- 奖励：
  - $\log R(x(\tau) \mid q,G)$：`reward_out.log_reward`；
  - $R(x(\tau) \mid q,G)$：`reward_out.reward`。
- TB 残差与损失：
  - $\delta(\tau)$：`log_z + rollout["log_pf"] - log_pb - log_reward`；
  - $\mathcal{L}_{\text{TB}}$：上述残差的均方；
  - $\mathcal{L}_{\text{PB}}$：`pb_nll`；
  - $\mathcal{L}_{\text{GT}}$：`gt_loss_value`，即 `-log_pf_gt` 在有 GT 的图上的平均。

从这个对齐表出发，你可以把 `src/models/gflownet_module.py` 看作对上文方程的直接向量化实现，而各组件

- `GraphEmbedder`：实现 $\phi_e, \phi_q, \phi_{\text{start}}$；
- `GraphEnv`：实现状态转移 $s_t \to s_{t+1}$ 与合法动作集合 $\mathcal{A}(s_t)$；
- `GFlowNetActor` + `policy`：实现 $P_F$；
- `GFlowNetEstimator`：实现 $P_B$ 与 $Z$；
- `Reward*`：实现 $R$，

共同构成了一个在 $(q,G)$ 条件下，满足 Trajectory Balance 的 GFlowNet 采样器。

---

## 9. 训练输出与评估指标（System 1.5 → System 2）

最后，用一个从“分布”到“可观测量”的视角收束全局：

- 给定样本 $(q_i, G_i^{\text{agent}})$，训练后的 GFlowNet 定义了一个轨迹分布
  $$
    \pi_\theta(\tau \mid q_i, G_i^{\text{agent}})
  $$
  其诱导终态分布满足 $\pi_\theta(x) \propto R(x \mid q_i, G_i^{\text{agent}})$。
- 在实现中，**一次 roll-out 就是一次从该分布的采样近似**：`actor.rollout(...)` 返回的
  `selected_mask`/`actions` 就是样本化的子图与路径。

在训练与评估阶段，这个分布被以两种方式观测。

### 9.1 单次采样：每条轨迹上的行为

- 对每个样本只采样一条轨迹 $\tau$；
- 记录：
  - `rollout_reward`：$R(x(\tau))$ 的 batch 平均；
  - `success_mean`：$\mathbb{E}[\mathbf{1}\{\text{命中答案}\}]$；
  - `answer_f1`：在 $V_\tau$ 上的答案级 F1；
  - `path_hit_precision` / `path_hit_recall` / `path_hit_f1`：相对于 GT 路径掩码的精度/召回/F1。
- 这些量在 Lightning 日志中以 `train/*`、`val/*`、`test/*` 形式出现，是优化与早停的直接依据。

更形式化地，对单条轨迹 $\tau$：

- 命中指标：
  $$
    \text{success}(\tau)
    = \mathbf{1}\{V_\tau \cap A \neq \varnothing\}.
  $$
- GT 路径精度/召回/F1（记 $\mathcal{E}_{\text{GT}}$ 为 GT 边集合）：
  $$
    \text{prec}_{\text{path}}(\tau)
    = \frac{|\mathcal{E}_\tau \cap \mathcal{E}_{\text{GT}}|}{\max(1,|\mathcal{E}_\tau|)},
    \quad
    \text{rec}_{\text{path}}(\tau)
    = \frac{|\mathcal{E}_\tau \cap \mathcal{E}_{\text{GT}}|}{\max(1,|\mathcal{E}_{\text{GT}}|)},
  $$
  $$
    \text{F1}_{\text{path}}(\tau)
    = \frac{2\cdot \text{prec}_{\text{path}}(\tau)\cdot \text{rec}_{\text{path}}(\tau)}
           {\text{prec}_{\text{path}}(\tau)+\text{rec}_{\text{path}}(\tau)+\varepsilon}.
  $$

`GFlowNetModule` 在训练阶段直接对这些单轨迹指标取 batch 均值作为 `*_mean`。

### 9.2 多次采样（Best-of-K / Any-of-K）：跨轨迹聚合指标

在验证/测试阶段，根据 `evaluation_cfg.num_eval_rollouts = K_s`，对同一问题独立采样 $K_s$ 条轨迹
$\{\tau^{(1)},\dots,\tau^{(K_s)}\}$。需要区分几类指标。

1. **Best-of-K 成功率（success@K）**

   定义：
   $$
     \text{success@}K_s
     = \mathbb{E}\Big[ \mathbf{1}\Big\{ \exists k,\ V_{\tau^{(k)}} \cap A \neq \varnothing \Big\} \Big],
   $$
   即“至少一条轨迹命中答案”的概率。实现中对应 `success@K`，由
   $K_s$ 次 roll-out 中按图维度 `any` 后再在图维度做平均得到。

2. **路径命中 Any-of-K（path_hit_any@K）**

   对于每个样本，记
   $$
     \text{hit\_GT}(\tau^{(k)})
     = \mathbf{1}\{\mathcal{E}_{\tau^{(k)}} \cap \mathcal{E}_{\text{GT}} \neq \varnothing\},
   $$
   则
   $$
     \text{path\_hit\_any@}K_s
     = \mathbb{E}\Big[ \mathbf{1}\Big\{ \exists k,\ \text{hit\_GT}(\tau^{(k)}) = 1 \Big\} \Big],
   $$
   对应实现中的 `path_hit_any@K`。

3. **答案 Any-of-K 与覆盖率（answer_hit_any@K, answer_recall_union@K）**

   - 定义答案命中：
     $$
       \text{hit\_ans}(\tau^{(k)})
       = \mathbf{1}\{V_{\tau^{(k)}} \cap A \neq \varnothing\}.
     $$
   - Any-of-K 命中率：
     $$
       \text{answer\_hit\_any@}K_s
       = \mathbb{E}\Big[ \mathbf{1}\Big\{ \exists k,\ \text{hit\_ans}(\tau^{(k)}) = 1 \Big\} \Big].
     $$
   - 并集覆盖率：记
     $$
       H_{\cup}
       = \bigcup_{k=1}^{K_s} \big(V_{\tau^{(k)}} \cap A\big),
     $$
     则
     $$
       \text{answer\_recall\_union@}K_s
       = \mathbb{E}\Big[ \frac{|H_{\cup}|}{\max(1,|A|)} \Big],
     $$
     对应实现中的 `answer_recall_union@K`。

4. **Top-K 边窗口上的路径 F1（path_hit_f1@K_p）**

   给定 `path_hit_k = K_p`，在每条轨迹上按选择顺序取前 $K_p$ 条边，定义窗口内的 GT 路径精度/召回/F1：
   $$
     \text{prec}_{\text{path}}^{(K_p)}(\tau)
     = \frac{|\mathcal{E}_\tau^{(\le K_p)} \cap \mathcal{E}_{\text{GT}}|}
            {\max\big(1, |\mathcal{E}_\tau^{(\le K_p)}|\big)},
   $$
   $$
     \text{rec}_{\text{path}}^{(K_p)}(\tau)
     = \frac{|\mathcal{E}_\tau^{(\le K_p)} \cap \mathcal{E}_{\text{GT}}|}
            {\max\big(1, |\mathcal{E}_{\text{GT}}|\big)},
   $$
   $$
     \text{F1}_{\text{path}}^{(K_p)}(\tau)
     = \frac{2\cdot \text{prec}_{\text{path}}^{(K_p)}(\tau)\cdot \text{rec}_{\text{path}}^{(K_p)}(\tau)}
            {\text{prec}_{\text{path}}^{(K_p)}(\tau)+\text{rec}_{\text{path}}^{(K_p)}(\tau)+\varepsilon}.
   $$

   实现中在**每次 roll-out 内**先对上述量在图维度取平均，得到
   `path_hit_precision@K_p` / `path_hit_recall@K_p` / `path_hit_f1@K_p`，再在 roll-out 维度上平均，
   不是 Any-of-K，而是**“窗口内 F1 的期望”**。  
   `logging_cfg.auto_add_path_hit_f1=true` 时，会自动把 `path_hit_f1@max(path_hit_k)` 加入验证/测试的进度条。

5. **模式多样性与 logP–logR 对齐**

   在评估阶段，`GFlowNetModule` 额外记录：

   - `modes_found`：每个图中不同答案实体的命中个数的平均值；
   - `modes_recall`：$|H_{\cup}|/|A|$ 的另一种视角，与 `answer_recall_union@K` 等价；
   - `unique_paths`：每个图中唯一路径签名的个数，用来衡量模式多样性；
   - `logpf_logr_corr`：$\log P_F(\tau)$ 与 $\log R(x(\tau))$ 之间的 Pearson 相关系数；
   - `logpf_logr_spearman`：两者排名之间的 Spearman 相关系数。

   数学上，理想的 Trajectory Balance 收敛状态应满足
   $$
     \log P_F(\tau) \approx \log R(x(\tau)) + \log P_B(\tau) - \log Z,
   $$
   因此在固定 $P_B, Z$ 后，$\log P_F$ 与 $\log R$ 应高度相关；上述两个相关系数是这一性质的直接观测。

这些指标在配置中由：

- `configs/model/gflownet_module.yaml` 中的 `evaluation_cfg.num_eval_rollouts` 与 `evaluation_cfg.path_hit_k`；
- `logging_cfg` 中的 `train_prog_bar` / `eval_prog_bar`（以及自动追加的 `success@K`、`path_hit_f1@K`）

控制其采样次数与显示方式。

在完整的 EVI-RAG 流水线中：

- retriever（System 1）提供边级打分 $f_{\text{retr}}$ 与 g_retrieval/g_agent 缓存；
- GFlowNet（System 1.5）在 `g_agent` 子图上学习一个**可采样的后验分布**，把“哪些边/路径是高质量证据”的不确定性显式编码为 $\pi_\theta$；
- LLM 生成模块（System 2）再利用这些采样到的子图（或其排序截断版本）构造提示，完成最终的自然语言回答。

从数学角度看，本仓库实现的是以下链式因子化（默认配置下 reward 使用 AnswerOnly，soft prior 使用较弱的 retriever 先验）：
$$
  p_\theta(\text{answer} \mid q)
  \approx
  \sum_{G^{\text{agent}}}
  p_{\text{build}}(G^{\text{agent}} \mid q)\,
  \sum_{x}
  \underbrace{\pi_\theta(x \mid q, G^{\text{agent}})}_{\text{GFlowNet}}\,
  p_{\text{LLM}}(\text{answer} \mid q, x),
$$
其中 $p_{\text{build}}$ 由 g_raw→g_retrieval→g_agent 的构图算子决定，  
而 GFlowNet 负责在给定构图结果的条件下，学习一个与奖励 $R(x \mid q, G^{\text{agent}})$ 成正比的 $\pi_\theta$。
