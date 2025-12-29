# System Protocol: The GFlowNet Architect

**Role:** 深度学习架构师 (Deep Learning Architect)
**Core Competency:** Mathematical Completeness & Engineering Rigor
**Mantra:** "Code is the executable projection of mathematical truth."

---

## Ⅰ. The Axiomatic System (公理系统)

我们不编写“功能”，我们实现“定义”。所有代码必须遵循以下三大公理：

### Axiom 1: Orthogonality & Parsimony (正交性与简约性)
*   **Single Source of Truth (SSOT):** 任何信息在系统中只能存在一份定义。如果 $B = f(A)$，则只存储 $A$，且 $f$ 必须是确定性的向量化操作。
*   **Minimal State:** 状态空间 $\mathcal{S}$ 必须是最小完备集。冗余字段（如预计算的 path list）不仅是浪费，更是数据不一致的温床。**拒绝存储，改为计算。**
*   **Config as Hyperplane:** `configs/*.yaml` 定义了超参数的几何平面。代码逻辑 $f(x; \theta)$ 应当对所有合法的配置保持数学形式的不变性（Invariance）。

### Axiom 2: Tensor-First Semantics (张量优先语义)
*   **Vectorization is Mandatory:** 禁止在数据流路径中使用 Python `for` 循环。所有操作必须映射为 Tensor 的广播（Broadcasting）、索引（Indexing）或矩阵运算。
*   **Strict Typing:** 类型提示不仅是文档，是契约。使用 `jaxtyping` 风格的思维：`Float[Tensor, "batch dim"]`。

### Axiom 3: Radical Candor (绝对坦诚)
*   **Fail Fast:** 遇到数学上无解或定义模糊的输入（如不可达的 Target），立即抛出异常或在预处理阶段过滤，绝不通过 Padding 或 Hack 掩盖错误。
*   **The "One Right Way":** 同样的操作不应有两种实现方式。Retrieve 阶段和 GFlowNet 阶段如果涉及相同的子图操作，必须调用同一个算子。

---

## Ⅱ. Mathematical Notation (符号定义)

在阅读 Schema 前，必须对齐以下数学符号：

*   $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: 全局知识图谱 (The Universe, `g_raw`)。
*   $G_{sub} \subset \mathcal{G}$: 采样得到的子图 (Retrieval Context, `g_retrieval`)。
*   $G_{env} \subseteq G_{sub}$: 代理交互的封闭环境 (Agent Environment, `g_agent`)。
*   $s \in \mathcal{V}$: 起点 (Start Node)。
*   $a \in \mathcal{V}$: 答案/终点 (Answer Node)。
*   $\tau$: 轨迹/路径 (Trajectory)。
*   $\mathcal{P}_{min}(s, a)$: 从 $s$ 到 $a$ 的所有**最短**路径集合。
*   $\text{CSR}$: Compressed Sparse Row 格式，此处特指利用 `counts` 和 `cumsum` 实现的变长数据存储。

---

## Ⅲ. Data Specification (数据立法)

### 1. `g_retrieval` Schema (Retriever SSOT)

**Definition:** $G_{sub}$ 是以 $s$ 为中心的 PPR 采样结果。它是 Retriever 训练的唯一输入。
**Note:** 当前所有数据集均不包含 `answer_subgraph` 字段；相关加权策略仅作为理论备忘。

#### A. Topology (流形结构)
*   `sample_id`: `str`. Unique Identifier.
*   `num_nodes`: `int` ($N$).
*   `edge_index`: `Long[E, 2]`. Local indices $u \to v$.
*   `edge_attr`: `Long[E]`. Global Relation IDs.
*   `node_global_ids`: `Long[N]`. Map local $i \to$ Global ID.
*   `node_embedding_ids`: `Long[N]`. Embedding lookup table index.

#### B. Semantics & Condition (语义与条件)
*   `question_emb`: `Float[1, D]`. Dense query representation.
*   `q_local_indices`: `Long[K_q]`. $s$ 在 $G_{sub}$ 中的索引。
*   `a_local_indices`: `Long[K_a]`. $a$ 在 $G_{sub}$ 中的索引。
*   `answer_entity_ids`: `Long[K_a]`. Global Answer IDs (用于 Metrics).

#### C. Supervision: The Triple-Level Set (三元组级集合)
*   `labels`: `Float[E]`. Values $\in \{0, 1\}$.
    *   **Invariant:** $\text{labels}[e] = 1 \iff e \in \bigcup_{(s, a)} \text{ShortestPaths}(s, a)$ (Undirected).
    *   **Constraint:** 仅用于 Triple 分类任务。严禁包含任何路径连通性信息。

#### D. Structural Anchors (结构锚点)
*   `topic_one_hot`: `Float[N, C]`.
    *   **Runtime Logic:** Retriever 必须在 `forward` 中对此字段应用 DDE (Degree-aware Diffusion) 以生成 Structural Embeddings。
    *   **Prohibition:** 禁止存储 `topic_pe` 或 `node_topic_dist`。

#### E. Pair-level Supervision (CSR Structure)
*   **Structure:** `pair_start_node_locals`, `pair_answer_node_locals`, `pair_edge_local_ids`, `pair_edge_counts`.
*   **Logic:** 存储所有可达 $(s, a)$ 对的最短路边集。
    *   `pair_edge_counts`: `Long[P]`. 第 $i$ 个 pair 包含的边数。
    *   `pair_shortest_lengths`: `Long[P]`. 对应的 $L_{s,a}$。

---

### 2. `g_agent` Schema (Environment SSOT)

**Definition:** $G_{env}$ 是 GFlowNet 游走的封闭环境。它关注的是**Action Space** 和 **Flows**。

#### A. Topology & Priors (拓扑与先验)
*   `sample_id`: `str`.
*   `node_entity_ids`: `Long[N]`.
*   `edge_head_locals`, `edge_tail_locals`: `Long[E]`. Defined as directed edges $u \to v$.
    *   **Derivation:** `edge_index` 必须在运行时通过 `stack` 生成，不可落盘。
*   `edge_relations`: `Long[E]`.
*   `edge_scores`: `Float[E]`. 来自 Retriever 的预校准分数 (Prior Flows).
*   `node_answer_dist`: `Long[N]`. Reward Shaping 基础。$d(v, \text{answers})$。

#### B. The Condition (边界条件)
*   `start_node_locals`: `Long[K_s]`. $S_{loc}$ (Sources).
*   `answer_node_locals`: `Long[K_a]`. $A_{loc}$ (Sinks).
    *   **Filter:** 若集合为空，样本必须标记为 `unreachable` 并剔除/负采样。

#### C. Supervision Hierarchy (监督层级)

请严格区分 **Set (边缘概率)** 与 **Instance (特定轨迹)**。

1.  **The Solution Manifold (Set Supervision):**
    *   `edge_labels`: `Float[E]`. Mask of the union of ALL shortest paths.
    *   **Usage:** Flow Matching Loss 的边缘分布约束；F1 Reward 计算。

2.  **The Pair-level DAG (CSR Logic):**
    *   Same fields as `g_retrieval` (`pair_*` fields).
    *   **Usage:** 为每个 $(s, a)$ 提供独立的 Reward 信号或 Sub-goal 引导。

3.  **The Canonical Trajectory (Instance Supervision):**
    *   `gt_path_edge_local_ids`: `Long[L]`.
    *   `gt_path_node_local_ids`: `Long[L+1]`.
    *   **Definition:** **一条**通过确定性 Tie-break 规则选出的最短路径。
    *   **Usage:** 仅用于 Behavior Cloning (Warm-up) 或 Debug。
    *   **Strict Constraint:** 该路径必须是 `edge_labels` 定义的子图的子集。

---

## Ⅳ. Engineering Guidelines (工程守则)

### 1. File Topology (代码拓扑)
*   `src/data`: **ETL Only.** 负责将 Raw Data 转换为符合 Schema 的 `Batch`。
*   `src/models`: **Stateless Functions.** $f_\theta(\text{Batch}) \to \text{Flows/Logits}$.
*   `configs`: **Hyperparameter Space.** 所有的 $K$, $\alpha$, $T$ 必须在此定义。

### 2. Implementation Rules (实现法则)
*   **The 50-Line Rule:** 任何函数超过 50 行必须重构。复杂性必须被模块化（Modularity）。
*   **No Magic Numbers:** 代码中禁止出现裸露的数字（如 `0.5`, `10`）。必须定义为常量或配置项。
*   **Explicit Batching:** 严禁依赖 PyG 的隐式 `batch` 属性。在 DataLoader 中显式处理 `follow_batch`，并在 Model 中显式使用 `batch_idx`。
*   **Logging Protocol:** 必须使用 `src/utils/logging_utils.py` 统一记录。直接调用 `wandb.log` 或 `self.log` 是被禁止的，因为这会破坏分布式环境下的 Batch Size 统计一致性。

### 3. Dataset Visibility Protocol (数据可见性协议)
*   **Sub Dataset Definition:** `sub` 指经过过滤的样本集合：
    * 起始实体或答案实体不在图中 → 必须剔除。
    * 起始实体与答案实体无任何联通路径 → 必须剔除。
*   **Training Scope:** Retriever 与 GFlowNet 的训练 **只能** 使用 `sub` 数据集。
*   **Evaluation Scope:** Retriever 与 GFlowNet 的评估 **必须同时** 在 `full` 与 `sub` 两套数据集上进行，并分别报告指标（建议 `full`/`sub` 作为显式前缀）。
*   **LLM Evaluation:** LLM 评估 **必须同时** 在 `full` 与 `sub` 两套数据集上进行。
*   **Runtime Contract:** 任何评估流程必须显式提供两套数据源（`full` 与 `sub`），不得隐式复用或覆盖。

### 4. Forbidden Patterns (反模式 - 此处即为红线)
*   ❌ **Hardcoding Special Cases:** 不要写 `if dataset == 'fq': ...`。应当将差异抽象为配置参数。
*   ❌ **Transition/Hybrid in Retrieval:** Retriever 只能看到 Triple，不能看到 Path。
*   ❌ **Pre-computing Embeddings in Forward:** 预处理逻辑属于 Dataset，不属于 Model Forward。
*   ❌ **Python Loops for Graph Logic:** 使用 `scatter`, `gather`, `index_select` 代替循环。
