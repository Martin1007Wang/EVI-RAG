# System Protocol: The GFlowNet Architect

**Role:** 深度学习架构师 (Deep Learning Architect)
**Core Competency:** Mathematical Completeness & Engineering Rigor
**Mantra:** "Code is the executable projection of mathematical truth."

---

## 0. Anchor Semantics (QA vs Flow)

*   **Data SSOT:** `q_local_indices` / `a_local_indices`（及其 ptr）仅表示**问题/答案实体集合**，是数据层唯一真相，严禁被覆盖或互换。
*   **Flow-Derived:** `start_node_locals` / `target_node_locals` 仅表示 **GFlowNet 流向起终点**，由 `(q, a) + direction + selector/sampling` 派生，运行时存在，不落盘。
*   **Direction Mapping:** `forward`: start = q, target = a；`backward`: start = a, target = q。
*   **No Swapping:** 禁止通过交换 `q/a` 实现反向流；必须通过显式的 flow 映射/override。
*   **Mask Semantics:** `dummy_mask` 仅由 data-level answer 计算；`node_is_target` 必须由 flow target 计算。

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
*   $G_{env} \subseteq G_{sub}$: 代理交互的封闭环境（运行时从 `g_retrieval` 派生，不落盘）。
*   $s \in \mathcal{V}$: 起点 (Start Node)。
*   $a \in \mathcal{V}$: 答案/终点 (Answer Node)。
*   $\tau$: 轨迹/路径 (Trajectory)。
*   $\mathcal{P}_{min}(s, a)$: 从 $s$ 到 $a$ 的所有**最短**路径集合。
*   $\text{CSR}$: Compressed Sparse Row 格式，此处特指利用 `counts` 和 `cumsum` 实现的变长数据存储。

---

## Ⅲ. Data Specification (数据立法)

### 1. `g_retrieval` Schema (GFlowNet Subgraph SSOT)

**Definition:** $G_{sub}$ 是以 $s$ 为中心的 PPR 采样结果。它是 GFlowNet 训练的唯一输入。
**Note:** 当前训练管线是 label-free；任何监督字段（如 `labels`, `pair_*`）出现即视为数据错误。
**Storage Note:** `g_retrieval` 仅物化 core LMDB（`<split>.lmdb`）。不读取/不维护 aux LMDB。

#### A. Topology (流形结构)
*   `sample_id`: `str`. Unique Identifier.
*   `num_nodes`: `int` ($N$).
*   `edge_index`: `Long[2, E]`. Local indices $u \to v$.
*   `edge_attr`: `Long[E]`. Global Relation IDs.
*   `node_global_ids`: `Long[N]`. Map local $i \to$ Global ID.
*   `node_embedding_ids`: `Long[N]`. Embedding lookup table index.
*   `node_type_counts`: `Long[N]`. Per-node type counts (CSR header).
*   `node_type_ids`: `Long[sum(node_type_counts)]`. Flattened type ids.

#### B. Semantics & Condition (语义与条件)
*   `question_emb`: `Float[1, D]`. Dense query representation.
*   `q_local_indices`: `Long[K_q]`. $s$ 在 $G_{sub}$ 中的索引。
*   `a_local_indices`: `Long[K_a]`. $a$ 在 $G_{sub}$ 中的索引。
*   `answer_entity_ids`: `Long[K_a]`. Global Answer IDs (用于 Metrics).
*   `question`: `Optional[str]`. 仅用于日志/可视化；可缺省。

#### C. Forbidden Legacy Fields (禁止出现)
*   `labels`, `pair_*`, `answer_subgraph`, `topic_pe`, `node_topic_dist` 等历史字段必须在预处理阶段剔除。

---

### 2. Runtime Contract (Current Pipeline)
*   **Training:** 仅提供 GFlowNet 训练（`configs/experiment/train_gflownet.yaml`，`override /data: g_retrieval`）；训练阶段仅消费 `g_retrieval`（LMDB）。
*   **Evaluation/Reasoning:** 评估/推理产物仅使用 `eval_gflownet` 缓存；不生成/不读取 `g_agent`。
*   **Shortcut Suppression/SP-Dropout:** 训练阶段默认启用 SP-Dropout（预选遮蔽）与 Safety Net（最短路保底），仅作用于 `g_retrieval` 子图采样。

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
*   **Training Scope:** GFlowNet 的训练 **只能** 使用 `sub` 数据集。
*   **Evaluation Scope:** GFlowNet 的评估 **必须同时** 在 `full` 与 `sub` 两套数据集上进行，并分别报告指标（建议 `full`/`sub` 作为显式前缀）。
*   **LLM Evaluation:** LLM 评估 **必须同时** 在 `full` 与 `sub` 两套数据集上进行。
*   **Runtime Contract:** 任何评估流程必须显式提供两套数据源（`full` 与 `sub`），不得隐式复用或覆盖。

### 4. Forbidden Patterns (反模式 - 此处即为红线)
*   ❌ **Hardcoding Special Cases:** 不要写 `if dataset == 'fq': ...`。应当将差异抽象为配置参数。
*   ❌ **Pre-computing Embeddings in Forward:** 预处理逻辑属于 Dataset，不属于 Model Forward。
*   ❌ **Python Loops for Graph Logic:** 使用 `scatter`, `gather`, `index_select` 代替循环。

---

## Ⅴ. Policy Semantics (行动语义)

### 1. Action Definition (动作定义)
*   **Factorized Edge Policy:** 前向策略使用关系/实体的可微因子分解：
    \[
    P_F(e \mid s_t, Q)=P_R(r \mid s_t, Q)\cdot P_E(v \mid s_t, r, Q)
    \]
*   **Soft Weighting (Training):** 训练期采用软重加权（关系分布与实体分布均可微），避免硬采样导致的梯度断裂。
*   **Hard Selection (Inference-Optional):** 推理期允许先采样关系再采样实体，但训练期默认不使用硬切断。

### 2. Termination Rule (终止规则)
*   **No Explicit STOP Action:** 终止由条件触发：命中答案、无出边或达到最大步数。
*   **Reward Semantics:** 成功路径 $R=1$（$\log R=0$）；失败路径 $R=\epsilon$（$\log R \approx -C$）。

### 3. Backward Policy Contract (反向策略契约)
*   **Uniform Backward:** 反向策略固定为均匀分布：
    \[
    P_B(s_t \mid s_{t+1}) = \frac{1}{\deg_{\text{in}}(s_{t+1})}
    \]
*   **Log Form:** 训练中累积 $\log P_B = -\log \deg_{\text{in}}$，不引入可学习参数。

### 4. Multi-Start Handling (多起点处理)
*   **Set Semantics:** `q_local_indices` 表示完整起点集合，严禁覆盖或互换。
*   **Single-Start Trajectory:** 每条轨迹仅选择一个起点；当前实现对集合**均匀采样**。
*   **Fail Fast:** 若未解析出有效起点，立即抛错。

### 5. Replay Invariance (回放不变性)
*   **Local Edge IDs:** Replay 必须以 per-graph local edge id 存储；通过 `edge_ptr` 在 add/fetch 时映射。

### 6. Multi-Endpoint Reality (多终点现实)
*   **Multi-Start & Multi-Target:** 数据可能同时包含多个起点与多个终点；在反向流中亦然。
*   **No Pairwise Connectivity Guarantee:** 起点与终点两两配对不保证连通；可达性必须由数据过滤或奖励函数显式处理，模型不得隐式假设全连通。

---

## Ⅵ. Known Limitation (已知局限)

*   **Subgraph-bounded supervision:** 当前训练/评估均在 `g_retrieval` 子图上进行（动作空间由采样与掩码裁剪），最短路监督与奖励都定义在该裁剪子图内，而非全图最短路。**原因**：$G_{raw}$ 极其庞大，$G_{sub}$ 也很大（平均约 10k 条边），直接在该规模上训练/推理不可行；因此必须裁剪动作空间。若需要第一性“全图”语义，必须改变构图策略或在全图上计算监督。
