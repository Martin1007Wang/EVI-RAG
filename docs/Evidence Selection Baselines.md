## 第二个独立测评：Evidence Selection Baselines（BFS/Beam vs GFlowNet Rollout）

本文档是对 `docs/Semantic Dissipation.md` 的**正交补充**：前者给出“检索→接口→推理”的因果账本与耗散分解；本文只关心其中第一段的一个更细粒度问题：

> 在固定上下文预算与固定图预算下，**证据选择算子**如何影响可见证据质量与推理上限？为什么需要 GFlowNet？

结论先行：
1) **节点级 BFS**（只做可达性/最短路）不会组合爆炸；  
2) 会爆炸的是**路径级枚举**（需要输出“多条候选路径/子图证据”且推理时未知答案），这才是与 GFlowNet Rollout 可比的 baseline；  
3) GFlowNet 的价值是把“在巨大路径空间里找高奖励轨迹”的代价**摊销到训练**，推理时用固定预算采样获得更高的 `success@K / answer_hit_any@K`，从而在同样窗口预算下提升 `S_ret^{vis}` 与 `Acc_hit`。

---

### 0. 统一对象：把“证据生成”视为算子
对每个样本 \(i\)，我们把 retriever 产生的子图（或其派生缓存）视为给证据选择器的输入：

- \(G_i=(V_i,E_i)\)：retriever 子图（例如 Top-\(K_e\) 边，经 `g_agent` 物化后的局部图）。
- \(S_i\subseteq V_i\)：起点集合（seed entities，对应 `start_node_locals`）。
- \(\mathcal{A}_i\subseteq V_{\text{global}}\)：答案实体集合（**推理时不可用**；离线评测时可用）。

定义一个通用的证据选择算子（可确定/可随机）：
$$ \mathcal{G}_\theta:\ (q_i, G_i, S_i)\ \mapsto\ \mathcal{E}_i $$
其中 \(\mathcal{E}_i\) 可以是：
- **边集合**（triplets）：\(\mathcal{E}_i \subseteq E_i\)
- **路径集合**：\(\mathcal{E}_i = \{P_{i,1},\dots,P_{i,m}\}\)，每条路径是边序列

后续所有 baseline（Retriever-TopK / BFS / Beam / GFlowNet）只是不同的 \(\mathcal{G}_\theta\) 实例；评测必须在同一预算与同一接口定义下进行，否则比较不成立。

---

### 1. 关于“组合爆炸”：节点 BFS vs 路径枚举
常见误解是“200 条边的图 BFS 会爆炸”。需要区分两种 BFS：

#### 1.1 节点级 BFS（可达性/最短路）
状态是“节点”，每条边最多被扫描一次：
$$T = O(|V|+|E|)$$
在 \(E\approx 200\) 的规模下，它几乎不可能成为瓶颈。

但它回答的问题也很弱：**能不能到达**、或**最短步数是多少**。如果推理时不知道答案，它无法告诉你“该输出哪些路径作为证据”。

#### 1.2 路径级枚举（把“部分路径”当状态）
当你需要输出“若干候选路径/证据子图”（供 LLM 或后续模块），队列里存的是一条条**部分路径**。若平均分支数为 \(b\)，最大步数为 \(L\)，潜在路径数近似：
$$\#\text{paths}\ \approx\ O(b^L)$$
这才是组合爆炸的来源，也是 GFlowNet 试图解决的核心困难：在不知道答案的情况下，你需要用有限预算从指数级的路径空间里抽取高质量轨迹。

---

### 2. Baseline 家族（必须区分 Oracle 与 Non-oracle）
为了展示 GFlowNet 的必要性，baseline 必须分两类：

#### 2.1 Oracle baselines（只做上界/诊断，**不可部署**）
这类方法显式使用 \(\mathcal{A}_i\)（答案）或 GT path 来停止/筛选，因此是上界：
- **Oracle shortest-path**：从 \(S_i\) 到 \(\mathcal{A}_i\cap V_i\) 的最短路径
- **Oracle k-shortest/beam-to-answer**：输出多条到答案的高分路径

用途：测“子图是否包含可达证据”与“在固定约束下的理论上限”，但不能与部署系统混用。

#### 2.2 Non-oracle baselines（可部署，推理时不看答案）
这些才是与 GFlowNet 可公平比较的 baseline：

1) **Retriever-TopK（边集合基线）**  
直接取 Top-\(K_e\) 边（或其 union/dedup），不显式建路径结构。

2) **BFS-frontier（节点级扩展 → 选边）**  
从 \(S_i\) 做深度限制 BFS 到 \(L\)，收集被访问到的边，再按某种启发式选出 \(K_e\) 条边作为证据（例如按 retriever score 排序、或按层次优先）。
它不爆炸，但通常“噪声大”：可达边集合迅速增大，窗口预算下易稀释关键边。

3) **Beam search（路径级近似枚举）**  
维护宽度 \(W\) 的 partial paths，根据路径分数（例如 \(\sum \log \text{edge\_score}\)）扩展到最大步数 \(L\)，得到 top-\(M\) 完整路径，再把这些路径的边并集作为证据。
它是“路径级 baseline”的标准做法，也是最容易展示“没有学习时必须强剪枝”的对照组。

4) **GFlowNet Rollout（学习到的路径分布采样）**  
学习策略 \(\pi_\theta(P\mid q_i,G_i,S_i)\)，推理时采样 \(K_s\) 条轨迹（每条长度 \(\le L\)），把轨迹边并集（或 top-\(K_e\) 截断）作为证据。
它用训练把“剪枝规则”从手工启发式变成可学习的、query-conditioned 的分布。

> 重要：为了可比，Non-oracle baselines 应尽量匹配环境约束（例如 `forbid_revisit/forbid_backtrack/max_steps`），否则你在解的是另一个问题。

---

### 3. 公平性：预算如何对齐
比较必须显式对齐两个预算，否则结论不可信：

1) **图预算（Graph budget）**：最终进入证据池的边数 \(K_e\)（或等价的 token 预算下的 \(K_{eff}\)）  
2) **搜索预算（Search budget）**：BFS/Beam 的 expansion 次数 vs GFlowNet 的 rollouts 次数 \(K_s\) 与最大步数 \(L\)

推荐的最小公平协议：
- 固定 `max_steps = L`（与环境一致）
- 固定证据池大小 `K_e`（或固定 token budget \(B\)，再由模板决定等效 \(K_{eff}\)）
- 对 Beam：报告 \(W\)、展开节点/边数（真实计算代价）
- 对 GFlowNet：报告 \(K_s\)、平均轨迹长度（真实计算代价）

---

### 4. 指标：两层评测（无 LLM / 有 LLM）

#### 4.1 无 LLM（只评“证据选择质量”）
这层用来直接展示 GFlowNet 的采样优势，避免 LLM 噪声掩盖信号：
- `success@K`：\(K_s\) 次 rollouts/paths 中是否至少命中一个答案实体（离线用 \(\mathcal{A}_i\) 判断）
- `answer_hit_any@K`、`answer_recall_union@K`：多答案覆盖的并集指标
- `path_hit_any@K` / `path_hit_f1@k`：若存在 GT path，则评测与 GT 的重合（注意这不是 reward 必然优化的目标）

#### 4.2 有 LLM（回到 Semantic Dissipation 的账本）
把每个 baseline 产出的证据池当作 \(R_i\)，复用 `docs/Semantic Dissipation.md` 的定义：
- \(S_{ret}^{vis}(K,B,\phi)\)、\(Acc_{hit}\)、\(\mathcal{D}_{rate}\)、\(\mathcal{D}_{mass}\)

这层用来回答：“同样 token 预算下，GFlowNet 是否提升可见证据与利用率？”

---

### 5. 在本仓库中的落点（数据/产物）
本项目中，GFlowNet 训练与评测消费的是 `g_agent` 缓存（唯一持久化 schema）：
- 生成：`stage=materialize_g_agent` → `${dataset.materialized_dir}/g_agent/<split>_g_agent.pt`
- GFlowNet 评测：`stage=gflownet_eval` → `${dataset.materialized_dir}/eval_gflownet/test_gflownet_eval.pt`
- Retriever 评测缓存：`stage=retriever_eval` → `${dataset.materialized_dir}/eval_retriever/test_retriever_eval.pt`

建议把 BFS/Beam baselines 的产物也做成“可解析缓存”，并沿用相同的下游评测接口（truth/LLM stages 只看缓存，不关心生成器来自哪里）。这样才符合“算子/变量分离”的可组合性。

---

### 6. 推荐呈现（最能体现 GFlowNet 的必要性）
建议至少做两张图 + 一张表：

1) **固定证据预算（\(K_e\) 或 \(B\)），扫搜索预算**：  
横轴 \(K_s\)（rollouts）或 Beam 宽度 \(W\)，纵轴 `answer_hit_any@K / success@K`。  
你会看到：Beam 在小预算下要么性能差，要么必须爆算力；GFlowNet 在固定 \(K_s\) 下更快贴近上界。

2) **固定搜索预算，扫 token 预算 \(B\)**：  
对比 `S_ret^{vis}` 与 `Acc_hit`，展示“噪声稀释/截断”如何让 BFS-frontier 退化，而 GFlowNet 的集中采样更稳。

3) **表：可部署性与代价**：  
列出是否 oracle、是否使用答案、平均展开边数/耗时、最终证据边数、关键指标。

