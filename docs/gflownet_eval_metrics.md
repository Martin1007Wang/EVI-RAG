# GFlowNet 评估指标与理论上限（WebQSP）

聚焦策略输出的边/节点有效性、多样性与分布一致性，列出定义、理论上限和可参考的检索基线。

## 指标定义（对齐“给 LLM 的 K 条候选路径”）
- 单条路径质量（诊断用，按 roll-out 均值）
  - `rollout_success`：单次采样命中答案的比率。
  - `path_hit_precision/recall/f1`：该路径选中边与 GT 路径边的命中。
  - `answer_precision/recall/f1`、`answer_coverage`：该路径选中边端点覆盖答案节点。
  - `rollout_length`、`avg_step_entropy`：路径长度与策略熵。
- 候选集合质量（LLM 关心，Best-of/Union over K = num_eval_rollouts）
  - `success@K`：K 条候选中至少一条命中答案（Best-of-K）。
  - `path_hit_any@K`：K 条候选中是否至少一次命中 GT 边（仅对可达样本）。
  - `answer_hit_any@K`：K 条候选的节点并集是否命中任一答案（仅对有答案且可达的样本）。
  - `answer_recall_union@K`：K 条候选的节点并集覆盖答案节点的比例。
- Top-K（单条路径前 K 条边的窗口诊断）
  - `path_hit_precision@K` / `path_hit_recall@K` / `path_hit_f1@K`：一条路径按选择顺序截取前 K 条边，对 GT 边的命中（仅对可达样本）。
- 多样性（K 条候选的去重）
  - `modes_found` / `modes_recall`：命中的不同答案实体数及其召回。
  - `unique_paths`：唯一路径数（有序边序列去重）。
- 分布一致性（分布匹配诊断）
  - `logpf_logr_corr`：log P 与 log R 的 Pearson 相关。
  - `logpf_logr_spearman`：log P 与 log R 的 Spearman 排名相关。

## 理论上限（数据无关的上界）
- `success@K` / `path_hit_any@K` / `answer_hit_any@K`：上限 1.0，且随 K 单调不减。
- `answer_recall_union@K` / `answer_*`：上限 1.0（并集/单条路径完全覆盖答案）。
- `path_hit_*`：上限 1.0（GT 边全命中且无多余）。
- `modes_found` 上限为该样本答案数；`modes_recall` 上限 1.0。
- `unique_paths` 上限为该图内可行路径总数（理论 >1）。
- 相关性：`logpf_logr_corr` / `spearman` 上限 1.0（完美分布匹配）。

## 当前评估体系的检索上界（两版）
> 指标与评估系统一致：success/answer_hit_any/answer_recall_union/path_hit_any 以候选集合（Best-of/并集）计分；path_hit_precision/recall/f1@K 为诊断。  
> 计算方式：脚本 `scripts/analyze_gflownet_data.py` 基于缓存 `edge_scores`，并屏蔽不可达/无答案样本。

1) **检索贪心版（单步，起点约束，edge_score 降序）**  
   相当于一次性选前 K 条起点可达边，不走多步。

**WebQSP train (n=1956)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9172 | 0.9172 | 0.5900 | 0.6871 | 0.6871 | 0.6871 | 0.6871 |
| 5  | 0.9335 | 0.9335 | 0.8234 | 0.9254 | 0.1853 | 0.9254 | 0.3087 |
| 10 | 0.9351 | 0.9351 | 0.8693 | 0.9586 | 0.0965 | 0.9586 | 0.1752 |
| 20 | 0.9361 | 0.9361 | 0.8991 | 0.9744 | 0.0503 | 0.9744 | 0.0953 |

**WebQSP val (n=170)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9059 | 0.9059 | 0.5352 | 0.5235 | 0.5235 | 0.5235 | 0.5235 |
| 5  | 0.9235 | 0.9235 | 0.8066 | 0.8824 | 0.1765 | 0.8824 | 0.2941 |
| 10 | 0.9294 | 0.9294 | 0.8604 | 0.9176 | 0.0923 | 0.9176 | 0.1677 |
| 20 | 0.9294 | 0.9294 | 0.8931 | 0.9471 | 0.0489 | 0.9471 | 0.0928 |

**WebQSP test (n=1113)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9146 | 0.9146 | 0.5658 | 0.5804 | 0.5804 | 0.5804 | 0.5804 |
| 5  | 0.9425 | 0.9425 | 0.8208 | 0.8697 | 0.1749 | 0.8697 | 0.2909 |
| 10 | 0.9443 | 0.9443 | 0.8727 | 0.9191 | 0.0937 | 0.9191 | 0.1692 |
| 20 | 0.9488 | 0.9488 | 0.9031 | 0.9506 | 0.0505 | 0.9497 | 0.0948 |

2) **Beam 贪心版（多步，环境约束，beam=K，max_steps=6）**  
   模拟与评估环境一致的“起点首步 + 禁回退/重访”约束，使用 edge_score 作为贪心打分，选 score 最高的 K 条路径，按并集/Best-of 计分。

**WebQSP train (n=1956)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9238 | 0.9238 | 0.6111 | 0.6943 | 0.4779 | 0.6943 | 0.5366 |
| 5  | 0.9489 | 0.9489 | 0.7454 | 0.8124 | 0.6628 | 0.8124 | 0.6997 |
| 10 | 0.9586 | 0.9586 | 0.8007 | 0.8569 | 0.7233 | 0.8569 | 0.7554 |
| 20 | 0.9693 | 0.9693 | 0.8460 | 0.8829 | 0.7586 | 0.8829 | 0.7885 |

**WebQSP val (n=170)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9118 | 0.9118 | 0.5509 | 0.5353 | 0.3588 | 0.5353 | 0.4049 |
| 5  | 0.9235 | 0.9235 | 0.7056 | 0.7176 | 0.5646 | 0.7176 | 0.6020 |
| 10 | 0.9471 | 0.9471 | 0.7733 | 0.7765 | 0.6364 | 0.7765 | 0.6695 |
| 20 | 0.9471 | 0.9471 | 0.8115 | 0.8176 | 0.6800 | 0.8176 | 0.7133 |

**WebQSP test (n=1113)**

| K | success@K≤ | answer_hit_any@K≤ | answer_recall_union@K≤ | path_hit_any@K≤ | path_hit_precision@K≤ | path_hit_recall@K≤ | path_hit_f1@K≤ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1  | 0.9236 | 0.9236 | 0.5916 | 0.5831 | 0.3811 | 0.5831 | 0.4343 |
| 5  | 0.9461 | 0.9461 | 0.7287 | 0.7206 | 0.5740 | 0.7206 | 0.6091 |
| 10 | 0.9560 | 0.9560 | 0.7840 | 0.7673 | 0.6350 | 0.7673 | 0.6661 |
| 20 | 0.9668 | 0.9668 | 0.8277 | 0.8095 | 0.6864 | 0.8095 | 0.7150 |

## 检索基线（历史值，供对照）
> 训练/推理已不再直接使用 retriever 的 edge_scores，以下表格仅作历史对照。**表格不包含新引入的集合指标 `answer_hit_any@K` / `answer_recall_union@K`，这些需要重跑评估后补充数值。**

| split | answer_precision | answer_recall | answer_f1 | path_hit_precision | path_hit_recall | path_hit_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| validation (n=170) | 0.1799 | 0.8958 | 0.2432 | 0.0489 | 0.9471 | 0.0928 |
| test (n=1113) | 0.1769 | 0.9130 | 0.2383 | 0.0505 | 0.9488 | 0.0947 |

说明：检索基线仅刻画“跟随检索排序”能到的下界/上限，真实评估以 roll-out 指标为准。检索排序上限（path_recall@k 等）若需，可另查，但不影响策略评估。

## 数据集上限（检索排序视角，供定位检索瓶颈）
基于 g_agent 缓存对 GT 路径边排名的上限（与策略无关）：

| split | retriever_path_recall@1 | @5 | @10 | @20 |
| --- | --- | --- | --- | --- |
| validation (n=170) | 0.5294 | 0.8824 | 0.9176 | 0.9471 |
| test (n=1113) | 0.5804 | 0.8688 | 0.9173 | 0.9488 |

含义：假设策略完美遵循检索排序，路径命中可达到的理论上限。若 roll-out 指标远低于此，问题可能在策略/训练；若两者都低，问题在检索/数据。

基于检索分数 Top-K 选边的路径命中（纯检索 greedy，对应 path_hit_* 的上限）：

**validation (n=170)**

| K | path_hit_precision@K | path_hit_recall@K | path_hit_f1@K |
| --- | --- | --- | --- |
| 1  | 0.5294 | 0.5294 | 0.5294 |
| 5  | 0.1765 | 0.8824 | 0.2941 |
| 10 | 0.0923 | 0.9176 | 0.1677 |
| 20 | 0.0489 | 0.9471 | 0.0928 |

**test (n=1113)**

| K | path_hit_precision@K | path_hit_recall@K | path_hit_f1@K |
| --- | --- | --- | --- |
| 1  | 0.5804 | 0.5804 | 0.5804 |
| 5  | 0.1748 | 0.8688 | 0.2906 |
| 10 | 0.0935 | 0.9173 | 0.1689 |
| 20 | 0.0505 | 0.9488 | 0.0947 |

含义：用检索分数选前 K 条边，计算与 GT 边的精确率/召回/F1，作为 path_hit_precision/recall/f1 的检索上限对照。

## Easy / Hard 子集：GFlowNet 相对 Beam Search 的增益（WebQSP, max_steps=3）

### 切分方式（基于 g_agent 与 Beam 上限）

- 数据来源：`test_g_agent.pt`（WebQSP test, n=1113），由 `GAgentBuilder` 生成，包含：
  - `edge_scores`（retriever 分数）、`gt_path_edge_local_ids`（GT 路径边）、`is_answer_reachable` / `gt_path_exists` 等字段。
- Beam 贪心上限（多步、环境约束）：
  - 使用 `scripts/analyze_gflownet_data.py` 中的 `_beam_greedy_metrics`。
  - 配置：`beam_k = 5`，`max_steps = 3`（与 WebQSP 的 `GraphEnv.max_steps` 对齐），`mode=subgraph`、起点首步+禁回退/重访。
  - 对每个样本计算 `beam_success = success@K=5 ∈ {0,1}`。
- 可达性判定：
  - `answerable := is_answer_reachable AND gt_path_exists`。
- 子集定义（由 `scripts/build_g_agent_difficulty_split.py` 实现）：
  - **Unreachable**：`answerable == False`（当前 WebQSP test 中为 0）。
  - **Easy**：`answerable == True` 且 `beam_success == 1`。
  - **Hard**：`answerable == True` 且 `beam_success == 0`。
- 结果（WebQSP test, n=1113）：
  - Easy：1045 个样本。
  - Hard：68 个样本。
  - Unreachable：0 个样本。

脚本 `scripts/build_g_agent_difficulty_split.py` 会输出：
- JSON：记录每个 `sample_id` 的分组、`beam_success`、GT edge rank 等。
- 子集缓存：`easy_g_agent.pt` / `hard_g_agent.pt`，记录字段与原始 g_agent 完全一致，仅样本子集不同。

### GFlowNet 在 Easy / Hard 上的表现（成功率与路径命中）

评估设置：
- 配置：`configs/experiment/eval_gflownet_webqsp_hard.yaml`（模型同 WebQSP 主实验，`num_eval_rollouts = [1, 5, 10, 20]`）。
- 环境：`max_steps = 3`，与 Beam 上限脚本一致。
- 指标：`success@K`、`path_hit_any@K`、`answer_hit_any@K`、`answer_recall_union@K` 等，均为 Best-of-K / Union 统计。

#### Easy 子集（Beam 成功的“顺风局”）

定义：Beam Search (K=5, max_steps=3) 成功命中答案的样本（1045 条）。

**代表性指标（WebQSP test, Easy 子集）**

| 指标                        | K=1      | K=5      | K=10     | K=20     |
|---------------------------|----------|----------|----------|----------|
| `success@K`               | 0.9789   | 0.9904   | 0.9904   | 0.9914   |
| `answer_hit_any@K`       | 0.9789   | 0.9904   | 0.9904   | 0.9914   |
| `answer_recall_union@K`  | 0.6006   | 0.7327   | 0.7814   | 0.8150   |
| `path_hit_any@K`         | 0.5828   | 0.7493   | 0.8019   | 0.8440   |
| `path_hit_f1@K`          | 0.5769   | 0.5770   | 0.5770   | 0.5770   |

其他：

| 指标                 | 数值    |
|----------------------|---------|
| `answer_f1`          | 0.4748 |
| `path_hit_f1`        | 0.5770 |
| `modes_recall`       | 0.8150 |
| `unique_paths`       | 2.71  |

**解读：**
- 在检索已给出强信号的样本上，GFlowNet 仍保持接近 0.99 的 `success@K`，说明策略没有因采样引入明显退化。
- 但路径多样性显著不足（`unique_paths ≈ 2.7`，`modes_recall ≈ 0.815`），说明 roll-out 温度/entropy 可能过低，采样过早塌缩到头部路径，需要在探索力度上补偿。

#### Hard 子集（Beam 失败的“逆风局”）

定义：`answerable == True` 但 Beam Search (K=5, max_steps=3) 失败（`beam_success == 0`），共 68 条样本。

**代表性指标（WebQSP test, Hard 子集）**

| 指标                        | K=1      | K=5      | K=10     | K=20     |
|---------------------------|----------|----------|----------|----------|
| `success@K`               | 0.1765   | 0.1912   | 0.1912   | 0.2059   |
| `answer_hit_any@K`       | 0.1765   | 0.1912   | 0.1912   | 0.2059   |
| `answer_recall_union@K`  | 0.0739   | 0.0979   | 0.1125   | 0.1229   |
| `path_hit_any@K`         | 0.1176   | 0.1912   | 0.2647   | 0.3382   |
| `path_hit_f1@K`          | 0.1228   | 0.1228   | 0.1228   | 0.1228   |

其他：

| 指标                 | 数值    |
|----------------------|---------|
| `answer_f1`          | 0.0590 |
| `path_hit_f1`        | 0.1228 |
| `modes_recall`       | 0.1229 |
| `unique_paths`       | 6.69  |

注意：Hard 子集按照定义，**Retriever + Beam Search (K=5)** 在这些样本上的 `success@5` 为 0（即 Beam 无法命中答案）。
在同一 g_agent 图和环境约束下，GFlowNet 通过 Best-of-10 采样达到：
- `success@10 ≈ 0.191`：约五分之一的困难样本被成功“翻盘”。
- `path_hit_any@10 ≈ 0.265`：在约四分之一的困难样本上，至少有一条采样路径完整命中 GT 边。

**解读：**
- Easy 集展示的是“锦上添花”：在检索已经可靠的区域，GFlowNet 维持高命中，但当前采样多样性不够，需调优温度/熵或 rollout 数以避免过度贪婪。
- Hard 集展示的是“雪中送炭”：在 Beam Search 判定失败的样本上，GFlowNet 将成功率从 0 提升到 ~0.2，GT 路径命中率到 ~0.34：
  - 说明策略依旧能从 retriever 低分尾部挖掘有效轨迹，但覆盖度明显低于前一版结果，提示训练或探索强度不足。
  - `path_hit_any@K` 高于 `success@K` 依然成立，意味着即使停点未对齐答案节点，轨迹中出现的 GT 边仍可为 LLM 提供推理证据。

综上：
- **Easy 子集**验证：GFlowNet 保持高命中，但采样多样性塌缩，应优先调参恢复探索。
- **Hard 子集**提供：在 retriever + Beam 完全失效的区域，GFlowNet 仍能将成功率提升到 ~20%、路径命中到 ~34%，量化了相对纯检索的增益，但提升幅度不及预期，需要针对困难样本加强探索与奖励设计。
