# VERA：针对检索任务的变分校准框架

本文件描述的是 **在极度不平衡的检索任务中，如何把“排序”和“置信度/不确定性”解耦，并在 query 级别做变分校准**。  
这里的 VERA 不是一个单独的模型，而是放在现有检索 backbone 之上的一套 **损失与结构设计原则**。

---

## 1. 问题形式化：候选级 vs Query 级

对每个 query \(i = 1,\dots,N\) 有候选集合 \(C_i = \{j=1,\dots,M_i\}\)：

- 候选级标签：\(y_{ij} \in \{0,1\}\)，表示候选三元组是否为正确答案；
- 真实分布在候选级极端不平衡：  
  \[
  P(y_{ij}=1) \approx 5\times 10^{-3},\quad P(y_{ij}=0) \approx 1.
  \]

我们关心的目标有两个层级：

1. **候选级排序**（ranking）  
   对固定的 query \(i\)，希望正例在候选集合 \(C_i\) 内排在前面，指标是 MRR / NDCG@k。
2. **query 级检索成功概率**（selective / calibration）  
   - 定义 query 级事件，例如  
     \[
     Z_i^{(k)} = \mathbf{1}\{\exists j \le k : y_{ij}=1\} \quad \text{(top-k 内至少一个正确)}
     \]
   - 我们希望模型给出 \(P(Z_i^{(k)}=1\mid x_i)\) 的可靠估计，用于 selective prediction。

重要的是：  
**VERA 的 KL 与熵项应该作用在 query 级 \(Z_i\) 上，而不是直接作用在极度不平衡的候选级标签 \(y_{ij}\) 上。**

---

## 2. 总体结构：Backbone + 排序头 + Evidential 头

我们将模型拆成三部分：

1. 共享 backbone：  
   \[
   z_{ij} = f_\theta(x_{ij}) \quad (\text{图结构 + 语义特征})
   \]
2. 排序头（deterministic）：  
   \[
   s_{ij} = g_\phi(z_{ij}) \in \mathbb{R}
   \]
   用于 InfoNCE / 对比损失，只关心「谁比谁大」。
3. evidential 头（Beta）：  
   \[
   (\alpha_{ij}, \beta_{ij}) = u_\psi(z_{ij}),\quad
   \alpha_{ij} = \lambda^+ + \text{softplus}(a_{ij}),\;
   \beta_{ij} = \lambda^- + \text{softplus}(b_{ij})
   \]

候选级：

- 概率均值：\(\mu_{ij} = \frac{\alpha_{ij}}{\alpha_{ij}+\beta_{ij}}\) 作为 \(P(y_{ij}=1\mid x_{ij})\) 的 proxy；
- 证据量：\(\kappa_{ij} = \alpha_{ij}+\beta_{ij}\) 表示不确定性（\(\kappa\) 小 → 不确定性高）。

query 级聚合：

- 按排序头分数取前 k：
  \[
  \text{Top-k}(s_i) = \text{argsort}_j(s_{ij})[:k]
  \]
- 使用独立近似得到检索成功概率：
  \[
  c_i^{(k)} = 1 - \prod_{j \in \text{Top-k}(s_i)} (1 - \mu_{ij})
  \]

这个 \(c_i^{(k)}\) 就是 VERA 要在 query 级上校准的对象。

---

## 3. VERA 的核心：用熵调节 KL 方向（在 query 级）

对每个 query 定义：

- 经验分布（由真实标签给出）：
  \[
  p_i(1) = Z_i^{(k)},\quad p_i(0) = 1 - Z_i^{(k)}
  \]
- 模型分布（由聚合概率给出）：
  \[
  q_i(1) = c_i^{(k)},\quad q_i(0) = 1 - c_i^{(k)}
  \]

**注意：这里的 \(p_i\) 是 query 级分布，不再是候选级极端不平衡的 \(P(y_{ij})\)。**

定义熵：

$$
H(p_i) = -\sum_{y \in \{0,1\}} p_i(y)\log p_i(y), \quad
H_{\max} = \log 2 = 1,\quad
H_{\text{norm}}(p_i) = \frac{H(p_i)}{H_{\max}}.
$$

VERA 的 query 级损失：

$$
\boxed{
L_{\text{VERA},i} = 
\alpha \cdot KL(p_i\|q_i)
 + (1-\alpha) \cdot KL(q_i\|p_i)
 + \beta \cdot (1 - H_{\text{norm}}(p_i)) \cdot KL(p_i\|q_i)
}
$$

展开为交叉熵形式：

$$
KL(p_i\|q_i) = -\sum_y p_i(y)\log q_i(y)
$$

$$
KL(q_i\|p_i) = \sum_y q_i(y)[\log q_i(y) - \log p_i(y)]
$$

**直觉**：

- 当某类 query 的真实标签很“确定”（例如几乎总是检索成功），\(H_{\text{norm}}(p_i)\) 很小，\((1-H_{\text{norm}}(p_i))\approx 1\)，  
  这会放大 \(KL(p_i\|q_i)\) 的权重，强制模型在这类 query 上高度对齐。
- 当真实标签本身比较混乱时，附加项权重减弱，避免过拟合噪声。

与传统只选一个方向的 KL 相比，VERA **同时使用 KL(p‖q) 与 KL(q‖p)，再用真实分布熵自适应调节重点**。

---

## 4. 完整的损失分解

在检索任务中，我们不再指望“一个 EDL/Type-II loss 搞定一切”，而是把目标拆成三层：

### 4.1 排序目标：\(L_{\text{rank}}\)（候选级，对比式）

在每个 query 内，对 scores \(s_{ij}=g_\phi(z_{ij})\) 做 softmax：

$$
q_{ij}^{\text{rank}} = \frac{\exp(s_{ij}/\tau)}{\sum_{u\in C_i}\exp(s_{iu}/\tau)}
$$

候选级（归一化）标签 \(\tilde{y}_{ij}\)（对正例均匀分配质量），InfoNCE 风格：

$$
L_{\text{rank}} = -\frac{1}{N}\sum_i\sum_{j\in C_i}\tilde{y}_{ij}\log q_{ij}^{\text{rank}}.
$$

这项只负责「把正例排在前面」，不关心绝对概率刻度。

### 4.2 候选级 evidential 目标：\(L_{\text{cand-evi}}\)（候选级，弱监督）

在候选级，我们只用 Beta 均值 \(\mu_{ij}\) 做一个轻量的 Type-II 风格约束，并显式按 query 平衡：

$$
L_{\text{cand-evi}}^{(i)} =
\frac{1}{|P_i|}\sum_{j\in P_i} -\log \mu_{ij}
 + \lambda_{\text{neg}} \cdot \frac{1}{|N_i|}\sum_{j\in N_i} -\log (1-\mu_{ij}),
$$

其中：

- \(P_i = \{j : y_{ij}=1\}\)，\(N_i = \{j : y_{ij}=0\}\)；
- \(\lambda_{\text{neg}}\ll 1\)（负例特别多，不需要强推）。

全局：

$$
L_{\text{cand-evi}} = \frac{1}{N'}\sum_{i:|P_i|>0}L_{\text{cand-evi}}^{(i)}.
$$

这项只训练 evidential 头 \(u_\psi\)，**不负责排序**。

### 4.3 Query 级 VERA 校准：\(L_{\text{query-evi}}\)

按上一节定义的 \(L_{\text{VERA},i}\)，在 query 级做 KL+熵调节：

$$
L_{\text{query-evi}} = \frac{1}{N}\sum_i L_{\text{VERA},i}.
$$

它只关心 **从 \(\{\mu_{ij}\}\) 聚合出来的 query 级成功概率 \(c_i^{(k)}\)** 是否贴近真实 \(Z_i^{(k)}\)。

### 4.4 一致性项：\(L_{\text{consistency}}\)

排序头输出的 logits 可以经 sigmoid 变成另一种概率刻度：

$$
\sigma_{ij} = \sigma(s_{ij}).
$$

我们希望排序概率与 evidential 均值不要严重背离，可以加：

$$
L_{\text{consistency}} = \mathbb{E}_{i,j}[(\sigma_{ij} - \mu_{ij})^2].
$$

这项把 ranking 头和 evidential 头在概率空间里对齐，避免一个说“很有把握”，另一个说“完全不确定”的矛盾。

### 4.5 总损失

最终：

$$
L_{\text{total}} =
L_{\text{rank}}
 + \lambda_{\text{cand}}L_{\text{cand-evi}}
 + \lambda_{\text{query}}L_{\text{query-evi}}
 + \lambda_{\text{cons}}L_{\text{consistency}}.
$$

- 训练早期：主要优化 \(L_{\text{rank}}\)，保证 backbone + 排序头有足够好的检索质量；
- 中期：冻结或弱化 backbone，把重点放在 \(L_{\text{cand-evi}}\) + \(L_{\text{cons}}\) 上，学好 Beta 参数；
- 后期：加入 \(L_{\text{query-evi}}\)，在 query 级别做 VERA 校准。

---

## 5. 架构示意（对应代码里的拆分）

```text
Graph / Dense Backbone: (query, graph_triple) → z_ij (hidden_dim)
                                  │
             ┌────────────────────┴──────────────────────┐
             │                                           │
   Ranking Head g_φ                                 Evidential Head u_ψ
     Dense → 1-d score s_ij                         Dense → 2-d (a_ij, b_ij)
             │                                           │
   InfoNCE / softmax over C_i                     softplus → (α_ij, β_ij)
             │                                           │
   排序：q^rank_ij = softmax(s_ij / τ)           概率：μ_ij = α_ij / (α_ij+β_ij)
                                                 证据：κ_ij = α_ij + β_ij
```

query 级：

1. 对每个 query，利用排序头取 Top‑k；
2. 用对应候选的 \(\mu_{ij}\) 计算 \(c_i^{(k)} = 1 - \prod_{j\in\text{Top-k}}(1-\mu_{ij})\)；
3. 用 \(Z_i^{(k)}\) 构造 \(p_i\)，在 \((p_i, q_i)\) 上施加 VERA。

---

## 6. 超参数建议（针对检索场景）

| 参数                      | 建议值           | 说明                                                       |
| ------------------------- | ---------------- | ---------------------------------------------------------- |
| \(\alpha\)                | 0.7              | VERA 中 KL(p‖q) 的基础权重，偏向“真实分布主导”            |
| \(\beta\)                 | 1.0              | 熵调节项强度，极度确定时放大 KL(p‖q)                      |
| \(\lambda_{\text{cand}}\) | 0.1–0.3          | 候选级 evidential 目标权重，不宜过大                      |
| \(\lambda_{\text{query}}\)| 0.5–1.0          | query 级 VERA 校准权重                                     |
| \(\lambda_{\text{cons}}\) | 0.1–0.3          | 排序与 Beta 均值一致性权重                                 |
| \(\lambda_{\text{neg}}\)  | 0.01–0.1         | 候选级负例在 Type-II 风格损失中的相对权重（按 query 归一）|

整体原则：

- **排序优先**：先确保 `ranking/mrr`、`ndcg@k` 达到或超过当前 EDL 模型；
- **校准第二**：再用 VERA 逐步校正 query 级的 `selective/score/*` 与 `calib/*`；
- **不把所有目标塞进一个 loss**：候选级 y、query 级 Z、排序和不确定性，各自有清晰的对象与损失。

这就是 VERA 在当前检索项目中的“最佳设计版本”：  
**它不是再造一个巨大的一体化 loss，而是在一个已经很强的检索 backbone 上，增加一个有原则的、可解释的 query 级变分校准层。**
