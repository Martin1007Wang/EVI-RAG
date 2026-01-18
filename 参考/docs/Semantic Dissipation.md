## Semantic Dissipation：从检索到推理的信息耗散（更科学的定义）

### 0. 目标（Problem Statement）
我们关心的不是“检索分高、LLM 分低”这种经验现象，而是一个可证伪、可分解、可复现的量：

> 当检索器已经提供了**足够的证据**时，LLM 仍然无法把证据转化为正确答案，这中间丢掉了什么（结构/噪声/窗口截断）？

因此必须把系统拆成三个正交环节：**检索**（内容是否包含证据）→ **接口**（证据是否进入窗口/以何种形式进入）→ **推理**（LLM 是否利用证据）。

---

### 1. 符号与接口（Notation & Interface）
* $D=\{1,\dots,N\}$：测试样本索引。
* $q_i$：第 $i$ 个问题。
* $\mathcal{A}_i$：第 $i$ 个样本的答案集合（允许多答案）。
* $\mathcal{P}_i$：第 $i$ 个样本的黄金证据路径集合（允许多条路径）。
  * 每条路径 $p\in\mathcal{P}_i$ 是若干三元组的集合（或序列），且任意 $p$ 足以推出 $\mathcal{A}_i$ 中至少一个答案。
* $R_i(K)$：检索器返回的 Top-$K$ 三元组集合（可带分数）。
* $\mathbb{I}(\cdot)$：指示函数，条件为真取 $1$，否则取 $0$。

为了显式刻画“窗口大小”和“数据形式”，引入接口函数：
* $\mathrm{lin}(\cdot;\,\phi)$：线性化模板（格式/顺序规则由 $\phi$ 决定），把三元组集合映射为文本。
* $\mathrm{tok}(\cdot)$：tokenizer。
* $B$：证据窗口的 token 预算（不含问题/系统提示词，或明确约定包含在内）。
* $\mathrm{trunc}_B(\cdot)$：按 token 预算截断。

定义实际喂给 LLM 的证据文本：
$$ E_i(K,B,\phi) \triangleq \mathrm{trunc}_B\left(\mathrm{lin}(R_i(K);\phi)\right) $$

以及它所“可见”的三元组集合（可通过可逆模板解析，或用严格约定的 serialization 保证可解析）：
$$ V_i(K,B,\phi) \triangleq \mathrm{parse}\left(E_i(K,B,\phi)\right) $$

---

### 2. 三个关键事件（Hit / Visible-Hit / Correct）
#### 2.1 检索命中（Set-Level Path Hit）
检索成功定义为：Top-$K$ 中完整包含至少一条黄金路径：
$$ H_i^{set}(K)\triangleq\mathbb{I}\left(\exists p\in\mathcal{P}_i\ \mathrm{s.t.}\ p\subseteq R_i(K)\right) $$
对应指标：
$$ S_{ret}^{set}(K)\triangleq \frac{1}{N}\sum_{i=1}^N H_i^{set}(K) $$

#### 2.2 可见命中（Visible Path Hit，显式刻画窗口/格式）
即使 $p\subseteq R_i(K)$，也可能因为**线性化顺序**或 **token 截断**导致证据不进入 LLM 窗口。因此定义：
$$ H_i^{vis}(K,B,\phi)\triangleq\mathbb{I}\left(\exists p\in\mathcal{P}_i\ \mathrm{s.t.}\ p\subseteq V_i(K,B,\phi)\right) $$
对应指标：
$$ S_{ret}^{vis}(K,B,\phi)\triangleq \frac{1}{N}\sum_{i=1}^N H_i^{vis}(K,B,\phi) $$

> **结论**：若要讨论“窗口下的理论上限”，上界应与接口一致，即 $S_{ret}^{vis}$，而不是 $S_{ret}^{set}$。

定义接口/截断损耗（非负）：
$$ \mathcal{L}_{iface}(K,B,\phi)\triangleq S_{ret}^{set}(K)-S_{ret}^{vis}(K,B,\phi)\ge 0 $$

#### 2.3 推理正确（Answer Correctness）
定义一个答案评分函数 $s(\hat{a}_i,\mathcal{A}_i)\in[0,1]$：
* **Any-hit/Accuracy**：$s=\mathbb{I}(\hat{a}_i\in\mathcal{A}_i)$。
* **Set-F1**：当 $\hat{a}_i$、$\mathcal{A}_i$ 都是集合时用集合 F1；多类别任务可用 **macro-F1**（先算每类 F1 再平均）。

端到端指标（显式依赖窗口与格式）：
$$ S_{llm}(K,B,\phi)\triangleq \frac{1}{N}\sum_{i=1}^N s\!\left(\hat{a}_i(E_i(K,B,\phi),q_i),\mathcal{A}_i\right) $$
记单样本得分随机变量：
$$ s_i(K,B,\phi)\triangleq s\!\left(\hat{a}_i(E_i(K,B,\phi),q_i),\mathcal{A}_i\right)\in[0,1] $$

---

### 3. 从检索到推理的“因果账本”（闭合分解）
下文把 $i$ 视为从 $D$ 上均匀采样的随机样本并省略下标（例如 $H^{vis}$ 表示随机样本的 $H_i^{vis}$）。
定义条件性能：
$$ Acc_{hit}(K,B,\phi)\triangleq \mathbb{E}\left[s\mid H^{vis}=1\right],\quad Acc_{miss}(K,B,\phi)\triangleq \mathbb{E}\left[s\mid H^{vis}=0\right] $$

则恒等分解成立（不依赖任何假设）：
$$ S_{llm}=Acc_{hit}\cdot S_{ret}^{vis}+Acc_{miss}\cdot(1-S_{ret}^{vis}) $$

这条式子把“检索可见性”和“推理利用率”正交拆开：你想要解释的性能差距，必须落到 $S_{ret}^{vis}$、$Acc_{hit}$、$Acc_{miss}$ 三个量上。

---

### 4. 语义耗散（Semantic Dissipation）与泄漏（Leakage）
直接用 $S_{ret}-S_{llm}$ 作为 gap 会出现负值（LLM 猜对/记住答案），不再表示“信息损耗”。更科学的做法是定义**非负**的耗散量：

#### 4.1 耗散率（有证据但用不上）
$$ \mathcal{D}_{rate}(K,B,\phi)\triangleq 1-Acc_{hit}(K,B,\phi)\in[0,1] $$

#### 4.2 耗散质量（在人群层面丢了多少）
$$ \mathcal{D}_{mass}(K,B,\phi)\triangleq \mathbb{E}\left[(1-s)\cdot\mathbb{I}(H^{vis}=1)\right] = S_{ret}^{vis}\cdot(1-Acc_{hit})\ge 0 $$
它是“有证据的人群里，平均丢了多少分”（$s$ 可为 0/1、Set-F1 或 macro-F1）。

#### 4.3 泄漏（无证据也能答对）
$$ \mathcal{L}_{leak}(K,B,\phi)\triangleq \mathbb{E}\left[s\cdot\mathbb{I}(H^{vis}=0)\right]=(1-S_{ret}^{vis})\cdot Acc_{miss} $$
它量化“闭卷知识/数据偏置/标注噪声”对端到端分数的贡献，也解释为什么“检索命中率”不能直接当 LLM 上界。
若希望把 $\mathcal{L}_{leak}$ 压到接近 $0$（从而让指标更“证据驱动”），可以采用 **grounded 评估**：要求模型输出可解析的三元组/边 ID 引用，未引用或引用不匹配则判错。

---

### 5. Oracle 对照：分离“结构损耗”与“噪声/长上下文损耗”
要分解结构与噪声，必须控制 **信息内容** 与 **token 预算**，否则比较不成立。

#### 5.1 Oracle 内容选择（控制信息量）
为每个样本选择一条最小黄金路径 $p_i^*$（例如按边数最短，或按在同一模板下 token 最短）：
$$ p_i^*\in\arg\min_{p\in\mathcal{P}_i}\ \mathrm{tok}\!\left(\mathrm{lin}(p;\phi_{struct})\right) $$

#### 5.2 Oracle-Structured（结构化输入）
* 内容：仅 $p_i^*$。
* 形式：拓扑有序（链式、分层、带显式节点对齐），记为 $\phi_{struct}$。
* 定义：$E_i^*(B,\phi)\triangleq \mathrm{trunc}_B(\mathrm{lin}(p_i^*;\phi))$。
* 指标：$Acc_{struct}(B)\triangleq \mathbb{E}\left[s(\hat{a}_i(E_i^*(B,\phi_{struct}),q_i),\mathcal{A}_i)\right]$。

#### 5.3 Oracle-Linear（同内容乱序）
* 内容：仍仅 $p_i^*$。
* 形式：bag-of-triples 随机乱序（多次 shuffle 取均值），记为 $\phi_{linear}$；$\pi$ 表示对 $p_i^*$ 中三元组的随机排列。
* 指标（对乱序取期望）：
$$ Acc_{linear}(B)\triangleq \mathbb{E}_{\pi}\,\mathbb{E}\left[s\!\left(\hat{a}_i\!\left(\mathrm{trunc}_B(\mathrm{lin}(p_i^*;\phi_{linear},\pi)),q_i\right),\mathcal{A}_i\right)\right] $$

#### 5.4 结构损耗（纯形式差异，信息不变）
$$ \mathcal{L}_{struct}(B)\triangleq Acc_{struct}(B)-Acc_{linear}(B) $$

#### 5.5 噪声/长上下文损耗（在同一线性化范式下）
用与 Oracle-Linear 相同的模板 $\phi_{linear}$ 线性化真实检索结果，并只在 $H^{vis}=1$ 的子集上比较：
$$ \mathcal{L}_{noise}(K,B)\triangleq Acc_{linear}(B)-Acc_{hit}(K,B,\phi_{linear}) $$
它同时捕获两类现实效应：
1) 噪声使信噪比下降（注意力被稀释）；2) 在固定预算 $B$ 下，噪声把关键边挤出窗口（$S_{ret}^{vis}$ 下降）。

---

### 6. 必须收集的指标（为了“解释”，不只是“汇报”）
对每个 $(K,B,\phi)$，至少记录：
* **检索侧**：$S_{ret}^{set}(K)$、以及你已有的 precision/recall/F1（可选，但要说明它们不等价于路径命中）。
* **接口侧**：$S_{ret}^{vis}(K,B,\phi)$、$\mathcal{L}_{iface}(K,B,\phi)$、证据 token 数、是否截断、截断后可见三元组数 $K_{eff}$。
* **推理侧**：$S_{llm}(K,B,\phi)$、$Acc_{hit}(K,B,\phi)$、$Acc_{miss}(K,B,\phi)$。
* **耗散/泄漏**：$\mathcal{D}_{rate}$、$\mathcal{D}_{mass}$、$\mathcal{L}_{leak}$。

---

### 7. 图表建议（能一眼讲清“窗口 vs 形式”）
#### 7.1 主图：固定 $B$，扫 $K$
在同一张图上画：
* $S_{ret}^{set}(K)$（检索能力上限，集合级）。
* $S_{ret}^{vis}(K,B,\phi_{linear})$（进入窗口后的证据可达性）。
* $S_{llm}(K,B,\phi_{linear})$（端到端）。
* 阴影或副轴：$\mathcal{D}_{rate}(K,B,\phi_{linear})$ 或 $\mathcal{D}_{mass}(K,B,\phi_{linear})$。

这张图直接展示：$K$ 增大时检索命中增加，但由于噪声与截断，$S_{ret}^{vis}$ 与 $S_{llm}$ 可能出现平台甚至下降。

#### 7.2 副图：固定内容（Oracle），扫 $B$
画 $Acc_{struct}(B)$ 与 $Acc_{linear}(B)$ 两条曲线，以及它们差值 $\mathcal{L}_{struct}(B)$。

这张图把“LLM 处理长上下文的能力（随 $B$ 增长）”与“数据形式的结构优势（$\mathcal{L}_{struct}$）”分离开，避免把两者混成一个模糊的 gap。
