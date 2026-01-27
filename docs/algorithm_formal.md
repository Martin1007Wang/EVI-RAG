# DualFlow 数学化算法规格（Code-Exact）

本文以**代码为唯一真相**，给出本仓库当前实现的数学化算法描述（适配 ICLR 论文级别规范）。所有符号与流程严格对齐 `src/models/dual_flow_module.py` 及相关模块。

---

## 1. 问题设定与数据语义

每个样本对应一个检索子图（`g_retrieval`）：

- 图：$G = (\mathcal{V}, \mathcal{E})$，边为有向关系 $(u, r, v)$。
- 起点集合（数据 SSOT）：$S \subseteq \mathcal{V}$，对应 `q_local_indices`。
- 答案集合（数据 SSOT）：$A \subseteq \mathcal{V}$，对应 `a_local_indices`。
- 问题向量：$q \in \mathbb{R}^{d}$，对应 `question_emb`。

**重要语义约束**：

- $S$ 与 $A$ 仅代表问题/答案实体集合，**不互换**。
- 流向由方向 $\text{forward/backward}$ 派生，不能通过交换 $S/A$ 实现。
- 训练仅使用 `g_retrieval` 子图；缺失答案样本由 `dummy_mask` 过滤。

---

## 2. 记号与结构

- 节点嵌入：$x_u$ 为原始节点嵌入（由 LMDB 载入）。
- 关系嵌入：$x_r$ 为原始关系嵌入。
- 时间步：$t \in \{0, \dots, T\}$，最大步数为 $T$。
- 方向性边集：
  - $\mathcal{E}_F$：正向边（非逆关系）。
  - $\mathcal{E}_B$：逆向边（逆关系）。
- 逆边映射：$\text{inv}(e) \leftrightarrow e'$，由 relation vocab 提供的逆关系映射生成。
- **严格无自环**：运行时强制检查 $u \neq v$（任何自环边直接抛错）。

---

## 3. 模型组件

### 3.1 双向 GNN Backbone

前向与后向**完全独立参数**的 GNN：

- $h_u^F = \text{GNN}_F(x_u, x_r, G)$
- $h_u^B = \text{GNN}_B(x_u, x_r, G)$

其中 $\text{GNN}$ 为多层关系图卷积（`EmbeddingBackbone` + `RelationalGNNLayer`）。CVT 节点采用邻接均值初始化（如开启）。

可选的 **Embedding Adapter** 对实体/关系嵌入进行残差微调（默认开启，Pre-LN）：  
\[
\\tilde{x} = x + \\text{Adapter}(\\text{LN}(x)), \\quad \\text{Adapter}(x)=W_2(\\sigma(W_1 x))
\]
其作用是允许预训练图谱嵌入对当前 QA 任务进行轻量适配，而无需解冻整张 embedding 表。

### 3.2 起点选择器（Virtual Super-Source）

在起点集合 $S$ 上构建虚拟 ROOT，通过可学习选择器确定起点：

\[
\ell_i = g_\psi([q, h_{s_i}^F]), \quad s_i \in S
\]

使用 Gumbel-Max 采样并采用 Straight-Through (ST) 权重：

\[
\pi_i = \text{softmax}(\ell)_i,\quad i^* = \arg\max(\ell_i + \gamma_i)
\]

\[
\tilde{w}_i = \mathbf{1}[i=i^*] - \pi_i^{\text{stopgrad}} + \pi_i
\]

得到：

- **离散起点**：$s = s_{i^*}$（用于 rollouts）。
- **起点 token**：$\bar{h}_S^F = \sum_i \tilde{w}_i h_{s_i}^F$，$\bar{h}_S^B = \sum_i \tilde{w}_i h_{s_i}^B$。

> 选择器最后一层权重零初始化，初始等价于均匀分布（代码精确实现）。

### 3.3 策略网络（QC-BiA）

前向策略（Student）与后向策略（Teacher）分别使用独立 QCBiA 网络：

\[
\text{logit}_e = \text{QCBiA}(c, h_u, h_r, h_v)
\]

其中 QCBiA 采用 Query-Conditioned FiLM 调制关系向量，具体为：

\[
(z_q, z_u, z_r, z_v) = (W_q h_q, W_u h_u, W_r h_r, W_v h_v)
\]

\[
(\gamma, \beta) = W_{gate}(\text{GELU}(W_{inter} z_q))
\]

\[
\tilde{z}_r = (1+\gamma) \odot z_r + \beta, \quad \text{logit} = \langle z_u \odot \tilde{z}_r, z_v \rangle / \sqrt{d}
\]

### 3.4 Context 构造

- 前向 context：
  \[
  c_F = \text{MLP}_F([q, \bar{h}_S^F])
  \]

- 后向 context：
  \[
  c_B = \text{MLP}_B([q, \bar{h}_S^B, h_{a_{seed}}^B])
  \]

其中 $a_{seed}$ 来自目标轮盘赌（见后文）。

### 3.5 流势能网络（LogZ）

Z 网络仅依赖 $(q, u, t)$，不看答案：

\[
\log Z(u,t) = f_\theta(h_u + \text{TimeEmbed}(t), c_F)
\]

与 Student 信息同构，避免答案泄漏。

---

## 4. 策略分布

### 4.1 前向策略（Student）

在当前节点 $u$，对所有出边 $e=(u,r,v) \in \mathcal{E}_F$：

\[
P_F(e \mid u, t, q) = \text{softmax}(\text{QCBiA}(c_F, h_u^F, h_r, h_v^F) / T_{samp})
\]

训练中使用温度 $T_{samp}$；推理可用 $T=1$。

### 4.2 后向策略（Teacher）

在当前节点 $v$，对所有逆边 $e'=(v,r^{-1},u) \in \mathcal{E}_B$：

\[
P_B(e' \mid v, t+1, q, a_{seed}) = \text{softmax}(\text{QCBiA}(c_B, h_v^B, h_{r^{-1}}, h_u^B))
\]

Teacher 仅用于训练，不共享 Student 参数。

**Teacher Edge Dropout（训练期）**：对 Teacher 可用的逆边集合施加随机边丢弃（结构级 Dropout），在每次后向 rollout 中采样一个全局掩码 $m_e \sim \text{Bernoulli}(1-p)$，仅保留 $m_e=1$ 的边用于 $P_B$ 计算与采样。

---

## 5. 目标轮盘赌（Target Roulette）

对每个样本在答案集合 $A$ 中均匀采样：

\[
a_{seed} \sim \text{Uniform}(A)
\]

该 $a_{seed}$ **同时**用于：

1. 作为后向 rollouts 的起点；
2. 构造后向 context 的答案 token $h_{a_{seed}}^B$。

**注意**：奖励判断仍基于**全体答案集合** $A$，并未收窄到 $a_{seed}$。

---

## 6. Rollout 动力学

### 6.1 Forward Exploration

- 起点：$s \in S$（StartSelector 采样）。
- 动作：按 $P_F$ 在 $\mathcal{E}_F$ 上采样。
- 终止：
  - 命中 $A$（terminal hit）
  - 无出边（dead end）
  - 达到最大步数 $T$

### 6.2 Backward Demonstration

- 起点：$a_{seed} \in A$（Target Roulette）。
- 动作：按 $P_B$ 在 $\mathcal{E}_B$ 上采样。
- 终止：
  - 命中 $S$（terminal hit）
  - 无出边（dead end）
  - 达到最大步数 $T$

后向轨迹的边 $e'$ 通过逆映射 $\text{inv}$ 转换为前向边 $e$，用于统一 DB Loss 计算。

---

## 7. Detailed Balance Loss

对任意轨迹 $\tau = (u_0 \to u_1 \to \dots \to u_L)$，每一步的 DB 残差：

\[
\Delta_{DB}(u_t,u_{t+1}) = \log Z(u_t,t) + \log P_F(u_{t+1}\mid u_t,t)
- \log Z(u_{t+1},t+1) - \log P_B(u_t\mid u_{t+1},t+1)
\]

**边界硬约束**：

- 若 $u_{t+1} \in A$，则 $\log Z(u_{t+1},t+1) \leftarrow 0$。
- 若终止于 dead end，$\log Z(u_{t+1},t+1) \leftarrow \log R_{dead}$（固定负值）。

**损失**：

\[
\mathcal{L}_{DB}(\tau) = \sum_{t=0}^{L-1} \left(\Delta_{DB}(u_t,u_{t+1})\right)^2
\]

失败轨迹可按权重 $w_{dead}$ 进行放大（代码中支持）。

---

## 8. 双向混合采样目标

训练使用两类轨迹：

- **Forward Exploration**：$\tau_F \sim P_F$（覆盖性）
- **Backward Demonstration**：$\tau_B \sim P_B$（高势能通道）

混合目标为：

\[
\mathcal{L} = \tfrac{1}{2}\mathbb{E}_{\tau_F}[\mathcal{L}_{DB}(\tau_F)] +
\tfrac{1}{2}\mathbb{E}_{\tau_B}[\mathcal{L}_{DB}(\tau_B)]
\]

实现中每个 batch 可进行多次 rollout（`num_rollouts`），并对损失求平均。

---

## 9. 训练算法（伪代码）

**Algorithm 1: Mixed-Trajectory Detailed Balance Training**

1. 输入 batch：$(G, q, S, A)$
2. 通过双向 GNN 得到 $h^F, h^B$
3. StartSelector：采样 $s \in S$，构造 $\bar{h}_S^F, \bar{h}_S^B$
4. Target Roulette：采样 $a_{seed} \in A$
5. 构造上下文：
   - $c_F = \text{MLP}_F([q, \bar{h}_S^F])$
   - $c_B = \text{MLP}_B([q, \bar{h}_S^B, h_{a_{seed}}^B])$
6. Forward Rollout：$\tau_F \sim P_F(\cdot \mid c_F)$
7. Backward Rollout：$\tau_B \sim P_B(\cdot \mid c_B)$，并映射为前向边序列
8. 对 $\tau_F$ 与 $\tau_B$ 分别计算 DB Loss
9. 取均值作为训练目标，反向传播更新 $(\theta, \phi)$

---

## 10. 与代码实现的对应关系

- 主训练逻辑与 DB Loss：`src/models/dual_flow_module.py`
- QCBiA 定义：`src/models/components/qc_bia_network.py`
- GNN Backbone 与 LogZ：`src/models/components/gflownet_layers.py`
- 数据 SSOT 约束：`src/data/g_retrieval_dataset.py`

---

## 11. 结论性说明

该实现形成了**条件化流场 + 双向混合采样**的 GFlowNet 训练体系：

- Student 覆盖探索空间；Teacher 提供从答案回溯的高势能通道。
- StartSelector 引入虚拟 ROOT，学习“有效起点分布”。
- Target Roulette 在训练动态中逼迫多模态路径的共同收敛。

这是一个可直接对应到代码实现的、可发表级别的数学化算法规范。
