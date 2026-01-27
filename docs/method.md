# Methodology: Off-Policy Detailed Balance (Student Explore, Teacher Grade)

我们采用 **“Student 探路，Teacher 判卷”** 的 Off-Policy Detailed Balance (DB) 训练范式。Student 负责采样轨迹，Teacher 只对同一条轨迹做反向评估；损失由前向流与后向流的严格平衡约束驱动，**双向更新、无 detach**。

---

## 1. 核心逻辑 (The Core Logic)

**Forward Student ($P_F$)**  
输入：$q, s_{start}, u$。  
任务：预测下一步 $v$，并实际采样轨迹 $\tau$。

**Backward Teacher ($P_B$)**  
输入：$q, s_{start}, A, v$。  
任务：在 Student 轨迹上评估 $P_B(u|v)$，不进行采样。

**Detailed Balance 约束**  
$$
\log Z(u) + \log P_F(v|u) = \log Z(v) + \log P_B(u|v)
$$
梯度同时回传给 $Z, P_F, P_B$。

---

## 2. 网络架构 (The Architecture)

- **两套独立的 Policy**：Forward/Backward 各自一套 `QCBiANetwork`（参数不共享）。  
- **共享 Z 网络**：只维护一套 `Z(u)` 预测器，输入为 $(c_{fwd}, u)$。  
- **Backward Context 融合**：$h_{ctx}^{bwd} = \text{Linear}(\text{Concat}(h_q, h_{start}, \text{AttnPool}(h_A)))$。  

> 说明：Backward 必须同时看 $q$、$s_{start}$ 与 $\text{Pool}(A)$，避免语义冲突采用 **Concat + Projection**。

---

## 3. 训练流程 (The Algorithm)

### 3.1 Forward Rollout (Student)
- Context: $h_{ctx} = \text{Linear}([h_q; h_{start}])$  
- 采样：Gumbel-Max 或 Categorical  
- 输出：轨迹 $\tau=\{(u_t, r_t, v_t)\}_{t=0}^{T}$

### 3.2 Dual Evaluation
**A. Forward Eval (Student)**  
计算 $\log Z(u_t)$ 与 $\log P_F(v_t|u_t)$。  

**B. Backward Eval (Teacher)**  
Context: $h_{ctx}^{bwd} = \text{Linear}([h_q; h_{start}; \text{AttnPool}(h_A)])$  
计算 $\log Z(v_t)$ 与 $\log P_B(u_t|v_t)$，候选边为 $r^{-1}$。

### 3.3 Detailed Balance Loss
$$
\Delta_t = (\log Z(u_t)+\log P_F(v_t|u_t)) - (\log Z(v_t)+\log P_B(u_t|v_t))
$$
$$
\mathcal{L}=\mathbb{E}_t[\Delta_t^2]
$$

**边界条件 (必须硬编码)**  
- **命中答案集合**：$\log Z(a)=0, \forall a \in A$。  
- **死胡同**：对终止失败节点施加负奖励或降低权重。

---

## 4. 选边策略 (QC-BiA Policy)

`QCBiANetwork` 直接对边打分，避免 $P(r|u)P(v|u,r)$ 的幼稚分解：
$$
\text{logit}(u\!\to\!v)=\langle z_u \odot \text{FiLM}_Q(z_r), z_v\rangle/\sqrt{d}
$$

实现位置：`src/models/components/qc_bia_network.py`。

---

## 5. 工程映射 (Code + Config)

**核心代码**  
- 主模块：`src/models/dual_flow_module.py`  
- QC-BiA：`src/models/components/qc_bia_network.py`

**关键配置**  
- `configs/model/gflownet.yaml`
  - `training_cfg.db_cfg.sampling_temperature_schedule` (默认: cosine)
  - `training_cfg.db_cfg.sampling_temperature_start`
  - `training_cfg.db_cfg.sampling_temperature_end`
  - `training_cfg.db_cfg.sampling_temperature` (仅 constant 时使用)
  - `training_cfg.db_cfg.teacher_edge_dropout`
  - `training_cfg.db_cfg.dead_end_log_reward`
  - `training_cfg.db_cfg.dead_end_weight`
  - `embedding_adapter_cfg.enabled`
  - `embedding_adapter_cfg.adapter_dim`
  - `embedding_adapter_cfg.dropout`
  - `actor_cfg.edge_inter_dim`
  - `actor_cfg.edge_dropout`

---

## 6. 数据语义对齐 (Flow vs QA)

- **Data SSOT**：`q_local_indices` / `a_local_indices` 仅表示实体集合真相。  
- **Flow-Derived**：`start_nodes_fwd` 来自 `q_local_indices`；`node_is_target` 由 `a_local_indices`（答案集合）派生。  
- **Mask 语义**：`dummy_mask` 来自 data-level answer；`node_is_target` 来自 flow target。  
- **禁止交换 q/a**：反向流必须依靠方向映射而非 swap。

---

## 7. 方案对齐检查

**必须满足**  
1. Student 负责采样，Teacher 只评估，不采样。  
2. Backward Policy 必须看 $q + s_{start} + \text{Pool}(A)$（Concat+Proj）。  
3. DB 约束双向更新，无 detach。  
4. Target 节点 LogZ 必须硬置为 0。  
5. Dead End 必须引入惩罚或降低权重。  
