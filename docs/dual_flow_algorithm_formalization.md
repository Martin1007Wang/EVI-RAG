# DualFlow 算法复盘与数学形式化（当前实现）

> 版本说明：本文档对应仓库当前代码实现（截至 `2026-03-04`），目标是把“实现语义”完整映射为“数学对象”。

---

## 1. 问题定义与数据契约

我们在检索子图上训练双向流模型。对每个样本，给定子图
\[
G_{sub}=(V,E),
\]
其中包含问题实体集合与答案实体集合的局部索引：

- 起点集合（数据 SSOT）：`q_local_indices`
- 终点集合（数据 SSOT）：`a_local_indices`

在当前实现中，DataLoader 可选注入双 super node（每图两个）：

- `question_super`（记为 \(v_q^\star\)）
- `answer_super`（记为 \(v_a^\star\)）

并注入两类虚拟边：

- \(v_q^\star \to q,\ \forall q\in Q\)
- \(a \to v_a^\star,\ \forall a\in A\)

对应实现：

- super-source 注入：`src/datasets/batch_adapter.py:504`
- super 边构造：`src/datasets/batch_adapter.py:369`
- 运行时静态图上下文：`src/models/environment/context.py:26`

---

## 2. 双向 MDP 形式化

### 2.1 状态

当前策略满足 Node-Markov 契约，状态不显式保留完整路径历史。时间步 \(t\) 的状态写作：
\[
s_t = (u_t, q),
\]
其中 \(u_t\in V\) 是当前节点，\(q\) 为问题条件。

对应实现：

- Node-Markov 校验：`src/models/policy/path.py:6`
- 动态状态结构：`src/models/environment/state.py:8`

### 2.2 动作空间

给定状态 \(s_t\)，动作集合为
\[
\mathcal A(s_t)=\mathcal A_{\text{move}}(s_t)\cup\{\text{STOP}\},
\]
其中 move 动作来自 CSR 邻接中当前节点的所有出边。

对应实现：

- CSR 并行抽取：`src/models/policy/action.py:52`
- STOP logit 计算：`src/models/policy/action.py:9`

### 2.3 前向 / 后向流

前向流：
- 起点：\(v_q^\star\)
- 目标：\(v_a^\star\)

后向流：
- 起点：\(v_a^\star\)
- 目标：\(v_q^\star\)
- 转移拓扑：在反图上 rollout（交换 `adj_t_fwd/adj_t_bwd`）

对应实现：

- 起点与目标解析：`src/rollout/engine/state_init.py:47`, `src/rollout/engine/state_init.py:93`
- 后向反图采样：`src/models/algorithms/dual_flow.py:232`

---

## 3. 策略参数化

### 3.1 编码器

策略先对节点、关系、问题做编码：
\[
h_v,\ h_r,\ h_q=\text{Encoder}_\theta(G_{sub},q).
\]

对应实现：`src/models/policy/dual_flow_policy.py:80`

### 3.2 节点势函数（logF）

节点势函数由 `NodeFlowHead` 给出：
\[
\log F_\theta(v)=\frac{z_\theta(v,q)}{T_{\text{node}}}.
\]

对应实现：`src/models/policy/dual_flow_policy.py:236`

### 3.3 边动作打分（分层）

对边 \(e=(u,r,v)\)，策略先做两级分解：

1. 关系级：
\[
\log P_\theta(r\mid s)
\]
2. 关系内边级：
\[
\log P_\theta(e\mid s,r)
\]

并叠加 Doob-\(h\) 形态的节点势偏置：
\[
\ell_e = \log P_\theta(r\mid s)+\log P_\theta(e\mid s,r)+\alpha\log F_\theta(v).
\]

其中 \(\alpha=\texttt{doob\_h\_alpha}\ge 0\)。

对应实现：

- 分层 relation/conditional 归一化：`src/models/policy/edge.py:173`, `src/models/policy/edge.py:187`
- Doob-\(h\) 偏置：`src/models/policy/edge.py:205`

### 3.4 STOP 机制

先计算状态相关偏置
\[
\delta(s)=c\cdot\tanh\!\left(\frac{w^\top \phi(s)}{T_{\text{stop}}}\right),
\]
再定义
\[
\ell_{\text{stop}}
\approx
\log\!\sum_{e\in\mathcal A_{\text{move}}(s)}\exp(\ell_e)+\delta(s).
\]

于是最终动作分布为
\[
P_F(a\mid s)=\text{softmax}\big(\{\ell_e\}_{e\in\mathcal A_{\text{move}}(s)}\cup\{\ell_{\text{stop}}\}\big).
\]

当 move 动作非空且有限时，有
\[
P_F(\text{STOP}\mid s)=\sigma(\delta(s)).
\]

对应实现：

- stop delta：`src/models/policy/dual_flow_policy.py:386`
- stop logits：`src/models/policy/action.py:9`
- 最终拼接 softmax 采样：`src/rollout/engine/sampler.py:58`, `src/rollout/engine/sampler.py:84`

---

## 4. 反向流一致性 \(P_B\)

当前实现将反向流定义为正向流的对称：使用同一套策略网络在反向拓扑上计算 \(\log P_B\)。

对 move 动作 \(u\to v\)：
\[
\log P_B(s_t\mid s_{t+1})=\log \pi_\theta(u\mid s_{t+1}),
\]
对 STOP 动作：
\[
\log P_B=\log P_F(\text{STOP}\mid s_{t+1}).
\]

注意：super node 不再作为特例处理，保持与正向流完全对称。

---

## 5. 终止奖励

对终止节点（或 `stop_prev_nodes`）计算 hit：

- 命中目标集合：\(R=1\)
- 未命中：\(R=\epsilon\)

即
\[
R(\tau)=
\begin{cases}
1, & \text{hit}\\
\epsilon, & \text{miss}
\end{cases}
\]
训练中使用 \(\beta\log R\)。

对应实现：

- hit mask：`src/models/reward/reward_engine.py:71`
- reward 计算：`src/models/reward/reward_engine.py:278`

---

## 6. Forward-Looking SubTB 推导（核心）

记一条 rollout 为 \(\tau\)，长度为 \(L\)（按实现中的 `num_steps` 有效步定义）。

定义每步差分：
\[
\Delta_i=\log P_F(a_i\mid s_i)-\log P_B(s_i\mid s_{i+1}).
\]

Forward-Looking 约束对每个前缀状态 \(s_t\) 要满足：
\[
\log F(s_t)+\sum_{i=t}^{L-1}\Delta_i=\beta\log R(\tau).
\]

残差写为
\[
\varepsilon_t=
\log F(s_t)+\sum_{i=t}^{L-1}\Delta_i-\beta\log R(\tau).
\]

实现中通过“suffix cumsum”一次性向量化构造：
\[
\text{suffix\_delta}_t=\sum_{i=t}^{L-1}\Delta_i.
\]

再做加权平方：
\[
\mathcal L_{\text{SubTB}}(\tau)=
\frac{\sum_t \lambda^{d_t}\varepsilon_t^2}{\sum_t \lambda^{d_t}},
\]
其中 \(d_t\) 是到轨迹末端的距离（代码里由 `effective_state_len` 和 `step_idx` 计算）。

最终 batch loss 为 valid rollout 上平均。

对应实现：

- 残差构造：`src/models/objectives/subtb_loss.py:76`
- \(\lambda\) 加权与归一化：`src/models/objectives/subtb_loss.py:30`
- 主 forward：`src/models/objectives/subtb_loss.py:123`

---

## 7. 总训练目标

训练目标由三部分组成：
\[
\mathcal L=
\mathcal L_{\text{fwd}}
+w_{\text{bwd}}\mathcal L_{\text{bwd}}
+w_{\text{rank}}\mathcal L_{\text{rank}}.
\]

其中：

- \(\mathcal L_{\text{fwd}}\)：前向 rollout 的 SubTB
- \(\mathcal L_{\text{bwd}}\)：后向 rollout（反图）上的 SubTB
- \(\mathcal L_{\text{rank}}\)：答案质量辅助损失（可关）

对应实现：`src/models/algorithms/dual_flow.py:393`

### 7.1 排序辅助项（可选）

当 `ranking_weight>0` 时，节点打分 \(z_v\) 经温度 \(T_r\) 后，最小化答案质量负对数：
\[
\mathcal L_{\text{rank}}
=-\log\frac{\sum_{v\in A}\exp(z_v/T_r)}{\sum_{v\in V}\exp(z_v/T_r)}.
\]

对应实现：`src/models/algorithms/dual_flow.py:432`

---

## 8. 训练流程（与代码一致）

```text
for each batch:
  1) 构建 GraphEnvContext（含 CSR、q/a 索引、question_ctx）
  2) 前向 rollout 采样，得到 log_pf/log_pb/log_f 轨迹张量
  3) 根据 stop 节点计算奖励 R 与 hit_mask
  4) 计算前向 SubTB loss
  5) 若 ranking_weight > 0：
       计算 ranking 辅助损失
  6) 加权求和，自动优化器更新
```

对应实现入口：

- 训练 step：`src/models/algorithms/dual_flow.py:538`
- 在线 rollout：`src/rollout/engine/engines/online.py:100`

---

## 9. 评估与数据可见性

- 验证/测试/预测统一使用 exact-flow 的答案分布导出路径；
- 每个样本的导出路径数由 exact-flow 的 top-entity 概率分布自适应决定，而不是固定 beam 大小；
- 指标按 `stage/scope` 打前缀，scope 由 datamodule 注入（`full`/`sub`）；
- `eval.py` 强制 DualFlow 评估同时提供 full 与 sub 变体，防止隐式覆盖。

对应实现：

- exact-flow 导出：`src/models/algorithms/dual_flow.py`
- exact planner：`src/models/planner/exact_flow.py`
- scope 注入：`src/datasets/g_retrieval_datamodule.py`
- 双域校验：`src/eval.py`

---

## 10. 设计动机复盘

1. **防标签泄漏**  
   Policy 前向不读取答案标签，答案仅在 reward/metric 端使用；有单测守护。

2. **结构化动作建模**  
   关系级 + 条件级分解，把巨大的边空间 softmax 分成可学习的两层结构。

3. **稳定终止概率**  
   STOP 与 move 总质量解耦，得到近似 logistic 形式，训练更稳。

4. **双向一致性约束**  
   前后向都用 SubTB 约束同一势函数，减少单向训练偏差。

对应测试：

- label leakage guard：`tests/test_policy_label_leakage_guard.py:172`
- backward flow contract：`tests/test_backward_flow_contract.py:173`

---

## 11. 当前局限（实现层面）

1. **动作空间仍受子图裁剪**  
   训练/评估语义定义在 `g_retrieval` 子图，而非全图。

2. **edge dropout 未启用且已移除配置入口**  
   数据配置若出现 legacy edge dropout 字段会直接报错。

3. **P_B 父节点候选被错误过滤**  
   反向父节点候选在收集阶段不应依赖 visited_mask；否则会错误排除真实父节点，破坏后向支撑集。该问题发生在 `src/models/policy/action.py` 的 parent gathering 逻辑（由 `src/models/policy/dual_flow_policy.py` 调用）。

对应实现：

- edge dropout 配置拒绝：`src/datasets/g_retrieval_dataset.py:394`
- P_B 父节点过滤问题：`src/models/policy/action.py:121`

---

## 12. 可直接复现的关键配置

- 模型主配置：`configs/model/dual_flow.yaml:1`
- 训练实验配置：`configs/experiment/train_dual_flow.yaml:1`
- 数据入口配置：`configs/data/g_retrieval.yaml:1`

关键超参数（当前默认）：

- `num_rollouts=8`
- `stop_min_steps=1`
- `sampling.max_steps=${dataset.max_steps}`（当前数据集多为 4，按真实边数计）
- `subtb.lambda_weight=0.8`（实验覆盖）
- `subtb.ranking_weight=0.0`
