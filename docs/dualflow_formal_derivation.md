# DualFlow 当前实现的形式化推导（代码对齐版）

本文档给出当前代码实现（`src/models/dual_flow_module.py`）的数学语义与训练/评估流程。  
目标是让“符号定义 = 运行时行为”，并覆盖当前 `predict + eval_llm` 的关键接口语义。

---

## 1. 数据语义与图构造

对每个样本图 \(b\)，给定检索子图
\[
G_b=(V_b,E_b),\qquad E_b\subseteq V_b\times\mathcal R\times V_b.
\]

数据层 SSOT（不可交换）：
\[
Q_b=\texttt{q\_local\_indices},\qquad
A_b=\texttt{a\_local\_indices}.
\]

方向定义固定：
- forward: \(Q_b \rightarrow A_b\)
- backward: \(A_b \rightarrow Q_b\)

训练图掩码：
\[
\texttt{graph\_mask}=\neg \texttt{dummy\_mask},
\]
其中 `dummy_mask` 仅由 `answer_entity_ids_ptr` 推导。

---

## 2. Super-Source 注入（当前默认启用）

当 `runtime_cfg.super_source_enabled=true` 时，每图新增一个虚拟根节点 \(s^\star_b\)：
\[
V'_b=V_b\cup\{s^\star_b\}.
\]

并添加根到问题实体的边：
\[
(s^\star_b, r_{\text{ss}}, q),\quad q\in Q_b.
\]

语义要点：
- `start_nodes_fwd` 直接取每图根节点（不再从 \(Q_b\) 抽样）。
- `super_source_injection` 在当前严格契约下必须是 `flow`。
- `r_ss` 使用保留 ID（默认 `2147483647`），并对 relation id 冲突做 fail-fast 检查。
- 运行时严格契约还要求：`stop_enabled=true`、`p0_mode=semantic`。

---

## 3. 参数化对象

定义状态势能：
\[
f_\theta(v;c)\equiv \log F_\theta(v;c),
\]
由 `z_predictor` 给出。

上下文：
- 前向上下文 \(c_b^{\mathrm{fwd}}\)：由 question token 与起点 token 融合。
- 后向上下文 \(c_b^{\mathrm{bwd}}\)：由 question token 与答案集合 pooled token 融合。

---

## 4. 前向策略 \(P_F\)（分层）

对状态 \(u\) 与候选边 \(e=(u,r,v)\)，代码中的基础 edge logit 为
\[
\psi_\theta(e\mid u,t)
=
\omega\log P_0(e\mid u,t,c)
+\omega\cdot\bigl(-\log d^-_{\mathrm{in}}(v)\bigr)
+\alpha f_\theta(v;c)
+\log\gamma,
\]
其中：
- \(\alpha=\exp(\texttt{logit\_scale})\)
- \(\omega=\texttt{prior\_weight}\)
- \(\gamma\in(0,1]\)
- \(P_0\in\{\texttt{uniform},\texttt{indicator},\texttt{semantic}\}\)（当前严格配置为 `semantic`）。

温度 \(T\) 时使用 \(\psi_\theta/T\)。

关系级聚合：
\[
m_\theta(r\mid u,t)=\log\sum_{e\in\mathcal E(u,r)}\exp(\psi_\theta(e\mid u,t)),
\]
\[
\Lambda_\theta(u,t)=\log\sum_r\exp(m_\theta(r\mid u,t)).
\]

STOP 质量（显式 STOP action）：
\[
\log R_{\text{stop}}(u)=
\begin{cases}
0, & u\in A_b,\\
\log\epsilon_b, & u\notin A_b.
\end{cases}
\]
\(\epsilon_b\) 由 `stop_reward_mode` 决定（`constant` 或 `uniform_node`）。

归一化：
\[
D_\theta(u,t)=\log\Big(\exp(\log R_{\text{stop}}^{\text{masked}}(u))+\exp(\Lambda_\theta(u,t))\Big).
\]

于是
\[
\log P_F(\text{STOP}\mid u,t)=\log R_{\text{stop}}^{\text{masked}}(u)-D_\theta(u,t),
\]
\[
\log P_F(e\mid u,t)=\psi_\theta(e\mid u,t)-D_\theta(u,t).
\]

采样实现是分层 Gumbel-Max（relation 与 STOP 竞争，再在 relation 内选 edge）。

---

## 5. 终止与 rollout 语义（P0 后）

前向 rollout（`_rollout_policy`）每步遵循：
1. 先检查当前节点是否在目标集合 \(A_b\)，若是则立即 `HIT`。
2. 计算可行动作；若无可行动作，STOP 可用。
3. `stop_min_steps`：当 \(t<\texttt{stop\_min\_steps}\) 时禁止主动 STOP。
4. 采样 STOP 或 move。

关键更新（已落地）：
- **不再**在最后一步强制 STOP（删除 `force_last`）。
- 若循环结束仍 active，则终止原因为 `MAX_STEPS`。

终止奖励（SubTB 用）：
\[
\log R(\tau)=
\begin{cases}
0, & \text{terminal=HIT},\\
\log\epsilon_b-\eta\cdot L(\tau), & \text{otherwise},
\end{cases}
\]
其中 \(\eta=\texttt{miss\_length\_penalty}\)，默认可为 0。

---

## 6. 反向策略 \(P_B\) 与后向 rollout

`P_B` 作用在反向候选边集合上，支持：
- `uniform`
- `outdegree`
- `learned`

当前默认配置是 `uniform`。

后向 rollout（`_rollout_pb`）：
- 起点从 \(A_b\) 均匀采样。
- 目标集合是 \(Q_b\)。
- 命中目标即 `HIT`。
- 无动作 `DEAD_END`；超步 `MAX_STEPS`。

---

## 7. SubTB 损失（代码等价）

定义逐步差值
\[
\Delta_{b,t}=\log P_{F,b,t}-\log P_{B,b,t}.
\]

在 stop 步处，用 \(\log P_F(\text{STOP})\) 覆盖 forward 项。  
状态势能序列（长度为 `max_steps`）：
\[
g_{b,t}=f_\theta(s_{b,t};c_b^{\mathrm{fwd}}).
\]

段终点目标：
\[
\hat g_{b,u}=
\begin{cases}
g_{b,u+1}, & u<t_{\text{stop}},\\
\log R(\tau_b), & u=t_{\text{stop}}.
\end{cases}
\]

残差：
\[
r_{b,t,u}=\sum_{k=t}^{u}\Delta_{b,k}+g_{b,t}-\hat g_{b,u}.
\]

带 \(\lambda\) 权重的 SubTB：
\[
\mathcal L_{\text{SubTB}}
=
\frac{
\sum_b w_b\sum_{0\le t\le u<L_b}\lambda^{u-t}r_{b,t,u}^2
}{
\sum_b w_b
}.
\]

其中：
- `normalize=true` 时，先做每图段内归一，再跨图加权。
- \(w_b\) 可来自后向分支的图权重（见下一节）。

---

## 8. 双向离策略训练（Backward Guidance）

若 `backward_rollouts > 0` 且 `backward_weight > 0`：
1. 从 \(A_b\) 采样后向轨迹，筛选命中 \(Q_b\) 的成功样本。
2. 将动作按有效长度翻转为前向动作。
3. 计算
\[
\rho_b=\exp\left(
\log P_F(\tau_b^{\mathrm{rev}})-\log P_B(\tau_b^{\mathrm{rev}})
\right)
\]
（可关闭 IS，此时 \(\rho_b=1\)；当前实现不再对 \(\rho_b\) 做截断，若出现非有限值直接 fail-fast）。
4. 图权重
\[
w_b^{\mathrm{bwd}}=\lambda_{\mathrm{bwd}}\rho_b.
\]
5. 用同一个 SubTB 公式计算后向分支。

总损失：
\[
\mathcal L=
\frac{\mathcal L_{\mathrm{fwd}}+\beta_{\mathrm{bwd}}\mathcal L_{\mathrm{bwd}}}{1+\beta_{\mathrm{bwd}}}
+\eta_{\mathrm{ans}}\mathcal L_{\mathrm{ans\_mass}}.
\]

注：
- 当前代码没有独立的 “PB auxiliary MLE” 项。
- `detach_pb_in_pf_loss` / `detach_pb_in_rho` 控制 PB 梯度解耦路径。
- 训练步骤不再执行梯度范数裁剪（no grad clipping）。

---

## 9. Beam / Sampling / Predict 导出（P0+P1 对齐）

### 9.1 Eval candidates

`_compute_eval_metrics`：
- 若 `sampling_eval.enabled=true`，用采样候选集（默认开启）。
- 否则用 beam。

### 9.2 Beam 完结语义

beam 搜索结束后执行 horizon finalization：
\[
\texttt{beam\_done} \leftarrow \texttt{beam\_done} \lor (\texttt{beam\_nodes}\ge 0),
\]
避免 `require_done=True` 时丢失有效候选。

### 9.3 Predict 导出策略

`evaluation_cfg.predict_export` 支持：
- `beam`
- `sampling`
- `union`

`union` 会按 `dedup in {path,node}` 合并并按 score 保留 `max_candidates`。

---

## 10. LLM 推理链路（与 DualFlow 产物对齐）

`eval_llm` 当前默认：
- prompt mode: `subgraphrag_icl_dc`
- `constrain_to_candidates=true`
- vLLM: `seed`、`pretrim_to_budget`、`budget_margin`

实现要点：
1. 轨迹文本 fallback（当 `trajectory_text` 缺失）会过滤 super-source 边。
2. 最终答案可做“候选约束”（只保留在候选终点集合中的答案）。
3. prompt 超预算时先做预算裁剪，再调用模型。

---

## 11. 指标语义（当前实现）

LLM 侧：
- `no_ans` 判定优先基于已解析有效 `ans:` 行，避免“有答案却被判 no_ans”。
- `sub` 作用域也累计 HAL，`sub/hal_score` 不再退化常数。

DualFlow 侧：
- 评估指标区分 reach/hit/pass、top-k 命中与覆盖，并按有效图平均。

---

## 12. 代码锚点

- Batch 准备与 super-source：`_prepare_batch`, `_maybe_augment_super_source`
- 前向分层概率：`_compute_hierarchical_log_probs_with_optional_prior`
- 前向 rollout：`_rollout_policy`
- PB 与后向 rollout：`_compute_pb_log_prob`, `_rollout_pb`
- SubTB：`_compute_subtb_loss`, `_compute_subtb_loss_from_delta`
- 训练总损失：`_compute_training_loss`
- Eval 候选：`_sample_eval_candidates`, `_compute_eval_metrics`
- Predict 导出：`_resolve_predict_export_cfg`, `predict_step`
- LLM 解析：`_trajectory_text`, `_enforce_candidate_answers`, `_build_vllm_generate`
- LLM 指标：`compute_llm_metrics`, `_subgraphrag_no_answer`

以上定义即当前代码基线，可直接作为实现约束与方法描述来源。
