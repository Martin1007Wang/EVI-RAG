
== 架构实现 <sec:architecture>
本节给出 *code-exact* 的 DualFlow/GFlowRAG 说明：将理论变量（状态 $s_t$、前向策略 $P_F$、后向先验 $P_B$、状态流 $Z_\theta$）
逐一映射到 `src/models/dual_flow_module.py` 的实际实现，并明确哪些量参与 *step logits*、哪些量仅用于 *DB loss*。

=== 算法流程总览 (Code-Exact)
对每个样本（检索子图 $G_"sub"$）：
- 图编码：一次性将节点/边/问题嵌入投影到隐藏维度，并可选地做浅层关系 GNN 聚合，得到静态 token：$bold(h)(v)$、$bold(r)(e)$、$bold(h)_q$。
- 前向 rollout：从 $q_"local"indices$ 采样单一起点，使用策略 $P_F$ 在 forward 边空间上采样至 STOP/命中/死路/步数上限。
- 后向 rollout：从 $a_"local"indices$ 均匀采样单一起点，使用固定后向先验 $P_B$ 在 backward（inverse）边空间上游走至命中/死路/步数上限。
- 训练：将两种 rollout 都转化为“forward 边序列”，用同一 Detailed Balance (DB) 残差计算损失并取平均。

=== 状态表示：节点 + 时间 + 历史 + 上下文
实现中，状态 $s_t$ 的可学习输入不是一个“单向量黑盒”，而是以下可分解的四元组：

*静态节点 token.* 子图节点经投影与（可选）关系 GNN 编码后得到 $bold(h)(v) in bb(R)^d$。该 token 与时间/路径无关，只随子图变化。

*时间编码.* 通过正弦位置编码得到 $bold(e)_"time"(t) in bb(R)^d$，显式建模有限视界 MDP 的时间坐标。

*历史记忆.* 用 GRU 对路径上的“关系序列”做递归编码：
$ bold(h)_"hist"^t = op("GRU")(bold(h)_"hist"^(t-1), bold(r)_(t-1)), quad bold(h)_"hist"^0 = bold(e)_"null". $
其中 $bold(r)_(t-1)$ 是上一跳选择边的关系 token。实现对应 `self.path_gru` 与 `self.null_relation_emb`。

*查询条件上下文.* 对每个图构造一个条件向量 $bold(c)_"flow" in bb(R)^d$：
$ bold(c)_"flow" = op("MLP")([bold(h)_q; bold(h)_(v_0)]). $
其中 $v_0$ 是起点（训练时由 StartSelector 采样；评估时对所有起点做 multi-start 展开）。

我们将 *head state* 定义为
$ bold(x)_t := bold(h)(v_t) + bold(e)_"time"(t) + bold(h)_"hist"^t, $
并将 $(bold(x)_t, bold(c)_"flow")$ 视为策略与 $Z$ 网络共同消费的最小马尔可夫充分统计量。

=== 解耦的两网结构：$Z_\theta$ 只进 Loss，$pi_\theta$ 只管转移
实现采用显式解耦（Decoupling）：
- $Z_\theta$ 网络（`z_predictor`）：输出 $log Z_\theta(s_t)$，*不参与* step logits；仅用于 DB loss 的 $log Z$ 项与终止接地（grounding）。
- 策略网络 $pi_\theta$（`policy_fwd` + `stop_predictor`）：专门用于 $u -> v$ 的转移打分与 STOP 打分。

=== Step Logits：Query-Conditional Bilinear + InDegree Prior
对任意候选边 $e=(u, r, v)$（forward 边空间内），step logit 的神经部分由 Query-Conditional Bilinear scorer 给出：

$ score_\theta(e|s_t)
= frac(1, sqrt(d)) sum_(j=1)^d bold(x)_t[j] * (bold(r)(e)[j] + (W_c bold(c)_"flow")[j]) * bold(h)(v)[j]. $

实现对应 `BilinearStepScorer`：将上下文 $bold(c)_"flow"$ 线性变换后加到关系 token 上（context shift），并做三线性积后乘 $d^{-1/2}$。
随后注入可学习放大器（amplifier）：
$ ell_"nn"(e|s_t) = alpha * score_\theta(e|s_t), quad alpha = exp(logit_scale) > 0. $

拓扑先验采用子图内的 *入度惩罚*（forward 掩码下）：
$ ell_"topo"(e) = -log d_"in"(v). $
其尺度通过可学习的非负权重控制：
$ lambda_"prior" = op("softplus")(w_"prior") >= 0, quad ell(e|s_t) = ell_"nn"(e|s_t) + lambda_"prior" ell_"topo"(e). $

上述设计的关键点是：打分只需要 head 的动态状态 $bold(x)_t$，tail 侧仅使用静态 $bold(h)(v)$，避免对每个邻居显式构造复杂的 $v$ 侧状态。

=== STOP Logit：独立的终止预测器 + 最小步约束
STOP 动作与扩展动作在同一 softmax 中竞争。实现中 STOP logit 由独立网络给出：
$ ell_"stop"(s_t) = g_\theta([bold(x)_t; bold(c)_"flow"]). $
并施加硬约束：
- `runtime_cfg.stop_min_steps`: 当 $t < "min_steps"$ 时强制 $ell_"stop" = -infinity$（禁止过早终止）。
- 命中答案且 $t >= "min_steps"$ 时强制 STOP；达到最大步数也强制 STOP（保证 episode 有界）。

=== 后向先验 $P_B$：inverse 边上的静态均匀分布
后向策略在 backward（inverse）边空间上取静态均匀分布：
$ log P_B(s_t | s_(t+1)) = -log |cal(Out)_b(s_(t+1))|. $
由于实现显式构造逆边集合（并校验 inverse map 对称性），该项在实践中等价于对 forward 入度的均匀先验（以实现中的方向掩码为准）。

=== 训练目标：双向 rollout 的 Detailed Balance 残差
对采样得到的 edge 序列（均以 forward 边 id 表示），DB 残差在每一步 $t$ 上为：
$ delta_t =
  (log Z_\theta(s_t) + log P_F(a_t|s_t))
  - (log Z^dagger(s_(t+1)) + log P_B(s_t|s_(t+1))). $

其中 $log Z^dagger$ 是终止接地（hard grounding）的 *代码口径*：
- 终止命中：$log Z^dagger = 0$（对应 $R=1$）。
- 终止失败（死路/超时等）：$log Z^dagger = "dead_end_log_reward"$（对应 $R=epsilon$）。

若 rollout 中显式选择了 STOP，还会额外加入 STOP 残差项：
$ delta_"stop" = log Z_\theta(s_"stop") + log P_F("STOP"|s_"stop") - log R, $
其中 $log R$ 在代码中取：
`0`（命中答案）、`emit_log_reward`（主动 STOP 但未命中）、或 `dead_end_log_reward`（其他失败）。

最终损失为加权均方：
$ cal(L)(theta) = op("mean")( sum delta_t^2 + delta_"stop"^2 ), $
并对 forward rollout 与 backward rollout 的损失取平均（训练时默认每个 batch 重复多个 rollout 取均值）。

=== 生成式证据蒸馏（推理/评估）
推理阶段使用 fixed-budget 的 beam search（支持 diverse beam）在子图内生成 Top-$K$ 轨迹，并输出：
- 终止实体（预测答案）
- 轨迹边序列（结构化证据）

随后可将轨迹线性化为文本证据链，交由 LLM 阅读器做最终答案归纳，以降低“纯生成”带来的知识幻觉并提升可解释性。
