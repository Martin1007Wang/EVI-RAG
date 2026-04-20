# EVI-RAG / RankFlow 算法数学推导

本文档严格基于当前仓库代码实现与默认 Hydra 配置，总结本仓库主线 `RankFlow` 模型实际对应的数学对象、状态转移、策略参数化、奖励定义、训练目标与评估指标。

本文档对应的核心实现文件如下：

- `configs/model/gflownet.yaml`
- `configs/train.yaml`
- `configs/pipeline/default.yaml`
- `src/models/gflownet.py`
- `src/models/state.py`
- `src/models/policy.py`
- `src/models/rollout.py`
- `src/models/losses.py`
- `src/models/reward.py`
- `src/models/modules/backbone.py`
- `src/models/modules/heads.py`
- `src/eval/metrics.py`
- `src/utils/graph_utils.py`
- `src/utils/path_utils.py`

当前主线模型不是“纯 Detailed Balance GFlowNet”，也不是“节点-关系-边三级动作”的旧式表述。按代码事实，它是一个：

- 以问题 `q` 和候选图 `G` 为条件的增量式子图扩张 GFlowNet
- 使用前向 `FL-SubTB` 训练目标的流模型
- 使用显式结构规则定义的一步后向策略 `P_B`
- 在终止子图上定义 reward
- 训练时带 teacher 正边 bonus 与 reward-matching 正则

## 1. 默认配置对应的具体算法

默认训练模型配置来自 `configs/model/gflownet.yaml`：

```yaml
max_steps: 3
num_rollout: 8
temperature: 1.0
temperature_schedule:
  start: 0.5
  end: 1.0
  warmup_steps: 2000

reward:
  recall_scale: 5.0
  connectivity_penalty: 3.0
  edge_penalty: 0.05
  log_r_min: -5.0

loss:
  variant: fl_subtb
  subtb_lambda: 0.9
  positive_edge_bonus: 1.0
  reward_matching_coef: 0.1
```

因此默认训练目标可先概括为：

$$
\mathcal L
=
\mathcal L_{\mathrm{FL\text{-}SubTB}(\lambda=0.9)}
+
0.1\,\mathcal L_{\mathrm{RM}}.
$$

其中终止奖励的默认形式为：

$$
\log R(x)
=
5\,\mathrm{recall}(x)
- 3\,\delta(x)
- 0.05\,|E_x|,
$$

再叠加 no-hit floor 与全局截断，详见第 8 节。

## 2. 数据、图与监督对象

### 2.1 候选图

对每个样本，预处理和物化阶段会构造一个局部候选图：

$$
G^{\mathrm{base}}=(V,E).
$$

并提供下列张量字段，对应 `RetrievalBatch`：

- `question_emb`，记为问题条件向量 `q`
- `is_anchor_mask`，对应锚点集合 $A \subseteq V$
- `is_target_mask`，对应图中可见金答案集合 $Y \subseteq V$
- `positive_edge_mask`，对应 teacher 正边集合 $E^+ \subseteq E$

因此模型学习的对象不是完整知识图上的答案后验，而是候选图内部的终止子图分布：

$$
P_\theta(x\mid q,G^{\mathrm{base}},A),
$$

其中 $x$ 是一个终止子图状态。

### 2.2 正边弱监督

`positive_edge_mask` 来自 `src/utils/path_utils.py::compute_shortest_path_teacher_targets`。默认预处理配置 `configs/pipeline/default.yaml` 中：

```yaml
path_mode: qa_directed
```

因此默认以有向最短路定义 teacher 边。对任意边 $e=(u,v)$，若存在锚点 $a\in A$ 与可达答案 $y\in Y$ 满足：

$$
d(a,u)+1+d(v,y)=d(a,y),
$$

则这条边属于某条 anchor-to-answer 最短路径，被标记为正边。即

$$
E^+=\left\{(u,v)\in E\mid \exists a\in A,\exists y\in Y,\ d(a,u)+1+d(v,y)=d(a,y)\right\}.
$$

当前主线训练中，这个弱监督只通过 rollout 里的 step bonus 使用，不直接进入终止奖励函数。

## 3. MDP 形式化

### 3.1 状态

`src/models/state.py` 中的状态可形式化为：

$$
s_t=(V_t,E_t,\phi_t),\qquad \phi_t\in\{\mathrm{ACTIVE},\mathrm{TERMINAL}\}.
$$

其中：

- $E_t$ 对应 `active_edges`
- $V_t$ 对应 `active_nodes`
- $\phi_t$ 对应 phase

语义上，$V_t$ 是由当前活跃边与锚点共同诱导的活跃节点集合。

### 3.2 初始状态

`State.create_initial()` 定义初始状态为锚点诱导子图：

$$
V_0=A,
$$

$$
E_0=\{(u,v)\in E\mid u\in A,\ v\in A\}.
$$

因此：

$$
s_0=(A,E_0,\mathrm{ACTIVE}).
$$

也就是说，轨迹不是从空图开始，而是从锚点及其诱导边开始。

### 3.3 动作空间

代码中存在两类动作：

- `expand`
- `stop`

若执行 `expand`，则需要从候选 frontier 边中选择一条边。`Policy._build_candidate_mask()` 给出的候选集合是：

$$
\mathcal C(s_t)
=
\left\{e=(u,v)\in E\setminus E_t\mid u\in V_t\ \text{or}\ v\in V_t\right\}.
$$

同时，`Policy` 默认 `undirected=True`，会额外施加 `src < dst` 的去重条件。因此更精确地，当前真正参与 expand 采样的集合是：

$$
\mathcal C_{\mathrm{canon}}(s_t)
=
\left\{e=(u,v)\in E\setminus E_t\mid (u\in V_t\lor v\in V_t),\ u<v\right\}.
$$

于是动作空间可以写为：

$$
\mathcal A(s_t)=\{\mathrm{stop}\}\cup\{\mathrm{add}(e):e\in \mathcal C_{\mathrm{canon}}(s_t)\}.
$$

### 3.4 状态转移

若在状态 $s_t$ 选择边 $e_t=(u_t,v_t)$，则 `State.apply_expansion()` 对应的转移为：

$$
E_{t+1}=E_t\cup\{e_t\},
$$

$$
V_{t+1}=V_t\cup\{u_t,v_t\}.
$$

若选择 `stop`，则 `State.apply_stop()` 把状态转为终止态，但不改变子图结构：

$$
s_{t+1}=(V_t,E_t,\mathrm{TERMINAL}).
$$

### 3.5 有限时域

`RolloutEngine(max_steps=3)` 会限制最大扩张步数。实现上缓冲长度为：

$$
T = \mathrm{max\_steps}+1 = 4,
$$

因为轨迹记录的是“状态数”，而状态数 = 转移数 + 1。

代码中当 `t >= max_steps` 时强制 stop，因此：

- 最多发生 `max_steps = 3` 次 expand
- 终止动作仍计入轨迹长度
- `traj_len` 表示 transition 总数，包含 stop

## 4. 前向策略分解

`src/models/rollout.py` 中，前向动作由两级采样组成：

1. 先采样动作类型 `expand/stop`
2. 若为 `expand`，再在候选边集合上采样一条边

因此目标前向策略可以写为：

$$
P_F^\theta(a_t\mid s_t)
=
\begin{cases}
P_{\mathrm{type}}^\theta(\mathrm{stop}\mid s_t), & a_t=\mathrm{stop},\\
P_{\mathrm{type}}^\theta(\mathrm{expand}\mid s_t)\cdot P_{\mathrm{edge}}^\theta(e_t\mid s_t,\mathrm{expand}), & a_t=\mathrm{add}(e_t).
\end{cases}
$$

下面分别写出各部分的参数化。

## 5. 表示学习与网络参数化

### 5.1 Backbone 投影

`src/models/modules/backbone.py` 中，节点、关系、问题 embedding 先投影到隐藏维 `H=512`：

$$
h_v^{(0)} = W_n\,\mathrm{LN}(x_v),
$$

$$
r_e = W_r\,\mathrm{LN}(z_e),
$$

$$
q = W_q\,\mathrm{LN}(q_{\mathrm{raw}}).
$$

对 anchor 节点，初始 node state 直接被问题向量替换：

$$
h_v^{\mathrm{input}}=
\begin{cases}
q, & v\in A,\\
h_v^{(0)}, & v\notin A.
\end{cases}
$$

### 5.2 Edge state id

backbone 不是只编码当前子图，而是编码整张候选图，同时用 edge state id 标记当前 rollout 状态。代码中：

- `0`: inactive edge
- `1`: frontier edge
- `2`: traversed edge

记为：

$$
\xi_e(s_t)\in\{0,1,2\}.
$$

只有 `edge_state_id > 0` 的边参与 NBF 消息传递。

### 5.3 NBF-style 消息传递

每一层消息传递都用问题条件向量调制。忽略 LayerNorm 和 dropout 后，可写成：

$$
m^{\mathrm{fwd}}_{u\to v} = \mathrm{MLP}_{\mathrm{fwd}}([h_u,r_{uv},q]),
$$

$$
m^{\mathrm{bwd}}_{v\to u} = \mathrm{MLP}_{\mathrm{bwd}}([h_v,r^{-1}_{uv},q]),
$$

$$
\bar m_v = \mathrm{mean}_{u:(u,v)\in E_{\xi>0}} m^{\mathrm{fwd}}_{u\to v}
+ \mathrm{mean}_{u:(v,u)\in E_{\xi>0}} m^{\mathrm{bwd}}_{v\to u},
$$

$$
h_v' = \mathrm{LN}(h_v + W_u\bar m_v).
$$

默认 `gnn_num_layers = 3`，因此这个更新重复 3 次。

### 5.4 Flow state 与 action state 的分离

`policy.py` 明确区分两种图级状态表示：

- `flow_state_h` 只用于 `FlowHead/ZHead`
- `state_h` 用于动作打分

这对应两层不同的数学对象。

#### Flow state

`_encode_flow_state()` 只对 active nodes 做 query-conditioned attention pooling：

$$
\alpha_i = \mathrm{softmax}_{i\in V_t}\big(a([h_i,q])\big),
$$

$$
h^{\mathrm{flow}}(s_t)=\sum_{i\in V_t}\alpha_i h_i.
$$

这是当前代码里的 Markov-faithful flow readout。

#### Action state

`_encode_state()` 把以下三路图级信息拼接后再编码：

- `node_pool = h^{flow}(s_t)`
- `edge_pool =` 已激活边上下文的均值池化
- `anchor_pool =` 锚点节点静态表示均值池化

因此：

$$
h^{\mathrm{state}}(s_t)
=
\mathrm{MLP}_{\mathrm{state}}
\left(
\left[
h^{\mathrm{flow}}(s_t),
\mathrm{mean}_{e\in E_t} c^{\mathrm{act}}_e,
\mathrm{mean}_{v\in A} h_v^{\mathrm{static}}
\right]
\right),
$$

其中

$$
c^{\mathrm{act}}_e = \mathrm{Enc}_{\mathrm{act}}([h_u,r_e,h_v,q]).
$$

## 6. `log Z`、状态流与动作分布

### 6.1 根流 `log Z`

`Policy.root_log_z()` 用 `ZHead` 预测根状态流：

$$
\log Z_\theta(q)
=
\log \widetilde F_\theta(s_0\mid q)
=
\mathrm{ZHead}(q,h^{\mathrm{flow}}(s_0)).
$$

### 6.2 中间状态流

`Policy.state_log_flow()` 用 `FlowHead` 预测中间状态流：

$$
\log \widetilde F_\theta(s_t\mid q)
=
\mathrm{FlowHead}(q,h^{\mathrm{flow}}(s_t)).
$$

### 6.3 Bilinear scalar head 的统一写法

`ZHead` 和 `FlowHead` 都继承自 `_MultiHeadBilinearScalar`。可以把它们统一写成：

$$
\mathrm{score}(q,s)
=
\sum_{m=1}^{M}\rho_m\langle Q_m q, K_m s\rangle
+ \mathrm{MLP}_{\mathrm{res}}([q,s,q\odot s]),
$$

其中 `M=4`，且

$$
\rho = \mathrm{softmax}(w^{\mathrm{head}}).
$$

于是：

$$
\log Z_\theta(q)=\mathrm{score}(q,h^{\mathrm{flow}}(s_0)),
$$

$$
\log \widetilde F_\theta(s_t\mid q)=\mathrm{score}(q,h^{\mathrm{flow}}(s_t)).
$$

### 6.4 动作类型分布

图级动作类型只有两列：

- 第 0 列：expand
- 第 1 列：stop

代码会先构造一组图级 `type_features`：

$$
f^{\mathrm{type}}(s_t)
=
\mathrm{Enc}_{\mathrm{type}}([q,\mathrm{cand\_pool},\mathrm{stats}(s_t)]).
$$

其中 `stats(s_t)` 的 7 个标量分量来自代码：

$$
\mathrm{stats}(s_t)=
\left[
\frac{|V_t|}{|V|},
\frac{|E_t|}{|E|},
\frac{|\mathcal C(s_t)|}{|E|},
\mathbf 1(|\mathcal C(s_t)|>0),
\frac{t}{T_{\max}},
\frac{T_{\max}-t}{T_{\max}},
\mathbf 1(T_{\max}-t=0)
\right].
$$

于是动作类型 logits 为：

$$
\ell^{\mathrm{type}}(s_t)
=
\mathrm{MLP}_{\mathrm{type}}([h^{\mathrm{state}}(s_t),f^{\mathrm{type}}(s_t)])\in\mathbb R^2.
$$

合法动作 mask 后得到：

$$
P_{\mathrm{type}}^\theta(g\mid s_t)
=
\mathrm{softmax}(\ell^{\mathrm{type}}(s_t))_g.
$$

### 6.5 候选边分布

对候选边 $e=(u,r,v)$，`ExpandEdgeScorer` 的分数由三项相加：

$$
\ell_e = \ell_e^{\mathrm{prior}} + \ell_e^{\mathrm{res}} + \ell_e^{\mathrm{bias}}.
$$

#### Semantic prior

代码中的 prior 用的是 query 与三元组语义表示的余弦相似度：

$$
\ell_e^{\mathrm{prior}}
=
\alpha\cdot
\cos\left(q, W_{\mathrm{tri}}[h_u^{\mathrm{static}},r_e,h_v^{\mathrm{static}}]\right).
$$

#### Gated residual

先用当前图状态和候选边上下文生成门控：

$$
g_e=\sigma\big(W_g[h^{\mathrm{state}}(s_t),c_e^{\mathrm{cand}}]+b_g\big),
$$

$$
\tilde c_e = c_e^{\mathrm{cand}}\odot(1+\gamma g_e),
$$

再计算 residual score：

$$
\ell_e^{\mathrm{res}}=\mathrm{MLP}_{\mathrm{res}}(\tilde c_e).
$$

#### Edge-state bias

还要加一项只依赖 edge state id 的偏置：

$$
\ell_e^{\mathrm{bias}}=b_{\xi_e(s_t)}.
$$

因此边选择分布为：

$$
P_{\mathrm{edge}}^\theta(e\mid s_t,\mathrm{expand})
=
\frac{\exp(\ell_e)}{\sum_{e'\in \mathcal C_{\mathrm{canon}}(s_t)}\exp(\ell_{e'})}.
$$

## 7. 行为采样分布与目标前向分布的区别

`src/models/rollout.py` 中一个关键实现细节是：

- rollout 采样时使用温度化的 behavior logits
- 写入 loss 的 `step_log_pf` 用的是未温度化 target log-prob

若温度为 $\tau$，则行为分布为：

$$
P_{\mathrm{beh}}(a\mid s)
\propto
\exp\left(\frac{\ell(a)}{\tau}\right),
$$

而进入 FL-SubTB 的目标前向分布是：

$$
P_F^\theta(a\mid s)
\propto
\exp(\ell(a)).
$$

默认训练温度随 global step 线性升温：

$$
\tau(g)
=
0.5 + (1.0-0.5)\cdot \min\left(\frac{g}{2000},1\right).
$$

因此当前实现是“用温度行为策略采样轨迹，但在 loss 中使用原始目标策略的 `\log P_F`”。代码中没有显式 importance weight 修正项。

## 8. 后向策略 `P_B`

### 8.1 可逆父状态的定义

`src/utils/graph_utils.py::compute_valid_backward_removals` 不是简单地把所有非 root 边都视为可删边，而是要求删除某条边后得到的父状态必须满足：

- 父状态本身是 forward-reachable
- 被删边在父状态中确实是合法 frontier expansion
- 从父状态重新添加该边后能精确恢复当前状态

记满足条件的可删边集合为：

$$
\mathcal R(s).
$$

其中 forward-reachable 的代码判定是：当前 active subgraph 的每个活跃连通分量都至少包含一个 anchor。

### 8.2 均匀后向策略

在当前主线实现中，后向策略是对合法可删边均匀分布：

$$
P_B(s'\mid s)
=
\begin{cases}
\frac{1}{|\mathcal R(s)|}, & s'\text{ 是删去一条合法可删边后的父状态},\\
0, & \text{otherwise}.
\end{cases}
$$

因此扩边后的 backward log-prob 为：

$$
\log P_B(s_t\mid s_{t+1}) = -\log |\mathcal R(s_{t+1})|.
$$

对 stop 终止动作，`RolloutEngine` 默认 `terminal_backward_mode="deterministic"`，所以：

$$
P_B(s_{L-1}\mid x)=1,
\qquad
\log P_B(s_{L-1}\mid x)=0.
$$

## 9. 终止奖励 `R(x)`

`src/models/reward.py` 给出的终止奖励定义在终止子图 $x=(V_x,E_x)$ 上。

### 9.1 Recall

记图内可见答案集合为 $Y$，终止子图命中的答案集合为 $V_x\cap Y$，则 recall 定义为：

$$
\mathrm{recall}(x)=\frac{|V_x\cap Y|}{|Y|},\qquad |Y|>0.
$$

代码中若某图没有任何 gold 节点，即 $|Y|=0$，则这类图不触发 no-hit floor，并按“无答案监督图”处理。

### 9.2 连通性惩罚

代码并不要求整个子图完全连通，而是只检查以下 key nodes：

- anchor 节点
- 已命中的 gold 节点

设：

$$
K_x = A \cup (V_x\cap Y).
$$

若这些 key nodes 分属多个连通分量，或者一个 key node 都没有被激活，则视为 disconnected。记指示变量：

$$
\delta(x)=
\mathbf 1\big(\#\mathrm{components}(K_x) \neq 1\big).
$$

### 9.3 边数惩罚

边数惩罚直接使用活跃边条数：

$$
m(x)=|E_x|.
$$

### 9.4 默认 reward 公式

默认配置下：

$$
\log R(x)
=
5\,\mathrm{recall}(x)
-3\,\delta(x)
-0.05\,m(x).
$$

### 9.5 No-hit floor 与全局截断

代码还额外施加两层规则。

若图中存在 gold 节点，但当前终止子图一个都没有 hit，则强制：

$$
|Y|>0,\quad |V_x\cap Y|=0
\Longrightarrow
\log R(x)=-5.
$$

随后再施加全局下界截断：

$$
\log R(x)\leftarrow \max(\log R(x),-5).
$$

因此始终有：

$$
\log R(x)\ge -5,
\qquad
R(x)\ge e^{-5}>0.
$$

这与 GFlowNet 所需的正奖励假设一致。

## 10. 训练时的正边 bonus

训练 rollout 时，若当前 expand 的边属于 `positive_edge_mask`，则会给该步加入一个 forward-looking bonus。默认配置中：

```yaml
positive_edge_bonus: 1.0
```

因此每一步的 bonus 为：

$$
\beta_t = \mathbf 1(e_t\in E^+).
$$

整条轨迹的累计 bonus 为：

$$
B(\tau)=\sum_t \beta_t.
$$

这部分只在训练 rollout 中启用；评估和推理时 `positive_edge_bonus=0.0`。

## 11. FL-SubTB 残差的精确定义

这是当前实现的核心。

记一条轨迹为：

$$
\tau=(s_0,a_0,s_1,a_1,\dots,a_{L-1},x),
$$

其中 $x$ 是 stop 后得到的终止状态，$L$ 是 transition 总数，包含 stop。

对每个有效步 $t$，代码维护：

- $f_t = \log \widetilde F_\theta(s_t\mid q)$
- $\log p_t^F = \log P_F^\theta(a_t\mid s_t)$
- $\log p_t^B = \log P_B(s_t\mid s_{t+1})$
- $\beta_t$ 为 step bonus

定义净对数比：

$$
\eta_t = \log p_t^F - \log p_t^B + \beta_t.
$$

则 `src/models/losses.py::_compute_residuals` 实现的子轨迹残差为：

$$
\Delta_{i,j}
=
f_i + \sum_{t=i}^{j}\eta_t - u_j,
\qquad 0\le i\le j\le L-1,
$$

其中 target $u_j$ 分两种情况：

$$
u_j=
\begin{cases}
f_{j+1}, & j<L-1,\\
\log R(x)+B(\tau), & j=L-1.
\end{cases}
$$

即：

$$
\Delta_{i,j}
=
\begin{cases}
f_i + \sum_{t=i}^{j}(\log p_t^F-\log p_t^B+\beta_t)-f_{j+1}, & j<L-1,\\
f_i + \sum_{t=i}^{L-1}(\log p_t^F-\log p_t^B+\beta_t)-\big(\log R(x)+B(\tau)\big), & j=L-1.
\end{cases}
$$

## 12. FL-SubTB 损失

`SubTrajectoryBalanceLoss` 中的几何权重矩阵是：

$$
w_{i,j}=\lambda^{j-i},\qquad i\le j,
$$

默认 `subtb_lambda = 0.9`。因此不是 detailed balance，也不是 pure trajectory balance，而是中间形态的 SubTB。

默认 `global_normalize=True`，所以 batch 上的 FL-SubTB 损失写为：

$$
\mathcal L_{\mathrm{FL\text{-}SubTB}}
=
\frac{
\sum_b\sum_{0\le i\le j<L_b}
w_{i,j}\,\Delta_{b,i,j}^2
}{
\sum_b\sum_{0\le i\le j<L_b}
w_{i,j}
}.
$$

这里 $L_b$ 是 batch 中第 $b$ 个样本轨迹的有效长度。

## 13. Reward-Matching 正则

代码中还加了一个 reward-matching 正则项：

$$
\mathcal L_{\mathrm{RM}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\left(f_{L_b-1}^{(b)}-\log R(x^{(b)})\right)^2.
$$

需要注意，按当前实现，`f_{L_b-1}` 对应的是 stop 前最后一个 active state 的 flow，不是 stop 后终止态单独重新估计的 flow。

默认总损失为：

$$
\mathcal L
=
\mathcal L_{\mathrm{FL\text{-}SubTB}}
+ 0.1\,\mathcal L_{\mathrm{RM}}.
$$

## 14. 代入默认配置后的最终训练目标

把默认超参数直接代入，可得当前主线默认优化的目标是：

$$
\mathcal L
=
\frac{
\sum_b\sum_{0\le i\le j<L_b}
0.9^{j-i}\,\Delta_{b,i,j}^2
}{
\sum_b\sum_{0\le i\le j<L_b}
0.9^{j-i}
}
+ 0.1\cdot
\frac{1}{B}
\sum_b
\left(f_{L_b-1}^{(b)}-\log R(x^{(b)})\right)^2,
$$

其中

$$
\Delta_{b,i,j}
=
f_{b,i} + \sum_{t=i}^{j}\left(\log p_{b,t}^F-\log p_{b,t}^B+\beta_{b,t}\right)-u_{b,j},
$$

$$
u_{b,j}=
\begin{cases}
f_{b,j+1}, & j<L_b-1,\\
\log R(x^{(b)})+B(\tau_b), & j=L_b-1.
\end{cases}
$$

且

$$
\beta_{b,t}=\mathbf 1(e_{b,t}\in E_b^+),
$$

$$
\log R(x)
=
\max\big(5\,\mathrm{recall}(x)-3\,\delta(x)-0.05|E_x|,-5\big),
$$

并在 no-hit 时强制置为 `-5`。

## 15. Monte Carlo 训练估计

`GFlowNetModule.training_step()` 中，对同一 mini-batch 会采样 `num_rollout = 8` 次 rollout，并对这些 rollout 的 loss 做平均后反向传播。因此 batch-level 训练估计可写为：

$$
\widehat{\mathcal L}_{\mathrm{batch}}
=
\frac{1}{8}\sum_{k=1}^{8}\mathcal L(\tau^{(k)}).
$$

这里每个 $\tau^{(k)}$ 都是一整个 batched rollout，即 batch 中每张图都独立走一条轨迹。

## 16. 优化器与学习率调度

`src/utils/optimization_utils.py` 使用 `AdamW`，默认：

- 基础学习率 `1e-4`
- `weight_decay = 1e-4`
- `betas = (0.9, 0.999)`

并把 `z_head` 和 `flow_head` 参数组的学习率乘以 5：

$$
\eta_{\mathrm{flow/z}}(t)=5\eta(t),
\qquad
\eta_{\mathrm{other}}(t)=\eta(t).
$$

调度器是 `cosine_with_warmup`，warmup steps 为 1000。若记总训练 horizon 为 $H$，则：

$$
\eta(t)=10^{-4}\cdot s(t),
$$

$$
s(t)=
\begin{cases}
\frac{t}{1000}, & t<1000,\\
\frac12\left(1+\cos\left(\pi\cdot\frac{t-1000}{H-1000}\right)\right), & t\ge 1000.
\end{cases}
$$

## 17. 推理与评估指标

`forward()` 和 `evaluate_subgraph_retrieval()` 都会生成若干 terminal rollouts。默认评估使用 `num_rollout = 8`，并把 `eval_budgets` 裁剪到不超过 rollout 数，因此有效 `K` 为：

$$
K\in\{1,2,4,8\}.
$$

### 17.1 Expected recall

`compute_distribution_expectations()` 中，单次 rollout 下的自然期望是：

$$
\mathrm{expected\_recall}
=
\mathbb E_{\tau\sim \pi_\theta}
\left[
\frac{|V_\tau\cap Y|}{|Y|}
\right].
$$

### 17.2 Best-of-K / oracle max recall

`compute_high_reward_discovery()` 中：

$$
\mathrm{oracle\_max\_recall@K}
=
\mathbb E\left[
\max_{1\le k\le K}
\frac{|V_{\tau_k}\cap Y|}{|Y|}
\right].
$$

同时定义 perfect recall 发现率：

$$
\mathrm{success@K}
=
\mathbb P\left(
\max_{1\le k\le K}
\frac{|V_{\tau_k}\cap Y|}{|Y|}=1
\right).
$$

### 17.3 子图规模与 dangling ratio

评估还统计：

- `expected_nodes`
- `expected_dangling_ratio`

其中 dangling edge 是指加进子图但不属于 anchor 或 active gold 保护核的边。数学上可视为：

$$
\mathrm{dangling\_ratio}(x)
=
\frac{|E_x^{\mathrm{added}}\setminus E_x^{\mathrm{core}}|}{|E_x^{\mathrm{added}}|},
$$

对所有 added-edge 非空的 `(graph, rollout)` 对求平均。

### 17.4 探索多样性

`compute_exploration_diversity()` 使用终止子图边集之间的 Jaccard 距离：

$$
d_{\mathrm{Jaccard}}(E_i,E_j)
=
1-
\frac{|E_i\cap E_j|}{|E_i\cup E_j|}.
$$

对每张图的 rollout 两两组合求平均，得到 `edge_jaccard_diversity`。

## 18. 当前实现中必须单独说明的事实

以下内容不是理论猜测，而是由代码直接决定的实现事实。

### 18.1 当前默认不是 Detailed Balance

因为 `configs/model/gflownet.yaml` 中：

```yaml
subtb_lambda: 0.9
```

所以默认训练目标是 `FL-SubTB(lambda=0.9)`，不是：

- `lambda = 0` 的 Detailed Balance
- 也不是 `lambda = inf` 的 Trajectory Balance

### 18.2 训练里真正用到的 teacher 信号只有正边 bonus

预处理虽然还产生了：

- `anchor_signed_distance`
- `node_to_target_distance`
- `shortest_suffix_count`
- `max_path_length`
- `heuristic_log_v`

但在当前主线训练代码中，真正进入优化目标的 teacher 信号主要是 `positive_edge_mask -> positive_edge_bonus`。其它字段目前未进入主线损失。

### 18.3 `Policy.max_steps` 与 `RolloutEngine.max_steps` 当前没有对齐

这是当前实现里最值得显式记录的一个细节。

`GFlowNetModule` 中：

```python
self.policy = Policy(
    backbone_cfg=backbone or {},
    hidden_dim=policy_hidden_dim,
    action_head_cfg=action_head,
)
self.rollout_engine = RolloutEngine(max_steps=max_steps)
```

但 `Policy.__init__()` 的签名是：

```python
def __init__(..., max_steps: int = 0, ...)
```

因此当前代码下：

- rollout 的真实 horizon 由 `RolloutEngine(max_steps=3)` 控制
- `Policy` 内部构造 type features 时使用的 `self.max_steps` 仍是默认值 `0`

这意味着动作类型特征中的“剩余预算”分量按代码事实并未与 rollout 真实步数上限严格对齐。

### 18.4 Reward-matching 锚定的是 stop 前最后一个 active state

当前 `reward_matching_loss` 使用的是：

$$
f_{L-1} = \log \widetilde F_\theta(s_{L-1}),
$$

而不是 stop 后终止相位单独重新编码的 terminal flow。这一点在解释 diagnostics `terminal_flow_mean` 和 `terminal_flow_vs_reward` 时需要保持一致。

## 19. 总结

按当前仓库代码，主线算法可以用一句话概括为：

> 在候选知识图上，从锚点诱导子图出发，按“先判 expand/stop、再选 frontier edge”的策略逐步扩张子图；用结构化终止奖励定义目标分布；用显式均匀后向策略和前向 FL-SubTB 目标训练状态流；再用 reward-matching 正则把流的绝对量级锚定到 reward。

对应的核心目标函数是：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{FL\text{-}SubTB}(\lambda=0.9)}
+ 0.1\,\mathcal L_{\mathrm{RM}}
}
$$

其中终止奖励默认是：

$$
\boxed{
\log R(x)
=
\max\big(5\,\mathrm{recall}(x)-3\,\delta(x)-0.05|E_x|,-5\big)
}
$$

并在 no-hit 时强制取 `-5`。
