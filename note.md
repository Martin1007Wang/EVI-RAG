# Weaver 特征工程重设计：数据约束下的 KGQA 子图搜索方案

## 0. 先把边界定死

当前数据不是完整 semantic parsing 数据，也不是人工标注 query graph 数据。它是 WebQSP/CWQ 风格的 KGQA 检索训练数据，清洗后进入 LMDB 的字段主要是：

```text
edge_index                         # $2, E$
node_entity_catalog_ids            # $N$
edge_relation_catalog_ids          # $E$
question_emb                       # $D$
anchor_node_ids                    # $A$
target_node_ids                    # $Y$
reachable_target_node_ids          # $Y_reach$
anchor_node_forward_distances_flat
anchor_node_backward_distances_flat
target_node_distances_flat
target_shortest_path_count_flat
target_shortest_path_edge_mask_flat
```

全局可用资源是：

```text
entity_catalog.pt
relation_catalog.pt
entity_embeddings.pt
relation_embeddings.pt
```

重要限制：

1. LMDB 中默认没有自然语言 question text。
2. 没有人工 query graph role 标注。
3. 没有显式 answer type 标注。
4. 没有完整 Freebase schema ontology。
5. 没有人工证明链，只有自动计算的 shortest-path / target-distance 监督。
6. 训练时有 target / shortest-path labels；测试时不能把 target-derived labels 编进 policy feature。

所以，任何依赖以下信息的特征都不能作为默认方案：

```text
question text pattern
wh-word answer type
人工 query graph role
人工 constraint label
完整 relation taxonomy
答案类型标签
teacher path 作为 policy 输入
```

上一版提出的 `g_constraint`、`g_type`、复杂 query-graph role taxonomy 太理想化。它们可以作为未来扩展，但不是当前仓库能直接落地的第一版。

当前真正可做、也最应该做的是：

$
\boxed{
\text{PLM L2 semantic atoms}
+\text{anchor-conditioned structural coordinates}
+\text{state-local action geometry}
+\text{frontier-value Stop gate}
}
$

目标不是凭空构造 query graph parser，而是在现有 tensor schema 下，把 sequential subgraph search 的 proposal geometry 做对。

---

## 1. 当前数据实际支持哪些特征？

给定单条样本：

$
G=(V,E),\qquad e=(u,r,v)\in E
$

已知：

$
q\in\mathbb{R}^{D}
$

来自 `question_emb`。

节点全局 id：

$
c_v=\texttt{node\_entity\_catalog\_ids}[v]
$

边关系全局 id：

$
c_r=\texttt{edge\_relation\_catalog\_ids}[e]
$

通过 catalog 和 embedding 表可查：

$
x_v,\quad x_r
$

anchor：

$
A\subseteq V
$

训练时还知道 target / shortest-path labels：

$
Y\subseteq V
$

以及 target-distance、shortest-path edge mask。但这些只能用于 reward、teacher、diagnostics、auxiliary loss，不能进入 policy feature，否则测试不可用且会泄漏答案。

因此 policy feature 的合法输入只有：

```text
question_emb
node/entity embeddings
relation embeddings
edge_index
anchor_node_ids
current rollout state s=(V_s,E_s)
frontier C(s)
图结构统计，如 degree、frontier size、是否 active、是否 root、是否 internal edge
```

---

## 2. 设计原则：不要造不存在的信息

当前数据下，特征工程的核心原则应改成四条。

### 2.1 从 anchor 出发，而不是从全图语义相似度出发

KGQA 检索的基本约束是：答案子图必须从 question entity / anchor 出发可达。

所以节点和边的第一类特征必须表达：

$
\text{这个节点/边相对于 anchor 在什么结构位置？}
$

当前 shortest-distance bucket 可以表达“几跳远”，但 DDE 更适合作为结构坐标，因为它表达正反向传播强度，而不仅是离散距离。

### 2.2 单边语义相似度只能是弱信号

可以使用：

$
\langle q,x_r\rangle,
\quad
\langle q,x_u\rangle,
\quad
\langle q,x_v\rangle
$

但这些只能是 feature，不能再作为手写主 logit。因为 KGQA 的正确边不一定文本上最像问题，尤其是中间桥接边、CVT 边、第二跳 relation。

### 2.3 当前能做的“role”只能是图结构 role，不是语义 role

没有 question text、answer type、人工 role label，就不要写 `person/location/date constraint` 这种特征。

当前可以可靠构造的是：

```text
is_outgoing_from_active
is_incoming_to_active
is_internal_edge
introduces_new_node
new_node_degree
relation_frequency
frontier_growth
DDE transition
path relation history similarity
```

这些是数据真实支持的 action role。

### 2.4 Stop 必须看 frontier

Stop 的问题不是“当前用了几步”，而是：

$
\text{继续扩展是否还有足够高价值的 frontier edge？}
$

所以 Stop/Expand gate 必须把 expand logit 定义为 frontier continuation value，而不是由 MLP 单独猜。

---

## 3. 静态语义特征

对每个节点：

$
x_v=
\begin{cases}
\text{entity PLM embedding}(c_v), & v\text{ is text entity}\
e_{nontext}, & v\text{ is non-text / CVT}
\end{cases}
$

这里保留当前 non-text shared embedding，不做 CVT schema pooling。

对每条边：

$
x_{r_e}=\text{relation PLM embedding}(c_r)
$

问题：

$
q=\texttt{question_emb}
$

如果上游已经 L2 normalize，保持即可；但训练启动时应 assert norm 分布，避免 dot product 被 embedding norm 污染。

---

## 4. DDE：anchor-conditioned structural coordinate

定义 anchor one-hot：

$
a_0(v)=\mathbf{1}$v\in A$
$

正向传播：

$
a_{k+1}^{+}(v)=
\operatorname{mean}_{(u,r,v)\in E}a_k^{+}(u)
$

反向传播：

$
a_{k+1}^{-}(v)=
\operatorname{mean}_{(v,r,u)\in E}a_k^{-}(u)
$

最终：

$
d_v=$a_0(v),a_1^+(v),\ldots,a_{K_f}^+(v),a_1^-(v),\ldots,a_{K_b}^-(v)$
$

(d_v) 是节点相对于 anchor 的结构坐标。它是 answer-agnostic，可以在 train/val/test 一致使用。

这一步替代或至少优先于当前 anchor shortest-distance bucket embedding。

---

## 5. FeatureEncoder：只做静态表示

节点：

$
h_v=\operatorname{LN}(W_xx_v+W_dd_v)
$

关系：

$
h_r=\operatorname{LN}(W_rx_r)
$

问题：

$
h_q=\operatorname{LN}(W_qq)
$

FeatureBank 应包含：

```python
@dataclass(frozen=True)
class FeatureBank:
    node_sem: torch.Tensor      # $N, D$
    rel_sem: torch.Tensor       # $E, D$
    query_sem: torch.Tensor     # $B, D$

    node_h: torch.Tensor        # $N, H$
    rel_h: torch.Tensor         # $E, H$
    query_h: torch.Tensor       # $B, H$

    node_dde: torch.Tensor      # $N, D_dde$
```

注意：DDE 已经进入 (h_v)，后面不要无意义重复拼。只有当某个 action feature 明确需要 (d_u\rightarrow d_v) 的结构转移时，才显式使用 raw DDE。

---

## 6. EdgeEncoder：只表示 directed fact

边表示负责回答：

$
(u,r,v)\text{ 是什么事实？}
$

定义：

$
\phi_e=\operatorname{EdgeEnc}(h_u,h_{r_e},h_v)
$

最小实现：

$
\phi_e=\operatorname{LN}(W_e$h_u,h_{r_e},h_v$)
$

不要在 EdgeEncoder 里输入 (h_q)。不要在 EdgeEncoder 里直接算 action score。不要在 EdgeEncoder 里默认重复拼 (d_u,d_v)。

---

## 7. 当前数据下可实现的 action features

候选边：

$
e=(u,r,v)\in C(s)
$

状态：

$
s=(V_s,E_s)
$

frontier：

$
C(s)={e\in E\setminus E_s:u\in V_s\lor v\in V_s}
$

ActionFeatureBuilder 构造：

$
\chi(e,s)=
$
g_{geom}(e,s),
g_{sem}(e),
g_{path}(e,s),
g_{branch}(e,s),
g_{status}(e,s)
$
$

这五类都能从现有 tensor schema 得到。

### 7.1 几何/结构转移特征 (g_{geom})

表达这条边相对于当前 active subgraph 的结构移动：

$
g_{geom}(e,s)=
$
\mathbf{1}$u\in V_s,v\notin V_s$,
\mathbf{1}$v\in V_s,u\notin V_s$,
\mathbf{1}$u\in V_s,v\in V_s$,
\mathbf{1}$u\notin V_s,v\notin V_s$
$
$

最后一项理论上对 frontier 应为 0，但保留可作为 sanity check。

如果使用 DDE transition：

$
$d_u,d_v,d_v-d_u,d_u\odot d_v$
$

这有明确意义：它描述从 active 端点到新端点的 anchor-conditioned structural movement。

### 7.2 语义先验与 residual 复用特征

使用已有 PLM L2 特征：

$
g_{sem}(e)=
$
\langle q,x_{r_e}\rangle,
\langle q,x_u\rangle,
\langle q,x_v\rangle
$
$

这些分数首先构成强 semantic prior：

$
b_e=\tau(\langle q,x_{r_e}\rangle+\alpha\langle q,x_{n(e,s)}\rangle)
$

同时它们也可以作为 residual 的输入特征。这里的角色不是让 MLP 从零学习边排序，而是在一个已经有效的 semantic prior 上学习状态相关上调/下调。

如果担心 non-text entity 的 cosine 无意义，可加入 mask：

$
\mathbf{1}$u\text{ is text}$,\quad \mathbf{1}$v\text{ is text}$
$

并令 non-text cosine 为 0 或 learned default。

### 7.3 Path/history 特征 (g_{path})

当前状态已经选中的非 root 关系集合：

$
R_s={r_e:e\in E_s\setminus E_0}
$

构造 relation history：

$
p_s=\operatorname{Pool}{h_{r_e}:e\in E_s\setminus E_0}
$

第一版用 mean / attention pool，不要先上复杂 GRU。

候选 relation 与当前 relation history 的交互：

$
g_{path}(e,s)=
$
\langle h_q,h_{r_e}\rangle,
\langle p_s,h_{r_e}\rangle,
\langle h_q,p_s\rangle
$
$

如果状态还没有非 root edge，则 (p_s) 使用 learned empty vector 或零向量。

这里的意义：不要只看单边 relation 和 question；要看候选 relation 是否和当前 partial path 一致。

### 7.4 Branching / hub-risk 特征 (g_{branch})

从图结构可直接得到：

$
g_{branch}(e,s)=
$
\log(1+\deg(u)),
\log(1+\deg(v)),
\log(1+\deg_{out}(u)),
\log(1+\deg_{out}(v)),
\log(1+\operatorname{freq}(r_e))
$
$

可选加入 frontier growth estimate：

$
\widehat{\Delta |C|}(e,s)=|C(s+e)|-|C(s)|
$

这项能惩罚 hub drift。很多错误扩展不是语义不相关，而是把搜索带进高分支噪声区。

### 7.5 状态/动作标记特征 (g_{status})

包括：

$
g_{status}(e,s)=
$
\mathbf{1}$e\in E_0$,
\mathbf{1}$e\in E_s$,
\rho_s,
|E_s\setminus E_0|,
|V_s|
$
$

对 frontier edge，(e\in E_s) 应为 0；保留这些特征主要用于 sanity 和统一接口。

---

## 8. Edge action scorer

新的 edge scorer 是：

$
z_e(s)=b_e+\lambda_{\mathrm{eff}}\Delta_\theta(h_s,\phi_e,\chi(e,s),b_e)
$

其中：

* (h_s)：当前 partial subgraph 状态；
* (\phi_e)：候选 directed fact 表示；
* (\chi(e,s))：当前数据实际支持的 action geometry / semantic / path / branch features；
* (b_e)：PLM semantic prior base logit；
* (\Delta_\theta)：只负责修正 prior 的 residual。

这保留了诊断中已经有效的 semantic prior，同时让当前 partial evidence graph 影响下一步边选择。它不再采用纯 action MLP：

$
z_e=f_\theta(h_s,\phi_e,\chi(e,s))
$

作为默认主路径，因为随机 residual/MLP 初期会破坏 prior ranking。

也不要使用：

$
z_e=\langle W_qh_q,W_e\phi_e\rangle
$

作为主方案。(h_q) 和 (\phi_e) 不天然处在同一个“越相似越应该扩展”的物理空间。KGQA 需要的是结构化 action utility，而不是 query-edge embedding similarity。

---

## 9. StateReadout：状态要看 active subgraph 和 frontier

状态读出包含：

$
m_V(s)=\operatorname{AttnPool}_{h_q}{h_v:v\in V_s}
$

$
m_E(s)=\operatorname{AttnPool}_{h_q}{\phi_e:e\in E_s}
$

relation history：

$
p_s=\operatorname{Pool}{h_{r_e}:e\in E_s\setminus E_0}
$

frontier summary 可以先用一个 preliminary/base edge score，也可以用纯结构/语义 summary。第一版建议用 action feature 中的弱 score 形成 summary：

$
m_C(s)=
$
\max_{e\in C(s)} \tilde{z}*e,
\operatorname{logmeanexp}*{e\in C(s)}\tilde{z}_e,
\log(1+|C(s)|)
$
$

其中 (\tilde{z}_e) 可以是一个不依赖 (h_s) 的 lightweight base scorer：

$
\tilde{z}*e=f*{base}(\phi_e,g_{sem}(e),g_{geom}(e,s),g_{branch}(e,s))
$

注意：这是为了给状态读出提供 frontier summary，不是最终 action logit。

progress：

$
\rho_s=\frac{|E_s\setminus E_0|}{B}
$

最终：

$
h_s=\operatorname{LN}(W_s$h_q,m_V(s),m_E(s),p_s,m_C(s),\rho_s$)
$

---

## 10. Stop gate：frontier continuation value

最终 edge logits：

$
z_e(s)=f_\theta(h_s,\phi_e,\chi(e,s))
$

Expand 价值：

$
z_{expand}(s)=\operatorname{logsumexp}_{e\in C(s)}z_e(s)-c_B\rho_s
$

Stop 价值：

$
z_{stop}(s)=v_\theta(h_s,m_C(s),\rho_s)
$

Policy：

$
P(Stop/Expand\mid s)=\operatorname{softmax}(z_{stop},z_{expand})
$

$
P(e\mid s,Expand)=\operatorname{softmax}_{e\in C(s)}z_e(s)
$

完整：

$
P_F(Expand(e)\mid s)=P(Expand\mid s)P(e\mid s,Expand)
$

无 frontier、inactive、budget exhausted 时强制 Stop。

---

## 11. 训练中可用但不能进 policy 的监督信息

当前数据有 target 和 shortest-path labels。它们很有价值，但角色必须清楚。

可以用于：

```text
terminal reward
coverage reward
teacher/proposal sampling
auxiliary diagnostics
edge-rank diagnostics
stop oracle diagnostics
```

不应该用于：

```text
FeatureEncoder 输入
ActionFeatureBuilder 输入
StateReadout 输入
Policy inference 输入
```

例如：

```text
target_shortest_path_edge_mask_flat
```

可以用来计算 `policy_answer_edge_rank_at_root`、teacher sampling、edge auxiliary loss，但不能作为 edge feature。

否则训练和测试分布不一致，且本质上泄漏答案。

---

## 12. 模块修改

### 12.1 `src/weaver/nn/dde.py`：新增

```python
class DirectionalDDE(nn.Module):
    def __init__(
        self,
        num_forward_rounds: int = 2,
        num_backward_rounds: int = 2,
        include_anchor_indicator: bool = True,
    ) -> None:
        ...

    def forward(
        self,
        edge_index: torch.Tensor,
        anchor_node_ids: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        ...
```

用 `scatter_mean` 实现正反向传播。

### 12.2 `src/weaver/nn/feature_encoder.py`：修改

新增 DDE。关闭旧 anchor distance embedding。

输出 `FeatureBank.node_dde`。

### 12.3 `src/weaver/nn/edge_encoder.py`：收紧

只做：

```python
phi_e = EdgeEnc(node_h$src$, rel_h$edge$, node_h$dst$)
```

### 12.4 `src/weaver/nn/action_features.py`：新增

职责：从 state、frontier、FeatureBank、图结构统计构造：

```text
g_geom
g_sem
g_path
g_branch
g_status
```

第一版不要实现不可得的 answer type / natural-language constraint features。

### 12.5 `src/weaver/nn/edge_scorer.py`：重写

职责：

```python
frontier_logits = EdgeScorer(h_s, phi_frontier, action_features)
```

删除手写主 logit：

```python
q_rel_cos + alpha * q_new_entity_cos
```

### 12.6 `src/weaver/nn/state_readout.py`：修改

加入：

```text
path memory
frontier summary
progress
```

### 12.7 `src/weaver/nn/stop_gate.py`：重写

```python
expand_logit = segmented_logsumexp(frontier_logits, graph_ids) - progress_penalty
stop_logit = stop_value_mlp($h_s, frontier_summary, progress$)
```

### 12.8 `src/weaver/policy.py`：重排

```text
FeatureEncoder
EdgeEncoder
Preliminary frontier summary / base scorer
StateReadout
ActionFeatureBuilder
EdgeScorer
StopGate
FlowHead
```

### 12.9 `src/weaver/loss.py`：不改主体

SubTB 仍然训练 GFlowNet balance。当前重点是 proposal geometry，不是 loss。

---

## 13. 配置建议

```yaml
feature_encoder_cfg:
  hidden_dim: 1024
  normalize_semantic: false
  use_anchor_distance_embedding: false
  dde:
    enabled: true
    num_forward_rounds: 2
    num_backward_rounds: 2
    include_anchor_indicator: true
  non_text:
    type: shared_embedding
    init_std: 0.02
```

```yaml
action_feature_cfg:
  use_geom: true
  use_semantic_weak_features: true
  use_path_history: true
  use_branching: true
  use_status: true
  use_answer_type: false
  use_nl_constraint: false
```

```yaml
state_readout_cfg:
  use_active_node_pool: true
  use_active_edge_pool: true
  use_path_memory: true
  use_frontier_summary: true
  frontier_summary:
    stats: $max, logmeanexp, log_size$
```

```yaml
edge_scorer_cfg:
  type: semantic_prior_residual
  hidden_dim: 1024
  residual_warmup_start_step: 500
  residual_warmup_steps: 1500
  zero_init_residual_output: true
```

```yaml
stop_gate_cfg:
  type: frontier_value
  expand_logit: frontier_logsumexp
  use_progress_penalty: true
  progress_penalty_init: 0.0
```

---

## 14. 最小迁移顺序

第一步：实现 DDE，替换旧 anchor distance bucket。先不改 edge scorer。

第二步：收紧 EdgeEncoder，只输出 directed fact representation。

第三步：实现 ActionFeatureBuilder 的最小集合：

```text
g_geom
g_sem
g_path
g_branch
g_status
```

第四步：重写 EdgeScorer，从纯 action MLP 改成 semantic-prior residual：

$
b_e=\tau(s_r(e)+\alpha s_n(e,s))
$

$
\Delta_\theta(s,e)=f_\theta(h_s,\phi_e,\chi(e,s),b_e)
$

$
z_e(s)=b_e+\lambda_{\mathrm{eff}}\Delta_\theta(s,e)
$

residual head 最后一层 zero-init，warmup 前 $\lambda_{\mathrm{eff}}=0$，保证初始化时 $z_e(s)\approx b_e$。

第五步：改 StopGate：

$
z_{expand}=\operatorname{LSE}_{e\in C(s)}z_e(s)
$

第六步：只用数据已有监督做 diagnostics，不泄漏进 policy feature。

重点指标：

```text
prior_answer_edge_rank_at_root
policy_answer_edge_rank_at_root
answer_edge_rank_delta_mean
final_worse_than_prior_rate
residual_to_prior_std_ratio
root_frontier_contains_answer_edge_rate
teacher_answer_edge_sampling_rate
budget_exhaust_ratio
stop_depth_hist
frontier_logsumexp_mean
hub_drift_rate
frontier_branch_growth
path_edge_recall_by_depth
```

---

## 15. 最终判断

这版方案比上一版更受数据约束。它不再假设有自然语言 question text、answer type、人工 constraint label 或 schema taxonomy。

当前数据最可靠的信息是：

```text
anchor
candidate graph
entity/relation PLM embeddings
question embedding
图结构
rollout state
训练期 target/shortest-path labels
```

所以特征工程必须围绕这些信息设计。

最终新 Weaver 应该是：

$
\text{old Weaver}=\text{unconstrained action scorer}+\text{GFlowNet shell}
$

$
\text{new Weaver}=\text{semantic prior preservation}+\text{state-conditioned residual}+\text{frontier value stop}
$

这才符合当前 WebQSP/CWQ tensor schema，也更接近 KGQA 检索真正需要的归纳偏置。
