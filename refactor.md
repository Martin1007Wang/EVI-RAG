我的选择很明确：**不要把当前仓库改成重 GNN，不要引入 GRU，不要堆一堆 handcrafted interaction features。**

最有品位的改法是：

[
\boxed{
z_e
===

z_{\text{semantic-prior}}(q,e)
+
z_{\text{evidence-residual}}(q,s_t,e)
}
]

其中：

[
s_t=(V_t,E_t)
]

仍然是图状态，不引入 trajectory memory。

核心变化只有一句话：**当前 policy 从“frontier semantic ranking”改成“partial evidence graph conditioned transition flow”。**

也就是：当前的 relation cosine 不是删掉，而是降级成先验；真正的 GFlowNet reasoning 能力来自一个 query-conditioned evidence graph readout，它让已选子图 (G_t) 直接影响下一条边和 STOP。

---

## 0. 总体设计：我会选择的最终算法形态

当前版本是：

[
z_e = \tau\left(
\langle h_q,h_r\rangle
+
\alpha \mathbf{1}[x_e\text{ text}]\langle h_q,h_{x_e}\rangle
\right)
]

我建议改成：

[
z_e =
z_{\text{prior}}(q,e)
+
\lambda_{\text{res}}\cdot g_\theta(h_q,h_s,\phi_e)
]

其中：

[
z_{\text{prior}}(q,e)
=====================

\tau\left(
\langle h_q,h_r\rangle
+
\alpha \mathbf{1}[x_e\text{ text}]\langle h_q,h_{x_e}\rangle
\right)
]

保留原来的 semantic prior。

然后：

[
\phi_e=\mathrm{EdgeEnc}_\theta(h_u,h_r,h_v,b_e)
]

[
h_s=\mathrm{EvidenceReadout}_\theta(q,G_t)
]

[
z_{\text{res}}=g_\theta([h_q,h_s,\phi_e])
]

这里 (b_e) 只保留最小结构标识，例如：

[
b_e=[
\mathbf{1}(u\in V_t),
\mathbf{1}(v\in V_t),
\mathbf{1}(u\in A),
\mathbf{1}(v\in A),
\mathbf{1}(x_e\text{ text})
]
]

不要放太多东西。anchor distance bucket、frontier rank、path distance、target distance 这类东西先不要塞进 online edge scorer。你一塞进去，算法就变成“特征工程调参”，品位立刻下降。

状态 readout 改成 query-conditioned evidence readout：

[
\mathcal{E}_t^+=E_t\setminus E_0
]

[
\phi_i=\mathrm{EdgeEnc}*\theta(h*{u_i},h_{r_i},h_{v_i},b_i)
]

[
\alpha_i=
\mathrm{softmax}_{i\in \mathcal{E}_t^+}
\left(
(h_qW_Q)^\top(\phi_iW_K)/\sqrt{d}
\right)
]

[
m_t=\sum_{i\in \mathcal{E}_t^+}\alpha_i(\phi_iW_V)
]

节点 evidence readout：

[
\beta_v=
\mathrm{softmax}_{v\in V_t}
\left(
(h_qU_Q)^\top(h_vU_K)/\sqrt{d}
\right)
]

[
n_t=\sum_{v\in V_t}\beta_v(h_vU_V)
]

anchor readout：

[
a=\mathrm{mean}_{v\in A}h_v
]

最后：

[
h_s=\mathrm{MLP}_s([h_q,a,m_t,n_t,\rho_t])
]

其中：

[
\rho_t=\frac{|E_t\setminus E_0|}{K}
]

这样做有四个优点。

第一，状态仍然是图：

[
s_t=(V_t,E_t)
]

没有引入顺序记忆，不破坏 subgraph-level GFlowNet。

第二，edge scorer 真的依赖状态：

[
z_e=f(q,G_t,e)
]

而不是当前这种：

[
z_e\approx f(q,e)
]

第三，semantic prior 保留，训练不会从零开始乱飘。

第四，结构很干净，论文里能讲清楚：**pretrained semantic geometry provides local prior; GFlowNet learns state-conditioned evidence residual.**

这比“MLP 拼一堆特征”高级得多。

---

# 1. `src/weaver/policy.py`

## 原先负责什么

现在 `Policy` 大概是一个总装配器：

1. 调 `backbone / feature encoder` 得到 `FeatureBank`；
2. 调 `state_readout` 得到 `state_h`；
3. 调 `action_head` 得到 Stop / Continue logits；
4. 调 `edge_scorer` 得到候选边 logits；
5. 输出 rollout 需要的 policy logits、flow value、可能还有状态值。

但当前的结构有一个隐性问题：`state_h` 主要服务 `flow_head`，没有真正进入 edge decision。于是 `Policy` 形式上是 state-conditioned，实质上不是。

## 现在应该负责什么

`Policy` 应该成为 **state-conditioned flow policy 的唯一组装点**。

它应该明确组织三件事：

[
\text{FeatureBank}
\rightarrow
\text{EvidenceContext}(q,G_t)
\rightarrow
\text{Action/Transition Flow}
]

也就是：

1. `backbone` 只负责静态 embedding；
2. `state_readout` 负责把当前子图 (G_t) 编码成 evidence context；
3. `edge_scorer` 使用 evidence context 给 frontier edge 打分；
4. `action_head` 使用 evidence context 判断 STOP；
5. `flow_head` 使用同一个 evidence context 估计 (\log F(s))。

## 怎么改

我会让 `Policy.forward(...)` 输出一个更干净的数据结构，比如：

```python
@dataclass(frozen=True)
class PolicyOutput:
    state_log_flow: torch.Tensor       # [B]
    stop_logits: torch.Tensor          # [B]
    edge_logits: torch.Tensor          # [num_candidates]
    context: EvidenceContext
```

再定义：

```python
@dataclass(frozen=True)
class EvidenceContext:
    query_h: torch.Tensor          # [B, H]
    state_h: torch.Tensor          # [B, H]
    node_h: torch.Tensor           # [num_nodes, H]
    rel_h: torch.Tensor            # [num_relations, H]
    progress: torch.Tensor         # [B]
```

不要在 `Policy` 里散落各种 `node_pool`, `edge_pool`, `state_summary` 的半成品。`Policy` 只认一个 `EvidenceContext`。

从职责上讲：

```text
Policy
= static features
+ evidence context
+ log F(s)
+ logit STOP
+ logit Expand(e)
```

它不应该知道 reward 怎么算，不应该知道 teacher label 怎么算，不应该知道 rollout buffer 怎么存。

---

# 2. `src/weaver/nn/backbone.py`

## 原先负责什么

`SemanticFeatureEncoder` 现在负责构造 `FeatureBank`：

[
h_q,\ h_v,\ h_r
]

并处理：

1. question embedding；
2. text entity embedding；
3. non-text entity shared embedding；
4. relation embedding；
5. anchor mask；
6. node_is_non_text；
7. anchor distance bucket。

当前问题是：它里面有一些静态特征算了但没有用，尤其 anchor distance bucket。职责稍微混了：有些是“静态语义表征”，有些已经接近“状态/结构特征”。

## 现在应该负责什么

`backbone.py` 应该只负责 **static semantic feature bank**。

也就是：

[
h_q,\ h_v,\ h_r
]

以及最基础的静态 mask。

它不应该负责当前状态 (s_t)，也不应该负责 evidence readout。

## 怎么改

保留：

```python
@dataclass(frozen=True)
class FeatureBank:
    query_h: torch.Tensor          # [B, H]
    node_h: torch.Tensor           # [N, H]
    rel_h: torch.Tensor            # [R, H]
    anchor_mask: torch.Tensor      # [N]
    node_is_non_text: torch.Tensor # [N]
```

如果需要每个 node 属于哪个 graph，应该来自 batch/schema，不要在 feature encoder 里临时拼。

anchor distance bucket 怎么办？

我的选择：**先移出 online policy 主路径。**

理由很简单：anchor distance 是强结构 heuristic，但它不是 evidence reasoning。你现在要避免把算法变成“特征工程 soup”。如果后面要用，也应该作为 `StateStructuralFeatures` 的小组件，而不是 backbone 的核心输出。

所以：

```text
backbone.py:
    静态语义 embedding

state_readout.py:
    当前子图 evidence encoding

edge_scorer.py:
    state-conditioned transition scoring
```

这三层必须切开。

---

# 3. `src/weaver/nn/state_readout.py`

这是最该改的模块。

## 原先负责什么

现在它做：

[
a=\mathrm{mean}_{v\in A}h_v
]

[
n=\mathrm{mean}_{v\in V_t\setminus A}h_v
]

[
p=\mathrm{mean}_{e\in E_t\setminus E_0}\phi_e
]

[
g=\mathrm{norm}(w_Aa+w_Nn+w_Ep)
]

[
h_s=\mathrm{norm}(\mathrm{MLP}([h_q,a,p,g]))
]

这个设计有两个问题。

第一，`n` 只进入 `g`，没有直接进入最终 MLP，信息被压得太早。

第二，`p` 是 mean pooling，不看 question。一个 evidence edge 对某个问题重要，对另一个问题可能无关。你现在的 readout 不区分这个。

更关键的是：当前 `h_s` 没有进入 edge scorer，所以它再好也没用。

## 现在应该负责什么

它应该负责：

[
h_s=\mathrm{EvidenceReadout}_\theta(q,G_t)
]

也就是把当前 partial subgraph 编码成 **query-conditioned evidence state**。

我会把它设计成轻量 set/graph readout，不上 GRU，不上重 GNN。

## 怎么改

建议定义：

```python
@dataclass(frozen=True)
class EvidenceContext:
    query_h: torch.Tensor
    state_h: torch.Tensor
    node_h: torch.Tensor
    rel_h: torch.Tensor
    progress: torch.Tensor
```

`StateReadout.forward(...)` 输入：

```python
query_h
node_h
rel_h
selected_edge_index
selected_edge_rel_ids
selected_edge_mask / selected_nonroot_edge_ids
node_in_state_mask
anchor_mask
batch_ids
progress
```

输出：

```python
EvidenceContext
```

内部做四件事。

### 3.1 Anchor readout

保留均值就够：

[
a_b=\mathrm{mean}_{v\in A_b}h_v
]

anchor 是问题入口，没必要复杂化。

### 3.2 Evidence edge encoder

对已选非 root 边：

[
e_i=(u_i,r_i,v_i)
]

[
\phi_i=\mathrm{MLP}*e([h*{u_i},h_{r_i},h_{v_i}])
]

这里不需要 normalize 到死。当前代码到处 L2 normalize，适合 cosine prior，但对 MLP 表示不一定好。我的选择是：

* `h_q, h_v, h_r` 作为输入仍然 L2 norm；
* MLP 输出不要强制 L2 norm；
* 只在 prior cosine 处依赖 norm。

### 3.3 Query-aware edge pooling

[
\alpha_i=
\mathrm{softmax}_{i\in E_t^+}
\left(
(h_qW_Q)^\top(\phi_iW_K)/\sqrt{d}
\right)
]

[
m_t=\sum_i\alpha_i(\phi_iW_V)
]

这一步比 mean pooling 有品位得多，因为它表达：

> 当前子图里的哪些 evidence edges 对这个问题重要。

它也是 permutation-invariant，不依赖 rollout 顺序。

### 3.4 Query-aware node pooling

对 (V_t)：

[
\beta_v=
\mathrm{softmax}_{v\in V_t}
\left(
(h_qU_Q)^\top(h_vU_K)/\sqrt{d}
\right)
]

[
n_t=\sum_v\beta_v(h_vU_V)
]

这解决了当前 expanded node pool 被弱化的问题。

### 3.5 Final state context

[
h_s=
\mathrm{MLP}_s([h_q,a,m_t,n_t,\rho_t])
]

注意这里 `progress` 可以进，因为 STOP 和 flow 需要知道预算进度。但我不建议让 `progress` 主导 edge scorer。它进入 `state_h` 可以，但不要再手写一大堆 progress-based edge features。

最终职责：

```text
state_readout.py
负责把当前子图 G_t 编码成 query-conditioned evidence context。
不负责打边分。
不负责 STOP。
不负责 reward。
```

---

# 4. `src/weaver/nn/edge_scorer.py`

这是第二个必须重写的模块。

## 原先负责什么

原来它基本做：

[
z_e=
\tau\left(
\langle h_q,h_r\rangle
+
\alpha \mathbf{1}[x_e\text{text}]\langle h_q,h_{x_e}\rangle
\right)
]

并且显式：

```python
del state_h
```

这句是当前算法最大的问题之一。它直接宣告：边选择不看当前 evidence state。

## 现在应该负责什么

它应该负责：

[
z_e =
z_{\text{prior}}(q,e)
+
z_{\text{residual}}(q,s_t,e)
]

也就是 **semantic prior + state-conditioned residual**。

这比直接 MLP 重写全部边分数更稳，也更有论文品位。

## 怎么改

### 4.1 保留原 prior

原 prior 不删：

[
z_{\text{prior}}=
\tau\left(
\langle h_q,h_r\rangle+
\alpha\mathbf{1}[x_e\text{text}]\langle h_q,h_{x_e}\rangle
\right)
]

它的解释是：

> pretrained semantic geometry defines a local proposal prior over frontier relations and newly introduced textual entities.

这个 prior 是合理的。问题只是它不能是全部。

### 4.2 增加 edge encoder

对 candidate edge：

[
c=(u,r,v)
]

[
\phi_c=\mathrm{MLP}_c([h_u,h_r,h_v,b_c])
]

其中 (b_c) 很少：

[
b_c=[
\mathbf{1}(u\in V_t),
\mathbf{1}(v\in V_t),
\mathbf{1}(u\in A),
\mathbf{1}(v\in A),
\mathbf{1}(x_c\text{text})
]
]

不建议加入：

* target distance；
* shortest path edge mask；
* teacher label；
* rank position；
* frontier degree；
* relation frequency；
* 一堆 handcrafted bucket。

这些会污染算法主张。

### 4.3 state-conditioned residual

[
z_{\text{res}}=
\mathrm{MLP}_z([h_s,h_q,\phi_c])
]

最后：

[
z_c=z_{\text{prior}}+\lambda_{\text{res}}z_{\text{res}}
]

我建议 (\lambda_{\text{res}}) 初始很小，例如 0 或 0.1，并设为可学习标量：

[
\lambda_{\text{res}}=\mathrm{softplus}(\eta)
]

或者简单一点，用配置：

```yaml
residual_scale_init: 0.1
```

如果你想训练稳，可以初始化最后一层为 near-zero。这样初始 policy 近似当前 semantic prior，训练后逐渐学习 residual。

这很重要。否则你一改 MLP，早期 rollout 可能直接崩。

### 4.4 它不应该做什么

`edge_scorer.py` 不应该：

1. 计算 state readout；
2. 访问 reward；
3. 访问 target labels；
4. 管 STOP；
5. 管 sampling temperature。

它只做：

```text
candidate edge + evidence context -> edge logits
```

---

# 5. `src/weaver/nn/action_head.py`

## 原先负责什么

现在 STOP logit 是：

[
\ell_{\text{stop}}=b+\beta\rho(s)+\gamma A_R(s)
]

其中：

[
A_R(s)=\max(0,\log R(s)-\log\epsilon)
]

这个设计的问题很大。

第一，STOP 不是 evidence sufficiency，而是 progress + reward advantage。

第二，如果 reward advantage 依赖训练期 gold answer，那么 inference 语义很危险。即使当前实现只是在训练/rollout里算，也会让 STOP 的论文叙事难讲。

第三，STOP 和 (h_s) 的语义脱节。

## 现在应该负责什么

`ActionHead` 应该负责：

[
P_F(\text{STOP}|s,q)
]

也就是 **evidence sufficiency decision**。

更准确地说，STOP 流应该是：

[
F_\theta(s\to \bot)
]

而 Continue/Expand 边流是：

[
F_\theta(s\to s\oplus e)
]

如果你暂时保持当前二阶段结构：

[
P(\text{STOP}|s)
]

[
P(e|s,\text{Continue})
]

那 STOP 至少要从 `EvidenceContext` 来：

[
\ell_{\text{stop}}=
g_\theta(h_s,h_q,\rho_t)
]

而不是只用 progress 和 reward advantage。

## 怎么改

我会改成：

[
\ell_{\text{stop}}
==================

\mathrm{MLP}*{stop}([h_s,h_q,\rho_t])
+
b*{\text{progress}}\rho_t
]

这里保留一个 progress bias 是可以的，因为 budget 是客观约束。但它只能是 bias，不是 STOP 的核心。

不要直接把 `terminal_reward_advantage` 作为主要输入。更干净的做法是：

* online policy inference：只使用 (q,s_t,\rho_t)；
* training diagnostics / auxiliary：可以用 oracle improvement 训练 stop sufficiency；
* reward 不进入 STOP input。

你可以加一个辅助 stop loss，但不要默认开太大权重：

[
y_{\text{stop}}(s)=
\mathbf{1}
\left[
\max_{x\succeq s, |x|\le K}R(x)-R(s)\le \epsilon
\right]
]

辅助损失：

[
\mathcal{L}_{stop}
==================

\mathrm{BCEWithLogits}(\ell_{\text{stop}},y_{\text{stop}})
]

但这只是 optional warmup，不是主目标。主目标仍是 SubTB。

### ActionHead 最终职责

```text
action_head.py
输入 EvidenceContext 和 progress。
输出 stop logit。
不直接读 gold reward。
不计算 edge logits。
不做合法性 mask。
```

合法性 mask 仍在 executor/sampling 层做。

---

# 6. `src/weaver/nn/flow_head.py` / 现有 flow head

你没有特别列这个文件，但当前 policy 里肯定有 flow head。

## 原先负责什么

估计：

[
\log F(s)
]

供 SubTB 使用。

问题是：如果 `state_h` 不影响 edge scorer，flow head 学到的东西对实际 transition policy 影响有限。它像一个为了 loss residual 服务的旁支。

## 现在应该负责什么

`flow_head` 应该估计：

[
\log Z_\theta(s)
]

也就是 partial evidence graph 的未来总潜力：

[
Z_\theta(s)
\approx
\sum_{x\succeq s} R(x)
]

它要和 edge scorer 共享同一个 `EvidenceContext`。这样论文里可以说：

> The same evidence state representation parameterizes both the state flow and the transition residual.

## 怎么改

简单保留：

[
\log F(s)=\mathrm{MLP}_F(h_s)
]

但需要保证：

1. `h_s` 是新的 evidence context；
2. `edge_scorer` 也用 `h_s`；
3. `action_head` 也用 `h_s`。

这样 `h_s` 才是真正的 single source of policy state。

我不建议现在做 child potential scoring：

[
z_e=\log F(s\oplus e)
]

虽然更漂亮，但工程成本高，容易慢。先用 shared context + residual 就够优雅。

---

# 7. `src/weaver/rollout/executor.py`

## 原先负责什么

它负责执行一步：

1. 根据当前 state 找 legal action；
2. 调 policy；
3. 应用 inactive graph / no frontier / budget exhausted 的合法性约束；
4. 采样或执行 action；
5. 更新 state；
6. 写 StepResult。

当前 executor 还有一个问题：它在某处计算 `terminal_reward_advantage` 给 action head。这个让 STOP 和 reward 纠缠太深。

## 现在应该负责什么

`executor.py` 应该只负责 **environment transition + legality**。

它不应该让 action policy 读取 gold-derived reward advantage。它可以计算当前 stop reward用于：

1. 如果真的 stop，写 terminal reward；
2. 诊断；
3. loss 需要 terminal reward。

但不要作为 policy input 的核心特征。

## 怎么改

### 7.1 删除或降级 `terminal_reward_advantage` 输入

原来：

```text
engine/executor computes terminal_reward_advantage
action_head uses it
```

改成：

```text
executor computes current_stop_log_reward only for terminal evaluation / metrics
policy does not consume it by default
```

如果你想保留兼容：

```yaml
use_reward_advantage_in_stop: false
```

默认 false。

有品位的版本应该默认不使用。否则 inference 信息来源说不干净。

### 7.2 保留合法性约束

这些保留：

* inactive graph 只能 STOP；
* no candidate edge 只能 STOP；
* budget exhausted 只能 STOP；
* already terminal 不再 expand。

合法性 mask 是环境规则，不是 policy 学习内容。

### 7.3 状态更新保持原样

[
s_{t+1}=s_t\oplus e
]

仍然是加入边和新节点。

这里不要引入 GRU memory，也不要记录 trajectory hidden state 到 state。否则你会破坏同一子图状态的一致性。

---

# 8. `src/weaver/rollout/sampling.py`

## 原先负责什么

它负责：

1. 对 action logits 做 softmax；
2. behavior temperature；
3. sample action；
4. 写入 target log-prob；
5. 处理 conditional edge policy。

当前有一点是对的：behavior 可以 temperature，但 GFlowNet loss 里写 untempered target log-prob。这点保留。

## 现在应该负责什么

它还是只负责 sampling，不应该承担算法语义。

但是随着 policy 改成：

[
z_e=z_{\text{prior}}+z_{\text{res}}
]

sampling 不需要知道 prior/residual。它只看 final logits。

## 怎么改

基本不大改。只需要保证：

1. `edge_logits` 来自新的 `edge_scorer`；
2. `stop_logits` 来自新的 `action_head`；
3. target log-prob 仍然用 untempered logits；
4. behavior log-prob 如需要可单独记录；
5. 不把 teacher/proposal 混进 target log-prob。

如果用 proposal mixture：

[
\pi_b=(1-\epsilon)\pi_\theta+\epsilon\pi_{\text{proposal}}
]

也必须保证写入 loss 的是：

[
\log \pi_\theta(a|s)
]

不是 behavior policy 的 log-prob。

这个模块的职责边界应该非常窄：

```text
sampling.py
只做 categorical sampling 和 log-prob bookkeeping。
不算 reward。
不算 teacher。
不算 edge feature。
```

如果现在它里面有很多结构定义和策略逻辑，可以继续拆。

---

# 9. `src/weaver/reward.py`

## 原先负责什么

当前 reward 是：

[
\log R(G_t)=
\max\left(
\log(\epsilon+\mathrm{F1}_A(G_t))
---------------------------------

c\frac{|E_t\setminus E_0|}{K},
\log R_{\min}
\right)
]

它评价答案覆盖和紧凑性。

问题是：它奖励的是“答案节点在不在子图里”，不是“证据路径是否支撑答案”。

这对 KGQA + LLM 很不够。LLM 需要的是 support subgraph，不只是 answer-containing subgraph。

## 现在应该负责什么

`reward.py` 应该定义 **terminal evidence reward**：

[
R(x)=R_{\text{answer}}(x)\cdot R_{\text{support}}(x)\cdot R_{\text{compact}}(x)
]

或者 log 形式：

[
\log R(x)
=========

\lambda_A \log(\epsilon+\mathrm{F1}_{answer})
+
\lambda_P \log(\epsilon+\mathrm{PathSupport})
---------------------------------------------

\lambda_C\mathrm{Cost}(x)
]

我会选择最小版：

[
\mathrm{PathSupport}(x)
=======================

\frac{
|{y\in Y_{\text{reachable}}:
\exists a\in A,\ \text{path}*{x}(a\leadsto y)}|
}{
|Y*{\text{reachable}}|
}
]

也就是答案不是孤零零出现在子图里，而是要能从 anchor 连过去。

如果你已经有 shortest-path teacher edge mask，可以定义：

[
\mathrm{TeacherEdgeRecall}(x)
=============================

\frac{
|E_x\cap E^*_{\text{teacher}}|
}{
|E^*_{\text{teacher}}|
}
]

但是我会谨慎使用。因为过强的 teacher-path reward 会把 GFlowNet 变成 shortest-path imitation。更干净的是用 weak support：

[
\exists \text{ anchor-answer connected path in selected subgraph}
]

而不是必须匹配 teacher shortest path。

## 怎么改

建议 reward 配置：

```yaml
reward_cfg:
  answer_f1_weight: 1.0
  support_weight: 0.5
  edge_cost: 0.03
  normalize_edge_cost_by_budget: true
  log_reward_clip_min: -30.0
```

函数层面：

```python
compute_answer_f1(...)
compute_anchor_answer_support(...)
compute_edge_cost(...)
compute_log_reward(...)
```

不要在 reward.py 里放 policy / proposal / teacher action 逻辑。

### 关键品位点

Reward 不要做 step shaping。

不要写：

[
r_t=\text{distance progress}+\text{relation match}
]

你之前已经很接近这个原则：reward 只评价 terminal evidence，policy 学如何生成它。

---

# 10. `src/weaver/losses.py`

## 原先负责什么

SubTB residual：

[
\delta_{i,j}
============

\log F(s_i)
+
\sum_{t=i}^{j}
[
\log P_F(a_t|s_t)
-----------------

\log P_B(s_t|s_{t+1})
]
-

\mathrm{target}(j)
]

终止：

[
\mathrm{target}(j)=\log R(x)
]

非终止：

[
\mathrm{target}(j)=\log F(s_{j+1})
]

loss 是加权平方。

这个大方向是对的。不要为了“创新”乱改 loss。

## 现在应该负责什么

`losses.py` 继续只负责 GFlowNet objective。

但要保证它不再夹杂过多 auxiliary：

* reward matching；
* edge auxiliary；
* teacher imitation；
* metrics；
* proposal loss。

这些都应该从核心 loss 中剥离，最多作为显式 optional auxiliary loss。

## 怎么改

核心保持：

```python
subtb_loss(rollout_batch) -> LossOutput
```

`LossOutput` 最好只含：

```python
@dataclass(frozen=True)
class LossOutput:
    loss: torch.Tensor
    subtb_loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
```

如果有 auxiliary：

```python
total_loss = subtb_loss
if stop_aux_coef > 0:
    total_loss += stop_aux_coef * stop_aux_loss
if teacher_aux_coef > 0:
    total_loss += teacher_aux_coef * teacher_aux_loss
```

但是这几个 auxiliary 不要藏在 SubTB 内部。

品位要求是：**主 objective 必须纯。**

你要能在论文里一句话写清楚：

[
\mathcal{L}=\mathcal{L}*{SubTB}
+
\lambda*{\text{stop}}\mathcal{L}*{stop}
+
\lambda*{\text{teacher}}\mathcal{L}_{teacher}
]

默认先只开 SubTB。auxiliary 是训练稳定器，不是算法核心。

---

# 11. `src/weaver/nn/backward_scorer.py` 或现有 backward policy

如果当前 backward 是 uniform parent，我会给一个分阶段选择。

## 最小版本

保留 uniform backward：

[
P_B(s_t|s_{t+1})=\frac{1}{|\mathrm{Parents}(s_{t+1})|}
]

先不要同时改 forward、stop、reward、backward。否则 debug 会爆炸。

## 更有品位的升级版本

新增 learned backward decomposition：

[
P_B(s_t|s_{t+1})
================

\mathrm{softmax}*{e\in E*{t+1}\setminus E_0}
b_\theta(q,s_{t+1},e)
]

其中删除边 (e) 得到 parent：

[
s_t=s_{t+1}\ominus e
]

它的语义是：完整 evidence graph 中哪条边像最后一步推理扩展。

但这个我建议放第二阶段。第一阶段先把 forward 变成 state-conditioned。

---

# 12. `src/weaver/guidance.py` / proposal / teacher

## 原先负责什么

你现在的 shortest-path、target-distance、coverage teacher 主要用于 proposal、diagnostic 或 disabled 链路。默认：

```yaml
coverage_cfg.enabled: false
proposal_cfg.enabled: false
```

所以它们不进入 online policy。

## 现在应该负责什么

Teacher 不应该变成 policy 的硬特征。

它有两个合理用途：

### 用途一：behavior proposal

[
\pi_b=(1-\epsilon_t)\pi_\theta+\epsilon_t\pi_{\text{teacher}}
]

只影响采样探索，不改变 target policy。

### 用途二：auxiliary warmup

[
\mathcal{L}_{teacher}
=====================

-\sum_{e\in C^*(s)}
w_e\log P_\theta(e|s)
]

小权重、短 warmup。

## 怎么改

不要把 teacher shortest-path mask 直接拼进 edge scorer：

错误做法：

[
z_e = f(...,\mathbf{1}[e\in E^*])
]

这是泄漏式监督特征，会把算法做脏。

正确做法：

```text
teacher only affects behavior exploration or auxiliary loss
target policy at inference sees no teacher labels
```

如果要论文干净，默认主实验可以：

1. no teacher；
2. teacher proposal only；
3. teacher auxiliary warmup；

分别 ablate。

---

# 13. `src/weaver/state.py` 和 `src/weaver/state_ops.py`

## 原先负责什么

`state.py` 定义：

[
s_t=(V_t,E_t)
]

`state_ops.py` 定义 frontier：

[
C(s_t)={e=(u,r,v)\in E\setminus E_t:u\in V_t\lor v\in V_t}
]

这些其实是对的。

## 现在应该负责什么

继续保持纯环境状态，不要塞神经网络 hidden state。

也就是说，不要改成：

[
s_t=(V_t,E_t,m_t)
]

除非你要做 history-augmented GFlowNet。但我不建议。

## 怎么改

基本不改。只需要保证它能高效提供给 `state_readout`：

1. 当前 selected edge ids；
2. selected non-root edge ids；
3. node_in_state mask；
4. candidate edge ids；
5. endpoint flags；
6. progress。

这些是结构状态，不是模型特征。

可以新增一个轻量结构：

```python
@dataclass(frozen=True)
class StateView:
    node_in_state: torch.Tensor
    edge_in_state: torch.Tensor
    nonroot_edge_ids: torch.Tensor
    candidate_edge_ids: torch.Tensor
    progress: torch.Tensor
```

但不要过度包装。如果当前已有等价对象，就复用。

---

# 14. `src/weaver/rollout/buffers.py`

## 原先负责什么

保存每一步：

* log_pf；
* log_pb；
* stop_mask；
* selected_edge_ids；
* terminal_log_rewards；
* state_log_flows；
* traces。

## 现在应该负责什么

它继续只保存 loss 必需信息和诊断信息。

随着新 policy，你可能需要额外保存：

```python
state_log_flow
stop_logit
edge_logit_selected
log_pf
log_pb
terminal_log_reward
```

但不要保存整个 `EvidenceContext`。那会浪费内存，也让 buffer 变成模型内部状态仓库。

## 怎么改

保持 buffer 简洁：

```text
buffer stores scalars/tensors required by SubTB and metrics.
buffer does not store node_h/state_h/attention weights by default.
```

如果需要 attention 可视化，单独 debug path，不进默认 buffer。

---

# 15. `src/weaver/rollout/engine.py`

## 原先负责什么

它可能在更高层调 executor、组织 rollout group、算训练/eval需要的 traces、metrics。

当前你提到 `engine.py:254` 计算 reward advantage，这说明 engine 混入了 policy input 构造。

## 现在应该负责什么

`engine.py` 只负责 rollout orchestration：

```text
for t in horizon:
    call executor.step(...)
    buffer.write(...)
return rollout batch
```

不要让它定义 STOP 的算法语义。

## 怎么改

删除或默认禁用：

```python
terminal_reward_advantage -> action_head
```

保留：

```python
current_stop_log_reward -> trace / terminal if stopped
```

如果某些 metrics 需要 “stop_now_reward vs continue_reward”，放在 diagnostics，不放进 policy.

---

# 16. `configs/model/weaver.yaml`

## 原先负责什么

当前配置大概有：

```yaml
hidden_dim: 1024
temperature
reward_cfg
coverage_cfg
proposal_cfg
action_head_cfg
edge_scorer_cfg
```

## 现在应该负责什么

配置要反映新算法主张，而不是暴露一堆可调旋钮。

我会这样整理：

```yaml
policy:
  hidden_dim: 1024

  state_readout:
    type: query_evidence_readout
    edge_pool: query_attention
    node_pool: query_attention
    dropout: 0.1

  edge_scorer:
    type: semantic_prior_residual
    use_semantic_prior: true
    prior_scale_init: 10.0
    new_text_weight_init: 0.1
    residual_scale_init: 0.1
    residual_hidden_dim: 1024
    residual_dropout: 0.1

  stop_head:
    type: evidence_sufficiency
    use_progress_bias: true
    use_reward_advantage: false

  flow_head:
    type: state_potential
```

Reward：

```yaml
reward:
  answer_f1_weight: 1.0
  support_weight: 0.5
  edge_cost: 0.03
  normalize_edge_cost_by_budget: true
  log_reward_clip_min: -30.0
```

Teacher/proposal：

```yaml
proposal:
  enabled: false
  teacher_mixture_warmup_steps: 0

auxiliary:
  stop_oracle_coef: 0.0
  teacher_edge_coef: 0.0
```

默认不要开 teacher auxiliary。先保证主算法干净。

---

# 17. 最终模块职责表

我给你压成一张“开发者能照着改”的职责表。

| 模块                      | 原先职责                                   | 问题                                              | 新职责                                         | 修改重点                                                             |
| ----------------------- | -------------------------------------- | ----------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------- |
| `policy.py`             | 拼 backbone/readout/head/scorer         | 表面 state-conditioned，实质 edge 不看 state           | 组装 evidence-flow policy                     | 输出 `EvidenceContext`，edge/stop/flow 共用 `state_h`                 |
| `backbone.py`           | 构造 q/node/relation embedding，附带一些结构特征  | 静态语义和结构状态混在一起                                   | 只做 static semantic feature bank             | 保留 `query_h,node_h,rel_h,anchor_mask,node_is_non_text`           |
| `state_readout.py`      | mean pooling anchor/node/edge，主要给 flow | 不 query-conditioned；不进入 edge scorer             | query-conditioned evidence graph readout    | selected edge attention + node attention + progress -> `state_h` |
| `edge_scorer.py`        | relation cosine + new text cosine      | `del state_h`，不是 reasoning                      | semantic prior + state-conditioned residual | `z = prior + MLP([h_s,h_q,edge_h])`                              |
| `action_head.py`        | progress + reward advantage STOP       | STOP 不是 evidence sufficiency；有 gold reward 依赖风险 | evidence sufficiency STOP                   | `stop_logit = MLP([h_s,h_q,progress]) + progress_bias`           |
| `flow_head`             | 估计 (\log F(s))                         | 和 action policy 弱耦合                             | partial evidence potential                  | `logF = MLP(h_s)`，与 edge/stop 共用状态                               |
| `executor.py`           | 执行动作、合法性、reward advantage              | policy input 混入 reward                          | environment transition + legality           | 不再把 reward advantage 默认喂给 STOP                                   |
| `sampling.py`           | temperature sampling + logprob         | 基本合理                                            | 只做采样和 logprob bookkeeping                   | target logprob 仍用 untempered policy                              |
| `reward.py`             | answer F1 + edge cost                  | 不奖励 support evidence                            | terminal evidence reward                    | 加 anchor-answer support/path connectivity                        |
| `losses.py`             | SubTB + 可能 auxiliary                   | 容易不纯                                            | 纯 SubTB，auxiliary 外挂                        | 主 loss 保持干净                                                      |
| `guidance.py`           | teacher/proposal/diagnostic            | 默认不进 online policy                              | behavior proposal / optional auxiliary      | 不把 teacher label 当 online feature                                |
| `state.py/state_ops.py` | 子图状态/frontier                          | 基本正确                                            | 保持纯图状态                                      | 不引入 GRU hidden state                                             |
| `buffers.py`            | 保存 rollout traces                      | 可能过多                                            | 只存 loss 必需量                                 | 不存大 context                                                      |

---

# 18. 为什么这套选择“有品位”

因为它克制。

它没有说“为了 reasoning，上 GNN”。这很粗糙。

它也没有说“为了 state-conditioned，拼 20 个特征”。这很土。

它保留当前算法里最合理的东西：

[
z_{\text{prior}}(q,e)
]

因为 pretrained relation/entity geometry 的确是 KGQA frontier expansion 的好先验。

但它补上当前最致命的缺口：

[
z_e=f(q,G_t,e)
]

也就是当前 partial evidence graph 真的影响下一步边选择。

它还避免了 GRU 的理论问题。状态仍然是：

[
s_t=(V_t,E_t)
]

不是：

[
s_t=(V_t,E_t,\tau_t)
]

所以同一个子图不会因为生成顺序不同变成不同状态。GFlowNet 的 subgraph-level distribution 叙事保持干净：

[
P(x|q)\propto R(x)
]

而不是绕到 trajectory marginalization 里。

---

# 19. 我建议的实际修改顺序

不要一口气全改。最优雅的工程顺序是：

## 第一步：只改 `state_readout.py` 和 `edge_scorer.py`

目标：

[
z_e=z_{\text{prior}}+\lambda z_{\text{res}}(h_s,h_q,\phi_e)
]

保留原 STOP、原 reward、原 SubTB。

这一步能回答最核心问题：**state-conditioned edge policy 是否比 local cosine 强。**

## 第二步：改 `action_head.py`

把 STOP 从 reward advantage 改成：

[
\ell_{\text{stop}}=\mathrm{MLP}([h_s,h_q,\rho])
]

保留 progress bias，默认不使用 reward advantage。

这一步解决 inference 语义问题。

## 第三步：改 `reward.py`

加入 support reward：

[
\log R
======

\lambda_A\log(\epsilon+\mathrm{F1})
+
\lambda_P\log(\epsilon+\mathrm{Support})
----------------------------------------

\lambda_C\mathrm{Cost}
]

这一步把 reward 从 answer-containing 子图改成 evidence-support 子图。

## 第四步：清理 `losses.py`

保证 SubTB 是主 loss，其他辅助项不混在里面。

## 第五步：再考虑 learned backward

如果前四步站住，再改 (P_B)。不要提前动。

---

# 20. 最后的判断

你现在的算法最大问题不是“没有 GNN”，而是：

[
\text{state representation 没有进入 transition decision}
]

所以最精准的手术是：

[
\boxed{
\text{local semantic prior}
\quad\longrightarrow\quad
\text{semantic prior + state-conditioned evidence residual}
}
]

这一步最小，但足以改变算法性质。

改完以后，你的论文主张可以从：

> 用 GFlowNet 采样 KG 子图。

升级成：

> 用 GFlowNet 在 partial evidence graph space 上学习 reward-proportional reasoning subgraph distribution；transition flow consists of a pretrained semantic prior and a learned state-conditioned evidence residual.

这句话就顺了。

现在不该做 GRU，不该堆特征，不该急着上复杂 GNN。先把 `state_readout -> edge_scorer / stop_head / flow_head` 这条链打通。否则你所有 GFlowNet 叙事都会被一句话打穿：**你的边选择根本不看当前推理状态。**
