# Trajectory GFlowNet 彻底重构设计（非兼容）

> 文档目的：为本仓库下一阶段的大重构提供唯一设计蓝图。本文档明确要求硬切换，不做兼容层、不保留双语义路径、不接受旧接口共存为长期状态。
>
> 基线时间：`2026-03-07`
>
> 当前实现参考：`src/models/algorithms/dual_flow.py`、`src/models/planner/exact_flow.py`、`src/models/policy/dual_flow_policy.py`、`src/rollout/engine/sampler.py`、`src/callbacks/rollout_pipeline.py`

---

## 1. 重构结论

本次重构的目标不是“把现有 exact planner 和 rollout engine 拼起来”，而是把仓库改造成一个单一语义的 `Trajectory GFlowNet` 系统。新的系统只承认一个核心对象：

\[
P_F(\tau \mid q, G),
\]

即在问题条件下、检索子图上的前向轨迹分布。

训练、采样、搜索、评估、导出都必须围绕这一对象展开。

因此，本次重构做如下强制决定：

1. 删除 exact planner 作为主训练/预测路径，不再以 `ExactFlowPlanner` 为算法中心。
2. 删除 super-source / super-node 运行时增强，不再允许模型输入图被 DataModule 隐式改写。
3. 删除 child-normalized transfer kernel 语义，统一改为 source-normalized forward policy。
4. 删除 callback 驱动的 rollout 兼容输出层，统一使用 typed result/schema。
5. 删除“双路径长期共存”策略；旧 `DualFlow` 路径只允许在开发分支短暂存在，最终合并时必须清除。

这意味着：最终主分支上不应再存在“exact path 是主路径，rollout path 是支线”的架构。

---

## 2. 当前仓库的结构性问题

### 2.1 当前 active path 不是路径级 GFlowNet runtime

当前训练/验证/测试/预测主路径为：

```text
DualFlowModule
  -> compute_policy_kernel()
  -> ExactFlowPlanner.run()
  -> entity marginal
  -> per-entity best path decode
```

对应文件：

- `src/models/algorithms/dual_flow.py`
- `src/models/planner/exact_flow.py`
- `src/models/planner/path_decoder.py`

这条路径的优点是稳定，但它把轨迹分布压缩成了实体边际分布，最后只为每个实体返回一条 best path。它不是真正意义上的“路径级采样系统”。

### 2.2 rollout 栈存在，但没有成为主语义

仓库内已有完整 rollout 体系：

- `src/rollout/engine/sampler.py`
- `src/rollout/engine/engines/online.py`
- `src/rollout/engine/engines/beam_decoder.py`
- `src/models/policy/dual_flow_policy.py`

但它没有成为 train/predict 的唯一执行路径；exact planner 和 rollout engine 各维护了一套状态语义、步数语义和输出语义。

### 2.3 数据层被运行时偷偷改写

`src/datasets/batch_adapter.py` 会在 `prepare_batch()` 中自动注入双 super node 和 super edges。这样做直接破坏了以下设计原则：

- 数据 SSOT 应该是 `q_local_indices` / `a_local_indices`
- 子图应反映真实检索上下文，而不是运行时人工改造后的扩展图
- 模型层的 start/stop 语义不应通过改图来实现

### 2.4 输出协议没有唯一真相

当前 `RolloutPipelineCallback` 同时兼容：

- `list[dict]`
- `RolloutBatch`

这表明预测输出协议没有唯一 SSOT。对于弹性窗口、概率质量分析、未检索质量统计而言，这是不可接受的。

### 2.5 配置被拆成两套不可对齐语义

当前存在两类核心配置：

- exact planner：`src/models/configs/objective.py`
- rollout/beam：`src/models/configs/search.py`

这导致：

- `emit_min_steps` 和 `stop_min_steps` 同时存在
- `predict_topk` / `predict_mass_threshold` 与 beam/rollout 参数割裂
- exact 和 rollout 的步数语义不同

这类分裂在彻底重构中必须被消灭。

---

## 3. 新系统的设计原则

### 3.1 单一概率对象

新系统只承认一类生成对象：终止轨迹 \(\tau\)。

一切监督、采样、搜索、评估都围绕：

\[
P_F(\tau \mid q, G).
\]

### 3.2 路径优先，答案次之

答案分布不再是模型的第一输出；答案分布应由路径分布边际化得到：

\[
P(y \mid q, G)=\sum_{\tau:\operatorname{term}(\tau)=y} P_F(\tau \mid q, G).
\]

### 3.3 数据图不允许被运行时改造

新系统中：

- `q_local_indices` / `a_local_indices` 是数据 SSOT
- 模型直接在真实检索子图上定义 start / target / stop 语义
- 不允许 DataModule 注入虚拟节点、虚拟边

### 3.4 训练与推理共享同一前向策略

训练采样器与预测搜索器都必须调用同一个 forward policy：

- 训练：从 \(P_F\) 采样轨迹
- 预测：按 \(P_F\) 做 mass-adaptive trajectory search

不允许出现“训练用一套语义，推理用另一套 exact kernel”的长期并存。

### 3.5 弹性窗口是一级概念

预测输出不再是固定 top-k。窗口大小由轨迹概率质量自动决定：

\[
W_\rho(x)=\text{cover-at-least-}\rho\text{-mass 的最小 rollout 集合}.
\]

### 3.6 未检索质量必须显式量化

预测阶段必须输出：

- tail rollout mass
- missed gold-answer mass
- covered gold-answer mass

而不是仅输出 `hit@k` 一类指标。

---

## 4. 目标算法：Trajectory GFlowNet

### 4.1 问题定义

对每个样本，给定问题条件 \(q\) 与检索子图

\[
G=(V,E), \quad E \subseteq V \times R \times V.
\]

数据提供：

- 起点集合 \(Q \subseteq V\)，由 `q_local_indices` 给出
- 答案节点集合 \(A \subseteq V\)，由 `a_local_indices` 给出
- gold 答案实体集合 \(Y^*\)，由 `answer_entity_ids` 给出

设最大真实步数为 \(H\)，最小允许停止步数为 \(m\)。

### 4.2 状态与动作

前向状态定义为：

\[
s_t=(u_t, t, q, G),
\]

其中 \(u_t\) 为当前节点，\(t\) 为已经走过的真实边数。

动作集合：

\[
\mathcal A(s_t)=\operatorname{Out}(u_t) \cup \{\text{STOP}\}.
\]

约束：

- 当 \(t < m\) 时禁止 `STOP`
- 当 \(t = H\) 时禁止继续 move，必须终止

### 4.3 起点分布

起点策略定义为：

\[
P_{\text{start}}(u_0 \mid q,G)=\operatorname{softmax}_{u\in Q} \; \omega_\theta(u,q,G).
\]

不再通过 super-source 边来表达起点选择。

### 4.4 前向策略

对任意状态 \(s_t\)，前向策略定义为：

\[
P_F(a_t \mid s_t)=\operatorname{softmax}(\ell^{\text{fwd}}_\theta(s_t, a_t)), \quad a_t \in \mathcal A(s_t).
\]

这里 `STOP` 与真实边动作同处一个 source-wise softmax 中。新的 forward head 直接输出状态上的动作 logits，不再构造 child-normalized transfer kernel。

### 4.5 后向策略

为了训练 GFlowNet，定义后向策略：

\[
P_B(a_t^{-1} \mid s_{t+1})
\]

它只用于训练，不再作为预测模式暴露给用户。后向策略的职责是给 DB/SubTB 提供逆向归一化概率，而不是承担另一套推理流程。

### 4.6 状态流

定义状态流头：

\[
\log F_\theta(s_t).
\]

以及图级配分项：

\[
\log Z_\theta(q,G).
\]

其中：

- `logF` 是状态级势函数
- `logZ` 用于把图级起点选择纳入同一 flow 方程

### 4.7 完整轨迹概率

一条终止轨迹记为：

\[
\tau=(u_0, a_0, u_1, a_1, \ldots, u_T, \text{STOP}), \quad T \le H.
\]

其概率定义为：

\[
P_F(\tau \mid q,G)
= P_{\text{start}}(u_0 \mid q,G)
  \cdot \prod_{t=0}^{T} P_F(a_t \mid s_t).
\]

这是新系统唯一承认的生成分布。

### 4.8 终止奖励

终止奖励先采用最小版本：

\[
R(\tau)=
\begin{cases}
1, & \operatorname{term}(\tau) \in Y^* \\
\epsilon, & \operatorname{term}(\tau) \notin Y^*
\end{cases}
\]

其中 \(\epsilon > 0\) 很小。

v1 不把 diversity shaping 写入 reward。本次重构先保证轨迹分布与窗口搜索的一致性，多样性主要来自 GFlowNet 的多峰轨迹建模。

### 4.9 训练目标：Stepwise Detailed Balance

新系统默认训练目标改为 stepwise DB。对采样得到的轨迹 \(\tau\)：

起点约束：

\[
\mathcal L_{\text{start}}
= \left(
  \log Z(x) + \log P_{\text{start}}(u_0 \mid x) - \log F(s_0)
\right)^2.
\]

中间步约束：

\[
\mathcal L_{\text{move},t}
= \left(
  \log F(s_t)
  + \log P_F(a_t \mid s_t)
  - \log F(s_{t+1})
  - \log P_B(a_t^{-1} \mid s_{t+1})
\right)^2.
\]

终止约束：

\[
\mathcal L_{\text{stop}}
= \left(
  \log F(s_T)
  + \log P_F(\text{STOP} \mid s_T)
  - \log R(\tau)
\right)^2.
\]

总损失：

\[
\mathcal L
= \lambda_{\text{start}} \mathcal L_{\text{start}}
+ \lambda_{\text{move}} \sum_t \mathcal L_{\text{move},t}
+ \lambda_{\text{stop}} \mathcal L_{\text{stop}}.
\]

后续若需要更稳的长轨迹训练，可在同一架构上切换到 SubTB，但 v1 先落地 stepwise DB。

### 4.10 Gold-answer mass analyzer

为了量化“模型知道多少 gold 质量”与“窗口丢了多少 gold 质量”，新系统增加一个独立分析器，在有限时域上对 source-normalized 前向策略做精确 DP。

定义：

\[
V_{\text{gold}}(u,t)
= P_F(\text{STOP}\mid u,t) \cdot \mathbf 1[\operatorname{entity}(u) \in Y^*]
+ \sum_{u\to v} P_F(u\to v\mid u,t) V_{\text{gold}}(v,t+1).
\]

则样本总 gold 质量为：

\[
P_{\text{gold}}(x)=\sum_{u\in Q} P_{\text{start}}(u\mid x) V_{\text{gold}}(u,0).
\]

这个量与训练损失解耦，只用于分析和评估。

### 4.11 弹性窗口搜索：Mass-Adaptive Trajectory Search

设前缀 \(\alpha\) 的概率为 \(p(\alpha)\)。由于后缀条件概率和为 1，前缀整棵后代子树的总质量即为：

\[
\operatorname{Mass}(\text{subtree}(\alpha)) = p(\alpha).
\]

因此，可在 prefix tree 上做 best-first expansion，并保持如下不变量：

\[
\sum_{\tau \in \mathcal C} P_F(\tau \mid x)
+ \sum_{\alpha \in \mathcal F} p(\alpha)
= 1,
\]

其中：

- \(\mathcal C\)：已完成轨迹集合
- \(\mathcal F\)：当前前沿前缀集合

对给定质量阈值 \(\rho\)，定义：

\[
K_\rho(x)=\min\left\{k:\sum_{i=1}^{k} P_F(\tau_{(i)}\mid x) \ge \rho\right\},
\]

其中 \(\tau_{(i)}\) 是按概率降序枚举出的完整轨迹。返回窗口：

\[
W_\rho(x)=\{\tau_{(1)},\ldots,\tau_{(K_\rho(x))}\}.
\]

这就是新的预测输出核心。

---

## 5. 预测与评估指标

### 5.1 窗口内检索质量

对窗口 \(W_\rho(x)\)，定义：

\[
A(W)=\{\operatorname{term}(\tau): \tau \in W\},
\]

\[
V(W)=\bigcup_{\tau \in W} \operatorname{nodes}(\tau).
\]

指标：

- `elastic_hit@rho`
- `elastic_answer_recall@rho`
- `elastic_context_recall@rho`
- `elastic_context_precision@rho`
- `elastic_context_f1@rho`

### 5.2 窗口效率

- `elastic_window_size@rho`
- `elastic_mass@rho`
- `elastic_unique_answers@rho`
- `elastic_unique_paths@rho`

### 5.3 多样性

路径熵：

\[
H_{\text{path}}(W)=-\sum_{\tau \in W} \tilde p(\tau) \log \tilde p(\tau),
\]

答案熵：

\[
H_{\text{ans}}(W)=-\sum_{y} \tilde P(y\mid W) \log \tilde P(y\mid W).
\]

并报告：

- `path_entropy@rho`
- `answer_entropy@rho`
- `effective_paths@rho = exp(H_path)`
- `effective_answers@rho = exp(H_ans)`

### 5.4 未检索质量

总尾质量：

\[
\operatorname{tail\_rollout\_mass}(W)=1-\sum_{\tau \in W} P_F(\tau \mid x).
\]

gold 覆盖质量：

\[
\operatorname{covered\_gold\_mass}(W)
= \sum_{\tau \in W,\operatorname{term}(\tau)\in Y^*} P_F(\tau \mid x).
\]

gold 丢失质量：

\[
\operatorname{missed\_gold\_mass}(W)
= P_{\text{gold}}(x)-\operatorname{covered\_gold\_mass}(W).
\]

这三个量是本次重构后最重要的检索质量解释指标。

---

## 6. 新的数据契约

### 6.1 目标：`TrajectoryBatch`

新模型输入必须是 typed batch，而不是 `{"inputs": prepared, "metadata": metadata}` 的双 dict。

建议定义：

```python
@dataclass(frozen=True)
class TrajectoryBatch:
    num_graphs: int
    node_ptr: torch.Tensor
    edge_index: torch.Tensor
    edge_rel_global: torch.Tensor
    edge_batch: torch.Tensor
    node_batch: torch.Tensor

    node_embeddings: torch.Tensor
    edge_embeddings: torch.Tensor
    question_emb: torch.Tensor
    question_ctx: torch.Tensor
    question_ctx_mask: torch.Tensor

    q_local_indices: torch.Tensor
    q_ptr: torch.Tensor
    a_local_indices: torch.Tensor
    a_ptr: torch.Tensor
    answer_entity_ids: torch.Tensor
    answer_ptr: torch.Tensor

    node_global_ids: torch.Tensor
    sample_ids: list[str]
    questions: list[str]
    dataset_scope: str
```

### 6.2 明确删除的运行时字段

以下字段不再作为新模型输入的一部分：

- super nodes
- super edges
- `relation_tokens` 作为 batch 预计算字段
- `node_tokens` 作为 batch 预计算字段
- `dummy_mask` 作为模型主逻辑分叉条件
- `prepared/metadata` 双 dict 协议

### 6.3 保留的数据 SSOT

以下字段必须保持不变：

- `q_local_indices`
- `a_local_indices`
- `q_ptr`
- `a_ptr`
- `answer_entity_ids`
- `answer_ptr`

它们是数据层真相，不允许通过交换、覆盖或 super-node 注入来改写语义。

---

## 7. 新的代码拓扑

建议建立一个全新的子系统，而不是在现有 `dual_flow` 文件上做 patch。目标目录：

```text
src/models/trajectory_gfn/
  __init__.py
  batch.py
  module.py
  encoder.py
  state.py
  heads.py
  policy.py
  transition.py
  sampler.py
  losses.py
  reward.py
  search.py
  analyzer.py
  schema.py
  metrics.py
```

职责划分如下。

### 7.1 `batch.py`

- 定义 `TrajectoryBatch`
- 定义 batch-level 校验函数
- 只处理 typed contract，不做模型逻辑

### 7.2 `module.py`

- 定义 `TrajectoryGFlowNetModule`
- 负责 `training_step` / `validation_step`
- 负责调用 sampler、loss、analyzer、search
- 不直接写 artifact，不自己拼 schema dict

### 7.3 `encoder.py`

- 复用当前图编码器和 question conditioning 思路
- 输出节点表示、关系表示、问题表示
- 不关心 rollout、loss、search

### 7.4 `state.py`

- 定义训练/推理共用的状态结构
- 至少包含：`current_node`, `step_t`, `done_mask`
- 明确建模 step embedding / remaining-step embedding

### 7.5 `heads.py`

- `StartHead`
- `ForwardActionHead`
- `BackwardActionHead`
- `FlowHead`
- 可选 `GraphPartitionHead`

### 7.6 `policy.py`

统一暴露四个唯一入口：

- `compute_start_distribution()`
- `compute_forward_distribution()`
- `compute_backward_distribution()`
- `compute_log_flow()`

训练与预测都只能调用这四类接口。

### 7.7 `transition.py`

- 唯一状态转移函数
- 统一处理 STOP、move、step 增加
- sampler 与 search 都必须复用这里

### 7.8 `sampler.py`

- 训练期 on-policy forward rollout sampler
- 输出 `TrajectorySampleBatch`
- 不负责任何导出格式兼容

### 7.9 `losses.py`

- 实现 stepwise DB
- 后续若需要可加入 SubTB
- loss 层不直接读 raw graph，只消费 sample batch 与 model outputs

### 7.10 `reward.py`

- 只负责终止 reward 计算
- 不承担 artifact/export 功能

### 7.11 `search.py`

- 实现 `MassAdaptiveTrajectorySearch`
- 在 prefix tree 上做 best-first search
- 输出 `ElasticWindowBatch`

### 7.12 `analyzer.py`

- 实现 `AnswerMassAnalyzer`
- 精确计算 entity marginal、gold mass、missed gold mass 所需 DP 量

### 7.13 `schema.py`

定义新的 typed 输出：

- `TrajectoryRecord`
- `ElasticWindowResult`
- `ElasticEvalBatch`

### 7.14 `metrics.py`

- 计算 elastic metrics
- 计算 diversity metrics
- 计算 missed-mass metrics
- 只接受 typed outputs，不接受 `list[dict]` 兼容协议

---

## 8. 与现有代码的关系：删除、重写、复用

### 8.1 直接删除的文件

这些文件代表旧语义中心，最终主分支应删除：

- `src/models/algorithms/dual_flow.py`
- `src/models/planner/exact_flow.py`
- `src/models/planner/path_decoder.py`
- `src/models/policy/policy_kernel.py`
- `src/datasets/batch_adapter.py`
- `src/callbacks/rollout_pipeline.py`
- `src/rollout/export/schema.py`
- `src/rollout/engine/sampler.py`
- `src/rollout/engine/engines/online.py`
- `src/rollout/engine/engines/beam_decoder.py`
- `src/models/configs/objective.py`
- `src/models/configs/search.py`

这些文件不是“局部有问题”，而是它们承载了旧架构的多中心语义：exact planner、rollout engine、callback exporter、search config 分裂共存。新系统不应保留这种历史包袱。

### 8.2 必须重写的文件

以下文件的核心思想可保留，但实现必须彻底重写：

- `src/models/policy/dual_flow_policy.py`
- `src/models/policy/action_head.py`
- `src/models/policy/backward_head.py`
- `src/models/reward/reward_engine.py`

重写要求：

- 删除 child-normalized transfer kernel
- 删除 exact-only kernel 构造逻辑
- 删除兼容旧 rollout engine 的输出结构
- 统一切到 source-wise forward/backward distributions

### 8.3 可复用的文件

以下模块可作为底层积木保留：

- `src/models/backbone/backbone.py`
- `src/models/backbone/gnn.py`
- `src/models/policy/edge.py`
- `src/models/policy/question.py`
- `src/utils/segment_ops.py`
- `src/datasets/g_retrieval_dataset.py`
- `src/datasets/g_retrieval_collate.py`
- `src/datasets/g_retrieval_datamodule.py`

但保留并不意味着接口不变。尤其 `g_retrieval_datamodule.py` 需要切换到新的 typed batch 输出，不再经过旧 `DualFlowBatchAdapter`。

---

## 9. 新的配置拓扑

本次重构后，模型配置只允许保留一棵树，不再拆成 planner/search 两套核心配置。

建议：

```yaml
model:
  _target_: src.models.trajectory_gfn.module.TrajectoryGFlowNetModule

  horizon:
    max_steps: 4
    min_stop_steps: 1

  policy:
    embedding_dim: 1024
    hidden_dim: 512
    gnn_layers: 2
    gnn_dropout: 0.1
    use_adapter: true
    adapter_dim: 128

  training:
    loss_type: db
    rollout_batch_size: 8
    lambda_start: 1.0
    lambda_move: 1.0
    lambda_stop: 1.0
    reward_epsilon: 1.0e-3

  inference:
    mode: adaptive_mass
    mass_threshold: 0.90
    max_expansions: 20000
    max_frontier_size: 4096

  analyzer:
    compute_gold_mass: true
    compute_entity_marginals: true
```

新增配置文件建议：

- `configs/model/trajectory_gfn.yaml`
- `configs/experiment/train_trajectory_gfn.yaml`
- `configs/run/eval_trajectory_gfn.yaml`

旧字段名如 `planner_cfg`, `rollout_cfg`, `beam_cfg`, `predict_topk`, `predict_min_paths` 不再保留。

---

## 10. 训练、预测、评估流程

### 10.1 训练流程

```text
for each batch:
  1) build TrajectoryBatch
  2) encode graph and question
  3) sample on-policy forward trajectories
  4) compute log_pf / log_pb / logF / rewards
  5) compute DB loss
  6) optimize
```

关键点：训练不再调用 exact planner。

### 10.2 验证流程

验证分两层：

1. 训练层验证：
   - DB residual statistics
   - rollout hit / reward / entropy

2. 搜索层验证：
   - adaptive mass search
   - elastic retrieval quality
   - missed mass quality

### 10.3 预测流程

```text
for each batch:
  1) encode graph and question
  2) run MassAdaptiveTrajectorySearch until cumulative mass >= rho
  3) run AnswerMassAnalyzer for gold mass accounting
  4) emit ElasticWindowBatch
  5) compute/write metrics and artifacts
```

预测的第一输出是“概率有序的路径窗口”，不是“实体 top-k + 每实体 best path”。

### 10.4 评估驱动方式

建议预测/评估不再依赖 callback 去偷偷消费 `trainer.predict()` 输出。应改为显式 evaluator：

- `src/eval.py` 或新建 `src/eval_trajectory.py`
- 直接迭代 dataloader
- 显式调用 `model.generate_windows(batch)`
- 显式聚合 metrics / 写文件

这样可以彻底消除隐式兼容逻辑。

---

## 11. 新的输出协议

建议定义如下 schema。

### 11.1 `TrajectoryRecord`

```python
@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    rollout_rank: int
    log_prob: float
    prob: float
    cumulative_mass: float
    terminal_entity_id: int
    is_gold: bool
    edges: list[EdgeRecord]
```

### 11.2 `ElasticWindowResult`

```python
@dataclass(frozen=True)
class ElasticWindowResult:
    sample_id: str
    dataset_scope: str
    mass_threshold: float
    window_size: int
    covered_mass: float
    tail_rollout_mass: float
    gold_total_mass: float
    covered_gold_mass: float
    missed_gold_mass: float
    unique_answer_count: int
    unique_path_count: int
    trajectories: list[TrajectoryRecord]
```

所有下游 artifact writer、metric processor、可视化工具都只接受这一协议。

---

## 12. 实施顺序

本次重构不做长期兼容，但仍建议按依赖顺序实现，避免在开发期自相矛盾。

### 阶段 0：分支切断

- 新建重构分支
- 明确 merge 目标是“完全切换”而不是“保留旧逻辑”
- 在文档与 issue 中写清楚：不接受旧 API 的兼容诉求

### 阶段 1：数据契约切换

- 新建 `TrajectoryBatch`
- 修改 `g_retrieval_datamodule.py`
- 删除 `DualFlowBatchAdapter`
- 停止 super-source augmentation

完成标准：模型可以拿到 typed batch，且 batch 中不存在虚拟节点/虚拟边。

### 阶段 2：策略重写

- 建立 `trajectory_gfn/policy.py`
- 建立 `StartHead` / `ForwardActionHead` / `BackwardActionHead` / `FlowHead`
- 加入 step embedding / remaining-step embedding

完成标准：forward/backward/start/logF 四类分布全部可前向计算。

### 阶段 3：训练路径切换

- 实现 `ForwardRolloutSampler`
- 实现 DB loss
- 新建 `TrajectoryGFlowNetModule`
- 删掉对 `ExactFlowPlanner` 的依赖

完成标准：训练可以在 toy graph 上稳定收敛，且 train loop 中不再出现 exact planner。

### 阶段 4：预测路径切换

- 实现 `MassAdaptiveTrajectorySearch`
- 实现 `AnswerMassAnalyzer`
- 输出 `ElasticWindowResult`

完成标准：predict 输出的是概率有序路径窗口，而不是 entity-best-path 列表。

### 阶段 5：评估与导出重写

- 重做 metrics
- 重做 artifact writer
- 去掉 callback 兼容层

完成标准：评估主指标变成 `elastic_*` / `missed_*`。

### 阶段 6：旧文件清理

- 删除旧 exact path
- 删除旧 rollout engines
- 删除旧 configs
- 删除所有兼容分支

完成标准：主分支不存在第二套 forward semantics。

---

## 13. 测试与验收标准

建议新增以下测试文件：

- `tests/trajectory_gfn/test_policy_normalization.py`
- `tests/trajectory_gfn/test_db_identity.py`
- `tests/trajectory_gfn/test_sampler_shapes.py`
- `tests/trajectory_gfn/test_search_prefix_mass_conservation.py`
- `tests/trajectory_gfn/test_search_monotonicity.py`
- `tests/trajectory_gfn/test_gold_mass_analyzer.py`
- `tests/trajectory_gfn/test_elastic_metrics.py`

必须保证的数学不变量：

1. `start` 分布归一化
2. 每个状态上 forward policy 归一化
3. 每个状态上 backward policy 归一化
4. DB 残差在可解 toy case 上接近 0
5. prefix search 过程中 `completed_mass + frontier_mass = 1`
6. `elastic_window_size@rho` 对 \(\rho\) 单调不减
7. `covered_mass@rho` 对 \(\rho\) 单调不减
8. `missed_gold_mass@rho` 对 \(\rho\) 单调不增

---

## 14. 完成定义

只有当以下条件全部满足时，本次重构才算完成：

1. `train` / `val` / `predict` 都以 `TrajectoryGFlowNetModule` 为唯一模型入口。
2. 仓库中不存在 exact planner 主路径。
3. 仓库中不存在 super-source runtime augmentation。
4. 预测输出是 typed elastic trajectory window，而不是 `list[dict]` 兼容结构。
5. 主评估指标切换为 `elastic_*`、`path_*`、`answer_*`、`missed_*`。
6. 所有旧配置与旧模块在 merge 前被清除，不保留长期 deprecated 状态。

---

## 15. 一句话版本

本次重构的本质，是把仓库从“exact planner 主导、rollout 为辅的 DualFlow 混合系统”，改造成“以轨迹分布为唯一真相的 Trajectory GFlowNet 系统”，并把路径级采样、答案多样性、弹性窗口与未检索质量统一纳入同一套数学与工程协议中。
