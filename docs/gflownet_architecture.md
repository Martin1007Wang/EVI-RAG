# GFlowNet Mainline Architecture

这份文档只回答一个问题：

当前仓库里，`gflownet`、`tasks/answer_reachability` 和 `archive/policy` 分别负责什么，边界应该怎么理解。

## 1. 一眼看懂目录

```text
configs/model/gflownet.yaml
src/models/gflownet_module.py
src/models/policy/
src/models/training/
src/tasks/answer_reachability/
src/models/components/
src/archive/policy/
src/graph_runtime/
src/tasks/answer_reachability/exact_reachability.py
src/utils/metrics_io.py
```

## 2. 每层的职责

### 2.1 `configs/model/gflownet.yaml`

唯一模型配置入口。

这里定义：

- `policy_cfg`
- `training_cfg`
- `heuristic_cfg`
- `inference_cfg`
- optimizer / scheduler

如果你在做默认训练配置变更，应该先看这里。

### 2.2 `src/models/gflownet_module.py` + `src/models/policy/` + `src/models/training/`

这是唯一模型主线。

核心文件：

- `gflownet_module.py`：Lightning 入口，只负责组装组件并组织 train / val / predict
- `policy/search_policy.py`：主线 `GFlowNetPolicy`，负责 start / move / `log Z` / heuristic wiring
- `policy/heuristic.py`：heuristic bias 的单独封装层
- `policy/trajectory_policy.py`：不带 heuristic 的基础 trajectory policy machinery
- `training/sampler.py`：通用 sampled rollouts
- `training/losses.py`：`SubTB`

这层负责的事情是：

- 定义参数化策略
- 采样前向轨迹
- 计算训练损失
- 暴露训练、验证、预测接口

不应该把任务级指标聚合逻辑塞回这里以外的第二套模型壳。

### 2.3 `src/tasks/answer_reachability/`

这是任务层，不是第二个模型层。

核心文件：

- `src/tasks/runtime_protocol.py`：task runtime factory / runtime protocol
- `execution.py`：把模型策略接到 exact 分析和预测结果上
- `task_runtime.py`：answer reachability runtime factory 与 bound runtime
- `search.py`：support-window search
- `metrics.py`：answer / support metrics
- `posterior.py`：posterior 和 rank-only 结果整理
- `artifacts.py`：预测产物写出
- `runtime.py`：训练/评估入口校验

这层负责的事情是：

- 先把 `SearchPolicyProtocol` 组装成 task runtime
- exact answer mass 解释
- support coverage 计算
- 预测输出与下游产物

一句话说，它回答的是“这套策略在 answer reachability 任务上表现如何”，而不是“再实现一套模型”。

### 2.4 `src/tasks/answer_reachability/exact_reachability.py`

这是 exact 分析内核。

它依赖 `SearchPolicyProtocol`，也就是说它只关心：

- 怎么准备 batch
- 怎么得到 start distribution
- 怎么得到 forward distribution
- 怎么得到 state score

这让 exact 分析与具体模型类名解耦，但在当前仓库里实际主线仍然只有 `GFlowNetPolicy`。

而 task runtime factory 依赖的是 `SearchPolicyProtocol` 加上任务配置，负责把训练 sampler 和评估执行面绑定成一个 runtime；这层抽象与 exact 分析是相邻关系，不应该再塞回 `src/models/`。

### 2.5 `src/graph_runtime/`

这是共享图运行时层，不属于具体模型实现。

核心内容：

- `batch.py`：`TrajectoryBatch`
- `builder.py`：`build_graph_batch`
- `topology.py`：`GraphTopology`
- `observation.py`：`GraphObservation`
- `protocol.py`：`GraphBatchProtocol`

它被 `src/models/gflownet_module.py`、`src/models/policy/`、`src/models/training/`、task 运行时、
`datasets` 和归档实验分支共同依赖，因此不再继续挂在 `src/models/` 下面。

### 2.6 `src/models/components/`

这里暴露主线第一层、可直接由 `GFlowNetModule` 组装的 `torch.nn.Module` 组件。

- `embedding.py`
- `scoring.py`

也就是：

- `EmbeddingBackbone`
- `NodeFlowHead`
- `StartLogitHead`
- `GraphLogZHead`

它们分别对应 `BackboneConfig`、`StateScoreHeadConfig`、`StartHeadConfig`、`GraphLogZHeadConfig`，由 Lightning 组合层负责实例化。

### 2.7 `src/models/policy/` 内部运行时辅助

策略主线自己的运行时结构和纯函数辅助逻辑现在都收回到了
`src/models/policy/` 包内。

- `types.py`：`PreparedSearchBatch`、`HeuristicCache`
- `heuristic.py`：`TrajectoryHeuristic`
- `heuristic_utils.py`：`compute_topology_log_heuristic`、`compute_embedding_log_heuristic`
- `components/embedding.py`：`BackboneInput`、`BackboneOutput`

`src/models/components/gnn.py` 和 `src/models/components/heuristic_heads.py` 仍然存在，但它们是
`EmbeddingBackbone` / learned trajectory heuristic 的内部实现模块，不作为第一层公开组件导出。

这样可以保证公开的 `components` 只表达第一层模型组件，而运行时 glue 不再漂在
`src/models/` 顶层。

### 2.8 `src/archive/policy/`

这里是实验组件收纳区。

当前保留的内容包括：

- `PolicyEncoder`
- `QuestionContextModule`
- `EdgeScoreModule`
- 若干 token/path/edge 级辅助模块

这层的原则是：

- 可以保留研究型模块
- 可以保留对应测试
- 不能反向污染默认主线依赖

### 2.9 `src/tasks/answer_reachability/rollout.py`

这里放 answer reachability 专属的 rollout 监督语义：

- 哪些节点算 success terminal
- failure reward 如何归一化

这样 `src/models/training/sampler.py` 只保留通用 rollout 机制，不再直接知道 answer 节点或 gold-terminal 规则。

## 3. 推荐依赖方向

当前推荐的依赖方向是：

```text
graph_runtime / components / configs
    -> policy / training / gflownet_module
    -> task runtime / exact reachability analysis
```

以及：

```text
archive/policy  (side branch, not on default runtime path)
```

更具体地说：

- `policy` / `training` / `gflownet_module` 可以依赖 `graph_runtime`、`components`、`configs`
- `answer_reachability` 可以依赖 `src/models/policy` 暴露的 protocol 和 task-local exact inference
- `archive/policy` 不应该成为默认训练路径的必经依赖

## 4. 修改代码时该去哪一层

### 4.1 想改模型行为

优先看：

- `src/models/policy/search_policy.py`
- `src/models/policy/heuristic.py`
- `src/models/components/heuristic_heads.py`（learned trajectory heuristic 内部 head）
- `src/models/policy/heuristic_utils.py`
- `src/models/policy/trajectory_policy.py`

### 4.2 想改训练损失

优先看：

- `src/models/training/losses.py`
- `src/models/configs/gflownet.py`
- `configs/model/gflownet.yaml`

### 4.3 想改验证 / 预测指标

优先看：

- `src/tasks/answer_reachability/execution.py`
- `src/tasks/answer_reachability/metrics.py`
- `src/tasks/answer_reachability/posterior.py`

### 4.4 想改 support-window search

优先看：

- `src/tasks/answer_reachability/search.py`
- `src/tasks/answer_reachability/exact_reachability.py`

### 4.5 想做高风险实验

先放到：

- `src/archive/policy/`

等它真的进入默认训练/评估面，再决定是否提到主线目录。

## 5. 当前架构的约束

为了保持主线稳定，当前建议遵守下面几条：

- 只保留一个默认模型配置：`configs/model/gflownet.yaml`
- 只保留一个默认模型类：`src.models.gflownet_module.GFlowNetModule`
- 只保留一个默认训练目标：`SubTB`
- `heuristic.kind` 只作为主线内部变体，不要再演化出第二套模型族
- experimental 模块默认不对外 re-export 到主线命名空间

## 6. 最后一句话

现在这套结构可以用一句话概括：

```text
src/models/gflownet_module.py + src/models/policy/ + src/models/training/
负责“学一个轨迹策略”，
answer_reachability 负责“精确分析这个策略在任务上的答案质量与支持覆盖”，
archive/policy 负责“先把新想法放到隔离区里试”。
```
