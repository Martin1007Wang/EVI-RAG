# Answer Reachability 重构复盘

这份文档不再试图维护旧实现的逐行对照，而是解释这次架构收口到底做了什么、为什么这么做。

当前结论很明确：

- 仓库只保留一条 `gflownet` 主线
- `h` 函数只保留三个变体：`topology`、`embedding`、`learned`
- 训练目标只保留 `SubTB`
- `answer_reachability` 只保留任务层职责：exact 分析、support-window search、评估产物

## 1. 我们最终淘汰了什么

### 1.1 旧的命名分叉

这次重构前，仓库里同时存在过这些命名：

- `trajectory_policy`
- `trajectory_gfn`
- `answer_reachability`
- `guidance`

它们的问题不是名字不同这么简单，而是会造成：

- 同一概念被多次包装
- 配置入口重复
- 测试和运行时依赖历史 alias
- 新代码很难判断哪条才是主线

现在这些分叉都已经收敛到：

- model: `GFlowNetModule`
- policy: `GFlowNetPolicy`
- config: `configs/model/gflownet.yaml`
- task runtime: `src/tasks/answer_reachability/*`

### 1.2 旧的训练面分叉

更大的问题其实在训练目标上。之前仓库先后尝试过：

- reward / stop 语义驱动的轨迹训练
- exact gold-marginal 任务中心化训练
- 多种 balance 风格目标并行存在的历史痕迹

这些探索并不全是错的，但同时保留在主线上会让代码长期处于“迁移中间态”。

这次决定是明确的：

- 主训练目标只有 `SubTB`
- exact answer reachability 不再承担训练目标，而是承担解释和评估任务

## 2. 为什么最后选择单一 GFlowNet 主线

这次收口背后的判断有三个。

### 2.1 仓库真实使用面已经只有一条主线

无论从配置、训练入口还是测试面来看，仓库现在真正持续维护的只有一套图搜索策略。

既然不存在并行维护两种模型家族的现实需求，那么继续保留多套入口只会增加维护噪音。

### 2.2 trajectory heuristic 是合理的可变因子，损失不是

从实验角度看，`h` 函数确实是一个自然变体：

- 你可以换成 topology prior
- 可以换成 embedding prior
- 也可以让模型学一个 small head

但损失函数不是同一个层级的问题。损失一旦并行存在：

- 日志面会变乱
- 配置会重复
- 训练语义会变得难以解释

因此我们保留“同一主线下的 trajectory heuristic 变体”，删除“多个主损失同时存在”的结构。

### 2.3 exact answer reachability 更适合作为任务层

`answer_reachability` 这部分最有价值的，不是再维护一套模型，而是：

- exact analyzer
- support-window search
- posterior aggregation
- artifact writing

这些能力都属于任务层解释与评估，不应该再与训练模型壳纠缠在一起。

## 3. 当前架构的最终分工

### 3.1 `src/models/gflownet_module.py` + `src/models/policy/` + `src/models/training/`

这里是唯一主线，负责：

- start distribution
- move policy
- graph log Z
- trajectory heuristic integration
- rollout sampler
- SubTB loss
- Lightning train/val/predict surface

### 3.2 `src/tasks/answer_reachability`

这里负责任务层：

- exact answer mass analysis
- support-window search
- rank metrics
- predict artifacts

这层消费 `gflownet` 策略，但不再自带第二套模型实现。

### 3.3 `src/archive/policy`

这里保留实验性质较强的 token-level policy building blocks。它们有研究价值，但：

- 不再伪装成主线依赖
- 不再通过 `src/models/policy` 兼容壳暴露
- 不再影响默认训练/评估路径

## 4. 这次重构的直接收益

### 4.1 入口更少

现在只需要记住：

- `configs/model/gflownet.yaml`
- `src.models.gflownet_module.GFlowNetModule`

### 4.2 责任边界更清楚

- 模型训练逻辑在 `src/models/gflownet_module.py`、`src/models/policy/`、`src/models/training/`
- 任务解释逻辑在 `answer_reachability`
- 实验组件在 `archive/policy`

### 4.3 实验面更容易解释

现在一个实验最核心的自由度就是：

- 选哪种 `heuristic.kind`
- 调哪些 SubTB / sampler / inference 超参

而不是先解释“这次到底走的是哪条历史分支”。

## 5. 当前明确不做的事

为了避免架构再次回到漂移状态，当前主线明确不做：

- 恢复 `trajectory_policy` 兼容壳
- 恢复 `guidance_cfg` 旧命名
- 同时保留多个主损失
- 在 `src/models/policy` 根目录重新塞回实验组件

如果以后真的要引入第二个模型家族，应该以独立目标和独立运行面重新设计，而不是继续复活这批历史 alias。

## 6. 建议的阅读方式

如果你想继续在这套架构上改代码，建议按下面顺序读：

1. `docs/gflownet_architecture.md`
2. `docs/answer_reachability_algorithm.md`
3. `docs/answer_reachability_math_derivation.md`
4. `src/models/gflownet_module.py`
5. `src/tasks/answer_reachability/execution.py`

这样会比从历史分支名倒推当前实现轻松很多。
