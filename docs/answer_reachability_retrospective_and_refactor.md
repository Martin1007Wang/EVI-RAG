# Answer Reachability Refactor Notes

这份笔记只保留当前仍然有参考价值的重构结论，旧路径与旧模块名已经更新到现行实现。

## 1. 这轮重构解决了什么

- 用 `src/models/gflownet_module.py` 取代历史 `trajectory_gfn/module.py`。
- 把训练逻辑集中到 `src/models/training/`。
- 把评估逻辑集中到 `src/metrics/answer_reachability/`。
- 把图批处理抽到 `src/graph_runtime/`，避免 model/metric 重复维护 batch 语义。
- 把任务级调度抽到 `src/runs/`，让 train/eval contract、dataset variant、输出落盘不再散落在入口脚本里。

## 2. 当前目录分工

- `src/models/`: 可训练主线与策略组件。
- `src/models/training/`: 只放训练期采样、reward supervisor、loss。
- `src/metrics/answer_reachability/`: 只放验证/测试/预测期 exact/search/metrics。
- `src/runs/`: 只放任务运行器和输出编排。
- `src/archive/`: 只放历史兼容代码，不参与主线。

## 3. 旧路径到新路径的迁移表

- `src/models/trajectory_gfn/module.py` -> `src/models/gflownet_module.py`
- `src/models/trajectory_gfn/sampler.py` -> `src/models/training/sampler.py`
- `src/models/trajectory_gfn/losses.py` -> `src/models/training/losses.py`
- `src/models/trajectory_gfn/reward.py` -> `src/models/training/answer_reachability.py`
- `src/models/trajectory_gfn/batch.py` -> `src/graph_runtime/__init__.py`
- `src/tasks/answer_reachability/execution.py` -> `src/metrics/answer_reachability/execution.py`
- `src/tasks/answer_reachability/posterior.py` -> `src/metrics/answer_reachability/posterior.py`
- `src/tasks/answer_reachability/metrics.py` -> `src/metrics/answer_reachability/metrics.py`
- `src/tasks/answer_reachability/search.py` -> `src/metrics/answer_reachability/search.py`

## 4. 当前仍保留的约束

- 训练主线只支持 `-sub` 数据集。
- eval 默认只允许单 GPU。
- `rank_only` 验证仍然是训练期默认，因为 full search 成本更高。
- prediction artifacts 现在由 runner 末尾统一落盘，不再依赖默认 callback 链路。
- `run.split` 现在会显式传给 datamodule，因此 `run_all_splits` 能真正切换数据源。

## 5. 后续建议

- 继续把 `GFlowNetModule` 拆薄，只保留 Lightning 生命周期钩子。
- 为 `answer/*`、`window/*`、`cert/*`、`edge/*` 保持统一 glossary。
- 新文档应一律引用 `src/metrics/answer_reachability/*`，不要再写历史 `tasks/*` 路径。
