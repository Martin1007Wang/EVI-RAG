 这是一次静态代码审查，没跑 profiler，所以结论是“高置信热点定位”，不是精确显存曲线。

  核心结论

  当前显存和计算开销的最大来源不是模型参数，而是 rollout 期间把图、候选边、状态读出反复放大。默认配置下 batch_size=8、train_num_rollout=8、hidden_dim=1024、precision=32-true，实际训练每步接近 64 个物理图的 rollout 规模。

  显存开销

  1. 默认路径会物理复制整批图。

     src/weaver/rollout/engine.py:239 在 use_static_batch_rollouts=false 时调用 repeat_retrieval_batch(...)，而 src/data/schema/repeat.py:45 会复制 edge_index、batch、edge_batch、节点字段、边字段、flat labels、question embedding 等。

     这意味着静态图数据和 FeatureBank 都随 train_num_rollout=8 放大。FeatureBank 至少包括 node_h/rel_h/query_h 和 node_sem_h/rel_sem_h/query_sem_h，默认都是 1024 维 float32。粗略估算，单个节点或边仅两套表示就约 2 * 1024 * 4 = 8KB，再乘 rollout 数和 batch 图规模，显存很容易被静态重复吞掉。
  2. doob_value_prior 的 successor value 计算非常重。

     src/weaver/policy.py:373 对每条候选边构造 successor state，然后 src/weaver/policy.py:379 又跑一次 StateReadout + FlowHead。这把每步开销从“每个 rollout state 一次读出”扩展到“每条 candidate edge 一次 successor 读出”。

     如果 frontier 候选数大，显存峰值会跟候选边总数近似线性增长。
  3. fused/static rollout 已经有雏形，但默认没启用。

     configs/model/weaver.yaml 里 use_static_batch_rollouts=false、use_fused_static_batch_rollouts=false。建议优先评估 fused static path，因为它避免复制静态 batch 和 FeatureBank。不过当前 fused path 仍用二维状态 mask，见 src/weaver/nn/state_readout.py:544，会生成 [num_rollouts, num_edges]
     的 belongs/frontier_mask，大图上仍可能很占显存。

  Python 计算开销

  1. 奖励函数有明显 Python 循环和 GPU 同步点。

     src/weaver/reward.py:198 对每个 rollout 行循环；src/weaver/reward.py:446 对每个 graph 循环；连通性用 Python set/list。默认 doob.stop_mode=reward 且 stop_tb_coef=1.0 会让训练每步 eager 计算 stop reward，这部分会拖慢 GPU 利用率。
  2. 分段 log-softmax 是 Python 按 segment 循环。

     src/graph/segments.py:67 对 range(num_segments) 循环并多次做 boolean indexing。这里应改成 torch_scatter.scatter_logsumexp 或等价 scatter 版本。
  3. 采样 expand edge 逐 graph 循环且 .item()。

     src/weaver/rollout/sampling.py:319 每个 expanding graph 单独筛 candidate、构造 Categorical、.item()。batch/rollout 增大时会成为 CPU 调度瓶颈。已有 sample_segmented_positions 的 Gumbel + scatter_max 思路，可以统一替换。

  实现不简洁/逻辑不清晰

  1. rollout 有三套路径：physical repeat、static batch、fused static batch。逻辑分叉较多，且默认仍走最浪费的 physical repeat。建议收敛到 fused/static 为主路径，physical repeat 仅作为测试兼容路径。
  2. State 和 RolloutState 双实现导致很多函数同时处理 1D/2D mask，复杂度高。当前语义清楚，但实现负担大，后续优化时容易引入不一致。
  3. reward 里“诊断指标”和“训练 reward”混在同一次计算中。answer_f1、answer_degree_excess 等诊断字段会被每次 stop reward 计算携带，训练热路径可以拆成轻量 reward 与可选 diagnostics。

  优先优化建议

  1. 先启用并验证 use_fused_static_batch_rollouts=true，比较显存、速度、结果一致性。
  2. 把 scatter_log_softmax 改成真正 scatter 实现。
  3. 向量化或缓存 reward 连通性，至少把 eager stop reward 的 Python BFS 从训练热路径移出。
  4. 避免 doob_value_prior 对全 frontier 做 successor readout：可以先做 top-k successor value、分块 successor readout，或训练时近似、评估时全量。
  5. 如果 GPU 显存紧张，优先试 precision: bf16-mixed 或 16-mixed，再调小 train_chunk_size；但结构性收益仍来自去掉物理 batch repeat。