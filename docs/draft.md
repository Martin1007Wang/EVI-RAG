我看了一遍当前训练主路径，这个 OOM 更像“真实峰值过高”，不太像单纯碎片化。
- 你现在训练配置把边数上限放到了 configs/experiment/train_answer_reachability.yaml:28 的 1920000，节点上限是 configs/experiment/train_answer_reachability.yaml:27 的 640000
- 模型主干隐藏维度是 configs/model/gflownet.yaml:68 的 512，而且有 configs/model/gflownet.yaml:69 的 2 层 GNN
- trainer 虽然是 bf16-mixed，见 configs/experiment/train_answer_reachability.yaml:44，但热路径里很多大张量又被强制转回了 float32
最可能的爆点
- src/models/components/gnn.py:52、src/models/components/gnn.py:68、src/models/components/gnn.py:69、src/models/components/gnn.py:96、src/models/components/gnn.py:157
- 这里的 PNA GNN 会连续物化多个大张量：messages、mean_gathered、diff_sq、stats、update_in
- 只要有一个 [E, 512] 的 float32 张量，内存就是 E * 512 * 4 bytes；当 E 在 140 万左右时，单个张量就接近你报错里的 2.81 GiB
- 这意味着：你这个 OOM 很可能就是在 GNN 统计量阶段被某一个 edge-level buffer 直接打满的
第二大头
- src/models/gflownet/policy.py:858、src/models/gflownet/policy.py:859、src/models/gflownet/policy.py:983、src/models/components/scoring.py:179、src/models/components/scoring.py:184
- forward candidate scoring 虽然有 chunk，但 relation_features = index_select(...) 是在 chunk 之前对“全部候选边”一次性 gather 的
- 这会让 chunking 只保护 MLP 前向，不保护最贵的候选关系特征物化
- 如果 rollout 阶段活跃状态多、平均出度高，这里同样会一次性造出一个接近几 GiB 的 [num_candidates, 512] 张量
系统性低效点
- src/models/components/gnn.py:52、src/models/components/gnn.py:157、src/models/gflownet/policy.py:602、src/models/gflownet/policy.py:618、src/models/components/scoring.py:180、src/models/components/scoring.py:186
- 这些地方把大激活显式转成 float32，会明显削弱 bf16-mixed 的省显存效果；稳定性是有了，但代价很高
- src/models/gflownet_module.py:1087、src/models/gflownet_module.py:1090、src/models/gflownet/policy.py:237、src/models/components/embedding.py:140、src/models/components/embedding.py:146
- training_step 里先 prepare_batch() 再 without_raw_features()，所以编码峰值阶段会同时保留原始 1024 维输入特征和编码后的图/问题张量，峰值偏高
- src/models/gflownet/policy.py:619、src/models/gflownet/policy.py:631
- question-context attention 会按图构造 [states_in_graph, context_len] 的注意力矩阵；单次不一定最大，但 rollout/search 里会反复发生
- src/metrics/search_backends.py:373、src/metrics/search_backends.py:385、src/metrics/search_backends.py:408
- eval 的 flow-frontier 会 clone 路径并保留 control states，前沿一大时显存会持续堆高
当前不是主因，但有潜在风险
- src/models/gflownet/losses.py:240、src/models/gflownet/losses.py:242、src/models/gflownet/losses.py:259
- MS-SubTB 的 pairwise residual 会造 [B, R, T, T] 级别张量；现在 T 小时问题不大，但如果以后 hop 数上去，这里会变成二次爆炸点
怎么判断是哪一块先炸
- 如果栈顶在 src/models/components/gnn.py，基本就是 GNN 边消息统计在炸
- 如果栈顶在 src/models/gflownet/policy.py 或 src/models/components/scoring.py，基本就是 forward candidate scoring 在炸
- 你这次只给了 OOM 摘要，没有完整栈；但从 2.81 GiB 这个量级看，前两者都非常像
最值得先动的优化
- 1. 先改 src/models/gflownet/policy.py:858 这条路径，把 relation_features 的 gather 挪进 chunk 循环，让 chunking 真正覆盖候选特征物化
- 2. 再改 src/models/components/gnn.py:52 这条路径，避免整批 edge message 全量转 float32；至少把 PNA 统计做成 edge-chunk/streaming
- 3. 立刻下调 configs/experiment/train_answer_reachability.yaml:24 的 batch_size，以及 configs/experiment/train_answer_reachability.yaml:27、configs/experiment/train_answer_reachability.yaml:28 的 node/edge 上限
- 4. 审查热路径里的 .to(dtype=torch.float32)，只把真正需要高精度的归约/softmax 留在 fp32
- 5. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 可以试，但它更像缓解碎片；你这里 45GB 已经被实打实占了，大概率不是根治手段
如果你要，我可以直接继续做两步高收益改动：
1. 先把 candidate scoring 改成“真正按 chunk gather”
2. 再把 GNN 的 PNA 聚合改成分块统计，优先处理最可能的 2.81 GiB 申请点