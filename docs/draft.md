• 理解了：你们的覆盖表来自“未清洗”的原始数据，当前 pipeline 因为清洗 common.*/属性关系而导致覆盖下降，这点是合理的。既然训练强制在 sub 上过滤，那我更关注“算法层面如何提升判别性”。下面给出两部分：方向性策略 + 判别性增强。

  方向性策略（一定要一致）

  - 推荐：有向 + 显式反向边（r / r⁻¹）
    这是 KG 任务最稳的做法：保持关系语义，同时让 q→a 和 a→q 都可达。
    关键是 全链路一致：构图、sub_filter、distance cache、环境动作空间都基于同一“扩展图”。
    这样可以明显缓解“因方向导致不可达”的问题，又不会像无向图那样丢失关系语义。
  - 不建议纯无向
    无向会把关系语义完全抹平，判别性通常更差（尤其多跳 CWQ）。
  - 如果坚持纯有向
    那么 sub_filter 必须用同一方向（你们现在是 qa_directed），并接受 coverage 损失；这条路线会把问题完全推向“判别性”。

  判别性增强（算法层面）

  1. 关系选择应该用关系语义
     目前 relation logits 只用 state MLP，没有用 relation embedding。
     建议让 relation logits = f(state)·g(relation_emb)（矩阵乘法即可，完全向量化），这样关系选择具备语义判别力。
     对应位置：src/models/components/gflownet_actor.py、src/models/gflownet_module.py（relation head 设计）。
  2. 把“硬裁剪 corridor”改成“软偏置”
     现在 training_cfg.corridor.enabled=true 会把边硬裁剪到“能到答案”的子空间，判别性训练信号被削弱。
     建议保留 distance_bias_alpha，但关闭硬 mask，只用软引导（仍是 label‑free）。
     对应位置：src/gfn/ops.py（edge_corridor_mask vs edge_distance_bias）。
  3. 利用 node_type_ 做结构嵌入（你们现在完全没用）*
     node_type_counts/node_type_ids 已经落盘，但模型侧没有用。
     用 type‑bag + type embedding 做 DDE/结构特征，可以显著提升区分能力。
     对应位置：src/models/gflownet_module.py（backbone）。
  4. 用“距离进展”替代 shortest‑path 监督
     你们已经有 node_min_dists，但目前只用在 corridor / distance prior。
     可以加一个进展项：奖励“距离减少”的动作（soft shaping），而不是 path label。
     对应位置：src/models/components/gflownet_reward.py（注意当前 reward 禁止 potential shaping，需要解禁或新 reward）。
  5. 把 question‑node 相关性显式注入到 policy
     已有 cosine_bias_alpha 和 cosine_relation_bias_alpha，默认 0。
     这是“零标注”提升判别性的最低成本方式之一；在多跳数据上通常很有效。
     对应位置：configs/experiment/train_gflownet.yaml / configs/model/gflownet_module.yaml。
  6. max_steps 对判别性影响很大
     多跳数据如果 max_steps 小于真实距离分布，会直接把正确轨迹截断，loss 会被拉高。
     建议用 node_min_dists 统计实际分布，再决定 dataset.max_steps。
     对应位置：configs/dataset/*.yaml。

  如果你要我给出“最小有效修改集合”
  我建议顺序是：

  1. 有向 + 反向边（统一构图/距离/环境）
  2. 关系语义注入 relation logits
  3. corridor 由硬裁剪改软偏置
  4. 引入 type 结构嵌入
  5. 适当增加 cosine_bias_alpha

  我需要你确认两点才能给更精确的建议

  1. 你们现在 eval 阶段是否也启用了 corridor（training_cfg.corridor 在 eval 也生效）？
  2. 你更倾向于“保留关系方向语义 + 加反向边”，还是“直接无向化”？

  给我这两个信息，我可以把方案收敛成一个最小改动版本，并标注涉及的具体文件/参数。