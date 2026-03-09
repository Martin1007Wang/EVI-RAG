1. 先把状态从“节点状态”改成“有限时域状态”
- 正确语义是 s_t = (u_t, t, q, G)，所以同一个节点在不同步数下，logF(s_t) 和 P_F(.|s_t) 必须允许不同。
- 现在的问题在 src/models/trajectory_gfn/policy.py:212、src/models/trajectory_gfn/policy.py:226、src/models/trajectory_gfn/encoder.py:99：logF 和动作分布只依赖当前节点/问题，不依赖 t。
- 最直接的改法是：把 node_log_f 从 encoder 预计算里拿掉，改成“state-conditioned 现算”。
- 具体做法：
  1. 在 src/models/trajectory_gfn/policy.py 里新增 step_embedding 和 remaining_embedding，大小都设成 max_steps + 1。
  2. 新增一个统一函数，比如 build_state_features(context, state)，输出每个 agent 的状态表示。
  3. 这个状态表示至少应是：
          state_repr = LN(
       node_repr(u_t)
       + step_emb(t)
       + remain_emb(H - t)
     )
       4. compute_log_flow() 改成基于 state_repr 算，而不是从 context.node_log_f 里 index_select。
  5. compute_forward_distribution() 里的 agent_history 也改成 state_repr，不要再直接拿 context.node_tokens[current]。
  6. compute_backward_distribution() 同样用 next state 的 state_repr，否则 DB 两边不是同一个状态定义。
- 你现在 EdgeScoreModule 里还有 Doob-h 项，见 src/models/policy/edge.py:323。这个也要一起改：它现在加的是 node_log_f(target_node)，但正确的应该是 logF(s_{t+1})，也就是“目标节点 + 下一时刻”的 flow。
- 所以 Doob-h 最好不要在 src/models/policy/edge.py 里直接吃 node_log_f 向量了，而应该在 src/models/trajectory_gfn/policy.py 里先算出每条边对应的 next_state_log_f，再把它加回 edge logits。
- 最小改动路径是：
  - src/models/trajectory_gfn/encoder.py：不再缓存 node_log_f
  - src/models/trajectory_gfn/policy.py：新增时间嵌入 + state repr + state flow
  - src/models/policy/edge.py：去掉对 node_log_f 的直接依赖，改为接收外部传入的 edge_next_log_f
- 要补的测试：
  - tests/trajectory_gfn/test_state_time_conditioning.py
  - 构造同一个节点 u，分别在 t=0 和 t=H 调 compute_log_flow()，断言二者不必相等
  - 同时断言 forward logits 也允许不同
2. 把 min_stop_steps 变成真正的硬支持约束
- 现在 src/models/trajectory_gfn/transition.py:45 的逻辑是：只有“有边可走”时才 ban STOP。
- 这会导致“在 min_stop_steps 之前走到 dead-end 时，STOP 反而变合法”，这是错的。
- 正确语义应该二选一，但你必须选一个并全仓统一：
  1. 严格版：t < min_stop_steps 时 STOP 永远非法；若此时又没有 move，则该状态 support 为空，直接判 invalid。
  2. 宽松版：允许 dead-end 早停，但要把文档、训练、评估、analyzer 全部改成这个定义。
- 结合你前面的目标，我建议用严格版。
- 具体怎么改：
  1. 在 src/models/trajectory_gfn/transition.py 里把 stop mask 改成：
          ban_stop = active_flat & (~force_stop) & (num_moves_flat < min_stop_steps)
          不要再依赖 out_degrees > 0 和 has_finite_edges。
  2. 再额外构造：
          invalid_rows = active_flat & (~force_stop) & (num_moves_flat < min_stop_steps) & (~has_finite_edges)
       3. 把 invalid_rows 挂到 ForwardActionDistribution 里，或者让 apply_forward_constraints() 返回 (distribution, invalid_rows)。
  4. 在 src/models/trajectory_gfn/sampler.py 里，采样前如果出现 invalid_rows.any()：
     - 训练阶段建议直接 raise ValueError
     - 不要偷偷让 invalid_logits_policy=stop
  5. 在 src/models/trajectory_gfn/analyzer.py 和 src/models/trajectory_gfn/search.py 里也一样，如果遇到这种状态，要么报错，要么在更早的数据层过滤掉。
- 更进一步，我建议把这个约束前移到数据层：
  - 在 src/datasets/g_retrieval_dataset.py 或预处理阶段增加一个可达性过滤：
    “每个 start 至少存在一条长度属于 [min_stop_steps, max_steps] 的合法终止轨迹”
- 要补的测试：
  - tests/trajectory_gfn/test_min_stop_steps_hard_constraint.py
  - toy graph: q=0，0 无出边，min_stop_steps=1
  - 预期：sampler / analyzer / search 都不能把它当作合法 STOP 终止
3. 重写 search：不能再用“发现的 completed mass 达到阈值就停”
- 你现在 src/models/trajectory_gfn/search.py:158 的停机条件不正确。
- 根本原因是：你当前 heap 里存的是前缀，但 completed 里的叶子只是在“某个前缀被展开时顺手发现”的，不是按完整轨迹概率全局有序产生的。
- 正确做法是：不要分成“frontier heap + completed list”两个世界，而要用“统一候选堆”。
- 统一候选堆里有两类对象：
  1. prefix candidate：优先级 = log p(prefix)，这是该子树所有后代的上界
  2. terminal candidate：优先级 = log p(trajectory)，这是精确叶子概率
- 算法应该是：
    push all start prefixes
  emitted = []
  emitted_mass = 0
  while emitted_mass < rho:
      cand = pop_max()
      if cand is terminal:
          emit cand
          emitted_mass += cand.prob
          continue
      # cand is prefix
      expand prefix
      if STOP legal:
          push terminal(prefix + STOP)
      for each move:
          push child_prefix
  - 这样一个 terminal 被 pop 出来时，才可以证明它是“当前全局剩余最高概率的完整轨迹”，因为所有未展开前缀的上界都不超过它。
- 这才是 exact W_rho 需要的顺序。
- 你现在 src/models/trajectory_gfn/search.py:190 是把 STOP completion 直接塞进 completed，这一步要改成“塞回堆里作为 terminal candidate”。
- 你现在 src/models/trajectory_gfn/search.py:214 也不能再在最后统一 sort(completed) 了，因为 emitted 顺序本身就应该已经是正确顺序。
- 建议你把 search.py 里的私有结构改成：
  - _PrefixCandidate
  - _TerminalCandidate
  - _SearchCandidate = Union[...]
- 要补的关键反例测试：
  - tests/trajectory_gfn/test_search_exact_top_order.py
  - 反例建议：
    - start -> A 概率 0.6
    - A -> STOP 概率 0.1，叶子概率 0.06
    - A -> C 概率 0.9，C -> STOP=1.0，叶子概率 0.54
    - start -> B 概率 0.4，B -> STOP=1.0，叶子概率 0.4
  - 正确 top terminal 顺序应是 0.54, 0.4, 0.06
  - 你当前实现很容易先把 0.06 放进窗口，这是错的
4. ElasticWindowResult 必须只包含“刚好达到阈值”的最小窗口
- 现在 src/models/trajectory_gfn/search.py:77 到 src/models/trajectory_gfn/search.py:124 的 _finalize_results() 会把所有已发现轨迹都放进去。
- 这不符合你自己定义的最小窗口：
    W_rho = 概率降序下，累计质量首次达到 rho 的最短前缀
  - 正确改法是：
  1. search 主循环里维护 emitted_trajectories
  2. 只有当 terminal candidate 被正式 emit 时，才进入结果列表
  3. 一旦 emitted_mass >= rho，立刻停
  4. 最终 ElasticWindowResult.trajectories = emitted_trajectories
  5. covered_mass = emitted_mass
  6. window_size = len(emitted_trajectories)
  7. covered_gold_mass 只对 emitted 集合求和
  8. missed_gold_mass = gold_total_mass - covered_gold_mass
- 这样 src/models/trajectory_gfn/metrics.py 里的 diversity、window size、elastic mass 才有意义。
- 现在 src/models/trajectory_gfn/search.py:102 的 covered_mass = cumulative_mass 本身没问题，问题是这个 cumulative_mass 不是“最小窗口”的，而是“所有已发现轨迹”的。
- 修好 search 后，这里自然就对了。
- 要补的测试：
  - tests/trajectory_gfn/test_window_is_minimal_prefix.py
  - 对同一个 toy distribution，手工列出 leaf probs，验证返回结果的最后一条恰好让累计质量第一次越过 rho
5. exact mode 和 approximate mode 必须分开，不要混在一个结果对象里装作一样
- 现在 src/models/trajectory_gfn/search.py:39 会 trim frontier，src/models/trajectory_gfn/search.py:157 也会因为 max_expansions 提前退出。
- 一旦发生这两件事中的任何一个，你的结果就不再是 exact elastic window。
- 最简单、最推荐的 v1 修法是：
  - exact search 不允许 trim
  - 命中 max_expansions 或 max_frontier_size 直接报错，不返回结果
- 也就是：
  - 删掉 _trim_frontier()
  - max_frontier_size 改成 guard，不是 pruning 机制
  - max_expansions 改成 fail-fast guard，不是近似停机条件
- 如果你一定要保留近似版，那就必须升级 src/models/trajectory_gfn/schema.py：
  - is_exact: bool
  - stop_reason: str
  - remaining_frontier_mass: float
  - pruned_prefix_mass: float
  - mass_lower_bound: float
  - mass_upper_bound: float
- 然后 tail_rollout_mass 就不能再简单写成 1 - covered_mass，因为你已经丢掉了一部分前缀。
- 对 v1，我强烈建议先走 exact-only：
  - 这样你整个“概率质量窗口”的论文叙事才是干净的
  - 等 exact 版跑稳后，再单独加 approximate search
- 要补的测试：
  - tests/trajectory_gfn/test_search_raises_on_truncation.py
  - 当 frontier 超上限或 expansions 超上限时，断言 search 直接报错，而不是返回一个看起来正常的 ElasticWindowResult
建议的具体改动顺序
1. 先改 src/models/trajectory_gfn/policy.py，把时间条件化打进去。
2. 再改 src/models/trajectory_gfn/transition.py 和 src/models/trajectory_gfn/sampler.py，把 min_stop_steps 变成硬支持约束。
3. 再重写 src/models/trajectory_gfn/search.py，换成统一候选堆的 exact enumeration。
4. 再改 src/models/trajectory_gfn/schema.py 和 src/models/trajectory_gfn/metrics.py，让结果对象只反映最小窗口。
5. 最后补测试，把这 4 类语义锁死。
我建议你优先改的文件
- src/models/trajectory_gfn/policy.py
- src/models/trajectory_gfn/encoder.py
- src/models/trajectory_gfn/transition.py
- src/models/trajectory_gfn/sampler.py
- src/models/trajectory_gfn/search.py
- src/models/trajectory_gfn/schema.py
一句话总结
- 先把“状态定义”修对，
- 再把“轨迹支持集”修对，
- 最后把“窗口搜索顺序”修对。