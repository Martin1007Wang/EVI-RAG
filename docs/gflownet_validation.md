# GFlowNet 终极审计清单 (Deep Audit Checklist)

**审计目标**：确保代码不仅跑得通（No Bugs），而且在数学上是对的（Correct Distribution）。拒绝“Loss下降就是好”的幻想。

**实现现状提示**：当前代码基线已移除行为克隆（BC）分支，训练目标仅包含 Online TB 与 GT 轨迹的 TB 式重放；文档中涉及 “Offline/BC” 的检查应理解为“Offline 轨迹合法性校验/重放”。

---

### 第一维度：理论架构与定义 (Theoretical Integrity)
**[BLOCKER] 只有这一层通过，模型才有训练的意义。**

1.  **$Z$ 的条件独立性检查 (Conditional Z Check)**
    *   **审查点**：检查 `src/models/gflow_net.py` 中 `logZ` 的定义。
    *   **判定标准**：如果输入数据包含 condition (如 `start_entity_ids` 或 prompt)，`logZ` **必须**是一个接受该 condition 的神经网络模块 (MLP)，绝不能是单一的 `nn.Parameter` 标量。
    *   **动作**：若为标量且任务为 conditional，立即重构为 `self.logZ_model(condition)`。

2.  **$P_B$ 定义的统一性 (Unified Backward Policy)**
    *   **审查点**：对比 Loss 计算中的 `log_pb` 项与代码注释/文档。
    *   **判定标准**：
        *   若 $P_B$ 固定（Uniform）：必须显式计算父节点数量 `1/num_parents`，严禁隐式假设 `log_pb=0` 除非是 Tree 结构（每个节点仅1个父节点）。
        *   若 $P_B$ 学习（Learned）：必须有独立的 `backward_policy` 网络，且保证参数与 $P_F$ 解耦（或共享躯干但独立Head）。
    *   **动作**：删除所有“假定 PB=1”的误导性注释，代码中必须显式体现 $P_B$ 的计算逻辑。

3.  **流守恒公式闭环 (TB Exactness)**
    *   **审查点**：TB Loss 实现代码。
    *   **判定标准**：严格对应公式 `loss = (log_z + sum_log_pf - sum_log_pb - log_reward)**2`。确保 `log_reward` 的正负号没有写反。

---

### 第二维度：逻辑与数据一致性 (Implementation Logic)
**[CRITICAL] 消除“训练数据”与“模型能力”之间的断裂。**

4.  **前向与回放的连通性对齐 (Mask Consistency)**
    *   **审查点**：对比 `get_mask_forward` (Online) 和 `validate_trajectory` (Offline)。
    *   **判定标准**：
        *   **双向行走**：如果前向允许双向，Teacher 轨迹中的双向移动不应被 mask 掉。
        *   **Stop Action**：Teacher 轨迹中的 Stop 动作必须在当前策略的 `action_space` 中是合法的。
    *   **动作**：编写单测，输入同一状态，断言 Online Mask 与 Offline Verification Mask 完全一致。

5.  **Off-Policy 合法性校验 (Teacher Compatibility)**
    *   **审查点**：Replay Buffer 中取出的轨迹在当前 $P_B$ 下的概率。
    *   **判定标准**：计算 Buffer 中 Teacher 轨迹的 $\sum \log P_B$。如果出现 `-inf`，说明 Teacher 走了当前模型认为“不可逆”的路，这将导致梯度爆炸。
    *   **动作**：在 Loss 计算前增加断言或过滤：丢弃那些在当前拓扑下无法回溯的 Teacher 轨迹。

6.  **Stop 动作的采样权 (Termination Sampling)**
    *   **审查点**：`_sample_uniform_actions` 或 $\epsilon$-greedy 逻辑。
    *   **判定标准**：随机探索时，Stop 动作必须有概率被选中。如果只在 Edge 中随机，模型将无法学会正确的长度分布。
    *   **动作**：将 Stop 动作纳入 Uniform 分布的采样池。

7.  **死参数与硬编码清洗 (Parameter Hygiene)**
    *   **审查点**：`reward_cfg`, `semantic_prior`, `path_reward_term`。
    *   **动作**：
        *   删除未使用的 `beta_score`。
        *   将 `+0.1`, `*0.5` 等 Magic Number 全部移入 Hydra Config。
        *   确认 `success_bonus=0` 是否符合预期，否则给个默认值。

---

### 第三维度：优化动力学 (Optimization Dynamics)
**[High Priority] 确保优化器是在“爬山”而不是“乱撞”。**

8.  **Z 与 Actor 的速率解耦 (Z-LR Decoupling)**
    *   **审查点**：Optimizer 的参数组配置。
    *   **判定标准**：`logZ` (或 Z-Model) 的学习率必须独立配置，且建议 `lr_z >= 10 * lr_actor`。
    *   **动作**：在 Config 中分离 `lr_z`，代码中区分 parameter groups。

9.  **数值稳定性防御 (Numerical Stability)**
    *   **审查点**：Masking 操作和 Log 操作。
    *   **动作**：
        *   Mask 填充值：从 `finfo.min` 改为 `-1e9` (防止 AMP 下 NaN)。
        *   Log 安全网：所有 `torch.log(p)` 必须改为 `torch.log(p + 1e-9)` 或使用 `logits.log_softmax`。

---

### 第四维度：分布验证实验 (Empirical Verification)
**[GOLD STANDARD] 只有通过这些测试，才算“验证完成”。**

10. **Toy DAG 真实分布拟合 (Log-Log Plot)**
    *   **实验设计**：在 $4 \times 4$ Grid World 或小型 Graph 上训练。
    *   **指标**：采样 10k 条轨迹，计算经验概率 $\hat{P}(x)$。
    *   **验收标准**：Plot $\log \hat{P}(x)$ vs $\log R(x)$。Pearson Correlation $r > 0.95$，回归线斜率 $k \in [0.9, 1.1]$。

11. **条件 $Z$ 差异性测试 (Conditional Z Test)**
    *   **实验设计**：构造两个起点 $S_1$ (高总奖励), $S_2$ (低总奖励)。
    *   **验收标准**：$\log Z(S_1) - \log Z(S_2) \approx \log(\sum R_{S_1}) - \log(\sum R_{S_2})$。如果两者相等，测试失败。

12. **梯度范数比率 (Grad Norm Ratio)**
    *   **实验设计**：训练前 1000 steps。
    *   **指标**：$\|\nabla_{\log Z}\| / \|\nabla_{\theta}\|$。
    *   **验收标准**：比率不应持续趋近于 0。$\log Z$ 必须有显著的梯度流。

---

### 交付给 Codex 的指令 (Prompt for Codex)

你可以直接复制下面的 Prompt 给 Codex，让它执行具体的代码审计和测试编写：

```markdown
Role: You are a Senior Deep Learning Engineer and GFlowNet Specialist.
Task: Audit an existing GFlowNet codebase and implement verification tests based on a rigorous checklist.

I have defined a 4-tier audit checklist covering Theoretical Integrity, Implementation Logic, Optimization, and Empirical Verification. 

Please perform the following actions step-by-step:

Phase 1: Static Code Analysis (Fix Logic Errors)
1. Check `src/models/gflow_net.py`. Does `logZ` handle conditional inputs (e.g., `start_entity_ids`)? If it's a global scalar but the task is conditional, flag this as a CRITICAL ERROR and suggest a refactor to a Neural Network Z-estimator.
2. Unify the definition of `P_B` (Backward Policy). Identify if code assumes Uniform P_B but calculates it incorrectly. Ensure `log_pb` logic in TB Loss matches the graph topology (handling parents count).
3. Audit Masking: Check `get_mask_forward` vs `validate_trajectory`. Ensure Teacher trajectories in the Replay Buffer are not deemed 'invalid' by the current mask logic.
4. Clean up: Identify hardcoded rewards (like +0.1) and unused params in `reward_cfg`. Suggest moving them to Hydra config.

Phase 2: Optimization Configuration
1. Modify the Optimizer setup to allow different Learning Rates for `logZ` (or Z-model) and the rest of the network. Suggest a default `lr_z = 10 * lr_actor`.
2. Replace any `finfo.min` in masking with a safe constant (e.g., -1e9) to prevent NaN in AMP.

Phase 3: Test Implementation (Write these tests)
1. Write a `test_toy_log_linear_fit.py`:
   - Create a micro-GridWorld (3x3).
   - Train the GFlowNet.
   - Sample 5000 trajectories.
   - Plot log(Empirical_Prob) vs log(Reward) and assert Pearson correlation > 0.95.
2. Write a `test_conditional_z.py`:
   - Define two contexts with known ground-truth total rewards R_total_A = 100, R_total_B = 1.
   - Train model.
   - Assert `abs((model.logZ(A) - model.logZ(B)) - log(100))` < threshold.

Please start by analyzing the code based on Phase 1.
```

---

## Greedy Baseline（retriever 下界）

- 脚本：`python scripts/greedy_baseline.py --dataset webqsp --split test --max_steps 2`
- 数据源：`/mnt/data/retrieval_dataset/webqsp/materialized/g_agent/test_g_agent.pt`
- 过滤：缺答案或答案不在图内的样本跳过；无起点的样本跳过。
- 结果（webqsp/test，max_steps=2）：
  - Used (reachable): 1557 / total_seen: 1628，reachable_ratio: 0.9564，skipped_missing_answer: 71，skipped_no_start: 0
  - Answer Precision (mean): 0.838150
  - Answer Recall (mean): 0.623818
  - Answer F1 (mean): 0.617446
  - Success rate (reachable): 0.989724；Success rate (overall incl. unreachable): 0.946560
  - Avg steps used: 1.3102（max_steps=2）；Avg selected edges: 1.3102
  
若后续模型的 test Answer Recall/F1 低于此贪心下界，应停止训练，先排查数据/策略/奖励。
