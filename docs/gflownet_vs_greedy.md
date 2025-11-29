# Greedy Beam Filter vs. GFlowNet：生态位与实验设计

## 1. 角色分工：Filter 与 Agent
- **阶段 1：Filter（Score-aware 2-Hop Beam）**  
  - 目标：高召回，把 4000 边缩到 ~50–100 条候选，别漏真。  
  - 算法：2-hop 路径分波束筛（已在 g_agent 落地），保留潜在正确路径，即使有冗余噪声。
- **阶段 2：Agent（GFlowNet）**  
  - 目标：高精度 + 多样性，在候选子图中精选 5–10 条喂给 LLM，控制 Token 开销。  
  - 算法：2-step GFlowNet 在奖励驱动下做组合优化，抑制 System1 误判、去冗余、补齐模式覆盖。

## 2. Greedy 的三大死穴 & GFlowNet 的修复点
1) **校准误差**：System1 高分≠真实。Greedy盲信分数；GFlowNet 在 Reward 环境中学习抑制假阳性、校准偏差。  
2) **缺乏多样性**：Greedy 只取最高分路径，易重复模式；GFlowNet 拟合分布 \(P \propto R\)，有概率覆盖互补模式，提高答案完整性。  
3) **缺乏全局视野**：Greedy 路径独立排序，不看组合；GFlowNet 的序列决策基于已选上下文，可做组合优化、避免冲突/冗余。

## 3. 实验对比建议
- **Baseline1（System1 Only）**：Retriever Top-K 边 → LLM。预期断路多。  
- **Baseline2（Beam Filter）**：2-Hop Beam 选 Top-K 路径 → LLM。预期召回高，Token 多，含噪声。  
- **Ours（Beam Filter + GFlowNet）**：Beam 先筛 100 边 → GFlowNet 采样/精选 10 边 → LLM。预期同等或更高准确率，Token 显著下降，信噪比更高。

## 4. g_agent 实现状态（已落地）
- 使用 Score-aware 前向 Beam：Hop1 限宽 + Hop2 路径分（log 相加）全局截断 `final_k`，可选注入 GT，输出瘦身子图供 System1.5/2。
- 目的：为 GFlowNet/LLM 提供小而全的候选空间，让 Agent 做“决赛圈精选”，而非“海选”。

## 5. 如何定位 System1.5 的价值
- **效率胜利**：若 GFlowNet 用 ~1/5 Token 达到 Beam Baseline 的 QA 准确率，即证明精选价值。  
  - Beam 做召回，GFlowNet 做高信噪比组合；两者协同，而非互斥。***

## 6. WebQSP：Greedy 基线观测（`/mnt/data/retrieval_dataset/webqsp/materialized/g_agent/test_g_agent.pt`）
- **Key 指标（Top-K）**：Reachability 在 Top-5≈0.892 ➜ Top-50≈0.946，说明主路不难走通；但 **Answer Coverage 从 49.6% 暴跌到 99.8%**（Top-5→Top-50），暴露严重的多样性坍缩。  
- **多答案缺失的根因**：Greedy 按分数填满窗口，重复同一实体/语义模态，导致高精度、低覆盖的模式坍缩。  
- **价值主张**：GFlowNet 在固定 Token 预算（5–10 边）下，通过奖励多样性（如 logdet / Answer Diversity），能把 Coverage 从 ~50–60% 推高到 80%+。这也是 System1.5 的“必要性证明”。  
- **信噪视角**：Top-50 SNR≈0.024，说明 Beam 虽然几乎全覆盖答案，但噪声极大；GFlowNet 需要在低 K 下优化“覆盖/噪声”二元指标。

## 7. CWQ：实验阻塞 & 解决路径
- **阻塞原因**：缺少 g_agent 缓存 ➜ 预期路径 `/mnt/data/retrieval_dataset/cwq/materialized/g_agent/{train,validation,test}_g_agent.pt` 不存在，导致 `scripts/greedy_baseline.py --dataset cwq` 抛 `FileNotFoundError`。  
- **解法（先造 g_agent，再评估）**：  
  1) 生成 g_agent：`PYTHONPATH=. MKL_THREADING_LAYER=GNU MPLCONFIGDIR=/mnt/wangjingxiong/EVI-RAG/.cache/matplotlib python src/eval.py experiment=eval_retriever_cwq g_agent.enabled=true g_agent.output_path=/mnt/data/retrieval_dataset/cwq/materialized/g_agent/{split}_g_agent.pt`  
  2) 然后跑 Greedy：`PYTHONPATH=. MKL_THREADING_LAYER=GNU MPLCONFIGDIR=/mnt/wangjingxiong/EVI-RAG/.cache/matplotlib python scripts/greedy_baseline.py --dataset cwq`  
- **预期对比**（待实测）：Top-5 Reachability 可能掉到 60–70%，Top-50 SNR 可能更低，进一步凸显 GFlowNet 在“连通性 + 多样性”上的增益空间。
