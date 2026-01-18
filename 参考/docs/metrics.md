#### A. 检索质量维度 (Retrieval Quality)
*目标：Context Window 里的“含金量”。*

1.  **Context_Recall@K**
    *   **公式**：$ \frac{|\{\text{Nodes in Path}\} \cap \{\text{Gold Answers}\}|}{|\{\text{Gold Answers}\}|} $
    *   **状态**：**核心指标**。建议与 F1 同步看。
2.  **Context_Precision@K**
    *   **公式**：$ \frac{|\{\text{Nodes in Path}\} \cap \{\text{Gold Answers}\}|}{|\{\text{Nodes in Path}\}|} $
    *   **状态**：**辅助指标**。接受你的定义，保留起始节点。注意：这个数值通常会很低（<0.1），这是正常的，只要它在涨就行。
3.  **Context_F1@K**
    *   **状态**：**综合指标**。Recall 和 Precision 的调和平均。Monitor 选这个（Context_F1@K）。
4.  **Context_Hit@K**
    *   **状态**：**兜底指标**。二值化的 Recall。

#### B. 导航质量维度 (Navigation Quality)
*目标：GFlowNet 的“微操”能力。*

5.  **Terminal_Hit@K**
    *   **定义**：仅看路径终点。
    *   **状态**：**验证指标**。用来证明模型学会了 Stop Action。
6.  **Pass@1**
    *   **定义**：单路径命中率。
    *   **状态**：**效率指标**。

#### C. 系统健康度维度 (System Health)
*目标：训练过程监控。*

7.  **Path_Diversity**
    *   **阈值**：警惕 < 0.1。
8.  **Length_Mean**
    *   **阈值**：警惕 < 1.2 (Shortcut) 或 > Max_Step (迷路)。
9.  **Reward_Gap**
    *   **定义**：Hit Reward - Miss Reward。
    *   **状态**：**Debug 指标**。必须 > 0。

#### D. 逻辑保真度 (Logical Fidelity) —— **The "Kaiming He" Defense**
*目标：证明模型懂逻辑，不是在瞎蒙。由于当前评估受检索子图限制（g_retrieval），本部分仅代表“检索条件下的逻辑保真度”，不是全图语义正确性。建议在 Evaluation/Test 阶段重点计算，训练阶段可选。*

**Coverage Protocol（必须报告）**
*   **SPARQL_Executable_Rate**：SPARQL 解析与谓词映射成功的样本比例。
*   **Subgraph_Answer_Reachable_Rate**：在 g_retrieval 子图内答案可达的样本比例。
*   **Relation_Coverage_Rate**：Gold 谓词在 g_retrieval 关系词表内可映射的比例。
*所有 D 部分指标仅在可覆盖样本上计算，并同时报告覆盖率。*

10. **Semantic Relation Recall (SRR)**
    *   **实现**：
        *   预计算所有 Relations 的 SBERT Embedding。
        *   解析 Gold SPARQL 里的谓词。
        *   计算 Cosine Similarity Soft Match。
    *   **范围**：仅在 g_retrieval 关系词表可映射的样本上评估。
    *   **价值**：这是你论文里 **Qualitative Analysis（定性分析）** 章节的核心数据支撑。
11. **Hub-Node Penalty (HNP)**
    *   **实现**：在 g_retrieval 子图内统计 Top-50 Degree 节点，计算路径经过率。
    *   **范围**：仅在子图可达样本上评估。
    *   **价值**：证明模型没有通过“蹭热点”作弊。
12. **Random Walk Baseline**
    *   **实现**：在 g_retrieval 子图内跑一组随机游走，计算上述所有指标。
    *   **价值**：**Control Group**。你的模型必须显著优于这个基线。
