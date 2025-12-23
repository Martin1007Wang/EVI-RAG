假设：
GT Path: $[(e_{start}, r_1, e_1), (e_1, r_2, e_{ans})]$
Rollout Path: $[(e_{start}, \hat{r}_1, \hat{e}_1), (\hat{e}_1, \hat{r}_2, \hat{e}_{ans})]$

Step 1: 对齐检查 (Alignment Check)
遍历 GT Path 中的每一个三元组（Step），检查 Rollout 是否匹配。
def calculate_process_acc(gt_path, rollout_path):
    # gt_path: list of (head, relation, tail)
    # rollout_path: list of (head, relation, tail)
    
    score = 0.0
    total_steps = len(gt_path)
    
    # 你的场景通常是序列决策，所以可以直接按顺序比对
    # 如果 rollout 长度不足，直接截断或补零
    max_len = min(len(gt_path), len(rollout_path))
    
    for i in range(max_len):
        gt_triple = gt_path[i]
        rollout_triple = rollout_path[i]
        
        # 1. 检查关系是否一致 (对应论文中的 dependency check)
        relation_match = (gt_triple.relation == rollout_triple.relation)
        
        # 2. 检查尾实体是否一致 (对应论文中的 value check)
        entity_match = (gt_triple.tail == rollout_triple.tail)
        
        if relation_match and entity_match:
            score += 1.0
        else:
            # 在序列模型中，通常如果这一步错了，后续大概率都错了
            # 论文中是 sum，你可以选择是否在这里 break
            # 为了提供 dense reward，建议继续计算，或者给予部分分数
            pass 
            
    return score / total_steps

Step 2: 整合进奖励函数
正如论文第 6 节提到的，将这个分数作为密集奖励（Dense Reward）的一部分：
$$
R(s_t, a_t) = \underbrace{\mathbb{I}(s_T = \text{Answer})}_{\text{稀疏结果奖励}} + \lambda \cdot \underbrace{\text{ProcessAcc}(s_t)}_{\text{密集过程奖励}}
$$

3. 一个关键的风险提示 (对于 GFlowNet)
在使用这个算法时，不要只比对一条 GT Path。
Any-Match 策略：如果你的数据集标注了多条推理路径，计算 Rollout 与所有 GT Paths 的 ProcessAcc，取最大值。