# GFlowNet 策略模块对照表

聚焦策略网络（policy）层，列出已实现的可插拔策略、输入/处理/输出约定，便于快速切换对比。

## 通用接口约定
- 调用签名（`__call__` / `forward`）：
  - `edge_tokens: FloatTensor[B, E, D]`：边级特征，已由 `GFlowNetModule` 投影到隐藏维。
  - `question_tokens: FloatTensor[B, D]`：问题向量，经 `query_projector` 处理。
  - `edge_mask: BoolTensor[B, E]`：哪些位置是真实边。
  - `selected_mask: BoolTensor[B, E]`：已选边标记。
  - `selection_order: LongTensor[B, E]`：已选边的选择步骤（未选为 -1），轻量策略可忽略。
  - 可选前沿信息：
    - `edge_heads: LongTensor[B, E]`, `edge_tails: LongTensor[B, E]`：边端点局部索引。
    - `current_tail: LongTensor[B]`：环境当前游走节点（局部索引），用于前沿判断。
  - 其他可忽略参数（例如 `direction_tokens`）会以 `**_` 接收。
- 返回值：
  - `edge_logits: FloatTensor[B, E]`：每条边的 logit（无 softmax）。
  - `stop_logits: FloatTensor[B]`：每个样本的 stop 动作 logit。
  - `cls_out: FloatTensor[B, D]`：策略内部的全局摘要，用于 `log_z` 条件（可复用 pooled 边特征）。

## 已实现策略

### （已下线）TransformerPolicy / RoleAwarePolicy
- 变长 PyG 扁平 batch 无法直接喂给 TransformerEncoder；原实现需要 [B, E, H] dense 输入，易发生跨图串扰或维度错误。
- 已从代码与导出移除；如需全局注意力，请先实现 varlen/padding 安全的版本，再行启用。

### EdgeMLPMixerPolicy
- **文件**：`src/models/components/gflownet_policies.py`
- **输入特征**：edge_tokens + 类型嵌入（candidate/selected/pad）。
- **处理**：堆叠 MLP + GELU + LayerNorm，逐边独立（O(E)）；stop 头用边特征均值 + question 拼接。
- **输出**：边 logits（线性头）、stop logits（MLP over pooled_edges + question），`cls_out` 使用 pooled_edges。
- **适用场景**：轻量对比基线，关注局部得分，不依赖全局 self-attention。

### EdgeFrontierPolicy
- **文件**：`src/models/components/gflownet_policies.py`
- **输入特征**：edge_tokens + 辅助二元特征（candidate_mask, frontier_mask），前沿基于 `current_tail` 与 `edge_heads/edge_tails`。
- **处理**：LayerNorm + MLP 生成边表示；对前沿边添加 `frontier_bonus`；stop 头同样用均值池化 + question。
- **输出**：边 logits（含前沿加分），stop logits，`cls_out` 为 pooled_edges。
- **适用场景**：2-hop/局部游走，利用“当前节点邻居”先验加速决策。

### EdgeGATPolicy
- **文件**：`src/models/components/gflownet_policies.py`
- **输入特征**：edge_tokens，前沿标记与 Transformer 相同的 mask；可用 `current_tail` 计算前沿加分。
- **处理**：单层「全局 query + 边」注意力：edge_proj(edge) + query_proj(question) → 内积 att，LeakyReLU，softmax over 边；对前沿边加分后 softmax，再取 log 作为 logits。
- **输出**：`edge_logits = log softmax(att)`，stop logits 基于 pooled edge repr + question，`cls_out` 为 pooled_edges。
- **适用场景**：需要轻量注意力、但仍希望跨边归一化的设置；复杂度 O(E)。

## 切换方式
- Hydra 配置 `model.policy_cfg` 指向：
  - `configs/model/policy/transformer.yaml` → TransformerPolicy
  - `configs/model/policy/mixer.yaml` → EdgeMLPMixerPolicy（默认）
  - `configs/model/policy/frontier.yaml` → EdgeFrontierPolicy
  - `configs/model/policy/gat.yaml` → EdgeGATPolicy
- 若需要更短路径，`configs/model/env/graph.yaml` 控制 `max_steps`（默认 3）。

## 环境交互要点
- 所有策略均假定动作空间为「选边或 stop」，stop_index = num_edges。
- 前沿类策略需从 batch 传入 `edge_heads/edge_tails/current_tail`；`GFlowNetModule` 已统一传递。
- 选边 logits 会叠加 retriever 分数偏置（`policy_edge_score_bias`）后与 stop logit 拼接，再经过 softmax/探索混合。***
