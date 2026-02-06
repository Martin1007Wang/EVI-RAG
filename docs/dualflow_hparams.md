# DualFlow 超参与魔法数字说明

本文档覆盖两类内容：
1) **可配置超参**（来自 `configs/model/dual_flow.yaml`，以及少量与模型相关的 dataset 配置）  
2) **代码内魔法数字/常量**（来自 `src/models/dual_flow_constants.py` 与组件实现）

> 说明：本文以当前配置文件为准（你已做过部分修改），并标注代码默认值与实际值的区别。

---

## 1. 模型配置超参（`configs/model/dual_flow.yaml`）

### 1.1 模型结构
- `hidden_dim`  
  - 含义：图节点/关系/问题投影后的隐空间维度。  
  - 影响：参数规模、双线性策略头与 GNN 的容量与算力消耗。
- `emb_dim`  
  - 含义：输入嵌入维度（节点/关系/问题的预计算向量维度）。  
  - 影响：投影层输入维度，与数据侧 embedding 维度一致。
- `gnn_layers`  
  - 含义：RelationalGNNLayer 层数。  
  - 影响：图消息传播深度（越深越慢、越易过平滑）。
- `gnn_dropout`  
  - 含义：GNN 层的 Dropout 概率。  
  - 影响：正则化强度。
- CVT 节点初始化固定开启（用入边邻居均值替代 CVT 初值）。
- EmbeddingAdapter 固定开启（低秩残差适配）。
- `embedding_adapter_cfg.adapter_dim`  
  - 含义：Adapter 中间瓶颈维度。  
  - 影响：适配容量与参数量。
- `embedding_adapter_cfg.dropout`  
  - 含义：Adapter dropout。  
  - 影响：适配正则化。

### 1.2 策略头（Potential Scale）
- `actor_cfg.logit_scale_init`  
  - 含义：势能差打分的可学习尺度参数初始化值（实际使用 `exp(logit_scale)` 作为放大系数）。  
  - 影响：势能差与 `-log d_in` 的相对尺度。

### 1.3 时间与步数
- `max_steps`  
  - 含义：最大推理步数 K（由 dataset 配置提供）。  
  - 影响：轨迹长度上限；过小可能截断、过大增加噪声。

### 1.4 运行时策略
- `runtime_cfg.avoid_revisit`  
  - 含义：是否禁止回访已访问节点。  
  - 影响：探索多样性 vs. 搜索空间覆盖。
  - 约束：当前向量化 DB loss 不支持 `true`（会抛错），需保持为 `false`。
- `runtime_cfg.stop_min_steps`  
  - 含义：最小步数约束（step < min_steps 时禁止 STOP）。  
  - 影响：抑制 t=0/t=1 早停塌缩；验证/测试同样生效（不涉及标签）。
- 逆关系映射固定使用 `inverse_relations.json`，严格互逆为必备断言。  
- 预测模式固定为 full（不再暴露超参）。
- STOP 动作固定启用（不再暴露开关）。
- 命中答案节点后强制 STOP（STOP 前所在实体命中则奖励，否则惩罚）。

### 1.5 优化器
- `optimizer_cfg.type`  
  - 含义：优化器类型。  
- `optimizer_cfg.lr`  
  - 含义：基础学习率。  
- `optimizer_cfg.weight_decay`  
  - 含义：L2 正则权重。  
- 参数分组固定在代码中：LayerNorm/Embedding/BatchNorm/`*.bias` 不做 weight decay。

### 1.6 学习率调度
- `scheduler_cfg.type`  
  - 含义：调度类型（如 `onecycle`）。  
- `scheduler_cfg.interval`  
  - 含义：step 或 epoch 级调度。  
- `scheduler_cfg.max_lr`  
  - 含义：OneCycle 的峰值学习率。  
- `scheduler_cfg.pct_start`  
  - 含义：OneCycle 上升阶段比例。

### 1.7 训练控制
- `training_cfg.accumulate_grad_batches`  
  - 含义：梯度累积步数。  
  - 影响：等效 batch size。
- `training_cfg.num_rollouts`  
  - 含义：每 batch 采样 rollout 次数。  
  - 影响：信号稳定性与速度。
- `training_cfg.grad_clip_norm`  
  - 含义：梯度裁剪阈值。  
  - 影响：训练稳定性。
- 起点采样温度固定绑定到 `training_cfg.db_cfg.sampling_temperature_*`。

#### Detailed Balance 相关 (`training_cfg.db_cfg`)
- `sampling_temperature_start/end`  
  - 含义：动作采样温度余弦退火区间。  
  - 影响：探索强度与收敛速度。
- `dead_end_log_reward_start`  
  - 含义：冷启动 dead-end 的 log reward（退火起点）。  
  - 影响：早期惩罚力度。
- `dead_end_log_reward`  
  - 含义：最终 dead-end 的 log reward。  
  - 影响：失败惩罚强度。
- `dead_end_weight`  
  - 含义：失败样本的损失权重系数。  
  - 影响：失败路径在损失中的占比。
- `emit_log_reward`  
  - 含义：错误 STOP（emit）的 log reward（可比 dead_end 更重）。  
  - 影响：抑制“乱停”策略。

### 1.8 评估控制
- `evaluation_cfg.beam_size`  
  - 含义：beam 搜索宽度。  
  - 影响：召回与计算量。
- `evaluation_cfg.diverse_beam.*`  
  - 含义：多样性约束配置（分组、惩罚方式等）。  
  - 影响：路径多样性 vs. 纯得分。
- 答案级去重固定开启（仅对答案节点去重）。  
- `evaluation_cfg.answer_gain_stop.*`  
  - 含义：答案增益停止规则（patience/epsilon/min_beam）。  
  - 影响：自适应截断 beam。

---

## 2. 数据集相关模型超参（`configs/dataset/*.yaml`）

这些与模型直接耦合的超参只有少量：
- `max_steps`（如 `configs/dataset/webqsp.yaml: max_steps: 3`）  
  - 控制最大推理步数 K。

---

## 3. 代码内“魔法数字 / 常量”

### 3.1 `src/models/dual_flow_constants.py`
- **ID / 标记类**
  - `_NEG_ONE = -1`：非法节点/边占位  
  - `_NEG_TWO = -2`：STOP 动作 ID  
  - `_SELF_RELATION_ID = -1`：自环关系 ID  
  - `_INVALID_EDGE_ID = -1`：非法边 ID
- **终止类型**
  - `_TERMINAL_NONE = 0`  
  - `_TERMINAL_HIT = 1`  
  - `_TERMINAL_DEAD_END = 2`  
  - `_TERMINAL_MAX_STEPS = 3`  
  - `_TERMINAL_INVALID_START = 4`  
  - `_TERMINAL_EMIT = 5`
- **默认超参（代码默认值）**
  - `_DEFAULT_GNN_LAYERS = 2`  
  - `_DEFAULT_EDGE_DROPOUT = 0.1`  
  - `_DEFAULT_TRAIN_ROLLOUTS = 1`  
  - OneCycle 默认：`pct_start=0.3`, `div_factor=25`, `final_div_factor=10000` 等  
  - Diverse beam 默认：`groups=4`, `penalty=hard`, `lambda=1.0`, `similarity=tail`

### 3.2 `src/models/components/backbone.py`
- `_PNA_EPS = 1e-6`：PNA 统计稳定性  
- `_PNA_SCALERS = 3`, `_PNA_AGGREGATORS = 4` → `_PNA_FEATURE_MULT=12`  
  - 代表 PNA 聚合特征维度倍数  
- `SinusoidalPositionalEncoding` 中使用常数 **10000** 作为频率基底  
- `_DEFAULT_ADAPTER_DIM_DIVISOR = 4`：adapter 默认 bottleneck 比例  
- `_LOGZ_OUTPUT_DIM = 1`：logZ/stop 头输出维度

### 3.3 `src/models/components/bilinear_step_scorer.py`
- `scale = d^{-0.5}`  
  - 双线性打分的缩放系数  
- `w_query_shift` 线性映射  
  - 用于将 `c_flow` 投影到关系空间做动态平移

### 3.4 DualFlow 路径记忆（新增）
- `null_relation_emb`：学习到的空关系向量（t=0）。  
  - 作用：初始化路径记忆 \(h_0\)。  
- `path_gru`：GRUCell，隐维度 `hidden_dim`。  
  - 作用：递推路径记忆 \(h_t\)。

---

## 4. 建议维护方式（避免“魔法数字”失控）
- **配置优先**：凡能在 `configs/` 里暴露的不要写死在代码里。  
- **默认值只做兜底**：代码默认值应与配置保持一致。  
- **新增超参必须写文档**：可参考本文格式追加。

---

如需我把某一段代码的常量全部收敛到配置（完全消灭魔法数字），告诉我优先级和范围。  
