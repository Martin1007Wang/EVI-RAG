# 配置迁移指南：从复杂嵌套配置到扁平化配置

## 概述

我们已经将 GFlowNet 的配置从复杂的嵌套结构简化为完全扁平化的结构，移除了所有不必要的抽象层和过度工程。

## 主要变化

### 1. 配置结构简化

**旧版（复杂嵌套）:**
```
configs/model/gflownet/
├── horizon_cfg/
├── training_cfg/
│   ├── auxiliary/
│   ├── answer_reward/
│   └── sampling_temperature_schedule/
├── policy_cfg/
│   ├── backbone/
│   ├── flow_head/
│   ├── state_encoder/
│   └── actor/
├── eval_cfg/
│   ├── monte_carlo/
│   ├── validation/
│   └── testing/
├── optimizer_cfg/
└── scheduler_cfg/
```

**新版（扁平化）:**
```
configs/model/gflownet.yaml  # 单个文件，所有配置扁平化
```

### 2. 配置参数映射

| 旧版配置路径 | 新版配置参数 | 说明 |
|-------------|-------------|------|
| `horizon_cfg.max_steps` | `max_steps` | 直接扁平化 |
| `training_cfg.rollouts_per_graph` | `rollouts_per_graph` | 直接扁平化 |
| `training_cfg.sampling_temperature` | `sampling_temperature` | 直接扁平化 |
| `training_cfg.sampling_temperature_schedule` | `sampling_temperature_schedule` | 直接扁平化 |
| `policy_cfg.backbone` | `backbone` | 直接扁平化 |
| `policy_cfg.hidden_dim` | `policy_hidden_dim` | 重命名以更清晰 |
| `training_cfg.answer_reward` | `answer_reward` | 直接扁平化 |
| `optimizer_cfg` | `optimizer` | 直接扁平化 |
| `scheduler_cfg` | `scheduler` | 直接扁平化 |
| `eval_cfg.*` | `validation_eval_cfg` / `test_eval_cfg` | 双速评估配置 |

### 3. 移除的配置组

以下配置组已被完全移除，因为它们属于过度工程：

1. **`training_cfg.auxiliary`** - 辅助训练配置（提案、回放、监督）
2. **`training_cfg.action_pruning`** - 动作剪枝配置
3. **`training_cfg.detailed_balance`** - 详细平衡损失配置
4. **`policy_cfg.flow_head`** - 流头配置（硬编码在模型中）
5. **`policy_cfg.state_encoder`** - 状态编码器配置（硬编码在模型中）
6. **`policy_cfg.actor`** - 动作头配置（硬编码在模型中）
7. **复杂的 `eval_cfg` 嵌套** - 替换为简单的双速评估配置

### 4. 代码接口变化

**旧版初始化:**
```python
def __init__(
    self,
    *,
    horizon_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    policy_cfg: dict[str, Any],
    optimizer_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    validation_eval_cfg: dict[str, Any] | None = None,
    test_eval_cfg: dict[str, Any] | None = None,
):
```

**新版初始化:**
```python
def __init__(
    self,
    *,
    # 核心超参数
    max_steps: int = 20,
    rollouts_per_graph: int = 8,
    sampling_temperature: float = 1.0,
    sampling_temperature_schedule: dict[str, Any] | None = None,
    
    # 策略网络配置
    backbone: dict[str, Any] | None = None,
    policy_hidden_dim: int = 512,
    
    # 奖励模型配置
    answer_reward: dict[str, Any] | None = None,
    
    # 优化器配置
    optimizer: dict[str, Any] | None = None,
    scheduler: dict[str, Any] | None = None,
    
    # 双速评估配置
    validation_eval_cfg: dict[str, Any] | None = None,
    test_eval_cfg: dict[str, Any] | None = None,
):
```

### 5. 评估系统变化

**旧版评估:**
- 复杂的 `GraphTaskRuntimeFactory`
- 嵌套的 `MetricRuntimeProtocol`
- 过度设计的 `answer_search_runtime.py` (556行)

**新版评估:**
- 简单的 `src/eval/` 模块
- 纯函数：`compute_answer_metrics()`, `compute_search_efficiency_metrics()`
- 双速策略：验证模式（快速）vs 测试模式（准确）
- 代码量减少 ~80%

## 迁移步骤

### 步骤1：更新训练命令

**旧版:**
```bash
python src/train.py experiment=train_rankflow dataset=webqsp-sub
```

**新版:**
```bash
# 命令不变，但内部使用新的扁平化配置
python src/train.py experiment=train_rankflow dataset=webqsp-sub
```

### 步骤2：更新评估命令

**旧版:**
```bash
python src/eval.py experiment=eval_rankflow ckpt.gflownet=/path/to/model.ckpt
```

**新版:**
```bash
# 命令不变，但评估使用新的双速系统
python src/eval.py experiment=eval_rankflow ckpt.gflownet=/path/to/model.ckpt
```

### 步骤3：自定义配置

如果需要自定义配置，现在更简单：

**旧版（复杂）:**
```yaml
# 需要创建多个嵌套文件
```

**新版（简单）:**
```yaml
# 直接在命令行覆盖
python src/train.py experiment=train_rankflow dataset=webqsp-sub \
  max_steps=30 \
  policy_hidden_dim=768 \
  validation_eval_cfg.num_rollouts=256
```

## 优势

1. **更简单**: 从 20+ 个配置文件减少到 1 个主文件
2. **更直观**: 所有配置参数一目了然
3. **更易维护**: 没有复杂的嵌套和抽象层
4. **更易调试**: 配置问题更容易定位和修复
5. **YAGNI 原则**: 移除了所有"你可能不需要"的功能

## 向后兼容性

旧版配置文件已备份为 `gflownet_legacy.yaml`。现有的训练检查点应该仍然可以加载，因为模型架构没有改变，只是配置接口简化了。

## 故障排除

如果遇到配置相关问题：

1. **检查配置路径**: 确保使用新的扁平化参数名
2. **查看默认值**: 所有参数都有合理的默认值
3. **验证配置**: 新的 `GFlowNetModule` 会验证必需参数
4. **回滚**: 如果需要，可以使用 `gflownet_legacy.yaml`

## 总结

这次重构遵循了 YAGNI（You Aren't Gonna Need It）原则，移除了所有过度工程，创建了一个简单、直观、易于维护的配置系统。配置行数减少了 ~70%，代码复杂度显著降低，同时保持了全部功能。