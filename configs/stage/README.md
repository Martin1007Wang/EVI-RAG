# Stage config quick map

评估入口统一为 `python src/eval.py stage=<name> dataset=<dataset> ...`。

## 标准流水（单 GPU，逐步产物单一）

1. `cache_g_agent`：单次 Retriever 前向，产出指标 + `g_agent/{split}_g_agent.pt`（并可选 `eval_retriever/{split}_retriever_eval.pt`）。
2. `gflownet_eval`：评估 GFlowNet（统一走 `trainer.predict`），产出指标 + `eval_gflownet/{split}_gflownet_eval.pt`。
3. `llm_reasoner_*`：
   - `llm_reasoner_triplet`：LLM over triplets（消费 `g_agent`）。
   - `llm_reasoner_paths`：LLM over paths（消费 `eval_gflownet`）。
   - `llm_reasoner_truth`：Oracle 上界（消费 retriever eval 缓存，可由 step1 打开 textualize 持久化）。

## Checkpoints（环境变量输入）

所有 stage 的 checkpoint 统一从 `ckpt.*` 注入：

- `ckpt.retriever`: Retriever checkpoint 路径
- `ckpt.gflownet`: GFlowNet checkpoint 路径

推荐 CLI 覆写（不要在仓库 yaml 里硬编码绝对路径）：

```bash
python src/eval.py stage=retriever_eval dataset=webqsp ckpt.retriever=/path/to/retriever.ckpt
```
