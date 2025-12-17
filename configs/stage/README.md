# Stage config quick map

评估入口统一为 `python src/eval.py stage=<name> dataset=<dataset> ...`。

## Stages（每次运行只产一种“算子输出”）

- `retriever_eval`：评估 Retriever（`trainer.test`），并持久化 `eval_retriever/{split}_retriever_eval.pt`（供 truth/oracle）。
- `materialize_g_agent`：用 Retriever checkpoint 生成 g_agent 缓存（`trainer.predict`），输出 `g_agent/{split}_g_agent.pt`（供 GFlowNet）。
- `gflownet_eval`：评估 GFlowNet（`trainer.test`），并持久化 `eval_gflownet/{split}_gflownet_eval.pt`（供 paths reasoner）。
- `llm_reasoner_truth`：Retriever oracle 上界（不调用 LLM，消费 `eval_retriever` 缓存）。
- `llm_reasoner_paths`：LLM over paths（消费 `eval_gflownet` 缓存）。
- `llm_reasoner_triplet`：LLM over triplets（消费 `g_agent` 缓存）。

## Checkpoints（环境变量输入）

所有 stage 的 checkpoint 统一从 `ckpt.*` 注入：

- `ckpt.retriever`: Retriever checkpoint 路径
- `ckpt.gflownet`: GFlowNet checkpoint 路径

推荐 CLI 覆写（不要在仓库 yaml 里硬编码绝对路径）：

```bash
python src/eval.py stage=retriever_eval dataset=webqsp ckpt.retriever=/path/to/retriever.ckpt
```

