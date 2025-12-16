# Data config quick map

- Hydra group: `data` selects the DataModule +其超参。算子在 `src/data`，变量在这里。
- 扁平化管线：
  - `retriever.yaml` → `GRetrievalDataModule`（graph retrieval），自动读取 `${dataset.*}`。
  - `gflownet.yaml` → `GAgentDataModule`（G_flow 架线的 g_agent 缓存）。
  - `llm_reasoner_triplet.yaml` → `LLMReasonerTripletDataModule`（triplet 版离线 QA 推理）。
  - `llm_reasoner.yaml` → `LLMReasonerPathDataModule` + `LLMReasonerPathDataset`（路径版，消费 `eval_gflownet/*_gflownet_eval.pt`，内含 chain_text）。
  - `llm_reasoner_truth.yaml` → `LLMReasonerTruthDataModule`（retriever oracle 上限：answer hit/recall@k）。
- 只需在 `defaults` 里 override `/dataset:<name>`，即可在所有管线上复用同一数据集路径；必要时可用 CLI `dataset=<name>` 切换。
- 如需新增数据集：添加 `configs/dataset/<name>.yaml`；若管线需特化字段，可在数据文件中局部覆写，但保持“管线文件”无数据集后缀。
