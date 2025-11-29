# EVI-RAG 核心数据结构与算法速览

面向「正交算子 + 显式数据形态」的最小梳理，只保留特征处理 / 算法步骤 / 关键张量。

## 1. 局部图规范化（scripts/build_retrieval_parquet.py）
- **输入**：`${paths.data_dir}/${dataset}/raw/data/{split}-*.parquet`，字段 `id, question, graph(三元组列表), q_entity, a_entity, answer`。
- **实体/关系词表**：`EntityVocab` 拆分结构 ID 与文本 embedding ID（非文本→embedding_id=0，文本从 1 递增）；`RelationVocab` 递增 ID。
- **GraphRecord**（写入 `graphs.parquet`）：`graph_id`, `node_entity_ids`, `node_embedding_ids`, `edge_src`, `edge_dst`, `edge_relation_ids`, `positive_edge_mask`（可选仅最短路为正），`gt_path_edge_indices`, `gt_path_node_indices`。
- **Question 记录**（写入 `questions.parquet`）：`question_uid`, `dataset/split/kb`, `question`, `seed_entity_ids`, `answer_entity_ids`, `seed_embedding_ids`, `answer_embedding_ids`, `answer_texts`, `graph_id`, `metadata`。

## 2. G_retrieval 物化（scripts/build_retrieval_dataset.py）
- **词表 LMDB**：`vocabulary/vocabulary.lmdb` 存储 entity/relation ↔ id（pickle）。
- **静态嵌入**：`embeddings/entity_embeddings.pt`（文本实体池，0 行保留非文本占位），`relation_embeddings.pt`。
- **样本 LMDB (`embeddings/{split}.lmdb`)**：key=`graph_id`，value 为 pickle dict，主要字段：
  - 图：`edge_index[2,E]`, `edge_attr[E]`(全局关系ID), `labels[E]`, `num_nodes`, `node_global_ids[N]`, `node_embedding_ids[N]`, `topic_one_hot[N,num_topics]`
  - 问题：`question_emb[D]`, `question`，`q_local_indices`，`a_local_indices`
  - 监督：`gt_paths_nodes`, `gt_paths_triples`
  - 元数据：`sample_id`, `idx`
- **在线视角**：`GRetrievalDataset` 返回 PyG Data，仅包含 ID；`UnifiedDataLoader` 在迭代时拉取全局实体/关系嵌入并可做 hard negative 采样。

## 3. System 1 检索（src/models/retriever_module.py）
- **模型/损失/优化**：全部 Hydra 注入，`create_retriever` + `create_loss_function` + `setup_optimizer`。
- **训练/评估**：forward 产出 `scores`（可带不确定性），loss 由配置控制；`_gather_predictions` 收集每样本 scores/labels/head/tail/answer_ids 以计算 MRR/Precision@k/Recall@k/答案召回 + 不确定性统计。
- **输出**：Lightning checkpoint，日志/metrics.db（可选）。

## 4. g_agent 子图（src/eval.py ➜ src/hrag/g_agent.py）
- **定位**：在 eval 阶段附加运行，默认输出 `${data_dir}/${dataset}/materialized/g_agent/{split}_g_agent.pt`。
- **当前算法：Score-aware Forward Beam（2-hop 路径分）**  
  1) 构建无向邻接；种子 = question 实体（为空则回退为最高分边端点）。  
  2) Hop1：对每个种子按边分降序保留前 `beam_width_hop1`（默认 50），Hub 抑制。  
  3) Hop2：对 Hop1 边扩展邻居，路径分 = `score(S→M) × score(M→T)`（log 域相加）；单跳 `(S→M)` 也纳入候选。  
  4) 全局按路径分排序取前 `final_k`（默认 50）条路径，收集其中所有唯一边为 `selected_edges`，并可选注入 GT 正边。  
  5) 终端节点 = incident 边最高分聚合的 top_n；选点 = 终端节点 ∪ 选边端点。  
- **缓存字段（每 sample）**：`sample_id`, `question`, `selected_edges`(local_index, head_entity_id, tail_entity_id, relation_id, score, label), `selected_nodes`(local_index, entity_id), `top_edge_local_indices`（与选边集合一致）, `gt_path_edge_local_indices`, `gt_path_node_local_indices`, `core_component_count`, `retrieval_failed`。

## 5. GFlowNet（System 1.5，src/models/gflow_net.py 等）
- **数据**：`GAgentPyGDataset` 读取 g_agent 缓存，PyG Batch 直接消费：
  - 边：`edge_heads/tails/relations/scores/labels`, `edge_mask`, `top_edge_mask`
  - 点：`node_entity_ids`, `start_entity_ids`（问题实体）, `answer_entity_ids`
  - GT：`gt_path_edge_local_ids`, `gt_path_node_local_ids`, `gt_path_exists`
- **环境 GraphEnv**：状态 = 静态图 + `selected_mask/selection_order/current_tail/step_counts/done`；动作 = 选择未用边或 stop，遵守起点/连通/防回退约束。
- **策略 RoleAwarePolicy**：CLS + 问题 + 边 token（candidate/selected/pad 类型嵌入 + 选中顺序嵌入）经 Transformer，输出边 logits 与 stop logit。
- **奖励**：
  - `System1GuidedReward`：综合正边/答案/GT 路径覆盖 + retriever 分数（长度惩罚、terminal 奖励、语义阈值，可选“仅语义成功”）。
  - `AnswerOnlyReward`：命中答案为主，reach_fraction/score 作为平滑 shaping。

## 6. LLM 生成（src/llm_generation.py）
- **数据准备**：`LLMGenerationDataModule` 读取 g_agent 缓存与 `questions/entity_vocab/relation_vocab.parquet`，按分数降序截断 `triplet_limit`，反查三元组文本并构造 `system_prompt` + `user_prompt`。
- **推理**：`LLMGenerationModule` 通过 vLLM 或 OpenAI 接口运行 `run_chat`，写出 `${paths.output_dir}/{dataset}-{prompt_tag}-{model}-{split}.jsonl` 与 `.metrics.json`。

## 7. 已知风险
- g_agent 聚合全量样本后一次性保存，长尾数据集可能占用大内存。
- LMDB/Parquet 缺少显式 schema_version，字段变更易在运行期抛 KeyError。
- g_agent 目前无 Steiner/USP 联通，若需要连通性增强需在 `_build_sample_record` 里重新启用相应逻辑。***
