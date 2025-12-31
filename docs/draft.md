  - g_agent 仍是检索子图裁剪：构图阶段有 edge_top_k/max_hops，最短路监督与 reward 都在这个裁剪子图里定义，并非全图最短路。若要第一性“全
    图”语义，需要改构图策略。src/data/components/g_agent_builder.py:245 configs/callbacks/g_agent_materializer.yaml
