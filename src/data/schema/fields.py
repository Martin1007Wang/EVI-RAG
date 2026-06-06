class SampleFields:
    SAMPLE_ID = "sample_id"  # scalar uint8 tensor containing utf-8 bytes
    EDGE_INDEX = "edge_index"  # LongTensor [2, num_edges]
    NODE_ENTITY_CATALOG_IDS = "node_entity_catalog_ids"  # LongTensor [num_nodes]
    EDGE_RELATION_CATALOG_IDS = "edge_relation_catalog_ids"  # LongTensor [num_edges]
    NUM_NODES = "num_nodes"  # scalar LongTensor
    NUM_EDGES = "num_edges"  # scalar LongTensor
    QUESTION_EMB = "question_emb"  # FloatTensor [hidden_dim]
    ANCHOR_NODE_IDS = "anchor_node_ids"  # LongTensor [num_anchors]
    TARGET_NODE_IDS = "target_node_ids"  # LongTensor [num_answers_in_graph]
    REACHABLE_TARGET_NODE_IDS = (
        "reachable_target_node_ids"  # LongTensor [num_reachable_targets]
    )
    NODE_TARGET_DISTANCE = "node_target_distance"  # LongTensor [num_nodes]
    EDGE_ON_SHORTEST_PATH = "edge_on_shortest_path"  # BoolTensor [num_edges]
    REACHABLE_TARGET_MAX_DISTANCE = "reachable_target_max_distance"  # scalar LongTensor
    REPLAY_BANK_EDGE_IDS = "replay_bank_edge_ids"
    REPLAY_BANK_EDGE_COUNT = "replay_bank_edge_count"
    REPLAY_BANK_PRIORITY = "replay_bank_priority"
