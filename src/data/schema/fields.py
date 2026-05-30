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
    REPLAY_CANDIDATE_EDGE_IDS = (
        "replay_candidate_edge_ids"  # LongTensor [num_candidate_edges], graph-local edge ids
    )
    REPLAY_CANDIDATE_PTR = (
        "replay_candidate_ptr"  # LongTensor [num_candidates + 1], CSR pointer into replay_candidate_edge_ids
    )
    REPLAY_CANDIDATE_TARGET_POSITIONS = (
        "replay_candidate_target_positions"  # LongTensor [num_candidate_target_refs], reachable-target positions per candidate
    )
    REPLAY_CANDIDATE_TARGET_PTR = (
        "replay_candidate_target_ptr"  # LongTensor [num_candidates + 1], CSR pointer into replay_candidate_target_positions
    )
    REPLAY_EDGE_TO_CANDIDATE_IDS = (
        "replay_edge_to_candidate_ids"  # LongTensor [num_edge_candidate_refs], graph-local candidate ids
    )
    REPLAY_EDGE_TO_CANDIDATE_PTR = (
        "replay_edge_to_candidate_ptr"  # LongTensor [num_edges + 1], CSR pointer into replay_edge_to_candidate_ids
    )
    REPLAY_PATH_TRUNCATED = (
        "replay_path_truncated"  # scalar BoolTensor serialized as LongTensor {0,1}
    )
