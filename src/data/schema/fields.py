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
    ANCHOR_NODE_FORWARD_DISTANCE_FLAT = (
        "anchor_node_forward_distances_flat"  # LongTensor [num_nodes]
    )
    ANCHOR_NODE_BACKWARD_DISTANCE_FLAT = (
        "anchor_node_backward_distances_flat"  # LongTensor [num_nodes]
    )
    NODE_TARGET_DISTANCE = "node_target_distance"  # LongTensor [num_nodes]
    NODE_TARGET_DISTANCES_FLAT = (
        "node_target_distances_flat"  # LongTensor [T * num_nodes]
    )
    NODE_TARGET_SHORTEST_PATH_COUNT_FLAT = (
        "node_target_shortest_path_count_flat"  # FloatTensor [T * num_nodes]
    )
    NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_INDICES = (
        "node_target_shortest_path_edge_count_indices"  # LongTensor [nnz] into [T * num_edges]
    )
    NODE_TARGET_SHORTEST_PATH_EDGE_COUNT_VALUES = (
        "node_target_shortest_path_edge_count_values"  # FloatTensor [nnz]
    )
    REPLAY_TRAJECTORY_EDGE_IDS = (
        "replay_trajectory_edge_ids"  # LongTensor [sum replay trajectory lengths]
    )
    REPLAY_TRAJECTORY_LENGTHS = (
        "replay_trajectory_lengths"  # LongTensor [num replay trajectories]
    )
