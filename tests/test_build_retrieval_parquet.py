import scripts.build_retrieval_parquet as brp


def _write_raw_split(tmp_path, split: str, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": pa.array([r["id"] for r in rows], type=pa.string()),
            "question": pa.array([r["question"] for r in rows], type=pa.string()),
            "answer": pa.array([r["answer"] for r in rows], type=pa.list_(pa.string())),
            "q_entity": pa.array([r["q_entity"] for r in rows], type=pa.list_(pa.string())),
            "a_entity": pa.array([r["a_entity"] for r in rows], type=pa.list_(pa.string())),
            "graph": pa.array([r["graph"] for r in rows], type=pa.list_(pa.list_(pa.string()))),
        }
    )
    pq.write_table(table, tmp_path / f"{split}-0000.parquet")


def test_shortest_path_directed_forward_only():
    # Simple forward path 0 -> 1 -> 2 should be recovered.
    edges_src = [0, 1]
    edges_dst = [1, 2]
    path_edges, path_nodes = brp.shortest_path_edge_indices_directed(
        num_nodes=3,
        edge_src=edges_src,
        edge_dst=edges_dst,
        seeds=[0],
        answers=[2],
        undirected=False,
    )
    assert path_edges == [0, 1]
    assert path_nodes == [0, 1, 2]


def test_shortest_path_includes_backward_when_forward_missing():
    # Only backward path exists; it should be used.
    edges_src = [1]
    edges_dst = [0]
    path_edges, path_nodes = brp.shortest_path_edge_indices_directed(
        num_nodes=2,
        edge_src=edges_src,
        edge_dst=edges_dst,
        seeds=[0],
        answers=[1],
        undirected=False,
    )
    assert path_edges == [0]
    assert path_nodes == [0, 1]


def test_shortest_path_preserves_parallel_edges():
    # SubgraphRAG parity (nx.DiGraph): parallel edges collapse and keep the LAST edge id for (u,v).
    edges_src = [0, 0, 1]
    edges_dst = [1, 1, 2]
    path_edges, path_nodes = brp.shortest_path_edge_indices_directed(
        num_nodes=3,
        edge_src=edges_src,
        edge_dst=edges_dst,
        seeds=[0],
        answers=[2],
        undirected=False,
    )
    assert path_edges == [1, 2]  # last edge id for (0,1) is 1
    assert path_nodes == [0, 1, 2]


def test_preprocess_writes_base_and_sub_only_when_filtered(tmp_path):
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    sub_out_dir = tmp_path / "normalized-sub"
    raw_root.mkdir(parents=True)

    _write_raw_split(
        raw_root,
        "train",
        rows=[
            {
                "id": "t0",
                "question": "q",
                "answer": ["a"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a"],
                "graph": [["m.q", "r", "m.a"]],
            },
            {
                "id": "t1",
                "question": "q",
                "answer": ["a_missing"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a_missing"],
                "graph": [["m.q", "r", "m.x"]],
            },
        ],
    )
    _write_raw_split(
        raw_root,
        "validation",
        rows=[
            {
                "id": "v0",
                "question": "q",
                "answer": ["a_missing"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a_missing"],
                "graph": [["m.q", "r", "m.x"]],
            }
        ],
    )
    _write_raw_split(
        raw_root,
        "test",
        rows=[
            {
                "id": "e0",
                "question": "q",
                "answer": ["a_missing"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a_missing"],
                "graph": [["m.q", "r", "m.x"]],
            }
        ],
    )

    train_filter = brp.SplitFilter(
        skip_no_topic=False,
        skip_no_ans=True,
        skip_no_path=False,
        use_shortest_path_positive=True,
    )
    eval_filter = brp.SplitFilter(
        skip_no_topic=False,
        skip_no_ans=True,
        skip_no_path=False,
        use_shortest_path_positive=True,
    )
    override_filters = {
        "test": brp.SplitFilter(
            skip_no_topic=False,
            skip_no_ans=False,
            skip_no_path=False,
            use_shortest_path_positive=True,
        )
    }

    brp.preprocess(
        dataset="webqsp",
        kb="freebase",
        raw_root=raw_root,
        out_dir=out_dir,
        sub_out_dir=sub_out_dir,
        column_map={
            "question_id_field": "id",
            "question_field": "question",
            "answer_text_field": "answer",
            "q_entity_field": "q_entity",
            "a_entity_field": "a_entity",
            "graph_field": "graph",
        },
        entity_normalization="none",
        undirected_traversal=False,
        train_filter=train_filter,
        eval_filter=eval_filter,
        override_filters=override_filters,
        write_sub_if_filtered=True,
    )

    import pyarrow.parquet as pq

    base_questions = pq.read_table(out_dir / "questions.parquet")
    sub_questions = pq.read_table(sub_out_dir / "questions.parquet")
    assert base_questions.num_rows == 4
    assert sub_questions.num_rows == 2  # train(t0) + test(e0)


def test_preprocess_skips_sub_when_no_rows_filtered(tmp_path):
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    sub_out_dir = tmp_path / "normalized-sub"
    raw_root.mkdir(parents=True)

    _write_raw_split(
        raw_root,
        "train",
        rows=[
            {
                "id": "t0",
                "question": "q",
                "answer": ["a"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a"],
                "graph": [["m.q", "r", "m.a"]],
            }
        ],
    )
    _write_raw_split(
        raw_root,
        "test",
        rows=[
            {
                "id": "e0",
                "question": "q",
                "answer": ["a"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a"],
                "graph": [["m.q", "r", "m.a"]],
            }
        ],
    )

    train_filter = brp.SplitFilter(
        skip_no_topic=False,
        skip_no_ans=True,
        skip_no_path=False,
        use_shortest_path_positive=True,
    )
    eval_filter = brp.SplitFilter(
        skip_no_topic=False,
        skip_no_ans=True,
        skip_no_path=False,
        use_shortest_path_positive=True,
    )

    brp.preprocess(
        dataset="webqsp",
        kb="freebase",
        raw_root=raw_root,
        out_dir=out_dir,
        sub_out_dir=sub_out_dir,
        column_map={
            "question_id_field": "id",
            "question_field": "question",
            "answer_text_field": "answer",
            "q_entity_field": "q_entity",
            "a_entity_field": "a_entity",
            "graph_field": "graph",
        },
        entity_normalization="none",
        undirected_traversal=False,
        train_filter=train_filter,
        eval_filter=eval_filter,
        override_filters={},
        write_sub_if_filtered=True,
    )
    assert not sub_out_dir.exists()


def test_preprocess_drops_empty_graph_samples(tmp_path):
    raw_root = tmp_path / "raw"
    out_dir = tmp_path / "normalized"
    raw_root.mkdir(parents=True)

    _write_raw_split(
        raw_root,
        "train",
        rows=[
            {
                "id": "t0",
                "question": "q",
                "answer": ["a"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a"],
                "graph": [["m.q", "r", "m.a"]],
            },
            {
                "id": "t1",
                "question": "q",
                "answer": ["a"],
                "q_entity": ["m.q"],
                "a_entity": ["m.a"],
                "graph": [],
            },
        ],
    )

    filter_all = brp.SplitFilter(
        skip_no_topic=False,
        skip_no_ans=False,
        skip_no_path=False,
        use_shortest_path_positive=True,
    )

    brp.preprocess(
        dataset="webqsp",
        kb="freebase",
        raw_root=raw_root,
        out_dir=out_dir,
        sub_out_dir=None,
        column_map={
            "question_id_field": "id",
            "question_field": "question",
            "answer_text_field": "answer",
            "q_entity_field": "q_entity",
            "a_entity_field": "a_entity",
            "graph_field": "graph",
        },
        entity_normalization="none",
        undirected_traversal=False,
        train_filter=filter_all,
        eval_filter=filter_all,
        override_filters={},
        write_sub_if_filtered=False,
    )

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    graphs = pq.read_table(out_dir / "graphs.parquet", columns=["graph_id", "edge_src"])
    questions = pq.read_table(out_dir / "questions.parquet", columns=["graph_id"])
    assert graphs.num_rows == 1
    assert questions.num_rows == 1
    assert int(pc.list_value_length(graphs["edge_src"])[0].as_py()) > 0
    assert not bool(pc.any(pc.equal(questions["graph_id"], "webqsp/train/t1")).as_py())


def _build_minimal_vocab_for_sample(sample: brp.Sample):
    entity_vocab = brp.EntityVocab(kb=sample.kb)
    relation_vocab = brp.RelationVocab(kb=sample.kb)
    for h, r, t in sample.graph:
        entity_vocab.add_entity(h)
        entity_vocab.add_entity(t)
        relation_vocab.relation_id(r)
    for ent in sample.q_entity + sample.a_entity:
        entity_vocab.add_entity(ent)
    entity_vocab.finalize()
    return entity_vocab, relation_vocab


def test_build_graph_emits_triple_mask_and_gt_source_answer_subgraph():
    sample = brp.Sample(
        dataset="ds",
        split="train",
        question_id="0",
        kb="freebase",
        question="q",
        graph=[
            ("m.h", "r1", "m.t"),
            ("m.h", "r2", "m.t"),
        ],
        q_entity=["m.h"],
        a_entity=["m.t"],
        answer_texts=["a"],
        answer_subgraph=[("m.h", "r1", "m.t")],
    )
    entity_vocab, relation_vocab = _build_minimal_vocab_for_sample(sample)
    graph = brp.build_graph(
        sample,
        entity_vocab,
        relation_vocab,
        graph_id="ds/train/0",
        use_shortest_path_positive=True,
        undirected_traversal=False,
    )
    assert graph.gt_source == "answer_subgraph"
    assert graph.gt_path_edge_indices == [0]
    assert graph.positive_triple_mask == [True, False]
    # Transition-level supervision is lifted to all parallel edges between the same endpoints.
    assert graph.positive_edge_mask == [True, True]


def test_build_graph_emits_gt_source_shortest_path():
    sample = brp.Sample(
        dataset="ds",
        split="train",
        question_id="1",
        kb="freebase",
        question="q",
        graph=[("m.h", "r", "m.t")],
        q_entity=["m.h"],
        a_entity=["m.t"],
        answer_texts=["a"],
        answer_subgraph=[],
    )
    entity_vocab, relation_vocab = _build_minimal_vocab_for_sample(sample)
    graph = brp.build_graph(
        sample,
        entity_vocab,
        relation_vocab,
        graph_id="ds/train/1",
        use_shortest_path_positive=True,
        undirected_traversal=False,
    )
    assert graph.gt_source == "shortest_path"
