import argparse
import glob
import os
import warnings
from collections import defaultdict
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

# 忽略 pandas 的未来警告，避免噪声
warnings.filterwarnings("ignore")


def parse_list_field(field: object) -> List[str]:
    """解析可能是 list/ndarray/string 的字段，输出字符串列表。"""
    if field is None:
        return []
    if isinstance(field, np.ndarray):
        field = field.tolist()
    if isinstance(field, list):
        return [str(x) for x in field]
    if isinstance(field, str):
        cleaned = field.strip("[]'\" ")
        if not cleaned:
            return []
        return [x.strip().strip("'\"") for x in cleaned.split(",")]
    return []


def get_shortest_path_stats(
    G: nx.MultiDiGraph, q_nodes: Sequence[str], a_nodes: Sequence[str]
) -> Tuple[float, bool]:
    """使用 multi-source BFS 计算最短路径长度。返回 (最短 hop, 是否存在路径)。"""
    if not q_nodes or not a_nodes or len(G) == 0:
        return float("inf"), False

    q_nodes = [q for q in q_nodes if q in G]
    a_nodes = [a for a in a_nodes if a in G]
    if not q_nodes or not a_nodes:
        return float("inf"), False

    # 一次 multi-source BFS，避免 q×a 重复搜索
    lengths = nx.multi_source_shortest_path_length(G, q_nodes)
    a_dists = [lengths[a] for a in a_nodes if a in lengths]
    if not a_dists:
        return float("inf"), False

    return float(min(a_dists)), True


def describe_distribution(name: str, values: Iterable[int]) -> str:
    arr = np.array(list(values), dtype=float)
    if len(arr) == 0:
        return f"{name}: N/A"
    return (
        f"{name}: mean {np.mean(arr):.2f}, "
        f"p50 {np.percentile(arr, 50):.2f}, "
        f"p90 {np.percentile(arr, 90):.2f}"
    )


def ensure_required_columns(df: pd.DataFrame, path: str, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列 {missing}，不符合 q/a/graph 数据格式。")


def analyze_split(split_name: str, files: Sequence[str]) -> None:
    print(f"\n{'='*20} 正在分析 Split: {split_name} {'='*20}")
    print(f"包含文件: {[os.path.basename(f) for f in files]}")

    stats = {
        "total_samples": 0,
        # 基础计数
        "q_entity_counts": [],
        "a_entity_counts": [],
        "graph_triples_counts": [],
        "graph_nodes_counts": [],
        # 覆盖率
        "q_entities_total": 0,
        "q_entities_in_graph": 0,
        "a_entities_total": 0,
        "a_entities_in_graph": 0,
        # 连通性
        "samples_with_path": 0,
        "samples_without_path": 0,
        "samples_with_q_missing": 0,
        "samples_with_a_missing": 0,
        # 新增统计量
        "path_hops": [],
        "graph_max_degrees": [],
        "graph_mean_degrees": [],
        "graph_p90_degrees": [],
        "snr_ratios": [],
    }

    for file_path in files:
        df = pd.read_parquet(file_path)
        ensure_required_columns(df, file_path, ["q_entity", "a_entity", "graph"])

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"
        ):
            stats["total_samples"] += 1

            # 1. 解析
            graph_data = row["graph"]
            if isinstance(graph_data, str):
                import json

                try:
                    graph_data = json.loads(graph_data)
                except Exception:
                    graph_data = []
            if graph_data is None:
                graph_data = []

            q_entities = parse_list_field(row["q_entity"])
            a_entities = parse_list_field(row["a_entity"])

            # 2. 建图 & 基础统计（有向多重图，保留关系标签，避免人为低估路径长度）
            G = nx.MultiDiGraph()
            graph_nodes = set()
            edge_triples = set()

            for triple in graph_data:
                if len(triple) >= 3:
                    h, r, t = str(triple[0]), str(triple[1]), str(triple[2])
                    G.add_edge(h, t, key=r)
                    graph_nodes.add(h)
                    graph_nodes.add(t)
                    edge_triples.add((h, r, t))

            stats["q_entity_counts"].append(len(q_entities))
            stats["a_entity_counts"].append(len(a_entities))
            stats["graph_triples_counts"].append(len(graph_data))
            stats["graph_nodes_counts"].append(len(graph_nodes))

            # 度分布统计
            if len(G) > 0:
                degrees = [d for _, d in G.degree()]
                stats["graph_max_degrees"].append(np.max(degrees))
                stats["graph_mean_degrees"].append(np.mean(degrees))
                stats["graph_p90_degrees"].append(np.percentile(degrees, 90))
            else:
                stats["graph_max_degrees"].append(0)
                stats["graph_mean_degrees"].append(0)
                stats["graph_p90_degrees"].append(0)

            # 3. 覆盖率
            current_q_in_graph = [q for q in q_entities if q in graph_nodes]
            current_a_in_graph = [a for a in a_entities if a in graph_nodes]
            stats["q_entities_total"] += len(q_entities)
            stats["q_entities_in_graph"] += len(current_q_in_graph)
            stats["a_entities_total"] += len(a_entities)
            stats["a_entities_in_graph"] += len(current_a_in_graph)

            # 4. 连通性 & 跳数 & SNR
            if not current_q_in_graph:
                stats["samples_with_q_missing"] += 1
            if not current_a_in_graph:
                stats["samples_with_a_missing"] += 1
            if not current_q_in_graph or not current_a_in_graph:
                stats["samples_without_path"] += 1
            else:
                min_len, found = get_shortest_path_stats(
                    G, current_q_in_graph, current_a_in_graph
                )
                if found:
                    stats["samples_with_path"] += 1
                    stats["path_hops"].append(min_len)

                    # 使用去重后的三元组数量，避免重复边稀释 SNR
                    denom = max(len(edge_triples), 1)
                    stats["snr_ratios"].append(min_len / denom)
                else:
                    stats["samples_without_path"] += 1

    # === 打印统计结果 ===
    print(f"\n--- 统计结果: {split_name} ---")
    print("1. 总体概况")
    print(f"   样本总数: {stats['total_samples']}")
    print(f"   图平均节点数: {np.mean(stats['graph_nodes_counts']):.1f}")
    print(f"   图平均边数: {np.mean(stats['graph_triples_counts']):.1f}")
    print(f"   {describe_distribution('Q entities/样本', stats['q_entity_counts'])}")
    print(f"   {describe_distribution('A entities/样本', stats['a_entity_counts'])}")

    print("\n2. 实体召回 (Recall)")
    q_recall = (
        stats["q_entities_in_graph"] / stats["q_entities_total"]
        if stats["q_entities_total"] > 0
        else 0
    )
    a_recall = (
        stats["a_entities_in_graph"] / stats["a_entities_total"]
        if stats["a_entities_total"] > 0
        else 0
    )
    print(f"   Q_Entity Hit: {q_recall:.2%}")
    print(f"   A_Entity Hit: {a_recall:.2%}")

    print("\n3. 连通性与难度分析 (关键决策指标)")
    success_rate = (
        stats["samples_with_path"] / stats["total_samples"]
        if stats["total_samples"] > 0
        else 0
    )
    print(f"   可达率 (Connectivity): {success_rate:.2%}")

    hops = np.array(stats["path_hops"])
    if len(hops) > 0:
        print("   GT 路径跳数分布 (Hop Distribution):")
        print(f"     Mean Hop: {np.mean(hops):.2f}")
        for k in range(1, 5):
            ratio = np.mean(hops == k)
            print(f"     {k}-hop: {ratio:.2%}")
        print(f"     >=5-hop: {np.mean(hops >= 5):.2%}")

    print("   图拓扑复杂度 (Topology):")
    print(f"     平均度数 (Mean Degree): {np.mean(stats['graph_mean_degrees']):.2f}")
    print(f"     Hub 节点平均最大度数 (Max Degree): {np.mean(stats['graph_max_degrees']):.2f}")
    print(f"     P90 度数: {np.mean(stats['graph_p90_degrees']):.2f}")

    snrs = np.array(stats["snr_ratios"])
    if len(snrs) > 0:
        print("   信噪比 (Signal-to-Noise Ratio):")
        print(f"     Mean SNR: {np.mean(snrs):.4%}")
        print(f"     Median SNR: {np.median(snrs):.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="统计 WebQSP/CWQ 数据集的高级图属性")
    parser.add_argument("--data_dir", type=str, required=True, help="Parquet 文件所在目录")
    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.data_dir, "**/*.parquet"), recursive=True)
    if not all_files:
        raise ValueError(f"{args.data_dir} 下未找到任何 parquet 文件，请传入具体 raw/data 路径。")

    splits = defaultdict(list)
    for f in all_files:
        basename = os.path.basename(f)
        if basename.startswith("train"):
            splits["train"].append(f)
        elif basename.startswith("val") or basename.startswith("dev"):
            splits["validation"].append(f)
        elif basename.startswith("test"):
            splits["test"].append(f)
        else:
            splits["unknown"].append(f)

    if splits["unknown"]:
        print(f"跳过未识别 split 文件: {[os.path.basename(f) for f in splits['unknown']]}")

    processed = False
    for split_name in ["train", "validation", "test"]:
        if splits[split_name]:
            processed = True
            analyze_split(split_name, sorted(splits[split_name]))

    if not processed:
        raise ValueError(f"{args.data_dir} 未包含 train/validation/test 前缀的 parquet 文件。")


if __name__ == "__main__":
    main()
