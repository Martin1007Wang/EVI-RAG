from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

logger = logging.getLogger(__name__)


@dataclass
class GAgentSample:
    """标准 g_agent 训练单元（确定性字段，不允许缺省/回退）。"""

    # 元数据
    sample_id: str
    question: str
    question_emb: torch.FloatTensor  # [1, D] (batched-friendly)
    # 本地拓扑 + 全局实体
    node_entity_ids: torch.LongTensor  # [N] global entity id
    edge_head_locals: torch.LongTensor  # [E] 0..N-1
    edge_tail_locals: torch.LongTensor  # [E] 0..N-1
    edge_relations: torch.LongTensor  # [E]
    edge_scores: torch.FloatTensor  # [E]
    edge_labels: torch.FloatTensor  # [E]
    top_edge_mask: torch.BoolTensor  # [E] true 表示保留边
    # 关键节点（全局/局部）
    start_entity_ids: torch.LongTensor  # [Ks]
    answer_entity_ids: torch.LongTensor  # [Ka]
    # GT 路径（本地边/节点索引）
    gt_path_edge_local_ids: torch.LongTensor  # [P]
    gt_path_node_local_ids: torch.LongTensor  # [P_node]
    # 可空节点索引（为空时使用 shape [0] 的 long tensor）
    start_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    answer_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    gt_path_exists: bool = False
    # 可达性标志
    is_answer_reachable: bool = False


def _load_g_agent_cache(cache_path: Path):
    """Load g_agent cache safely across torch versions (torch 2.6+ defaults weights_only=True)."""
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    return torch.load(cache_path, **load_kwargs)


def _builder_sample_to_record(sample: GAgentSample) -> Dict:
    """Normalize legacy builder-style dataclass into dict format consumed by parser."""
    num_edges = int(sample.edge_head_locals.numel())
    num_nodes = int(sample.node_entity_ids.numel())
    edge_local_ids = list(range(num_edges))
    nodes = [
        {
            "local_index": i,
            "entity_id": int(sample.node_entity_ids[i].item()),
            "id": int(sample.node_entity_ids[i].item()),
        }
        for i in range(num_nodes)
    ]
    edges = []
    for i in range(num_edges):
        edges.append(
            {
                "local_index": i,
                "head_local": int(sample.edge_head_locals[i].item()),
                "tail_local": int(sample.edge_tail_locals[i].item()),
                "relation": int(sample.edge_relations[i].item()),
                "label": float(sample.edge_labels[i].item()) if sample.edge_labels.numel() > i else 0.0,
                "score": float(sample.edge_scores[i].item()) if sample.edge_scores.numel() > i else 0.0,
            }
        )
    if hasattr(sample, "gt_path_edge_local_ids"):
        gt_edges = sample.gt_path_edge_local_ids.tolist()
    else:
        gt_edges = []
    if hasattr(sample, "gt_path_node_local_ids") and sample.gt_path_node_local_ids.numel() > 0:
        gt_nodes = sample.gt_path_node_local_ids.tolist()
    else:
        gt_nodes_set = set()
        for e_idx in gt_edges:
            if 0 <= e_idx < num_edges:
                h_local = int(sample.edge_head_locals[e_idx].item())
                t_local = int(sample.edge_tail_locals[e_idx].item())
                gt_nodes_set.add(h_local)
                gt_nodes_set.add(t_local)
        gt_nodes = sorted(gt_nodes_set)
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "question_emb": sample.question_emb,
        "selected_edges": edges,
        "selected_nodes": nodes,
        "top_edge_local_indices": edge_local_ids,
        "gt_path_edge_local_indices": gt_edges,
        "gt_path_node_local_indices": gt_nodes,
        "gt_path_exists": bool(sample.gt_path_exists),
        "start_entity_ids": sample.start_entity_ids.tolist(),
        "start_node_locals": sample.start_node_locals.tolist(),
        "answer_entity_ids": sample.answer_entity_ids.tolist(),
        "answer_node_locals": sample.answer_node_locals.tolist(),
        "is_answer_reachable": bool(sample.is_answer_reachable),
    }


def _parse_sample(record: Dict) -> GAgentSample:
    required_keys = [
        "sample_id",
        "question",
        "question_emb",
        "edge_relations",
        "edge_scores",
        "edge_labels",
        "edge_head_locals",
        "edge_tail_locals",
        "node_entity_ids",
        "top_edge_mask",
        "start_entity_ids",
        "start_node_locals",
        "answer_entity_ids",
        "answer_node_locals",
        "gt_path_edge_local_ids",
        "gt_path_node_local_ids",
        "gt_path_exists",
        "is_answer_reachable",
    ]
    missing = [k for k in required_keys if k not in record]
    if missing:
        raise ValueError(f"g_agent record missing keys: {missing}")

    def as_long(key): return torch.as_tensor(record[key], dtype=torch.long).view(-1)
    def as_float(key): return torch.as_tensor(record[key], dtype=torch.float32).view(-1)
    def as_bool(key): return torch.as_tensor(record[key], dtype=torch.bool).view(-1)

    node_entity_ids = as_long("node_entity_ids")
    edge_head_locals = as_long("edge_head_locals")
    edge_tail_locals = as_long("edge_tail_locals")
    edge_relations = as_long("edge_relations")
    edge_scores = as_float("edge_scores")
    edge_labels = as_float("edge_labels")
    top_edge_mask = as_bool("top_edge_mask")

    num_edges = edge_head_locals.numel()
    num_nodes = node_entity_ids.numel()
    if torch.unique(node_entity_ids).numel() != num_nodes:
        raise ValueError(f"node_entity_ids must be unique per sample: {record.get('sample_id')}")
    tensors_with_expected_edges = {
        "edge_tail_locals": edge_tail_locals,
        "edge_relations": edge_relations,
        "edge_scores": edge_scores,
        "edge_labels": edge_labels,
        "top_edge_mask": top_edge_mask,
    }
    for name, tensor in tensors_with_expected_edges.items():
        if tensor.numel() != num_edges:
            raise ValueError(f"{name} length {tensor.numel()} != num_edges {num_edges} for {record.get('sample_id')}")

    if num_nodes == 0:
        raise ValueError(f"g_agent record has no nodes: {record.get('sample_id')}")
    if num_edges == 0:
        raise ValueError(f"g_agent record has no edges: {record.get('sample_id')}")

    for name, locals_tensor in (("edge_head_locals", edge_head_locals), ("edge_tail_locals", edge_tail_locals)):
        if (locals_tensor < 0).any() or (locals_tensor >= num_nodes).any():
            raise ValueError(f"{name} out of range for {record.get('sample_id')}")

    derived_edge_heads = node_entity_ids[edge_head_locals]
    derived_edge_tails = node_entity_ids[edge_tail_locals]
    if "edge_heads" in record:
        edge_heads_disk = torch.as_tensor(record["edge_heads"], dtype=torch.long).view(-1)
        if edge_heads_disk.numel() != num_edges or not torch.equal(edge_heads_disk, derived_edge_heads):
            raise ValueError(
                f"edge_heads mismatch with node_entity_ids/edge_head_locals for {record.get('sample_id')}; "
                "drop redundant global IDs to fix inconsistency."
            )
    if "edge_tails" in record:
        edge_tails_disk = torch.as_tensor(record["edge_tails"], dtype=torch.long).view(-1)
        if edge_tails_disk.numel() != num_edges or not torch.equal(edge_tails_disk, derived_edge_tails):
            raise ValueError(
                f"edge_tails mismatch with node_entity_ids/edge_tail_locals for {record.get('sample_id')}; "
                "drop redundant global IDs to fix inconsistency."
            )

    start_entity_ids = as_long("start_entity_ids")
    answer_entity_ids = as_long("answer_entity_ids")
    start_node_locals = as_long("start_node_locals")
    answer_node_locals = as_long("answer_node_locals")
    if answer_entity_ids.numel() != answer_node_locals.numel():
        raise ValueError(
            f"answer_entity_ids length {answer_entity_ids.numel()} "
            f"!= answer_node_locals length {answer_node_locals.numel()} for {record.get('sample_id')}; "
            "builder must drop unreachable/duplicate answers to keep pointers aligned."
        )

    for name, locals_tensor in (("start_node_locals", start_node_locals), ("answer_node_locals", answer_node_locals)):
        if locals_tensor.numel() > 0 and ((locals_tensor < 0).any() or (locals_tensor >= num_nodes).any()):
            raise ValueError(f"{name} out of range for {record.get('sample_id')}")

    gt_path_edge_local_ids = as_long("gt_path_edge_local_ids")
    gt_path_node_local_ids = as_long("gt_path_node_local_ids")
    if gt_path_edge_local_ids.numel() > 0:
        if ((gt_path_edge_local_ids < 0) | (gt_path_edge_local_ids >= num_edges)).any():
            raise ValueError(f"gt_path_edge_local_ids out of range for {record.get('sample_id')}")
    if gt_path_node_local_ids.numel() > 0:
        if ((gt_path_node_local_ids < 0) | (gt_path_node_local_ids >= num_nodes)).any():
            raise ValueError(f"gt_path_node_local_ids out of range for {record.get('sample_id')}")

    computed_reachable = answer_node_locals.numel() > 0
    declared_reachable = bool(record["is_answer_reachable"])
    if computed_reachable != declared_reachable:
        raise ValueError(
            f"is_answer_reachable mismatch for {record.get('sample_id')}: "
            f"declared={declared_reachable}, computed={computed_reachable}"
        )

    sample = GAgentSample(
        sample_id=str(record["sample_id"]),
        question=str(record.get("question", "")),
        question_emb=torch.as_tensor(record["question_emb"], dtype=torch.float32).view(-1),
        node_entity_ids=node_entity_ids,
        edge_head_locals=edge_head_locals,
        edge_tail_locals=edge_tail_locals,
        edge_relations=edge_relations,
        edge_scores=edge_scores,
        edge_labels=edge_labels,
        top_edge_mask=top_edge_mask,
        start_entity_ids=start_entity_ids,
        answer_entity_ids=answer_entity_ids,
        start_node_locals=start_node_locals,
        answer_node_locals=answer_node_locals,
        gt_path_edge_local_ids=gt_path_edge_local_ids,
        gt_path_node_local_ids=gt_path_node_local_ids,
        gt_path_exists=bool(record["gt_path_exists"]),
        is_answer_reachable=declared_reachable,
    )
    if sample.question_emb.dim() == 1:
        sample.question_emb = sample.question_emb.unsqueeze(0)
    if sample.start_entity_ids.numel() == 0:
        raise ValueError(f"g_agent record missing non-empty start_entity_ids: {record.get('sample_id')}")
    if sample.answer_entity_ids.numel() == 0:
        raise ValueError(f"g_agent record missing non-empty answer_entity_ids: {record.get('sample_id')}")
    if sample.start_node_locals.numel() == 0:
        raise ValueError(
            f"g_agent record missing start_node_locals for {record.get('sample_id')}; "
            "builder must emit per-graph start_node_locals aligned to start_entity_ids instead of relying on inference."
        )
    return sample


def load_g_agent_samples(
    cache_path: Path,
    *,
    drop_unreachable: bool = False,
) -> List[GAgentSample]:
    cache_path = Path(cache_path).expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"g_agent cache not found: {cache_path}")
    payload = _load_g_agent_cache(cache_path)
    raw_samples: Sequence[Dict] = payload.get("samples") or []
    if not raw_samples:
        raise ValueError(f"No samples found in {cache_path}")

    samples: List[GAgentSample] = []
    dropped = 0
    for record in raw_samples:
        sample = _parse_sample(record)
        if drop_unreachable and not sample.is_answer_reachable:
            dropped += 1
            continue
        samples.append(sample)
    if drop_unreachable and dropped > 0:
        logger.info("Filtered %d unreachable samples (is_answer_reachable=False) from %s", dropped, cache_path)
    if not samples:
        raise ValueError(f"No usable samples loaded from {cache_path} (drop_unreachable={drop_unreachable})")
    logger.info("Loaded g_agent cache %s: %d samples.", cache_path, len(samples))
    return samples


class GAgentData(PyGData):
    """PyG Data 携带 g_agent 字段，并定义增量规则避免批内索引错乱。"""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return int(self.num_nodes)
        if key in {"start_node_locals", "answer_node_locals", "gt_path_node_local_ids"}:
            return int(self.num_nodes)
        if key == "gt_path_edge_local_ids":
            return int(self.edge_index.size(1))
        return super().__inc__(key, value, *args, **kwargs)


def _sample_to_pyg_data(sample: GAgentSample) -> GAgentData:
    edge_index = torch.stack([sample.edge_head_locals, sample.edge_tail_locals], dim=0)
    question_emb = sample.question_emb
    if question_emb.dim() == 1:
        question_emb = question_emb.unsqueeze(0)
    start_node_locals = sample.start_node_locals
    if start_node_locals.numel() == 0:
        raise ValueError(
            f"GAgentSample {sample.sample_id} missing start_node_locals; "
            "dataset must materialize start anchors instead of inferring from start_entity_ids."
        )
    data = GAgentData(
        edge_index=edge_index,
        edge_attr=sample.edge_relations,
        edge_scores=sample.edge_scores,
        edge_labels=sample.edge_labels,
        top_edge_mask=sample.top_edge_mask,
        node_global_ids=sample.node_entity_ids,
        question_emb=question_emb,
        start_node_locals=start_node_locals,
        answer_node_locals=sample.answer_node_locals,
        start_entity_ids=sample.start_entity_ids,
        answer_entity_ids=sample.answer_entity_ids,
        gt_path_edge_local_ids=sample.gt_path_edge_local_ids,
        gt_path_node_local_ids=sample.gt_path_node_local_ids,
        gt_path_exists=torch.tensor([sample.gt_path_exists], dtype=torch.bool),
        is_answer_reachable=torch.tensor([sample.is_answer_reachable], dtype=torch.bool),
        sample_id=sample.sample_id,
        question=sample.question,
    )
    data.num_nodes = int(sample.node_entity_ids.numel())
    return data


def pyg_batch_to_dense(batch: PyGData) -> Dict[str, torch.Tensor | List[str | None]]:
    """将 PyG Batch 转换为旧的密集张量批格式（仍需 mask，但基于 PyG 批原语确定切片）。"""
    num_graphs = int(batch.num_graphs)
    node_ptr = batch.ptr
    edge_batch = batch.batch[batch.edge_index[0]]
    edge_counts = torch.bincount(edge_batch, minlength=num_graphs)
    edge_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=edge_counts.device), edge_counts.cumsum(0)])

    max_edges = int(edge_counts.max().item())
    max_nodes = int((node_ptr[1:] - node_ptr[:-1]).max().item())

    device = batch.edge_index.device
    edge_heads = torch.full((num_graphs, max_edges), -1, dtype=torch.long, device=device)
    edge_tails = torch.full((num_graphs, max_edges), -1, dtype=torch.long, device=device)
    edge_relations = torch.full((num_graphs, max_edges), -1, dtype=torch.long, device=device)
    edge_scores = torch.zeros(num_graphs, max_edges, dtype=torch.float32, device=device)
    edge_labels = torch.zeros(num_graphs, max_edges, dtype=torch.float32, device=device)
    edge_local_ids = torch.full((num_graphs, max_edges), -1, dtype=torch.long, device=device)
    edge_mask = torch.zeros(num_graphs, max_edges, dtype=torch.bool, device=device)
    top_edge_mask = torch.zeros(num_graphs, max_edges, dtype=torch.bool, device=device)

    node_local_ids = torch.full((num_graphs, max_nodes), -1, dtype=torch.long, device=device)
    node_entity_ids = torch.full((num_graphs, max_nodes), -1, dtype=torch.long, device=device)
    node_mask = torch.zeros(num_graphs, max_nodes, dtype=torch.bool, device=device)

    max_start_locals = 0
    max_answer_locals = 0
    start_node_locals_buf: List[torch.Tensor] = []
    answer_node_locals_buf: List[torch.Tensor] = []
    start_entities_buf: List[torch.Tensor] = []
    answer_entities_buf: List[torch.Tensor] = []
    gt_edge_buf: List[torch.Tensor] = []
    gt_node_buf: List[torch.Tensor] = []

    for g in range(num_graphs):
        es, ee = int(edge_ptr[g].item()), int(edge_ptr[g + 1].item())
        ns, ne = int(node_ptr[g].item()), int(node_ptr[g + 1].item())

        edge_slice = slice(es, ee)
        node_slice = slice(ns, ne)

        e_count = ee - es
        n_count = ne - ns
        edge_heads[g, :e_count] = batch.node_global_ids[batch.edge_index[0, edge_slice]]
        edge_tails[g, :e_count] = batch.node_global_ids[batch.edge_index[1, edge_slice]]
        edge_relations[g, :e_count] = batch.edge_attr[edge_slice]
        edge_scores[g, :e_count] = batch.edge_scores[edge_slice]
        edge_labels[g, :e_count] = batch.edge_labels[edge_slice]
        edge_local_ids[g, :e_count] = torch.arange(e_count, device=device)
        edge_mask[g, :e_count] = True
        top_edge_mask[g, :e_count] = batch.top_edge_mask[edge_slice]

        node_entity_ids[g, :n_count] = batch.node_global_ids[node_slice]
        node_local_ids[g, :n_count] = torch.arange(n_count, device=device)
        node_mask[g, :n_count] = True

        start_nodes = batch.start_node_locals
        start_nodes = start_nodes[(start_nodes >= ns) & (start_nodes < ne)] - ns
        answer_nodes = batch.answer_node_locals
        answer_nodes = answer_nodes[(answer_nodes >= ns) & (answer_nodes < ne)] - ns
        start_node_locals_buf.append(start_nodes)
        answer_node_locals_buf.append(answer_nodes)
        max_start_locals = max(max_start_locals, int(start_nodes.numel()))
        max_answer_locals = max(max_answer_locals, int(answer_nodes.numel()))

        start_entities_buf.append(node_entity_ids[g, start_nodes] if start_nodes.numel() > 0 else torch.empty(0, dtype=torch.long, device=device))
        answer_entities_buf.append(node_entity_ids[g, answer_nodes] if answer_nodes.numel() > 0 else torch.empty(0, dtype=torch.long, device=device))

        gt_edges = batch.gt_path_edge_local_ids
        gt_edges = gt_edges[(gt_edges >= es) & (gt_edges < ee)] - es
        gt_nodes = batch.gt_path_node_local_ids
        gt_nodes = gt_nodes[(gt_nodes >= ns) & (gt_nodes < ne)] - ns
        gt_edge_buf.append(gt_edges)
        gt_node_buf.append(gt_nodes)

    # pad variable-length fields
    if max_start_locals > 0:
        start_node_locals = torch.full((num_graphs, max_start_locals), -1, dtype=torch.long, device=device)
        start_node_mask = torch.zeros(num_graphs, max_start_locals, dtype=torch.bool, device=device)
    else:
        start_node_locals = torch.empty(num_graphs, 0, dtype=torch.long, device=device)
        start_node_mask = torch.empty(num_graphs, 0, dtype=torch.bool, device=device)

    if max_answer_locals > 0:
        answer_node_locals = torch.full((num_graphs, max_answer_locals), -1, dtype=torch.long, device=device)
        answer_node_mask = torch.zeros(num_graphs, max_answer_locals, dtype=torch.bool, device=device)
    else:
        answer_node_locals = torch.empty(num_graphs, 0, dtype=torch.long, device=device)
        answer_node_mask = torch.empty(num_graphs, 0, dtype=torch.bool, device=device)

    max_start_entities = max((t.numel() for t in start_entities_buf), default=0)
    max_answer_entities = max((t.numel() for t in answer_entities_buf), default=0)
    start_entity_ids = torch.full((num_graphs, max_start_entities), -1, dtype=torch.long, device=device) if max_start_entities > 0 else torch.empty(num_graphs, 0, dtype=torch.long, device=device)
    start_entity_mask = torch.zeros_like(start_entity_ids, dtype=torch.bool)
    answer_entity_ids = torch.full((num_graphs, max_answer_entities), -1, dtype=torch.long, device=device) if max_answer_entities > 0 else torch.empty(num_graphs, 0, dtype=torch.long, device=device)
    answer_mask = torch.zeros_like(answer_entity_ids, dtype=torch.bool)

    max_gt_edges = max((t.numel() for t in gt_edge_buf), default=0)
    gt_path_edges = torch.full((num_graphs, max_gt_edges), -1, dtype=torch.long, device=device) if max_gt_edges > 0 else torch.empty(num_graphs, 0, dtype=torch.long, device=device)
    gt_path_edge_mask = torch.zeros_like(gt_path_edges, dtype=torch.bool)
    gt_path_mask = torch.zeros(num_graphs, max_edges, dtype=torch.bool, device=device) if max_edges > 0 else torch.empty(num_graphs, 0, dtype=torch.bool, device=device)
    gt_path_exists = batch.gt_path_exists.view(-1).to(device)
    gt_path_length = torch.zeros(num_graphs, dtype=torch.float32, device=device)

    for g in range(num_graphs):
        if start_node_locals_buf[g].numel() > 0:
            dim = start_node_locals_buf[g].numel()
            start_node_locals[g, :dim] = start_node_locals_buf[g]
            start_node_mask[g, :dim] = True
        if answer_node_locals_buf[g].numel() > 0:
            dim = answer_node_locals_buf[g].numel()
            answer_node_locals[g, :dim] = answer_node_locals_buf[g]
            answer_node_mask[g, :dim] = True

        if start_entities_buf[g].numel() > 0:
            dim = start_entities_buf[g].numel()
            start_entity_ids[g, :dim] = start_entities_buf[g]
            start_entity_mask[g, :dim] = True
        if answer_entities_buf[g].numel() > 0:
            dim = answer_entities_buf[g].numel()
            answer_entity_ids[g, :dim] = answer_entities_buf[g]
            answer_mask[g, :dim] = True

        if gt_edge_buf[g].numel() > 0:
            plen = gt_edge_buf[g].numel()
            gt_path_edges[g, :plen] = gt_edge_buf[g]
            gt_path_edge_mask[g, :plen] = True
            gt_path_mask[g, gt_edge_buf[g]] = True
            gt_path_length[g] = float(plen)

    # question embedding reshape（假设同维度）
    question_emb = batch.question_emb
    if question_emb.dim() == 1:
        dim = question_emb.numel() // max(num_graphs, 1)
        question_emb = question_emb.view(num_graphs, dim) if dim > 0 else torch.empty(num_graphs, 0, device=device)
    elif question_emb.dim() == 2:
        question_emb = question_emb
    else:
        raise ValueError("Unsupported question_emb shape for batching.")

    sample_ids = list(batch.sample_id) if hasattr(batch, "sample_id") else [None] * num_graphs
    questions = list(batch.question) if hasattr(batch, "question") else [None] * num_graphs

    return {
        "sample_id": sample_ids,
        "question": questions,
        "start_node_locals": start_node_locals,
        "start_node_mask": start_node_mask,
        "edge_heads": edge_heads,
        "edge_tails": edge_tails,
        "edge_relations": edge_relations,
        "edge_scores": edge_scores,
        "edge_labels": edge_labels,
        "edge_local_ids": edge_local_ids,
        "edge_mask": edge_mask,
        "top_edge_mask": top_edge_mask,
        "node_local_ids": node_local_ids,
        "node_entity_ids": node_entity_ids,
        "node_mask": node_mask,
        "start_entity_ids": start_entity_ids,
        "start_entity_mask": start_entity_mask,
        "answer_entity_ids": answer_entity_ids,
        "answer_mask": answer_mask,
        "question_emb": question_emb,
        "answer_node_locals": answer_node_locals,
        "answer_node_mask": answer_node_mask,
        "gt_path_mask": gt_path_mask,
        "gt_path_edges": gt_path_edges,
        "gt_path_edge_mask": gt_path_edge_mask,
        "gt_path_exists": gt_path_exists,
        "gt_path_length": gt_path_length,
        "is_answer_reachable": batch.is_answer_reachable.view(-1).to(device),
    }


class GAgentPyGDataset(Dataset):
    """PyG 友好的 g_agent Dataset，输出 torch_geometric.data.Data。"""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        drop_unreachable: bool = False,
    ) -> None:
        super().__init__()
        self.samples = load_g_agent_samples(cache_path, drop_unreachable=drop_unreachable)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GAgentData:
        return _sample_to_pyg_data(self.samples[idx])


__all__ = [
    "GAgentSample",
    "GAgentData",
    "GAgentPyGDataset",
    "load_g_agent_samples",
    "_builder_sample_to_record",
    "_sample_to_pyg_data",
]
