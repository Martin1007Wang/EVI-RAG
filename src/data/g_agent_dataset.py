from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as PyGData

logger = logging.getLogger(__name__)


@dataclass
class GAgentSample:
    """标准 g_agent 训练单元（核心字段确定性；可选审计字段允许缺省）。"""

    # 元数据
    sample_id: str
    question: str
    question_emb: torch.FloatTensor  # [1, D] (batched-friendly)
    # 本地拓扑 + 全局实体
    node_entity_ids: torch.LongTensor  # [N] global entity id
    node_embedding_ids: torch.LongTensor  # [N] embedding table row ids
    edge_head_locals: torch.LongTensor  # [E] 0..N-1
    edge_tail_locals: torch.LongTensor  # [E] 0..N-1
    edge_relations: torch.LongTensor  # [E]
    edge_scores: torch.FloatTensor  # [E]
    edge_labels: torch.FloatTensor  # [E]
    # 可选审计字段：标记该边是否来自 retriever Top-K（或其它“保留规则”）。
    # 最终 GFlowNet 训练不依赖此字段；若缓存缺失，将在加载时确定性地补全为全 True。
    top_edge_mask: torch.BoolTensor  # [E]
    # 关键节点（全局/局部）
    start_entity_ids: torch.LongTensor  # [Ks]
    answer_entity_ids: torch.LongTensor  # [Ka]
    # GT 路径（本地边/节点索引）
    gt_path_edge_local_ids: torch.LongTensor  # [P]
    gt_path_node_local_ids: torch.LongTensor  # [P_node]
    # 可空节点索引（为空时使用 shape [0] 的 long tensor）
    start_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    answer_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    # Pair-level shortest-path supervision (CSR-style).
    pair_start_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    pair_answer_node_locals: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    pair_edge_local_ids: torch.LongTensor = field(default_factory=lambda: torch.empty(0, dtype=torch.long))
    pair_edge_ptr: torch.LongTensor = field(default_factory=lambda: torch.tensor([0], dtype=torch.long))
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
    """Normalize GAgentSample into flat dict schema consumed by parser."""

    def _tolist(t):
        return t.tolist() if hasattr(t, "tolist") else t

    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "question_emb": _tolist(sample.question_emb),
        "edge_relations": _tolist(sample.edge_relations),
        "edge_scores": _tolist(sample.edge_scores),
        "edge_labels": _tolist(sample.edge_labels),
        "edge_head_locals": _tolist(sample.edge_head_locals),
        "edge_tail_locals": _tolist(sample.edge_tail_locals),
        "node_entity_ids": _tolist(sample.node_entity_ids),
        "node_embedding_ids": _tolist(sample.node_embedding_ids),
        "top_edge_mask": _tolist(sample.top_edge_mask),
        "start_entity_ids": _tolist(sample.start_entity_ids),
        "answer_entity_ids": _tolist(sample.answer_entity_ids),
        "gt_path_edge_local_ids": _tolist(sample.gt_path_edge_local_ids),
        "gt_path_node_local_ids": _tolist(sample.gt_path_node_local_ids),
        "gt_path_exists": bool(sample.gt_path_exists),
        "start_node_locals": _tolist(sample.start_node_locals),
        "answer_node_locals": _tolist(sample.answer_node_locals),
        "pair_start_node_locals": _tolist(sample.pair_start_node_locals),
        "pair_answer_node_locals": _tolist(sample.pair_answer_node_locals),
        "pair_edge_local_ids": _tolist(sample.pair_edge_local_ids),
        "pair_edge_ptr": _tolist(sample.pair_edge_ptr),
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
        "node_embedding_ids",
        "start_entity_ids",
        "start_node_locals",
        "answer_entity_ids",
        "answer_node_locals",
        "pair_start_node_locals",
        "pair_answer_node_locals",
        "pair_edge_local_ids",
        "pair_edge_ptr",
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
    node_embedding_ids = as_long("node_embedding_ids")

    num_edges = edge_head_locals.numel()
    num_nodes = node_entity_ids.numel()
    if torch.unique(node_entity_ids).numel() != num_nodes:
        raise ValueError(f"node_entity_ids must be unique per sample: {record.get('sample_id')}")
    tensors_with_expected_edges = {
        "edge_tail_locals": edge_tail_locals,
        "edge_relations": edge_relations,
        "edge_scores": edge_scores,
        "edge_labels": edge_labels,
    }
    for name, tensor in tensors_with_expected_edges.items():
        if tensor.numel() != num_edges:
            raise ValueError(f"{name} length {tensor.numel()} != num_edges {num_edges} for {record.get('sample_id')}")

    top_edge_mask: torch.Tensor
    if "top_edge_mask" in record:
        top_edge_mask = as_bool("top_edge_mask")
        if top_edge_mask.numel() != num_edges:
            raise ValueError(
                f"top_edge_mask length {top_edge_mask.numel()} != num_edges {num_edges} for {record.get('sample_id')}"
            )
    elif "top_edge_local_indices" in record:
        top_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        idx = torch.as_tensor(record["top_edge_local_indices"], dtype=torch.long).view(-1)
        if idx.numel() > 0:
            if (idx < 0).any() or (idx >= num_edges).any():
                raise ValueError(f"top_edge_local_indices out of range for {record.get('sample_id')}")
            top_edge_mask[idx] = True
    else:
        top_edge_mask = torch.ones(num_edges, dtype=torch.bool)

    if node_embedding_ids.numel() != num_nodes:
        raise ValueError(
            f"node_embedding_ids length {node_embedding_ids.numel()} != num_nodes {num_nodes} for {record.get('sample_id')}"
        )

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
    pair_start_node_locals = as_long("pair_start_node_locals")
    pair_answer_node_locals = as_long("pair_answer_node_locals")
    pair_edge_local_ids = as_long("pair_edge_local_ids")
    pair_edge_ptr = as_long("pair_edge_ptr")

    for name, locals_tensor in (("start_node_locals", start_node_locals), ("answer_node_locals", answer_node_locals)):
        if locals_tensor.numel() > 0 and ((locals_tensor < 0).any() or (locals_tensor >= num_nodes).any()):
            raise ValueError(f"{name} out of range for {record.get('sample_id')}")

    if pair_start_node_locals.numel() != pair_answer_node_locals.numel():
        raise ValueError(
            f"pair_start_node_locals length {pair_start_node_locals.numel()} != "
            f"pair_answer_node_locals length {pair_answer_node_locals.numel()} for {record.get('sample_id')}"
        )
    for name, locals_tensor in (
        ("pair_start_node_locals", pair_start_node_locals),
        ("pair_answer_node_locals", pair_answer_node_locals),
    ):
        if locals_tensor.numel() > 0 and ((locals_tensor < 0).any() or (locals_tensor >= num_nodes).any()):
            raise ValueError(f"{name} out of range for {record.get('sample_id')}")
    if pair_edge_local_ids.numel() > 0:
        if ((pair_edge_local_ids < 0) | (pair_edge_local_ids >= num_edges)).any():
            raise ValueError(f"pair_edge_local_ids out of range for {record.get('sample_id')}")
    if pair_edge_ptr.numel() == 0:
        if pair_start_node_locals.numel() != 0:
            raise ValueError(f"pair_edge_ptr empty but pairs exist for {record.get('sample_id')}")
    else:
        if pair_edge_ptr.numel() != pair_start_node_locals.numel() + 1:
            raise ValueError(
                f"pair_edge_ptr length {pair_edge_ptr.numel()} != pair_count+1 "
                f"({pair_start_node_locals.numel() + 1}) for {record.get('sample_id')}"
            )
        if pair_edge_ptr[0].item() != 0:
            raise ValueError(f"pair_edge_ptr must start at 0 for {record.get('sample_id')}")
        if (pair_edge_ptr[1:] < pair_edge_ptr[:-1]).any():
            raise ValueError(f"pair_edge_ptr must be non-decreasing for {record.get('sample_id')}")
        if pair_edge_ptr[-1].item() != pair_edge_local_ids.numel():
            raise ValueError(
                f"pair_edge_ptr end {int(pair_edge_ptr[-1].item())} != "
                f"pair_edge_local_ids length {pair_edge_local_ids.numel()} for {record.get('sample_id')}"
            )

    gt_path_edge_local_ids = as_long("gt_path_edge_local_ids")
    gt_path_node_local_ids = as_long("gt_path_node_local_ids")
    if gt_path_edge_local_ids.numel() > 0:
        if ((gt_path_edge_local_ids < 0) | (gt_path_edge_local_ids >= num_edges)).any():
            raise ValueError(f"gt_path_edge_local_ids out of range for {record.get('sample_id')}")
    if gt_path_node_local_ids.numel() > 0:
        if ((gt_path_node_local_ids < 0) | (gt_path_node_local_ids >= num_nodes)).any():
            raise ValueError(f"gt_path_node_local_ids out of range for {record.get('sample_id')}")
    gt_path_exists = bool(record["gt_path_exists"])
    if gt_path_exists:
        if gt_path_edge_local_ids.numel() == 0:
            if gt_path_node_local_ids.numel() != 1:
                raise ValueError(f"gt_path zero-length but node path size != 1 for {record.get('sample_id')}")
            if start_node_locals.numel() > 0:
                if not bool((start_node_locals == gt_path_node_local_ids[0]).any().item()):
                    raise ValueError(f"gt_path does not start from start_node_locals for {record.get('sample_id')}")
            if answer_node_locals.numel() > 0:
                if not bool((answer_node_locals == gt_path_node_local_ids[-1]).any().item()):
                    raise ValueError(f"gt_path terminal not in answer_node_locals for {record.get('sample_id')}")
        else:
            if gt_path_node_local_ids.numel() == 0:
                raise ValueError(f"gt_path_exists=True but gt_path_node_local_ids is empty for {record.get('sample_id')}")
            if gt_path_node_local_ids.numel() != gt_path_edge_local_ids.numel() + 1:
                raise ValueError(
                    f"gt_path_node_local_ids length {gt_path_node_local_ids.numel()} != "
                    f"gt_path_edge_local_ids+1 ({gt_path_edge_local_ids.numel() + 1}) for {record.get('sample_id')}"
                )
            if start_node_locals.numel() > 0:
                if not bool((start_node_locals == gt_path_node_local_ids[0]).any().item()):
                    raise ValueError(f"gt_path does not start from start_node_locals for {record.get('sample_id')}")
            if answer_node_locals.numel() > 0:
                if not bool((answer_node_locals == gt_path_node_local_ids[-1]).any().item()):
                    raise ValueError(f"gt_path terminal not in answer_node_locals for {record.get('sample_id')}")
            for idx, edge_id in enumerate(gt_path_edge_local_ids.tolist()):
                h = int(edge_head_locals[edge_id].item())
                t = int(edge_tail_locals[edge_id].item())
                a = int(gt_path_node_local_ids[idx].item())
                b = int(gt_path_node_local_ids[idx + 1].item())
                if not ((a == h and b == t) or (a == t and b == h)):
                    raise ValueError(f"gt_path edge/node mismatch at step {idx} for {record.get('sample_id')}")
            if gt_path_edge_local_ids.numel() > 0:
                if not bool((edge_labels[gt_path_edge_local_ids] > 0.5).all().item()):
                    raise ValueError(f"gt_path edges must be subset of edge_labels for {record.get('sample_id')}")
    elif gt_path_edge_local_ids.numel() > 0 or gt_path_node_local_ids.numel() > 0:
        raise ValueError(f"gt_path_exists=False but gt_path_* is non-empty for {record.get('sample_id')}")

    computed_reachable = bool(pair_start_node_locals.numel() > 0)
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
        node_embedding_ids=node_embedding_ids,
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
        pair_start_node_locals=pair_start_node_locals,
        pair_answer_node_locals=pair_answer_node_locals,
        pair_edge_local_ids=pair_edge_local_ids,
        pair_edge_ptr=pair_edge_ptr,
        gt_path_edge_local_ids=gt_path_edge_local_ids,
        gt_path_node_local_ids=gt_path_node_local_ids,
        gt_path_exists=gt_path_exists,
        is_answer_reachable=declared_reachable,
    )
    if sample.question_emb.dim() == 1:
        sample.question_emb = sample.question_emb.unsqueeze(0)
    if sample.start_entity_ids.numel() == 0:
        raise ValueError(f"g_agent record missing non-empty start_entity_ids: {record.get('sample_id')}")
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
    if isinstance(payload, dict):
        raw_samples: Sequence[Dict] = payload.get("samples") or []
    elif isinstance(payload, list):
        raw_samples = payload
    else:
        raise TypeError(f"Unsupported g_agent cache format at {cache_path}: {type(payload)}")
    if not raw_samples:
        raise ValueError(f"No samples found in {cache_path}")

    samples: List[GAgentSample] = []
    dropped = 0
    for record in raw_samples:
        if isinstance(record, GAgentSample):
            sample = record
        else:
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
        if key in {
            "start_node_locals",
            "answer_node_locals",
            "gt_path_node_local_ids",
            "pair_start_node_locals",
            "pair_answer_node_locals",
        }:
            return int(self.num_nodes)
        if key in {"gt_path_edge_local_ids", "pair_edge_local_ids"}:
            return int(self.edge_index.size(1))
        if key == "pair_edge_ptr":
            return int(self.pair_edge_local_ids.numel())
        return super().__inc__(key, value, *args, **kwargs)


def _sample_to_pyg_data(sample: GAgentSample) -> GAgentData:
    edge_index = torch.stack([sample.edge_head_locals, sample.edge_tail_locals], dim=0)
    question_emb = sample.question_emb
    if question_emb.dim() == 1:
        question_emb = question_emb.unsqueeze(0)
    start_node_locals = sample.start_node_locals
    data = GAgentData(
        edge_index=edge_index,
        edge_attr=sample.edge_relations,
        edge_scores=sample.edge_scores,
        edge_labels=sample.edge_labels,
        node_global_ids=sample.node_entity_ids,
        node_embedding_ids=sample.node_embedding_ids,
        question_emb=question_emb,
        start_node_locals=start_node_locals,
        answer_node_locals=sample.answer_node_locals,
        start_entity_ids=sample.start_entity_ids,
        answer_entity_ids=sample.answer_entity_ids,
        pair_start_node_locals=sample.pair_start_node_locals,
        pair_answer_node_locals=sample.pair_answer_node_locals,
        pair_edge_local_ids=sample.pair_edge_local_ids,
        pair_edge_ptr=sample.pair_edge_ptr,
        gt_path_edge_local_ids=sample.gt_path_edge_local_ids,
        gt_path_node_local_ids=sample.gt_path_node_local_ids,
        gt_path_exists=torch.tensor([sample.gt_path_exists], dtype=torch.bool),
        is_answer_reachable=torch.tensor([sample.is_answer_reachable], dtype=torch.bool),
        sample_id=sample.sample_id,
        question=sample.question,
    )
    data.num_nodes = int(sample.node_entity_ids.numel())
    # 强制 PyG 校验，防止批内 offset 错乱
    data.validate(raise_on_error=True)
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
        top_mask_src = getattr(batch, "top_edge_mask", None)
        if top_mask_src is None:
            top_edge_mask[g, :e_count] = True
        else:
            top_edge_mask[g, :e_count] = top_mask_src[edge_slice].to(device=device, dtype=torch.bool)

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
        prefer_jsonl: bool = True,
        convert_pt_to_jsonl: bool = False,
    ) -> None:
        super().__init__()
        path = Path(cache_path)
        self.drop_unreachable = drop_unreachable
        target = path
        if path.suffix.lower() == ".pt" and convert_pt_to_jsonl:
            target = self._convert_pt_to_jsonl(path)
        elif prefer_jsonl:
            candidate = path.with_suffix(".jsonl")
            if candidate.exists():
                target = candidate
        self._is_jsonl = target.suffix.lower() == ".jsonl"
        self.cache_path = target
        if self._is_jsonl:
            self._offsets = self._build_index(self.cache_path)
            self.samples = None
        else:
            if convert_pt_to_jsonl:
                converted = self._convert_pt_to_jsonl(path)
                self.cache_path = converted
                self._is_jsonl = True
                self._offsets = self._build_index(converted)
                self.samples = None
            else:
                self.samples = load_g_agent_samples(path, drop_unreachable=drop_unreachable)
            self._offsets = []

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return len(self._offsets)

    def __getitem__(self, idx: int) -> GAgentData:
        if self.samples is not None:
            return _sample_to_pyg_data(self.samples[idx])
        record = self._read_jsonl_record(idx)
        sample = _parse_sample(record)
        if self.drop_unreachable and not sample.is_answer_reachable:
            # 线性探测下一个样本，确保 DataLoader 不返回空
            return self.__getitem__((idx + 1) % len(self))
        return _sample_to_pyg_data(sample)

    def _build_index(self, path: Path) -> List[int]:
        offsets: List[int] = []
        with path.open("rb") as f:
            offset = 0
            for line in f:
                offsets.append(offset)
                offset += len(line)
        return offsets

    def _read_jsonl_record(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self._offsets):
            raise IndexError(idx)
        with self.cache_path.open("rb") as f:
            f.seek(self._offsets[idx])
            line = f.readline()
        if not line:
            raise ValueError(f"Empty line at idx={idx}")
        return json.loads(line)

    def _convert_pt_to_jsonl(self, path: Path) -> Path:
        samples = load_g_agent_samples(path, drop_unreachable=self.drop_unreachable)
        out = path.with_suffix(".jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for sample in samples:
                rec = _builder_sample_to_record(sample) if isinstance(sample, GAgentSample) else sample
                f.write(json.dumps(rec) + "\n")
        # 释放内存
        del samples
        return out


__all__ = [
    "GAgentSample",
    "GAgentData",
    "GAgentPyGDataset",
    "load_g_agent_samples",
    "_builder_sample_to_record",
    "_sample_to_pyg_data",
]
