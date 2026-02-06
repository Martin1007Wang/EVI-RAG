from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pyarrow.parquet as pq

from src.data.context import StageContext
from src.data.io.lmdb_utils import ensure_dir
from src.data.schema.constants import EntityVocabFields, GraphFields, RelationVocabFields
from src.data.utils.inverse_relations_llm import generate_inverse_relations_llm
from src.utils.logging_utils import log_event


@dataclass(frozen=True)
class _RelationStats:
    top_candidates: List[Tuple[str, int]]
    total: int


def _dedup_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix:
        return ""
    return prefix if prefix.endswith("/") else f"{prefix}/"


def _build_inverse_relation_key(rel: str, *, prefix: str, suffix: str) -> str:
    if prefix:
        return f"{prefix}{rel}"
    if not suffix:
        raise ValueError("inverse_relations.kg_id_suffix must be non-empty.")
    return f"{rel}{suffix}"


def _is_generated_inverse_relation(kg_id: str, *, prefix: str, suffix: str) -> bool:
    if prefix and kg_id.startswith(prefix):
        return True
    if suffix and kg_id.endswith(suffix):
        return True
    return False


def _extract_json_payload(text: str) -> Any:
    if not text:
        raise ValueError("empty LLM output")
    s = text.strip()
    if s.startswith("{") or s.startswith("["):
        return json.loads(s)
    start = s.find("[")
    end = s.rfind("]")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return json.loads(s[start : end + 1])
    raise ValueError("no JSON object found in LLM output")


def _load_relation_vocab(path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    table = pq.read_table(path, columns=[RelationVocabFields.RELATION_ID, RelationVocabFields.KG_ID, RelationVocabFields.LABEL])
    rel_ids = table.column(RelationVocabFields.RELATION_ID).to_pylist()
    kg_ids = table.column(RelationVocabFields.KG_ID).to_pylist()
    labels = table.column(RelationVocabFields.LABEL).to_pylist()
    rel_id_to_kg: Dict[int, str] = {}
    rel_id_to_label: Dict[int, str] = {}
    for rid, kg, label in zip(rel_ids, kg_ids, labels):
        if rid is None or kg is None:
            continue
        rel_id_to_kg[int(rid)] = str(kg)
        rel_id_to_label[int(rid)] = "" if label is None else str(label)
    return rel_id_to_kg, rel_id_to_label


def _load_entity_vocab(path: Path) -> Dict[int, str]:
    table = pq.read_table(path, columns=[EntityVocabFields.ENTITY_ID, EntityVocabFields.KG_ID])
    ent_ids = table.column(EntityVocabFields.ENTITY_ID).to_pylist()
    kg_ids = table.column(EntityVocabFields.KG_ID).to_pylist()
    mapping: Dict[int, str] = {}
    for eid, kg in zip(ent_ids, kg_ids):
        if eid is None or kg is None:
            continue
        mapping[int(eid)] = str(kg)
    return mapping


def _iter_graph_rows(path: Path) -> Iterable[Tuple[List[int], List[int], List[int], List[int]]]:
    pf = pq.ParquetFile(path)
    columns = [GraphFields.NODE_IDS, GraphFields.EDGE_SRC, GraphFields.EDGE_DST, GraphFields.EDGE_REL_IDS]
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=columns)
        node_ids_col = table.column(GraphFields.NODE_IDS).to_pylist()
        edge_src_col = table.column(GraphFields.EDGE_SRC).to_pylist()
        edge_dst_col = table.column(GraphFields.EDGE_DST).to_pylist()
        edge_rel_col = table.column(GraphFields.EDGE_REL_IDS).to_pylist()
        for node_ids, edge_src, edge_dst, edge_rel in zip(node_ids_col, edge_src_col, edge_dst_col, edge_rel_col):
            yield (
                list(node_ids or []),
                list(edge_src or []),
                list(edge_dst or []),
                list(edge_rel or []),
            )


def _collect_inverse_candidates(
    graphs_path: Path,
    *,
    ignore_rel_ids: set[int],
    sample_per_rel: int,
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, List[Tuple[int, int]]]]:
    counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    samples: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for node_ids, edge_src, edge_dst, edge_rel in _iter_graph_rows(graphs_path):
        pair_to_rels: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for src, dst, rel in zip(edge_src, edge_dst, edge_rel):
            rel_id = int(rel)
            if rel_id in ignore_rel_ids:
                continue
            pair_to_rels[(int(src), int(dst))].append(rel_id)
        for src, dst, rel in zip(edge_src, edge_dst, edge_rel):
            rel_id = int(rel)
            if rel_id in ignore_rel_ids:
                continue
            rev = pair_to_rels.get((int(dst), int(src)))
            if rev:
                for rev_rel in rev:
                    if rev_rel in ignore_rel_ids:
                        continue
                    counts[rel_id][rev_rel] += 1
            if sample_per_rel > 0 and len(samples[rel_id]) < sample_per_rel:
                head_gid = node_ids[int(src)] if int(src) < len(node_ids) else None
                tail_gid = node_ids[int(dst)] if int(dst) < len(node_ids) else None
                if head_gid is not None and tail_gid is not None:
                    samples[rel_id].append((int(head_gid), int(tail_gid)))
    return counts, samples


def _build_candidate_stats(
    counts: Dict[int, Dict[int, int]],
    *,
    rel_id_to_kg: Dict[int, str],
    topk: int,
) -> Dict[str, _RelationStats]:
    stats: Dict[str, _RelationStats] = {}
    for rel_id, cand_counts in counts.items():
        if rel_id not in rel_id_to_kg:
            continue
        total = sum(cand_counts.values())
        top = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)[:topk]
        top_named = [(rel_id_to_kg[rid], int(cnt)) for rid, cnt in top if rid in rel_id_to_kg]
        stats[rel_id_to_kg[rel_id]] = _RelationStats(top_candidates=top_named, total=int(total))
    return stats


def _resolve_mutual_pairs(
    stats: Dict[str, _RelationStats],
    *,
    min_support: int,
    min_ratio: float,
) -> Dict[str, str]:
    top_choice: Dict[str, Optional[str]] = {}
    for rel, stat in stats.items():
        if not stat.top_candidates:
            top_choice[rel] = None
            continue
        top_rel, top_cnt = stat.top_candidates[0]
        ratio = float(top_cnt) / float(stat.total) if stat.total else 0.0
        if top_cnt >= min_support and ratio >= min_ratio:
            top_choice[rel] = top_rel
        else:
            top_choice[rel] = None
    inverse_map: Dict[str, str] = {}
    for rel, cand in top_choice.items():
        if cand is None:
            continue
        if top_choice.get(cand) == rel:
            inverse_map[rel] = cand
    return inverse_map


def _format_samples(
    samples: List[Tuple[int, int]],
    *,
    entity_id_to_kg: Optional[Dict[int, str]],
    limit: int,
) -> List[str]:
    out: List[str] = []
    for head_id, tail_id in samples[:limit]:
        if entity_id_to_kg:
            head = entity_id_to_kg.get(head_id, str(head_id))
            tail = entity_id_to_kg.get(tail_id, str(tail_id))
        else:
            head = str(head_id)
            tail = str(tail_id)
        out.append(f"{head} -> {tail}")
    return out


def _resolve_with_llm(
    unresolved: List[str],
    *,
    stats: Dict[str, _RelationStats],
    samples_by_rel: Dict[str, List[Tuple[int, int]]],
    entity_id_to_kg: Optional[Dict[int, str]],
    llm_cfg: Mapping[str, Any],
) -> Dict[str, Dict[str, str]]:
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("vllm is required for inverse relation resolution.") from exc

    model = llm_cfg.get("model")
    if not model:
        raise ValueError("inverse_relations.llm.model must be set for LLM resolution.")
    temperature = float(llm_cfg.get("temperature", 0.0))
    top_p = float(llm_cfg.get("top_p", 1.0))
    max_tokens = int(llm_cfg.get("max_tokens", 128))
    tensor_parallel_size = int(llm_cfg.get("tensor_parallel_size", 1))
    max_model_len = llm_cfg.get("max_model_len")
    dtype = llm_cfg.get("dtype")
    trust_remote_code = bool(llm_cfg.get("trust_remote_code", False))

    llm_kwargs = {
        "model": str(model),
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = int(max_model_len)
    if dtype not in (None, "", "null", "None"):
        llm_kwargs["dtype"] = dtype
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    system = (
        "You are a knowledge graph relation expert. "
        "Select the best inverse relation among candidates or say NONE."
    )
    outputs: Dict[str, Dict[str, str]] = {}
    for rel in unresolved:
        stat = stats.get(rel)
        candidates = [] if stat is None else [c for c, _ in stat.top_candidates]
        sample_lines = _format_samples(samples_by_rel.get(rel, []), entity_id_to_kg=entity_id_to_kg, limit=3)
        candidate_sections: List[str] = []
        for cand in candidates:
            cand_samples = _format_samples(samples_by_rel.get(cand, []), entity_id_to_kg=entity_id_to_kg, limit=2)
            if cand_samples:
                cand_block = "\n".join(f"  - {line}" for line in cand_samples)
                candidate_sections.append(f"{cand} examples:\n{cand_block}")
            else:
                candidate_sections.append(f"{cand} examples:\n  - (no examples)")
        prompt = (
            "Return JSON only: "
            "{\"forward\":\"<rel>\",\"inverse_relation\":\"<candidate or NONE>\",\"inverse_label\":\"<label if NONE>\"}\n"
            f"Forward relation: {rel}\n"
            "Examples (head -> tail):\n"
            + ("\n".join(f"- {line}" for line in sample_lines) if sample_lines else "- (no examples)")
            + "\nCandidates:\n"
            + ("\n".join(f"- {cand}" for cand in candidates) if candidates else "- (none)")
            + ("\nCandidate examples:\n" + "\n".join(candidate_sections) if candidate_sections else "")
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        resp = llm.chat([messages], sampling_params=sampling, use_tqdm=False)
        text = resp[0].outputs[0].text if resp and resp[0].outputs else ""
        payload = _extract_json_payload(text)
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            raise ValueError(f"LLM returned non-JSON for relation {rel}: {text[:200]}")
        inverse_rel = str(payload.get("inverse_relation") or "").strip()
        inverse_label = str(payload.get("inverse_label") or "").strip()
        outputs[rel] = {
            "inverse_relation": inverse_rel,
            "inverse_label": inverse_label,
        }
    return outputs


def _load_entries(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    inner = payload.get("inverse_relations", payload)
    entries: Dict[str, Dict[str, Any]] = {}
    if isinstance(inner, list):
        for item in inner:
            if isinstance(item, dict) and item.get("forward"):
                entries[str(item["forward"])] = dict(item)
    return entries


def _write_entries(path: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    ordered = [entries[key] for key in sorted(entries.keys())]
    payload = {"inverse_relations": ordered}
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_inverse_relations_detect(ctx: StageContext) -> Path:
    cfg = ctx.cfg
    inv_cfg = cfg.get("inverse_relations") if hasattr(cfg, "get") else None
    if not inv_cfg or not hasattr(inv_cfg, "get"):
        raise ValueError("inverse_relations config missing.")
    build_cfg = inv_cfg.get("build") or {}
    graphs_path = ctx.parquet_dir / "graphs.parquet"
    relation_vocab_path = ctx.parquet_dir / "relation_vocab.parquet"
    if not graphs_path.exists() or not relation_vocab_path.exists():
        raise FileNotFoundError("graphs.parquet and relation_vocab.parquet are required for inverse_relations detect.")
    detect_path = ctx.resolve_path(build_cfg.get("detect_path", ctx.parquet_dir / "inverse_relations.detect.json"))
    prefix = _normalize_prefix(inv_cfg.get("kg_id_prefix", ""))
    suffix = str(inv_cfg.get("kg_id_suffix", "__inv"))
    min_support = int(build_cfg.get("min_support", 5))
    min_ratio = float(build_cfg.get("min_ratio", 0.9))
    topk = int(build_cfg.get("topk_candidates", 3))
    sample_per_rel = int(build_cfg.get("sample_per_relation", 5))

    rel_id_to_kg, rel_id_to_label = _load_relation_vocab(relation_vocab_path)
    ignore_rel_ids = {
        rid for rid, kg in rel_id_to_kg.items() if _is_generated_inverse_relation(kg, prefix=prefix, suffix=suffix)
    }
    counts, samples = _collect_inverse_candidates(
        graphs_path,
        ignore_rel_ids=ignore_rel_ids,
        sample_per_rel=sample_per_rel,
    )
    stats = _build_candidate_stats(counts, rel_id_to_kg=rel_id_to_kg, topk=topk)
    inverse_map = _resolve_mutual_pairs(stats, min_support=min_support, min_ratio=min_ratio)

    entries: Dict[str, Dict[str, Any]] = {}
    for rel, inv_rel in inverse_map.items():
        entry = {"forward": rel, "inverse_relation": inv_rel}
        label = rel_id_to_label.get(next((rid for rid, kg in rel_id_to_kg.items() if kg == rel), -1))
        if label:
            entry["forward_label"] = label
        entries[rel] = entry

    _write_entries(detect_path, entries)
    log_event(ctx.logger, "inverse_relations_detect_done", path=str(detect_path), count=len(entries))
    return detect_path


def build_inverse_relations_resolve(ctx: StageContext) -> Path:
    cfg = ctx.cfg
    inv_cfg = cfg.get("inverse_relations") if hasattr(cfg, "get") else None
    if not inv_cfg or not hasattr(inv_cfg, "get"):
        raise ValueError("inverse_relations config missing.")
    build_cfg = inv_cfg.get("build") or {}
    detect_path = ctx.resolve_path(build_cfg.get("detect_path", ctx.parquet_dir / "inverse_relations.detect.json"))
    resolved_path = ctx.resolve_path(build_cfg.get("resolved_path", ctx.parquet_dir / "inverse_relations.resolved.json"))
    resolve_with_llm = bool(build_cfg.get("resolve_with_llm", False))
    graphs_path = ctx.parquet_dir / "graphs.parquet"
    relation_vocab_path = ctx.parquet_dir / "relation_vocab.parquet"
    entity_vocab_path = ctx.parquet_dir / "entity_vocab.parquet"
    if not detect_path.exists():
        raise FileNotFoundError(f"Missing detect mapping: {detect_path}")
    if not graphs_path.exists() or not relation_vocab_path.exists():
        raise FileNotFoundError("graphs.parquet and relation_vocab.parquet are required for inverse_relations resolve.")

    entries = _load_entries(detect_path)
    if resolve_with_llm:
        rel_id_to_kg, _ = _load_relation_vocab(relation_vocab_path)
        prefix = _normalize_prefix(inv_cfg.get("kg_id_prefix", ""))
        suffix = str(inv_cfg.get("kg_id_suffix", "__inv"))
        ignore_rel_ids = {
            rid for rid, kg in rel_id_to_kg.items() if _is_generated_inverse_relation(kg, prefix=prefix, suffix=suffix)
        }
        sample_per_rel = int(build_cfg.get("sample_per_relation", 5))
        topk = int(build_cfg.get("topk_candidates", 3))

        counts, samples = _collect_inverse_candidates(
            graphs_path,
            ignore_rel_ids=ignore_rel_ids,
            sample_per_rel=sample_per_rel,
        )
        stats = _build_candidate_stats(counts, rel_id_to_kg=rel_id_to_kg, topk=topk)
        samples_by_rel: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for rel_id, pairs in samples.items():
            kg = rel_id_to_kg.get(rel_id)
            if kg:
                samples_by_rel[kg].extend(pairs)

        unresolved = [rel for rel in stats.keys() if not entries.get(rel, {}).get("inverse_relation")]

        entity_id_to_kg = _load_entity_vocab(entity_vocab_path) if entity_vocab_path.exists() else None
        llm_cfg = dict(inv_cfg.get("llm") or {})
        outputs = _resolve_with_llm(
            unresolved,
            stats=stats,
            samples_by_rel=samples_by_rel,
            entity_id_to_kg=entity_id_to_kg,
            llm_cfg=llm_cfg,
        )
        for rel, result in outputs.items():
            entry = entries.get(rel, {"forward": rel})
            inv_rel = result.get("inverse_relation", "")
            inv_label = result.get("inverse_label", "")
            if inv_rel and inv_rel.upper() != "NONE":
                entry["inverse_relation"] = inv_rel
            if inv_label:
                entry["inverse"] = inv_label
            entries[rel] = entry

    _write_entries(resolved_path, entries)
    log_event(
        ctx.logger,
        "inverse_relations_resolve_done",
        path=str(resolved_path),
        count=len(entries),
        mode="llm" if resolve_with_llm else "copy",
    )
    return resolved_path


def build_inverse_relations_describe(ctx: StageContext) -> Path:
    cfg = ctx.cfg
    inv_cfg = cfg.get("inverse_relations") if hasattr(cfg, "get") else None
    if not inv_cfg or not hasattr(inv_cfg, "get"):
        raise ValueError("inverse_relations config missing.")
    build_cfg = inv_cfg.get("build") or {}
    resolved_path = ctx.resolve_path(build_cfg.get("resolved_path", ctx.parquet_dir / "inverse_relations.resolved.json"))
    mapping_path = ctx.resolve_path(inv_cfg.get("mapping_path", ctx.parquet_dir / "inverse_relations.json"))
    relation_vocab_path = ctx.parquet_dir / "relation_vocab.parquet"
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing resolved mapping: {resolved_path}")
    if not relation_vocab_path.exists():
        raise FileNotFoundError("relation_vocab.parquet is required for inverse_relations describe.")

    rel_id_to_kg, rel_id_to_label = _load_relation_vocab(relation_vocab_path)
    entries = _load_entries(resolved_path)
    prefix = _normalize_prefix(inv_cfg.get("kg_id_prefix", ""))
    suffix = str(inv_cfg.get("kg_id_suffix", "__inv"))
    llm_cfg = dict(inv_cfg.get("llm") or {})
    rel_list = _dedup_preserve_order(
        [kg for kg in rel_id_to_kg.values() if not _is_generated_inverse_relation(kg, prefix=prefix, suffix=suffix)]
    )
    graphs_path = ctx.parquet_dir / "graphs.parquet"
    examples_by_relation: Dict[str, List[str]] = {}
    if graphs_path.exists():
        ignore_rel_ids = {
            rid for rid, kg in rel_id_to_kg.items() if _is_generated_inverse_relation(kg, prefix=prefix, suffix=suffix)
        }
        sample_per_rel = int(build_cfg.get("sample_per_relation", 5))
        _, samples = _collect_inverse_candidates(
            graphs_path,
            ignore_rel_ids=ignore_rel_ids,
            sample_per_rel=sample_per_rel,
        )
        entity_vocab_path = ctx.parquet_dir / "entity_vocab.parquet"
        entity_id_to_kg = _load_entity_vocab(entity_vocab_path) if entity_vocab_path.exists() else None
        raw_example_limit = llm_cfg.get("example_limit", 3)
        example_limit = 3 if raw_example_limit is None else int(raw_example_limit)
        for rel_id, pairs in samples.items():
            kg = rel_id_to_kg.get(rel_id)
            if not kg:
                continue
            examples_by_relation[kg] = _format_samples(
                pairs,
                entity_id_to_kg=entity_id_to_kg,
                limit=example_limit,
            )
    desc_entries = generate_inverse_relations_llm(
        relations=rel_list,
        llm_cfg=llm_cfg,
        examples_by_relation=examples_by_relation or None,
    )
    desc_map = {desc["forward"]: desc for desc in desc_entries}
    forward_labels = {
        rel: desc_map.get(rel, {}).get("forward_label") or rel for rel in rel_list
    }
    forward_texts = {
        rel: desc_map.get(rel, {}).get("forward_text") or "" for rel in rel_list
    }
    inverse_labels = {
        rel: desc_map.get(rel, {}).get("inverse") or "" for rel in rel_list
    }
    inverse_texts = {
        rel: desc_map.get(rel, {}).get("inverse_text") or "" for rel in rel_list
    }
    forward_set = set(rel_list)
    for rel in rel_list:
        entry = entries.get(rel, {"forward": rel})
        entry["forward_label"] = forward_labels.get(rel, rel)
        entry["forward_text"] = forward_texts.get(rel, "")
        inv_rel = entry.get("inverse_relation")
        inv_rel = str(inv_rel) if inv_rel else ""
        if inv_rel == rel:
            entry.pop("inverse_relation", None)
            inv_rel = ""
        if not inv_rel:
            inv_rel = _build_inverse_relation_key(rel, prefix=prefix, suffix=suffix)
            if inv_rel in forward_set:
                raise ValueError(f"inverse_relations generated id collides with forward relation: {inv_rel!r}.")
            entry["inverse_relation"] = inv_rel
        if inv_rel in forward_set:
            back = entries.get(inv_rel, {}).get("inverse_relation")
            if back is None:
                raise ValueError(
                    f"inverse_relations missing mutual mapping for {rel!r} -> {inv_rel!r}."
                )
            if str(back) != rel:
                raise ValueError(
                    f"inverse_relations non-mutual mapping: {rel!r} -> {inv_rel!r} but {inv_rel!r} -> {back!r}."
                )
            entry["inverse"] = forward_labels.get(inv_rel, inv_rel)
            entry["inverse_text"] = forward_texts.get(inv_rel, "")
        else:
            inv_label = inverse_labels.get(rel, "") or entry.get("inverse", "")
            if not inv_label:
                raise ValueError(f"inverse_relations missing inverse label for {rel!r}.")
            entry["inverse"] = inv_label
            entry["inverse_text"] = inverse_texts.get(rel, "") or entry.get("inverse_text", "")
        entries[rel] = entry
    for rel in entries:
        if "forward_label" not in entries[rel]:
            for rid, kg in rel_id_to_kg.items():
                if kg == rel:
                    label = rel_id_to_label.get(rid)
                    if label:
                        entries[rel]["forward_label"] = label
                    break

    _write_entries(mapping_path, entries)
    log_event(ctx.logger, "inverse_relations_describe_done", path=str(mapping_path), count=len(entries))
    return mapping_path


def build_inverse_relations_all(ctx: StageContext) -> Path:
    detect_path = build_inverse_relations_detect(ctx)
    resolve_path = build_inverse_relations_resolve(ctx)
    final_path = build_inverse_relations_describe(ctx)
    log_event(
        ctx.logger,
        "inverse_relations_all_done",
        detect_path=str(detect_path),
        resolve_path=str(resolve_path),
        mapping_path=str(final_path),
    )
    return final_path
