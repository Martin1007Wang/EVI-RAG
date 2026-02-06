from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from src.data.io.lmdb_utils import ensure_dir
from src.data.schema.constants import GraphFields, RelationVocabFields, EntityVocabFields


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


def _load_llm_cfg(cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = OmegaConf.load(cfg_path)
    inv_cfg = cfg.get("inverse_relations") if cfg is not None else None
    llm_cfg = {}
    if inv_cfg and hasattr(inv_cfg, "get"):
        llm_cfg = dict(inv_cfg.get("llm") or {})
    llm_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return llm_cfg


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
    rel_id_to_kg: Dict[int, str],
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inverse relation mapping from graphs + optional LLM.")
    parser.add_argument("--graphs", help="graphs.parquet path (required for detect/resolve)")
    parser.add_argument("--relation-vocab", required=True, help="relation_vocab.parquet path")
    parser.add_argument("--entity-vocab", help="entity_vocab.parquet path (for LLM examples)")
    parser.add_argument("--input", help="existing inverse_relations.json to extend")
    parser.add_argument("--output", required=True, help="output inverse_relations.json")
    parser.add_argument("--config", default="configs/pipeline/default.yaml", help="pipeline config for LLM defaults")
    parser.add_argument("--model", default=None, help="override LLM model id")
    parser.add_argument("--batch-size", type=int, default=None, help="override LLM batch size (labels)")
    parser.add_argument("--temperature", type=float, default=None, help="override LLM temperature")
    parser.add_argument("--top-p", type=float, default=None, help="override LLM top_p")
    parser.add_argument("--max-tokens", type=int, default=None, help="override max tokens")
    parser.add_argument("--max-retries", type=int, default=None, help="override max retries")
    parser.add_argument("--suffix", default="__inv", help="suffix for generated inverse relations")
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-ratio", type=float, default=0.9)
    parser.add_argument("--topk-candidates", type=int, default=3)
    parser.add_argument("--sample-per-relation", type=int, default=5)
    parser.add_argument("--resolve-llm", action="store_true", help="use LLM to resolve ambiguous inverses")
    parser.add_argument("--describe-llm", action="store_true", help="use LLM to generate labels/texts")
    args = parser.parse_args()

    rel_id_to_kg, rel_id_to_label = _load_relation_vocab(Path(args.relation_vocab))
    kg_id_to_label = {kg: rel_id_to_label[rid] for rid, kg in rel_id_to_kg.items() if rel_id_to_label.get(rid)}
    ignore_rel_ids = {rid for rid, kg in rel_id_to_kg.items() if kg.endswith(args.suffix)}

    mapping_entries: Dict[str, Dict[str, Any]] = {}
    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        inner = payload.get("inverse_relations", payload)
        if isinstance(inner, list):
            for item in inner:
                if isinstance(item, dict) and item.get("forward"):
                    mapping_entries[str(item["forward"])] = dict(item)

    stats: Dict[str, _RelationStats] = {}
    samples_by_rel: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    if args.graphs:
        counts, samples = _collect_inverse_candidates(
            Path(args.graphs),
            rel_id_to_kg=rel_id_to_kg,
            ignore_rel_ids=ignore_rel_ids,
            sample_per_rel=args.sample_per_relation,
        )
        stats = _build_candidate_stats(counts, rel_id_to_kg=rel_id_to_kg, topk=args.topk_candidates)
        for rel_id, pairs in samples.items():
            kg = rel_id_to_kg.get(rel_id)
            if kg:
                samples_by_rel[kg].extend(pairs)
        inverse_map = _resolve_mutual_pairs(stats, min_support=args.min_support, min_ratio=args.min_ratio)
        for rel, inv_rel in inverse_map.items():
            entry = mapping_entries.get(rel, {"forward": rel})
            entry["inverse_relation"] = inv_rel
            mapping_entries[rel] = entry

    if args.resolve_llm:
        if not args.graphs:
            raise ValueError("--graphs is required for --resolve-llm")
        entity_id_to_kg = _load_entity_vocab(Path(args.entity_vocab)) if args.entity_vocab else None
        llm_cfg = _load_llm_cfg(Path(args.config), {
            "model": args.model,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
        })
        unresolved = [rel for rel in stats.keys() if not mapping_entries.get(rel, {}).get("inverse_relation")]
        llm_outputs = _resolve_with_llm(
            unresolved,
            stats=stats,
            samples_by_rel=samples_by_rel,
            entity_id_to_kg=entity_id_to_kg,
            llm_cfg=llm_cfg,
        )
        for rel, result in llm_outputs.items():
            entry = mapping_entries.get(rel, {"forward": rel})
            inv_rel = result.get("inverse_relation", "")
            inv_label = result.get("inverse_label", "")
            if inv_rel and inv_rel.upper() != "NONE":
                entry["inverse_relation"] = inv_rel
            if inv_label:
                entry["inverse"] = inv_label
            mapping_entries[rel] = entry

    if args.describe_llm:
        from src.data.utils.inverse_relations_llm import generate_inverse_relations_llm

        llm_cfg = _load_llm_cfg(Path(args.config), {
            "model": args.model,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
        })
        rel_list = _dedup_preserve_order([kg for kg in rel_id_to_kg.values() if not kg.endswith(args.suffix)])
        desc_entries = generate_inverse_relations_llm(relations=rel_list, llm_cfg=llm_cfg)
        for desc in desc_entries:
            rel = desc["forward"]
            entry = mapping_entries.get(rel, {"forward": rel})
            entry["forward_label"] = desc.get("forward_label", entry.get("forward_label", rel))
            entry["forward_text"] = desc.get("forward_text", entry.get("forward_text", ""))
            entry["inverse"] = desc.get("inverse", entry.get("inverse", ""))
            entry["inverse_text"] = desc.get("inverse_text", entry.get("inverse_text", ""))
            mapping_entries[rel] = entry

    # finalize entries
    entries = []
    for rel in sorted(mapping_entries.keys()):
        entry = mapping_entries[rel]
        entry.setdefault("forward", rel)
        if "forward_label" not in entry and rel in kg_id_to_label:
            entry["forward_label"] = kg_id_to_label[rel]
        entries.append(entry)

    payload = {"inverse_relations": entries}
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(entries)} relations to {output_path}")


if __name__ == "__main__":
    main()
