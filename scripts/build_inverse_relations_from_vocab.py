from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from src.data.schema.constants import RelationVocabFields
from src.data.utils.inverse_relations_llm import generate_inverse_relations_llm
from src.data.io.lmdb_utils import ensure_dir


def _dedup_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _load_llm_cfg(cfg_path: Path, overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = OmegaConf.load(cfg_path)
    inv_cfg = cfg.get("inverse_relations") if cfg is not None else None
    llm_cfg = {}
    if inv_cfg and hasattr(inv_cfg, "get"):
        llm_cfg = dict(inv_cfg.get("llm") or {})
    llm_cfg.update({k: v for k, v in overrides.items() if v is not None})
    return llm_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inverse_relations.json from relation_vocab.parquet.")
    parser.add_argument("--relation-vocab", required=True, help="Path to relation_vocab.parquet")
    parser.add_argument("--output", default="inverse_relations.json", help="Output JSON path")
    parser.add_argument("--config", default="configs/pipeline/default.yaml", help="Pipeline config for LLM defaults")
    parser.add_argument("--model", default=None, help="Override LLM model id")
    parser.add_argument("--batch-size", type=int, default=None, help="Override LLM batch size")
    parser.add_argument("--temperature", type=float, default=None, help="Override LLM temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Override LLM top_p")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--max-retries", type=int, default=None, help="Override max retries")
    parser.add_argument("--drop-suffix", default="__inv", help="Drop relations ending with this suffix")
    args = parser.parse_args()

    vocab_path = Path(args.relation_vocab)
    table = pq.read_table(vocab_path)
    kg_ids = table.column(RelationVocabFields.KG_ID).to_pylist()
    relations = [str(r) for r in kg_ids if r is not None and str(r).strip()]
    if args.drop_suffix:
        relations = [r for r in relations if not r.endswith(args.drop_suffix)]
    relations = _dedup_preserve_order(relations)

    llm_cfg = _load_llm_cfg(
        Path(args.config),
        {
            "model": args.model,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_retries": args.max_retries,
        },
    )
    entries = generate_inverse_relations_llm(relations=relations, llm_cfg=llm_cfg)
    payload = {"inverse_relations": entries}
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote {len(entries)} relations to {output_path}")


if __name__ == "__main__":
    main()
