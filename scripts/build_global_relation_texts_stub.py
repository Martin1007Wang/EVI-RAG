#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Set

import rootutils

root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

try:
    import hydra
except ModuleNotFoundError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]

from tqdm import tqdm

from src.data.io.raw_loader import iter_samples
from src.data.relation_cleaning_rules import DEFAULT_RELATION_CLEANING_RULES
from src.data.schema.constants import _ALLOWED_SPLITS, _ONE, _ZERO
from src.data.stages.step1_vocab import _partition_graph_edges
from src.utils.logging_utils import get_logger, init_logging, log_event


_KEY_FORWARD = "forward"
_KEY_FORWARD_TEXT = "forward_text"
_KEY_INVERSE = "inverse"
_KEY_RELATIONS = "inverse_relations"
_FORWARD_TEXT_KG = "kg_id"
_FORWARD_TEXT_BLANK = "blank"

LOGGER = get_logger(__name__)


def _parse_dataset_list(values: Sequence[str]) -> List[str]:
    datasets: List[str] = []
    for value in values:
        for entry in str(value).split(","):
            entry = entry.strip()
            if entry:
                datasets.append(entry)
    if not datasets:
        raise ValueError("At least one dataset must be provided.")
    return datasets


def _compose_cfgs(config_dir: Path, dataset_names: Sequence[str]) -> List[object]:
    if hydra is None:  # pragma: no cover
        raise ModuleNotFoundError("hydra-core is required to compose dataset configs.")
    cfgs: List[object] = []
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        for name in dataset_names:
            cfg = hydra.compose(config_name="build_retrieval_pipeline", overrides=[f"dataset={name}"])
            cfgs.append(cfg)
    return cfgs


def _collect_relations(cfg: object) -> Set[str]:
    dataset = str(cfg.get("dataset_name") or cfg.get("dataset") or "dataset")
    kb = cfg.get("kb")
    column_map = dict(cfg.get("column_map"))
    entity_normalization = cfg.get("entity_normalization")
    dataset_family = cfg.get("dataset_family")
    dataset_source = str(cfg.get("dataset_source", "hf")).strip().lower()
    hf_dataset = cfg.get("hf_dataset")
    hf_cache_dir_cfg = cfg.get("hf_cache_dir")
    hf_cache_dir = Path(hf_cache_dir_cfg).expanduser().resolve() if hf_cache_dir_cfg else None
    hf_offline = bool(cfg.get("hf_offline", False))
    remove_self_loops = bool(cfg.get("remove_self_loops", True))
    relation_cleaning_enabled = bool(cfg.get("relation_cleaning", True))
    relation_cleaning_rules = DEFAULT_RELATION_CLEANING_RULES

    if dataset_source != "hf":
        raise ValueError("dataset_source must be 'hf'; raw parquet ingestion is disabled.")

    relations: Set[str] = set()
    for sample in tqdm(
        iter_samples(
            dataset,
            kb,
            None,
            list(_ALLOWED_SPLITS),
            column_map,
            entity_normalization,
            dataset_source=dataset_source,
            dataset_family=dataset_family,
            hf_dataset=hf_dataset,
            hf_cache_dir=hf_cache_dir,
            hf_offline=hf_offline,
        ),
        desc=f"Relations from {dataset}",
    ):
        kept_edges, _ = _partition_graph_edges(
            sample.graph,
            relation_cleaning_rules,
            remove_self_loops=remove_self_loops,
            relation_cleaning_enabled=relation_cleaning_enabled,
        )
        for _, rel, _ in kept_edges:
            relations.add(rel)
    log_event(LOGGER, "dataset_relations_collected", dataset=dataset, relation_count=len(relations))
    return relations


def _resolve_forward_text(rel: str, mode: str) -> str:
    if mode == _FORWARD_TEXT_KG:
        return rel
    if mode == _FORWARD_TEXT_BLANK:
        return ""
    raise ValueError(f"Unsupported forward_text_mode: {mode!r}")


def _build_stub(relations: Sequence[str], *, forward_text_mode: str, inverse_placeholder: str) -> dict:
    entries = []
    for rel in relations:
        entries.append(
            {
                _KEY_FORWARD: rel,
                _KEY_FORWARD_TEXT: _resolve_forward_text(rel, forward_text_mode),
                _KEY_INVERSE: inverse_placeholder,
            }
        )
    return {_KEY_RELATIONS: entries}


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a global inverse-relations stub from multiple datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names (comma-separated or space-separated). Example: --datasets webqsp cwq",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to write inverse_relations stub JSON")
    parser.add_argument(
        "--forward-text-mode",
        type=str,
        default=_FORWARD_TEXT_KG,
        choices=[_FORWARD_TEXT_KG, _FORWARD_TEXT_BLANK],
        help="Populate forward_text with kg_id or leave blank.",
    )
    parser.add_argument(
        "--inverse-placeholder",
        type=str,
        default="",
        help="Placeholder inverse label (empty by default; fill before pipeline use).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(root) / "configs",
        help="Path to Hydra config directory (defaults to repo configs/).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    init_logging()
    dataset_names = _parse_dataset_list(args.datasets)
    cfgs = _compose_cfgs(args.config_dir, dataset_names)
    all_relations: Set[str] = set()
    for cfg in cfgs:
        all_relations |= _collect_relations(cfg)
    relations_sorted = sorted(all_relations)
    payload = _build_stub(
        relations_sorted,
        forward_text_mode=args.forward_text_mode,
        inverse_placeholder=args.inverse_placeholder,
    )
    out_path = args.output
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    log_event(LOGGER, "global_inverse_stub_written", path=str(out_path), relation_count=len(relations_sorted))
    return _ZERO


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[_ONE:]))
