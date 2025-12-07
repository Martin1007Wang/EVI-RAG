"""构建 g_agent 缓存的难度划分（Easy / Hard / Unreachable）。

定义：
- Easy: 可达且 beam 贪心（K=beam_k, max_steps）成功命中答案。
- Hard: 可达但 beam 贪心失败（典型场景：正确路径分数排在截断之后）。
- Unreachable: `is_answer_reachable=False` 或 `gt_path_exists=False`。

输出：
- JSON：包含各组 sample_id 列表与逐样本的 beam 指标、GT 最小 rank 信息。
- 可选：写出 easy / hard 的子集 g_agent 缓存，便于直接跑 eval。"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

# rootutils 在某些环境未预装，这里提供最小兜底，避免导入失败
import sys


def _setup_repo_root() -> None:
    try:
        import rootutils as _rootutils  # type: ignore
    except ModuleNotFoundError:
        class _DummyRootutils:
            @staticmethod
            def setup_root(path: str | Path, *, indicator: str = ".project-root", pythonpath: bool = True):
                repo_root = Path(path).resolve().parents[1]
                if pythonpath and str(repo_root) not in sys.path:
                    sys.path.append(str(repo_root))
                return repo_root

        sys.modules["rootutils"] = _DummyRootutils()
        _rootutils = sys.modules["rootutils"]  # type: ignore
    _rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)  # type: ignore


_setup_repo_root()

# 复用已有检索上限/beam 逻辑，避免重复实现
from scripts.analyze_gflownet_data import _beam_greedy_metrics, _gt_ranks  # type: ignore

# 直接动态加载 g_agent_dataset，避免触发 src.data/__init__ 对 hydra 等重依赖
_G_AGENT_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "g_agent_dataset.py"
spec = importlib.util.spec_from_file_location("local_g_agent_dataset", _G_AGENT_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load g_agent_dataset from {_G_AGENT_PATH}")
g_agent_module = importlib.util.module_from_spec(spec)
sys.modules["local_g_agent_dataset"] = g_agent_module
spec.loader.exec_module(g_agent_module)
GAgentPyGDataset = getattr(g_agent_module, "GAgentPyGDataset")
_load_g_agent_cache = getattr(g_agent_module, "_load_g_agent_cache")

DEFAULT_BASE = Path("/mnt/data/retrieval_dataset/webqsp/materialized/g_agent")
DEFAULT_G_AGENT = DEFAULT_BASE / "test_g_agent.pt"
DEFAULT_OUT_JSON = DEFAULT_BASE / "test_beamK5_difficulty.json"
DEFAULT_MAX_STEPS = 3  # 与 webqsp 实验 env.max_steps 对齐（train_gflownet_webqsp.yaml）


def _write_subset_cache_raw(cache_path: Path, sample_ids: Iterable[str], out_path: Path) -> None:
    """从原始 g_agent cache 中按 sample_id 过滤样本，保持记录原样。"""
    payload = _load_g_agent_cache(cache_path)
    raw_samples: Sequence[Dict] = payload.get("samples") or []
    id_set = set(sample_ids)
    selected = [rec for rec in raw_samples if str(rec.get("sample_id")) in id_set]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": selected}, out_path)


def build_difficulty_split(
    cache_path: Path,
    beam_k: int,
    max_steps: int,
    use_top_mask: bool,
) -> Dict:
    ds = GAgentPyGDataset(cache_path, drop_unreachable=False)
    easy: List[str] = []
    hard: List[str] = []
    unreachable: List[str] = []
    per_sample: List[Dict] = []

    for idx in range(len(ds)):
        data = ds[idx]
        sample = ds.samples[idx]
        sample_id = sample.sample_id
        answerable = bool(sample.is_answer_reachable) and bool(sample.gt_path_exists)

        beam = _beam_greedy_metrics(data, [beam_k], max_steps, use_top_mask)[beam_k]
        beam_success = float(beam["success"])
        mask = data.top_edge_mask if use_top_mask and hasattr(data, "top_edge_mask") else None
        ranks, gaps = _gt_ranks(data.edge_scores, data.gt_path_edge_local_ids, mask)
        min_rank = min(ranks) if ranks else None

        if not answerable:
            group = "unreachable"
            unreachable.append(sample_id)
        elif beam_success >= 0.5:
            group = "easy"
            easy.append(sample_id)
        else:
            group = "hard"
            hard.append(sample_id)

        per_sample.append(
            {
                "sample_id": sample_id,
                "group": group,
                "beam_success": beam_success,
                "answerable": answerable,
                "min_gt_rank": min_rank,
                "gt_ranks": ranks,
                "gt_score_gaps": gaps,
            }
        )

    return {
        "easy": easy,
        "hard": hard,
        "unreachable": unreachable,
        "per_sample": per_sample,
        "meta": {
            "cache_path": str(cache_path),
            "beam_k": beam_k,
            "max_steps": max_steps,
            "use_top_mask": use_top_mask,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct Easy/Hard split based on beam-greedy success@K for g_agent cache.")
    parser.add_argument(
        "--g-agent-path",
        type=Path,
        default=DEFAULT_G_AGENT,
        help=f"Path to g_agent cache .pt (default: {DEFAULT_G_AGENT})",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help=f"Where to save difficulty JSON (default: {DEFAULT_OUT_JSON})",
    )
    parser.add_argument("--beam-k", type=int, default=5, help="Beam width K for greedy upper bound")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Max steps for beam rollout (align with env.max_steps; default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument("--use-top-mask", action="store_true", help="Restrict beam to top_edge_mask if present")
    parser.add_argument(
        "--write-caches-dir",
        type=Path,
        default=DEFAULT_BASE,
        help=f"If set, write easy/hard subset caches to this directory (default: {DEFAULT_BASE}; filenames: easy_g_agent.pt, hard_g_agent.pt)",
    )
    args = parser.parse_args()

    result = build_difficulty_split(
        cache_path=args.g_agent_path,
        beam_k=args.beam_k,
        max_steps=args.max_steps,
        use_top_mask=args.use_top_mask,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    easy_cnt, hard_cnt, unreach_cnt = len(result["easy"]), len(result["hard"]), len(result["unreachable"])
    total = easy_cnt + hard_cnt + unreach_cnt
    print(f"[summary] total={total} easy={easy_cnt} hard={hard_cnt} unreachable={unreach_cnt}")
    print(f"[saved] difficulty json => {args.out_json}")

    if args.write_caches_dir is not None:
        easy_path = args.write_caches_dir / "easy_g_agent.pt"
        hard_path = args.write_caches_dir / "hard_g_agent.pt"
        _write_subset_cache_raw(args.g_agent_path, result["easy"], easy_path)
        _write_subset_cache_raw(args.g_agent_path, result["hard"], hard_path)
        print(f"[saved] easy cache => {easy_path}")
        print(f"[saved] hard cache => {hard_path}")


if __name__ == "__main__":
    main()
