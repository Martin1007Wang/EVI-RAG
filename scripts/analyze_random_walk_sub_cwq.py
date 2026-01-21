from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.ipc as ipc


DATASET = "rmanluo/RoG-cwq"
DEFAULT_SPLIT = "validation"
DEFAULT_K_MAX = 5
PERCENTILE = 90
COMPARE_ATOL = 1e-12


@dataclass(frozen=True)
class SplitPaths:
    builder_dir: Path
    dataset_name: str
    arrow_files: list[Path]


@dataclass(frozen=True)
class WalkMetrics:
    exact: np.ndarray  # [K, 2]
    within: np.ndarray  # [K, 2]


def _normalize_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.getenv("HF_DATASETS_CACHE")
    if env_root:
        roots.append(Path(env_root))
    roots.extend(
        [
            Path("/mnt/data/huggingface/datasets"),
            Path.home() / ".cache" / "huggingface" / "datasets",
        ]
    )
    return [root for root in roots if root.exists()]


def _find_dataset_dir(dataset: str, cache_roots: Iterable[Path]) -> Path:
    namespace, ds_name = dataset.split("/", 1)
    desired = _normalize_for_match(ds_name)
    namespace_prefix = f"{namespace.lower()}___"

    for cache_root in cache_roots:
        for child in cache_root.iterdir():
            if not child.is_dir():
                continue
            if not child.name.lower().startswith(namespace_prefix):
                continue
            _, tail = child.name.split("___", 1)
            if _normalize_for_match(tail) == desired:
                return child

    raise FileNotFoundError(
        f"Could not find a cached dataset dir for {dataset!r} under: "
        + ", ".join(str(p) for p in cache_roots)
    )


def _find_latest_builder_dir(dataset_dir: Path) -> Path:
    dataset_info_files: list[Path] = []
    for config_dir in dataset_dir.iterdir():
        if not config_dir.is_dir():
            continue
        for version_dir in config_dir.iterdir():
            if not version_dir.is_dir():
                continue
            for fingerprint_dir in version_dir.iterdir():
                if not fingerprint_dir.is_dir():
                    continue
                info = fingerprint_dir / "dataset_info.json"
                if info.exists():
                    dataset_info_files.append(info)

    if not dataset_info_files:
        raise FileNotFoundError(f"No dataset_info.json found under {dataset_dir}")

    latest = max(dataset_info_files, key=lambda p: p.stat().st_mtime)
    return latest.parent


def _resolve_builder_dir(dataset: str) -> Path:
    cache_roots = _candidate_cache_roots()
    if not cache_roots:
        raise FileNotFoundError("No HuggingFace datasets cache roots found")
    dataset_dir = _find_dataset_dir(dataset=dataset, cache_roots=cache_roots)
    return _find_latest_builder_dir(dataset_dir=dataset_dir)


def _load_split_paths(builder_dir: Path, split: str) -> SplitPaths:
    info_path = builder_dir / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    info = json.loads(info_path.read_text())
    dataset_name = info.get("dataset_name")
    if not isinstance(dataset_name, str) or not dataset_name:
        raise ValueError(f"Invalid dataset_name in {info_path}")

    arrow_files = sorted(builder_dir.glob(f"{dataset_name}-{split}-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(
            f"No .arrow files for split={split!r} under {builder_dir} "
            f"(expected pattern: {dataset_name}-{split}-*.arrow)"
        )

    return SplitPaths(builder_dir=builder_dir, dataset_name=dataset_name, arrow_files=arrow_files)


def _build_local_graph(
    triples: list[list[str]],
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    entity_to_local: dict[str, int] = {}
    src: list[int] = []
    dst: list[int] = []
    for head, _, tail in triples:
        head_idx = entity_to_local.get(head)
        if head_idx is None:
            head_idx = len(entity_to_local)
            entity_to_local[head] = head_idx
        tail_idx = entity_to_local.get(tail)
        if tail_idx is None:
            tail_idx = len(entity_to_local)
            entity_to_local[tail] = tail_idx
        src.append(head_idx)
        dst.append(tail_idx)

    if not entity_to_local:
        return {}, np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    return entity_to_local, np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)


def _unique_in_graph(entity_names: list[str], entity_to_local: dict[str, int]) -> np.ndarray:
    if not entity_names:
        return np.empty((0,), dtype=np.int64)
    locals_ = [entity_to_local.get(e) for e in entity_names]
    locals_ = [i for i in locals_ if i is not None]
    if not locals_:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.asarray(locals_, dtype=np.int64))


def _make_start_probs(
    num_nodes: int, start_nodes: np.ndarray, out_degree: np.ndarray
) -> np.ndarray:
    num_start = int(start_nodes.size)
    if num_start <= 0:
        raise ValueError("start_nodes must be non-empty")

    probs = np.zeros((num_nodes, 2), dtype=np.float64)
    probs[start_nodes, 0] = 1.0 / num_start

    weights = out_degree[start_nodes].astype(np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        probs[start_nodes, 1] = 1.0 / num_start
    else:
        probs[start_nodes, 1] = weights / weight_sum

    return probs


def _propagate_once(
    probs: np.ndarray, src: np.ndarray, dst: np.ndarray, out_degree: np.ndarray
) -> np.ndarray:
    num_nodes, num_cols = probs.shape
    dead_mask = out_degree == 0

    src_deg = out_degree[src]
    contrib = probs[src] / src_deg[:, None]

    next_probs = np.zeros_like(probs)
    for col in range(num_cols):
        next_probs[:, col] = np.bincount(dst, weights=contrib[:, col], minlength=num_nodes)

    if bool(dead_mask.any()):
        next_probs[dead_mask, :] += probs[dead_mask, :]

    return next_probs


def _propagate_absorbing_once(
    probs: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    out_degree: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    if not bool(target_mask.any()):
        return _propagate_once(probs=probs, src=src, dst=dst, out_degree=out_degree)

    moving = probs.copy()
    moving[target_mask, :] = 0.0
    next_probs = _propagate_once(probs=moving, src=src, dst=dst, out_degree=out_degree)
    next_probs[target_mask, :] += probs[target_mask, :]
    return next_probs


def _compute_walk_metrics(
    *,
    src: np.ndarray,
    dst: np.ndarray,
    out_degree: np.ndarray,
    start_nodes: np.ndarray,
    target_nodes: np.ndarray,
    k_max: int,
) -> WalkMetrics:
    num_nodes = int(out_degree.shape[0])
    target_mask = np.zeros((num_nodes,), dtype=bool)
    target_mask[target_nodes] = True

    probs0 = _make_start_probs(num_nodes=num_nodes, start_nodes=start_nodes, out_degree=out_degree)
    probs_exact = probs0.copy()
    probs_within = probs0.copy()

    exact = np.zeros((k_max, 2), dtype=np.float64)
    within = np.zeros((k_max, 2), dtype=np.float64)

    for k in range(k_max):
        probs_exact = _propagate_once(probs=probs_exact, src=src, dst=dst, out_degree=out_degree)
        probs_within = _propagate_absorbing_once(
            probs=probs_within,
            src=src,
            dst=dst,
            out_degree=out_degree,
            target_mask=target_mask,
        )
        exact[k, :] = probs_exact[target_mask, :].sum(axis=0)
        within[k, :] = probs_within[target_mask, :].sum(axis=0)

    return WalkMetrics(exact=exact, within=within)


def _summarize(values: np.ndarray) -> tuple[float, float, float]:
    return float(values.mean()), float(np.median(values)), float(np.percentile(values, PERCENTILE))


def _fmt_triplet(values: np.ndarray) -> str:
    mean, median, p90 = _summarize(values)
    return f"{mean:.4f} / {median:.4f} / {p90:.4f}"


def _compare_counts(fwd: np.ndarray, bwd: np.ndarray) -> tuple[int, int, int]:
    diff = bwd - fwd
    bwd_gt = int((diff > COMPARE_ATOL).sum())
    fwd_gt = int((diff < -COMPARE_ATOL).sum())
    eq = int(fwd.size - bwd_gt - fwd_gt)
    return bwd_gt, fwd_gt, eq


def _render_scope(*, dataset: str, split: str, num_total: int, num_sub: int) -> list[str]:
    return [
        "## Scope",
        f"- Dataset: `{dataset}`",
        f"- Split: `{split}`",
        "- Sub filter: keep samples where `q_entity` and `a_entity` both appear in the graph node set.",
        "- Graph: `graph` is a list of directed (head, relation, tail) triples; node set is union of heads/tails.",
        f"- Total samples in split: {num_total}",
        f"- Sub graphs used: {num_sub}",
        "",
    ]


def _render_entity_stats(
    *,
    num_sub: int,
    start_total: int,
    start_unique: int,
    start_counts: np.ndarray,
    answer_total: int,
    answer_unique: int,
    answer_counts: np.ndarray,
) -> list[str]:
    start_mean, start_median, start_p90 = _summarize(start_counts)
    ans_mean, ans_median, ans_p90 = _summarize(answer_counts)
    return [
        "## Start/Answer Set Sizes (sub)",
        f"- Questions: {num_sub}",
        "- Start entities (`q_entity`)",
        f"  - Total: {start_total}",
        f"  - Unique: {start_unique}",
        f"  - Per-question: mean {start_mean:.4f}, median {start_median:.0f}, p90 {start_p90:.0f}, max {int(start_counts.max())}",
        "- Answer entities (`a_entity`)",
        f"  - Total: {answer_total}",
        f"  - Unique: {answer_unique}",
        f"  - Per-question: mean {ans_mean:.4f}, median {ans_median:.0f}, p90 {ans_p90:.0f}, max {int(answer_counts.max())}",
        "",
    ]


def _render_degree_stats(
    *,
    start_degrees: np.ndarray,
    answer_degrees: np.ndarray,
    mean_degree_cmp: tuple[int, int, int],
    sum_degree_cmp: tuple[int, int, int],
    ratio_equal_nodes: np.ndarray,
) -> list[str]:
    sd_mean, sd_median, sd_p90 = _summarize(start_degrees.astype(np.float64))
    ad_mean, ad_median, ad_p90 = _summarize(answer_degrees.astype(np.float64))
    eq_mean, eq_median, eq_p90 = _summarize(ratio_equal_nodes)

    ms, ma, me = mean_degree_cmp
    ss, sa, se = sum_degree_cmp

    return [
        "## Degree Stats (sub)",
        "Edges are directed; inverse edges are not guaranteed, so in-degree and out-degree can differ.",
        f"Across graphs, the fraction of nodes with `out_degree == in_degree` is: mean {eq_mean:.4f}, median {eq_median:.4f}, p90 {eq_p90:.4f}.",
        "",
        "Per-node out-degree:",
        f"- Start nodes: mean {sd_mean:.2f}, median {sd_median:.0f}, p90 {sd_p90:.1f}, min {int(start_degrees.min())}, max {int(start_degrees.max())} (count {int(start_degrees.size)})",
        f"- Answer nodes: mean {ad_mean:.2f}, median {ad_median:.0f}, p90 {ad_p90:.1f}, min {int(answer_degrees.min())}, max {int(answer_degrees.max())} (count {int(answer_degrees.size)})",
        "",
        "Per-graph comparisons:",
        f"- Mean out-degree: start > answer for {ms} graphs, answer > start for {ma} graphs, equal for {me} graphs",
        f"- Total out-degree (sum over nodes in set): start > answer for {ss} graphs, answer > start for {sa} graphs, equal for {se} graphs",
        "",
    ]


def _render_random_walk_setup(k_max: int) -> list[str]:
    k_line = "- K in {1, 2, 3, 4, 5}" if k_max == 5 else f"- K in {{1..{k_max}}}"
    return [
        "## Random Walk Setup",
        "- Graph: directed edges from `graph` (no edge reversal; inverse edges may or may not exist).",
        "- Start/target sets: local node indices mapped from `q_entity` / `a_entity`; duplicates removed.",
        "- Step rule: at each step, choose uniformly among outgoing edges; if out-degree is 0, stay in place.",
        "- Start distribution:",
        "  1) `uniform`: uniform over nodes in the set",
        "  2) `degree`: weighted by out-degree within the set (fallback to uniform if sum is 0)",
        "- Metrics:",
        "  - `exact@K`: probability of being in the target set exactly at step K",
        "  - `within@K`: probability of hitting the target within K steps (target is absorbing)",
        k_line,
        "- Computation: exact probability propagation (no Monte Carlo)",
        "",
    ]


def _render_start_distribution_block(
    *,
    title: str,
    k_max: int,
    exact_fwd: np.ndarray,
    exact_bwd: np.ndarray,
    within_fwd: np.ndarray,
    within_bwd: np.ndarray,
) -> list[str]:
    lines: list[str] = [f"### Start distribution: {title}"]
    for k in range(k_max):
        kk = k + 1
        lines.extend(
            [
                f"K={kk}",
                f"- exact fwd: {_fmt_triplet(exact_fwd[:, k])}",
                f"- exact bwd: {_fmt_triplet(exact_bwd[:, k])}",
                f"- within fwd: {_fmt_triplet(within_fwd[:, k])}",
                f"- within bwd: {_fmt_triplet(within_bwd[:, k])}",
            ]
        )

        ex_bgt, ex_fgt, ex_eq = _compare_counts(exact_fwd[:, k], exact_bwd[:, k])
        wi_bgt, wi_fgt, wi_eq = _compare_counts(within_fwd[:, k], within_bwd[:, k])
        if (ex_fgt, ex_eq) == (wi_fgt, wi_eq):
            lines.append(
                f"- bwd > fwd: exact {ex_bgt}, within {wi_bgt} (fwd > bwd: {ex_fgt}, equal: {ex_eq})"
            )
        else:
            lines.append(
                f"- bwd > fwd: exact {ex_bgt}, within {wi_bgt} (fwd > bwd: exact {ex_fgt}, within {wi_fgt}; equal: exact {ex_eq}, within {wi_eq})"
            )
        lines.append("")
    return lines


def _render_results(
    *,
    k_max: int,
    exact_fwd_uniform: np.ndarray,
    exact_bwd_uniform: np.ndarray,
    within_fwd_uniform: np.ndarray,
    within_bwd_uniform: np.ndarray,
    exact_fwd_degree: np.ndarray,
    exact_bwd_degree: np.ndarray,
    within_fwd_degree: np.ndarray,
    within_bwd_degree: np.ndarray,
) -> list[str]:
    lines: list[str] = ["## Results (mean/median/p90)", ""]
    lines.extend(
        _render_start_distribution_block(
            title="uniform",
            k_max=k_max,
            exact_fwd=exact_fwd_uniform,
            exact_bwd=exact_bwd_uniform,
            within_fwd=within_fwd_uniform,
            within_bwd=within_bwd_uniform,
        )
    )
    lines.extend(
        _render_start_distribution_block(
            title="degree-weighted",
            k_max=k_max,
            exact_fwd=exact_fwd_degree,
            exact_bwd=exact_bwd_degree,
            within_fwd=within_fwd_degree,
            within_bwd=within_bwd_degree,
        )
    )
    return lines


def _render_takeaways(
    *,
    split: str,
    within_fwd_uniform: np.ndarray,
    within_bwd_uniform: np.ndarray,
    within_fwd_degree: np.ndarray,
    within_bwd_degree: np.ndarray,
) -> list[str]:
    k_max = within_fwd_uniform.shape[1]
    mean_u_fwd = within_fwd_uniform.mean(axis=0)
    mean_u_bwd = within_bwd_uniform.mean(axis=0)
    mean_d_fwd = within_fwd_degree.mean(axis=0)
    mean_d_bwd = within_bwd_degree.mean(axis=0)

    def ks_where_bwd_gt(mean_fwd: np.ndarray, mean_bwd: np.ndarray) -> str:
        ks = [str(i + 1) for i in range(k_max) if mean_bwd[i] > mean_fwd[i] + COMPARE_ATOL]
        return ", ".join(ks) if ks else "none"

    return [
        "## Takeaways",
        f"- On cwq sub graphs (`{split}` split), mean `within@K` bwd > fwd at K={ks_where_bwd_gt(mean_u_fwd, mean_u_bwd)} for uniform starts.",
        f"- On cwq sub graphs (`{split}` split), mean `within@K` bwd > fwd at K={ks_where_bwd_gt(mean_d_fwd, mean_d_bwd)} for degree-weighted starts.",
    ]


def _render_markdown(
    *,
    dataset: str,
    split: str,
    num_total: int,
    num_sub: int,
    start_total: int,
    start_unique: int,
    start_counts: np.ndarray,
    answer_total: int,
    answer_unique: int,
    answer_counts: np.ndarray,
    start_degrees: np.ndarray,
    answer_degrees: np.ndarray,
    mean_degree_cmp: tuple[int, int, int],
    sum_degree_cmp: tuple[int, int, int],
    ratio_equal_nodes: np.ndarray,
    k_max: int,
    exact_fwd_uniform: np.ndarray,
    exact_bwd_uniform: np.ndarray,
    within_fwd_uniform: np.ndarray,
    within_bwd_uniform: np.ndarray,
    exact_fwd_degree: np.ndarray,
    exact_bwd_degree: np.ndarray,
    within_fwd_degree: np.ndarray,
    within_bwd_degree: np.ndarray,
) -> str:
    lines: list[str] = [f"# Random Walk Analysis on cwq (sub)", ""]
    lines.extend(_render_scope(dataset=dataset, split=split, num_total=num_total, num_sub=num_sub))
    lines.extend(
        _render_entity_stats(
            num_sub=num_sub,
            start_total=start_total,
            start_unique=start_unique,
            start_counts=start_counts,
            answer_total=answer_total,
            answer_unique=answer_unique,
            answer_counts=answer_counts,
        )
    )
    lines.extend(
        _render_degree_stats(
            start_degrees=start_degrees,
            answer_degrees=answer_degrees,
            mean_degree_cmp=mean_degree_cmp,
            sum_degree_cmp=sum_degree_cmp,
            ratio_equal_nodes=ratio_equal_nodes,
        )
    )
    lines.extend(_render_random_walk_setup(k_max=k_max))
    lines.extend(
        _render_results(
            k_max=k_max,
            exact_fwd_uniform=exact_fwd_uniform,
            exact_bwd_uniform=exact_bwd_uniform,
            within_fwd_uniform=within_fwd_uniform,
            within_bwd_uniform=within_bwd_uniform,
            exact_fwd_degree=exact_fwd_degree,
            exact_bwd_degree=exact_bwd_degree,
            within_fwd_degree=within_fwd_degree,
            within_bwd_degree=within_bwd_degree,
        )
    )
    lines.extend(
        _render_takeaways(
            split=split,
            within_fwd_uniform=within_fwd_uniform,
            within_bwd_uniform=within_bwd_uniform,
            within_fwd_degree=within_fwd_degree,
            within_bwd_degree=within_bwd_degree,
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Random walk analysis on RoG-cwq (sub).")
    parser.add_argument("--dataset", type=str, default=DATASET)
    parser.add_argument("--split", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--k_max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--builder_dir", type=Path, default=None)
    parser.add_argument("--output_md", type=Path, required=True)
    args = parser.parse_args()

    builder_dir = args.builder_dir or _resolve_builder_dir(args.dataset)
    split_paths = _load_split_paths(builder_dir=builder_dir, split=args.split)

    k_max = int(args.k_max)
    if k_max <= 0:
        raise ValueError("k_max must be > 0")

    max_examples = args.max_examples
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be > 0")

    num_total = 0
    num_sub = 0

    start_counts: list[int] = []
    answer_counts: list[int] = []
    start_entities: set[str] = set()
    answer_entities: set[str] = set()

    start_degrees: list[int] = []
    answer_degrees: list[int] = []
    ratio_equal_nodes: list[float] = []

    mean_degree_start_gt = 0
    mean_degree_ans_gt = 0
    mean_degree_eq = 0
    sum_degree_start_gt = 0
    sum_degree_ans_gt = 0
    sum_degree_eq = 0

    exact_fwd_uniform: list[np.ndarray] = []
    exact_bwd_uniform: list[np.ndarray] = []
    within_fwd_uniform: list[np.ndarray] = []
    within_bwd_uniform: list[np.ndarray] = []
    exact_fwd_degree: list[np.ndarray] = []
    exact_bwd_degree: list[np.ndarray] = []
    within_fwd_degree: list[np.ndarray] = []
    within_bwd_degree: list[np.ndarray] = []

    for arrow_path in split_paths.arrow_files:
        with arrow_path.open("rb") as f:
            reader = ipc.open_stream(f)
            for batch in reader:
                q_arr = batch.column(batch.schema.get_field_index("q_entity"))
                a_arr = batch.column(batch.schema.get_field_index("a_entity"))
                g_arr = batch.column(batch.schema.get_field_index("graph"))

                for i in range(batch.num_rows):
                    num_total += 1
                    if max_examples is not None and num_total > max_examples:
                        break

                    q_entities = q_arr[i].as_py() or []
                    a_entities = a_arr[i].as_py() or []
                    triples = g_arr[i].as_py() or []

                    entity_to_local, src, dst = _build_local_graph(triples)
                    if not entity_to_local:
                        continue

                    num_nodes = len(entity_to_local)
                    out_degree = np.bincount(src, minlength=num_nodes)
                    in_degree = np.bincount(dst, minlength=num_nodes)

                    q_local = _unique_in_graph(q_entities, entity_to_local)
                    a_local = _unique_in_graph(a_entities, entity_to_local)
                    if q_local.size == 0 or a_local.size == 0:
                        continue

                    num_sub += 1
                    ratio_equal_nodes.append(float((out_degree == in_degree).mean()))

                    start_counts.append(int(q_local.size))
                    answer_counts.append(int(a_local.size))
                    start_entities.update(e for e in q_entities if e in entity_to_local)
                    answer_entities.update(e for e in a_entities if e in entity_to_local)

                    start_degrees.extend(out_degree[q_local].tolist())
                    answer_degrees.extend(out_degree[a_local].tolist())

                    start_mean_deg = float(out_degree[q_local].mean())
                    ans_mean_deg = float(out_degree[a_local].mean())
                    if abs(start_mean_deg - ans_mean_deg) <= COMPARE_ATOL:
                        mean_degree_eq += 1
                    elif start_mean_deg > ans_mean_deg:
                        mean_degree_start_gt += 1
                    else:
                        mean_degree_ans_gt += 1

                    start_sum_deg = float(out_degree[q_local].sum())
                    ans_sum_deg = float(out_degree[a_local].sum())
                    if abs(start_sum_deg - ans_sum_deg) <= COMPARE_ATOL:
                        sum_degree_eq += 1
                    elif start_sum_deg > ans_sum_deg:
                        sum_degree_start_gt += 1
                    else:
                        sum_degree_ans_gt += 1

                    fwd = _compute_walk_metrics(
                        src=src,
                        dst=dst,
                        out_degree=out_degree,
                        start_nodes=q_local,
                        target_nodes=a_local,
                        k_max=k_max,
                    )
                    bwd = _compute_walk_metrics(
                        src=src,
                        dst=dst,
                        out_degree=out_degree,
                        start_nodes=a_local,
                        target_nodes=q_local,
                        k_max=k_max,
                    )

                    exact_fwd_uniform.append(fwd.exact[:, 0].copy())
                    exact_fwd_degree.append(fwd.exact[:, 1].copy())
                    within_fwd_uniform.append(fwd.within[:, 0].copy())
                    within_fwd_degree.append(fwd.within[:, 1].copy())
                    exact_bwd_uniform.append(bwd.exact[:, 0].copy())
                    exact_bwd_degree.append(bwd.exact[:, 1].copy())
                    within_bwd_uniform.append(bwd.within[:, 0].copy())
                    within_bwd_degree.append(bwd.within[:, 1].copy())

                if max_examples is not None and num_total > max_examples:
                    break
        if max_examples is not None and num_total > max_examples:
            break

    if num_sub == 0:
        raise RuntimeError("No sub graphs kept; check dataset/cache paths and filter assumptions")

    start_counts_np = np.asarray(start_counts, dtype=np.int64)
    answer_counts_np = np.asarray(answer_counts, dtype=np.int64)
    start_degrees_np = np.asarray(start_degrees, dtype=np.int64)
    answer_degrees_np = np.asarray(answer_degrees, dtype=np.int64)
    ratio_equal_nodes_np = np.asarray(ratio_equal_nodes, dtype=np.float64)

    exact_fwd_uniform_np = np.stack(exact_fwd_uniform, axis=0)
    exact_bwd_uniform_np = np.stack(exact_bwd_uniform, axis=0)
    within_fwd_uniform_np = np.stack(within_fwd_uniform, axis=0)
    within_bwd_uniform_np = np.stack(within_bwd_uniform, axis=0)
    exact_fwd_degree_np = np.stack(exact_fwd_degree, axis=0)
    exact_bwd_degree_np = np.stack(exact_bwd_degree, axis=0)
    within_fwd_degree_np = np.stack(within_fwd_degree, axis=0)
    within_bwd_degree_np = np.stack(within_bwd_degree, axis=0)

    md = _render_markdown(
        dataset=args.dataset,
        split=args.split,
        num_total=num_total if max_examples is None else min(num_total, max_examples),
        num_sub=num_sub,
        start_total=int(start_counts_np.sum()),
        start_unique=len(start_entities),
        start_counts=start_counts_np,
        answer_total=int(answer_counts_np.sum()),
        answer_unique=len(answer_entities),
        answer_counts=answer_counts_np,
        start_degrees=start_degrees_np,
        answer_degrees=answer_degrees_np,
        mean_degree_cmp=(mean_degree_start_gt, mean_degree_ans_gt, mean_degree_eq),
        sum_degree_cmp=(sum_degree_start_gt, sum_degree_ans_gt, sum_degree_eq),
        ratio_equal_nodes=ratio_equal_nodes_np,
        k_max=k_max,
        exact_fwd_uniform=exact_fwd_uniform_np,
        exact_bwd_uniform=exact_bwd_uniform_np,
        within_fwd_uniform=within_fwd_uniform_np,
        within_bwd_uniform=within_bwd_uniform_np,
        exact_fwd_degree=exact_fwd_degree_np,
        exact_bwd_degree=exact_bwd_degree_np,
        within_fwd_degree=within_fwd_degree_np,
        within_bwd_degree=within_bwd_degree_np,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md)


if __name__ == "__main__":
    main()
