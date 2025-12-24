from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


def triplet_to_str(triplet: Tuple[str, str, str]) -> str:
    head, relation, tail = triplet
    return f"({head}, {relation}, {tail})"


def build_triplet_prompt(question: str, triplets: Sequence[Tuple[str, str, str]], limit: int) -> str:
    lines = [triplet_to_str(t) for t in triplets[:limit]]
    triplet_block = "Triplets:\n" + "\n".join(lines) if lines else "Triplets:\n"
    question_block = f"Question:\n{question}"
    instruction = (
        "Return JSON only with the full answer list: "
        "{\"answers\": [\"<entity>\", ...]}. Use [] if no answer can be derived. "
        "Use entity strings exactly as they appear in the triplets."
    )
    return "\n\n".join([triplet_block, question_block, instruction])


def build_path_prompt(
    *,
    question: str,
    chains: Sequence[Dict[str, object]],
    limit: int,
    include_meta: bool,
    instruction: str,
) -> str:
    """
    Build a prompt block from candidate chains (paths).
    Each chain dict should include chain_text, frequency, length.
    """
    lines: List[str] = []
    for i, chain in enumerate(chains[:limit], 1):
        prefix = ""
        if include_meta:
            prefix = f"[freq={chain.get('frequency', 0)},len={chain.get('length', 0)}] "
        lines.append(f"{i}. {prefix}{chain.get('chain_text', '')}")
    paths_block = "Paths:\n" + ("\n".join(lines) if lines else "")
    question_block = f"Question:\n{question}"
    return "\n\n".join([paths_block, question_block, instruction])


__all__ = ["build_triplet_prompt", "build_path_prompt", "triplet_to_str"]
