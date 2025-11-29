from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def triplet_to_str(triplet: Tuple[str, str, str]) -> str:
    head, relation, tail = triplet
    return f"({head}, {relation}, {tail})"


def build_user_prompt(question: str, triplets: Sequence[Tuple[str, str, str]], limit: int) -> str:
    lines = [triplet_to_str(t) for t in triplets[:limit]]
    triplet_block = "Triplets:\n" + "\n".join(lines) if lines else "Triplets:\n"
    question_block = f"Question:\n{question}"
    instruction = "Answer with a concise entity string prefixed by 'Ans:' using only the provided triplets."
    return "\n\n".join([triplet_block, question_block, instruction])


__all__ = ["build_user_prompt", "triplet_to_str"]
