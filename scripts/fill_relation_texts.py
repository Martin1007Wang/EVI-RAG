#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import rootutils
except ModuleNotFoundError:  # pragma: no cover
    rootutils = None  # type: ignore[assignment]

if rootutils is not None:
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
else:  # pragma: no cover
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

from src.utils.logging_utils import get_logger, init_logging, log_event


LOGGER = get_logger(__name__)

_EMPTY = ""
_KEY_FORWARD = "forward"
_KEY_FORWARD_TEXT = "forward_text"
_KEY_FORWARD_LABEL = "forward_label"
_KEY_INVERSE = "inverse"
_KEY_INVERSE_TEXT = "inverse_text"

_VERB_LEADS = (
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
)
_VERB_PREP = (
    "by",
    "in",
    "on",
    "at",
    "from",
    "to",
    "for",
    "with",
    "about",
    "into",
    "onto",
    "over",
    "under",
    "between",
    "among",
    "within",
    "without",
    "around",
    "through",
    "across",
    "during",
    "before",
    "after",
    "since",
    "until",
    "via",
    "per",
)

_PROPERTY_TEXT_OVERRIDES: Dict[str, Tuple[str, str]] = {
    "place_of_birth": ("born in", "birthplace of"),
    "date_of_birth": ("born on", "birth date of"),
    "place_of_death": ("died in", "death place of"),
    "date_of_death": ("died on", "death date of"),
    "cause_of_death": ("cause of death", "cause of death of"),
    "location": ("located in", "location of"),
    "located_in": ("located in", "location of"),
    "headquarters": ("headquartered in", "headquarters of"),
    "administrative_headquarters": ("administrative headquarters in", "administrative headquarters of"),
    "headquarters_location": ("headquartered in", "headquarters of"),
    "headquarters_city": ("headquartered in", "headquarters of"),
    "headquartered_in": ("headquartered in", "headquarters of"),
    "based_in": ("based in", "base of"),
    "residence": ("resides in", "residence of"),
    "place_of_residence": ("resides in", "residence of"),
    "resides_in": ("resides in", "residence of"),
    "member_of": ("member of", "has member"),
    "is_member_of": ("member of", "has member"),
    "members": ("has member", "member of"),
    "contains": ("contains", "part of"),
    "contained_by": ("contained by", "contains"),
    "containedby": ("contained by", "contains"),
    "part_of": ("part of", "has part"),
    "is_part_of": ("part of", "has part"),
    "spouse": ("married to", "married to"),
    "spouse_s": ("married to", "married to"),
    "spouses": ("married to", "married to"),
    "spouse_or_domestic_partner": ("married to", "married to"),
    "married_to": ("married to", "married to"),
    "partner": ("partner of", "partner of"),
    "partners": ("partner of", "partner of"),
    "friend": ("friend of", "friend of"),
    "friends": ("friend of", "friend of"),
    "sibling": ("sibling of", "sibling of"),
    "sibling_s": ("sibling of", "sibling of"),
    "siblings": ("sibling of", "sibling of"),
    "brother": ("brother of", "brother of"),
    "sister": ("sister of", "sister of"),
    "children": ("parent of", "child of"),
    "child": ("parent of", "child of"),
    "parents": ("child of", "parent of"),
    "parent": ("child of", "parent of"),
    "capital": ("capital", "capital of"),
    "employer": ("employed by", "employer of"),
    "employers": ("employed by", "employer of"),
    "employee": ("employer of", "employed by"),
    "employees": ("employer of", "employed by"),
    "owned_by": ("owned by", "owner of"),
    "owner": ("owned by", "owner of"),
    "owners": ("owned by", "owner of"),
    "owner_s": ("owned by", "owner of"),
}

_FORWARD_TO_INVERSE: Dict[str, str] = {
    "born in": "birthplace of",
    "born on": "birth date of",
    "died in": "death place of",
    "died on": "death date of",
    "located in": "location of",
    "headquartered in": "headquarters of",
    "based in": "base of",
    "resides in": "residence of",
    "member of": "has member",
    "part of": "has part",
    "contained by": "contains",
    "contains": "part of",
    "married to": "married to",
    "partner of": "partner of",
    "friend of": "friend of",
    "sibling of": "sibling of",
    "employed by": "employer of",
    "owned by": "owner of",
    "produced by": "producer of",
    "directed by": "director of",
    "written by": "author of",
    "composed by": "composer of",
    "designed by": "designer of",
    "developed by": "developer of",
    "invented by": "inventor of",
    "founded by": "founder of",
    "published by": "publisher of",
    "operated by": "operator of",
    "manufactured by": "manufacturer of",
    "edited by": "editor of",
    "created by": "creator of",
    "coached by": "coach of",
    "played by": "player of",
    "acted by": "actor of",
}

_VERB_TOKEN_TO_AGENT: Dict[str, str] = {
    "acted": "actor",
    "built": "builder",
    "coached": "coach",
    "composed": "composer",
    "conducted": "conductor",
    "created": "creator",
    "designed": "designer",
    "developed": "developer",
    "directed": "director",
    "edited": "editor",
    "founded": "founder",
    "invented": "inventor",
    "led": "leader",
    "made": "maker",
    "managed": "manager",
    "manufactured": "manufacturer",
    "owned": "owner",
    "operated": "operator",
    "painted": "painter",
    "performed": "performer",
    "played": "player",
    "produced": "producer",
    "published": "publisher",
    "recorded": "recording artist",
    "sang": "singer",
    "sung": "singer",
    "taught": "teacher",
    "written": "writer",
}

_ROLE_NOUN_TO_PAST: Dict[str, str] = {
    "producer": "produced",
    "producers": "produced",
    "director": "directed",
    "directors": "directed",
    "composer": "composed",
    "composers": "composed",
    "editor": "edited",
    "editors": "edited",
    "creator": "created",
    "creators": "created",
    "designer": "designed",
    "designers": "designed",
    "developer": "developed",
    "developers": "developed",
    "manufacturer": "manufactured",
    "manufacturers": "manufactured",
    "publisher": "published",
    "publishers": "published",
    "inventor": "invented",
    "inventors": "invented",
    "founder": "founded",
    "founders": "founded",
    "owner": "owned",
    "owners": "owned",
    "operator": "operated",
    "operators": "operated",
    "coach": "coached",
    "coaches": "coached",
    "actor": "acted",
    "actors": "acted",
    "writer": "written",
    "writers": "written",
    "author": "written",
    "authors": "written",
}
_AGENT_EXCEPTIONS: Dict[str, str] = {
    "act": "actor",
    "compose": "composer",
    "conduct": "conductor",
    "create": "creator",
    "design": "designer",
    "develop": "developer",
    "direct": "director",
    "edit": "editor",
    "found": "founder",
    "invent": "inventor",
    "manage": "manager",
    "manufacture": "manufacturer",
    "operate": "operator",
    "produce": "producer",
    "publish": "publisher",
    "record": "recording artist",
}


def _tokenize(segment: str) -> List[str]:
    return [tok for tok in segment.replace("_", " ").replace("/", " ").split() if tok]


def _phrase_from_segment(segment: str) -> str:
    return " ".join(_tokenize(segment))


def _is_verb_like(words: Sequence[str]) -> bool:
    if not words:
        return False
    if words[0] in _VERB_LEADS:
        return True
    if any(word in _VERB_PREP for word in words):
        return True
    last = words[-1]
    if last.endswith("ed") or last.endswith("ing"):
        return True
    return False


def _agent_from_base(base: str) -> str:
    if base in _AGENT_EXCEPTIONS:
        return _AGENT_EXCEPTIONS[base]
    if base.endswith("e"):
        return f"{base}r"
    return f"{base}er"


def _agent_from_verb_token(token: str) -> str:
    token = token.lower()
    direct = _VERB_TOKEN_TO_AGENT.get(token)
    if direct is not None:
        return direct
    base = token
    if base.endswith("ied"):
        base = f"{base[:-3]}y"
    elif base.endswith("ed"):
        base = base[:-2]
    elif base.endswith("ing"):
        base = base[:-3]
    return _agent_from_base(base)


def _build_forward_text(property_phrase: str, property_words: Sequence[str]) -> str:
    if not property_phrase:
        return "relation"
    if _is_verb_like(property_words):
        return property_phrase
    return property_phrase


def _build_inverse_text(
    property_phrase: str,
    property_words: Sequence[str],
    *,
    forward_text: str,
) -> str:
    if not property_phrase:
        return f"inverse of {forward_text}".strip()
    mapped = _FORWARD_TO_INVERSE.get(forward_text)
    if mapped is not None:
        return mapped
    if _is_verb_like(property_words):
        if property_words and property_words[-1] == "by" and len(property_words) >= 2:
            agent = _agent_from_verb_token(property_words[-2])
            return f"{agent} of"
        return f"inverse of {forward_text}".strip()
    return f"{property_phrase} of"


def _apply_property_override(property_segment: str) -> Tuple[str, str] | None:
    override = _PROPERTY_TEXT_OVERRIDES.get(property_segment)
    if override is not None:
        return override
    return None


def _apply_role_noun(property_segment: str) -> Tuple[str, str] | None:
    past = _ROLE_NOUN_TO_PAST.get(property_segment)
    if past is None:
        return None
    forward = f"{past} by"
    inverse = _FORWARD_TO_INVERSE.get(forward, f"{property_segment} of")
    return forward, inverse


def _apply_by_suffix(property_segment: str) -> Tuple[str, str] | None:
    if not property_segment.endswith("_by"):
        return None
    stem = property_segment[:-3]
    if not stem:
        return None
    stem_phrase = _phrase_from_segment(stem)
    forward = f"{stem_phrase} by".strip()
    inverse = _FORWARD_TO_INVERSE.get(forward)
    if inverse is not None:
        return forward, inverse
    stem_words = _tokenize(stem)
    if stem_words:
        agent = _agent_from_verb_token(stem_words[-1])
        return forward, f"{agent} of"
    return forward, f"inverse of {forward}"


def _derive_texts(relation: str) -> Tuple[str, str]:
    segments = relation.split(".")
    if not segments:
        return "relation", "inverse of relation"
    property_segment = segments[-1]
    property_phrase = _phrase_from_segment(property_segment)
    property_words = _tokenize(property_segment)
    override = _apply_property_override(property_segment)
    if override is not None:
        return override
    role_override = _apply_role_noun(property_segment)
    if role_override is not None:
        return role_override
    by_override = _apply_by_suffix(property_segment)
    if by_override is not None:
        return by_override
    forward_text = _build_forward_text(property_phrase, property_words)
    inverse_text = _build_inverse_text(
        property_phrase,
        property_words,
        forward_text=forward_text,
    )
    return forward_text, inverse_text


def _update_entries(entries: List[dict], *, overwrite: bool) -> Tuple[int, int]:
    forward_updates = 0
    inverse_updates = 0
    for entry in entries:
        relation = str(entry.get(_KEY_FORWARD, "")).strip()
        if not relation:
            raise ValueError("inverse_relations entry missing forward field.")
        forward_text, inverse_text = _derive_texts(relation)
        existing_forward = str(entry.get(_KEY_FORWARD_TEXT, "") or entry.get(_KEY_FORWARD_LABEL, "")).strip()
        if overwrite or not existing_forward or existing_forward == relation:
            entry[_KEY_FORWARD_TEXT] = forward_text
            forward_updates += 1
        existing_inverse = str(entry.get(_KEY_INVERSE, "") or entry.get(_KEY_INVERSE_TEXT, "")).strip()
        if overwrite or not existing_inverse:
            entry[_KEY_INVERSE] = inverse_text
            inverse_updates += 1
    return forward_updates, inverse_updates


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill forward/inverse texts for relations.")
    parser.add_argument("--input", type=Path, required=True, help="Path to inverse_relations.json")
    parser.add_argument("--output", type=Path, required=True, help="Path to write updated JSON")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing texts")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)
    init_logging()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    entries = payload.get("inverse_relations")
    if not isinstance(entries, list):
        raise ValueError("inverse_relations.json must contain an inverse_relations list.")
    forward_updates, inverse_updates = _update_entries(entries, overwrite=args.overwrite)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    log_event(
        LOGGER,
        "relation_texts_filled",
        forward_updates=forward_updates,
        inverse_updates=inverse_updates,
        output=str(output_path),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
