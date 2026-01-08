from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Pattern, Tuple

RELATION_ACTION_KEEP = "keep"
RELATION_ACTION_TYPE = "type_feature"
RELATION_ACTION_DROP = "drop"


@dataclass(frozen=True)
class RelationCleaningRules:
    type_exact: frozenset[str]
    type_prefixes: Tuple[str, ...]
    type_regexes: Tuple[Pattern[str], ...]
    drop_exact: frozenset[str]
    drop_prefixes: Tuple[str, ...]
    drop_regexes: Tuple[Pattern[str], ...]


RELATION_TYPE_EXACT = frozenset(
    (
        "common.topic.notable_for",
        "common.topic.notable_types",
        "freebase.type_hints.included_types",
        "freebase.type_profile.strict_included_types",
        "type.object.type",
    )
)
RELATION_TYPE_PREFIXES: Tuple[str, ...] = ()
RELATION_TYPE_REGEXES: Tuple[Pattern[str], ...] = (
    re.compile(r"(?:^|[._])gender(?:[._]|$)"),
    re.compile(r"(?:^|[._])sex(?:[._]|$)"),
    re.compile(r"(?:^|[._])nationality(?:[._]|$)"),
    re.compile(r"(?:^|[._])languages?(?:[._]|$)"),
)
RELATION_DROP_EXACT = frozenset(
    (
        "organization.organization.organization_type",
        "organization.organization_type.organizations_of_this_type",
    )
)
RELATION_DROP_PREFIXES = (
    "common.document.",
    "common.image.",
    "common.topic.",
    "common.webpage.",
    "dataworld.",
    "freebase.type_hints.",
    "freebase.type_profile.",
    "type.domain.",
    "type.extension.",
    "type.object.",
    "type.property.",
    "type.type.",
)
RELATION_DROP_REGEXES: Tuple[Pattern[str], ...] = (
    re.compile(r"^freebase\\.valuenotation\\."),
    re.compile(r"^rdf-schema#"),
)

DEFAULT_RELATION_CLEANING_RULES = RelationCleaningRules(
    type_exact=RELATION_TYPE_EXACT,
    type_prefixes=RELATION_TYPE_PREFIXES,
    type_regexes=RELATION_TYPE_REGEXES,
    drop_exact=RELATION_DROP_EXACT,
    drop_prefixes=RELATION_DROP_PREFIXES,
    drop_regexes=RELATION_DROP_REGEXES,
)


def relation_action(rel: str, rules: RelationCleaningRules, *, enabled: bool) -> str:
    if not enabled:
        return RELATION_ACTION_KEEP
    if rel in rules.drop_exact or any(rel.startswith(prefix) for prefix in rules.drop_prefixes):
        return RELATION_ACTION_DROP
    if any(pattern.search(rel) for pattern in rules.drop_regexes):
        return RELATION_ACTION_DROP
    if rel in rules.type_exact or any(rel.startswith(prefix) for prefix in rules.type_prefixes):
        return RELATION_ACTION_TYPE
    if any(pattern.search(rel) for pattern in rules.type_regexes):
        return RELATION_ACTION_TYPE
    return RELATION_ACTION_KEEP
