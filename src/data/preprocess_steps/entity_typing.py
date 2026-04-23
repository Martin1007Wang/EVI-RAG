from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityTyping:
    kb_prefixes: tuple[str, ...] = ("m.", "g.")
    def is_text_entity(self, entity: str) -> bool:
        return not entity.startswith(self.kb_prefixes)
    def is_cvt_entity(self, entity: str) -> bool:
        return entity.startswith(self.kb_prefixes)