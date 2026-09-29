"""Loads config/lexicon.yaml into canonical entities and an alias index."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from ..ontology.models import CanonicalEntity
from ..ontology.schema import EntityType
from .text import match_key

DEFAULT_ALIAS_CONFIDENCE = 0.95


def canonical_id(entity_type: str, canonical_name: str) -> str:
    return f"{entity_type}:{canonical_name}"


@dataclass
class AliasEntry:
    text: str
    entity_type: str
    canonical_id: str
    confidence: float
    case_sensitive: bool


@dataclass
class Lexicon:
    entities: dict[str, CanonicalEntity] = field(default_factory=dict)
    aliases: list[AliasEntry] = field(default_factory=list)
    index: dict[tuple[str, str], list[tuple[str, float]]] = field(default_factory=dict)
    file_hash: Optional[str] = None

    @classmethod
    def load(cls, path: Path) -> "Lexicon":
        raw_text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text) or {}
        lex = cls(file_hash=hashlib.sha1(raw_text.encode()).hexdigest()[:12])
        valid_types = {t.value for t in EntityType.concept_types()}
        for etype, items in data.items():
            if etype not in valid_types:
                raise ValueError(f"Unknown entity type in lexicon: {etype}")
            for item in items or []:
                lex._add_entry(etype, item)
        return lex

    def _add_entry(self, etype: str, item: dict) -> None:
        name = item["canonical"]
        cid = canonical_id(etype, name)
        attrs = {k: v for k, v in item.items() if k not in {"canonical", "aliases", "family", "display"}}
        ent = CanonicalEntity(
            id=cid, entity_type=etype, canonical_name=name,
            display_name=item.get("display"), family=item.get("family"),
            attributes=attrs, aliases=[], normalization_confidence=1.0, in_lexicon=True,
        )
        alias_items = [name] + list(item.get("aliases") or [])
        seen: set[str] = set()
        for a in alias_items:
            if isinstance(a, dict):
                text, conf = str(a["text"]), float(a.get("confidence", DEFAULT_ALIAS_CONFIDENCE))
                cs = a.get("case_sensitive")
            else:
                text, conf, cs = str(a), (1.0 if str(a) == name else DEFAULT_ALIAS_CONFIDENCE), None
            if text == name:
                conf = 1.0
            if cs is None:  # short ALL-CAPS acronyms are case-sensitive (CT, HIP, SEM ...)
                letters = [c for c in text if c.isalpha()]
                cs = bool(letters) and text.upper() == text and len(text) <= 5
            if text in seen:
                continue
            seen.add(text)
            if text != name:
                ent.aliases.append(text)
            self.aliases.append(AliasEntry(text, etype, cid, conf, bool(cs)))
            key = (etype, match_key(text))
            bucket = self.index.setdefault(key, [])
            if all(c != cid for c, _ in bucket):
                bucket.append((cid, conf))
        self.entities[cid] = ent

    # ------------------------------------------------------------------
    def lookup(self, surface: str, entity_type: str) -> Optional[tuple[str, float]]:
        hits = self.index.get((entity_type, match_key(surface)))
        if not hits:
            return None
        return max(hits, key=lambda h: h[1])

    def lookup_any_type(self, surface: str) -> list[tuple[str, str, float]]:
        key = match_key(surface)
        out = []
        for (etype, k), hits in self.index.items():
            if k == key:
                out.extend((etype, cid, conf) for cid, conf in hits)
        return out

    def dimensions(self, cid: str) -> list[str]:
        ent = self.entities.get(cid)
        if not ent:
            return []
        return list(ent.attributes.get("dimensions") or [])

    def by_type(self, entity_type: str) -> list[CanonicalEntity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]
