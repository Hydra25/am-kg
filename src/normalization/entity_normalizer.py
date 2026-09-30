"""Entity normalisation: surface form -> canonical entity, with a confidence.

Resolution order (first hit wins):
  1. manual overrides / merges from the review UI       (confidence 1.0)
  2. exact lexicon alias for the given type             (alias confidence)
  3. paper-local abbreviation ("volumetric energy density (VED)")
  4. exact lexicon alias under a *different* type        (x0.85, re-typed)
  5. fuzzy match against same-type aliases (difflib)     (ratio x0.85)
  6. optional LLM choice among fuzzy candidates          (<= 0.8)
  7. new, unreviewed canonical entity                     (0.5 -> review)

Entities are never merged silently below the fuzzy threshold: unknown terms
become *new* entities flagged for review, which a human can later merge.
"""
from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..ontology.models import CanonicalEntity
from .lexicon import Lexicon, canonical_id
from .text import clean_whitespace, match_key, normalize_chars, singularize

log = logging.getLogger(__name__)

FUZZY_THRESHOLD = 0.88
_GENERIC_HEAD = re.compile(r"\s+(defects?|phenomen(on|a)|formation|content|level|values?|behaviou?r|"
                           r"characteristics?|features?)$", re.I)
NEW_ENTITY_CONFIDENCE = 0.5


@dataclass
class NormResult:
    canonical_id: str
    canonical_name: str
    entity_type: str
    confidence: float
    method: str
    in_lexicon: bool


class NormalizationOverrides:
    """Manual corrections persisted as JSON (edited through the review UI)."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self.aliases: dict[str, str] = {}   # "Type|key" -> canonical_id
        self.merges: dict[str, str] = {}    # source canonical_id -> target canonical_id
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.aliases = data.get("aliases", {})
            self.merges = data.get("merges", {})

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"aliases": self.aliases, "merges": self.merges},
                                        indent=2, ensure_ascii=False), encoding="utf-8")

    def add_alias(self, entity_type: str, surface: str, target_cid: str) -> None:
        self.aliases[f"{entity_type}|{match_key(surface)}"] = target_cid

    def add_merge(self, source_cid: str, target_cid: str) -> None:
        if source_cid == target_cid:
            return
        self.merges[source_cid] = target_cid

    def resolve(self, cid: str) -> str:
        seen = set()
        while cid in self.merges and cid not in seen:
            seen.add(cid)
            cid = self.merges[cid]
        return cid


class EntityNormalizer:
    def __init__(self, lexicon: Lexicon, overrides: Optional[NormalizationOverrides] = None,
                 llm_chooser: Optional[Callable[[str, str, list[str]], Optional[str]]] = None) -> None:
        self.lexicon = lexicon
        self.overrides = overrides or NormalizationOverrides()
        self.llm_chooser = llm_chooser
        self.new_entities: dict[str, CanonicalEntity] = {}
        # per-paper context, set by the pipeline for each paper
        self.paper_abbreviations: dict[str, str] = {}
        self.paper_symbols: dict[str, list[str]] = {}

    def set_paper_context(self, abbreviations: Optional[dict[str, str]] = None,
                          symbols: Optional[dict[str, list[str]]] = None) -> None:
        """Abbreviations ("VED") and symbols ("P", "v", "h") defined in the current paper."""
        self.paper_abbreviations = dict(abbreviations or {})
        self.paper_symbols = dict(symbols or {})

    def resolve_symbol(self, surface: str, entity_type: Optional[str] = None) -> Optional[tuple[str, float]]:
        """'P' -> laser power if the paper defines "laser power (P)" or "P is the laser power".
        A symbol the paper defines as two different lexicon terms is ambiguous -> None."""
        hits: dict[str, float] = {}
        for long_form in self.paper_symbols.get(surface.strip(), []):
            words = long_form.split()
            for n in range(min(len(words), 5), 0, -1):      # longest trailing phrase first
                phrase = " ".join(words[-n:])
                hit = (self.lexicon.lookup(phrase, entity_type) if entity_type
                       else next(iter((h[1], h[2]) for h in self.lexicon.lookup_any_type(phrase)), None))
                if hit:
                    hits[hit[0]] = max(hit[1], hits.get(hit[0], 0.0))
                    break
        if len(hits) != 1:
            return None
        return next(iter(hits.items()))

    # ------------------------------------------------------------------
    def entity(self, cid: str) -> Optional[CanonicalEntity]:
        return self.lexicon.entities.get(cid) or self.new_entities.get(cid)

    def _result(self, cid: str, conf: float, method: str) -> NormResult:
        cid = self.overrides.resolve(cid)
        ent = self.entity(cid)
        if ent is None:  # override points at an entity created in another run
            etype, _, name = cid.partition(":")
            ent = self._create_new(etype, name, register_only=True)
        return NormResult(cid, ent.canonical_name, ent.entity_type, round(conf, 3), method, ent.in_lexicon)

    def normalize(self, surface: str, entity_type: str,
                  local_abbreviations: Optional[dict[str, str]] = None) -> NormResult:
        surface = clean_whitespace(normalize_chars(surface))
        key = match_key(surface)

        # 1. manual override
        ov = self.overrides.aliases.get(f"{entity_type}|{key}")
        if ov:
            return self._result(ov, 1.0, "manual_override")
        # 2. exact alias, same type
        hit = self.lexicon.lookup(surface, entity_type)
        if hit:
            return self._result(hit[0], hit[1], "lexicon")
        # 3. paper-local abbreviation definition
        abbreviations = local_abbreviations if local_abbreviations is not None else self.paper_abbreviations
        if abbreviations and surface in abbreviations:
            long_form = abbreviations[surface]
            hit = self.lexicon.lookup(long_form, entity_type)
            if hit:
                return self._result(hit[0], hit[1] * 0.95, "local_abbreviation")
        # 3a. paper-local symbol ("P", "v", "h", "E" defined in the text)
        if surface in self.paper_symbols:
            hit = self.resolve_symbol(surface, entity_type)
            if hit:
                return self._result(hit[0], hit[1] * 0.9, "paper_symbol")
        # 3b. plural / singular variants
        sing = singularize(surface)
        if sing != surface:
            hit = self.lexicon.lookup(sing, entity_type)
            if hit:
                return self._result(hit[0], hit[1] * 0.97, "lexicon_singular")
        # 3c. generic head nouns: "lack of fusion defects" -> "lack of fusion"
        stripped = _GENERIC_HEAD.sub("", surface).strip()
        if stripped and stripped != surface:
            hit = self.lexicon.lookup(stripped, entity_type) or self.lexicon.lookup(singularize(stripped), entity_type)
            if hit:
                return self._result(hit[0], hit[1] * 0.95, "lexicon_head_noun_stripped")
        # 4. exact alias under another type (e.g. LLM typed porosity as Property)
        others = [h for h in self.lexicon.lookup_any_type(surface) if h[0] != entity_type]
        if len(others) == 1:
            etype, cid, conf = others[0]
            return self._result(cid, conf * 0.85, f"lexicon_retyped_from_{entity_type}")
        # 5. fuzzy
        candidates = self._fuzzy_candidates(surface, entity_type)
        if (candidates and candidates[0][1] >= FUZZY_THRESHOLD
                and fuzzy_compatible(surface, self._best_alias.get(candidates[0][0], ""))):
            cid, ratio = candidates[0]
            return self._result(cid, ratio * 0.85, "fuzzy")
        # 6. LLM choice among candidates (restricted list -> no invented entities)
        # Only candidates whose head noun occurs in the term are offered: the small
        # LLM otherwise maps e.g. "mechanical properties" -> "hardness" or
        # "titanium" -> "Ti-6Al-4V" (seen on a real paper). No candidate left = no call.
        candidates = [(c, r) for c, r in candidates if head_compatible(surface, self.lexicon.entities[c].canonical_name)]
        if self.llm_chooser and candidates:
            names = [self.lexicon.entities[c].canonical_name for c, _ in candidates[:5]]
            try:
                choice = self.llm_chooser(surface, entity_type, names)
            except Exception as exc:  # pragma: no cover - network errors
                log.warning("LLM normalisation failed for %r: %s", surface, exc)
                choice = None
            if choice and choice in names:
                return self._result(canonical_id(entity_type, choice), 0.7, "llm_candidate_choice")
        # 7. new entity
        ent = self._create_new(entity_type, self._clean_new_name(surface))
        return NormResult(ent.id, ent.canonical_name, entity_type, NEW_ENTITY_CONFIDENCE, "new_entity", False)

    # ------------------------------------------------------------------
    def _fuzzy_candidates(self, surface: str, entity_type: str) -> list[tuple[str, float]]:
        key = match_key(surface)
        if len(key) < 4:  # short strings/acronyms: fuzzy matching is unreliable
            return []
        best: dict[str, float] = {}
        self._best_alias: dict[str, str] = {}      # canonical id -> the alias that matched best
        for alias in self.lexicon.aliases:
            if alias.entity_type != entity_type:
                continue
            akey = match_key(alias.text)
            if len(akey) < 4:
                continue
            r = difflib.SequenceMatcher(None, key, akey).ratio() * min(1.0, alias.confidence + 0.05)
            if r > best.get(alias.canonical_id, 0.0):
                best[alias.canonical_id] = r
                self._best_alias[alias.canonical_id] = alias.text
        return sorted(best.items(), key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def _clean_new_name(surface: str) -> str:
        s = surface.strip(" .,;:()[]")
        if not re.search(r"[A-Z]{2,}|\d", s):   # keep acronyms / alloy designations
            s = s.lower()
        words = s.split()
        if words:
            words[-1] = singularize(words[-1])
        return " ".join(words)

    def _create_new(self, entity_type: str, name: str, register_only: bool = False) -> CanonicalEntity:
        cid = canonical_id(entity_type, name)
        if cid not in self.new_entities:
            self.new_entities[cid] = CanonicalEntity(
                id=cid, entity_type=entity_type, canonical_name=name, aliases=[],
                normalization_confidence=NEW_ENTITY_CONFIDENCE if not register_only else 1.0,
                in_lexicon=False,
            )
        return self.new_entities[cid]


_HEAD_STOP = {"of", "and", "the", "a", "an", "in", "on", "for", "with", "by", "to", "at"}


def _tokens(text: str) -> list[str]:
    return [singularize(t) for t in re.findall(r"[a-z0-9]+", normalize_chars(text).lower()) if t not in _HEAD_STOP]


def head_compatible(term: str, candidate: str) -> bool:
    """True if the candidate's head noun (its last content word) occurs in the term.
    "tensile strength" is compatible with "ultimate tensile strength" and "strength",
    but not with "tensile properties"; "heat treatment" is compatible with
    "treatment at 800 °C"."""
    cand = _tokens(candidate)
    return bool(cand) and cand[-1] in set(_tokens(term))


def _fuzzy_tokens(text: str) -> list[str]:
    """Like _tokens, but keeps Greek letters and primes: α' (martensite) is not α."""
    t = normalize_chars(text).lower().replace("′", "'")
    return [singularize(w) for w in re.findall(r"[a-z0-9α-ω']+", t) if w not in _HEAD_STOP]


def _similar(a: str, b: str) -> bool:
    if a == b:
        return True
    # a spelling variant keeps the beginning of the word: "tomograpy" ~ "tomography",
    # but "toughness" !~ "roughness" and "macrostructure" !~ "microstructure"
    return a[:3] == b[:3] and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


# measurement-location / statistic words that make a term more specific without changing
# what it is: "top surface roughness" is still surface roughness
_QUALIFIERS = {"top", "side", "bottom", "lateral", "upper", "lower", "upskin", "downskin", "average", "mean",
               "overall", "total", "measured", "final", "maximum", "minimum", "max", "min", "bulk", "local"}


def fuzzy_compatible(term: str, candidate: str) -> bool:
    """Guard for fuzzy (string-similarity) matches. Every word of the term must have a
    spelling-variant counterpart in the candidate and vice versa. Blocks the wrong
    mappings seen on real papers: toughness -> surface roughness, α' phase -> α phase,
    macrostructure -> microstructure, α+β microstructure -> microstructure."""
    a, b = _fuzzy_tokens(term), _fuzzy_tokens(candidate)
    if not a or not b:
        return False
    if "".join(a) == "".join(b):          # "key hole" vs "keyhole"
        return True
    return (all(x in _QUALIFIERS or any(_similar(x, y) for y in b) for x in a)
            and all(any(_similar(y, x) for x in a) for y in b))


_ABBREV_RE = re.compile(r"(?P<long>[A-Za-z][\w\-α-ω ]{3,80}?)\s*\((?P<short>[A-Z][A-Za-z0-9\-/]{1,9})\)")


def find_abbreviations(text: str) -> dict[str, str]:
    """Schwartz–Hearst-style detection of 'long form (SHORT)' definitions."""
    out: dict[str, str] = {}
    for m in _ABBREV_RE.finditer(normalize_chars(text)):
        short = m.group("short")
        words = m.group("long").split()
        letters = [c.lower() for c in short if c.isalpha()]
        if not letters:
            continue
        # take the minimal number of trailing words whose initials cover the short form
        for n in range(1, min(len(words), len(letters) + 2) + 1):
            cand = words[-n:]
            if cand[0][0].lower() == letters[0]:
                initials = "".join(w[0].lower() for w in cand if w)
                if all(ch in initials or ch in " ".join(cand).lower() for ch in letters):
                    out[short] = " ".join(cand)
                    break
    return out


# "laser power (P)", "scanning speed (v)", "hatch distance (h)"
_SYM_PAREN_RE = re.compile(r"(?P<long>[A-Za-z][A-Za-z\- ]{2,60}?)\s*\(\s*(?P<sym>[A-Za-z][a-z]?)\s*\)")
# "where P is the laser power, v is the scanning speed ...", "P: laser power", "P = laser power"
_SYM_DEF_RE = re.compile(r"(?<![\w'])(?P<sym>[A-Za-z][a-z]?)(?:\s+(?:is|denotes|represents)\s+|\s*[:=]\s*)(?:the\s+)?"
                         r"(?P<long>[A-Za-z][A-Za-z\- ]{2,50}?)(?=\s*(?:[,;.(\[]|\band\b|\bin\b|\bwith\b|$))")


def find_symbol_definitions(text: str) -> dict[str, list[str]]:
    """Single-letter (or letter + subscript letter) symbols defined in the text.

    Returns symbol -> candidate long forms in order of appearance. The long forms are
    raw text; EntityNormalizer.resolve_symbol only accepts one that ends in a lexicon
    term, so "It is a common process" never turns "It" into anything."""
    found: list[tuple[int, str, str]] = []
    norm = normalize_chars(text)
    for rx in (_SYM_PAREN_RE, _SYM_DEF_RE):
        for m in rx.finditer(norm):
            sym, long_form = m.group("sym"), clean_whitespace(m.group("long"))
            if sym.lower() in {"a", "an", "i", "it", "is", "in", "on", "of", "or", "to", "as", "at", "by", "we"}:
                continue
            found.append((m.start(), sym, long_form))
    out: dict[str, list[str]] = {}
    for _pos, sym, long_form in sorted(found):
        lst = out.setdefault(sym, [])
        if long_form not in lst:
            lst.append(long_form)
    return out
