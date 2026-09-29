"""Fast dictionary matcher: finds lexicon aliases in a sentence.

Longest match wins; overlapping shorter matches are dropped ("keyhole
porosity" beats "porosity"). Short ALL-CAPS acronyms are case-sensitive.
Matching runs on a length-preserving normalised copy of the sentence, so the
returned offsets are valid on the verbatim sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..normalization.lexicon import AliasEntry, Lexicon
from ..normalization.text import normalize_chars

_SEP = r"[ \-]"


@dataclass
class Match:
    start: int
    end: int
    text: str
    entity_type: str
    canonical_id: str
    confidence: float


def _alias_pattern(alias: str, entity_type: str) -> str:
    a = normalize_chars(alias)
    parts = re.split(r"[ \-]+", a)
    sep = r"[ \-]?" if entity_type == "Material" else _SEP
    body = sep.join(re.escape(p) for p in parts if p)
    # allow simple plurals for multi-letter, non-acronym words
    plural = "(?:s|es)?" if a[-1:].isalpha() and not a.isupper() and len(a) > 3 else ""
    if plural and a.endswith("y") and not a.endswith(("ay", "ey", "oy", "uy")):
        body, plural = body[:-1], "(?:y|ies)"
    return rf"(?<![A-Za-z0-9α-ω]){body}{plural}(?![A-Za-z0-9α-ω])"


class LexiconMatcher:
    def __init__(self, lexicon: Lexicon) -> None:
        self.lexicon = lexicon
        entries = sorted(lexicon.aliases, key=lambda a: len(a.text), reverse=True)
        self._cs: list[tuple[re.Pattern, AliasEntry]] = []
        self._ci: list[tuple[re.Pattern, AliasEntry]] = []
        for e in entries:
            pat = _alias_pattern(e.text, e.entity_type)
            if e.case_sensitive:
                self._cs.append((re.compile(pat), e))
            else:
                self._ci.append((re.compile(pat, re.IGNORECASE), e))

    def find(self, sentence: str) -> list[Match]:
        norm = normalize_chars(sentence)
        cands: list[Match] = []
        for pat, e in self._cs + self._ci:
            for m in pat.finditer(norm):
                cands.append(Match(m.start(), m.end(), sentence[m.start():m.end()], e.entity_type,
                                   e.canonical_id, e.confidence))
        # longest-first, then earliest; drop overlaps
        cands.sort(key=lambda m: (-(m.end - m.start), m.start, -m.confidence))
        chosen: list[Match] = []
        occupied: list[tuple[int, int]] = []
        for c in cands:
            if any(c.start < b and a < c.end for a, b in occupied):
                continue
            chosen.append(c)
            occupied.append((c.start, c.end))
        return sorted(chosen, key=lambda m: m.start)
