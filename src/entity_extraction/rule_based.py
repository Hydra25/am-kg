"""Rule-based (lexicon + regex) entity and value extraction.

This is the deterministic baseline of the pipeline. It is fast, reproducible
and gives exact character offsets; the LLM extractor adds recall on top.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.lexicon import Lexicon
from ..normalization.text import normalize_chars
from ..normalization.units import QuantitySpan, parse_quantities
from ..ontology.models import EntityMention, Evidence, Sentence, stable_id
from ..ontology.schema import EntityType
from .matcher import LexiconMatcher, Match

E = EntityType
VALUE_BEARING = {E.PARAMETER.value, E.PROPERTY.value, E.DEFECT.value, E.MICROSTRUCTURE.value,
                 E.CONDITION.value}
PREV_WINDOW = 90   # max chars between an entity mention and its value (value after mention)
NEXT_WINDOW = 25   # max chars when the value precedes the mention ("200 W laser power")


@dataclass
class ValueAssignment:
    mention_index: int
    qspan: QuantitySpan
    distance: int


@dataclass
class SentenceAnalysis:
    sentence: Sentence
    mentions: list[EntityMention] = field(default_factory=list)
    quantities: list[QuantitySpan] = field(default_factory=list)
    assignments: list[ValueAssignment] = field(default_factory=list)

    def of_type(self, *types: str) -> list[tuple[int, EntityMention]]:
        return [(i, m) for i, m in enumerate(self.mentions) if m.entity_type in types]


def evidence_for(sentence: Sentence, paper_id: str) -> Evidence:
    return Evidence(paper_id=paper_id, page=sentence.page, section=sentence.section_title,
                    section_type=sentence.section_type, sentence_id=sentence.sentence_id,
                    text=sentence.text, source_type=sentence.source_type, table_id=sentence.table_id)


class RuleEntityExtractor:
    def __init__(self, lexicon: Lexicon, normalizer: EntityNormalizer) -> None:
        self.lexicon = lexicon
        self.normalizer = normalizer
        self.matcher = LexiconMatcher(lexicon)

    # ------------------------------------------------------------------
    def paper_abbreviation_matches(self, sentence: str, abbreviations: dict[str, str]) -> list[Match]:
        """Paper-local abbreviations ("... energy input (EI)") whose long form is a lexicon term."""
        out: list[Match] = []
        norm = normalize_chars(sentence)
        for short, long_form in abbreviations.items():
            hits = self.lexicon.lookup_any_type(long_form)
            if len(hits) != 1 or self.lexicon.lookup_any_type(short):
                continue
            etype, cid, conf = hits[0]
            for m in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(short)}(?![A-Za-z0-9])", norm):
                out.append(Match(m.start(), m.end(), sentence[m.start():m.end()], etype, cid, conf * 0.95))
        return out

    def analyze(self, sentence: Sentence, paper_id: str,
                abbreviations: Optional[dict[str, str]] = None) -> SentenceAnalysis:
        text = sentence.text
        matches = self.matcher.find(text)
        if abbreviations:
            extra = [m for m in self.paper_abbreviation_matches(text, abbreviations)
                     if not any(m.start < x.end and x.start < m.end for x in matches)]
            matches = sorted(matches + extra, key=lambda m: m.start)
        quantities = parse_quantities(text)
        # a quantity overlapping a mention (e.g. "18Ni300", "T6") is not a value
        quantities = [q for q in quantities if not any(q.start < m.end and m.start < q.end for m in matches)]

        ev = evidence_for(sentence, paper_id)
        mentions: list[EntityMention] = []
        for mt in matches:
            norm = self.normalizer.normalize(mt.text, mt.entity_type)
            conf = min(mt.confidence, norm.confidence)
            if sentence.source_type == "table":
                conf -= 0.03
            mentions.append(EntityMention(
                mention_id=stable_id("m", sentence.sentence_id, mt.start, mt.end),
                text=mt.text, entity_type=norm.entity_type, canonical_id=norm.canonical_id,
                canonical_name=norm.canonical_name, normalization_confidence=norm.confidence,
                normalization_method=norm.method, sentence_id=sentence.sentence_id,
                char_start=mt.start, char_end=mt.end, confidence=round(conf, 3),
                method="rule", evidence=ev,
            ))
        analysis = SentenceAnalysis(sentence=sentence, mentions=mentions, quantities=quantities)
        analysis.assignments = self.assign_values(analysis)
        for a in analysis.assignments:
            m = analysis.mentions[a.mention_index]
            if m.quantity is None:
                m.quantity = a.qspan.quantity
        return analysis

    # ------------------------------------------------------------------
    def _compatible(self, mention: EntityMention, qs: QuantitySpan) -> bool:
        dims = self.lexicon.dimensions(mention.canonical_id or "")
        if not dims:
            return False
        return qs.quantity.dimension in dims

    def assign_values(self, analysis: SentenceAnalysis) -> list[ValueAssignment]:
        """Attach each value to the nearest dimension-compatible entity mention.

        "laser power, scanning speed and layer thickness of 200 W, 800 mm/s and
        30 um" is resolved correctly because only dimension-compatible mentions
        are candidates (W->power, mm/s->velocity, um->length).
        """
        out: list[ValueAssignment] = []
        text = analysis.sentence.text
        for qs in analysis.quantities:
            if qs.quantity.dimension is None:
                continue
            best: Optional[tuple[int, int]] = None
            for i, m in enumerate(analysis.mentions):
                if m.entity_type not in VALUE_BEARING or not self._compatible(m, qs):
                    continue
                if m.char_end is None or m.char_start is None:
                    continue
                if m.char_end <= qs.start:
                    dist = qs.start - m.char_end
                    if dist > PREV_WINDOW or ";" in text[m.char_end:qs.start]:
                        continue
                elif m.char_start >= qs.end:
                    dist = (m.char_start - qs.end) + 15   # prefer preceding mentions
                    if dist - 15 > NEXT_WINDOW:
                        continue
                else:
                    continue
                if best is None or dist < best[1]:
                    best = (i, dist)
            if best is not None:
                out.append(ValueAssignment(best[0], qs, best[1]))
        return out
