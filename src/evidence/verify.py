"""Evidence verification: is the stored evidence really in the source paper?

This is what lets the system answer "why does the KG believe this?" with a
verbatim, checkable quote (paper -> page -> section -> sentence/table row).
"""
from __future__ import annotations

from typing import Optional

from ..normalization.text import clean_whitespace, normalize_chars
from ..ontology.models import Evidence, ParsedDocument


def _norm(t: str) -> str:
    return clean_whitespace(normalize_chars(t)).lower()


class EvidenceIndex:
    def __init__(self, doc: ParsedDocument) -> None:
        self.doc = doc
        self.sentences = doc.sentence_map()
        self.title = doc.paper.title or ""

    def verify(self, ev: Optional[Evidence]) -> tuple[bool, str]:
        if ev is None or not (ev.text or "").strip():
            return False, "missing_evidence"
        if ev.paper_id != self.doc.paper.paper_id:
            return False, "evidence_paper_mismatch"
        if ev.sentence_id is None:
            if ev.section and ev.section.lower() == "title" and _norm(ev.text) == _norm(self.title):
                return True, "ok_title"
            return False, "evidence_without_sentence_id"
        s = self.sentences.get(ev.sentence_id)
        if s is None:
            return False, "evidence_sentence_not_found"
        parts = [self.sentences[c].text for c in ev.context_sentence_ids if c in self.sentences]
        if len(parts) != len(ev.context_sentence_ids):
            return False, "context_sentence_not_found"
        expected = " ".join(parts + [s.text])
        if _norm(ev.text) != _norm(expected):
            return False, "evidence_text_mismatch"
        if ev.page is not None and s.page is not None and ev.page != s.page:
            return False, "evidence_page_mismatch"
        return True, "ok"


def explain_evidence(ev: Evidence, title: Optional[str] = None) -> str:
    """Human-readable provenance line."""
    where = [f"paper {ev.paper_id}" + (f" ('{title}')" if title else "")]
    if ev.page:
        where.append(f"page {ev.page}")
    if ev.section:
        where.append(f"section '{ev.section}'")
    if ev.source_type == "table":
        where.append(f"table {ev.table_id}")
    return ", ".join(where) + f'\n  "{ev.text}"'
