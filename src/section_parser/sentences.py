"""Rule-based sentence splitter tuned for scientific text.

Avoids splitting after common abbreviations (Fig., et al., e.g., approx.),
initials, and decimal numbers. Returns character offsets so that page numbers
can be mapped back to each sentence.
"""
from __future__ import annotations

import re

_ABBREV = {
    "fig", "figs", "eq", "eqs", "ref", "refs", "al", "e.g", "i.e", "approx", "vs", "no",
    "ca", "resp", "cf", "etc", "sect", "sec", "tab", "vol", "wt", "at", "max", "min",
    "avg", "dr", "prof", "mr", "ms", "st", "viz", "ed", "eds", "pp", "p", "incl",
}
_BOUNDARY = re.compile(r"[.!?][\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9Α-Ωα-ω])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences in `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.start() + 1
        prev = text[start:end]
        if re.fullmatch(r"\s*(?:Table|Fig\.?|Figure)\s*\d+[a-z]?\.", prev):
            continue
        last_tok = re.search(r"(?:^|[\s(])([A-Za-z.]+)\.$", prev)
        if last_tok:
            tok = last_tok.group(1).lower().rstrip(".")
            if tok in _ABBREV or (len(tok) == 1 and tok.isalpha()):
                continue
        # do not split "... Fig. 2 ..." style or a number list like "1. 2."
        if re.search(r"\b(?:Fig|Figs|Table|Eq|Ref)\.$", prev):
            continue
        seg = text[start:end].strip()
        if len(seg) < 3:
            continue
        spans.append(_strip_span(text, start, m.end()))
        start = m.end()
    if text[start:].strip():
        spans.append(_strip_span(text, start, len(text)))
    return [s for s in spans if s[1] > s[0]]


def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e
