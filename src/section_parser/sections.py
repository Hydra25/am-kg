"""Scientific section detection and document assembly.

Input: a flat list of text lines (with page, font size, boldness) produced by a
PDF parser. Output: sections with canonical types and page-tagged sentences.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..ontology.models import Section, Sentence, stable_id
from ..ontology.schema import SectionType
from .sentences import split_sentences

S = SectionType

_SECTION_KEYWORDS: list[tuple[SectionType, tuple[str, ...]]] = [
    (S.ABSTRACT, ("abstract",)),
    (S.REFERENCES, ("references", "bibliography", "literature cited", "reference list")),
    (S.ACKNOWLEDGEMENTS, ("acknowledg", "funding", "declaration of competing", "competing interest",
                          "credit author", "author contribution", "data availability", "conflict of interest")),
    (S.RESULTS_DISCUSSION, ("results and discussion", "results & discussion", "result and discussion")),
    (S.INTRODUCTION, ("introduction", "background", "literature review", "state of the art")),
    (S.RESULTS, ("results", "result")),
    (S.MATERIALS_METHODS, ("material and method", "materials and method", "methods", "methodology",
                           "experimental", "experiment", "procedure", "setup", "set-up", "materials",
                           "fabrication", "sample preparation", "specimen preparation", "process parameters",
                           "characterization", "characterisation", "numerical model", "simulation")),
    (S.DISCUSSION, ("discussion",)),
    (S.CONCLUSION, ("conclusion", "concluding", "summary", "outlook")),
]
_NUMBERING = re.compile(r"^\s*(?P<num>(?:\d+(?:\.\d+)*\.?)|(?:[IVX]+\.))\s+(?P<rest>\S.*)$")


_LETTER_SPACED = re.compile(r"^(?:[A-Za-z] ){2,}[A-Za-z](?:\s{2,}(?:[A-Za-z] ){1,}[A-Za-z])*$")


def classify_heading(title: str) -> Optional[SectionType]:
    t = title.lower().strip()
    if _LETTER_SPACED.match(title.strip()):
        # Elsevier style "A B S T R A C T", "A R T I C L E  I N F O"
        t = re.sub(r"\s{2,}", "  ", t)
        t = " ".join(w.replace(" ", "") for w in t.split("  "))
        if t.startswith("article info"):
            return S.FRONT_MATTER
    m = _NUMBERING.match(t)
    if m:
        t = m.group("rest")
    t = re.sub(r"[^a-z& ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    short = len(t.split()) <= 6          # a keyword deep inside a long line is not a heading type
    for stype, keys in _SECTION_KEYWORDS:
        if any(t.startswith(k) or (short and f" {k}" in f" {t}") for k in keys):
            return stype
    return None


@dataclass
class Line:
    text: str
    page: int                     # 1-based
    size: float = 10.0
    bold: bool = False
    y0: float = 0.0
    is_heading: Optional[bool] = None   # forced by some parsers (text/markdown)
    heading_level: int = 1


@dataclass
class _RawSection:
    title: str
    stype: SectionType
    page_start: int
    parts: list[tuple[str, int]] = field(default_factory=list)  # (text, page)


def body_font_size(lines: list[Line]) -> float:
    c: Counter[float] = Counter()
    for ln in lines:
        c[round(ln.size, 1)] += len(ln.text)
    return c.most_common(1)[0][0] if c else 10.0


def looks_like_heading(line: Line, body: float) -> bool:
    if line.is_heading is not None:
        return line.is_heading
    t = line.text.strip()
    if _LETTER_SPACED.match(t) and classify_heading(t) is not None:
        return True                     # "A B S T R A C T"
    if not t or len(t) > 90 or len(t.split()) > 12 or not re.search(r"[A-Za-z]{3,}", t):
        return False
    if re.match(r"^(fig(ure)?|table)\s*\d", t, re.I):
        return False
    if t.endswith((".", ",", ";", ":")) and not _NUMBERING.match(t):
        if not classify_heading(t.rstrip(".:")):
            return False
    bigger = line.size >= body + 0.8
    numbered = bool(_NUMBERING.match(t)) and t[0].isdigit() and not re.match(r"^\d+(\.\d+)?\s*(%|mm|μm|W|MPa)", t)
    keyword = classify_heading(t) is not None and len(t.split()) <= 6
    if bigger and (numbered or keyword or line.bold or t[0].isupper()):
        return True
    if line.bold and (numbered or keyword):
        return True
    if numbered and keyword:
        return True
    # bare keyword on its own line (e.g. "Abstract", "References")
    if keyword and len(t.split()) <= 3 and (line.bold or bigger or t.isupper() or t.istitle()):
        return True
    return False


_COMPOUND_PREFIXES = {
    "as", "heat", "lack", "of", "non", "micro", "nano", "post", "pre", "re", "co", "self", "high", "low", "multi",
    "sub", "inter", "intra", "cross", "well", "fine", "coarse", "laser", "electron", "wire", "powder", "layer",
    "build", "near", "over", "under", "semi", "anti", "ultra", "single", "double", "two", "three", "one", "x",
    "l", "pbf", "ded", "ti", "al", "ni", "fe", "cu", "in", "on", "out", "air", "gas", "water", "time",
    "state", "long", "short", "top", "bottom", "side", "open", "closed", "dog", "thin", "thick", "melt", "keyhole",
}


def dehyphenate_join(parts: list[tuple[str, int]]) -> tuple[str, list[tuple[int, int]]]:
    """Join lines, fixing end-of-line hyphenation. Returns text and (offset, page) marks."""
    buf: list[str] = []
    marks: list[tuple[int, int]] = []
    length = 0
    for text, page in parts:
        t = text.strip()
        if not t:
            continue
        if buf:
            prev = buf[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalnum():
                last_word = re.split(r"[\s(/]", prev[:-1])[-1].lower()
                if t[:1].islower() and last_word not in _COMPOUND_PREFIXES:
                    buf[-1] = prev[:-1]          # "fabri-\ncated" -> "fabricated"
                    length -= 1
                # else keep the hyphen, no space: "micro-\nCT", "as-\nbuilt"
            else:
                buf.append(" ")
                length += 1
        marks.append((length, page))
        buf.append(t)
        length += len(t)
    return "".join(buf), marks


def _page_at(marks: list[tuple[int, int]], offset: int) -> Optional[int]:
    page = None
    for off, p in marks:
        if off <= offset:
            page = p
        else:
            break
    return page


def build_sections(lines: list[Line], paper_id: str, title: Optional[str] = None,
                   start_index: int = 0) -> tuple[list[Section], list[Sentence]]:
    """Group lines into typed sections and split them into sentences."""
    body = body_font_size(lines)
    raw: list[_RawSection] = []
    current = _RawSection("Front matter", S.FRONT_MATTER, lines[0].page if lines else 1)
    last_top_type = S.FRONT_MATTER
    top_types: dict[str, SectionType] = {}
    title_norm = (title or "").strip().lower()
    body_started = False   # becomes True at Abstract / Introduction / section "1"

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        if title_norm and text.lower() in title_norm and current.stype == S.FRONT_MATTER:
            continue  # title lines are stored as metadata, not body text
        if looks_like_heading(ln, body):
            stype = classify_heading(text)
            m = _NUMBERING.match(text)
            num = m.group("num").rstrip(".") if m else None
            if (stype in (S.ABSTRACT, S.INTRODUCTION) or (num and (num.split(".")[0] == "1" or stype is not None))
                    or ln.is_heading is True           # explicit headings (markdown/text input) are trusted
                    or (ln.page >= 2 and stype in (S.MATERIALS_METHODS, S.RESULTS, S.RESULTS_DISCUSSION,
                                                   S.DISCUSSION, S.CONCLUSION))):   # e.g. Nature: no "Introduction"
                body_started = True
            elif not body_started and stype not in (S.REFERENCES, S.ACKNOWLEDGEMENTS):
                # Before the abstract/introduction every "heading" is front matter: journal
                # logos ("materials"), author lines, affiliations, "ARTICLE INFO" boxes.
                stype = S.FRONT_MATTER
            if stype is None:
                # unclassified subsection inherits its parent's type ("2.1 Powder" -> methods)
                parent = num.split(".")[0] if num and "." in num else None
                stype = top_types.get(parent or "", last_top_type)
                if current.stype == S.FRONT_MATTER and stype == S.FRONT_MATTER:
                    stype = S.OTHER
            if num and "." not in num:
                top_types[num] = stype
            if not (num and "." in num):
                last_top_type = stype
            if stype == S.FRONT_MATTER and current.stype == S.FRONT_MATTER:
                current.parts.append((text, ln.page))   # stay in one front-matter block
                continue
            if current.parts or current.stype != S.FRONT_MATTER:
                raw.append(current)
            current = _RawSection(text, stype, ln.page)
            # "Abstract: text on same line" is handled by parsers splitting lines
            continue
        # inline "Abstract" prefix without a separate heading line
        if current.stype == S.FRONT_MATTER and re.match(r"^abstract[\s.:—-]+\S", text, re.I):
            raw.append(current)
            current = _RawSection("Abstract", S.ABSTRACT, ln.page)
            text = re.sub(r"^abstract[\s.:—-]+", "", text, flags=re.I)
        current.parts.append((text, ln.page))
    raw.append(current)

    # A long front-matter block holds the abstract (and, in Nature-style papers without an
    # "Introduction" heading, the introduction too). Keep it as content, not front matter.
    kinds = {r.stype for r in raw}
    for r in raw:
        if r.stype == S.FRONT_MATTER:
            size = sum(len(t) for t, _ in r.parts)
            if size > 1500:
                if S.ABSTRACT not in kinds and size < 2500:
                    r.stype = S.ABSTRACT
                elif S.INTRODUCTION not in kinds:
                    r.stype = S.INTRODUCTION
                else:
                    r.stype = S.OTHER
                kinds.add(r.stype)

    sections: list[Section] = []
    sentences: list[Sentence] = []
    idx = start_index
    for n, rs in enumerate(raw):
        if not rs.parts:
            continue
        sec_id = f"{paper_id}_sec{n:02d}"
        text, marks = dehyphenate_join(rs.parts)
        sec = Section(section_id=sec_id, title=rs.title, section_type=rs.stype.value,
                      page_start=rs.page_start, text=text)
        for s, e in split_sentences(text):
            sid = f"{paper_id}_s{idx:04d}"
            sentences.append(Sentence(
                sentence_id=sid, text=text[s:e], page=_page_at(marks, s), section_id=sec_id,
                section_type=rs.stype.value, section_title=rs.title, index=idx,
            ))
            sec.sentence_ids.append(sid)
            idx += 1
        sections.append(sec)
    return sections, sentences


def section_id_for(paper_id: str, name: str) -> str:
    return stable_id("sec", paper_id, name, length=8)
