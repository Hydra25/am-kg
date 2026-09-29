"""Optional GROBID integration (https://github.com/kermitt2/grobid).

Start GROBID locally with Docker:
    docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
and set GROBID_URL=http://localhost:8070 in .env.

GROBID gives much better title/author/abstract/section structure than font
heuristics. Sentences carry page coordinates (teiCoordinates=s).
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import requests

from ..ontology.models import PaperMetadata, Section, Sentence, Table
from ..ontology.schema import SectionType
from ..section_parser.sections import classify_heading
from ..section_parser.sentences import split_sentences

log = logging.getLogger(__name__)
NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def grobid_available(url: str, timeout: float = 3.0) -> bool:
    if not url:
        return False
    try:
        return requests.get(f"{url.rstrip('/')}/api/isalive", timeout=timeout).ok
    except requests.RequestException:
        return False


def fetch_tei(pdf_path: Path, url: str, timeout: int = 180) -> str:
    with open(pdf_path, "rb") as fh:
        resp = requests.post(
            f"{url.rstrip('/')}/api/processFulltextDocument",
            files={"input": (pdf_path.name, fh, "application/pdf")},
            data={"teiCoordinates": ["s", "head", "figure"], "segmentSentences": "1",
                  "consolidateHeader": "0"},
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.text


def _text(el: Optional[ET.Element]) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _page_from_coords(el: ET.Element) -> Optional[int]:
    coords = el.get("coords")
    if not coords:
        return None
    try:
        return int(coords.split(";")[0].split(",")[0])
    except (ValueError, IndexError):
        return None


def parse_tei(tei_xml: str, paper_id: str, source_file: str) -> tuple[PaperMetadata, list[Section], list[Sentence], list[Table]]:
    root = ET.fromstring(tei_xml.encode("utf-8") if isinstance(tei_xml, str) else tei_xml)
    header = root.find(".//tei:teiHeader", NS)
    title = _text(header.find(".//tei:titleStmt/tei:title", NS)) if header is not None else ""
    authors = []
    for a in root.findall(".//tei:sourceDesc//tei:analytic/tei:author/tei:persName", NS):
        name = " ".join(filter(None, [_text(a.find("tei:forename", NS)), _text(a.find("tei:surname", NS))]))
        if name:
            authors.append(name)
    doi = _text(root.find(".//tei:sourceDesc//tei:idno[@type='DOI']", NS)) or None
    journal = _text(root.find(".//tei:sourceDesc//tei:monogr/tei:title[@level='j']", NS)) or None
    date = root.find(".//tei:sourceDesc//tei:date[@type='published']", NS)
    year = None
    if date is not None and date.get("when"):
        m = re.match(r"(\d{4})", date.get("when", ""))
        year = int(m.group(1)) if m else None
    abstract_el = root.find(".//tei:profileDesc/tei:abstract", NS)
    abstract = _text(abstract_el) or None
    meta = PaperMetadata(paper_id=paper_id, title=title or None, authors=authors, year=year,
                         journal=journal, doi=doi, source_file=source_file, abstract=abstract,
                         metadata_source="grobid")

    sections: list[Section] = []
    sentences: list[Sentence] = []
    idx = 0

    def add_section(title_: str, stype: str, blocks: list[ET.Element]) -> None:
        nonlocal idx
        sec_id = f"{paper_id}_sec{len(sections):02d}"
        sec = Section(section_id=sec_id, title=title_, section_type=stype)
        texts = []
        for p in blocks:
            s_elems = p.findall(".//tei:s", NS)
            units = [(_text(s), _page_from_coords(s)) for s in s_elems] if s_elems else []
            if not units:
                t = _text(p)
                units = [(t[a:b], None) for a, b in split_sentences(t)]
            for txt, page in units:
                if not txt:
                    continue
                sid = f"{paper_id}_s{idx:04d}"
                sentences.append(Sentence(sentence_id=sid, text=txt, page=page, section_id=sec_id,
                                          section_type=stype, section_title=title_, index=idx))
                sec.sentence_ids.append(sid)
                if sec.page_start is None and page:
                    sec.page_start = page
                texts.append(txt)
                idx += 1
        sec.text = " ".join(texts)
        if sec.sentence_ids:
            sections.append(sec)

    if abstract_el is not None:
        add_section("Abstract", SectionType.ABSTRACT.value, abstract_el.findall(".//tei:p", NS) or [abstract_el])

    last_type = SectionType.INTRODUCTION.value
    for div in root.findall(".//tei:text/tei:body/tei:div", NS):
        head = div.find("tei:head", NS)
        htext = _text(head)
        num = head.get("n") if head is not None else None
        stype_enum = classify_heading(htext) if htext else None
        stype = stype_enum.value if stype_enum else (last_type if num and "." in num else SectionType.OTHER.value)
        if stype_enum and not (num and "." in num):
            last_type = stype
        add_section(f"{num + ' ' if num else ''}{htext}".strip() or "Untitled", stype, div.findall("tei:p", NS))

    tables: list[Table] = []
    for i, fig in enumerate(root.findall(".//tei:figure[@type='table']", NS)):
        rows = [[_text(c) for c in row.findall("tei:cell", NS)] for row in fig.findall(".//tei:row", NS)]
        if len(rows) < 2:
            continue
        cap = " ".join(filter(None, [_text(fig.find("tei:head", NS)), _text(fig.find("tei:figDesc", NS))]))
        tables.append(Table(table_id=f"t{i + 1}", page=_page_from_coords(fig), caption=cap or None,
                            header=rows[0], rows=rows[1:], extractor="grobid"))
    return meta, sections, sentences, tables
