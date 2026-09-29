"""Document ingestion: file -> ParsedDocument (metadata, sections, sentences, tables).

Supported inputs:
  * .pdf  -> GROBID (if GROBID_URL is set and alive) else PyMuPDF
  * .txt / .md -> plain text with '# Heading' lines (used for demo data & tests)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from ..config import Settings
from ..normalization.units import unit_from_header
from ..ontology.models import PaperMetadata, ParsedDocument, Section, Sentence, Table
from ..pdf_parser import grobid_parser
from ..pdf_parser.pymupdf_parser import RawPdf, parse_pdf
from ..section_parser.sections import Line, build_sections

log = logging.getLogger(__name__)

SYNTHETIC_MARKER = re.compile(r"DEMO\s*/\s*SYNTHETIC", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>,;]+)", re.I)
_KV_RE = re.compile(r"^(title|authors?|year|journal|doi)\s*:\s*(.+)$", re.I)


def paper_id_for(path: Path) -> str:
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    return f"paper_{h}"


# ---------------------------------------------------------------------------
# metadata heuristics
# ---------------------------------------------------------------------------
def _kv_metadata(lines: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in lines[:40]:
        m = _KV_RE.match(ln.strip())
        if m:
            out.setdefault(m.group(1).lower().rstrip("s") if m.group(1).lower() != "authors" else "authors",
                           m.group(2).strip())
    return out


_JOURNAL_HEADER = re.compile(
    r"(journal\b|contents lists|homepage|sciencedirect|elsevier|materials (&|and) design|science (&|and) engineering"
    r"|letters$|procedia|^materials$|^metals$|^coatings$|^scientific reports$|www\.|https?://)", re.I)


def _plausible_title(t: str) -> bool:
    return (len(t.split()) >= 4 and not re.match(r"^(microsoft|untitled|document\d*|.*\.(docx?|pdf|tex))", t, re.I)
            and not _JOURNAL_HEADER.search(t))


def _guess_title(raw: RawPdf) -> Optional[str]:
    spans = [(t, s, y) for t, s, y in raw.first_page_spans
             if len(t) > 3 and not _KV_RE.match(t) and not _JOURNAL_HEADER.search(t)]
    if not spans:
        return None
    max_y = max(y for _, _, y in spans) or 1
    top = [sp for sp in spans if sp[2] <= 0.5 * max_y] or spans
    biggest = max(s for _, s, _ in top)
    title_lines = [t for t, s, _ in top if s == biggest]
    title = " ".join(title_lines).strip()
    return title if len(title) >= 10 else None


def _guess_year(text: str) -> Optional[int]:
    m = re.search(r"(?:received|accepted|published|available online|©|\(c\)|copyright)[^\n]{0,60}?\b((?:19|20)\d{2})\b", text, re.I)
    if not m:
        m = re.search(r"\b(20[0-3]\d|19[89]\d)\b", text)
    return int(m.group(1)) if m else None


def _guess_authors(line_texts: list[str], title: Optional[str]) -> list[str]:
    started = title is None
    for t in line_texts[:25]:
        if not started:
            if title and t.strip() and t.strip() in title:
                started = True
            continue
        if re.match(r"^abstract", t, re.I):
            break
        cand = re.sub(r"[\d*†‡§,]+(?=\s|,|$)", "", t)
        if ("," in t or " and " in t) and not re.search(r"(university|institute|department|dept|school|@|laborator)", t, re.I):
            names = [n.strip(" ,;*") for n in re.split(r",| and ", cand) if n.strip(" ,;*")]
            if names and all(1 <= len(n.split()) <= 4 and n[0].isupper() for n in names):
                return names
    return []


def _finalize_metadata(meta: PaperMetadata, full_text: str, line_texts: list[str],
                       sections: list[Section]) -> PaperMetadata:
    kv = _kv_metadata(line_texts)
    if kv.get("title") and not meta.title:
        meta.title = kv["title"]
    if kv.get("authors") and not meta.authors:
        meta.authors = [a.strip() for a in re.split(r";|,| and ", kv["authors"]) if a.strip()]
    if kv.get("year") and not meta.year:
        m = re.search(r"\d{4}", kv["year"])
        meta.year = int(m.group()) if m else None
    if kv.get("journal") and not meta.journal:
        meta.journal = kv["journal"]
    if not meta.doi:
        m = _DOI_RE.search(full_text[:6000])
        meta.doi = m.group(1).rstrip(".)]") if m else None
    if not meta.year:
        meta.year = _guess_year(full_text[:4000])
    if not meta.abstract:
        abs_secs = [s for s in sections if s.section_type == "abstract"]
        meta.abstract = abs_secs[0].text[:3000] if abs_secs else None
    meta.is_synthetic = bool(SYNTHETIC_MARKER.search(full_text[:5000]))
    return meta


# ---------------------------------------------------------------------------
# tables -> row sentences (so table facts are extracted, never discarded)
# ---------------------------------------------------------------------------
def table_row_sentences(tables: list[Table], sections: list[Section], paper_id: str,
                        start_index: int) -> list[Sentence]:
    out: list[Sentence] = []
    idx = start_index
    for tb in tables:
        units = [unit_from_header(h) for h in tb.header]
        clean_headers = [re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*", " ", h).strip() or f"col{i}"
                         for i, h in enumerate(tb.header)]
        host = None
        tnum = re.search(r"\d+", tb.caption or "")
        if tnum:  # the section whose text contains the caption "Table N"
            pat = re.compile(rf"\bTable\s*{tnum.group()}\b[.:]")
            host = next((s for s in sections if pat.search(s.text)), None)
        if host is None:
            for s in sections:
                if s.page_start is not None and tb.page is not None and s.page_start <= tb.page:
                    host = s
        label = (tb.caption or f"Table {tb.table_id}").split(".")[0][:40]
        for r, row in enumerate(tb.rows):
            cells = []
            for h, u, c in zip(clean_headers, units, row):
                c = (c or "").strip()
                if not c:
                    continue
                if u and re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:\s*(?:±|-|–)\s*\d+(?:\.\d+)?)?", c):
                    c = f"{c} {u}"
                cells.append(f"{h} = {c}")
            if not cells:
                continue
            text = f"{label}, row {r + 1}: " + "; ".join(cells) + "."
            out.append(Sentence(
                sentence_id=f"{paper_id}_s{idx:04d}", text=text, page=tb.page,
                section_id=host.section_id if host else f"{paper_id}_tables",
                section_type=host.section_type if host else "other",
                section_title=f"{label} ({host.title})" if host else label,
                index=idx, source_type="table", table_id=tb.table_id,
            ))
            idx += 1
    return out


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def _parse_text_file(path: Path, paper_id: str) -> tuple[list[Line], Optional[str], list[Table]]:
    lines: list[Line] = []
    tables: list[Table] = []
    title = None
    page = 1
    table_rows: list[list[str]] = []
    last_text = ""
    for raw in path.read_text(encoding="utf-8").splitlines() + [""]:
        if raw.strip().startswith("|"):
            cells = [c.strip() for c in raw.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                table_rows.append(cells)
            continue
        if table_rows:
            caption = last_text if re.match(r"^table\s+\d", last_text, re.I) else None
            tables.append(Table(table_id=f"t{len(tables) + 1}", page=page, caption=caption,
                                header=table_rows[0], rows=table_rows[1:], extractor="markdown"))
            table_rows = []
        if raw.strip():
            last_text = raw.strip()
        if raw.strip() == "\f" or raw.strip().lower().startswith("<<page"):
            page += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", raw)
        if m:
            if m.group(1) == "#" and title is None:
                title = m.group(2).strip()
                continue
            lines.append(Line(text=m.group(2).strip(), page=page, is_heading=True))
        elif raw.strip():
            lines.append(Line(text=raw.strip(), page=page, is_heading=False))
    return lines, title, tables


def parse_document(path: Path, settings: Settings) -> ParsedDocument:
    path = Path(path)
    paper_id = paper_id_for(path)
    suffix = path.suffix.lower()
    warnings: list[str] = []
    tables: list[Table] = []
    num_pages = 1

    if suffix == ".pdf":
        parser_choice = settings.pdf.parser
        use_grobid = parser_choice == "grobid" or (parser_choice == "auto" and settings.grobid_url
                                                   and grobid_parser.grobid_available(settings.grobid_url))
        if use_grobid:
            try:
                tei = grobid_parser.fetch_tei(path, settings.grobid_url)
                meta, sections, sentences, tables = grobid_parser.parse_tei(tei, paper_id, path.name)
                import pymupdf as fitz  # page count only
                with fitz.open(str(path)) as d:
                    num_pages = len(d)
                full = " ".join(s.text for s in sentences)
                meta = _finalize_metadata(meta, full, [], sections)
                sentences += table_row_sentences(tables, sections, paper_id, len(sentences))
                return ParsedDocument(paper=meta, parser="grobid", num_pages=num_pages, sections=sections,
                                      sentences=sentences, tables=tables, warnings=warnings)
            except Exception as exc:
                warnings.append(f"GROBID failed ({exc}); fell back to PyMuPDF")
                log.warning("GROBID failed for %s: %s", path.name, exc)
        raw = parse_pdf(path, settings.pdf.extract_tables, settings.pdf.remove_headers_footers)
        warnings += raw.warnings
        tables = raw.tables
        num_pages = raw.num_pages
        # Publisher PDFs (Elsevier, MDPI, Nature) carry the real title in their metadata;
        # the biggest font on page 1 is often the journal name ("Materials & Design").
        pdf_title = ((raw.pdf_metadata or {}).get("title") or "").strip()
        if _plausible_title(pdf_title):
            title = pdf_title
        else:
            title = _guess_title(raw)
        lines = raw.lines
        parser_name = "pymupdf"
    elif suffix in {".txt", ".md"}:
        lines, title, tables = _parse_text_file(path, paper_id)
        parser_name = "text"
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    line_texts = [ln.text for ln in lines]
    kv = _kv_metadata(line_texts)
    title = kv.get("title") or title
    # key:value front-matter lines are metadata, not body text
    body_lines = [ln for ln in lines if not _KV_RE.match(ln.text.strip())]
    sections, sentences = build_sections(body_lines, paper_id, title=title)
    full_text = "\n".join(line_texts)
    meta = PaperMetadata(paper_id=paper_id, title=title, source_file=path.name)
    if parser_name == "pymupdf":
        meta.authors = _guess_authors(line_texts, title)
    meta = _finalize_metadata(meta, full_text, line_texts, sections)
    sentences += table_row_sentences(tables, sections, paper_id, len(sentences))
    if not any(s.section_type not in ("front_matter", "other") for s in sections):
        warnings.append("No standard scientific section headings detected; section labels are unreliable.")
    return ParsedDocument(paper=meta, parser=parser_name, num_pages=num_pages, sections=sections,
                          sentences=sentences, tables=tables, warnings=warnings)


def save_parsed(doc: ParsedDocument, out_dir: Path) -> Path:
    out = Path(out_dir) / f"{doc.paper.paper_id}.json"
    out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_parsed(path: Path) -> ParsedDocument:
    return ParsedDocument.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def discover_inputs(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in {".pdf", ".txt", ".md"}
                  and not p.name.lower().startswith("readme"))
