"""PyMuPDF-based PDF parser.

Produces font-annotated lines (for section detection) and tables. Handles:
  * two-column layouts (column-aware block ordering)
  * running headers / footers / page numbers (removed if repeated)
  * tables via `page.find_tables()` (text inside table areas is not duplicated
    into the body; tables are returned separately and never silently dropped)

Limitations (documented extension points): scanned PDFs need OCR; complex
tables without ruling lines may be missed by find_tables — plug a better table
extractor into `extract_tables()` (e.g. Camelot, GROBID figures/tables, or a
table-transformer model).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ontology.models import Table
from ..section_parser.sections import Line

log = logging.getLogger(__name__)

try:
    import pymupdf as fitz  # PyMuPDF >= 1.24
except ImportError:  # pragma: no cover
    try:
        import fitz  # type: ignore
    except ImportError:
        fitz = None  # type: ignore


_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
              "\ufb05": "st", "\ufb06": "st"}


def fix_ligatures(text: str) -> str:
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    return text


@dataclass
class RawPdf:
    lines: list[Line]
    tables: list[Table]
    num_pages: int
    first_page_spans: list[tuple[str, float, float]]   # (text, size, y0) for title detection
    pdf_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ((x1 - x0) * (y1 - y0)) / area if area > 0 else 0.0


def _order_blocks(blocks: list[dict], page_width: float) -> list[dict]:
    """Column-aware reading order: full-width blocks split the page into bands;
    inside a band, the left column is read before the right column."""
    mid = page_width / 2
    blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    ordered: list[dict] = []
    band: list[dict] = []

    def flush() -> None:
        left = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 < mid]
        right = [b for b in band if (b["bbox"][0] + b["bbox"][2]) / 2 >= mid]
        ordered.extend(sorted(left, key=lambda b: b["bbox"][1]))
        ordered.extend(sorted(right, key=lambda b: b["bbox"][1]))
        band.clear()

    for b in blocks:
        width = b["bbox"][2] - b["bbox"][0]
        crosses_mid = b["bbox"][0] < mid - 20 and b["bbox"][2] > mid + 20
        if width > 0.6 * page_width or crosses_mid:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def extract_tables(page: Any, page_no: int, start_idx: int) -> tuple[list[Table], list[tuple]]:
    """Extension point: returns (tables, bboxes). Uses PyMuPDF's find_tables."""
    tables: list[Table] = []
    bboxes: list[tuple] = []
    if not hasattr(page, "find_tables"):
        return tables, bboxes
    try:
        found = page.find_tables()
    except Exception as exc:  # pragma: no cover - depends on PDF
        log.warning("find_tables failed on page %s: %s", page_no, exc)
        return tables, bboxes
    for i, t in enumerate(getattr(found, "tables", []) or []):
        try:
            data = t.extract()
        except Exception:  # pragma: no cover
            continue
        rows = [[fix_ligatures((c or "").replace("\n", " ").strip()) for c in r] for r in data if r]
        if len(rows) < 2:
            continue
        header = rows[0]
        tables.append(Table(table_id=f"t{start_idx + i + 1}", page=page_no, header=header,
                            rows=rows[1:], extractor="pymupdf.find_tables"))
        bboxes.append(tuple(t.bbox))
    return tables, bboxes


def parse_pdf(path: Path, extract_tables_flag: bool = True, remove_headers: bool = True) -> RawPdf:
    if fitz is None:
        raise ImportError("PyMuPDF is required: pip install pymupdf")
    doc = fitz.open(str(path))
    lines: list[Line] = []
    tables: list[Table] = []
    warnings: list[str] = []
    first_spans: list[tuple[str, float, float]] = []
    margin_keys: Counter[str] = Counter()
    per_page_margin: list[list[tuple[str, Line]]] = []
    total_chars = 0

    for pno, page in enumerate(doc, start=1):
        width, height = page.rect.width, page.rect.height
        page_tables, tboxes = ([], [])
        if extract_tables_flag:
            page_tables, tboxes = extract_tables(page, pno, len(tables))
            # attach captions ("Table 1. ...") found just above each table
            for tb, bbox in zip(page_tables, tboxes):
                clip = fitz.Rect(0, max(0, bbox[1] - 45), width, bbox[1])
                cap = fix_ligatures(page.get_text("text", clip=clip).strip().replace("\n", " "))
                m = re.search(r"(Table\s+\d+.*)$", cap, re.I)
                tb.caption = m.group(1).strip() if m else None
            tables.extend(page_tables)
        d = page.get_text("dict")
        blocks = [b for b in d.get("blocks", []) if b.get("type") == 0]
        blocks = [b for b in blocks if not any(_overlap_ratio(tuple(b["bbox"]), tb) > 0.5 for tb in tboxes)]
        page_margin: list[tuple[str, Line]] = []
        for b in _order_blocks(blocks, width):
            for ln in b.get("lines", []):
                spans = [s for s in ln.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = fix_ligatures("".join(s["text"] for s in ln["spans"]).strip())
                sizes = [s["size"] for s in spans]
                size = max(set(sizes), key=sizes.count)
                bold = all((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                y0 = ln["bbox"][1]
                line = Line(text=text, page=pno, size=round(size, 1), bold=bool(bold), y0=y0)
                total_chars += len(text)
                if pno == 1:
                    first_spans.append((text, round(size, 1), y0))
                if y0 < 0.07 * height or ln["bbox"][3] > 0.94 * height:
                    key = re.sub(r"\d+", "#", text.lower())
                    margin_keys[key] += 1
                    page_margin.append((key, line))
                lines.append(line)
        per_page_margin.append(page_margin)

    num_pages = len(doc)
    if remove_headers:
        repeated = {k for k, c in margin_keys.items() if num_pages >= 2 and c >= max(2, int(0.4 * num_pages))}
        drop = {id(line) for pm in per_page_margin for k, line in pm
                if k in repeated or re.fullmatch(r"#|page #|# of #|- # -", k)
                or re.search(r"\bpage #( of #)?$", k)}
        lines = [ln for ln in lines if id(ln) not in drop]

    if num_pages and total_chars / num_pages < 200:
        warnings.append("Very little text extracted: the PDF may be scanned; OCR is not implemented "
                        "(extension point: run ocrmypdf before ingestion).")
    meta = dict(doc.metadata or {})
    doc.close()
    return RawPdf(lines=lines, tables=tables, num_pages=num_pages, first_page_spans=first_spans,
                  pdf_metadata=meta, warnings=warnings)
