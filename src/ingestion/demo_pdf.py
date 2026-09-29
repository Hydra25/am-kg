"""Render the DEMO / SYNTHETIC markdown papers in data/demo/ to PDF.

Uses PyMuPDF's Story API (HTML -> PDF), so no extra dependency is needed. The
PDFs exercise the real PDF path (fonts, headings, ruled tables, page numbers)
of the pipeline. Every page carries a "DEMO / SYNTHETIC" footer.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF
    import fitz  # type: ignore

CSS = """
body { font-family: serif; font-size: 10pt; }
h1 { font-size: 16pt; font-weight: bold; }
h2 { font-size: 12.5pt; font-weight: bold; margin-top: 8pt; }
h3 { font-size: 11pt; font-weight: bold; margin-top: 6pt; }
p { text-align: justify; margin-bottom: 4pt; }
table { border-collapse: collapse; margin: 6pt 0; }
td, th { border: 1px solid black; padding: 3pt; font-size: 9.5pt; }
p.meta { font-size: 9.5pt; margin: 0; }
"""


def markdown_to_html(md: str) -> str:
    out: list[str] = []
    rows: list[list[str]] = []
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{html.escape(' '.join(para))}</p>")
            para.clear()

    def flush_table() -> None:
        if rows:
            cells = "".join(
                "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{html.escape(c)}</{'th' if i == 0 else 'td'}>"
                                 for c in r) + "</tr>"
                for i, r in enumerate(rows))
            out.append(f"<table>{cells}</table>")
            rows.clear()

    for line in md.splitlines():
        s = line.strip()
        if s.startswith("|"):
            flush_para()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
            continue
        flush_table()
        m = re.match(r"^(#{1,3})\s+(.*)$", s)
        if m:
            flush_para()
            level = len(m.group(1))
            out.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>")
        elif re.match(r"^(Authors|Year|Journal|DOI)\s*:", s):
            flush_para()
            out.append(f"<p class='meta'>{html.escape(s)}</p>")
        elif not s:
            flush_para()
        else:
            if re.match(r"^Table\s+\d", s):
                flush_para()
                out.append(f"<p>{html.escape(s)}</p>")
            else:
                para.append(s)
    flush_para()
    flush_table()
    return "\n".join(out)


def render_markdown_pdf(md_path: Path, pdf_path: Path) -> Path:
    body = markdown_to_html(Path(md_path).read_text(encoding="utf-8"))
    story = fitz.Story(html=body, user_css=CSS)
    writer = fitz.DocumentWriter(str(pdf_path))
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (56, 56, -56, -70)
    more = True
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    # footer with page numbers (removed again by the header/footer filter)
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        page.insert_text((56, mediabox.height - 35), f"DEMO / SYNTHETIC document - page {i}",
                         fontsize=8, fontname="helv")
    # fixed metadata + no random file id -> identical bytes -> identical paper_id on every run
    doc.set_metadata({"title": "DEMO / SYNTHETIC document", "creationDate": "D:20250101000000",
                      "modDate": "D:20250101000000", "producer": "AM-KG demo generator", "creator": "AM-KG"})
    data = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    doc.close()
    Path(pdf_path).write_bytes(data)
    return pdf_path


def build_demo_pdfs(demo_dir: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [render_markdown_pdf(md, out_dir / f"{md.stem}.pdf") for md in sorted(Path(demo_dir).glob("*.md"))]
