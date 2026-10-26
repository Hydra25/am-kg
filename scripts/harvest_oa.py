"""Free paper harvester: open-access LPBF papers from OpenAlex (free API, no key).

    python scripts/harvest_oa.py --max 100                 # search + download into data/harvest/papers/
    python scripts/harvest_oa.py --max 100 --dry-run       # only write the manifest (no downloads)

Rules (see the project notes):
* only open-access papers with a Creative Commons / public-domain licence and a direct PDF link;
* NO automated downloads from ScienceDirect / Elsevier hosts (their terms forbid it) - such papers are skipped;
* the pilot/spare benchmark papers (data/benchmark_papers.csv) are excluded, so the auto-lexicon never
  learns from the DEV/TEST papers (no leakage);
* polite crawling: one request per second, resumable, a manifest records every decision.

Output: data/harvest/papers/*.pdf and data/harvest/papers/manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]

API = "https://api.openalex.org/works"
QUERIES = ["laser powder bed fusion", "selective laser melting", "L-PBF metal", "PBF-LB/M"]
OPEN_LICENCES = {"cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd", "cc0", "public-domain"}
BLOCKED_HOSTS = ("sciencedirect.com", "elsevier.com", "els-cdn.com")
LPBF_RE = re.compile(r"laser powder[- ]bed fusion|selective laser melting|\bL-?PBF\b|\bPBF-LB|\bSLM\b|"
                     r"direct metal laser (?:sintering|melting)|\bDMLS\b", re.I)
METAL_RE = re.compile(r"Ti-?6Al-?4V|\bTi64\b|titanium|316L|stainless|AlSi10Mg|AlSi|alumin|Inconel|IN ?718|IN ?625|"
                      r"nickel|superalloy|maraging|steel|CoCr|cobalt|copper|tungsten|alloy|metal", re.I)
MANIFEST_FIELDS = ["file", "doi", "title", "year", "journal", "publisher", "license", "pdf_url", "status"]
USER_AGENT = "am-kg research harvester (academic, open-access only)"


def abstract_text(inv: Optional[dict]) -> str:
    """OpenAlex stores abstracts as an inverted index {word: [positions]}."""
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def norm_doi(doi: Optional[str]) -> str:
    return (doi or "").lower().replace("https://doi.org/", "").strip()


def excluded_dois(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {norm_doi(r.get("doi")) for r in csv.DictReader(fh) if r.get("doi")}


def judge(work: dict, exclude: set[str]) -> tuple[Optional[dict], str]:
    """-> (manifest row, '') if the work should be downloaded, else (row or None, reason)."""
    doi = norm_doi(work.get("doi"))
    loc = work.get("best_oa_location") or {}
    src = loc.get("source") or {}
    row = {"doi": doi, "title": (work.get("title") or "").strip(), "year": work.get("publication_year"),
           "journal": src.get("display_name") or "", "publisher": src.get("host_organization_name") or "",
           "license": loc.get("license") or "", "pdf_url": loc.get("pdf_url") or ""}
    if not doi:
        return row, "no_doi"
    if doi in exclude:
        return row, "benchmark_paper"
    # every open copy with a PDF link (publisher first, then repositories such as PMC / arXiv / university
    # repositories): if the publisher refuses scripted downloads, an identical repository copy often works
    urls = [loc.get("pdf_url")] + [x.get("pdf_url") for x in work.get("locations") or [] if x.get("is_oa")]
    urls = [u for u in dict.fromkeys(u for u in urls if u)
            if not any(h in urlparse(u).netloc.lower() for h in BLOCKED_HOSTS)]
    if not urls:
        return row, "blocked_host" if row["pdf_url"] else "no_pdf_link"
    row["pdf_url"], row["alt_urls"] = urls[0], urls[1:]
    if (row["license"] or "").lower() not in OPEN_LICENCES:
        return row, "licence_not_open"
    text = f"{row['title']} {abstract_text(work.get('abstract_inverted_index'))}"
    if not LPBF_RE.search(text):
        return row, "not_lpbf"
    if not METAL_RE.search(text):
        return row, "not_metal"
    return row, ""


def search(query: str, per_page: int, from_year: int, mailto: str) -> Iterator[dict]:
    import requests
    cursor = "*"
    params: dict[str, Any] = {
        "search": query, "per-page": per_page,
        "filter": f"open_access.is_oa:true,type:article,from_publication_date:{from_year}-01-01",
        "select": "id,doi,title,publication_year,best_oa_location,locations,abstract_inverted_index",
    }
    if mailto:
        params["mailto"] = mailto            # OpenAlex "polite pool" (optional, your own choice)
    headers = {"User-Agent": USER_AGENT}
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
    if api_key:                              # free OpenAlex key from .env; sent as a header, never in URLs/logs
        headers["Authorization"] = f"Bearer {api_key}"
    while cursor:
        params["cursor"] = cursor
        for attempt in range(4):             # rate limit: wait (Retry-After) and try again
            r = requests.get(API, params=params, headers=headers, timeout=60)
            if r.status_code != 429:
                break
            wait = float(r.headers.get("Retry-After") or 10 * (attempt + 1))
            print(f"  OpenAlex 429 (rate limit): {r.text[:200].strip()} -> waiting {wait:.0f}s", flush=True)
            if wait > 120:
                break
            time.sleep(wait)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAlex HTTP {r.status_code}: {r.text[:300].strip()}")
        data = r.json()
        yield from data.get("results", [])
        cursor = (data.get("meta") or {}).get("next_cursor")
        time.sleep(1)


def file_name(doi: str) -> str:
    return "oa_" + re.sub(r"[^a-z0-9]+", "_", doi.lower()).strip("_")[:80] + ".pdf"


def download(url: str, dest: Path, max_mb: int = 40) -> str:
    import requests
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"},
                         timeout=90, stream=True, allow_redirects=True)
    except requests.RequestException as exc:
        return f"error:{type(exc).__name__}"
    if r.status_code != 200:
        return f"http_{r.status_code}"
    if any(h in urlparse(r.url).netloc.lower() for h in BLOCKED_HOSTS):
        return "blocked_host_redirect"
    data = b""
    for chunk in r.iter_content(65536):
        data += chunk
        if len(data) > max_mb * 1024 * 1024:
            return "too_large"
    if not data.startswith(b"%PDF"):
        return "not_a_pdf"                 # e.g. an HTML landing page or a bot check
    dest.write_bytes(data)
    return "ok"


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {r["doi"]: r for r in csv.DictReader(fh)}


def write_manifest(path: Path, rows: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})


def main() -> None:
    try:                                     # OPENALEX_API_KEY may live in .env (optional)
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max", type=int, default=100, help="stop after this many downloaded papers")
    ap.add_argument("--out", default="data/harvest/papers")
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--query", action="append", help="search query (repeatable); default: built-in LPBF queries")
    ap.add_argument("--mailto", default="", help="optional e-mail for the OpenAlex polite pool")
    ap.add_argument("--dry-run", action="store_true", help="manifest only, no PDF downloads")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    rows = read_manifest(manifest_path)
    exclude = excluded_dois(ROOT / "data" / "benchmark_papers.csv")
    have = sum(1 for r in rows.values() if r.get("status") == "ok")
    print(f"Already downloaded: {have}. Excluding {len(exclude)} benchmark papers.", flush=True)
    try:
        for q in args.query or QUERIES:
            print(f"\nSearch: {q}", flush=True)
            for work in search(q, 50, args.from_year, args.mailto):
                if have >= args.max:
                    break
                row, reason = judge(work, exclude)
                if row is None or not row["doi"]:
                    continue
                prev = rows.get(row["doi"], {}).get("status", "")
                if prev == "ok" or prev.startswith("skip:"):
                    continue                     # done before; failed downloads (http_403 ...) are retried
                if reason:
                    rows[row["doi"]] = {**row, "status": f"skip:{reason}"}
                    continue
                row["file"] = file_name(row["doi"])
                if args.dry_run:
                    row["status"] = "dry_run"
                else:
                    for url in [row["pdf_url"], *row.pop("alt_urls", [])]:
                        row["status"] = download(url, out / row["file"])
                        time.sleep(1)
                        if row["status"] == "ok":
                            row["pdf_url"] = url
                            break
                rows[row["doi"]] = row
                if row["status"] in ("ok", "dry_run"):
                    have += 1
                    print(f"  [{have}/{args.max}] {row['year']} {row['title'][:80]}", flush=True)
                else:
                    print(f"  - {row['status']}: {row['title'][:70]}", flush=True)
            if have >= args.max:
                break
    except KeyboardInterrupt:
        print("\nStopped (progress saved).")
    except Exception as exc:          # network trouble: keep what we have
        print(f"\nSearch stopped: {exc}", file=sys.stderr)
    finally:
        write_manifest(manifest_path, rows)
    from collections import Counter
    print("\nStatus:", dict(Counter(r.get("status", "") for r in rows.values())))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
