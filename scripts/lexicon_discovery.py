"""Option B: grow the lexicon automatically from GLiNER discoveries on harvested open-access papers.

    python scripts/lexicon_discovery.py                  # extract (resumable) + aggregate + LLM check + write
    python scripts/lexicon_discovery.py --limit 20       # only the first 20 harvested papers
    python scripts/lexicon_discovery.py --no-llm         # no LLM: candidates that pass wait as 'llm_pending'
    python scripts/lexicon_discovery.py --check          # + regression check on DEV (rules-only, with/without)

Input: PDFs from scripts/harvest_oa.py (data/harvest/papers/, never the DEV/TEST papers).
Output:
  config/lexicon_auto.yaml                 new concepts + extra aliases (loaded automatically by Lexicon.load;
                                           move/rename the file to undo everything)
  data/harvest/discovery/candidates.csv    every candidate with its tier and the reason
  data/harvest/discovery/blocklist.json    terms that are never proposed again
  data/harvest/discovery/llm_verdicts.json LLM answers (cached, one call per term, not per mention)

Rule v1 (agreed 2026-10-16): a NEW GLiNER term goes to the auto lexicon only if
  1. it occurs in >= 3 different papers,
  2. those papers study >= 2 different materials,
  3. >= 70 % of its mentions have the same entity type,
  4. it passes the junk filter, and is not a short abbreviation (<= 4 characters / ALL CAPS),
  5. a large LLM (default Groq gpt-oss-120b, free tier) says it is a specific AM concept.
  LLM says "synonym of an existing lexicon term" -> alias of that term. LLM says "no" -> block list.
  Anything that misses a threshold stays on the watch list and is re-checked when more papers arrive.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from src.normalization.lexicon import AUTO_FILE_NAME, Lexicon  # noqa: E402
from src.normalization.text import match_key  # noqa: E402
from src.quality.junk import junk_reason  # noqa: E402

MIN_PAPERS, MIN_MATERIALS, MIN_TYPE_SHARE = 3, 2, 0.7
TYPES = ["Material", "Process", "Parameter", "Property", "Defect", "Microstructure", "PostProcessing",
         "CharacterizationMethod", "Geometry", "ExperimentalCondition"]
_DET = re.compile(r"^(the|a|an|this|that|these|those|such|its|their)\s+", re.I)
# generic tails that do not change the concept: "weak fusion defects" = "weak fusion"
_TAILS = {"defect", "defects", "value", "values", "phenomenon", "phenomena", "characteristic", "characteristics"}


# ---------------------------------------------------------------------------------------- grouping
def singular(word: str) -> str:
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return word[:-3] + "y"
    if len(w) > 4 and re.search(r"(sses|xes|ches|shes)$", w):
        return word[:-2]
    if len(w) > 3 and w.endswith("s") and not re.search(r"(ss|us|is|ys)$", w):
        return word[:-1]
    return word


def clean_name(text: str) -> str:
    t = _DET.sub("", " ".join(text.split())).strip(" .,;:")
    words = t.split(" ")
    while len(words) > 1 and words[-1].lower() in _TAILS:
        words = words[:-1]
    return " ".join(words)


def group_key(text: str) -> str:
    words = clean_name(text).split(" ")
    words[-1] = singular(words[-1])
    return match_key(" ".join(words))


@dataclass
class Candidate:
    key: str
    names: Counter = field(default_factory=Counter)
    types: Counter = field(default_factory=Counter)
    papers: set = field(default_factory=set)
    materials: set = field(default_factory=set)
    mentions: int = 0
    examples: list = field(default_factory=list)       # (paper, sentence)
    tier: str = ""
    reason: str = ""
    target: str = ""                                     # alias target / final type

    @property
    def name(self) -> str:
        """The most frequent written form, preferring a singular form ("pore", not "pores")."""
        forms = [n for n, _ in self.names.most_common()]
        single = [n for n in forms if n.split(" ")[-1] == singular(n.split(" ")[-1])]
        return (single or forms)[0]

    @property
    def type(self) -> str:
        return self.types.most_common(1)[0][0]

    @property
    def type_share(self) -> float:
        return self.types.most_common(1)[0][1] / max(1, sum(self.types.values()))


def aggregate(papers: list[dict]) -> dict[str, Candidate]:
    cands: dict[str, Candidate] = {}
    for p in papers:
        for m in p["mentions"]:
            key = group_key(m["text"])
            if len(key) < 2:
                continue
            c = cands.setdefault(key, Candidate(key))
            c.names[clean_name(m["text"])] += 1
            c.types[m["type"]] += 1
            c.mentions += 1
            c.papers.add(p["paper"])
            if p.get("material"):
                c.materials.add(p["material"])
            if len(c.examples) < 3 and all(e[0] != p["paper"] for e in c.examples):
                c.examples.append((p["paper"], m["sentence"][:400]))
    return cands


def pre_classify(c: Candidate, lexicon: Lexicon, blocklist: dict[str, str]) -> None:
    """Tier before the LLM: known | block | watch | llm (= passes rule v1 steps 1-4)."""
    name = c.name
    if lexicon.lookup_any_type(name) or any(lexicon.lookup_any_type(n) for n in c.names):
        c.tier, c.reason = "known", "already_in_lexicon"
    elif c.key in blocklist:
        c.tier, c.reason = "block", blocklist[c.key]
    elif len(name) <= 4 or name.isupper():
        c.tier, c.reason = "watch", "abbreviation_never_auto"     # may be real (BSE), but never automatic
    elif (jr := junk_reason(name, c.type, False)):
        c.tier, c.reason = "block", f"junk:{jr}"
    elif len(c.papers) < MIN_PAPERS:
        c.tier, c.reason = "watch", f"papers_{len(c.papers)}<{MIN_PAPERS}"
    elif len(c.materials) < MIN_MATERIALS:
        c.tier, c.reason = "watch", f"materials_{len(c.materials)}<{MIN_MATERIALS}"
    elif c.type_share < MIN_TYPE_SHARE:
        c.tier, c.reason = "watch", f"type_share_{c.type_share:.2f}<{MIN_TYPE_SHARE}"
    else:
        c.tier, c.reason = "llm", ""


# ---------------------------------------------------------------------------------------- LLM check
SYSTEM = (
    "You curate a lexicon for a knowledge graph about laser powder bed fusion (metal additive manufacturing). "
    "Entity types: Material, Process, Parameter (machine setting), Property (measured property), Defect, "
    "Microstructure (also fracture features), PostProcessing, CharacterizationMethod, Geometry (specimen/part "
    "shape), ExperimentalCondition (build/test condition).\n"
    "For each candidate term decide:\n"
    "- concept: true only if it is a SPECIFIC, reusable technical concept of the given domain (not a generic "
    "word, not a sample label, not a sentence fragment, not specific to one paper).\n"
    "- type: the best entity type from the list.\n"
    "- synonym_of: if the term means exactly the same as one of the EXISTING terms listed for it, give that "
    "existing term verbatim; otherwise null. Do not use synonym_of for broader/narrower terms.\n"
    'Answer JSON only: {"decisions": [{"id": 1, "concept": true, "type": "Defect", "synonym_of": null}, ...]}'
)


class TermChecker:
    def __init__(self, llm: Any, cache_path: Path, batch_size: int = 8) -> None:
        self.llm = llm
        self.model = f"{getattr(llm, 'provider', '')}:{getattr(llm, 'model', 'llm')}".strip(":")
        self.cache_path = cache_path
        self.cache: dict[str, dict] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        self.batch_size = batch_size
        self.stopped = ""

    def check(self, cands: list[Candidate], lexicon: Lexicon) -> None:
        todo = [c for c in cands if f"{c.key}|{c.type}" not in self.cache]
        for i in range(0, len(todo), self.batch_size):
            if self.stopped:
                break
            self._batch(todo[i:i + self.batch_size], lexicon)
            self.cache_path.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False), encoding="utf-8")
        for c in cands:
            v = self.cache.get(f"{c.key}|{c.type}")
            if v is None:
                c.tier, c.reason = "watch", f"llm_pending{(': ' + self.stopped) if self.stopped else ''}"
            elif not v.get("concept"):
                c.tier, c.reason = "block", "llm_not_a_concept"
            elif v.get("synonym_of") and lexicon.lookup(v["synonym_of"], v.get("type") or c.type):
                c.tier, c.reason, c.target = "alias", "llm_synonym", v["synonym_of"]
            elif (v.get("type") or c.type) != c.type:
                c.tier, c.reason = "watch", f"llm_type_{v.get('type')}_differs"
            else:
                c.tier, c.reason = "auto", "rule_v1+llm"

    def _batch(self, batch: list[Candidate], lexicon: Lexicon) -> None:
        lines = []
        for n, c in enumerate(batch, 1):
            existing = sorted(e.canonical_name for e in lexicon.by_type(c.type))
            ex = "\n".join(f'   example: "{s}"' for _, s in c.examples[:3])
            lines.append(f"{n}. candidate: \"{c.name}\"   proposed type: {c.type}\n{ex}\n"
                         f"   existing {c.type} terms: {', '.join(existing)}")
        try:
            raw = self.llm.chat(SYSTEM, "Candidates:\n" + "\n".join(lines), json_mode=True)
            from src.llm.json_utils import parse_json_lenient
            data = parse_json_lenient(raw)
        except Exception as exc:              # daily limit, network, bad JSON: stop asking, keep the rest pending
            self.stopped = str(exc)[:120]
            print(f"  LLM check stopped: {self.stopped}", flush=True)
            return
        decisions = data.get("decisions", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        for d in decisions:
            try:
                c = batch[int(d.get("id")) - 1]
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
            t = d.get("type") if d.get("type") in TYPES else c.type
            self.cache[f"{c.key}|{c.type}"] = {"concept": bool(d.get("concept")), "type": t,
                                               "synonym_of": d.get("synonym_of") or None, "model": self.model,
                                               "term": c.name}


# ---------------------------------------------------------------------------------------- outputs
def write_auto_lexicon(path: Path, cands: list[Candidate], lexicon: Lexicon, model: str,
                       doi_of: dict[str, str]) -> tuple[int, int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    data = data or {}
    entries = {(e["type"], e["canonical"]): e for e in data.get("entries") or []}
    aliases = {(a["type"], a["canonical"], a["alias"]): a for a in data.get("aliases") or []}
    today = time.strftime("%Y-%m-%d")
    added_e = added_a = 0
    for c in cands:
        src = {"papers": len(c.papers), "materials": sorted(c.materials),
               "dois": sorted(doi_of.get(p, p) for p in c.papers)[:10], "mentions": c.mentions,
               "type_share": round(c.type_share, 2), "checked_by": model, "date": today}
        if c.tier == "auto" and (c.type, c.name) not in entries:
            variants = sorted({n for n in c.names if n != c.name and not lexicon.lookup_any_type(n)})[:5]
            entries[(c.type, c.name)] = {"type": c.type, "canonical": c.name, "aliases": variants, "source": src}
            added_e += 1
        elif c.tier == "alias":
            for n in sorted(set(c.names)):
                if (c.type, c.target, n) not in aliases and not lexicon.lookup_any_type(n):
                    aliases[(c.type, c.target, n)] = {"type": c.type, "canonical": c.target, "alias": n,
                                                      "source": src}
                    added_a += 1
    if not entries and not aliases and not path.exists():
        return 0, 0                                   # nothing discovered yet: do not create an empty file
    header = ("# AUTO-GENERATED by scripts/lexicon_discovery.py (rule v1: >=3 papers, >=2 materials, >=70% same\n"
              "# type, junk-free, LLM-verified). Loaded automatically next to lexicon.yaml with confidence 0.8.\n"
              "# Never edit lexicon.yaml from here. To undo all auto terms: rename or move this file.\n")
    body = {"entries": sorted(entries.values(), key=lambda e: (e["type"], e["canonical"].lower())),
            "aliases": sorted(aliases.values(), key=lambda a: (a["type"], a["canonical"], a["alias"]))}
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    return added_e, added_a


def write_candidates(path: Path, cands: list[Candidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "reason", "term", "type", "type_share", "papers", "materials", "mentions", "target",
                    "example"])
        order = {"auto": 0, "alias": 1, "llm": 2, "watch": 3, "block": 4, "known": 5}
        for c in sorted(cands, key=lambda c: (order.get(c.tier, 9), -len(c.papers), c.name.lower())):
            w.writerow([c.tier, c.reason, c.name, c.type, round(c.type_share, 2), len(c.papers),
                        "; ".join(sorted(c.materials)), c.mentions, c.target, c.examples[0][1] if c.examples else ""])


# ---------------------------------------------------------------------------------------- extraction
def extraction_settings(args: argparse.Namespace, work: Path) -> Any:
    from src.config import load_settings
    s = load_settings()
    s.paths.processed_dir = str(work / "processed")
    s.paths.extracted_dir = str(work / "extracted")
    s.paths.review_dir = str(work / "review")
    for d in ("processed", "extracted"):
        (work / d).mkdir(parents=True, exist_ok=True)
    s.extraction.entity_backend = "gliner"
    s.gliner.model, s.gliner.context, s.gliner.label_set = args.model, args.context, "lpbf"
    s.gliner.threshold, s.gliner.verify_llm = args.threshold, ""
    return s


def paper_record(res: Any, lexicon: Lexicon, name: str) -> dict:
    materials = Counter(m.canonical_name for m in res.entities
                        if m.entity_type == "Material" and m.canonical_id in lexicon.entities)
    mentions = []
    for m in res.entities:
        if m.method != "gliner" or not m.canonical_id or m.canonical_id in lexicon.entities:
            continue
        mentions.append({"text": m.text, "type": m.entity_type, "canonical": m.canonical_name,
                         "sentence": m.evidence.text if m.evidence else "",
                         "section": m.evidence.section_type if m.evidence else ""})
    return {"paper": name, "material": materials.most_common(1)[0][0] if materials else "",
            "mentions": mentions}


def extract_all(args: argparse.Namespace, work: Path) -> list[dict]:
    pdf_dir = ROOT / args.papers
    manifest = pdf_dir / "manifest.csv"
    ok = None
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            ok = {r["file"] for r in csv.DictReader(fh) if r.get("status") == "ok"}
    pdfs = sorted(p for p in pdf_dir.glob("*.pdf") if ok is None or p.name in ok)[: args.limit or None]
    cache = work / "papers"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pdfs if not (cache / f"{p.stem}.json").exists()]
    print(f"{len(pdfs)} harvested paper(s); {len(todo)} still to extract", flush=True)
    if todo:
        from src.pipeline import ExtractionPipeline
        pipe = ExtractionPipeline(extraction_settings(args, work), mode="gliner")
        for i, pdf in enumerate(todo, 1):
            t0 = time.time()
            try:
                res = pipe.process_file(pdf)
                rec = paper_record(res, pipe.lexicon, pdf.name)
            except Exception as exc:          # a bad PDF must not stop the run
                print(f"  [{i}/{len(todo)}] FAILED {pdf.name}: {exc}", flush=True)
                continue
            (cache / f"{pdf.stem}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            print(f"  [{i}/{len(todo)}] {pdf.name}: {len(rec['mentions'])} new-term mentions, "
                  f"material={rec['material'] or '?'} ({time.time() - t0:.0f}s)", flush=True)
    return [json.loads((cache / f"{p.stem}.json").read_text(encoding="utf-8"))
            for p in pdfs if (cache / f"{p.stem}.json").exists()]


def regression_check() -> None:
    """Rules-only entity F1 on DEV with and without the auto lexicon (TEST stays untouched)."""
    from scripts.gliner_sweep import Sweep
    f1 = {}
    for flag in ("0", "1"):
        os.environ["AMKG_LEXICON_AUTO"] = flag
        sw = Sweep("dev", ROOT / "data/annotations_silver", ROOT / "data/experiments/lexicon_check",
                   ROOT / "data/processed")
        f1[flag] = sw.run("rules_only" + (" + auto lexicon" if flag == "1" else ""), {}, mode="rule")["entity_f1"]
    os.environ.pop("AMKG_LEXICON_AUTO", None)
    if f1["1"] < f1["0"] - 0.005:
        print(f"WARNING: DEV entity F1 dropped {f1['0']:.3f} -> {f1['1']:.3f}. To undo: rename "
              f"config/{AUTO_FILE_NAME} (e.g. to lexicon_auto.rejected.yaml).")
    else:
        print(f"Regression check OK: DEV entity F1 {f1['0']:.3f} -> {f1['1']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--papers", default="data/harvest/papers")
    ap.add_argument("--work", default="data/harvest/discovery")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="urchade/gliner_large-v2.1")
    ap.add_argument("--context", default="neighbours")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--llm", default="groq:openai/gpt-oss-120b", help="checker, provider:model (free tier)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--check", action="store_true", help="regression check on DEV after writing")
    args = ap.parse_args()
    work = ROOT / args.work
    work.mkdir(parents=True, exist_ok=True)

    papers = extract_all(args, work)
    if not papers:
        raise SystemExit("No extracted papers. Run scripts/harvest_oa.py first.")
    lexicon = Lexicon.load(ROOT / "config" / "lexicon.yaml")         # curated + current auto terms
    blocklist_path = work / "blocklist.json"
    blocklist = json.loads(blocklist_path.read_text(encoding="utf-8")) if blocklist_path.exists() else {}
    cands = list(aggregate(papers).values())
    for c in cands:
        pre_classify(c, lexicon, blocklist)
    to_check = [c for c in cands if c.tier == "llm"]
    print(f"\n{len(cands)} candidate terms; {len(to_check)} pass rule v1 steps 1-4", flush=True)

    model = "none"
    if to_check and args.no_llm:
        for c in to_check:
            c.tier, c.reason = "watch", "llm_pending (--no-llm)"
    elif to_check:
        from src.config import load_settings
        from src.llm.openai_compat import parse_llm_spec
        if args.llm.startswith("ollama:"):
            from src.llm.client import OllamaClient
            st = load_settings()
            client: Any = OllamaClient(st.llm.base_url, args.llm.split(":", 1)[1], 0.0, st.llm.num_ctx,
                                       st.llm.timeout_s, False, num_predict=1500)
        else:
            client = parse_llm_spec(args.llm, load_settings(), num_predict=1500)
        checker = TermChecker(client, work / "llm_verdicts.json")
        model = checker.model
        checker.check(to_check, lexicon)

    for c in cands:
        if c.tier == "block" and c.reason.startswith("llm_"):
            blocklist[c.key] = c.reason
    blocklist_path.write_text(json.dumps(blocklist, indent=1, ensure_ascii=False), encoding="utf-8")
    write_candidates(work / "candidates.csv", cands)
    doi_of = {}
    manifest = ROOT / args.papers / "manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8") as fh:
            doi_of = {r["file"]: r["doi"] for r in csv.DictReader(fh)}
    added_e, added_a = write_auto_lexicon(ROOT / "config" / AUTO_FILE_NAME, cands, lexicon, model, doi_of)

    tiers = Counter(c.tier for c in cands)
    print(f"\nTiers: {dict(tiers)}")
    for tier in ("auto", "alias"):
        for c in [c for c in cands if c.tier == tier]:
            extra = f" -> {c.target}" if tier == "alias" else ""
            print(f"  {tier:5} {c.type:22} {c.name}{extra}  ({len(c.papers)} papers, {len(c.materials)} materials)")
    print(f"\nAdded to config/{AUTO_FILE_NAME}: {added_e} new term(s), {added_a} alias(es)")
    print(f"All candidates: {work / 'candidates.csv'}")
    if args.check:
        regression_check()


if __name__ == "__main__":
    main()
