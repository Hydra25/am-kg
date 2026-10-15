"""Error analysis for new GLiNER entities (DEV only): what does GLiNER propose, what does the checker
keep or drop, and how does each candidate compare with the silver annotations?

    python scripts/gliner_error_analysis.py            # large + neighbours + lpbf @ 0.4, checker qwen2.5:3b

Output: data/experiments/gliner_errors/candidates.csv, one row per NEW (non-lexicon) GLiNER candidate:
paper, sentence, term, type, canonical, score, checker (keep/drop/off), silver match:
  exact      - same type + same name in silver (true positive)
  other_type - same name in silver but another type
  partial    - a silver entity of the paper contains this term or is contained in it (naming variant)
  none       - not in silver at all
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gliner_sweep import SPLITS  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.entity_extraction import gliner_verifier as gv  # noqa: E402
from src.ingestion.loader import load_parsed  # noqa: E402
from src.normalization.text import match_key  # noqa: E402
from src.pipeline import ExtractionPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="urchade/gliner_large-v2.1")
    ap.add_argument("--context", default="neighbours")
    ap.add_argument("--label-set", default="lpbf")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--verify", default="ollama:qwen2.5:3b", help='checker LLM ("" = off)')
    ap.add_argument("--gold", default="data/annotations_silver")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/experiments/gliner_errors")
    args = ap.parse_args()
    out = ROOT / args.out
    (out / "extracted").mkdir(parents=True, exist_ok=True)

    s = load_settings()
    s.paths.extracted_dir = str(out / "extracted")
    s.paths.review_dir = str(out / "review")
    s.extraction.entity_backend = "gliner"
    s.gliner.model, s.gliner.context, s.gliner.label_set = args.model, args.context, args.label_set
    s.gliner.threshold, s.gliner.verify_llm = args.threshold, args.verify

    verdicts: dict[tuple[str, str, str], str] = {}
    original = gv.GlinerVerifier.verify

    def recording_verify(self, candidates):        # remember every checker answer
        res = original(self, candidates)
        for c, v in zip(candidates, res):
            verdicts[(c.sentence, c.text, c.entity_type)] = "keep" if v.keep else "drop"
        return res

    gv.GlinerVerifier.verify = recording_verify
    pipe = ExtractionPipeline(s, mode="gliner")
    lexicon = pipe.lexicon

    gold = {}
    for g in sorted((ROOT / args.gold).glob("*.json")):
        d = json.loads(g.read_text(encoding="utf-8"))
        gold[Path(d.get("source_file") or g.stem).stem] = d

    rows = []
    for p in sorted((ROOT / args.processed).glob("*.json")):
        doc = load_parsed(p)
        stem = Path(doc.paper.source_file or "").stem
        if stem not in SPLITS["dev"] or stem not in gold:
            continue
        print(f"{stem} ...", flush=True)
        res = pipe.extract(doc)
        sent_text = {x.sentence_id: x.text for x in doc.sentences}
        g_ents = [(e["type"], e["canonical"]) for e in gold[stem]["entities"]]
        g_keys = {(t, match_key(n)) for t, n in g_ents}
        g_names = {match_key(n) for _, n in g_ents}
        kept = {(m.sentence_id, m.text) for m in res.entities if m.method == "gliner"}
        seen = set()
        cands = [(m.sentence_id, m.text, m.entity_type, m.canonical_name, m.confidence)
                 for m in res.entities if m.method == "gliner"]
        dropped_file = out / "extracted" / "llm_audit" / f"{res.paper.paper_id}_dropped.json"
        if dropped_file.exists():
            for d in json.loads(dropped_file.read_text(encoding="utf-8")):
                if d.get("stage") == "gliner_check":
                    cands.append((d["sentence_id"], d["text"], d["type"], "", d.get("score", 0)))
        for sid, text, etype, canon, score in cands:
            if (sid, text) in seen:
                continue
            seen.add((sid, text))
            if canon and lexicon.lookup(canon, etype):
                continue                                    # lexicon entity: not a NEW candidate
            name = canon or text
            k = match_key(name)
            if (etype, k) in g_keys:
                match = "exact"
            elif k in g_names:
                match = "other_type"
            elif any(len(gk) > 3 and (gk in k or k in gk) for gk in g_names):
                match = "partial"
            else:
                match = "none"
            sentence = sent_text.get(sid, "")
            checker = verdicts.get((sentence, text, etype), "off" if not args.verify else "?")
            if (sid, text) not in kept:
                checker = "drop"
            rows.append({"paper": stem, "term": text, "type": etype, "canonical": canon, "score": score,
                         "checker": checker, "silver": match, "sentence": sentence})
    gv.GlinerVerifier.verify = original

    with (out / "candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["paper", "term", "type", "canonical", "score", "checker", "silver",
                                           "sentence"])
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"\n{len(rows)} new GLiNER candidates")
    print("silver match:", dict(Counter(r["silver"] for r in rows)))
    print("checker x silver:", dict(Counter((r["checker"], r["silver"]) for r in rows)))
    print(f"Saved {out / 'candidates.csv'}")


if __name__ == "__main__":
    main()
