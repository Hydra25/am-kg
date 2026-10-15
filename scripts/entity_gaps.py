"""Which silver entities does the system miss (false negatives) and which extra ones does it add?

Runs the pipeline in rule mode (lexicon only, no GLiNER/LLM) on the DEV papers and compares the
document-level entity sets with the silver annotations. Use it to decide what to add to the lexicon.

    python scripts/entity_gaps.py                 # DEV only (never tune on TEST)
    python scripts/entity_gaps.py --mode gliner   # rules + GLiNER

Output: data/experiments/entity_gaps/{fn,fp}.csv and a summary per type.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.gliner_sweep import SPLITS  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.ingestion.loader import load_parsed  # noqa: E402
from src.normalization.text import match_key  # noqa: E402
from src.pipeline import ExtractionPipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--mode", choices=["rule", "gliner"], default="rule")
    ap.add_argument("--gold", default="data/annotations_silver")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/experiments/entity_gaps")
    args = ap.parse_args()
    stems = set(SPLITS[args.split])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    s = load_settings()
    s.paths.extracted_dir = str(out / "extracted")
    s.paths.review_dir = str(out / "review")
    (out / "extracted").mkdir(exist_ok=True)
    pipe = ExtractionPipeline(s, mode=args.mode)

    gold = {}
    for g in sorted((ROOT / args.gold).glob("*.json")):
        d = json.loads(g.read_text(encoding="utf-8"))
        stem = Path(d.get("source_file") or g.stem).stem
        if stem in stems:
            gold[stem] = d
    fn_rows, fp_rows = [], []
    by_type = defaultdict(Counter)
    for p in sorted((ROOT / args.processed).glob("*.json")):
        doc = load_parsed(p)
        stem = Path(doc.paper.source_file or "").stem
        if stem not in gold:
            continue
        res = pipe.extract(doc)
        text = " ".join(x.text for x in doc.sentences)
        tkey = match_key(text)
        pred: dict[tuple[str, str], str] = {}
        for m in res.entities:
            if m.canonical_id and m.canonical_id in res.canonical_entities:
                e = res.canonical_entities[m.canonical_id]
                pred[(e.entity_type, match_key(e.canonical_name))] = e.canonical_name
        pred_keys_any_type = {k for _, k in pred}
        g_set = {(e["type"], match_key(e["canonical"])): e["canonical"] for e in gold[stem]["entities"]}
        for (t, k), name in sorted(g_set.items()):
            hit = (t, k) in pred
            by_type[t]["gold"] += 1
            by_type[t]["tp"] += hit
            if hit:
                continue
            words = re.findall(r"[a-z0-9]+", name.lower())
            in_text = bool(k) and k in tkey
            head_in_text = bool(words) and words[-1] in text.lower()
            if k in pred_keys_any_type:
                why = "found_with_other_type"
            elif in_text:
                why = "in_text_not_in_lexicon"
            elif head_in_text:
                why = "paraphrase_or_variant"
            else:
                why = "not_literally_in_text"
            fn_rows.append({"paper": stem, "type": t, "canonical": name, "why": why})
        for (t, k), name in sorted(pred.items()):
            if (t, k) not in g_set:
                by_type[t]["fp"] += 1
                fp_rows.append({"paper": stem, "type": t, "canonical": name})

    for name, rows in (("fn", fn_rows), ("fp", fp_rows)):
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            if rows:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print(f"{'type':<24}{'gold':>6}{'tp':>6}{'fn':>6}{'fp':>6}{'recall':>8}")
    for t, c in sorted(by_type.items(), key=lambda kv: -kv[1]["gold"]):
        r = c["tp"] / c["gold"] if c["gold"] else 0
        print(f"{t:<24}{c['gold']:>6}{c['tp']:>6}{c['gold'] - c['tp']:>6}{c['fp']:>6}{r:>8.2f}")
    print("\nWhy missed:", dict(Counter(r["why"] for r in fn_rows)))
    print(f"Saved {out / 'fn.csv'} and {out / 'fp.csv'}")


if __name__ == "__main__":
    main()
