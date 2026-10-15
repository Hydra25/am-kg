"""GLiNER precision sweep: which labels / thresholds / filters give the best entity F1?

Tuning happens on the DEV papers only; TEST is used once, for the final numbers.

    python scripts/gliner_sweep.py --split dev            # sweep + per-type tuning -> best_config.json
    python scripts/gliner_sweep.py --split test           # rules only vs baseline vs best config (no tuning)

GLiNER runs once per label set (predictions are cached at a low threshold); every config then only
re-filters, so the sweep costs about two GLiNER passes over the papers. Gold = silver annotations.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_settings  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.ingestion.loader import load_parsed  # noqa: E402
from src.knowledge_graph.kb import KnowledgeBase  # noqa: E402
from src.normalization.entity_normalizer import EntityNormalizer  # noqa: E402
from src.normalization.lexicon import Lexicon  # noqa: E402
from src.pipeline import ExtractionPipeline  # noqa: E402

log = logging.getLogger("gliner_sweep")

SPLITS = {
    "dev": ["ti64_baskin2024", "ti64_cacace2026", "316l_lee2022", "alsi10mg_kumar2023", "in718_tan2023"],
    "test": ["ti64_lee2025", "ti64_squillaci2025", "316l_obeidi2021", "alsi10mg_minkowitz2023", "in718_baldi2023"],
}
SPLITS["all"] = SPLITS["dev"] + SPLITS["test"]
TYPES = ["Material", "Process", "Parameter", "Property", "Defect", "Microstructure", "PostProcessing",
         "CharacterizationMethod", "Geometry", "ExperimentalCondition"]
TYPE_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
PREDICT_FLOOR = 0.3
BASE = {"label_set": "default", "threshold": 0.5, "type_thresholds": {}, "require_support": False, "min_mentions": 2}


class Sweep:
    def __init__(self, split: str, gold_dir: Path, out_dir: Path, processed_dir: Path) -> None:
        self.base = load_settings()
        self.out_dir = out_dir
        self.processed_dir = processed_dir
        stems = SPLITS[split]
        self.docs: list[Path] = []
        for p in sorted(processed_dir.glob("*.json")):
            try:
                src = json.loads(p.read_text(encoding="utf-8"))["paper"].get("source_file") or ""
            except (OSError, ValueError, KeyError):
                continue
            if Path(src).stem in stems:
                self.docs.append(p)
        found = {Path(json.loads(p.read_text(encoding="utf-8"))["paper"]["source_file"]).stem for p in self.docs}
        missing = sorted(set(stems) - found)
        if missing:
            log.warning("No processed document for %s (run `python main.py extract` once first)", missing)
        # gold files of this split only
        self.gold_dir = out_dir / f"gold_{split}"
        shutil.rmtree(self.gold_dir, ignore_errors=True)
        self.gold_dir.mkdir(parents=True)
        for g in sorted(gold_dir.glob("*.json")):
            stem = json.loads(g.read_text(encoding="utf-8")).get("source_file") or g.stem
            if Path(stem).stem in found:
                shutil.copy(g, self.gold_dir / g.name)
        self.gliner: Any = None
        self.norm = EntityNormalizer(Lexicon.load(self.base.path("lexicon_file")))
        self.rows: list[dict[str, Any]] = []

    def run(self, name: str, cfg: dict[str, Any], mode: str = "gliner") -> dict[str, Any]:
        s = self.base.model_copy(deep=True)
        work = self.out_dir / "runs" / name
        s.paths.extracted_dir = str(work / "extracted")
        s.paths.review_dir = str(work / "review")
        (work / "extracted").mkdir(parents=True, exist_ok=True)
        s.extraction.entity_backend = "gliner" if mode == "gliner" else "none"
        for k, v in cfg.items():
            setattr(s.gliner, k, v)
        pipe = ExtractionPipeline(s, mode=mode, gliner=self.gliner if mode == "gliner" else None)
        if mode == "gliner":
            from src.entity_extraction.gliner_extractor import LABEL_SETS
            g = pipe.gliner
            g.labels = dict(s.gliner.labels or LABEL_SETS[s.gliner.label_set])
            g.threshold, g.type_thresholds = s.gliner.threshold, dict(s.gliner.type_thresholds)
            g.predict_floor = PREDICT_FLOOR
            self.gliner = g
        t0 = time.perf_counter()
        results = [pipe.extract(load_parsed(p)) for p in self.docs]
        kb = KnowledgeBase.from_results(results)
        rep = evaluate(kb, self.gold_dir, self.processed_dir, self.norm)
        e = rep["entities"]
        row = {"config": name, "entity_p": e["precision"], "entity_r": e["recall"], "entity_f1": e["f1"],
               "relation_f1": rep["relations_strict"]["f1"], "value_f1": rep["values"]["f1"],
               "gliner_mentions": sum(r.stats.gliner_mentions for r in results),
               "seconds": round(time.perf_counter() - t0, 1), "settings": json.dumps(cfg, sort_keys=True)}
        self.rows.append(row)
        print(f"  {name:<34} P {row['entity_p']:.3f}  R {row['entity_r']:.3f}  F1 {row['entity_f1']:.3f}"
              f"  (gliner mentions {row['gliner_mentions']}, {row['seconds']}s)", flush=True)
        return row

    def save(self, tag: str) -> None:
        out = self.out_dir / f"results_{tag}"
        with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        lines = ["| config | entity P | entity R | entity F1 | relation F1 | value F1 | GLiNER mentions |",
                 "|---|---|---|---|---|---|---|"]
        lines += [f"| {r['config']} | {r['entity_p']:.3f} | {r['entity_r']:.3f} | {r['entity_f1']:.3f} | "
                  f"{r['relation_f1']:.3f} | {r['value_f1']:.3f} | {r['gliner_mentions']} |" for r in self.rows]
        out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n" + "\n".join(lines))
        print(f"\nSaved: {out.with_suffix('.md')}")


def sweep_dev(sw: Sweep) -> dict[str, Any]:
    print("1) reference points", flush=True)
    sw.run("rules_only", {}, mode="rule")
    rows = {"default": sw.run("baseline (default labels, 0.5)", dict(BASE)),
            "lpbf": sw.run("lpbf labels, 0.5", {**BASE, "label_set": "lpbf"})}
    for t in (0.6, 0.7):
        sw.run(f"default labels, {t}", {**BASE, "threshold": t})
    label_set = max(rows, key=lambda k: rows[k]["entity_f1"])
    print(f"   -> label set: {label_set}", flush=True)

    print("2) per-type thresholds (one type at a time, others stay at 0.5)", flush=True)
    cfg = {**BASE, "label_set": label_set}
    best_f1 = rows[label_set]["entity_f1"]
    tuned: dict[str, float] = {}
    for etype in TYPES:
        best_t = 0.5
        for t in TYPE_GRID:
            if t == 0.5:
                continue
            r = sw.run(f"{label_set} | {etype}={t}", {**cfg, "type_thresholds": {**tuned, etype: t}})
            if r["entity_f1"] > best_f1 + 1e-4:
                best_f1, best_t = r["entity_f1"], t
        if best_t != 0.5:
            tuned[etype] = best_t
    print(f"   -> tuned thresholds: {tuned or 'none'}", flush=True)
    cfg = {**cfg, "type_thresholds": tuned}

    print("3) support filter (keep only useful new GLiNER entities)", flush=True)
    best = (sw.run("tuned", cfg), cfg)
    for n in (2, 3):
        c = {**cfg, "require_support": True, "min_mentions": n}
        r = sw.run(f"tuned + support (min {n})", c)
        if r["entity_f1"] > best[0]["entity_f1"] + 1e-4:
            best = (r, c)
    return best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--gold", default="data/annotations_silver")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/experiments/gliner_sweep")
    ap.add_argument("--config", default=None, help="best config JSON for --split test (default: from the dev run)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sw = Sweep(args.split, ROOT / args.gold, out, ROOT / args.processed)
    print(f"Split {args.split}: {len(sw.docs)} paper(s)", flush=True)
    if not sw.docs:
        raise SystemExit("No papers found - check --processed")

    if args.split == "dev":
        best = sweep_dev(sw)
        (out / "best_config.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"\nBest config (DEV): {json.dumps(best)}")
        rules = sw.rows[0]["entity_f1"]
        top = max(r["entity_f1"] for r in sw.rows[1:])
        if top <= rules:
            print(f"NOTE: no GLiNER config beats rules only on DEV (F1 {top:.3f} vs {rules:.3f}).")
    else:
        cfg_path = Path(args.config) if args.config else out / "best_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"{cfg_path} not found - run --split dev first")
        best = json.loads(cfg_path.read_text(encoding="utf-8"))
        sw.run("rules_only", {}, mode="rule")
        sw.run("baseline (default labels, 0.5)", dict(BASE))
        sw.run("best config from DEV", best)
    sw.save(args.split)


if __name__ == "__main__":
    main()
