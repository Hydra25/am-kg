"""GLiNER precision sweep: which labels / thresholds / filters give the best entity F1?

Tuning happens on the DEV papers only; TEST is used once, for the final numbers.

    python scripts/gliner_sweep.py --split dev            # sweep + per-type tuning -> best_config.json
    python scripts/gliner_sweep.py --split test           # rules only vs baseline vs best config (no tuning)
    python scripts/gliner_sweep.py --split dev --models urchade/gliner_large-v2.1,numind/NuNER_Zero
                                                          # compare zero-shot models (lpbf labels, 4 thresholds)
    python scripts/gliner_sweep.py --split dev --way3 --model urchade/gliner_large-v2.1
                                                          # context (sentence/neighbours) x label sets x thresholds
    python scripts/gliner_sweep.py --split dev --way2 --model urchade/gliner_large-v2.1 --context neighbours \
        --verify ollama:qwen2.5:3b                        # thresholds x (no checker / LLM checker)

GLiNER runs once per label set (predictions are cached at a low threshold); every config then only
re-filters, so the sweep costs about two GLiNER passes over the papers. Gold = silver annotations.
"""
from __future__ import annotations

import argparse
import csv
import gc
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
from src.normalization.text import match_key  # noqa: E402
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
MODEL_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
DEFAULT_MODELS = ["urchade/gliner_small-v2.1", "urchade/gliner_medium-v2.1", "urchade/gliner_large-v2.1",
                  "numind/NuNER_Zero", "knowledgator/gliner-multitask-large-v0.5"]
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
        self.gold: dict[str, dict] = {}
        for g in sorted(gold_dir.glob("*.json")):
            d = json.loads(g.read_text(encoding="utf-8"))
            stem = Path(d.get("source_file") or g.stem).stem
            if stem in found:
                shutil.copy(g, self.gold_dir / g.name)
                self.gold[stem] = d
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
        if self.gliner is not None and cfg.get("model") and cfg["model"] != self.gliner.model_name:
            self.gliner = None          # another model: free the old one before loading
            gc.collect()
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
        new_found, new_total = self.new_entity_recall(results)
        row = {"config": name, "entity_p": e["precision"], "entity_r": e["recall"], "entity_f1": e["f1"],
               "relation_f1": rep["relations_strict"]["f1"], "value_f1": rep["values"]["f1"],
               "new_entity_recall": round(new_found / new_total, 3) if new_total else 0.0,
               "gliner_mentions": sum(r.stats.gliner_mentions for r in results),
               "gliner_sec_per_paper": round(sum(r.stats.gliner_seconds for r in results) / len(results), 1),
               "seconds": round(time.perf_counter() - t0, 1), "settings": json.dumps(cfg, sort_keys=True)}
        self.rows.append(row)
        print(f"  {name:<44} P {row['entity_p']:.3f}  R {row['entity_r']:.3f}  F1 {row['entity_f1']:.3f}"
              f"  new-R {row['new_entity_recall']:.3f}  (gliner mentions {row['gliner_mentions']},"
              f" {row['gliner_sec_per_paper']}s/paper GLiNER)", flush=True)
        return row

    def new_entity_recall(self, results: list) -> tuple[int, int]:
        """Silver entities that are NOT in the lexicon: how many does the run find? (what GLiNER adds)"""
        lex = self.norm.lexicon
        found = total = 0
        for res in results:
            gold = self.gold.get(Path(res.paper.source_file or "").stem)
            if not gold:
                continue
            pred = set()
            for m in res.entities:
                ent = res.canonical_entities.get(m.canonical_id or "")
                if ent is not None:
                    pred.add((ent.entity_type, match_key(ent.canonical_name)))
            for g in {(e["type"], match_key(e["canonical"])): e for e in gold["entities"]}.values():
                if lex.lookup(g["canonical"], g["type"]):
                    continue
                total += 1
                found += (g["type"], match_key(g["canonical"])) in pred
        return found, total

    def save(self, tag: str) -> None:
        out = self.out_dir / f"results_{tag}"
        with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0]))
            w.writeheader()
            w.writerows(self.rows)
        lines = ["| config | entity P | entity R | entity F1 | new-entity R | relation F1 | value F1 | GLiNER mentions"
                 " | GLiNER s/paper |",
                 "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['config']} | {r['entity_p']:.3f} | {r['entity_r']:.3f} | {r['entity_f1']:.3f} | "
                  f"{r['new_entity_recall']:.3f} | {r['relation_f1']:.3f} | {r['value_f1']:.3f} | "
                  f"{r['gliner_mentions']} | {r['gliner_sec_per_paper']} |" for r in self.rows]
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


def compare_models(sw: Sweep, models: list[str], max_sec: float) -> dict[str, Any]:
    """Way 1: each zero-shot model with lpbf labels at a few thresholds (DEV)."""
    print("Rules only (reference)", flush=True)
    rules = sw.run("rules_only", {}, mode="rule")
    best: dict[str, tuple[dict, dict]] = {}
    for model in models:
        print(f"\nModel {model}", flush=True)
        for t in MODEL_THRESHOLDS:
            cfg = {**BASE, "model": model, "label_set": "lpbf", "threshold": t}
            try:
                r = sw.run(f"{model.split('/')[-1]} @ {t}", cfg)
            except Exception as exc:    # download / incompatible model: report and go on
                print(f"  {model}: could not run ({exc})", flush=True)
                break
            if model not in best or r["entity_f1"] > best[model][0]["entity_f1"]:
                best[model] = (r, cfg)
        sw.gliner = None
        gc.collect()
    speed = {m: max(x["gliner_sec_per_paper"] for x in sw.rows if x["config"].startswith(m.split("/")[-1] + " @"))
             for m in best}
    print("\nPer model (best threshold on DEV):")
    print(f"{'model':<44}{'thr':>5}{'P':>7}{'R':>7}{'F1':>7}{'new-R':>7}{'s/paper':>9}")
    for m, (r, cfg) in best.items():
        print(f"{m:<44}{cfg['threshold']:>5}{r['entity_p']:>7.3f}{r['entity_r']:>7.3f}{r['entity_f1']:>7.3f}"
              f"{r['new_entity_recall']:>7.3f}{speed[m]:>9}")
    usable = {m: v for m, v in best.items() if speed[m] <= max_sec} or best
    pick = max(usable, key=lambda m: usable[m][0]["entity_f1"])
    print(f"\nRules only: F1 {rules['entity_f1']:.3f}. Picked (<= {max_sec:.0f} s/paper): {pick}")
    return usable[pick][1]


def way3(sw: Sweep, model: str) -> None:
    """Way 3: more context and several labels per type, on one model (DEV)."""
    print("Rules only (reference)", flush=True)
    sw.run("rules_only", {}, mode="rule")
    for context in ("sentence", "neighbours"):
        for label_set in ("lpbf", "lpbf_multi"):
            print(f"\n{model} | context={context} | labels={label_set}", flush=True)
            for t in (0.5, 0.6, 0.7):
                cfg = {**BASE, "model": model, "label_set": label_set, "context": context, "threshold": t}
                sw.run(f"{context}/{label_set} @ {t}", cfg)


def way2(sw: Sweep, model: str, context: str, label_set: str, verifiers: list[str]) -> None:
    """Way 2: LLM checker for new GLiNER entities, at several thresholds (DEV)."""
    print("Rules only (reference)", flush=True)
    sw.run("rules_only", {}, mode="rule")
    for verify in ["", *verifiers]:
        tag = "no checker" if not verify else f"checker {verify}"
        print(f"\n{model.split('/')[-1]} | {context}/{label_set} | {tag}", flush=True)
        for t in (0.4, 0.5, 0.6, 0.7):
            cfg = {**BASE, "model": model, "label_set": label_set, "context": context, "threshold": t,
                   "verify_llm": verify}
            sw.run(f"{tag} @ {t}", cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--gold", default="data/annotations_silver")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/experiments/gliner_sweep")
    ap.add_argument("--config", default=None, help="best config JSON for --split test (default: from the dev run)")
    ap.add_argument("--models", default=None,
                    help="comma-separated GLiNER models to compare on DEV ('all' = the built-in list)")
    ap.add_argument("--max-sec", type=float, default=120, help="max GLiNER seconds per paper for --models")
    ap.add_argument("--way3", action="store_true", help="context x label-set sweep on --model (DEV)")
    ap.add_argument("--model", default="urchade/gliner_large-v2.1", help="GLiNER model for --way3 / --way2")
    ap.add_argument("--way2", action="store_true", help="LLM checker sweep on --model (DEV)")
    ap.add_argument("--context", default="neighbours", help="GLiNER context for --way2")
    ap.add_argument("--label-set", default="lpbf", help="GLiNER label set for --way2")
    ap.add_argument("--verify", default="ollama:qwen2.5:3b",
                    help="comma-separated checker LLMs for --way2, e.g. ollama:qwen2.5:3b,groq:openai/gpt-oss-120b")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    sw = Sweep(args.split, ROOT / args.gold, out, ROOT / args.processed)
    print(f"Split {args.split}: {len(sw.docs)} paper(s)", flush=True)
    if not sw.docs:
        raise SystemExit("No papers found - check --processed")

    if args.models:
        if args.split != "dev":
            raise SystemExit("--models is for model choice: use it with --split dev only")
        models = DEFAULT_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
        best = compare_models(sw, models, args.max_sec)
        (out / "best_model_config.json").write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
        print(f"Saved best model config: {out / 'best_model_config.json'}")
        sw.save("models_dev")
        return
    if args.way2:
        if args.split != "dev":
            raise SystemExit("--way2 is tuning: use it with --split dev only")
        way2(sw, args.model, args.context, args.label_set, [v.strip() for v in args.verify.split(",") if v.strip()])
        sw.save("way2_dev")
        return
    if args.way3:
        if args.split != "dev":
            raise SystemExit("--way3 is tuning: use it with --split dev only")
        way3(sw, args.model)
        sw.save("way3_dev")
        return
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
