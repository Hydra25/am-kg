"""Benchmark: run several extraction set-ups on the gold papers and compare them.

    python main.py benchmark data/papers --modes rule,gliner,hybrid_gliner,hybrid_llm

For every mode the extraction is written to data/benchmark/<mode>/extracted
(data/extracted is never touched) and scored with the same metrics as
`python main.py evaluate`. The output is the comparison table for the paper:
entity / relation / value P-R-F1 plus time, LLM calls and tokens per paper.

Human review decisions are NOT applied (each mode gets an empty review folder),
so the table measures the automatic system only.
"""
from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import Settings
from ..ingestion.loader import discover_inputs
from ..knowledge_graph.kb import KnowledgeBase
from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.lexicon import Lexicon
from .metrics import evaluate

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModeSpec:
    name: str
    mode: str                 # pipeline mode
    entities: Optional[str]   # entity backend (None = setting)
    description: str


MODES: dict[str, ModeSpec] = {
    "rule": ModeSpec("rule", "rule", None, "lexicon + regex rules only"),
    "gliner": ModeSpec("gliner", "gliner", "gliner", "rules + GLiNER entities, no LLM"),
    "hybrid_gliner": ModeSpec("hybrid_gliner", "hybrid", "gliner", "rules + GLiNER entities + LLM relations/findings"),
    "hybrid_llm": ModeSpec("hybrid_llm", "hybrid", "llm", "rules + LLM entities + LLM relations/findings"),
    "llm": ModeSpec("llm", "llm", "llm", "LLM only (no rule relations)"),
}


def gold_stems(gold_dir: Path) -> set[str]:
    stems = set()
    for f in Path(gold_dir).glob("*.json"):
        try:
            stems.add(json.loads(f.read_text(encoding="utf-8")).get("source_file") or f.stem)
        except (OSError, ValueError):
            log.warning("Unreadable gold file %s", f)
    return stems


def mode_settings(base: Settings, spec: ModeSpec, out_root: Path) -> Settings:
    s = base.model_copy(deep=True)
    root = out_root / spec.name
    s.paths.extracted_dir = str(root / "extracted")
    s.paths.review_dir = str(root / "review")      # empty: no human decisions applied
    if spec.entities:
        s.extraction.entity_backend = spec.entities
    return s


def run_benchmark(base: Settings, papers_path: Path, modes: list[str], gold_dir: Path,
                  out_root: Path, statuses: tuple[str, ...] = ("accepted", "review"),
                  llm: Any = None, gliner: Any = None) -> dict[str, Any]:
    from ..pipeline import ExtractionPipeline

    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s) {unknown}; choose from {sorted(MODES)}")
    stems = gold_stems(gold_dir)
    files = [f for f in discover_inputs(Path(papers_path)) if f.stem in stems]
    if not files:
        raise FileNotFoundError(f"No papers in {papers_path} have gold annotations in {gold_dir}")
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name in modes:
        spec = MODES[name]
        s = mode_settings(base, spec, out_root)
        log.info("Benchmark mode %s on %d paper(s)", name, len(files))
        uses_gliner = spec.entities == "gliner" or spec.mode == "gliner"
        pipe = ExtractionPipeline(s, llm=llm, mode=spec.mode, gliner=gliner if uses_gliner else None)
        if uses_gliner and pipe.gliner is not None:
            # load the GLiNER model once and share it between modes
            pipe.gliner.normalizer, pipe.gliner.rule_extractor = pipe.normalizer, pipe.rule_entities
            gliner = pipe.gliner
        t0 = time.perf_counter()
        results = []
        for f in files:
            results.append(pipe.process_file(f))
        wall = time.perf_counter() - t0
        kb = KnowledgeBase.load(s)
        norm = EntityNormalizer(Lexicon.load(s.path("lexicon_file")))
        rep = evaluate(kb, gold_dir, s.path("processed_dir"), norm, statuses)
        n = len(results)
        row = {
            "mode": name, "description": spec.description, "papers": n,
            "entity_p": rep["entities"]["precision"], "entity_r": rep["entities"]["recall"],
            "entity_f1": rep["entities"]["f1"],
            "relation_p": rep["relations_strict"]["precision"], "relation_r": rep["relations_strict"]["recall"],
            "relation_f1": rep["relations_strict"]["f1"], "relation_pair_f1": rep["relations_lenient_pair"]["f1"],
            "value_p": rep["values"]["precision"], "value_r": rep["values"]["recall"], "value_f1": rep["values"]["f1"],
            "sec_per_paper": round(sum(r.stats.seconds for r in results) / n, 1),
            "llm_calls_per_paper": round(sum(r.stats.llm_calls for r in results) / n, 1),
            "llm_out_tokens_per_paper": round(sum(r.stats.llm_output_tokens for r in results) / n),
            "cache_hits": sum(r.stats.llm_cache_hits for r in results),
            "wall_seconds": round(wall, 1),
        }
        rows.append(row)
        details[name] = {"metrics": rep, "stats": [r.stats.model_dump() for r in results]}
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {"papers": [f.name for f in files], "statuses": list(statuses), "rows": rows}
    (out_root / "results.json").write_text(json.dumps({**summary, "details": details}, indent=2,
                                                      ensure_ascii=False, default=str), encoding="utf-8")
    with (out_root / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_root / "results.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def to_markdown(summary: dict[str, Any]) -> str:
    lines = [f"# Extraction benchmark ({len(summary['papers'])} papers)", "",
             f"Claims counted: {', '.join(summary['statuses'])}. DERIVED relations excluded. No human review applied.", "",
             "| Mode | Entity P / R / F1 | Relation P / R / F1 | Pair F1 | Value P / R / F1 | s / paper | LLM calls / paper | out tokens / paper |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        lines.append(f"| {r['mode']} | {r['entity_p']} / {r['entity_r']} / **{r['entity_f1']}** | "
                     f"{r['relation_p']} / {r['relation_r']} / **{r['relation_f1']}** | {r['relation_pair_f1']} | "
                     f"{r['value_p']} / {r['value_r']} / **{r['value_f1']}** | {r['sec_per_paper']} | "
                     f"{r['llm_calls_per_paper']} | {r['llm_out_tokens_per_paper']} |")
    lines += ["", "Modes:"] + [f"- **{r['mode']}**: {r['description']}" for r in summary["rows"]]
    if any(r["cache_hits"] for r in summary["rows"]):
        lines += ["", "_Some LLM answers came from the cache; times of those modes are lower than a cold run._"]
    return "\n".join(lines)
