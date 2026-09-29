"""AM-KG command line interface.

Examples
  python main.py check                          # Ollama / Neo4j / GROBID status
  python main.py make-demo                      # render DEMO/SYNTHETIC PDFs into data/papers
  python main.py extract data/papers --mode auto
  python main.py extract data/papers --mode hybrid --entities gliner
  python main.py annotate-export data/papers        # Label Studio tasks for the gold benchmark
  python main.py annotate-import export.json        # Label Studio export -> data/annotations
  python main.py benchmark data/papers              # compare modes on the gold papers
  python main.py build-graph                    # Neo4j if configured, else local store
  python main.py query studies_linking --param a="laser power" --param b=porosity
  python main.py why <claim_id>
  python main.py ask "What parameters affect porosity in LPBF Ti-6Al-4V?"
  python main.py evaluate
  python main.py gaps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import get_settings, setup_logging


def _print_rows(rows: list[dict], limit: int = 50) -> None:
    if not rows:
        print("(no results)")
        return
    for r in rows[:limit]:
        print(json.dumps(r, ensure_ascii=False, default=str))
    if len(rows) > limit:
        print(f"... {len(rows) - limit} more")


def cmd_check(args: argparse.Namespace) -> int:
    from src.knowledge_graph.store import neo4j_available
    from src.llm.client import OllamaClient
    from src.pdf_parser.grobid_parser import grobid_available
    s = get_settings()
    if s.llm.provider != "ollama":
        from src.llm.openai_compat import make_llm_client
        try:
            c = make_llm_client(s)
            print(f"LLM provider '{s.llm.provider}' ({c.base_url}), model '{s.llm.model}': "
                  f"{'reachable' if c.is_available() else 'NOT reachable (check key / model / internet)'}")
        except Exception as exc:
            print(f"LLM provider '{s.llm.provider}': {exc}")
    client = OllamaClient.from_settings(s)
    try:
        models = client.list_models()
        print(f"Ollama at {s.llm.base_url}: reachable; models={models}")
        print(f"  configured model '{s.llm.model}': {'available' if client.is_available() else 'NOT pulled -> ollama pull ' + s.llm.model}")
    except Exception as exc:
        print(f"Ollama at {s.llm.base_url}: NOT reachable ({exc.__class__.__name__}) -> start with 'ollama serve'")
    ok, why = neo4j_available(s)
    print(f"Neo4j: {'connected' if ok else 'not used/unreachable'} ({why})")
    print(f"GROBID: {'available' if grobid_available(s.grobid_url) else 'not configured/unreachable (PyMuPDF will be used)'}")
    try:
        import gliner  # noqa: F401
        gl = f"installed (model {s.gliner.model})"
    except ImportError:
        gl = "not installed (pip install gliner) -> LLM entity prompt is used"
    print(f"GLiNER: {gl}; entity_backend setting: {s.extraction.entity_backend}")
    print(f"Extraction mode setting: {s.extraction.mode}; thresholds accept>={s.quality.accept_threshold}, review>={s.quality.review_threshold}")
    return 0


def cmd_make_demo(args: argparse.Namespace) -> int:
    from src.ingestion.demo_pdf import build_demo_pdfs
    s = get_settings()
    paths = build_demo_pdfs(s.path("demo_dir"), s.path("papers_dir"))
    print("Created DEMO / SYNTHETIC PDFs (not real research):")
    for p in paths:
        print("  ", p)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from src.ingestion.loader import discover_inputs, parse_document, save_parsed
    s = get_settings()
    for f in discover_inputs(Path(args.path)):
        doc = parse_document(f, s)
        out = save_parsed(doc, s.path("processed_dir"))
        print(f"{f.name}: {doc.paper.paper_id} | title={doc.paper.title!r} | pages={doc.num_pages} | "
              f"sections={[x.section_type for x in doc.sections]} | sentences={len(doc.sentences)} | "
              f"tables={len(doc.tables)} -> {out}")
        for w in doc.warnings:
            print("   WARNING:", w)
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from src.pipeline import ExtractionPipeline
    s = get_settings()
    if getattr(args, "entities", None):
        s.extraction.entity_backend = args.entities
    pipe = ExtractionPipeline(s, mode=args.mode)
    results = pipe.process_many(Path(args.path))
    for r in results:
        st = r.stats
        print(f"{r.paper.source_file}: {r.paper.paper_id} mode={r.provenance.extraction_mode if r.provenance else '?'} "
              f"mentions={st.num_mentions} relations={st.num_relations} (accepted={st.accepted}, review={st.review}, "
              f"rejected={st.rejected}) findings={st.num_findings} llm_calls={st.llm_calls} time={st.seconds}s"
              f" entities_by={r.provenance.entity_backend if r.provenance else '?'}"
              + ("  [DEMO/SYNTHETIC]" if r.paper.is_synthetic else ""))
        if st.gliner_seconds:
            print(f"   GLiNER {st.gliner_seconds}s | {st.gliner_mentions} new entity mentions")
        if st.llm_by_stage:
            print(f"   LLM time {st.llm_seconds}s | chunks {st.llm_chunks - st.llm_chunks_skipped}/{st.llm_chunks} sent "
                  f"({st.llm_chunks_skipped} skipped by gating) | cache hits {st.llm_cache_hits} | "
                  f"tokens in/out {st.llm_prompt_tokens}/{st.llm_output_tokens}")
            for stage, v in st.llm_by_stage.items():
                print(f"     - {stage:<13} calls={v['calls']:<3} cached={v['cache_hits']:<3} "
                      f"time={v['seconds']:>7}s  out_tokens={v['output_tokens']}")
        for w in r.warnings:
            print("   WARNING:", w)
    print(f"Extracted JSON written to {get_settings().path('extracted_dir')}")
    return 0 if results else 1


def cmd_build_graph(args: argparse.Namespace) -> int:
    from src.knowledge_graph.store import get_store
    store = get_store(backend=args.backend)
    counts = store.build(reset=not args.no_reset)
    print(f"Graph built with backend '{store.name}': {counts}")
    store.close()
    return 0


def cmd_process(args: argparse.Namespace) -> int:
    rc = cmd_extract(args)
    if rc == 0:
        args.no_reset = False
        rc = cmd_build_graph(args)
    return rc


def _parse_params(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip().strip('"')
    return out


def cmd_queries(args: argparse.Namespace) -> int:
    from src.knowledge_graph.queries import QUERIES
    for q in QUERIES.values():
        params = list(q.entity_params) + q.extra_params
        print(f"{q.name:32} params={params}\n    {q.description}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from src.knowledge_graph.queries import QUERIES
    from src.knowledge_graph.store import get_store
    store = get_store(backend=args.backend)
    statuses = ["accepted"] if args.accepted_only else ["accepted", "review"]
    if args.show_cypher:
        print(QUERIES[args.name].cypher, "\n")
    _print_rows(store.run_query(args.name, statuses=statuses, **_parse_params(args.param)), args.limit)
    store.close()
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    from src.knowledge_graph.kb import KnowledgeBase
    kb = KnowledgeBase.load()
    c = kb.claim(args.claim_id)
    if not c:
        print("Unknown claim id")
        return 1
    p = kb.papers.get(c.evidence.paper_id) if c.evidence else None
    print(f"CLAIM {c.claim_id}: {c.subject.name} -[{c.relation}]-> {c.object.name}")
    print(f"  status={c.status.value} confidence={c.confidence} knowledge_type={c.knowledge_type.value} method={c.method}")
    print(f"  qc_reasons={c.qc_reasons} reviewed={c.reviewed} {c.review_note or ''}")
    if c.evidence:
        ev = c.evidence
        print(f"  paper: {ev.paper_id} {('- ' + p.title) if p else ''} {'[DEMO/SYNTHETIC]' if p and p.is_synthetic else ''}")
        print(f"  page {ev.page} | section '{ev.section}' ({ev.section_type}) | sentence {ev.sentence_id} "
              f"| source={ev.source_type}{' ' + ev.table_id if ev.table_id else ''}")
        if ev.context_sentence_ids:
            print(f"  context sentences used: {ev.context_sentence_ids}")
        print(f"  evidence: \"{ev.text}\"")
    print(f"  attributes: {json.dumps(c.attributes, ensure_ascii=False, default=str)}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from src.knowledge_graph.kb import KnowledgeBase
    from src.retrieval.assistant import ResearchAssistant
    kb = KnowledgeBase.load()
    ans = ResearchAssistant(kb, use_llm=not args.no_llm).answer(args.question)
    print(f"Q: {ans.question}\nLinked entities: {[(e.name, e.type) for e in ans.linked_entities]}\n")
    print(f"ANSWER ({ans.mode}):\n{ans.answer}\n")
    if ans.interpretation:
        print(f"INTERPRETATION (LLM-generated, not a corpus fact): {ans.interpretation}\n")
    print("EVIDENCE:")
    for it in ans.evidence:
        mark = "*" if it.eid in ans.cited_ids else " "
        head = f"{it.claim['subject']} -{it.claim['relation']}-> {it.claim['object']} [{it.claim['knowledge_type']}, {it.claim['status']}]" if it.claim else "passage"
        print(f" {mark}[{it.eid}] {head} | {it.paper_id} p.{it.page} {it.section}{' [SYNTHETIC]' if it.synthetic else ''}\n       \"{it.text[:220]}\"")
    for w in ans.warnings:
        print("WARNING:", w)
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from src.knowledge_graph.kb import KnowledgeBase
    from src.review.store import ReviewDecision, ReviewStore
    s = get_settings()
    if args.action == "list":
        kb = KnowledgeBase.load()
        for c in kb.review_queue()[: args.limit]:
            print(f"{c.claim_id} {c.confidence:.2f} {c.subject.name} -{c.relation}-> {c.object.name} | {c.qc_reasons}\n"
                  f"     \"{(c.evidence.text if c.evidence else '')[:160]}\"")
        print(f"New (non-lexicon) entities pending review: {[e.id for e in kb.new_entities()]}")
        return 0
    store = ReviewStore(s.path("review_dir"))
    if args.action == "merge":
        from src.normalization.entity_normalizer import NormalizationOverrides
        ov = NormalizationOverrides(s.path("review_dir") / "normalization_overrides.json")
        ov.add_merge(args.target, args.into)
        ov.save()
        store.add(ReviewDecision("entity", args.target, "merge", {"into": args.into}, note=args.note))
        print(f"Merged {args.target} -> {args.into}. Run 'python main.py build-graph' to apply.")
        return 0
    kind = "entity" if ":" in args.target else ("finding" if args.target.startswith("f_") else "claim")
    store.add(ReviewDecision(kind, args.target, args.action, note=args.note))
    print(f"Recorded {args.action} for {kind} {args.target}. Run 'python main.py build-graph' to apply.")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from src.evaluation.metrics import evaluate, graph_quality, to_markdown
    from src.knowledge_graph.kb import KnowledgeBase
    from src.normalization.entity_normalizer import EntityNormalizer, NormalizationOverrides
    from src.normalization.lexicon import Lexicon
    s = get_settings()
    kb = KnowledgeBase.load()
    norm = EntityNormalizer(Lexicon.load(s.path("lexicon_file")),
                            NormalizationOverrides(s.path("review_dir") / "normalization_overrides.json"))
    statuses = ["accepted"] if args.accepted_only else ["accepted", "review"]
    rep = evaluate(kb, Path(args.gold or s.path("annotations_dir")), s.path("processed_dir"), norm, statuses,
                   include_derived=args.include_derived)
    q = graph_quality(kb, s.path("processed_dir"), statuses)
    out_dir = s.root / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evaluation_report.json").write_text(json.dumps({"metrics": rep, "graph_quality": q}, indent=2,
                                                               ensure_ascii=False, default=str), encoding="utf-8")
    md = to_markdown(rep, q)
    (out_dir / "evaluation_report.md").write_text(md, encoding="utf-8")
    print(md)
    if any(p.is_synthetic for p in kb.papers.values()):
        print("\nNOTE: scores on DEMO / SYNTHETIC papers only test the pipeline; annotate real papers for scientific evaluation.")
    return 0


def cmd_gaps(args: argparse.Namespace) -> int:
    from src.analysis.gaps import find_gaps
    from src.knowledge_graph.kb import KnowledgeBase
    s = get_settings()
    kb = KnowledgeBase.load()
    rep = find_gaps(kb, s.gaps.min_pair_papers, s.gaps.max_context_papers,
                    include_synthetic=not args.exclude_synthetic, limit=args.limit)
    print("EXPERIMENTAL research-gap candidates (NOT confirmed research gaps)")
    print("Corpus:", json.dumps(rep.corpus, ensure_ascii=False))
    for c in rep.candidates:
        print(f"- [{c.kind}] {c.statement}\n    context papers={c.context_papers} reference papers={c.reference_papers}")
    print("Caveats:")
    for cv in rep.caveats:
        print("  *", cv)
    out = s.root / "data" / "eval" / "gap_candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def cmd_annotate_export(args: argparse.Namespace) -> int:
    """Write Label Studio tasks (+ labeling config) for the given papers."""
    from src.evaluation.labelstudio import build_tasks, labeling_config
    from src.ingestion.loader import discover_inputs, load_parsed, paper_id_for, parse_document, save_parsed
    from src.pipeline import load_result
    s = get_settings()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labeling_config.xml").write_text(labeling_config(), encoding="utf-8")
    tasks, missing = [], []
    for f in discover_inputs(Path(args.path)):
        pid = paper_id_for(f)
        pp = s.path("processed_dir") / f"{pid}.json"
        doc = load_parsed(pp) if pp.exists() else parse_document(f, s)
        if not pp.exists():
            save_parsed(doc, s.path("processed_dir"))
        result = None
        if not args.no_predictions:
            ep = s.path("extracted_dir") / f"{pid}.json"
            if ep.exists():
                result = load_result(ep)
            else:
                missing.append(f.name)
        new = build_tasks(doc, result, max_sentences=args.chunk_sentences)
        tasks += new
        print(f"{f.name}: {len(new)} tasks{' (with pre-annotations)' if result else ''}"
              + ("  [DEMO/SYNTHETIC]" if doc.paper.is_synthetic else ""))
    (out_dir / "tasks.json").write_text(json.dumps(tasks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(tasks)} tasks -> {out_dir / 'tasks.json'}\nLabeling config -> {out_dir / 'labeling_config.xml'}")
    if missing:
        print("No extraction yet (tasks without pre-annotations) for: " + ", ".join(missing)
              + "\n  -> run: python main.py extract <pdf> --mode hybrid --entities gliner")
    return 0


def cmd_annotate_import(args: argparse.Namespace) -> int:
    """Convert a Label Studio JSON export into gold files in data/annotations/."""
    from src.evaluation.labelstudio import import_annotations, write_gold
    from src.normalization.entity_normalizer import EntityNormalizer
    from src.normalization.lexicon import Lexicon
    s = get_settings()
    export = json.loads(Path(args.export).read_text(encoding="utf-8"))
    norm = EntityNormalizer(Lexicon.load(s.path("lexicon_file")))
    papers = import_annotations(export, norm, annotator=args.annotator,
                                annotation_mode="from_scratch" if args.from_scratch else "model_assisted")
    written = write_gold(papers, Path(args.out or s.path("annotations_dir")), overwrite=args.overwrite)
    for stem, g in papers.items():
        print(f"{stem}: entities={len(g['entities'])} relations={len(g['relations'])} values={len(g['values'])} "
              f"normalization_pairs={len(g['normalization'])}")
    print(f"Wrote {len(written)} gold file(s).")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from src.evaluation.benchmark import run_benchmark
    s = get_settings()
    statuses = ("accepted",) if args.accepted_only else ("accepted", "review")
    summary = run_benchmark(s, Path(args.path), [m.strip() for m in args.modes.split(",") if m.strip()],
                            Path(args.gold or s.path("annotations_dir")), s.root / "data" / "benchmark", statuses)
    from src.evaluation.benchmark import to_markdown
    print(to_markdown(summary))
    print(f"\nSaved: {s.root / 'data' / 'benchmark'}/results.(md|csv|json)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from src.knowledge_graph.kb import KnowledgeBase
    from src.reporting.tables import build_tables, write_csvs, write_workbook
    s = get_settings()
    statuses = ["accepted", "review"] + (["rejected"] if args.include_rejected else [])
    if args.accepted_only:
        statuses = ["accepted"]
    kb = KnowledgeBase.load()
    tables = build_tables(kb, statuses, set(args.paper) if args.paper else None,
                          include_synthetic=not args.exclude_synthetic)
    out_dir = s.root / "data" / "exports"
    xlsx = write_workbook(tables, out_dir / (args.name + ".xlsx"))
    csvs = write_csvs(tables, out_dir / "csv")
    print(f"Excel workbook: {xlsx}")
    print(f"CSV files:      {csvs[0].parent}")
    for name, rows in tables.items():
        print(f"  {name:26} {len(rows):5d} rows")
    print(f"Statuses included: {statuses}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    from src.knowledge_graph.kb import KnowledgeBase
    print(json.dumps(KnowledgeBase.load().stats(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Additive Manufacturing Knowledge Graph")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("make-demo").set_defaults(fn=cmd_make_demo)
    sp = sub.add_parser("parse"); sp.add_argument("path"); sp.set_defaults(fn=cmd_parse)
    for name, fn in (("extract", cmd_extract), ("process", cmd_process)):
        sp = sub.add_parser(name)
        sp.add_argument("path")
        sp.add_argument("--mode", choices=["auto", "rule", "gliner", "llm", "hybrid"], default=None)
        sp.add_argument("--entities", choices=["auto", "gliner", "llm", "none"], default=None,
                        help="who proposes non-lexicon entities (default: settings.yaml extraction.entity_backend)")
        sp.add_argument("--backend", choices=["auto", "neo4j", "local"], default=None)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("build-graph")
    sp.add_argument("--backend", choices=["auto", "neo4j", "local"], default=None)
    sp.add_argument("--no-reset", action="store_true")
    sp.set_defaults(fn=cmd_build_graph)
    sub.add_parser("queries").set_defaults(fn=cmd_queries)
    sp = sub.add_parser("query")
    sp.add_argument("name")
    sp.add_argument("--param", action="append", default=[], help='key=value, e.g. material="Ti6Al4V"')
    sp.add_argument("--backend", choices=["auto", "neo4j", "local"], default=None)
    sp.add_argument("--accepted-only", action="store_true")
    sp.add_argument("--show-cypher", action="store_true")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(fn=cmd_query)
    sp = sub.add_parser("why"); sp.add_argument("claim_id"); sp.set_defaults(fn=cmd_why)
    sp = sub.add_parser("ask"); sp.add_argument("question"); sp.add_argument("--no-llm", action="store_true")
    sp.set_defaults(fn=cmd_ask)
    sp = sub.add_parser("review", help="list | accept | reject | merge")
    sp.add_argument("action", choices=["list", "accept", "reject", "merge"])
    sp.add_argument("target", nargs="?", help="claim id, finding id or entity id (Type:name)")
    sp.add_argument("--into", help="target entity id for merge")
    sp.add_argument("--note")
    sp.add_argument("--limit", type=int, default=30)
    sp.set_defaults(fn=cmd_review)
    sp = sub.add_parser("evaluate")
    sp.add_argument("--gold", default=None)
    sp.add_argument("--accepted-only", action="store_true")
    sp.add_argument("--include-derived", action="store_true")
    sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("annotate-export", help="Label Studio tasks (+ pre-annotations) for building gold data")
    sp.add_argument("path")
    sp.add_argument("--out", default="data/labelstudio")
    sp.add_argument("--no-predictions", action="store_true", help="empty tasks: annotate from scratch")
    sp.add_argument("--chunk-sentences", type=int, default=12)
    sp.set_defaults(fn=cmd_annotate_export)
    sp = sub.add_parser("annotate-import", help="Label Studio JSON export -> data/annotations/*.json")
    sp.add_argument("export")
    sp.add_argument("--out", default=None)
    sp.add_argument("--annotator", default="")
    sp.add_argument("--from-scratch", action="store_true", help="tasks were annotated without pre-annotations")
    sp.add_argument("--overwrite", action="store_true")
    sp.set_defaults(fn=cmd_annotate_import)
    sp = sub.add_parser("benchmark", help="compare extraction modes on the gold papers")
    sp.add_argument("path")
    sp.add_argument("--modes", default="rule,gliner,hybrid_gliner,hybrid_llm")
    sp.add_argument("--gold", default=None)
    sp.add_argument("--accepted-only", action="store_true")
    sp.set_defaults(fn=cmd_benchmark)
    sp = sub.add_parser("gaps")
    sp.add_argument("--exclude-synthetic", action="store_true")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(fn=cmd_gaps)
    sp = sub.add_parser("export", help="write Excel + CSV tables of the extracted knowledge")
    sp.add_argument("--paper", action="append", help="only this paper id (repeatable)")
    sp.add_argument("--include-rejected", action="store_true")
    sp.add_argument("--accepted-only", action="store_true")
    sp.add_argument("--exclude-synthetic", action="store_true")
    sp.add_argument("--name", default="amkg_extracted_tables")
    sp.set_defaults(fn=cmd_export)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    s = get_settings()
    setup_logging(args.log_level or s.log_level, s.root / s.paths.logs_dir / "amkg.log")
    if args.cmd == "review" and args.action != "list" and not args.target:
        print("review accept/reject/merge needs a target id")
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
