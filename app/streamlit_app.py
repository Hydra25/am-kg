"""AM-KG Streamlit interface.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402  (installed with streamlit)
import streamlit as st  # noqa: E402

from src.config import get_settings  # noqa: E402
from src.knowledge_graph.export import graph_rows, to_networkx  # noqa: E402
from src.knowledge_graph.kb import KnowledgeBase  # noqa: E402
from src.knowledge_graph.queries import QUERIES  # noqa: E402
from src.review.store import ReviewDecision, ReviewStore  # noqa: E402

st.set_page_config(page_title="AM Knowledge Graph", layout="wide")
S = get_settings()
TYPE_COLORS = {"Paper": "#7f7f7f", "Material": "#1f77b4", "Process": "#d62728", "Parameter": "#ff7f0e",
               "Property": "#2ca02c", "Defect": "#9467bd", "Microstructure": "#8c564b",
               "PostProcessing": "#e377c2", "CharacterizationMethod": "#17becf", "Geometry": "#bcbd22",
               "ExperimentalCondition": "#aec7e8", "ParameterValue": "#ffbb78", "PropertyValue": "#98df8a"}


def show_df(df) -> None:
    """st.dataframe across Streamlit versions (width= replaced use_container_width=)."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:  # pragma: no cover - older Streamlit
        st.dataframe(df, use_container_width=True)


@st.cache_resource(show_spinner="Loading knowledge base ...")
def load_kb(version: int) -> KnowledgeBase:
    return KnowledgeBase.load(S)


def kb() -> KnowledgeBase:
    return load_kb(st.session_state.get("kb_version", 0))


def refresh() -> None:
    st.session_state["kb_version"] = st.session_state.get("kb_version", 0) + 1


def claim_table(claims) -> pd.DataFrame:
    rows = []
    for c in claims:
        ev = c.evidence
        rows.append({"status": c.status.value, "conf": c.confidence, "type": c.knowledge_type.value,
                     "subject": c.subject.name, "relation": c.relation, "object": c.object.name,
                     "trend": c.attributes.get("trend"), "negated": c.attributes.get("negated"),
                     "page": ev.page if ev else None, "section": ev.section if ev else None,
                     "evidence": ev.text if ev else None, "method": c.method,
                     "qc": ", ".join(c.qc_reasons), "claim_id": c.claim_id})
    return pd.DataFrame(rows)


def render_graph(rows: dict, height: int = 650, include_values: bool = False) -> None:
    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("pip install pyvis to see the interactive graph")
        return
    g = to_networkx(rows, include_values=include_values)
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff")
    for n, d in g.nodes(data=True):
        net.add_node(n, label=d["label"], title=f"{d['entity_type']}: {d['label']}",
                     color=TYPE_COLORS.get(d["entity_type"], "#cccccc"), size=18 if d["kind"] == "Entity" else 10)
    for u, v, d in g.edges(data=True):
        net.add_edge(u, v, label=d["rel"], title=f"{d['rel']} | papers={d['papers']} | {d['status']} | {d['knowledge_types']}",
                     dashes=d["status"] != "accepted", width=1 + d["papers"])
    net.set_options('{"physics": {"barnesHut": {"springLength": 160}}, "edges": {"font": {"size": 9}, "arrows": {"to": {"enabled": true}}}}')
    html = net.generate_html()
    if hasattr(st, "iframe"):          # Streamlit >= 1.50
        st.iframe(html, height=height + 20)
    else:                              # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height + 20)


# ---------------------------------------------------------------------------
page = st.sidebar.radio("Page", ["Upload & Extract", "Review", "Knowledge Graph", "Graph Explorer",
                                 "Research Assistant", "Paper Evidence", "Export Tables", "Queries", "Research Gaps",
                                 "Evaluation"])
st.sidebar.caption(f"Model: {S.llm.model} @ {S.llm.base_url}")
st.sidebar.caption(f"Graph backend: {S.graph.backend} | Neo4j: {'configured' if S.neo4j.enabled else 'not configured'}")
if st.sidebar.button("Reload knowledge base"):
    refresh()
K = kb()
st.sidebar.metric("Papers", len(K.papers))
st.sidebar.metric("Claims (accepted / review)", f"{sum(c.status.value == 'accepted' for c in K.claims)} / "
                                                f"{sum(c.status.value == 'review' for c in K.claims)}")
if any(p.is_synthetic for p in K.papers.values()):
    st.sidebar.warning("Corpus contains DEMO / SYNTHETIC papers (not real research).")

# ---------------------------------------------------------------------------
if page == "Upload & Extract":
    st.header("Upload a paper and extract knowledge")
    up = st.file_uploader("PDF (or .txt/.md)", type=["pdf", "txt", "md"])
    mode = st.selectbox("Extraction mode", ["auto", "rule", "hybrid", "llm"],
                        help="auto = hybrid if Ollama is reachable, otherwise rule-based")
    if up and st.button("Run extraction"):
        dest = S.path("papers_dir") / up.name
        dest.write_bytes(up.getvalue())
        from src.pipeline import ExtractionPipeline
        try:
            with st.spinner("Parsing and extracting (LLM calls can take a few minutes on a laptop) ..."):
                res = ExtractionPipeline(mode=mode).process_file(dest)
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.stop()
        st.session_state["last_result"] = res.paper.paper_id
        refresh()
        st.success(f"{res.paper.title} -> {res.paper.paper_id} | mode={res.provenance.extraction_mode} | "
                   f"{res.stats.num_relations} claims ({res.stats.accepted} accepted, {res.stats.review} review, "
                   f"{res.stats.rejected} rejected) in {res.stats.seconds}s")
        for w in res.warnings:
            st.warning(w)
    pid = st.session_state.get("last_result") or (sorted(K.papers)[-1] if K.papers else None)
    if pid and pid in K.papers:
        from src.pipeline import load_result
        res = load_result(S.path("extracted_dir") / f"{pid}.json")
        st.subheader(f"Result: {res.paper.title}")
        if res.paper.is_synthetic:
            st.info("DEMO / SYNTHETIC document")
        t1, t2, t3 = st.tabs(["Entities", "Relations", "Findings"])
        with t1:
            show_df(pd.DataFrame([{"text": m.text, "type": m.entity_type, "canonical": m.canonical_name,
                                        "norm_conf": m.normalization_confidence, "norm_method": m.normalization_method,
                                        "value": m.quantity.raw if m.quantity else None,
                                        "normalized": f"{m.quantity.normalized_value or ''} {m.quantity.normalized_unit or ''}" if m.quantity else None,
                                        "conf": m.confidence, "method": m.method, "sentence": m.sentence_id}
                                       for m in res.entities]))
        with t2:
            show_df(claim_table(res.relations))
        with t3:
            show_df(pd.DataFrame([{"status": f.status.value, "type": f.finding_type, "conf": f.confidence,
                                        "page": f.evidence.page, "section": f.evidence.section, "text": f.text,
                                        "method": f.method} for f in res.findings]))
        if st.button("Build / update graph now"):
            from src.knowledge_graph.store import get_store
            store = get_store(S, K)
            st.success(f"Graph built ({store.name}): {store.build()}")
            store.close()

elif page == "Review":
    st.header("Human-in-the-loop review")
    store = ReviewStore(S.path("review_dir"))
    queue = K.review_queue()
    st.caption(f"{len(queue)} claims need review. Decisions are appended to data/review/decisions.jsonl "
               f"(audit log) and applied when the knowledge base is reloaded.")
    for c in queue[:30]:
        with st.expander(f"{c.confidence:.2f} | {c.subject.name} -[{c.relation}]-> {c.object.name}"):
            ev = c.evidence
            st.markdown(f"**Evidence** ({ev.paper_id}, p.{ev.page}, *{ev.section}*):\n\n> {ev.text}" if ev else "no evidence")
            st.write({"knowledge_type": c.knowledge_type.value, "method": c.method, "qc_reasons": c.qc_reasons,
                      "attributes": c.attributes})
            col1, col2, col3 = st.columns(3)
            note = st.text_input("Note", key=f"n_{c.claim_id}")
            if col1.button("Accept", key=f"a_{c.claim_id}"):
                store.add(ReviewDecision("claim", c.claim_id, "accept", note=note or None)); refresh(); st.rerun()
            if col2.button("Reject", key=f"r_{c.claim_id}"):
                store.add(ReviewDecision("claim", c.claim_id, "reject", note=note or None)); refresh(); st.rerun()
            with col3.popover("Edit"):
                from src.ontology.schema import RelationType
                rel = st.selectbox("Relation", [r.value for r in RelationType], index=[r.value for r in RelationType].index(c.relation),
                                   key=f"er_{c.claim_id}")
                ents = sorted(K.entities)
                subj = st.selectbox("Subject", ents, index=ents.index(c.subject.id) if c.subject.id in ents else 0, key=f"es_{c.claim_id}")
                obj = c.object.id
                if c.object.id in ents:
                    obj = st.selectbox("Object", ents, index=ents.index(c.object.id), key=f"eo_{c.claim_id}")
                if st.button("Save edit", key=f"ed_{c.claim_id}"):
                    payload = {"relation": rel, "subject_id": subj}
                    if obj != c.object.id:
                        payload["object_id"] = obj
                    store.add(ReviewDecision("claim", c.claim_id, "edit", payload, note=note or None)); refresh(); st.rerun()
    st.subheader("Entities not in the lexicon (merge or accept)")
    for e in K.new_entities():
        cols = st.columns([3, 3, 1, 1])
        cols[0].write(f"**{e.canonical_name}** ({e.entity_type}) aliases={e.aliases}")
        same_type = sorted(i for i, x in K.entities.items() if x.entity_type == e.entity_type and i != e.id)
        target = cols[1].selectbox("Merge into", ["-"] + same_type, key=f"m_{e.id}")
        if cols[2].button("Merge", key=f"mb_{e.id}") and target != "-":
            from src.normalization.entity_normalizer import NormalizationOverrides
            ov = NormalizationOverrides(S.path("review_dir") / "normalization_overrides.json")
            ov.add_merge(e.id, target); ov.save()
            store.add(ReviewDecision("entity", e.id, "merge", {"into": target})); refresh(); st.rerun()
        if cols[3].button("Accept", key=f"ea_{e.id}"):
            store.add(ReviewDecision("entity", e.id, "accept")); refresh(); st.rerun()
    st.subheader("Merge any two entities")
    ents = sorted(K.entities)
    c1, c2 = st.columns(2)
    src = c1.selectbox("Merge this entity", ents, key="msrc")
    dst = c2.selectbox("into", ents, key="mdst")
    if st.button("Merge entities") and src != dst:
        from src.normalization.entity_normalizer import NormalizationOverrides
        ov = NormalizationOverrides(S.path("review_dir") / "normalization_overrides.json")
        ov.add_merge(src, dst); ov.save()
        store.add(ReviewDecision("entity", src, "merge", {"into": dst})); refresh(); st.rerun()
    with st.expander("Decision log"):
        show_df(pd.DataFrame([d.__dict__ for d in store.all()]))

elif page == "Knowledge Graph":
    st.header("Knowledge graph")
    c1, c2, c3 = st.columns(3)
    statuses = c1.multiselect("Claim status", ["accepted", "review"], default=["accepted", "review"])
    include_values = c2.checkbox("Show value nodes", value=False)
    types = c3.multiselect("Entity types", list(TYPE_COLORS), default=[t for t in TYPE_COLORS if "Value" not in t])
    rows = graph_rows(K, statuses)
    keep = {e["id"] for e in rows["entities"] if e["entity_type"] in types} | {p["paper_id"] for p in rows["papers"] if "Paper" in types}
    rows["entities"] = [e for e in rows["entities"] if e["id"] in keep]
    rows["papers"] = [p for p in rows["papers"] if p["paper_id"] in keep]
    st.caption("Dashed edges contain only claims under review. Edge width = number of supporting papers.")
    render_graph(rows, include_values=include_values)
    st.json(K.stats(), expanded=False)

elif page == "Graph Explorer":
    st.header("Graph explorer")
    etype = st.selectbox("Entity type", ["Material", "Process", "Parameter", "Property", "Defect", "Microstructure",
                                         "PostProcessing", "CharacterizationMethod", "Geometry", "ExperimentalCondition"])
    options = sorted((e.canonical_name, cid) for cid, e in K.entities.items() if e.entity_type == etype)
    if not options:
        st.info("No entities of this type yet.")
    else:
        name = st.selectbox("Entity", [o[0] for o in options])
        cid = dict(options)[name]
        ent = K.entities[cid]
        st.write({"canonical_name": ent.canonical_name, "aliases": ent.aliases, "family": ent.family,
                  "papers": sorted(K.entity_papers.get(cid, []))})
        claims = [c for c in K.active_claims() if cid in (c.subject.id, c.object.id)]
        show_df(claim_table(claims))
        sub = graph_rows(K)
        ids = {cid} | {c.subject.id for c in claims} | {c.object.id for c in claims}
        sub["entities"] = [e for e in sub["entities"] if e["id"] in ids]
        sub["papers"] = [p for p in sub["papers"] if p["paper_id"] in ids]
        sub["edges"] = [e for e in sub["edges"] if e["s"] in ids and e["o"] in ids and cid in (e["s"], e["o"])]
        render_graph(sub, height=450)

elif page == "Research Assistant":
    st.header("Research assistant (Graph + RAG, local Qwen)")
    q = st.text_input("Question", "What parameters affect porosity in LPBF Ti-6Al-4V?")
    use_llm = st.checkbox("Use local LLM for the answer (otherwise deterministic graph-only answer)", value=True)
    if st.button("Ask") and q:
        from src.retrieval.assistant import ResearchAssistant
        with st.spinner("Retrieving evidence ..."):
            ans = ResearchAssistant(K, S, use_llm=use_llm).answer(q)
        st.caption(f"Linked entities: {[(e.name, e.type) for e in ans.linked_entities]} | mode: {ans.mode}")
        st.markdown("### Answer (from corpus evidence)")
        st.markdown(ans.answer.replace("\n", "  \n"))
        if ans.interpretation:
            st.markdown("### Interpretation (LLM-generated, NOT a corpus fact)")
            st.info(ans.interpretation)
        for w in ans.warnings:
            st.warning(w)
        st.markdown("### Evidence")
        show_df(pd.DataFrame([{"id": it.eid, "cited": it.eid in ans.cited_ids, "kind": it.kind,
                                    "claim": f"{it.claim['subject']} -{it.claim['relation']}-> {it.claim['object']}" if it.claim else "",
                                    "type": it.claim["knowledge_type"] if it.claim else "", "status": it.claim["status"] if it.claim else "",
                                    "context_match": it.context_match, "paper": it.paper_id, "page": it.page,
                                    "section": it.section, "text": it.text, "synthetic": it.synthetic}
                                   for it in ans.evidence]))

elif page == "Paper Evidence":
    st.header("Paper evidence")
    if not K.papers:
        st.info("No papers yet.")
    else:
        pid = st.selectbox("Paper", sorted(K.papers), format_func=lambda p: f"{p} - {(K.papers[p].title or '')[:80]}")
        p = K.papers[pid]
        st.write({"title": p.title, "authors": p.authors, "year": p.year, "journal": p.journal, "doi": p.doi,
                  "source_file": p.source_file, "metadata_source": p.metadata_source, "synthetic": p.is_synthetic})
        claims = [c for c in K.claims if c.evidence and c.evidence.paper_id == pid]
        status = st.multiselect("Status", ["accepted", "review", "rejected"], default=["accepted", "review", "rejected"])
        show_df(claim_table([c for c in claims if c.status.value in status]))
        cid = st.text_input("Explain a claim (paste claim_id): why does the graph believe it?")
        if cid:
            c = K.claim(cid.strip())
            if c:
                ev = c.evidence
                st.markdown(f"**{c.subject.name} -[{c.relation}]-> {c.object.name}**  \n"
                            f"{c.knowledge_type.value}, {c.status.value}, confidence {c.confidence}, method {c.method}  \n"
                            f"Paper `{ev.paper_id}` -> page {ev.page} -> section *{ev.section}* -> sentence `{ev.sentence_id}` ({ev.source_type})")
                st.markdown(f"> {ev.text}")
                st.json(c.attributes)

elif page == "Queries":
    st.header("Named graph queries")
    name = st.selectbox("Query", list(QUERIES), format_func=lambda n: QUERIES[n].description)
    q = QUERIES[name]
    params = {p: st.text_input(p, key=f"q_{name}_{p}") for p in list(q.entity_params) + q.extra_params}
    with st.expander("Cypher (Neo4j)"):
        st.code(q.cypher, language="cypher")
    if st.button("Run"):
        from src.knowledge_graph.store import get_store
        store = get_store(S, K)
        try:
            show_df(pd.DataFrame(store.run_query(name, **params)))
        except Exception as exc:
            st.error(str(exc))
        store.close()

elif page == "Research Gaps":
    st.header("Potentially under-studied combinations (experimental)")
    st.warning("These are candidates, not confirmed research gaps: 'Limited evidence was found in the current corpus'.")
    excl = st.checkbox("Exclude DEMO / SYNTHETIC papers", value=False)
    from src.analysis.gaps import combination_table, find_gaps
    rep = find_gaps(K, S.gaps.min_pair_papers, S.gaps.max_context_papers, include_synthetic=not excl)
    st.json(rep.corpus, expanded=False)
    show_df(pd.DataFrame([c.__dict__ for c in rep.candidates]))
    for cv in rep.caveats:
        st.caption("• " + cv)
    st.subheader("Material x Process x Parameter x Response coverage")
    show_df(pd.DataFrame(combination_table(K, include_synthetic=not excl)))

elif page == "Export Tables":
    st.header("Extracted knowledge as tables")
    from src.reporting.tables import build_tables, write_workbook
    c1, c2, c3 = st.columns(3)
    statuses = c1.multiselect("Status", ["accepted", "review", "rejected"], default=["accepted", "review"])
    excl = c2.checkbox("Exclude DEMO / SYNTHETIC papers", value=False)
    sel = c3.multiselect("Papers (empty = all)", sorted(K.papers),
                         format_func=lambda p: f"{p} - {(K.papers[p].title or '')[:40]}")
    tables = build_tables(K, statuses, set(sel) if sel else None, include_synthetic=not excl)
    tab_objs = st.tabs([f"{n} ({len(r)})" for n, r in tables.items()])
    for tab, (_name, rows) in zip(tab_objs, tables.items()):
        with tab:
            show_df(pd.DataFrame(rows))
    out = write_workbook(tables, S.root / "data" / "exports" / "amkg_extracted_tables.xlsx")
    st.download_button("Download Excel workbook", out.read_bytes(), file_name="amkg_extracted_tables.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

elif page == "Evaluation":
    st.header("Evaluation")
    rep_path = S.root / "data" / "eval" / "evaluation_report.md"
    if st.button("Run evaluation now"):
        from src.evaluation.metrics import evaluate, graph_quality, to_markdown
        from src.normalization.entity_normalizer import EntityNormalizer, NormalizationOverrides
        from src.normalization.lexicon import Lexicon
        norm = EntityNormalizer(Lexicon.load(S.path("lexicon_file")),
                                NormalizationOverrides(S.path("review_dir") / "normalization_overrides.json"))
        try:
            r = evaluate(K, S.path("annotations_dir"), S.path("processed_dir"), norm)
            qual = graph_quality(K, S.path("processed_dir"))
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text(to_markdown(r, qual), encoding="utf-8")
        except FileNotFoundError as exc:
            st.error(str(exc))
    if rep_path.exists():
        st.markdown(rep_path.read_text(encoding="utf-8"))
    else:
        st.info("No report yet. Add gold annotations to data/annotations and run the evaluation.")
