"""End-to-end: demo PDFs -> extraction -> KB -> local store queries, review, Neo4j loader (fake driver)."""
import json

from src.knowledge_graph.kb import KnowledgeBase
from src.knowledge_graph.store import LocalGraphStore, Neo4jGraphStore
from src.pipeline import load_all_results
from src.review.store import ReviewDecision, ReviewStore


def test_pipeline_outputs(demo_corpus):
    s = demo_corpus
    results = load_all_results(s.path("extracted_dir"))
    assert len(results) == 5
    r1 = [r for r in results if r.paper.source_file == "demo_01_lpbf_ti64.pdf"][0]
    assert r1.paper.is_synthetic and r1.paper.doi == "10.0000/demo.synthetic.001" and r1.paper.year == 2025
    assert r1.stats.accepted > 20 and r1.provenance.extraction_mode == "rule"
    # every non-rejected claim has verifiable evidence with page + section
    for r in results:
        for c in r.relations:
            if c.status.value != "rejected":
                assert c.evidence and c.evidence.text and c.evidence.page and c.evidence.section
    # table rows are extracted, not discarded
    assert any(c.evidence.source_type == "table" for c in r1.relations if c.evidence)
    # the derived table trend is labelled DERIVED and not auto-accepted
    derived = [c for c in r1.relations if c.method == "table_trend"]
    assert derived and all(c.knowledge_type.value == "DERIVED" and c.status.value == "review" for c in derived)
    assert (s.path("processed_dir") / f"{r1.paper.paper_id}.json").exists()


def test_local_queries(demo_corpus):
    kb = KnowledgeBase.load(demo_corpus)
    store = LocalGraphStore(demo_corpus, kb)
    counts = store.build()
    assert counts["papers"] == 5 and (demo_corpus.path("graph_dir") / "graph.json").exists()
    rows = store.run_query("processes_for_material", material="Ti6Al4V")
    assert rows[0]["process"] == "LPBF" and rows[0]["papers"] == 2
    rows = store.run_query("parameters_for_process", process="SLM")
    assert "laser power" in {r["parameter"] for r in rows}
    rows = store.run_query("studies_linking", a="laser power", b="porosity")
    assert len({r["paper_id"] for r in rows}) == 2 and all(r["evidence"] for r in rows)
    rows = store.run_query("property_values_for_material", material="Ti-6Al-4V", property="tensile strength")
    assert rows and rows[0]["normalized_value"] == 950
    rows = store.run_query("papers_with_material_and_process", material="Ti64", process="laser powder bed fusion")
    assert len(rows) == 2
    assert store.run_query("processes_for_material", material="unobtainium")[0].get("error")
    cid = store.run_query("studies_linking", a="laser power", b="porosity")[0]["claim_id"]
    why = store.run_query("why_claim", claim_id=cid)[0]
    assert why["evidence"] and why["page"] and why["section"]


def test_review_decisions_apply(demo_corpus):
    kb = KnowledgeBase.load(demo_corpus)
    target = kb.review_queue()[0]
    rs = ReviewStore(demo_corpus.path("review_dir"))
    rs.add(ReviewDecision("claim", target.claim_id, "accept", note="checked"))
    acc = [c for c in kb.claims if c.status.value == "accepted"][0]
    rs.add(ReviewDecision("claim", acc.claim_id, "reject"))
    rs.add(ReviewDecision("entity", "Microstructure:α' martensite", "merge", {"into": "Microstructure:microstructure"}))
    kb2 = KnowledgeBase.load(demo_corpus)
    assert kb2.claim(target.claim_id).status.value == "accepted" and kb2.claim(target.claim_id).reviewed
    assert kb2.claim(acc.claim_id).status.value == "rejected"
    assert "Microstructure:α' martensite" not in kb2.entities
    assert all(c.object.id != "Microstructure:α' martensite" for c in kb2.claims)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter([type("R", (), {"data": lambda self, r=r: r})() for r in self.rows])


class FakeSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def run(self, cypher, **params):
        self.log.append((cypher, params))
        return FakeResult([{"process": "LPBF", "papers": 2}] if "RETURN" in cypher and "UNWIND" not in cypher else [])


class FakeDriver:
    def __init__(self):
        self.log = []

    def session(self, database=None):
        return FakeSession(self.log)

    def close(self):
        pass


def test_neo4j_loader_cypher(demo_corpus):
    kb = KnowledgeBase.load(demo_corpus)
    drv = FakeDriver()
    store = Neo4jGraphStore(demo_corpus, kb, driver=drv)
    store.build()
    cy = "\n".join(c for c, _ in drv.log)
    assert "CREATE CONSTRAINT entity_id" in cy and "MERGE (c:Claim {claim_id: row.claim_id})" in cy
    assert "e:`Material`" in cy and "MERGE (x)-[r:`INFLUENCES`]->(y)" in cy and "[:SUPPORTS]" in cy
    # all parameters sent to Neo4j must be JSON-serialisable primitives / lists
    for _, params in drv.log:
        json.dumps(params, default=lambda o: (_ for _ in ()).throw(TypeError(str(type(o)))))
    claims_rows = [p["rows"] for c, p in drv.log if "MERGE (c:Claim" in c][0]
    assert all(isinstance(v, (str, int, float, bool, list, type(None))) for row in claims_rows for v in row.values())
    rows = store.run_query("processes_for_material", material="Ti6Al4V")
    assert rows == [{"process": "LPBF", "papers": 2}]
    assert drv.log[-1][1]["material_id"] == "Material:Ti-6Al-4V"      # resolved before the Cypher runs
