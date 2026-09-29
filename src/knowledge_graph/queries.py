"""Named graph queries: one Cypher version (Neo4j) and one Python version
(local store) per query, returning the same row format.

Entity parameters are given as free text ("Ti6Al4V", "SLM", "scan speed") and
resolved to canonical ids with the normaliser before the query runs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..ontology.models import Relation
from .kb import KnowledgeBase

EFFECT_RELS = ["AFFECTS", "INFLUENCES", "CAUSES", "IMPROVES", "CHANGES", "PRODUCES"]


def _ev_row(c: Relation, kb: KnowledgeBase) -> dict[str, Any]:
    ev = c.evidence
    p = kb.papers.get(ev.paper_id) if ev else None
    return {
        "paper_id": ev.paper_id if ev else None, "title": p.title if p else None, "year": p.year if p else None,
        "relation": c.relation, "subject": c.subject.name, "object": c.object.name,
        "trend": c.attributes.get("trend"), "negated": c.attributes.get("negated"),
        "hedged": c.attributes.get("hedged"), "knowledge_type": c.knowledge_type.value,
        "evidence": ev.text if ev else None, "page": ev.page if ev else None,
        "section": ev.section if ev else None, "confidence": c.confidence, "status": c.status.value,
        "claim_id": c.claim_id, "synthetic": bool(p and p.is_synthetic),
    }


# ---------------------------------------------------------------------------
# local implementations
# ---------------------------------------------------------------------------
def _local_processes_for_material(kb: KnowledgeBase, p: dict) -> list[dict]:
    agg: dict[str, dict] = {}
    for c in kb.active_claims(p["statuses"]):
        if c.relation == "PROCESSES" and c.object.id == p["material_id"]:
            row = agg.setdefault(c.subject.id, {"process": c.subject.name, "papers": set(), "claims": 0, "status": "review"})
            row["papers"].add(c.evidence.paper_id if c.evidence else None)
            row["claims"] += 1
            if c.status.value == "accepted":
                row["status"] = "accepted"
    rows = [{**r, "papers": len(r["papers"])} for r in agg.values()]
    return sorted(rows, key=lambda r: (-r["papers"], -r["claims"]))


def _local_parameters_for_process(kb: KnowledgeBase, p: dict) -> list[dict]:
    agg: dict[str, dict] = {}
    values: dict[str, list[str]] = defaultdict(list)
    for c in kb.active_claims(p["statuses"]):
        if c.relation == "HAS_VALUE":
            values[c.subject.id].append(c.object.name)
    for c in kb.active_claims(p["statuses"]):
        if c.relation == "HAS_PARAMETER" and c.subject.id == p["process_id"]:
            row = agg.setdefault(c.object.id, {"parameter": c.object.name, "papers": set(), "claims": 0})
            row["papers"].add(c.evidence.paper_id if c.evidence else None)
            row["claims"] += 1
    rows = [{"parameter": r["parameter"], "papers": len(r["papers"]), "claims": r["claims"],
             "example_values": sorted(set(values.get(cid, [])))[:10]} for cid, r in agg.items()]
    return sorted(rows, key=lambda r: (-r["papers"], -r["claims"]))


def _local_studies_linking(kb: KnowledgeBase, p: dict) -> list[dict]:
    a, b = p["a_id"], p["b_id"]
    rows = [_ev_row(c, kb) for c in kb.active_claims(p["statuses"])
            if {c.subject.id, c.object.id} == {a, b}]
    return sorted(rows, key=lambda r: (-(r["year"] or 0), r["paper_id"] or ""))


def _local_property_values(kb: KnowledgeBase, p: dict) -> list[dict]:
    papers = {c.subject.id for c in kb.active_claims(p["statuses"])
              if c.relation == "STUDIES" and c.object.id == p["material_id"]}
    rows = []
    for c in kb.active_claims(p["statuses"]):
        if c.relation == "HAS_VALUE" and c.subject.id == p["property_id"] and c.evidence and c.evidence.paper_id in papers:
            q = c.object.quantity
            ctx = kb.paper_context(c.evidence.paper_id)
            vctx = c.attributes.get("value_context") or {}
            rows.append({**_ev_row(c, kb), "value": c.object.name,
                         "normalized_value": q.normalized_value if q else None,
                         "normalized_min": q.normalized_min if q else None,
                         "normalized_max": q.normalized_max if q else None,
                         "normalized_unit": q.normalized_unit if q else None,
                         "material_state": vctx.get("state"),
                         "context_ambiguous": len(ctx["materials"]) > 1})
    return rows


def _local_papers_material_process(kb: KnowledgeBase, p: dict) -> list[dict]:
    rows = []
    for pid, paper in kb.papers.items():
        ctx = kb.paper_context(pid, p["statuses"])
        if p["material_id"] in ctx["materials"] and p["process_id"] in ctx["processes"]:
            rows.append({"paper_id": pid, "title": paper.title, "year": paper.year, "doi": paper.doi,
                         "synthetic": paper.is_synthetic})
    return rows


def _local_relations_between(kb: KnowledgeBase, p: dict) -> list[dict]:
    a, b = p["a_id"], p["b_id"]
    active = [c for c in kb.active_claims(p["statuses"]) if c.relation in EFFECT_RELS]
    rows = [{**_ev_row(c, kb), "path_type": "direct claim"} for c in active if {c.subject.id, c.object.id} == {a, b}]
    # two-hop paths a - x - b (DERIVED: a path in the graph is not a claim made by any paper)
    nbr: dict[str, dict[str, list[Relation]]] = defaultdict(lambda: defaultdict(list))
    for c in active:
        nbr[c.subject.id][c.object.id].append(c)
        nbr[c.object.id][c.subject.id].append(c)
    for x, cs1 in nbr.get(a, {}).items():
        if x == b:
            continue
        cs2 = nbr.get(x, {}).get(b)
        if cs2:
            rows.append({"path_type": "two-hop path (DERIVED, not stated by a paper)",
                         "via": kb.entity_name(x), "rel1": sorted({c.relation for c in cs1}),
                         "rel2": sorted({c.relation for c in cs2}),
                         "papers1": sorted({c.evidence.paper_id for c in cs1 if c.evidence}),
                         "papers2": sorted({c.evidence.paper_id for c in cs2 if c.evidence})})
    return rows


def _local_papers_reporting_both(kb: KnowledgeBase, p: dict) -> list[dict]:
    rows = []
    for pid, cnt in kb.mention_counts.items():
        if cnt.get(p["a_id"]) and cnt.get(p["b_id"]):
            paper = kb.papers[pid]
            linked = [c for c in kb.active_claims(p["statuses"])
                      if c.evidence and c.evidence.paper_id == pid and {c.subject.id, c.object.id} == {p["a_id"], p["b_id"]}]
            rows.append({"paper_id": pid, "title": paper.title, "year": paper.year,
                         "mentions_a": cnt[p["a_id"]], "mentions_b": cnt[p["b_id"]],
                         "direct_claims": len(linked), "synthetic": paper.is_synthetic})
    return sorted(rows, key=lambda r: -r["direct_claims"])


def _local_understudied(kb: KnowledgeBase, p: dict) -> list[dict]:
    from ..analysis.gaps import combination_table
    return combination_table(kb, p["statuses"])


def _local_why(kb: KnowledgeBase, p: dict) -> list[dict]:
    c = kb.claim(p["claim_id"])
    if not c:
        return []
    row = _ev_row(c, kb)
    row.update({"method": c.method, "qc_reasons": c.qc_reasons, "reviewed": c.reviewed,
                "source_type": c.evidence.source_type if c.evidence else None,
                "sentence_id": c.evidence.sentence_id if c.evidence else None,
                "context_sentence_ids": c.evidence.context_sentence_ids if c.evidence else [],
                "attributes": c.attributes})
    return [row]


def _local_neighbors(kb: KnowledgeBase, p: dict) -> list[dict]:
    agg: dict[tuple, dict] = {}
    for c in kb.active_claims(p["statuses"]):
        if p["entity_id"] not in (c.subject.id, c.object.id) or c.relation in ("STUDIES", "USES_PROCESS"):
            continue
        out = c.subject.id == p["entity_id"]
        other = c.object if out else c.subject
        key = (c.relation, out, other.id)
        row = agg.setdefault(key, {"relation": c.relation, "direction": "out" if out else "in",
                                   "neighbor": other.name, "neighbor_type": other.type, "neighbor_id": other.id,
                                   "claims": 0, "papers": set()})
        row["claims"] += 1
        if c.evidence:
            row["papers"].add(c.evidence.paper_id)
    rows = [{**r, "papers": len(r["papers"])} for r in agg.values()]
    return sorted(rows, key=lambda r: (r["relation"], -r["papers"]))


# ---------------------------------------------------------------------------
@dataclass
class NamedQuery:
    name: str
    description: str
    entity_params: dict[str, Optional[str]]   # param -> required entity type (None = any)
    cypher: str
    local: Callable[[KnowledgeBase, dict], list[dict]]
    extra_params: list[str] = field(default_factory=list)


QUERIES: dict[str, NamedQuery] = {q.name: q for q in [
    NamedQuery(
        "processes_for_material", "1. All AM processes used for a material",
        {"material": "Material"},
        """MATCH (pr:Process)-[r:PROCESSES]->(m:Material {id: $material_id})
WHERE r.status IN $statuses
RETURN pr.canonical_name AS process, size(r.paper_ids) AS papers, r.support_count AS claims, r.status AS status
ORDER BY papers DESC, claims DESC""",
        _local_processes_for_material),
    NamedQuery(
        "parameters_for_process", "2. All parameters reported for a process (with example values)",
        {"process": "Process"},
        """MATCH (pr:Process {id: $process_id})-[r:HAS_PARAMETER]->(pa:Parameter)
WHERE r.status IN $statuses
OPTIONAL MATCH (pa)-[hv:HAS_VALUE]->(v:ParameterValue)
WHERE v.paper_id IN r.paper_ids AND hv.status IN $statuses
RETURN pa.canonical_name AS parameter, size(r.paper_ids) AS papers, r.support_count AS claims,
       collect(DISTINCT v.raw)[..10] AS example_values
ORDER BY papers DESC, claims DESC""",
        _local_parameters_for_process),
    NamedQuery(
        "studies_linking", "3. Studies with an evidence-backed claim connecting two entities (e.g. laser power & porosity)",
        {"a": None, "b": None},
        """MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.status IN $statuses
  AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, c.relation AS relation,
       c.subject_name AS subject, c.object_name AS object, c.trend AS trend, c.negated AS negated,
       c.hedged AS hedged, c.knowledge_type AS knowledge_type, c.evidence AS evidence, c.page AS page,
       c.section AS section, c.confidence AS confidence, c.status AS status, c.claim_id AS claim_id,
       p.is_synthetic AS synthetic
ORDER BY year DESC""",
        _local_studies_linking),
    NamedQuery(
        "property_values_for_material", "4. Reported values of a property (e.g. tensile strength) for a material",
        {"material": "Material", "property": None},
        """MATCH (p:Paper)-[s:STUDIES]->(m:Material {id: $material_id})
WHERE s.status IN $statuses
MATCH (p)-[:SUPPORTS]->(c:Claim {relation: 'HAS_VALUE', subject_id: $property_id})
WHERE c.status IN $statuses
MATCH (v:Value {id: c.object_id})
OPTIONAL MATCH (p)-[s2:STUDIES]->(m2:Material) WHERE s2.status IN $statuses
WITH p, c, v, count(DISTINCT m2) AS n_materials
RETURN p.paper_id AS paper_id, p.title AS title, v.raw AS value, v.normalized_value AS normalized_value,
       v.normalized_min AS normalized_min, v.normalized_max AS normalized_max, v.normalized_unit AS normalized_unit,
       v.state AS material_state, c.evidence AS evidence, c.page AS page, c.section AS section,
       c.status AS status, c.claim_id AS claim_id, n_materials > 1 AS context_ambiguous""",
        _local_property_values),
    NamedQuery(
        "papers_with_material_and_process", "5. Papers using both a material and a process",
        {"material": "Material", "process": "Process"},
        """MATCH (p:Paper)-[s:STUDIES]->(:Material {id: $material_id}), (p)-[u:USES_PROCESS]->(:Process {id: $process_id})
WHERE s.status IN $statuses AND u.status IN $statuses
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, p.doi AS doi, p.is_synthetic AS synthetic""",
        _local_papers_material_process),
    NamedQuery(
        "relations_between", "6. Direct claims and two-hop paths between two entities (e.g. scan speed & tensile strength)",
        {"a": None, "b": None},
        """CALL {
  MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
  WHERE c.status IN $statuses AND c.relation IN $effect_rels
    AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
  RETURN 'direct claim' AS path_type, null AS via, c.relation AS rel1, null AS rel2,
         [p.paper_id] AS papers1, null AS papers2, c.evidence AS evidence
  UNION
  MATCH (a:Entity {id: $a_id})-[r1]-(x:Entity)-[r2]-(b:Entity {id: $b_id})
  WHERE type(r1) IN $effect_rels AND type(r2) IN $effect_rels
    AND r1.status IN $statuses AND r2.status IN $statuses AND x <> a AND x <> b
  RETURN 'two-hop path (DERIVED, not stated by a paper)' AS path_type, x.canonical_name AS via,
         type(r1) AS rel1, type(r2) AS rel2, r1.paper_ids AS papers1, r2.paper_ids AS papers2, null AS evidence
}
RETURN path_type, via, rel1, rel2, papers1, papers2, evidence""",
        _local_relations_between),
    NamedQuery(
        "papers_reporting_both", "7. Papers mentioning two entities together (e.g. porosity and laser power)",
        {"a": None, "b": None},
        """MATCH (p:Paper)-[ma:MENTIONS]->(:Entity {id: $a_id}), (p)-[mb:MENTIONS]->(:Entity {id: $b_id})
OPTIONAL MATCH (p)-[:SUPPORTS]->(c:Claim)
WHERE c.status IN $statuses
  AND ((c.subject_id = $a_id AND c.object_id = $b_id) OR (c.subject_id = $b_id AND c.object_id = $a_id))
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year, ma.count AS mentions_a, mb.count AS mentions_b,
       count(c) AS direct_claims, p.is_synthetic AS synthetic
ORDER BY direct_claims DESC""",
        _local_papers_reporting_both),
    NamedQuery(
        "understudied_combinations", "8. Material x Process x Parameter x Property/Defect combinations with paper counts",
        {},
        """MATCH (p:Paper)-[s:STUDIES]->(m:Material), (p)-[u:USES_PROCESS]->(pr:Process), (p)-[:SUPPORTS]->(c:Claim)
WHERE s.status IN $statuses AND u.status IN $statuses AND c.status IN $statuses
  AND c.relation IN ['AFFECTS', 'INFLUENCES'] AND c.subject_type = 'Parameter'
WITH m.canonical_name AS material, pr.canonical_name AS process, c.subject_name AS parameter,
     c.object_name AS response, collect(DISTINCT p.paper_id) AS papers
RETURN material, process, parameter, response, size(papers) AS n_papers, papers
ORDER BY n_papers ASC, material, process""",
        _local_understudied),
    NamedQuery(
        "why_claim", "Provenance: why does the graph believe this claim?",
        {}, """MATCH (p:Paper)-[:SUPPORTS]->(c:Claim {claim_id: $claim_id})
RETURN p.paper_id AS paper_id, p.title AS title, c.relation AS relation, c.subject_name AS subject,
       c.object_name AS object, c.evidence AS evidence, c.page AS page, c.section AS section,
       c.sentence_id AS sentence_id, c.source_type AS source_type, c.method AS method,
       c.knowledge_type AS knowledge_type, c.confidence AS confidence, c.status AS status,
       c.qc_reasons AS qc_reasons, c.reviewed AS reviewed""",
        _local_why, extra_params=["claim_id"]),
    NamedQuery(
        "entity_neighbors", "Graph explorer: all relations of one entity",
        {"entity": None},
        """MATCH (e:Entity {id: $entity_id})-[r]-(n)
WHERE NOT type(r) IN ['MENTIONS', 'SUBJECT', 'OBJECT', 'STUDIES', 'USES_PROCESS'] AND r.status IN $statuses
RETURN type(r) AS relation, CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END AS direction,
       coalesce(n.canonical_name, n.raw) AS neighbor, n.entity_type AS neighbor_type, n.id AS neighbor_id,
       r.support_count AS claims, size(r.paper_ids) AS papers
ORDER BY relation, papers DESC""",
        _local_neighbors),
]}
