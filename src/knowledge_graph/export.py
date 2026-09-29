"""Converts the KnowledgeBase into graph rows (nodes + aggregated edges).

Shared by the Neo4j loader, the local JSON graph store and the Streamlit
visualisation, so every backend shows exactly the same graph.
"""
from __future__ import annotations

import json
from typing import Any, Iterable

from ..ontology.models import Relation
from ..ontology.schema import EntityType
from .kb import KnowledgeBase

VALUE_TYPES = {EntityType.PARAMETER_VALUE.value, EntityType.PROPERTY_VALUE.value}


def node_kind(t: str) -> str:
    if t == EntityType.PAPER.value:
        return "Paper"
    if t in VALUE_TYPES:
        return "Value"
    return "Entity"


def claim_props(c: Relation) -> dict[str, Any]:
    ev = c.evidence
    a = c.attributes
    return {
        "claim_id": c.claim_id, "relation": c.relation,
        "subject_id": c.subject.id, "subject_name": c.subject.name, "subject_type": c.subject.type,
        "object_id": c.object.id, "object_name": c.object.name, "object_type": c.object.type,
        "subject_surface": c.subject.surface, "object_surface": c.object.surface,
        "status": c.status.value, "confidence": c.confidence, "method": c.method,
        "knowledge_type": c.knowledge_type.value, "reviewed": c.reviewed,
        "evidence": ev.text if ev else None, "paper_id": ev.paper_id if ev else None,
        "page": ev.page if ev else None, "section": ev.section if ev else None,
        "section_type": ev.section_type if ev else None, "sentence_id": ev.sentence_id if ev else None,
        "context_sentence_ids": list(ev.context_sentence_ids) if ev else [],
        "source_type": ev.source_type if ev else None, "table_id": ev.table_id if ev else None,
        "trend": a.get("trend"), "negated": a.get("negated"), "hedged": a.get("hedged"),
        "claim_type": a.get("claim_type"), "claim_origin": a.get("claim_origin"),
        "subject_change": a.get("subject_change"), "object_change": a.get("object_change"),
        "qc_reasons": list(c.qc_reasons), "attributes_json": json.dumps(a, ensure_ascii=False, default=str),
    }


def graph_rows(kb: KnowledgeBase, statuses: Iterable[str] = ("accepted", "review")) -> dict[str, list[dict]]:
    st = set(statuses)
    claims = [c for c in kb.claims if c.status.value in st]
    papers = [{"paper_id": p.paper_id, "title": p.title, "authors": p.authors, "year": p.year,
               "journal": p.journal, "doi": p.doi, "source_file": p.source_file, "abstract": p.abstract,
               "is_synthetic": p.is_synthetic, "metadata_source": p.metadata_source}
              for p in kb.papers.values()]
    used_entities = {r.id for c in claims for r in (c.subject, c.object) if node_kind(r.type) == "Entity"}
    used_entities |= {cid for cnt in kb.mention_counts.values() for cid in cnt}
    entities = []
    for cid in sorted(used_entities):
        e = kb.entities.get(cid)
        etype = e.entity_type if e else cid.split(":", 1)[0]
        entities.append({
            "id": cid, "entity_type": etype, "canonical_name": e.canonical_name if e else cid.split(":", 1)[-1],
            "display_name": e.display_name if e else None, "aliases": list(e.aliases) if e else [],
            "alias_text": " | ".join(e.aliases) if e else "", "family": e.family if e else None,
            "in_lexicon": e.in_lexicon if e else False,
            "normalization_confidence": e.normalization_confidence if e else 0.5,
            "dimensions": list((e.attributes or {}).get("dimensions") or []) if e else [],
            "paper_count": len(kb.entity_papers.get(cid, ())),
        })
    values: dict[str, dict] = {}
    for c in claims:
        if c.relation == "HAS_VALUE" and c.object.type in VALUE_TYPES:
            q = c.object.quantity
            vctx = c.attributes.get("value_context") or {}
            values[c.object.id] = {
                "id": c.object.id, "entity_type": c.object.type, "raw": c.object.name,
                "paper_id": c.evidence.paper_id if c.evidence else None, "of_entity": c.subject.id,
                "value": q.value if q else None, "value_min": q.value_min if q else None,
                "value_max": q.value_max if q else None, "uncertainty": q.uncertainty if q else None,
                "operator": q.operator if q else None, "unit": q.unit if q else None,
                "normalized_value": q.normalized_value if q else None,
                "normalized_min": q.normalized_min if q else None, "normalized_max": q.normalized_max if q else None,
                "normalized_unit": q.normalized_unit if q else None, "dimension": q.dimension if q else None,
                "context_materials": vctx.get("materials", []), "context_processes": vctx.get("processes", []),
                "state": vctx.get("state"),
            }
    # aggregate claims into one semantic edge per (subject, relation, object)
    agg: dict[tuple, dict] = {}
    for c in claims:
        key = (c.subject.id, c.relation, c.object.id)
        e = agg.setdefault(key, {
            "s": c.subject.id, "s_kind": node_kind(c.subject.type), "rel": c.relation, "o": c.object.id,
            "o_kind": node_kind(c.object.type), "claim_ids": [], "paper_ids": set(), "support_count": 0,
            "max_confidence": 0.0, "status": "review", "knowledge_types": set(), "trends": set(),
            "negated_count": 0, "hedged_count": 0})
        e["claim_ids"].append(c.claim_id)
        if c.evidence:
            e["paper_ids"].add(c.evidence.paper_id)
        e["support_count"] += 1
        e["max_confidence"] = max(e["max_confidence"], c.confidence)
        if c.status.value == "accepted":
            e["status"] = "accepted"
        e["knowledge_types"].add(c.knowledge_type.value)
        if c.attributes.get("trend"):
            e["trends"].add(c.attributes["trend"])
        e["negated_count"] += int(bool(c.attributes.get("negated")))
        e["hedged_count"] += int(bool(c.attributes.get("hedged")))
    edges = [{**e, "paper_ids": sorted(e["paper_ids"]), "knowledge_types": sorted(e["knowledge_types"]),
              "trends": sorted(e["trends"])} for e in agg.values()]
    findings = [{"finding_id": f.finding_id, "text": f.text, "paper_id": f.evidence.paper_id, "page": f.evidence.page,
                 "section": f.evidence.section, "finding_type": f.finding_type, "confidence": f.confidence,
                 "status": f.status.value, "method": f.method, "knowledge_type": f.knowledge_type.value,
                 "related_claim_ids": f.related_claim_ids}
                for f in kb.findings if f.status.value in st]
    mentions = [{"paper_id": pid, "entity_id": cid, "count": n}
                for pid, cnt in kb.mention_counts.items() for cid, n in cnt.items()]
    return {"papers": papers, "entities": entities, "values": list(values.values()),
            "claims": [claim_props(c) for c in claims], "edges": edges, "findings": findings, "mentions": mentions}


def to_networkx(rows: dict[str, list[dict]], include_values: bool = False):
    import networkx as nx
    g = nx.MultiDiGraph()
    for p in rows["papers"]:
        g.add_node(p["paper_id"], label=(p["title"] or p["paper_id"])[:40], kind="Paper", entity_type="Paper")
    for e in rows["entities"]:
        g.add_node(e["id"], label=e["canonical_name"], kind="Entity", entity_type=e["entity_type"])
    if include_values:
        for v in rows["values"]:
            g.add_node(v["id"], label=v["raw"], kind="Value", entity_type=v["entity_type"])
    for e in rows["edges"]:
        if not include_values and (e["o_kind"] == "Value" or e["s_kind"] == "Value"):
            continue
        if e["s"] in g and e["o"] in g:
            g.add_edge(e["s"], e["o"], key=e["rel"], rel=e["rel"], papers=len(e["paper_ids"]),
                       status=e["status"], support=e["support_count"], knowledge_types=e["knowledge_types"])
    return g
