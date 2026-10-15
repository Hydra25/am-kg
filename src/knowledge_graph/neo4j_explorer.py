"""Read-only Neo4j explorer used by the Streamlit "Neo4j Explorer" page.

* `DEMO_QUERIES`: ready-made Cypher for a demo (graph or table views, some with a picker).
* `run_read`: runs Cypher in a READ transaction, so the explorer can never change the graph.
* `to_graph` / `to_table`: turn driver records (nodes, relationships, paths, scalars) into plain
  dicts for pyvis and pandas. Duck-typed, so tests can use simple fake objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

EFFECT_RELS = ["AFFECTS", "INFLUENCES", "CAUSES", "IMPROVES", "CHANGES"]

# --------------------------------------------------------------------------------------------- queries
# Pickers fill parameters before the query runs:
#   target  -> $target_id   (an entity with incoming effect edges)
#   pair    -> $s_id, $o_id (an existing effect edge)
#   quantity-> $q_id        (a Parameter/Property with values)
#   material-> $m_id        (a Material studied by a paper)
# $statuses is always set (accepted, or accepted + review).


@dataclass
class DemoQuery:
    name: str
    description: str
    cypher: str
    view: str = "graph"                 # graph | table
    picker: Optional[str] = None        # None | target | pair | quantity | material
    params: dict[str, Any] = field(default_factory=dict)


DEMO_QUERIES: list[DemoQuery] = [
    DemoQuery("Overview: what is in the graph", "Node counts per label.",
              "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS nodes ORDER BY nodes DESC", "table"),
    DemoQuery("Papers, alloys and processes", "Each paper with the materials it studies and the AM process it uses.",
              """MATCH (p:Paper)-[r:STUDIES|USES_PROCESS]->(e:Entity)
WHERE r.status IN $statuses
RETURN p, r, e"""),
    DemoQuery("What affects …?", "Everything with an AFFECTS / INFLUENCES / CAUSES / IMPROVES / CHANGES edge into the chosen entity.",
              """MATCH (x:Entity)-[r]->(t:Entity {id: $target_id})
WHERE type(r) IN $effect_rels AND r.status IN $statuses
RETURN x, r, t""", picker="target"),
    DemoQuery("Evidence for a relation", "The sentences, pages and confidence behind one relation (why does the graph believe it?).",
              """MATCH (p:Paper)-[:SUPPORTS]->(c:Claim)
WHERE c.subject_id = $s_id AND c.object_id = $o_id AND c.status IN $statuses
RETURN p.source_file AS paper, c.page AS page, c.section AS section, c.relation AS relation,
       c.evidence AS evidence_sentence, round(c.confidence * 100) / 100 AS confidence, c.status AS status,
       c.method AS method
ORDER BY confidence DESC""", "table", picker="pair"),
    DemoQuery("Evidence graph for a relation", "Paper → Claim → subject / object: the evidence stored inside the graph.",
              """MATCH path = (p:Paper)-[:SUPPORTS]->(c:Claim)-[:SUBJECT|OBJECT]->(e)
WHERE c.subject_id = $s_id AND c.object_id = $o_id AND c.status IN $statuses
RETURN path""", picker="pair"),
    DemoQuery("Effects confirmed by ≥ 2 papers", "Process–parameter–property map: only effects reported in at least two papers.",
              """MATCH (a:Entity)-[r]->(b:Entity)
WHERE type(r) IN $effect_rels AND r.status IN $statuses AND size(r.paper_ids) >= 2
RETURN a, r, b"""),
    DemoQuery("Reported values", "All values of a parameter or property, normalised to one unit.",
              """MATCH (q:Entity {id: $q_id})-[h:HAS_VALUE]->(v:Value)
WHERE h.status IN $statuses
MATCH (p:Paper {paper_id: v.paper_id})
RETURN p.source_file AS paper, v.raw AS as_written, v.normalized_value AS value,
       v.normalized_min AS min, v.normalized_max AS max, v.normalized_unit AS unit
ORDER BY coalesce(v.normalized_value, v.normalized_min)""", "table", picker="quantity"),
    DemoQuery("Alloy neighbourhood", "Papers on the chosen alloy and the effects they report.",
              """MATCH (p:Paper)-[s:STUDIES]->(m:Material {id: $m_id}) WHERE s.status IN $statuses
MATCH (p)-[:MENTIONS]->(a:Entity)-[r]->(b:Entity)
WHERE type(r) IN $effect_rels AND r.status IN $statuses AND p.paper_id IN r.paper_ids
RETURN m, s, p, a, r, b LIMIT 200""", picker="material"),
    DemoQuery("Research gaps (single-paper effects)", "Effects reported by only one paper: weak evidence, candidates for new experiments.",
              """MATCH (a:Entity)-[r]->(b:Entity)
WHERE type(r) IN $effect_rels AND r.status IN $statuses AND size(r.paper_ids) = 1
RETURN a.canonical_name AS cause, type(r) AS relation, b.canonical_name AS effect,
       round(r.max_confidence * 100) / 100 AS confidence, r.paper_ids[0] AS paper
ORDER BY confidence DESC LIMIT 50""", "table"),
]

PICKER_QUERIES = {
    "target": """MATCH (x:Entity)-[r]->(t:Entity) WHERE type(r) IN $effect_rels AND r.status IN $statuses
RETURN t.id AS id, t.canonical_name AS name, t.entity_type AS type, count(DISTINCT x) AS n
ORDER BY n DESC, name LIMIT 200""",
    "pair": """MATCH (a:Entity)-[r]->(b:Entity) WHERE type(r) IN $effect_rels AND r.status IN $statuses
RETURN a.id AS s_id, b.id AS o_id, a.canonical_name AS s, type(r) AS rel, b.canonical_name AS o,
       size(r.paper_ids) AS papers
ORDER BY papers DESC, s LIMIT 300""",
    "quantity": """MATCH (q:Entity)-[h:HAS_VALUE]->(:Value) WHERE h.status IN $statuses
RETURN q.id AS id, q.canonical_name AS name, q.entity_type AS type, count(*) AS n
ORDER BY n DESC, name LIMIT 200""",
    "material": """MATCH (:Paper)-[s:STUDIES]->(m:Material) WHERE s.status IN $statuses
RETURN m.id AS id, m.canonical_name AS name, count(*) AS n ORDER BY n DESC, name LIMIT 100""",
}


# --------------------------------------------------------------------------------------------- running
def run_read(driver: Any, cypher: str, params: Optional[dict] = None, database: Optional[str] = None,
             max_rows: int = 2000) -> tuple[list[str], list[list[Any]]]:
    """Run Cypher in a READ transaction (writes are rejected by the server) -> (keys, rows)."""
    params = {"statuses": ["accepted"], "effect_rels": EFFECT_RELS, **(params or {})}

    def work(tx: Any) -> tuple[list[str], list[list[Any]]]:
        res = tx.run(cypher, **params)
        keys = list(res.keys())
        rows = []
        for i, rec in enumerate(res):
            if i >= max_rows:
                break
            rows.append(list(rec.values()))
        return keys, rows

    with driver.session(database=database) as session:
        return session.execute_read(work)


# --------------------------------------------------------------------------------------------- converting
def _is_path(v: Any) -> bool:
    return hasattr(v, "relationships") and hasattr(v, "nodes") and not hasattr(v, "labels")


def _is_rel(v: Any) -> bool:
    return hasattr(v, "start_node") and hasattr(v, "end_node") and hasattr(v, "type")


def _is_node(v: Any) -> bool:
    return hasattr(v, "labels") and hasattr(v, "element_id")


def node_type(labels: Any) -> str:
    labels = set(labels)
    for name in ("Paper", "Claim", "Finding"):
        if name in labels:
            return name
    specific = sorted(labels - {"Entity", "Value"})
    return specific[0] if specific else (next(iter(labels)) if labels else "Node")


def node_caption(labels: Any, props: dict) -> str:
    t = node_type(labels)
    if t == "Paper":
        return str(props.get("source_file") or props.get("title") or props.get("paper_id", "paper"))
    if t == "Claim":
        return str(props.get("relation", "claim"))
    if t in ("ParameterValue", "PropertyValue"):
        return str(props.get("raw", "value"))
    if t == "Finding":
        return str(props.get("finding_type") or "finding")
    return str(props.get("canonical_name") or props.get("id") or t)


def _short(v: Any, n: int = 300) -> str:
    s = ", ".join(map(str, v[:8])) + (" …" if len(v) > 8 else "") if isinstance(v, list) else str(v)
    return s if len(s) <= n else s[:n] + " …"


def node_tooltip(labels: Any, props: dict) -> str:
    t = node_type(labels)
    keys = {"Paper": ["title", "year", "doi", "source_file"],
            "Claim": ["relation", "subject_name", "object_name", "status", "confidence", "page", "section", "evidence"],
            "Finding": ["text", "page", "section", "status"]}.get(
        t, ["canonical_name", "entity_type", "aliases", "raw", "normalized_value", "normalized_unit", "paper_count"])
    lines = [t] + [f"{k}: {_short(props[k])}" for k in keys if props.get(k) not in (None, "", [])]
    return "\n".join(lines)


def to_graph(rows: list[list[Any]]) -> dict[str, Any]:
    """All nodes / relationships found in the result values (also inside paths and lists)."""
    nodes: dict[str, dict] = {}
    edges: dict[str, dict] = {}

    def add_node(n: Any) -> None:
        if n.element_id in nodes:
            return
        props = dict(n.items())
        labels = list(n.labels)
        nodes[n.element_id] = {"id": n.element_id, "type": node_type(labels), "label": node_caption(labels, props),
                               "title": node_tooltip(labels, props)}

    def add_rel(r: Any) -> None:
        add_node(r.start_node)
        add_node(r.end_node)
        props = dict(r.items())
        papers = props.get("paper_ids") or []
        bits = [r.type]
        if papers:
            bits.append(f"papers: {len(papers)}")
        for k in ("support_count", "status", "max_confidence", "trends"):
            if props.get(k) not in (None, "", []):
                bits.append(f"{k}: {_short(props[k], 80)}")
        edges[r.element_id] = {"id": r.element_id, "from": r.start_node.element_id, "to": r.end_node.element_id,
                               "label": r.type, "title": "\n".join(bits), "papers": len(papers),
                               "status": props.get("status")}

    def walk(v: Any) -> None:
        if _is_path(v):
            for n in v.nodes:
                add_node(n)
            for r in v.relationships:
                add_rel(r)
        elif _is_rel(v):
            add_rel(v)
        elif _is_node(v):
            add_node(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)

    for row in rows:
        for v in row:
            walk(v)
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _cell(v: Any) -> Any:
    if _is_path(v):
        return " → ".join(node_caption(n.labels, dict(n.items())) for n in v.nodes)
    if _is_rel(v):
        return v.type
    if _is_node(v):
        return node_caption(v.labels, dict(v.items()))
    if isinstance(v, list):
        return ", ".join(str(_cell(x)) for x in v)
    return v


def to_table(keys: list[str], rows: list[list[Any]]) -> list[dict]:
    return [{k: _cell(v) for k, v in zip(keys, row)} for row in rows]
