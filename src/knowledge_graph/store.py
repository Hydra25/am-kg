"""Graph stores: Neo4j (primary) and a dependency-free local store (fallback).

Both expose the same interface, so the CLI, Streamlit app and research
assistant work identically whether or not Neo4j is running.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

from ..config import Settings, get_settings
from ..normalization.entity_normalizer import EntityNormalizer, NormalizationOverrides
from ..normalization.lexicon import Lexicon
from ..ontology.schema import EntityType, RelationType
from .export import graph_rows
from .kb import KnowledgeBase
from .queries import EFFECT_RELS, QUERIES

log = logging.getLogger(__name__)

ENTITY_LABELS = {t.value for t in EntityType.concept_types()}
VALUE_LABELS = {EntityType.PARAMETER_VALUE.value, EntityType.PROPERTY_VALUE.value}
REL_TYPES = {r.value for r in RelationType}


class GraphStore(ABC):
    name = "abstract"

    def __init__(self, settings: Settings, kb: KnowledgeBase) -> None:
        self.settings = settings
        self.kb = kb
        lex = Lexicon.load(settings.path("lexicon_file"))
        self.normalizer = EntityNormalizer(lex, NormalizationOverrides(
            settings.path("review_dir") / "normalization_overrides.json"))

    # -- entity resolution -------------------------------------------------
    def resolve(self, text: str, entity_type: Optional[str] = None) -> Optional[str]:
        """Free text -> canonical id present in the KB (lexicon aware)."""
        if text in self.kb.entities:
            return text
        types = [entity_type] if entity_type else [None]
        for t in types:
            hits = self.kb.find_entities(text, t)
            if hits:
                return hits[0]
        candidates = [entity_type] if entity_type else sorted(ENTITY_LABELS)
        for t in candidates:
            n = self.normalizer.normalize(text, t)
            if n.in_lexicon or n.method == "manual_override":
                cid = self.kb.resolve_id(n.canonical_id)
                if cid in self.kb.entities or not self.kb.entities:
                    return cid
        return None

    def run_query(self, name: str, statuses: Iterable[str] = ("accepted", "review"), **params: Any) -> list[dict]:
        q = QUERIES[name]
        p: dict[str, Any] = {"statuses": list(statuses), "effect_rels": EFFECT_RELS}
        for pname, ptype in q.entity_params.items():
            if pname not in params:
                raise ValueError(f"query '{name}' needs parameter '{pname}'")
            cid = self.resolve(str(params[pname]), ptype)
            if cid is None:
                return [{"error": f"'{params[pname]}' not found in the knowledge graph"
                                  + (f" as {ptype}" if ptype else "")}]
            p[f"{pname}_id"] = cid
        for extra in q.extra_params:
            p[extra] = params[extra]
        return self._execute(q, p)

    @abstractmethod
    def _execute(self, q: Any, params: dict[str, Any]) -> list[dict]: ...

    @abstractmethod
    def build(self, reset: bool = True) -> dict[str, int]: ...

    def close(self) -> None:  # pragma: no cover - overridden by Neo4j
        pass


class LocalGraphStore(GraphStore):
    """In-memory store over the KnowledgeBase; snapshot written to data/graph/graph.json."""
    name = "local"

    def _execute(self, q: Any, params: dict[str, Any]) -> list[dict]:
        return q.local(self.kb, params)

    def build(self, reset: bool = True) -> dict[str, int]:
        statuses = ["accepted", "review"] if self.settings.graph.insert_review_items else ["accepted"]
        rows = graph_rows(self.kb, statuses)
        out = self.settings.path("graph_dir") / "graph.json"
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        return {k: len(v) for k, v in rows.items()}


class Neo4jGraphStore(GraphStore):
    name = "neo4j"

    def __init__(self, settings: Settings, kb: KnowledgeBase, driver: Any = None) -> None:
        super().__init__(settings, kb)
        if driver is None:
            from neo4j import GraphDatabase
            cfg = settings.neo4j
            driver = GraphDatabase.driver(cfg.uri, auth=(cfg.username, cfg.password))
        self.driver = driver
        self.database = settings.neo4j.database or None

    def close(self) -> None:
        self.driver.close()

    def _run(self, cypher: str, **params: Any) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            return [r.data() for r in session.run(cypher, **params)]

    def _execute(self, q: Any, params: dict[str, Any]) -> list[dict]:
        return self._run(q.cypher, **params)

    # -- schema -------------------------------------------------------------
    SCHEMA = [
        "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
        "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT value_id IF NOT EXISTS FOR (v:Value) REQUIRE v.id IS UNIQUE",
        "CREATE CONSTRAINT claim_id IF NOT EXISTS FOR (c:Claim) REQUIRE c.claim_id IS UNIQUE",
        "CREATE CONSTRAINT finding_id IF NOT EXISTS FOR (f:Finding) REQUIRE f.finding_id IS UNIQUE",
        "CREATE INDEX claim_subject IF NOT EXISTS FOR (c:Claim) ON (c.subject_id)",
        "CREATE INDEX claim_object IF NOT EXISTS FOR (c:Claim) ON (c.object_id)",
        "CREATE INDEX claim_status IF NOT EXISTS FOR (c:Claim) ON (c.status)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
        "CREATE INDEX value_paper IF NOT EXISTS FOR (v:Value) ON (v.paper_id)",
        "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.canonical_name, e.alias_text]",
    ]

    def create_schema(self) -> None:
        for stmt in self.SCHEMA:
            self._run(stmt)

    def reset(self) -> None:
        self._run("MATCH (n) WHERE n:Paper OR n:Entity OR n:Value OR n:Claim OR n:Finding DETACH DELETE n")

    @staticmethod
    def _chunks(rows: list[dict], n: int = 500) -> Iterable[list[dict]]:
        for i in range(0, len(rows), n):
            yield rows[i:i + n]

    def build(self, reset: bool = True) -> dict[str, int]:
        statuses = ["accepted", "review"] if self.settings.graph.insert_review_items else ["accepted"]
        rows = graph_rows(self.kb, statuses)
        self.create_schema()
        if reset:
            self.reset()
        for batch in self._chunks(rows["papers"]):
            self._run("UNWIND $rows AS row MERGE (p:Paper {paper_id: row.paper_id}) SET p += row", rows=batch)
        by_label: dict[str, list[dict]] = {}
        for e in rows["entities"]:
            if e["entity_type"] in ENTITY_LABELS:
                by_label.setdefault(e["entity_type"], []).append(e)
        for label, ents in by_label.items():  # labels come from the ontology whitelist, never from user text
            for batch in self._chunks(ents):
                self._run(f"UNWIND $rows AS row MERGE (e:Entity {{id: row.id}}) SET e += row, e:`{label}`", rows=batch)
        vals: dict[str, list[dict]] = {}
        for v in rows["values"]:
            if v["entity_type"] in VALUE_LABELS:
                vals.setdefault(v["entity_type"], []).append(v)
        for label, vs in vals.items():
            for batch in self._chunks(vs):
                self._run(f"UNWIND $rows AS row MERGE (v:Value {{id: row.id}}) SET v += row, v:`{label}`", rows=batch)
        for batch in self._chunks(rows["claims"]):
            self._run("""UNWIND $rows AS row
MERGE (c:Claim {claim_id: row.claim_id}) SET c += row
WITH c, row MATCH (p:Paper {paper_id: row.paper_id}) MERGE (p)-[:SUPPORTS]->(c)""", rows=batch)
        # Claim -> endpoint links (SUBJECT / OBJECT) for reification
        for role, key in (("SUBJECT", "subject"), ("OBJECT", "object")):
            for kind, match in (("Entity", "(n:Entity {id: row.id})"), ("Value", "(n:Value {id: row.id})"),
                                ("Paper", "(n:Paper {paper_id: row.id})")):
                links = [{"claim_id": c["claim_id"], "id": c[f"{key}_id"]} for c in rows["claims"]
                         if _kind(c[f"{key}_type"]) == kind]
                for batch in self._chunks(links):
                    self._run(f"""UNWIND $rows AS row MATCH (c:Claim {{claim_id: row.claim_id}})
MATCH {match} MERGE (c)-[:{role}]->(n)""", rows=batch)
        # aggregated semantic edges, one per (subject, relation, object)
        groups: dict[tuple, list[dict]] = {}
        for e in rows["edges"]:
            if e["rel"] in REL_TYPES:
                groups.setdefault((e["s_kind"], e["rel"], e["o_kind"]), []).append(e)
        match = {"Entity": "(x:Entity {id: row.s})", "Value": "(x:Value {id: row.s})", "Paper": "(x:Paper {paper_id: row.s})"}
        omatch = {"Entity": "(y:Entity {id: row.o})", "Value": "(y:Value {id: row.o})", "Paper": "(y:Paper {paper_id: row.o})"}
        for (sk, rel, ok), es in groups.items():
            props = [{**{k: v for k, v in e.items() if k not in ("s", "o", "s_kind", "o_kind", "rel")}, "s": e["s"], "o": e["o"]}
                     for e in es]
            for batch in self._chunks(props):
                self._run(f"""UNWIND $rows AS row MATCH {match[sk]} MATCH {omatch[ok]}
MERGE (x)-[r:`{rel}`]->(y)
SET r.claim_ids = row.claim_ids, r.paper_ids = row.paper_ids, r.support_count = row.support_count,
    r.max_confidence = row.max_confidence, r.status = row.status, r.knowledge_types = row.knowledge_types,
    r.trends = row.trends, r.negated_count = row.negated_count, r.hedged_count = row.hedged_count""", rows=batch)
        for batch in self._chunks(rows["findings"]):
            self._run("""UNWIND $rows AS row MERGE (f:Finding {finding_id: row.finding_id}) SET f += row
WITH f, row MATCH (p:Paper {paper_id: row.paper_id}) MERGE (p)-[:REPORTS]->(f)
WITH f, row UNWIND row.related_claim_ids AS cid MATCH (c:Claim {claim_id: cid}) MERGE (f)-[:EXPRESSES]->(c)""",
                      rows=batch)
        for batch in self._chunks(rows["mentions"], 2000):
            self._run("""UNWIND $rows AS row MATCH (p:Paper {paper_id: row.paper_id}) MATCH (e:Entity {id: row.entity_id})
MERGE (p)-[m:MENTIONS]->(e) SET m.count = row.count""", rows=batch)
        return {k: len(v) for k, v in rows.items()}

    def set_claim_status(self, claim_id: str, status: str) -> None:
        self._run("MATCH (c:Claim {claim_id: $cid}) SET c.status = $status, c.reviewed = true", cid=claim_id, status=status)


def _kind(t: str) -> str:
    if t == "Paper":
        return "Paper"
    if t in VALUE_LABELS:
        return "Value"
    return "Entity"


def neo4j_available(settings: Settings) -> tuple[bool, str]:
    if not settings.neo4j.enabled:
        return False, "NEO4J_URI not set"
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(settings.neo4j.uri, auth=(settings.neo4j.username, settings.neo4j.password))
        drv.verify_connectivity()
        drv.close()
        return True, "ok"
    except Exception as exc:  # pragma: no cover - needs a server
        return False, str(exc)


def get_store(settings: Optional[Settings] = None, kb: Optional[KnowledgeBase] = None,
              backend: Optional[str] = None) -> GraphStore:
    s = settings or get_settings()
    kb = kb or KnowledgeBase.load(s)
    choice = (backend or s.graph.backend).lower()
    if choice == "neo4j" or (choice == "auto" and s.neo4j.enabled):
        ok, why = neo4j_available(s)
        if ok:
            return Neo4jGraphStore(s, kb)
        if choice == "neo4j":
            raise RuntimeError(f"Neo4j not reachable: {why}")
        log.warning("Neo4j not reachable (%s); using the local graph store", why)
    return LocalGraphStore(s, kb)
