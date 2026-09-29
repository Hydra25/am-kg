"""DERIVED relations: computed by the system, never stated verbatim in one sentence.

1. Paper-level STUDIES / USES_PROCESS from where materials/processes are mentioned.
2. Monotonic trends computed from table rows (e.g. laser power vs. relative
   density across rows where all other parameters are constant).

These are stored with knowledge_type=DERIVED so they are never confused with
EXPLICIT statements from the paper.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from ..entity_extraction.matcher import LexiconMatcher
from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.units import unit_from_header
from ..ontology.models import EntityMention, EntityRef, Evidence, ParsedDocument, Relation, stable_id
from ..ontology.schema import EntityType, KnowledgeType, RelationType, infer_effect_relation

E = EntityType
_SECTION_WEIGHT = {"abstract": 0.95, "materials_methods": 0.93, "results": 0.9, "results_discussion": 0.9,
                   "discussion": 0.85, "conclusion": 0.9, "other": 0.8, "front_matter": 0.8,
                   "introduction": 0.55}
_PRIORITY = ["abstract", "materials_methods", "results", "results_discussion", "conclusion", "discussion",
             "other", "front_matter", "introduction"]


def paper_level_relations(doc: ParsedDocument, mentions: list[EntityMention]) -> list[Relation]:
    """(Paper)-[:STUDIES]->(Material), (Paper)-[:USES_PROCESS]->(Process)."""
    paper = doc.paper
    pref = EntityRef(id=paper.paper_id, name=paper.title or paper.paper_id, type=E.PAPER.value)
    title_l = (paper.title or "").lower()
    by_entity: dict[str, list[EntityMention]] = defaultdict(list)
    for m in mentions:
        if m.entity_type in (E.MATERIAL.value, E.PROCESS.value) and m.canonical_id:
            by_entity[m.canonical_id].append(m)
    smap = doc.sentence_map()
    out: list[Relation] = []
    for cid, ms in by_entity.items():
        sects = {smap[m.sentence_id].section_type for m in ms if m.sentence_id in smap}
        # A NEW (non-lexicon) entity needs >= 2 mentions outside the introduction before the
        # paper is said to study/use it; single passing mentions only filled the review queue.
        if all(m.normalization_method == "new_entity" for m in ms):
            core = [m for m in ms if m.sentence_id in smap and smap[m.sentence_id].section_type != "introduction"]
            if len(core) < 2:
                continue
        conf = max(_SECTION_WEIGHT.get(s, 0.8) for s in sects) if sects else 0.6
        in_title = any(m.text.lower() in title_l for m in ms)
        if in_title:
            conf = 0.97
        if len(ms) >= 3 and conf < 0.9 and sects != {"introduction"}:
            conf = 0.9
        best = sorted(ms, key=lambda m: _PRIORITY.index(smap[m.sentence_id].section_type)
                      if m.sentence_id in smap and smap[m.sentence_id].section_type in _PRIORITY else 99)[0]
        m0 = best
        rel = RelationType.STUDIES if m0.entity_type == E.MATERIAL.value else RelationType.USES_PROCESS
        eref = EntityRef(id=cid, name=m0.canonical_name or m0.text, type=m0.entity_type, surface=m0.text,
                         normalization_confidence=max(m.normalization_confidence for m in ms))
        out.append(Relation(
            claim_id=stable_id("c", paper.paper_id, rel.value, cid), subject=pref, relation=rel.value, object=eref,
            evidence=m0.evidence, confidence=conf, method="paper_mentions", knowledge_type=KnowledgeType.DERIVED,
            attributes={"mention_count": len(ms), "sections": sorted(sects), "in_title": in_title,
                        "surfaces": sorted({m.text for m in ms})[:10]},
        ))
    return out


def _to_float(cell: str) -> Optional[float]:
    m = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", cell or "")
    return float(m.group(1)) if m else None


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def table_trend_relations(doc: ParsedDocument, matcher: LexiconMatcher, normalizer: EntityNormalizer,
                          min_points: int = 3, min_abs_rho: float = 0.8) -> list[Relation]:
    out: list[Relation] = []
    row_sents = defaultdict(list)
    for s in doc.sentences:
        if s.source_type == "table" and s.table_id:
            row_sents[s.table_id].append(s)
    for tb in doc.tables:
        cols: list[Optional[EntityMention]] = []
        for h in tb.header:
            mt = matcher.find(h)
            if not mt:
                cols.append(None)
                continue
            nm = normalizer.normalize(mt[0].text, mt[0].entity_type)
            cols.append(EntityMention(mention_id="", text=mt[0].text, entity_type=nm.entity_type,
                                      canonical_id=nm.canonical_id, canonical_name=nm.canonical_name,
                                      normalization_confidence=nm.confidence, sentence_id=""))
        values = [[_to_float(c) for c in row] for row in tb.rows]
        factors = [i for i, c in enumerate(cols) if c and c.entity_type in (E.PARAMETER.value, E.CONDITION.value)]
        responses = [i for i, c in enumerate(cols) if c and c.entity_type in (E.PROPERTY.value, E.DEFECT.value, E.MICROSTRUCTURE.value)]
        sents = row_sents.get(tb.table_id, [])
        for f in factors:
            others = [o for o in factors if o != f]
            groups: dict[tuple, list[int]] = defaultdict(list)
            for ri, row in enumerate(values):
                groups[tuple(row[o] if o < len(row) else None for o in others)].append(ri)
            rows = max(groups.values(), key=len) if groups else []
            for r in responses:
                pts = [(values[i][f], values[i][r]) for i in rows
                       if f < len(values[i]) and r < len(values[i]) and values[i][f] is not None and values[i][r] is not None]
                if len(pts) < min_points or len({p[0] for p in pts}) < min_points:
                    continue
                rho = _spearman([p[0] for p in pts], [p[1] for p in pts])
                if abs(rho) < min_abs_rho:
                    continue
                fm, rm = cols[f], cols[r]
                assert fm is not None and rm is not None
                rel = infer_effect_relation(EntityType(fm.entity_type), EntityType(rm.entity_type), "affect")
                if rel is None:
                    continue
                used = [sents[i] for i in rows if i < len(sents)]
                if not used:
                    continue
                ev = Evidence(paper_id=doc.paper.paper_id, page=used[-1].page, section=used[-1].section_title,
                              section_type=used[-1].section_type, sentence_id=used[-1].sentence_id,
                              text=" ".join(s.text for s in used), source_type="table", table_id=tb.table_id,
                              context_sentence_ids=[s.sentence_id for s in used[:-1]])
                sref = EntityRef(id=fm.canonical_id or "", name=fm.canonical_name or fm.text, type=fm.entity_type,
                                 surface=fm.text, normalization_confidence=fm.normalization_confidence)
                oref = EntityRef(id=rm.canonical_id or "", name=rm.canonical_name or rm.text, type=rm.entity_type,
                                 surface=rm.text, normalization_confidence=rm.normalization_confidence)
                trend = "positive" if rho > 0 else "negative"
                out.append(Relation(
                    claim_id=stable_id("c", doc.paper.paper_id, sref.id, rel.value, oref.id, tb.table_id, "trend"),
                    subject=sref, relation=rel.value, object=oref, evidence=ev, confidence=0.78,
                    method="table_trend", knowledge_type=KnowledgeType.DERIVED,
                    attributes={"derivation": "monotonic trend over table rows (Spearman)", "spearman_rho": round(rho, 3),
                                "n_points": len(pts), "trend": trend, "table_id": tb.table_id,
                                "subject_unit": unit_from_header(tb.header[f]), "object_unit": unit_from_header(tb.header[r]),
                                "claim_type": "finding", "claim_origin": "this_study",
                                "caveat": "computed from reported table values; not a statistical test"},
                ))
    return out
