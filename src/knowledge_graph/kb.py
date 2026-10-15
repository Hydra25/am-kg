"""KnowledgeBase: the reviewed, corpus-level view of all extraction results.

Loads data/extracted/*.json, applies human review decisions and entity
merges, and exposes the claims used to build Neo4j or the local graph store.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..config import Settings, get_settings
from ..normalization.entity_normalizer import NormalizationOverrides
from ..normalization.text import match_key
from ..ontology.models import CanonicalEntity, EntityRef, ExtractionResult, Finding, PaperMetadata, Relation
from ..ontology.schema import ClaimStatus, EntityType
from ..quality.validator import QualityController
from ..review.store import ReviewStore

log = logging.getLogger(__name__)


@dataclass
class KnowledgeBase:
    papers: dict[str, PaperMetadata] = field(default_factory=dict)
    claims: list[Relation] = field(default_factory=list)          # all claims incl. rejected (audit)
    findings: list[Finding] = field(default_factory=list)
    entities: dict[str, CanonicalEntity] = field(default_factory=dict)
    entity_papers: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    mention_counts: dict[str, Counter] = field(default_factory=dict)  # paper_id -> Counter(canonical_id)
    merges: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_results(cls, results: Iterable[ExtractionResult], review: Optional[ReviewStore] = None,
                     overrides: Optional[NormalizationOverrides] = None,
                     qc: Optional[QualityController] = None) -> "KnowledgeBase":
        kb = cls()
        merges = dict(overrides.merges) if overrides else {}
        if review:
            merges.update(review.merges())
        kb.merges = merges
        for res in results:
            kb.papers[res.paper.paper_id] = res.paper
            for cid, ent in res.canonical_entities.items():
                if cid not in kb.entities:
                    kb.entities[cid] = ent.model_copy(deep=True)
                else:
                    known = kb.entities[cid]
                    for a in ent.aliases:
                        if a not in known.aliases:
                            known.aliases.append(a)
            cnt: Counter = Counter()
            for m in res.entities:
                if m.canonical_id:
                    cid = kb.resolve_id(m.canonical_id)
                    cnt[cid] += 1
                    kb.entity_papers[cid].add(res.paper.paper_id)
            kb.mention_counts[res.paper.paper_id] = cnt
            kb.claims.extend(r.model_copy(deep=True) for r in res.relations)
            kb.findings.extend(f.model_copy(deep=True) for f in res.findings)
        kb._apply_merges()
        if review:
            kb._apply_review(review, qc)
        return kb

    @classmethod
    def load(cls, settings: Optional[Settings] = None, exclude_synthetic: bool = False) -> "KnowledgeBase":
        from ..pipeline import load_all_results
        s = settings or get_settings()
        results = load_all_results(s.path("extracted_dir"))
        if exclude_synthetic:                 # real papers only (e.g. for a demo / the paper)
            results = [r for r in results if not r.paper.is_synthetic]
        review = ReviewStore(s.path("review_dir"))
        overrides = NormalizationOverrides(s.path("review_dir") / "normalization_overrides.json")
        return cls.from_results(results, review, overrides, QualityController(s.quality))

    # ------------------------------------------------------------------
    def resolve_id(self, cid: str) -> str:
        seen = set()
        while cid in self.merges and cid not in seen:
            seen.add(cid)
            cid = self.merges[cid]
        return cid

    def _remap(self, ref: EntityRef) -> None:
        new = self.resolve_id(ref.id)
        if new != ref.id:
            ref.id = new
            ent = self.entities.get(new)
            ref.name = ent.canonical_name if ent else new.split(":", 1)[-1]
            ref.normalization_confidence = 1.0

    def _apply_merges(self) -> None:
        for src, dst in self.merges.items():
            dst = self.resolve_id(dst)
            s_ent = self.entities.pop(src, None)
            if s_ent is None:
                continue
            d_ent = self.entities.get(dst)
            if d_ent is None:
                d_ent = s_ent.model_copy(update={"id": dst, "canonical_name": dst.split(":", 1)[-1], "aliases": []})
                self.entities[dst] = d_ent
            for a in [s_ent.canonical_name] + s_ent.aliases:
                if a not in d_ent.aliases and a != d_ent.canonical_name:
                    d_ent.aliases.append(a)
            self.entity_papers[dst] |= self.entity_papers.pop(src, set())
        for c in self.claims:
            self._remap(c.subject)
            self._remap(c.object)

    def _apply_review(self, review: ReviewStore, qc: Optional[QualityController]) -> None:
        latest = review.latest()
        accepted_entities = {tid for (kind, tid), d in latest.items() if kind == "entity" and d.action == "accept"}
        for (kind, tid), d in latest.items():
            if kind == "entity" and d.action == "edit" and d.payload.get("canonical_name"):
                ent = self.entities.get(tid)
                if ent:
                    ent.canonical_name = d.payload["canonical_name"]
                    ent.normalization_confidence = 1.0
        for ent_id in accepted_entities:
            if ent_id in self.entities:
                self.entities[ent_id].normalization_confidence = 1.0
        by_id = {c.claim_id: c for c in self.claims}
        for c in self.claims:  # entity acceptance can lift claims out of review
            touched = False
            for ref in (c.subject, c.object):
                if ref.id in accepted_entities and ref.normalization_confidence < 1.0:
                    ref.normalization_confidence = 1.0
                    touched = True
            if touched and qc and c.status != ClaimStatus.REJECTED:
                c.qc_reasons = [r for r in c.qc_reasons if r != "new_entity_needs_review"]
                qc.assess(c, None)
        for (kind, tid), d in latest.items():
            if kind == "claim" and tid in by_id:
                c = by_id[tid]
                if d.action == "accept":
                    c.status = ClaimStatus.ACCEPTED
                elif d.action == "reject":
                    c.status = ClaimStatus.REJECTED
                elif d.action == "edit":
                    self._edit_claim(c, d.payload)
                    c.status = ClaimStatus(d.payload.get("status", "accepted"))
                c.reviewed = True
                c.review_note = d.note or f"{d.action} by {d.reviewer} at {d.timestamp}"
            elif kind == "finding":
                for f in self.findings:
                    if f.finding_id == tid:
                        f.status = ClaimStatus.ACCEPTED if d.action == "accept" else (
                            ClaimStatus.REJECTED if d.action == "reject" else f.status)

    def _edit_claim(self, c: Relation, p: dict) -> None:
        if p.get("relation"):
            c.relation = p["relation"]
        for side in ("subject", "object"):
            ref: EntityRef = getattr(c, side)
            if p.get(f"{side}_id"):
                ref.id = p[f"{side}_id"]
                ent = self.entities.get(ref.id)
                ref.name = p.get(f"{side}_name") or (ent.canonical_name if ent else ref.id.split(":", 1)[-1])
                ref.type = p.get(f"{side}_type") or (ent.entity_type if ent else ref.type)
                ref.normalization_confidence = 1.0
        if p.get("value_raw") and c.object.quantity is not None:
            from ..normalization.units import parse_quantity_text
            q = parse_quantity_text(p["value_raw"])
            if q:
                c.object.quantity = q
                c.object.name = q.raw
        c.method = c.method + "+manual" if "manual" not in c.method else c.method
        c.attributes["edited"] = True

    # ------------------------------------------------------------------
    def active_claims(self, statuses: Iterable[str] = ("accepted", "review")) -> list[Relation]:
        st = set(statuses)
        return [c for c in self.claims if c.status.value in st]

    def entity_name(self, cid: str) -> str:
        e = self.entities.get(cid)
        return e.canonical_name if e else cid.split(":", 1)[-1]

    def find_entities(self, text: str, entity_type: Optional[str] = None) -> list[str]:
        """Resolve free text (canonical name or alias, any spelling) to canonical ids."""
        key = match_key(text)
        out = []
        for cid, e in self.entities.items():
            if entity_type and e.entity_type != entity_type:
                continue
            if key in {match_key(e.canonical_name)} | {match_key(a) for a in e.aliases}:
                out.append(cid)
        return out

    def paper_context(self, paper_id: str, statuses: Iterable[str] = ("accepted", "review")) -> dict[str, set[str]]:
        st = set(statuses)
        ctx: dict[str, set[str]] = {"materials": set(), "processes": set()}
        for c in self.claims:
            if c.subject.id == paper_id and c.status.value in st:
                if c.relation == "STUDIES":
                    ctx["materials"].add(c.object.id)
                elif c.relation == "USES_PROCESS":
                    ctx["processes"].add(c.object.id)
        return ctx

    def stats(self) -> dict:
        st = Counter(c.status.value for c in self.claims)
        kt = Counter(c.knowledge_type.value for c in self.claims if c.status != ClaimStatus.REJECTED)
        rel = Counter(c.relation for c in self.claims if c.status != ClaimStatus.REJECTED)
        types = Counter(e.entity_type for e in self.entities.values())
        return {"papers": len(self.papers), "entities": len(self.entities), "claims_total": len(self.claims),
                "claims_by_status": dict(st), "claims_by_knowledge_type": dict(kt),
                "claims_by_relation": dict(rel), "entities_by_type": dict(types),
                "findings": len(self.findings),
                "synthetic_papers": sum(p.is_synthetic for p in self.papers.values())}

    def review_queue(self) -> list[Relation]:
        return sorted((c for c in self.claims if c.status == ClaimStatus.REVIEW),
                      key=lambda c: c.confidence)

    def new_entities(self) -> list[CanonicalEntity]:
        return [e for e in self.entities.values() if not e.in_lexicon and e.normalization_confidence < 1.0]

    def claim(self, claim_id: str) -> Optional[Relation]:
        return next((c for c in self.claims if c.claim_id == claim_id), None)

    def is_value_type(self, t: str) -> bool:
        return t in (EntityType.PARAMETER_VALUE.value, EntityType.PROPERTY_VALUE.value)
