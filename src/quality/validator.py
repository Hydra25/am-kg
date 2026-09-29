"""Quality control: decides accepted / review / rejected for every claim.

Hard rules (-> rejected):
  missing subject/object, missing evidence, evidence not found in the source
  document, relation not allowed by the ontology, self-loops, value/unit
  dimension mismatch, confidence below review_threshold.
Soft rules (-> at most 'review'):
  endpoint is a new (non-lexicon) entity, LLM quote does not match the
  sentence, background/introduction claims, values with unknown units,
  derived (computed) relations.
Otherwise: confidence >= accept_threshold -> accepted, else review.
"""
from __future__ import annotations

from typing import Optional

from ..config import QualityConfig
from ..evidence.verify import EvidenceIndex
from ..ontology.models import Finding, Relation
from ..ontology.schema import ClaimStatus, KnowledgeType, is_valid_signature

HARD_EXTRACTOR_REASONS = ("value_dimension_mismatch",)
SOFT_EXTRACTOR_REASONS = ("llm_evidence_quote_not_in_sentence",)


class QualityController:
    def __init__(self, cfg: QualityConfig) -> None:
        self.cfg = cfg

    def status_for(self, conf: float) -> ClaimStatus:
        if conf >= self.cfg.accept_threshold:
            return ClaimStatus.ACCEPTED
        if conf >= self.cfg.review_threshold:
            return ClaimStatus.REVIEW
        return ClaimStatus.REJECTED

    def assess(self, rel: Relation, index: Optional[EvidenceIndex]) -> Relation:
        hard: list[str] = []
        soft: list[str] = []
        reasons = list(rel.qc_reasons)
        if not rel.subject.id or not rel.subject.name or not rel.object.id or not rel.object.name:
            hard.append("missing_entity")
        if self.cfg.require_evidence and (rel.evidence is None or not rel.evidence.text.strip()):
            hard.append("missing_evidence")
        elif self.cfg.require_evidence_in_source and index is not None:
            ok, why = index.verify(rel.evidence)
            if not ok:
                hard.append(why)
        if not is_valid_signature(rel.relation, rel.subject.type, rel.object.type):
            hard.append(f"invalid_signature:{rel.subject.type}-{rel.relation}->{rel.object.type}")
        if rel.subject.id == rel.object.id:
            hard.append("self_loop")
        for r in reasons:
            if r.startswith(HARD_EXTRACTOR_REASONS):
                hard.append(r)
            elif r.startswith(SOFT_EXTRACTOR_REASONS):
                soft.append(r)
        # normalisation quality of the endpoints scales the confidence
        norm_min = min(rel.subject.normalization_confidence, rel.object.normalization_confidence)
        if rel.subject.normalization_confidence <= 0.5 or rel.object.normalization_confidence <= 0.5:
            soft.append("new_entity_needs_review")
        if rel.attributes.get("claim_origin") == "background":
            soft.append("background_claim_from_cited_work")
        if rel.relation == "HAS_VALUE" and rel.object.quantity is not None and rel.object.quantity.normalized_unit is None:
            soft.append("unknown_unit")
        if rel.knowledge_type == KnowledgeType.DERIVED and rel.method != "paper_mentions":
            soft.append("derived_relation")

        raw = rel.attributes.get("confidence_raw", rel.confidence)
        final = round(raw * (0.6 + 0.4 * norm_min), 3)
        rel.attributes["confidence_raw"] = raw
        rel.confidence = final
        if hard:
            rel.status = ClaimStatus.REJECTED
        else:
            status = self.status_for(final)
            if soft and status == ClaimStatus.ACCEPTED:
                status = ClaimStatus.REVIEW
            rel.status = status
            if status == ClaimStatus.REJECTED:
                hard.append(f"confidence_below_{self.cfg.review_threshold}")
        rel.qc_reasons = sorted(set(hard + soft + [r for r in reasons if r not in hard + soft]))
        return rel

    def assess_finding(self, f: Finding, index: Optional[EvidenceIndex]) -> Finding:
        ok = True
        if index is not None:
            ok, _ = index.verify(f.evidence)
        f.status = self.status_for(f.confidence) if ok else ClaimStatus.REJECTED
        return f
