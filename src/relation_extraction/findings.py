"""Rule-based scientific finding detection.

A Finding is always the verbatim sentence (EXPLICIT); the system never
paraphrases findings into the graph.
"""
from __future__ import annotations

import re
from collections import defaultdict

from ..entity_extraction.rule_based import SentenceAnalysis, evidence_for
from ..ontology.models import Finding, Relation, stable_id
from ..ontology.schema import EFFECT_RELATIONS, KnowledgeType

_FINDING_SECTIONS = {"abstract", "results", "discussion", "results_discussion", "conclusion"}
_COMPARISON = re.compile(r"\b(highest|lowest|higher than|lower than|compared (to|with)|whereas|than|"
                         r"maximum|minimum|best|worst|superior|inferior|comparable)\b", re.I)


def rule_findings(paper_id: str, analyses: dict[str, SentenceAnalysis], relations: list[Relation]) -> list[Finding]:
    effect_names = {r.value for r in EFFECT_RELATIONS}
    by_sentence: dict[str, list[Relation]] = defaultdict(list)
    for r in relations:
        if (r.evidence and r.evidence.sentence_id and r.relation in effect_names
                and r.knowledge_type == KnowledgeType.EXPLICIT
                and r.attributes.get("claim_type") == "finding"
                and r.attributes.get("claim_origin") != "background"
                and not r.evidence.context_sentence_ids):
            by_sentence[r.evidence.sentence_id].append(r)
    out: list[Finding] = []
    for sid, an in analyses.items():
        s = an.sentence
        if s.section_type not in _FINDING_SECTIONS or s.source_type != "text":
            continue
        rels = by_sentence.get(sid, [])
        if rels:
            if any(r.attributes.get("negated") for r in rels):
                ftype = "null_result"
            elif s.section_type == "conclusion":
                ftype = "conclusion"
            elif any(r.attributes.get("trend") in ("positive", "negative") for r in rels):
                ftype = "trend"
            else:
                ftype = "result"
            conf = max(r.confidence for r in rels)
        elif an.assignments and _COMPARISON.search(s.text):
            ftype, conf = "comparison", 0.75
        else:
            continue
        out.append(Finding(finding_id=stable_id("f", paper_id, sid), text=s.text, evidence=evidence_for(s, paper_id),
                           finding_type=ftype, related_claim_ids=[r.claim_id for r in rels],
                           confidence=round(conf, 3), method="rule"))
    return out
