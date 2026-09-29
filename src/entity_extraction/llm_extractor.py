"""LLM-based extraction (entities, relations, findings) with hallucination guards.

The LLM (local Qwen via Ollama) proposes items; the code *verifies* them:
  * entity text must literally occur in the cited sentence
  * values must literally occur in the sentence and parse with a known unit
  * value dimension must match the entity (W for power, MPa for strength ...)
  * relation endpoints must be entities found in the cited sentence
  * relation type must satisfy the ontology signature
  * the evidence quote must overlap the cited sentence
Anything that fails is dropped (entities) or kept with a QC rejection reason
(relations) so that the audit trail shows what the model proposed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

from ..llm.client import LLMProvider
from ..llm.instrumented import llm_stage
from ..llm.json_utils import call_llm_json
from ..llm.prompts import PromptLibrary
from ..normalization.entity_normalizer import EntityNormalizer
from ..normalization.lexicon import Lexicon
from ..normalization.text import match_key, normalize_chars
from ..normalization.units import parse_quantity_text
from ..ontology.models import EntityMention, EntityRef, Finding, ParsedDocument, Relation, Sentence, stable_id
from ..ontology.schema import EntityType, KnowledgeType, RelationType, canonicalize_relation
from ..quality.junk import junk_reason
from .rule_based import SentenceAnalysis, evidence_for

log = logging.getLogger(__name__)

_TYPE_ALIASES = {
    "amprocess": "Process", "process": "Process", "manufacturingprocess": "Process",
    "material": "Material", "alloy": "Material",
    "parameter": "Parameter", "processparameter": "Parameter", "machineparameter": "Parameter",
    "property": "Property", "mechanicalproperty": "Property", "thermalproperty": "Property",
    "surfaceproperty": "Property",
    "defect": "Defect", "microstructure": "Microstructure",
    "postprocessing": "PostProcessing", "post-processing": "PostProcessing",
    "characterizationmethod": "CharacterizationMethod", "characterization": "CharacterizationMethod",
    "characterisationmethod": "CharacterizationMethod", "method": "CharacterizationMethod",
    "geometry": "Geometry", "experimentalcondition": "ExperimentalCondition", "condition": "ExperimentalCondition",
}
_EXPLICIT_CUE = re.compile(r"\b(increas|decreas|reduc|improv|enhanc|promot|affect|influenc|caus|led to|result|"
                           r"lower|rais|suppress|eliminat|induc|deteriorat|degrad|transform|dissolv|refin|"
                           r"fabricat|manufactur|produc|print|built|process|deposit|effect)", re.I)


def canonical_type(t: Optional[str]) -> Optional[str]:
    if not t:
        return None
    return _TYPE_ALIASES.get(re.sub(r"[\s_]", "", t).lower())


def _sid(v: Any) -> str:
    m = re.search(r"\d+", str(v))
    return f"S{m.group()}" if m else str(v)


class LLMEntity(BaseModel):
    text: str
    type: str
    sentence: str
    value: Optional[str] = None
    confidence: float = 0.5

    @field_validator("sentence", mode="before")
    @classmethod
    def norm_sentence(cls, v: Any) -> str:
        return _sid(v)


class LLMEntities(BaseModel):
    entities: list[LLMEntity] = Field(default_factory=list)


class LLMRelation(BaseModel):
    """Compact relation schema. The model does NOT copy the evidence sentence:
    the sentence id is enough (we already have its verbatim text), and copying it
    was the largest share of output tokens (= most of the relation time)."""
    subject: str
    relation: str
    object: str
    sentence: str
    effect: Optional[str] = None
    negated: bool = False
    confidence: float = 0.5

    @field_validator("sentence", mode="before")
    @classmethod
    def norm_sentence(cls, v: Any) -> str:
        return _sid(v)

    @field_validator("object", "subject", mode="before")
    @classmethod
    def _to_str(cls, v: Any) -> str:
        return "" if v is None else str(v)


class LLMRelations(BaseModel):
    relations: list[LLMRelation] = Field(default_factory=list)


class LLMFinding(BaseModel):
    sentence: str
    finding_type: str = "result"
    confidence: float = 0.5

    @field_validator("sentence", mode="before")
    @classmethod
    def norm_sentence(cls, v: Any) -> str:
        return _sid(v)


class LLMFindings(BaseModel):
    findings: list[LLMFinding] = Field(default_factory=list)


class NormChoice(BaseModel):
    choice: str
    confidence: float = 0.5


@dataclass
class LLMExtractionOutput:
    mentions: list[EntityMention] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)   # audit trail of rejected proposals
    calls: int = 0
    failures: int = 0
    chunks: int = 0            # chunks considered for the LLM
    chunks_skipped: int = 0    # chunks skipped by rule-first gating


_FINDING_SECTIONS = ("abstract", "results", "discussion", "results_discussion", "conclusion")


def chunk_is_relevant(sents: list[Sentence], analyses: dict[str, "SentenceAnalysis"]) -> bool:
    """Rule-first gating: a chunk is worth an LLM call only if the (fast, free)
    rule-based pass found at least one AM entity or a number with a unit in it.
    Chunks without either are usually affiliations, copyright lines, funding
    text or generic background, where the LLM mostly produces noise.
    Abstract and conclusion chunks are always kept: they carry the paper's main
    findings even when phrased without lexicon terms or numbers."""
    for s in sents:
        if s.section_type in ("abstract", "conclusion"):
            return True
        an = analyses.get(s.sentence_id)
        if an is not None and (an.mentions or an.quantities):
            return True
    return False


def _find_span(needle: str, hay: str) -> Optional[tuple[int, int]]:
    n = normalize_chars(needle).strip().lower()
    h = normalize_chars(hay).lower()
    if not n:
        return None
    i = h.find(n)
    if i >= 0:
        return i, i + len(n)
    # whitespace-insensitive search
    pat = r"\s*".join(re.escape(c) for c in n if not c.isspace())
    m = re.search(pat, h)
    return (m.start(), m.end()) if m else None


def chunk_sentences(doc: ParsedDocument, skip: list[str], max_chars: int) -> list[tuple[str, list[Sentence]]]:
    chunks: list[tuple[str, list[Sentence]]] = []
    cur: list[Sentence] = []
    cur_sec = None
    size = 0
    for s in doc.sentences:
        if s.section_type in skip:
            continue
        if cur and (s.section_id != cur_sec or size + len(s.text) > max_chars):
            chunks.append((cur[0].section_title, cur))
            cur, size = [], 0
        cur.append(s)
        cur_sec = s.section_id
        size += len(s.text) + 8
    if cur:
        chunks.append((cur[0].section_title, cur))
    return chunks


class LLMExtractor:
    def __init__(self, llm: LLMProvider, lexicon: Lexicon, normalizer: EntityNormalizer,
                 prompts: PromptLibrary, max_retries: int = 2, extract_findings: bool = True) -> None:
        self.llm = llm
        self.lexicon = lexicon
        self.normalizer = normalizer
        self.prompts = prompts
        self.max_retries = max_retries
        self.extract_findings = extract_findings

    # ------------------------------------------------------------------
    def choose_canonical(self, term: str, entity_type: str, candidates: list[str]) -> Optional[str]:
        """Used by EntityNormalizer: LLM may only pick from the given candidates."""
        user = self.prompts.render("normalization", term=term, entity_type=entity_type,
                                   candidates="\n".join(f"- {c}" for c in candidates))
        with llm_stage(self.llm, "normalization"):
            res, _ = call_llm_json(self.llm, self.prompts.raw("system_extraction"), user, NormChoice, 1)
        if res and res.choice in candidates and res.confidence >= 0.6:
            return res.choice
        return None

    # ------------------------------------------------------------------
    def extract(self, doc: ParsedDocument, analyses: dict[str, SentenceAnalysis], skip: list[str],
                chunk_chars: int, gate: bool = False, llm_entities: bool = True) -> LLMExtractionOutput:
        """`skip`: section types never sent to the LLM. `gate`: skip chunks in which
        the rule-based pass found nothing (see `chunk_is_relevant`). `llm_entities`:
        False when another extractor (GLiNER) already proposed the entities."""
        out = LLMExtractionOutput()
        paper_id = doc.paper.paper_id
        system = self.prompts.raw("system_extraction")
        for section, sents in chunk_sentences(doc, skip, chunk_chars):
            out.chunks += 1
            if gate and not chunk_is_relevant(sents, analyses):
                out.chunks_skipped += 1
                log.debug("Skipping chunk in %r: no rule-based entities or quantities", section)
                continue
            local = {f"S{i + 1}": s for i, s in enumerate(sents)}
            numbered = "\n".join(f"[{k}] {s.text}" for k, s in local.items())
            # 1) entities (skipped when GLiNER already did this job)
            if llm_entities:
                with llm_stage(self.llm, "entities"):
                    ents, clog = call_llm_json(self.llm, system,
                                               self.prompts.render("entity_extraction", section=section,
                                                                   sentences=numbered),
                                               LLMEntities, self.max_retries, wrap_key="entities")
                out.calls += clog.attempts
                if ents is None:
                    out.failures += 1
                    out.dropped.append({"stage": "entities", "section": section, "errors": clog.errors})
                else:
                    for e in ents.entities:
                        self._accept_entity(e, local, analyses, paper_id, out)
            # 2) relations (entity list = rule + LLM mentions of these sentences)
            ent_lines = []
            for k, s in local.items():
                an = analyses.get(s.sentence_id)
                if an:
                    ent_lines += [f"{m.text} ({m.entity_type}, {k})" for m in an.mentions]
            if ent_lines:
                with llm_stage(self.llm, "relations"):
                    rels, clog = call_llm_json(self.llm, system,
                                               self.prompts.render("relation_extraction", section=section,
                                                                   sentences=numbered, entities="; ".join(ent_lines)),
                                               LLMRelations, self.max_retries, wrap_key="relations")
                out.calls += clog.attempts
                if rels is None:
                    out.failures += 1
                    out.dropped.append({"stage": "relations", "section": section, "errors": clog.errors})
                else:
                    for r in rels.relations:
                        rel = self._accept_relation(r, local, analyses, paper_id, out)
                        if rel:
                            out.relations.append(rel)
            # 3) findings
            if self.extract_findings and any(s.section_type in _FINDING_SECTIONS for s in sents):
                with llm_stage(self.llm, "findings"):
                    fnd, clog = call_llm_json(self.llm, system,
                                              self.prompts.render("finding_extraction", section=section,
                                                                  sentences=numbered),
                                              LLMFindings, self.max_retries, wrap_key="findings")
                out.calls += clog.attempts
                if fnd is None:
                    out.failures += 1
                else:
                    for f in fnd.findings:
                        s = local.get(f.sentence)
                        if s is None:
                            out.dropped.append({"stage": "finding", "reason": "unknown_sentence", "item": f.model_dump()})
                            continue
                        ftype = f.finding_type if f.finding_type in {"result", "trend", "comparison", "null_result", "conclusion"} else "result"
                        out.findings.append(Finding(
                            finding_id=stable_id("f", paper_id, s.sentence_id), text=s.text,
                            evidence=evidence_for(s, paper_id), finding_type=ftype,
                            confidence=round(min(max(f.confidence, 0.0), 0.95) * 0.9, 3), method="llm"))
        return out

    # ------------------------------------------------------------------
    def _accept_entity(self, e: LLMEntity, local: dict[str, Sentence], analyses: dict[str, SentenceAnalysis],
                       paper_id: str, out: LLMExtractionOutput) -> None:
        s = local.get(e.sentence)
        etype = canonical_type(e.type)
        if s is None or etype is None or etype not in {t.value for t in EntityType.concept_types()}:
            out.dropped.append({"stage": "entity", "reason": "unknown_sentence_or_type", "item": e.model_dump()})
            return
        span = _find_span(e.text, s.text)
        if span is None:
            out.dropped.append({"stage": "entity", "reason": "text_not_in_sentence", "item": e.model_dump()})
            return
        an = analyses.setdefault(s.sentence_id, SentenceAnalysis(sentence=s))
        surface = s.text[span[0]:span[1]]
        # junk filter BEFORE normalisation (which may itself ask the LLM)
        reason = None if self.lexicon.lookup_any_type(surface) else junk_reason(surface, etype, False)
        if reason:
            out.dropped.append({"stage": "entity", "reason": f"junk:{reason}", "item": e.model_dump()})
            return
        norm = self.normalizer.normalize(surface, etype)
        quantity = None
        if e.value:
            vspan = _find_span(e.value, s.text)
            q = parse_quantity_text(s.text[vspan[0]:vspan[1]]) if vspan else None
            dims = self.lexicon.dimensions(norm.canonical_id)
            if q is None:
                out.dropped.append({"stage": "value", "reason": "value_not_in_sentence_or_unparseable", "item": e.model_dump()})
            elif dims and q.dimension not in dims:
                out.dropped.append({"stage": "value", "reason": f"dimension_mismatch({q.dimension} not in {dims})", "item": e.model_dump()})
            else:
                quantity = q
        # merge with an overlapping rule mention of the same canonical entity
        for m in an.mentions:
            if m.canonical_id == norm.canonical_id and m.char_start is not None and \
                    m.char_start < span[1] and span[0] < (m.char_end or 0):
                m.method = "rule+llm" if m.method == "rule" else m.method
                m.confidence = round(min(0.98, 1 - (1 - m.confidence) * (1 - min(e.confidence, 0.9))), 3)
                if m.quantity is None and quantity is not None:
                    m.quantity = quantity
                out.mentions.append(m)
                return
        mention = EntityMention(
            mention_id=stable_id("m", s.sentence_id, span[0], span[1], "llm"), text=surface, entity_type=norm.entity_type,
            canonical_id=norm.canonical_id, canonical_name=norm.canonical_name,
            normalization_confidence=norm.confidence, normalization_method=norm.method,
            sentence_id=s.sentence_id, char_start=span[0], char_end=span[1], quantity=quantity,
            confidence=round(0.5 * min(max(e.confidence, 0.0), 0.95) + 0.5 * norm.confidence, 3),
            method="llm", evidence=evidence_for(s, paper_id))
        an.mentions.append(mention)
        out.mentions.append(mention)

    def _find_mention(self, text: str, an: SentenceAnalysis) -> Optional[EntityMention]:
        key = match_key(text)
        for m in an.mentions:
            if match_key(m.text) == key or (m.canonical_name and match_key(m.canonical_name) == key):
                return m
        for m in an.mentions:
            mk = match_key(m.text)
            if len(key) >= 3 and (key in mk or mk in key):
                return m
        return None

    def _accept_relation(self, r: LLMRelation, local: dict[str, Sentence], analyses: dict[str, SentenceAnalysis],
                         paper_id: str, out: LLMExtractionOutput) -> Optional[Relation]:
        s = local.get(r.sentence)
        if s is None:
            out.dropped.append({"stage": "relation", "reason": "unknown_sentence", "item": r.model_dump()})
            return None
        an = analyses.get(s.sentence_id) or SentenceAnalysis(sentence=s)
        subj = self._find_mention(r.subject, an)
        if subj is None:
            out.dropped.append({"stage": "relation", "reason": "subject_not_in_sentence", "item": r.model_dump()})
            return None
        ev = evidence_for(s, paper_id)
        rel_name = (r.relation or "").strip().upper().replace(" ", "_")
        # Grounding: both endpoints were found in the cited sentence (checked above
        # and below), and the evidence stored with the claim is that sentence verbatim.
        llm_conf = min(max(r.confidence, 0.0), 0.95)
        attrs: dict[str, Any] = {"llm_sentence": r.sentence}
        qc: list[str] = []

        if rel_name == "HAS_VALUE":
            vspan = _find_span(r.object, s.text)
            q = parse_quantity_text(s.text[vspan[0]:vspan[1]]) if vspan else None
            if q is None:
                out.dropped.append({"stage": "relation", "reason": "value_not_in_sentence", "item": r.model_dump()})
                return None
            dims = self.lexicon.dimensions(subj.canonical_id or "")
            if dims and q.dimension not in dims:
                qc.append(f"value_dimension_mismatch:{q.dimension}")
            vtype = EntityType.PARAMETER_VALUE if subj.entity_type in ("Parameter", "ExperimentalCondition") else EntityType.PROPERTY_VALUE
            vid = stable_id("val", paper_id, s.sentence_id, subj.canonical_id, vspan[0] if vspan else 0, q.raw)
            obj_ref = EntityRef(id=vid, name=q.raw, type=vtype.value, surface=q.raw, quantity=q)
            canon, swapped = RelationType.HAS_VALUE.value, False
            ktype = KnowledgeType.EXTRACTED
        else:
            obj = self._find_mention(r.object, an)
            if obj is None:
                out.dropped.append({"stage": "relation", "reason": "object_not_in_sentence", "item": r.model_dump()})
                return None
            res = canonicalize_relation(rel_name, subj.entity_type, obj.entity_type)
            if res is None:
                out.dropped.append({"stage": "relation", "reason": f"unknown_relation:{rel_name}", "item": r.model_dump()})
                return None
            canon, swapped = res
            if swapped:
                subj, obj = obj, subj
            obj_ref = EntityRef(id=obj.canonical_id or "", name=obj.canonical_name or obj.text, type=obj.entity_type,
                                surface=obj.text, normalization_confidence=obj.normalization_confidence)
            ktype = KnowledgeType.EXPLICIT if _EXPLICIT_CUE.search(s.text) and canon not in (
                "HAS_PARAMETER", "HAS_CONDITION", "CHARACTERIZES") else KnowledgeType.EXTRACTED
            eff = (r.effect or "").lower() or None
            attrs.update({"object_change": "none" if r.negated else eff, "negated": bool(r.negated or eff == "none"),
                          "claim_type": "finding" if ktype == KnowledgeType.EXPLICIT else None,
                          "claim_origin": "background" if s.section_type == "introduction" else "this_study"})
        subj_ref = EntityRef(id=subj.canonical_id or "", name=subj.canonical_name or subj.text, type=subj.entity_type,
                             surface=subj.text, normalization_confidence=subj.normalization_confidence)
        conf = 0.5 * llm_conf + 0.45
        if s.section_type == "introduction":
            conf -= 0.15
        cid = stable_id("c", paper_id, subj_ref.id, canon, obj_ref.id, s.sentence_id)
        return Relation(claim_id=cid, subject=subj_ref, relation=canon, object=obj_ref, evidence=ev,
                        confidence=round(max(0.0, min(conf, 0.95)), 3), method="llm", knowledge_type=ktype,
                        qc_reasons=qc, attributes={k: v for k, v in attrs.items() if v is not None})
