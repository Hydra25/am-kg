"""End-to-end extraction pipeline.

    file -> parse (sections, sentences, tables)       data/processed/<paper_id>.json
         -> rule-based entities + values + relations
         -> [optional] Qwen/Ollama entities, relations, findings (verified)
         -> derived relations (paper-level, table trends)
         -> normalisation + merge + quality control
         -> ExtractionResult                          data/extracted/<paper_id>.json

Graph building is a separate step (src/knowledge_graph) that reads the
extracted JSON, so extraction can be re-run and reviewed independently.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable, Optional

from .config import PIPELINE_VERSION, Settings, get_settings
from .entity_extraction.llm_extractor import LLMExtractor
from .entity_extraction.rule_based import RuleEntityExtractor, SentenceAnalysis
from .evidence.verify import EvidenceIndex
from .ingestion.loader import discover_inputs, parse_document, save_parsed
from .llm.client import LLMProvider, OllamaClient
from .llm.prompts import PromptLibrary
from .normalization.entity_normalizer import EntityNormalizer, NormalizationOverrides, find_abbreviations
from .normalization.lexicon import Lexicon
from .ontology.models import (
    CanonicalEntity,
    EntityMention,
    ExtractionResult,
    ExtractionStats,
    Finding,
    ParsedDocument,
    Provenance,
    Relation,
)
from .ontology.schema import ClaimStatus, KnowledgeType
from .quality.validator import QualityController
from .relation_extraction.derived import paper_level_relations, table_trend_relations
from .relation_extraction.findings import rule_findings
from .relation_extraction.rule_based import RuleRelationExtractor

log = logging.getLogger(__name__)
VALID_MODES = {"rule", "llm", "hybrid", "auto"}


def _merge_key(r: Relation) -> tuple:
    ev = r.evidence.sentence_id if r.evidence else None
    if r.relation == "HAS_VALUE" and r.object.quantity is not None:
        q = r.object.quantity
        okey = ("q", q.normalized_value, q.normalized_min, q.normalized_max, q.normalized_unit or q.unit)
    else:
        okey = (r.object.id,)
    return (r.subject.id, r.relation, okey, ev)


def merge_relations(rels: Iterable[Relation]) -> list[Relation]:
    """Merge duplicate claims (same endpoints, relation and evidence sentence)."""
    merged: dict[tuple, Relation] = {}
    for r in rels:
        k = _merge_key(r)
        if k not in merged:
            merged[k] = r
            continue
        a = merged[k]
        methods = sorted(set(a.method.split("+")) | set(r.method.split("+")))
        if a.method != r.method:  # agreement between independent methods raises confidence
            a.confidence = round(min(0.98, 1 - (1 - a.confidence) * (1 - r.confidence)), 3)
        else:
            a.confidence = max(a.confidence, r.confidence)
        a.method = "+".join(methods)
        # EXPLICIT beats EXTRACTED when two methods disagree on the epistemic type
        if r.knowledge_type == KnowledgeType.EXPLICIT:
            a.knowledge_type = KnowledgeType.EXPLICIT
        for key, val in r.attributes.items():
            a.attributes.setdefault(key, val)
        a.qc_reasons = sorted(set(a.qc_reasons) & set(r.qc_reasons)) if a.method != r.method else a.qc_reasons
    return list(merged.values())


def consolidate_context_claims(rels: list[Relation]) -> list[Relation]:
    """Drop weak context-resolved HAS_PARAMETER/HAS_CONDITION claims when the same
    pair is already supported by a same-sentence claim in this paper. Keeps the
    review queue focused on claims that actually need a human decision."""
    strong = {(r.subject.id, r.relation, r.object.id) for r in rels
              if r.relation in ("HAS_PARAMETER", "HAS_CONDITION")
              and r.attributes.get("context_resolution") == "same_sentence"
              and r.attributes.get("claim_origin") != "background"}
    out = []
    for r in rels:
        if (r.relation in ("HAS_PARAMETER", "HAS_CONDITION")
                and r.attributes.get("context_resolution") not in (None, "same_sentence")
                and (r.subject.id, r.relation, r.object.id) in strong):
            continue
        out.append(r)
    return out


class ExtractionPipeline:
    def __init__(self, settings: Optional[Settings] = None, llm: Optional[LLMProvider] = None,
                 mode: Optional[str] = None) -> None:
        self.settings = settings or get_settings()
        s = self.settings
        self.lexicon = Lexicon.load(s.path("lexicon_file"))
        self.overrides = NormalizationOverrides(s.path("review_dir") / "normalization_overrides.json")
        self.normalizer = EntityNormalizer(self.lexicon, self.overrides)
        self.rule_entities = RuleEntityExtractor(self.lexicon, self.normalizer)
        self.prompts = PromptLibrary(s.path("prompts_dir"))
        self.qc = QualityController(s.quality)
        requested = (mode or s.extraction.mode).lower()
        if requested not in VALID_MODES:
            raise ValueError(f"extraction mode must be one of {VALID_MODES}")
        self.llm: Optional[LLMProvider] = llm
        if requested in ("auto", "llm", "hybrid") and self.llm is None:
            client = OllamaClient.from_settings(s)
            if client.is_available():
                self.llm = client
            elif requested != "auto":
                raise RuntimeError(f"Ollama model '{s.llm.model}' not reachable at {s.llm.base_url}. "
                                   f"Start Ollama and run: ollama pull {s.llm.model} (or use --mode rule)")
        self.mode = ("hybrid" if self.llm else "rule") if requested == "auto" else requested
        self.llm_extractor = None
        if self.llm and self.mode in ("llm", "hybrid"):
            self.llm_extractor = LLMExtractor(self.llm, self.lexicon, self.normalizer, self.prompts,
                                              s.llm.max_retries, s.extraction.llm_findings)
            if s.extraction.llm_normalization:
                self.normalizer.llm_chooser = self.llm_extractor.choose_canonical
        log.info("Extraction mode: %s%s", self.mode, f" (model {self.llm.model})" if self.llm else "")

    # ------------------------------------------------------------------
    def process_file(self, path: Path, save: bool = True) -> ExtractionResult:
        t0 = time.time()
        doc = parse_document(Path(path), self.settings)
        if save:
            save_parsed(doc, self.settings.path("processed_dir"))
        result = self.extract(doc)
        result.stats.seconds = round(time.time() - t0, 2)
        if save:
            self.save_result(result)
        return result

    def process_many(self, path: Path) -> list[ExtractionResult]:
        results = []
        for f in discover_inputs(Path(path)):
            try:
                log.info("Processing %s", f.name)
                results.append(self.process_file(f))
            except Exception as exc:  # keep going on a bad PDF
                log.exception("Failed to process %s: %s", f, exc)
        return results

    def save_result(self, result: ExtractionResult) -> Path:
        out = self.settings.path("extracted_dir") / f"{result.paper.paper_id}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return out

    # ------------------------------------------------------------------
    def extract(self, doc: ParsedDocument) -> ExtractionResult:
        s = self.settings
        pid = doc.paper.paper_id
        skip = set(s.extraction.skip_sections)
        full_text = " ".join(x.text for x in doc.sentences if x.section_type not in skip)
        abbreviations = find_abbreviations(full_text)
        warnings = list(doc.warnings)

        # 1) rule-based entities and values
        analyses: dict[str, SentenceAnalysis] = {}
        for sent in doc.sentences:
            if sent.section_type in skip:
                continue
            analyses[sent.sentence_id] = self.rule_entities.analyze(sent, pid, abbreviations)

        # 2) LLM proposals (verified against the text)
        llm_rel: list[Relation] = []
        llm_findings: list[Finding] = []
        dropped: list[dict] = []
        calls = failures = 0
        if self.llm_extractor is not None:
            out = self.llm_extractor.extract(doc, analyses, sorted(skip), s.extraction.chunk_chars)
            llm_rel, llm_findings, dropped = out.relations, out.findings, out.dropped
            calls, failures = out.calls, out.failures
            if failures:
                warnings.append(f"{failures} LLM chunk(s) returned unusable JSON after retries (see logs)")

        # 3) rule-based relations (after LLM so that LLM-found mentions/values are included)
        rrx = RuleRelationExtractor(pid)
        methods_procs = [m for an in analyses.values() for m in an.mentions
                         if m.entity_type == "Process" and an.sentence.section_type != "introduction"]
        rule_rel: list[Relation] = []
        prev: list[SentenceAnalysis] = []
        for sent in doc.sentences:
            an = analyses.get(sent.sentence_id)
            if an is None:
                continue
            rule_rel += rrx.effect_relations(an)
            rule_rel += rrx.structural_relations(an, prev, methods_procs)
            prev.append(an)
        if self.mode == "llm":  # pure-LLM mode (for comparing extractors): no rule relations
            rule_rel = []

        mentions: list[EntityMention] = [m for an in analyses.values() for m in an.mentions]

        # 4) derived relations
        derived = paper_level_relations(doc, mentions)
        derived += table_trend_relations(doc, self.rule_entities.matcher, self.normalizer)

        # 5) merge + QC
        relations = consolidate_context_claims(merge_relations(rule_rel + llm_rel + derived))
        index = EvidenceIndex(doc)
        for r in relations:
            self.qc.assess(r, index)

        # 6) findings (verbatim sentences)
        findings: dict[str, Finding] = {}
        for f in rule_findings(pid, analyses, relations) + llm_findings:
            if f.finding_id in findings:
                g = findings[f.finding_id]
                g.method = "rule+llm" if g.method != f.method else g.method
                g.confidence = round(min(0.98, 1 - (1 - g.confidence) * (1 - f.confidence)), 3)
                g.related_claim_ids = sorted(set(g.related_claim_ids) | set(f.related_claim_ids))
            else:
                findings[f.finding_id] = f
        for f in findings.values():
            self.qc.assess_finding(f, index)

        # 7) canonical entity records used by this paper
        canon: dict[str, CanonicalEntity] = {}
        for m in mentions:
            if not m.canonical_id:
                continue
            ent = self.normalizer.entity(m.canonical_id)
            if ent is None:
                continue
            rec = canon.setdefault(m.canonical_id, ent.model_copy(deep=True))
            if not rec.in_lexicon:
                rec.normalization_confidence = min(rec.normalization_confidence, m.normalization_confidence)
            if m.text != rec.canonical_name and m.text not in rec.aliases:
                rec.aliases.append(m.text)

        stats = ExtractionStats(
            num_sentences=len(analyses), num_mentions=len(mentions), num_relations=len(relations),
            accepted=sum(r.status == ClaimStatus.ACCEPTED for r in relations),
            review=sum(r.status == ClaimStatus.REVIEW for r in relations),
            rejected=sum(r.status == ClaimStatus.REJECTED for r in relations),
            num_findings=len(findings), llm_calls=calls, llm_failures=failures)
        prov = Provenance(
            pipeline_version=PIPELINE_VERSION, extraction_mode=self.mode,
            llm_model=self.llm.model if self.llm and self.mode != "rule" else None,
            prompt_hashes=self.prompts.hashes() if self.mode != "rule" else {},
            lexicon_hash=self.lexicon.file_hash,
            thresholds={"accept": s.quality.accept_threshold, "review": s.quality.review_threshold})
        result = ExtractionResult(paper=doc.paper, entities=mentions, canonical_entities=canon,
                                  relations=relations, findings=sorted(findings.values(), key=lambda f: f.finding_id),
                                  stats=stats, provenance=prov, warnings=warnings)
        if dropped:
            dump = s.path("extracted_dir") / "llm_audit"
            dump.mkdir(exist_ok=True)
            (dump / f"{pid}_dropped.json").write_text(json.dumps(dropped, indent=2, ensure_ascii=False), encoding="utf-8")
        return result


def load_result(path: Path) -> ExtractionResult:
    return ExtractionResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_all_results(extracted_dir: Path) -> list[ExtractionResult]:
    return [load_result(p) for p in sorted(Path(extracted_dir).glob("paper_*.json"))]
