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
from .entity_extraction.gliner_extractor import GlinerEntityExtractor
from .entity_extraction.llm_extractor import LLMExtractor
from .entity_extraction.rule_based import RuleEntityExtractor, SentenceAnalysis
from .evidence.verify import EvidenceIndex
from .ingestion.loader import discover_inputs, parse_document, save_parsed
from .llm.client import LLMError, LLMProvider
from .llm.instrumented import InstrumentedLLM
from .llm.openai_compat import make_llm_client
from .llm.prompts import PromptLibrary
from .normalization.entity_normalizer import (
    EntityNormalizer,
    NormalizationOverrides,
    find_abbreviations,
    find_symbol_definitions,
)
from .normalization.lexicon import Lexicon
from .normalization.process_context import resolve_generic_processes
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
# rule   = lexicon + regex only (fast, deterministic)
# gliner = rule + GLiNER entities, no LLM (fast baseline, no findings/LLM relations)
# hybrid = rule + entities (GLiNER or LLM) + LLM relations + LLM findings
# llm    = LLM only (for comparing extractors)
VALID_MODES = {"rule", "gliner", "llm", "hybrid", "auto"}


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
                 mode: Optional[str] = None, gliner: Optional[GlinerEntityExtractor] = None) -> None:
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
            try:
                client = make_llm_client(s)
                available = client.is_available()
            except LLMError as exc:          # e.g. cloud provider selected but no API key in .env
                if requested != "auto":
                    raise RuntimeError(str(exc)) from exc
                log.warning("%s -> falling back to rule mode", exc)
                client, available = None, False
            if available:
                self.llm = client
            elif requested != "auto":
                if s.llm.provider == "ollama":
                    raise RuntimeError(f"Ollama model '{s.llm.model}' not reachable at {s.llm.base_url}. "
                                       f"Start Ollama and run: ollama pull {s.llm.model} (or use --mode rule)")
                raise RuntimeError(f"LLM provider '{s.llm.provider}' not reachable (check the API key, "
                                   f"llm.model and your internet connection)")
        self.mode = ("hybrid" if self.llm else "rule") if requested == "auto" else requested
        if self.llm is not None and not isinstance(self.llm, InstrumentedLLM):
            # measure every call (time, tokens) and reuse answers for identical prompts
            self.llm = InstrumentedLLM(self.llm, s.path("cache_dir") if s.llm.cache else None,
                                       stage_max_tokens=s.llm.num_predict_by_stage)
        self.rel_llm: Optional[InstrumentedLLM] = None
        if self.llm and self.mode in ("llm", "hybrid") and s.extraction.relation_llm:
            from .llm.openai_compat import parse_llm_spec
            rel_client = parse_llm_spec(s.extraction.relation_llm, s)
            self.rel_llm = InstrumentedLLM(rel_client, s.path("cache_dir") if s.llm.cache else None)
            log.info("Relations use a separate model: %s", s.extraction.relation_llm)
        self.llm_extractor = None
        if self.llm and self.mode in ("llm", "hybrid"):
            self.llm_extractor = LLMExtractor(self.llm, self.lexicon, self.normalizer, self.prompts,
                                              s.llm.max_retries, s.extraction.llm_findings,
                                              relation_llm=self.rel_llm)
            if s.extraction.llm_normalization:
                self.normalizer.llm_chooser = self.llm_extractor.choose_canonical
        self.gliner = gliner
        self.entity_backend = self._resolve_entity_backend()
        log.info("Extraction mode: %s%s | entities: %s", self.mode,
                 f" (model {self.llm.model})" if self.llm else "", self.entity_backend)

    def _resolve_entity_backend(self) -> str:
        """Decide who proposes entities beyond the lexicon: gliner | llm | none."""
        if self.mode == "rule":
            return "none"
        wanted = self.settings.extraction.entity_backend.lower()
        if self.mode == "gliner":
            wanted = "gliner"
        elif self.mode == "llm":      # pure-LLM comparison mode: the LLM does everything
            wanted = "llm"
        if wanted not in {"auto", "gliner", "llm", "none"}:
            raise ValueError("extraction.entity_backend must be auto | gliner | llm | none")
        if wanted in ("auto", "gliner") and self.gliner is None:
            g = self.settings.gliner
            try:
                self.gliner = GlinerEntityExtractor.load(
                    g.model, self.normalizer, self.rule_entities, g.threshold, g.labels or None, g.device, g.batch_size,
                    set(g.skip_sections))
            except Exception as exc:   # not installed, no internet for the first download, ...
                if wanted == "gliner":
                    raise RuntimeError(f"GLiNER is not available ({exc}). Install it with: pip install gliner "
                                       f"(the model downloads on first use), or set extraction.entity_backend: llm"
                                       ) from exc
                log.warning("GLiNER not available (%s); using the LLM entity prompt instead", exc)
        if wanted in ("auto", "gliner") and self.gliner is not None:
            return "gliner"
        if wanted in ("auto", "llm") and self.llm_extractor is not None:
            return "llm"
        return "none"

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
        if isinstance(self.llm, InstrumentedLLM):
            # per-paper accounting; reset FIRST so LLM normalisation calls made while
            # GLiNER/rule entities are normalised are counted too
            self.llm.reset_stats()
        if self.rel_llm is not None:
            self.rel_llm.reset_stats()
        pid = doc.paper.paper_id
        skip = set(s.extraction.skip_sections)
        full_text = " ".join(x.text for x in doc.sentences if x.section_type not in skip)
        abbreviations = find_abbreviations(full_text)
        # symbols such as P, v, h, E are resolved during normalisation only (never matched on their own)
        self.normalizer.set_paper_context(abbreviations, find_symbol_definitions(full_text))
        warnings = list(doc.warnings)

        # 1) rule-based entities and values
        analyses: dict[str, SentenceAnalysis] = {}
        for sent in doc.sentences:
            if sent.section_type in skip:
                continue
            analyses[sent.sentence_id] = self.rule_entities.analyze(sent, pid, abbreviations)

        # 1b) GLiNER entities (fast local NER; replaces the LLM entity prompt)
        gliner_seconds, gliner_mentions = 0.0, 0
        gl_dropped: list[dict] = []
        if self.entity_backend == "gliner" and self.gliner is not None:
            g = self.gliner.extract(doc, analyses, skip | set(s.extraction.llm_skip_sections))
            gliner_seconds, gl_dropped = g.seconds, g.dropped
            gliner_mentions = sum(1 for m in g.mentions if m.method == "gliner")

        # 1c) "the printing process", "fabrication", "the metal printer" -> the paper's main process
        resolve_generic_processes(analyses, self.lexicon)

        # 2) LLM proposals (verified against the text)
        llm_rel: list[Relation] = []
        llm_findings: list[Finding] = []
        dropped: list[dict] = []
        calls = failures = chunks = chunks_skipped = 0
        if self.llm_extractor is not None:
            llm_skip = sorted(skip | set(s.extraction.llm_skip_sections))
            out = self.llm_extractor.extract(doc, analyses, llm_skip, s.extraction.chunk_chars,
                                             gate=s.extraction.llm_gate,
                                             llm_entities=self.entity_backend == "llm")
            llm_rel, llm_findings, dropped = out.relations, out.findings, out.dropped
            calls, failures = out.calls, out.failures
            chunks, chunks_skipped = out.chunks, out.chunks_skipped
            if failures:
                warnings.append(f"{failures} LLM chunk(s) returned unusable JSON after retries (see logs)")

        # 3) rule-based relations (after LLM so that LLM-found mentions/values are included)
        # only lexicon processes (incl. remapped generic mentions) anchor structural relations;
        # "remelting" or "Marangoni flow" typed as Process must not collect HAS_PARAMETER claims
        def process_ok(m: EntityMention) -> bool:
            return m.canonical_id in self.lexicon.entities

        rrx = RuleRelationExtractor(pid, process_ok=process_ok)
        methods_procs = [m for an in analyses.values() for m in an.mentions
                         if m.entity_type == "Process" and an.sentence.section_type != "introduction"
                         and process_ok(m)]
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
            if (m.text != rec.canonical_name and m.text not in rec.aliases
                    and m.normalization_method != "paper_main_process"):   # "printing" is no LPBF alias
                rec.aliases.append(m.text)

        stats = ExtractionStats(
            num_sentences=len(analyses), num_mentions=len(mentions), num_relations=len(relations),
            accepted=sum(r.status == ClaimStatus.ACCEPTED for r in relations),
            review=sum(r.status == ClaimStatus.REVIEW for r in relations),
            rejected=sum(r.status == ClaimStatus.REJECTED for r in relations),
            num_findings=len(findings), llm_calls=calls, llm_failures=failures,
            llm_chunks=chunks, llm_chunks_skipped=chunks_skipped,
            gliner_seconds=gliner_seconds, gliner_mentions=gliner_mentions)
        if isinstance(self.llm, InstrumentedLLM):
            usage = self.llm.summary()   # includes LLM normalization calls made above
            stats.llm_seconds = usage["seconds"]
            stats.llm_cache_hits = usage["cache_hits"]
            stats.llm_prompt_tokens = usage["prompt_tokens"]
            stats.llm_output_tokens = usage["output_tokens"]
            stats.llm_by_stage = usage["by_stage"]
            stats.llm_by_provider = usage.get("by_provider", {})
        if self.rel_llm is not None:          # add the separate relation model's usage
            ru = self.rel_llm.summary()
            stats.llm_seconds = round(stats.llm_seconds + ru["seconds"], 2)
            stats.llm_cache_hits += ru["cache_hits"]
            stats.llm_prompt_tokens += ru["prompt_tokens"]
            stats.llm_output_tokens += ru["output_tokens"]
            stats.llm_by_stage = {**stats.llm_by_stage, **ru["by_stage"]}
            for k, v in ru.get("by_provider", {}).items():
                stats.llm_by_provider[k] = stats.llm_by_provider.get(k, 0) + v
        prov = Provenance(
            pipeline_version=PIPELINE_VERSION, extraction_mode=self.mode,
            llm_model=self.llm.model if self.llm and self.mode not in ("rule", "gliner") else None,
            relation_llm_model=(f"{self.rel_llm.inner.provider}:{self.rel_llm.model}" if self.rel_llm else None),
            entity_backend=self.entity_backend,
            gliner_model=self.gliner.model_name if self.entity_backend == "gliner" and self.gliner else None,
            prompt_hashes=self.prompts.hashes() if self.mode not in ("rule", "gliner") else {},
            lexicon_hash=self.lexicon.file_hash,
            thresholds={"accept": s.quality.accept_threshold, "review": s.quality.review_threshold})
        result = ExtractionResult(paper=doc.paper, entities=mentions, canonical_entities=canon,
                                  relations=relations, findings=sorted(findings.values(), key=lambda f: f.finding_id),
                                  stats=stats, provenance=prov, warnings=warnings)
        dropped = gl_dropped + dropped
        if dropped:
            dump = s.path("extracted_dir") / "llm_audit"
            dump.mkdir(exist_ok=True)
            (dump / f"{pid}_dropped.json").write_text(json.dumps(dropped, indent=2, ensure_ascii=False), encoding="utf-8")
        return result


def load_result(path: Path) -> ExtractionResult:
    return ExtractionResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_all_results(extracted_dir: Path) -> list[ExtractionResult]:
    return [load_result(p) for p in sorted(Path(extracted_dir).glob("paper_*.json"))]
