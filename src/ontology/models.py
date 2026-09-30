"""Pydantic data models shared by every pipeline stage.

All extracted knowledge carries an Evidence object so that any fact in the
graph can be traced back to paper -> page -> section -> sentence/table row.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from .schema import ClaimStatus, KnowledgeType


def stable_id(prefix: str, *parts: Any, length: int = 12) -> str:
    """Deterministic id so re-running the pipeline yields identical ids."""
    h = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:length]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Parsed document
# --------------------------------------------------------------------------
class PaperMetadata(BaseModel):
    paper_id: str
    title: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    source_file: Optional[str] = None
    abstract: Optional[str] = None
    metadata_source: str = "heuristic"      # heuristic | grobid | pdf_metadata | manual
    is_synthetic: bool = False              # True for DEMO / SYNTHETIC documents


class Sentence(BaseModel):
    sentence_id: str
    text: str
    page: Optional[int] = None               # 1-based page number
    section_id: str
    section_type: str
    section_title: str
    index: int                               # position within the document
    source_type: str = "text"                # text | table | caption
    table_id: Optional[str] = None


class Section(BaseModel):
    section_id: str
    title: str
    section_type: str
    page_start: Optional[int] = None
    text: str = ""
    sentence_ids: list[str] = Field(default_factory=list)


class Table(BaseModel):
    table_id: str
    page: Optional[int] = None
    caption: Optional[str] = None
    header: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    extractor: str = "pymupdf"
    note: Optional[str] = None


class ParsedDocument(BaseModel):
    paper: PaperMetadata
    parser: str
    num_pages: int = 0
    sections: list[Section] = Field(default_factory=list)
    sentences: list[Sentence] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)

    def sentence_map(self) -> dict[str, Sentence]:
        return {s.sentence_id: s for s in self.sentences}


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
class Quantity(BaseModel):
    raw: str                                  # exactly as written in the paper
    value: Optional[float] = None
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    uncertainty: Optional[float] = None
    operator: Optional[str] = None            # <, >, ~, <=, >=
    unit: Optional[str] = None                # original unit string
    normalized_value: Optional[float] = None
    normalized_min: Optional[float] = None
    normalized_max: Optional[float] = None
    normalized_uncertainty: Optional[float] = None
    normalized_unit: Optional[str] = None
    dimension: Optional[str] = None


class Evidence(BaseModel):
    paper_id: str
    page: Optional[int] = None
    section: Optional[str] = None            # section title as written
    section_type: Optional[str] = None
    sentence_id: Optional[str] = None
    text: str                                # verbatim source sentence / table row
    source_type: str = "text"                # text | table | caption
    table_id: Optional[str] = None
    # extra sentences needed to support the claim (e.g. process named in the
    # previous sentence). `text` is then the verbatim concatenation.
    context_sentence_ids: list[str] = Field(default_factory=list)


class EntityMention(BaseModel):
    mention_id: str
    text: str                                 # surface form in the paper
    entity_type: str
    canonical_id: Optional[str] = None
    canonical_name: Optional[str] = None
    normalization_confidence: float = 0.0
    normalization_method: Optional[str] = None
    sentence_id: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quantity: Optional[Quantity] = None       # value attached to the mention (if any)
    confidence: float = 0.0
    method: str = "rule"                      # rule | llm | rule+llm | manual
    evidence: Optional[Evidence] = None


class EntityRef(BaseModel):
    """Endpoint of a relation."""
    id: str                                   # canonical id, or value-node id
    name: str                                 # canonical name / raw value text
    type: str
    surface: Optional[str] = None             # text as it appeared in the sentence
    normalization_confidence: float = 1.0
    quantity: Optional[Quantity] = None       # only for value nodes


class Relation(BaseModel):
    """A single evidence-backed claim (stored as a :Claim node in Neo4j)."""
    claim_id: str
    subject: EntityRef
    relation: str
    object: EntityRef
    evidence: Optional[Evidence] = None
    confidence: float = 0.0
    method: str = "rule"
    knowledge_type: KnowledgeType = KnowledgeType.EXTRACTED
    status: ClaimStatus = ClaimStatus.REVIEW
    qc_reasons: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    # e.g. trigger, claim_type (finding|investigated|setup), subject_change,
    # object_change, trend (positive|negative|unspecified), negated, hedged,
    # context_resolution, llm_evidence_quote
    reviewed: bool = False
    review_note: Optional[str] = None

    def triple(self) -> tuple[str, str, str]:
        return (self.subject.name, self.relation, self.object.name)


class Finding(BaseModel):
    finding_id: str
    text: str                                 # verbatim sentence (EXPLICIT) from the paper
    evidence: Evidence
    finding_type: str = "result"              # result | comparison | trend | null_result | conclusion
    related_claim_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    method: str = "rule"
    knowledge_type: KnowledgeType = KnowledgeType.EXPLICIT
    status: ClaimStatus = ClaimStatus.REVIEW


class CanonicalEntity(BaseModel):
    id: str
    entity_type: str
    canonical_name: str
    display_name: Optional[str] = None
    aliases: list[str] = Field(default_factory=list)
    family: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    normalization_confidence: float = 1.0
    in_lexicon: bool = True


class ExtractionStats(BaseModel):
    num_sentences: int = 0
    num_mentions: int = 0
    num_relations: int = 0
    accepted: int = 0
    review: int = 0
    rejected: int = 0
    num_findings: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    seconds: float = 0.0
    # Speed / cost accounting (Phase 1). Defaults keep old JSON files loadable.
    llm_seconds: float = 0.0
    llm_cache_hits: int = 0
    llm_prompt_tokens: int = 0
    llm_output_tokens: int = 0
    llm_chunks: int = 0            # chunks considered for the LLM
    llm_chunks_skipped: int = 0    # ...of which skipped by rule-first gating
    llm_by_stage: dict[str, Any] = Field(default_factory=dict)
    llm_by_provider: dict[str, int] = Field(default_factory=dict)   # calls per provider (fallback chain)
    gliner_seconds: float = 0.0
    gliner_mentions: int = 0


class Provenance(BaseModel):
    pipeline_version: str
    extraction_mode: str
    llm_model: Optional[str] = None
    entity_backend: Optional[str] = None     # which extractor proposed non-lexicon entities: llm | gliner | none
    gliner_model: Optional[str] = None
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    lexicon_hash: Optional[str] = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utcnow)


class ExtractionResult(BaseModel):
    paper: PaperMetadata
    entities: list[EntityMention] = Field(default_factory=list)
    canonical_entities: dict[str, CanonicalEntity] = Field(default_factory=dict)
    relations: list[Relation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    stats: ExtractionStats = Field(default_factory=ExtractionStats)
    provenance: Optional[Provenance] = None
    warnings: list[str] = Field(default_factory=list)
