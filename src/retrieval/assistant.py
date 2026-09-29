"""Graph-RAG research assistant.

    question -> entity linking (lexicon + KB) -> graph claims (primary evidence)
             + passage retrieval (BM25 / Chroma)  -> numbered evidence
             -> local Qwen answer that must cite evidence ids
             -> citation check; fallback to a deterministic graph-only answer

The LLM never sees anything but the retrieved evidence, and its optional
mechanistic explanation is returned separately, labelled INTERPRETED.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import Settings, get_settings
from ..entity_extraction.matcher import LexiconMatcher
from ..knowledge_graph.kb import KnowledgeBase
from ..llm.client import LLMError, LLMProvider, OllamaClient
from ..llm.prompts import PromptLibrary
from ..normalization.lexicon import Lexicon
from ..normalization.text import match_key
from ..ontology.models import Relation
from .vector_index import BM25Index, ChromaIndex, Passage, build_passages, load_docs

log = logging.getLogger(__name__)

_TARGET_WORDS = [
    (r"\bparameters?\b|\bsettings?\b", "Parameter"),
    (r"\bdefects?\b", "Defect"),
    (r"\bpropert(y|ies)\b", "Property"),
    (r"\bmicrostructur", "Microstructure"),
    (r"\bprocess(es)?\b(?! parameter)", "Process"),
    (r"\bmaterials?\b|\balloys?\b", "Material"),
    (r"\bpost[- ]?process|\bheat treatments?\b", "PostProcessing"),
    (r"\bcharacteri[sz]ation|\btechniques?\b|\bmethods?\b", "CharacterizationMethod"),
]
_VALUE_WORDS = re.compile(r"\b(values?|how (much|high|low)|reported|range|magnitude|what (is|was|were) the)\b", re.I)
CONTEXT_TYPES = {"Material", "Process"}


@dataclass
class LinkedEntity:
    id: str
    name: str
    type: str
    surface: str


@dataclass
class ParsedQuestion:
    text: str
    entities: list[LinkedEntity] = field(default_factory=list)
    target_types: list[str] = field(default_factory=list)
    wants_values: bool = False

    @property
    def focus(self) -> list[LinkedEntity]:
        return [e for e in self.entities if e.type not in CONTEXT_TYPES]

    @property
    def context(self) -> list[LinkedEntity]:
        return [e for e in self.entities if e.type in CONTEXT_TYPES]


@dataclass
class EvidenceItem:
    eid: str
    kind: str                     # claim | passage
    text: str
    paper_id: str
    title: Optional[str]
    page: Optional[int]
    section: Optional[str]
    claim: Optional[dict[str, Any]] = None
    score: float = 0.0
    synthetic: bool = False
    context_match: Optional[bool] = None   # does the source paper match the question's material/process?
    target_match: Optional[bool] = None    # is the other endpoint of the asked-for type?


@dataclass
class AssistantAnswer:
    question: str
    answer: str
    interpretation: Optional[str]
    mode: str                     # llm | graph_only
    linked_entities: list[LinkedEntity]
    evidence: list[EvidenceItem]
    cited_ids: list[str]
    warnings: list[str] = field(default_factory=list)


class ResearchAssistant:
    def __init__(self, kb: KnowledgeBase, settings: Optional[Settings] = None,
                 llm: Optional[LLMProvider] = None, use_llm: bool = True, embed_fn: Any = None) -> None:
        self.settings = settings or get_settings()
        self.kb = kb
        self.lexicon = Lexicon.load(self.settings.path("lexicon_file"))
        self.matcher = LexiconMatcher(self.lexicon)
        self.prompts = PromptLibrary(self.settings.path("prompts_dir"))
        self.llm = llm
        if self.llm is None and use_llm:
            client = OllamaClient.from_settings(self.settings)
            self.llm = client if client.is_available() else None
        docs = load_docs(self.settings.path("processed_dir"))
        docs = [d for d in docs if d.paper.paper_id in kb.papers] or docs
        self.passages = build_passages(docs, self.settings.retrieval.passage_window)
        self.index: Any = BM25Index(self.passages)
        if self.settings.retrieval.vector_backend == "chroma":
            try:
                embed = embed_fn or OllamaClient.from_settings(self.settings).embed
                self.index = ChromaIndex(self.passages, self.settings.path("index_dir") / "chroma", embed)
            except Exception as exc:
                log.warning("Chroma unavailable (%s); using BM25", exc)

    # ------------------------------------------------------------------
    def understand(self, question: str) -> ParsedQuestion:
        pq = ParsedQuestion(text=question, wants_values=bool(_VALUE_WORDS.search(question)))
        seen: set[str] = set()
        for m in self.matcher.find(question):
            cid = self.kb.resolve_id(m.canonical_id)
            if cid not in seen:
                seen.add(cid)
                pq.entities.append(LinkedEntity(cid, self.kb.entity_name(cid), m.entity_type, m.text))
        qk = match_key(question)
        for cid, e in self.kb.entities.items():  # entities created from papers (not in lexicon)
            if cid in seen or e.in_lexicon or len(e.canonical_name) < 5:
                continue
            if match_key(e.canonical_name) in qk:
                seen.add(cid)
                pq.entities.append(LinkedEntity(cid, e.canonical_name, e.entity_type, e.canonical_name))
        mentioned = {e.type for e in pq.entities}
        for pat, t in _TARGET_WORDS:
            if re.search(pat, question, re.I) and t not in pq.target_types:
                # "which parameters affect porosity": Parameter is a target, not a mention
                if t in mentioned and t not in ("Parameter", "Property", "Defect"):
                    continue
                pq.target_types.append(t)
        return pq

    def graph_evidence(self, pq: ParsedQuestion) -> list[tuple[Relation, float]]:
        focus = {e.id for e in pq.focus}
        context = {e.id for e in pq.context}
        anchor = focus or context
        if not anchor:
            return []
        scored = []
        for c in self.kb.active_claims():
            if c.relation in ("STUDIES", "USES_PROCESS"):
                continue
            if c.relation == "HAS_VALUE" and not pq.wants_values:
                continue
            ends = {c.subject.id, c.object.id}
            hit = ends & anchor
            if not hit:
                continue
            score = 3.0 * len(hit)
            other = c.object if c.subject.id in hit else c.subject
            if pq.target_types:
                score += 1.5 if other.type in pq.target_types else -1.0
            if context and c.evidence:
                pctx = self.kb.paper_context(c.evidence.paper_id)
                have = pctx["materials"] | pctx["processes"]
                score += 2.0 if context <= have else (-1.5 if focus else 0.0)
            score += c.confidence + (0.5 if c.status.value == "accepted" else 0.0)
            score += 0.5 if c.knowledge_type.value == "EXPLICIT" else 0.0
            if c.attributes.get("claim_origin") == "background":
                score -= 1.0
            scored.append((c, score))
        scored.sort(key=lambda x: -x[1])
        return scored[: self.settings.retrieval.top_k_claims]

    def passage_evidence(self, pq: ParsedQuestion) -> list[tuple[Passage, float]]:
        terms = [pq.text] + [e.name for e in pq.entities] + [e.surface for e in pq.entities]
        ctx_papers = None
        if pq.context:
            ids = {e.id for e in pq.context}
            ctx_papers = {pid for pid in self.kb.papers
                          if ids <= (lambda c: c["materials"] | c["processes"])(self.kb.paper_context(pid))}

        def boost(p: Passage) -> float:
            b = 1.0
            if ctx_papers is not None:
                b *= 1.5 if p.paper_id in ctx_papers else 0.7
            if p.section_type in ("results", "results_discussion", "discussion", "conclusion", "abstract"):
                b *= 1.2
            return b
        return self.index.search(" ".join(terms), k=self.settings.retrieval.top_k_passages, boost=boost)

    # ------------------------------------------------------------------
    def answer(self, question: str) -> AssistantAnswer:
        pq = self.understand(question)
        warnings: list[str] = []
        if not pq.entities:
            warnings.append("No known AM entities were recognised in the question; results rely on text search only.")
        items: list[EvidenceItem] = []
        used_sents: set[str] = set()
        ctx_ids = {e.id for e in pq.context}
        focus_ids = {e.id for e in pq.focus} or ctx_ids
        for i, (c, sc) in enumerate(self.graph_evidence(pq), start=1):
            p = self.kb.papers.get(c.evidence.paper_id) if c.evidence else None
            cmatch = None
            if ctx_ids and c.evidence:
                pc = self.kb.paper_context(c.evidence.paper_id)
                cmatch = ctx_ids <= (pc["materials"] | pc["processes"])
            other = c.object if c.subject.id in focus_ids else c.subject
            tmatch = (other.type in pq.target_types) if pq.target_types else None
            items.append(EvidenceItem(
                eid=f"E{i}", kind="claim", text=c.evidence.text if c.evidence else "", paper_id=c.evidence.paper_id if c.evidence else "",
                title=p.title if p else None, page=c.evidence.page if c.evidence else None,
                section=c.evidence.section if c.evidence else None, score=round(sc, 2), synthetic=bool(p and p.is_synthetic),
                claim={"claim_id": c.claim_id, "subject": c.subject.name, "relation": c.relation, "object": c.object.name,
                       "knowledge_type": c.knowledge_type.value, "status": c.status.value, "confidence": c.confidence,
                       "trend": c.attributes.get("trend"), "negated": c.attributes.get("negated"),
                       "hedged": c.attributes.get("hedged"), "origin": c.attributes.get("claim_origin")},
                context_match=cmatch, target_match=tmatch))
            if c.evidence and c.evidence.sentence_id:
                used_sents.add(c.evidence.sentence_id)
        j = 0
        for p, sc in self.passage_evidence(pq):
            if set(p.sentence_ids) <= used_sents:
                continue
            j += 1
            items.append(EvidenceItem(eid=f"P{j}", kind="passage", text=p.text, paper_id=p.paper_id, title=p.title,
                                      page=p.page, section=p.section, score=round(sc, 2), synthetic=p.is_synthetic))
        if any(it.synthetic for it in items):
            warnings.append("Some evidence comes from DEMO / SYNTHETIC documents and is not real research.")
        if not items:
            return AssistantAnswer(question, "The current corpus does not contain evidence related to this question.",
                                   None, "graph_only", pq.entities, [], [], warnings)
        if self.llm is not None:
            try:
                return self._llm_answer(pq, items, warnings)
            except LLMError as exc:
                warnings.append(f"LLM unavailable ({exc}); showing graph-only answer.")
        return self._graph_only(pq, items, warnings)

    def _format_evidence(self, items: list[EvidenceItem]) -> str:
        lines = []
        for it in items:
            src = f"{it.paper_id}{' (SYNTHETIC)' if it.synthetic else ''}, p.{it.page}, {it.section}"
            if it.claim:
                c = it.claim
                flags = [c["knowledge_type"], f"status={c['status']}"]
                if c.get("trend") and c["trend"] not in ("unspecified",):
                    flags.append(f"trend={c['trend']}")
                if c.get("negated"):
                    flags.append("NEGATED (no effect)")
                if c.get("hedged"):
                    flags.append("hedged")
                if c.get("origin") == "background":
                    flags.append("background/cited work")
                if it.context_match is False:
                    flags.append("DIFFERENT material/process context than asked")
                lines.append(f"[{it.eid}] CLAIM {c['subject']} -{c['relation']}-> {c['object']} ({', '.join(flags)}) "
                             f"| {src} | \"{it.text[:400]}\"")
            else:
                lines.append(f"[{it.eid}] PASSAGE | {src} | \"{it.text[:500]}\"")
        return "\n".join(lines)

    def _llm_answer(self, pq: ParsedQuestion, items: list[EvidenceItem], warnings: list[str]) -> AssistantAnswer:
        assert self.llm is not None
        user = self.prompts.render("answer_generation", question=pq.text, evidence=self._format_evidence(items))
        raw = self.llm.chat(self.prompts.raw("system_answer"), user, json_mode=False).strip()
        body, _, interp = raw.partition("Interpretation:")
        valid = {it.eid for it in items}
        cited = sorted(set(re.findall(r"\[([EP]\d+)\]", body)))
        bad = [c for c in cited if c not in valid]
        if bad:
            warnings.append(f"Removed citations to non-existent evidence: {bad}")
            for b in bad:
                body = body.replace(f"[{b}]", "")
        cited = [c for c in cited if c in valid]
        if not cited:
            warnings.append("The LLM answer cited no evidence; showing the graph-only answer instead.")
            return self._graph_only(pq, items, warnings)
        interp = re.sub(r"\[[EP]\d+\]", "", interp).strip() or None
        return AssistantAnswer(pq.text, body.strip(), interp, "llm", pq.entities, items, cited, warnings)

    def _graph_only(self, pq: ParsedQuestion, items: list[EvidenceItem], warnings: list[str]) -> AssistantAnswer:
        sections: dict[str, dict[tuple, list[EvidenceItem]]] = {
            "Direct answer (matching entity type and context)": defaultdict(list),
            "Related evidence (other entity types)": defaultdict(list),
            "Evidence from OTHER material/process contexts (may not transfer)": defaultdict(list)}
        names = list(sections)
        for it in items:
            if it.claim:
                c = it.claim
                key = (c["subject"], c["relation"], c["object"], c.get("trend") or "", bool(c.get("negated")),
                       c["knowledge_type"], c["status"])
                if it.context_match is False:
                    sections[names[2]][key].append(it)
                elif it.target_match is False:
                    sections[names[1]][key].append(it)
                else:
                    sections[names[0]][key].append(it)
        lines = ["Graph-only answer (deterministic summary of evidence-backed claims; no LLM used):"]
        for title, groups in sections.items():
            if not groups:
                continue
            lines.append(f"{title}:")
            for (s, r, o, trend, neg, kt, st), its in sorted(groups.items(), key=lambda kv: -len({i.paper_id for i in kv[1]})):
                papers = sorted({i.paper_id for i in its})
                desc = f"- {s} {'does NOT ' + r.lower().rstrip('s') if neg else r} {o}"
                if trend and trend not in ("unspecified", "none"):
                    desc += f" (trend: {trend})"
                desc += f" [{kt}{', under review' if st == 'review' else ''}; {len(papers)} paper(s)] "
                desc += "".join(f"[{i.eid}]" for i in its[:5])
                lines.append(desc)
        groups = {k: v for g in sections.values() for k, v in g.items()}
        passages = [it for it in items if it.kind == "passage"]
        if passages:
            lines.append("Related passages: " + " ".join(f"[{p.eid}]" for p in passages))
        if len(lines) == 1:
            lines.append("No graph claims matched; see the retrieved passages.")
        cited = sorted({i.eid for its in groups.values() for i in its} | {p.eid for p in passages})
        return AssistantAnswer(pq.text, "\n".join(lines), None, "graph_only", pq.entities, items, cited, warnings)
