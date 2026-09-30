"""GLiNER entity extraction: a fast local NER model instead of the LLM entity prompt.

GLiNER (Zaratiana et al., NAACL 2024) is a ~200M-parameter bidirectional
transformer that tags spans for any label names you give it ("zero-shot NER").
It reads text but does not *generate* text, so it is far faster than asking a
chat LLM to write a JSON list of entities, and it cannot hallucinate spans:
every entity is a character span of the sentence.

The same safeguards as for the LLM apply:
  * junk filter (src/quality/junk.py) for new, non-lexicon terms
  * normalisation to canonical entities (lexicon, aliases, fuzzy, ...)
  * values are attached by the rule-based, dimension-checked value assigner

Install (optional):  pip install gliner
The model is downloaded from Hugging Face on first use (~0.5 GB) and cached.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..normalization.entity_normalizer import EntityNormalizer
from ..ontology.models import EntityMention, ParsedDocument, stable_id
from ..quality.junk import junk_reason
from .rule_based import RuleEntityExtractor, SentenceAnalysis, evidence_for

log = logging.getLogger(__name__)
# "The weld", "This defect", "Such features", "their material properties" -> drop the determiner
_LEADING_DET = re.compile(r"^(the|a|an|this|that|these|those|such|their|its|our|his|her|each|both|all)\s+", re.I)

# GLiNER label (natural-language description) -> our ontology entity type.
DEFAULT_LABELS: dict[str, str] = {
    "material or alloy": "Material",
    "additive manufacturing process": "Process",
    "process parameter": "Parameter",
    "material property": "Property",
    "defect": "Defect",
    "microstructure feature": "Microstructure",
    "post-processing treatment": "PostProcessing",
    "characterization or testing method": "CharacterizationMethod",
    "specimen geometry": "Geometry",
    "experimental condition": "ExperimentalCondition",
}


@dataclass
class GlinerOutput:
    mentions: list[EntityMention] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    sentences: int = 0


class GlinerEntityExtractor:
    def __init__(self, model: Any, normalizer: EntityNormalizer, rule_extractor: RuleEntityExtractor,
                 threshold: float = 0.5, labels: Optional[dict[str, str]] = None, batch_size: int = 8,
                 model_name: str = "", skip_new_in: Optional[set[str]] = None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.rule_extractor = rule_extractor
        self.threshold = threshold
        self.labels = dict(labels or DEFAULT_LABELS)
        self.batch_size = batch_size
        self.model_name = model_name or getattr(model, "name", "gliner")
        self.skip_new_in = set(skip_new_in or ())

    @classmethod
    def load(cls, model_name: str, normalizer: EntityNormalizer, rule_extractor: RuleEntityExtractor,
             threshold: float = 0.5, labels: Optional[dict[str, str]] = None, device: str = "cpu",
             batch_size: int = 8, skip_new_in: Optional[set[str]] = None) -> "GlinerEntityExtractor":
        """Load the Hugging Face model. Raises ImportError if `gliner` is not installed."""
        from gliner import GLiNER  # optional dependency, imported lazily
        t0 = time.perf_counter()
        model = GLiNER.from_pretrained(model_name)
        if device and device != "cpu":
            model = model.to(device)
        log.info("GLiNER model %s loaded in %.1fs (%s)", model_name, time.perf_counter() - t0, device)
        return cls(model, normalizer, rule_extractor, threshold, labels, batch_size, model_name, skip_new_in)

    # ------------------------------------------------------------------
    def _predict(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        labels = list(self.labels)
        if hasattr(self.model, "inference"):          # gliner >= 0.2.2x (batch API)
            return self.model.inference(texts, labels, threshold=self.threshold, batch_size=self.batch_size)
        if hasattr(self.model, "batch_predict_entities"):   # older gliner versions
            out: list[list[dict[str, Any]]] = []
            for i in range(0, len(texts), self.batch_size):
                out += self.model.batch_predict_entities(texts[i:i + self.batch_size], labels,
                                                         threshold=self.threshold)
            return out
        return [self.model.predict_entities(t, labels, threshold=self.threshold) for t in texts]

    def extract(self, doc: ParsedDocument, analyses: dict[str, SentenceAnalysis], skip: set[str]) -> GlinerOutput:
        """Add GLiNER mentions to `analyses` (in place) and return what was added."""
        out = GlinerOutput()
        paper_id = doc.paper.paper_id
        # Table rows are flattened cell text ("Table 2, row 3: ...") - not prose, skip them.
        sents = [s for s in doc.sentences
                 if s.section_type not in skip and s.source_type != "table" and s.sentence_id in analyses]
        out.sentences = len(sents)
        if not sents:
            return out
        t0 = time.perf_counter()
        predictions = self._predict([s.text for s in sents])
        out.seconds = round(time.perf_counter() - t0, 2)

        for s, preds in zip(sents, predictions):
            an = analyses[s.sentence_id]
            added = False
            for p in sorted(preds, key=lambda e: -float(e.get("score", 0))):
                etype = self.labels.get(p.get("label", ""))
                start, end = int(p.get("start", -1)), int(p.get("end", -1))
                if etype is None or not (0 <= start < end <= len(s.text)):
                    continue
                det = _LEADING_DET.match(s.text[start:end])
                if det:
                    start += det.end()
                    if start >= end:
                        continue
                surface = s.text[start:end]
                score = min(float(p.get("score", 0.5)), 0.95)
                # junk filter BEFORE normalisation (which may ask the LLM to choose a canonical)
                in_lex = bool(self.normalizer.lexicon.lookup_any_type(surface)
                              or self.normalizer.resolve_symbol(surface))
                reason = None if in_lex else junk_reason(surface, etype, False)
                if reason:
                    out.dropped.append({"stage": "gliner_entity", "reason": reason, "text": surface,
                                        "type": etype, "sentence_id": s.sentence_id, "score": round(score, 3)})
                    continue
                overlapping = [m for m in an.mentions if m.char_start is not None
                               and m.char_start < end and start < (m.char_end or 0)]
                if overlapping:
                    # The rule / lexicon span wins on conflicts. If both found the same
                    # entity, that agreement raises the rule mention's confidence.
                    hit = self.normalizer.lexicon.lookup(surface, etype)
                    for m in overlapping:
                        if hit and m.canonical_id == hit[0] and "gliner" not in m.method:
                            m.method = f"{m.method}+gliner"
                            m.confidence = round(min(0.98, 1 - (1 - m.confidence) * (1 - score * 0.9)), 3)
                    continue
                if s.section_type in self.skip_new_in:
                    # only confirm lexicon terms here; do not create new entities
                    out.dropped.append({"stage": "gliner_entity", "reason": f"no_new_entities_in_{s.section_type}",
                                        "text": surface, "type": etype, "sentence_id": s.sentence_id,
                                        "score": round(score, 3)})
                    continue
                norm = self.normalizer.normalize(surface, etype)
                mention = EntityMention(
                    mention_id=stable_id("m", s.sentence_id, start, end, "gliner"), text=surface,
                    entity_type=norm.entity_type, canonical_id=norm.canonical_id,
                    canonical_name=norm.canonical_name, normalization_confidence=norm.confidence,
                    normalization_method=norm.method, sentence_id=s.sentence_id,
                    char_start=start, char_end=end,
                    confidence=round(0.5 * score + 0.5 * norm.confidence, 3),
                    method="gliner", evidence=evidence_for(s, paper_id))
                an.mentions.append(mention)
                out.mentions.append(mention)
                added = True
            if added:  # values ("200 W") are attached by the dimension-checked rule assigner
                an.assignments = self.rule_extractor.assign_values(an)
                for a in an.assignments:
                    m = an.mentions[a.mention_index]
                    if m.quantity is None:
                        m.quantity = a.qspan.quantity
        return out
