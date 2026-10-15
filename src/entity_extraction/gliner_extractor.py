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


# LPBF-specific descriptions: GLiNER is zero-shot, so the label wording steers what it tags.
LPBF_LABELS: dict[str, str] = {
    "metal alloy or powder material": "Material",
    "additive manufacturing process such as laser powder bed fusion": "Process",
    "laser process parameter such as laser power, scan speed, hatch spacing or layer thickness": "Parameter",
    "measured mechanical or physical property such as strength, hardness, density or roughness": "Property",
    "manufacturing defect such as pores, cracks or lack of fusion": "Defect",
    "microstructure feature such as grains, phases, precipitates or melt pools": "Microstructure",
    "post-processing treatment such as heat treatment, HIP or machining": "PostProcessing",
    "characterization or testing method such as SEM, XRD, CT or tensile testing": "CharacterizationMethod",
    "test specimen or part geometry": "Geometry",
    "build or test condition such as build orientation, atmosphere or preheating temperature": "ExperimentalCondition",
}
# Several descriptions per type (label -> type is many-to-one): the best-scoring label decides the type.
LPBF_MULTI_LABELS: dict[str, str] = {
    "metal alloy such as Ti-6Al-4V, 316L, AlSi10Mg or Inconel 718": "Material",
    "metal powder feedstock": "Material",
    "additive manufacturing process such as laser powder bed fusion or selective laser melting": "Process",
    "metal 3D printing or welding process": "Process",
    "laser process parameter such as laser power, scan speed, hatch spacing or layer thickness": "Parameter",
    "energy density or scan strategy": "Parameter",
    "laser beam setting such as spot size or wavelength": "Parameter",
    "mechanical property such as tensile strength, yield strength, elongation or hardness": "Property",
    "physical property such as density, surface roughness or residual stress": "Property",
    "fatigue or fracture property": "Property",
    "porosity or pore defect": "Defect",
    "crack or cracking defect": "Defect",
    "lack of fusion, keyhole or unmelted powder defect": "Defect",
    "grain structure such as columnar or equiaxed grains": "Microstructure",
    "phase or precipitate such as martensite, Laves phase or gamma prime": "Microstructure",
    "melt pool, dendrite, cell or fracture surface feature": "Microstructure",
    "heat treatment such as annealing, aging, stress relief or HIP": "PostProcessing",
    "surface finishing such as machining, polishing or shot peening": "PostProcessing",
    "microscopy method such as SEM, TEM, EBSD or optical microscopy": "CharacterizationMethod",
    "measurement or test method such as XRD, CT, tensile test or hardness test": "CharacterizationMethod",
    "test specimen such as tensile specimen, cube or cylinder": "Geometry",
    "part geometry such as lattice, thin wall or overhang": "Geometry",
    "build condition such as build orientation, atmosphere, argon or preheating temperature": "ExperimentalCondition",
    "test condition such as room temperature or strain rate": "ExperimentalCondition",
}
LABEL_SETS: dict[str, dict[str, str]] = {"default": DEFAULT_LABELS, "lpbf": LPBF_LABELS, "lpbf_multi": LPBF_MULTI_LABELS}
CONTEXTS = ("sentence", "neighbours")
MAX_WORDS = 150          # GLiNER truncates inputs at 384 tokens: longer sentences are split into windows
WINDOW_STRIDE = 110


@dataclass
class GlinerOutput:
    mentions: list[EntityMention] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    sentences: int = 0


class GlinerEntityExtractor:
    def __init__(self, model: Any, normalizer: EntityNormalizer, rule_extractor: RuleEntityExtractor,
                 threshold: float = 0.5, labels: Optional[dict[str, str]] = None, batch_size: int = 8,
                 model_name: str = "", skip_new_in: Optional[set[str]] = None,
                 type_thresholds: Optional[dict[str, float]] = None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.rule_extractor = rule_extractor
        self.threshold = threshold
        self.type_thresholds = dict(type_thresholds or {})   # e.g. {"Property": 0.7}; others use `threshold`
        self.predict_floor: Optional[float] = None
        self.context = "sentence"   # sentence | neighbours (previous + next sentence of the same section as context)
        self.verifier: Any = None   # optional GlinerVerifier (LLM checker for new entities)
        self.labels = dict(labels or DEFAULT_LABELS)
        self.batch_size = batch_size
        self.model_name = model_name or getattr(model, "name", "gliner")
        self.skip_new_in = set(skip_new_in or ())
        # raw predictions per (labels, floor threshold, text): threshold sweeps reuse one model pass
        self._cache: dict[tuple, list[dict[str, Any]]] = {}

    @classmethod
    def load(cls, model_name: str, normalizer: EntityNormalizer, rule_extractor: RuleEntityExtractor,
             threshold: float = 0.5, labels: Optional[dict[str, str]] = None, device: str = "cpu",
             batch_size: int = 8, skip_new_in: Optional[set[str]] = None,
             type_thresholds: Optional[dict[str, float]] = None) -> "GlinerEntityExtractor":
        """Load the Hugging Face model. Raises ImportError if `gliner` is not installed."""
        from gliner import GLiNER  # optional dependency, imported lazily
        t0 = time.perf_counter()
        model = GLiNER.from_pretrained(model_name)
        if device and device != "cpu":
            model = model.to(device)
        log.info("GLiNER model %s loaded in %.1fs (%s)", model_name, time.perf_counter() - t0, device)
        return cls(model, normalizer, rule_extractor, threshold, labels, batch_size, model_name, skip_new_in,
                   type_thresholds)

    def threshold_for(self, entity_type: str) -> float:
        return self.type_thresholds.get(entity_type, self.threshold)

    def _floor(self) -> float:
        # predict_floor: the sweep predicts once at a low threshold and re-filters per config
        return min([self.threshold, *self.type_thresholds.values(), *([self.predict_floor] if self.predict_floor else [])])

    # ------------------------------------------------------------------
    def _model_predict(self, texts: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        out = self._raw_predict(texts, threshold)
        if not self._lower_labels():
            return out
        back = {lab.lower(): lab for lab in self.labels}   # NuNER Zero wants lower-case labels
        return [[{**p, "label": back.get(p.get("label", ""), p.get("label", ""))} for p in preds] for preds in out]

    def _lower_labels(self) -> bool:
        return "nuner" in (self.model_name or "").lower()

    def _raw_predict(self, texts: list[str], threshold: float) -> list[list[dict[str, Any]]]:
        labels = [lab.lower() for lab in self.labels] if self._lower_labels() else list(self.labels)
        if hasattr(self.model, "inference"):          # gliner >= 0.2.2x (batch API)
            return self.model.inference(texts, labels, threshold=threshold, batch_size=self.batch_size)
        if hasattr(self.model, "batch_predict_entities"):   # older gliner versions
            out: list[list[dict[str, Any]]] = []
            for i in range(0, len(texts), self.batch_size):
                out += self.model.batch_predict_entities(texts[i:i + self.batch_size], labels, threshold=threshold)
            return out
        return [self.model.predict_entities(t, labels, threshold=threshold) for t in texts]

    @staticmethod
    def _windows(text: str) -> list[tuple[int, str]]:
        """(char offset, piece): short texts as they are, long ones as overlapping word windows."""
        words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        if len(words) <= MAX_WORDS:
            return [(0, text)]
        out = []
        for i in range(0, len(words), WINDOW_STRIDE):
            chunk = words[i:i + MAX_WORDS]
            a, b = chunk[0][0], chunk[-1][1]
            out.append((a, text[a:b]))
            if i + MAX_WORDS >= len(words):
                break
        return out

    def _predict(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Predictions with character offsets into each full text (long texts are windowed)."""
        floor = self._floor()
        key_labels = tuple(self.labels)
        pieces: list[tuple[int, int, str]] = []          # (text index, offset, piece)
        for ti, t in enumerate(texts):
            for off, piece in self._windows(t):
                pieces.append((ti, off, piece))
        todo = list(dict.fromkeys(p for _, _, p in pieces if (key_labels, floor, p) not in self._cache))
        if todo:
            for piece, preds in zip(todo, self._model_predict(todo, floor)):
                self._cache[(key_labels, floor, piece)] = preds
        out: list[dict[tuple, dict[str, Any]]] = [{} for _ in texts]
        for ti, off, piece in pieces:
            for p in self._cache[(key_labels, floor, piece)]:
                q = {**p, "start": int(p.get("start", -1)) + off, "end": int(p.get("end", -1)) + off}
                k = (q["start"], q["end"], q.get("label"))
                if k not in out[ti] or float(q.get("score", 0)) > float(out[ti][k].get("score", 0)):
                    out[ti][k] = q
        return [list(d.values()) for d in out]

    def _predict_with_neighbours(self, doc: ParsedDocument, sents: list) -> list[list[dict[str, Any]]]:
        """GLiNER reads 'previous + sentence + next' (same section, prose only); spans outside the
        middle sentence are dropped and offsets are mapped back to the sentence."""
        prose = [s for s in doc.sentences if s.source_type != "table"]
        pos = {s.sentence_id: i for i, s in enumerate(prose)}
        texts, offsets = [], []
        for s in sents:
            i = pos.get(s.sentence_id)
            prev = prose[i - 1].text if i is not None and i > 0 and prose[i - 1].section_id == s.section_id else ""
            nxt = (prose[i + 1].text if i is not None and i + 1 < len(prose)
                   and prose[i + 1].section_id == s.section_id else "")
            head = prev + " " if prev else ""
            texts.append(head + s.text + (" " + nxt if nxt else ""))
            offsets.append(len(head))
        out = []
        for s, off, preds in zip(sents, offsets, self._predict(texts)):
            keep = []
            for p in preds:
                a, b = int(p.get("start", -1)) - off, int(p.get("end", -1)) - off
                if 0 <= a < b <= len(s.text):
                    keep.append({**p, "start": a, "end": b})
            out.append(keep)
        return out

    def extract(self, doc: ParsedDocument, analyses: dict[str, SentenceAnalysis], skip: set[str]) -> GlinerOutput:
        """Add GLiNER mentions to `analyses` (in place) and return what was added.

        Pass 1 collects candidates (threshold, junk filter, overlap with rule mentions, normalisation);
        the optional LLM checker then judges the NEW (non-lexicon) ones; pass 2 adds the survivors."""
        out = GlinerOutput()
        paper_id = doc.paper.paper_id
        # Table rows are flattened cell text ("Table 2, row 3: ...") - not prose, skip them.
        sents = [s for s in doc.sentences
                 if s.section_type not in skip and s.source_type != "table" and s.sentence_id in analyses]
        out.sentences = len(sents)
        if not sents:
            return out
        t0 = time.perf_counter()
        if self.context == "neighbours":
            predictions = self._predict_with_neighbours(doc, sents)
        else:
            predictions = self._predict([s.text for s in sents])
        out.seconds = round(time.perf_counter() - t0, 2)

        # ---- pass 1: candidates
        pending: list[dict[str, Any]] = []
        for s, preds in zip(sents, predictions):
            an = analyses[s.sentence_id]
            taken: list[tuple[int, int]] = []          # spans already chosen in this sentence
            for p in sorted(preds, key=lambda e: -float(e.get("score", 0))):
                etype = self.labels.get(p.get("label", ""))
                start, end = int(p.get("start", -1)), int(p.get("end", -1))
                if etype is None or not (0 <= start < end <= len(s.text)):
                    continue
                if float(p.get("score", 0)) < self.threshold_for(etype):
                    continue
                det = _LEADING_DET.match(s.text[start:end])
                if det:
                    start += det.end()
                    if start >= end:
                        continue
                surface = s.text[start:end]
                score = min(float(p.get("score", 0.5)), 0.95)
                # junk filter BEFORE normalisation (which may ask the LLM to choose a canonical)
                # a paper abbreviation is trusted only if its long form is a lexicon term
                # ("AB" = as-built or "SS" = side surface are not entities)
                long_form = self.normalizer.paper_abbreviations.get(surface, "")
                in_lex = bool(self.normalizer.lexicon.lookup_any_type(surface)
                              or self.normalizer.resolve_symbol(surface)
                              or (long_form and self.normalizer.lexicon.lookup_any_type(long_form)))
                reason = None if in_lex else junk_reason(surface, etype, False)
                if reason:
                    out.dropped.append({"stage": "gliner_entity", "reason": reason, "text": surface,
                                        "type": etype, "sentence_id": s.sentence_id, "score": round(score, 3)})
                    continue
                if any(a < end and start < b for a, b in taken):
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
                taken.append((start, end))
                pending.append({"s": s, "start": start, "end": end, "surface": surface, "etype": etype,
                                "score": score, "norm": norm,
                                "new": norm.canonical_id not in self.normalizer.lexicon.entities})

        # ---- LLM checker: only NEW entities (lexicon entities are trusted)
        to_check = [c for c in pending if c["new"]]
        if self.verifier is not None and to_check:
            from .gliner_verifier import Candidate
            verdicts = self.verifier.verify([Candidate(c["s"].text, c["surface"], c["etype"]) for c in to_check])
            for c, v in zip(to_check, verdicts):
                if not v.keep:
                    c["drop"] = v.reason or "llm_checker_rejected"
                    out.dropped.append({"stage": "gliner_check", "reason": c["drop"], "text": c["surface"],
                                        "type": c["etype"], "sentence_id": c["s"].sentence_id,
                                        "score": round(c["score"], 3)})
                elif v.entity_type != c["etype"]:       # checker corrected the type
                    c["etype"] = v.entity_type
                    c["norm"] = self.normalizer.normalize(c["surface"], v.entity_type)

        # ---- pass 2: add the survivors
        touched: dict[str, SentenceAnalysis] = {}
        for c in pending:
            if c.get("drop"):
                continue
            s, norm = c["s"], c["norm"]
            mention = EntityMention(
                mention_id=stable_id("m", s.sentence_id, c["start"], c["end"], "gliner"), text=c["surface"],
                entity_type=norm.entity_type, canonical_id=norm.canonical_id,
                canonical_name=norm.canonical_name, normalization_confidence=norm.confidence,
                normalization_method=norm.method, sentence_id=s.sentence_id,
                char_start=c["start"], char_end=c["end"],
                confidence=round(0.5 * c["score"] + 0.5 * norm.confidence, 3),
                method="gliner", evidence=evidence_for(s, paper_id))
            an = analyses[s.sentence_id]
            an.mentions.append(mention)
            out.mentions.append(mention)
            touched[s.sentence_id] = an
        for an in touched.values():   # values ("200 W") are attached by the dimension-checked rule assigner
            an.assignments = self.rule_extractor.assign_values(an)
            for a in an.assignments:
                m = an.mentions[a.mention_index]
                if m.quantity is None:
                    m.quantity = a.qspan.quantity
        return out
